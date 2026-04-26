"""
Qwen3-VL portrait fallback (EXP-25, OpenRouter) — production module.

Adapted from tests/qwen_portrait.py with production-safe defaults:
  * No on-disk cache by default; opt in by setting QWEN_CACHE_DIR.
  * No tests/ path coupling; OPENROUTER_API_KEY read from process env
    (Cloud Run injects it as a configured env var or secret).

Same process_image(pil) -> (text, conf) contract as OcrProcessor. Used as a
CASCADE fallback for portrait crops when PaddleOCR returns no text OR text
that doesn't match the trailer-ID format whitelist.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

import requests

logger = logging.getLogger(__name__)

PROMPT = (
    "The image shows a trailer ID plate. Characters are upright and stacked "
    "vertically - read them top-to-bottom. Valid formats: JBHZ + 6 digits, "
    "JBHU + 6 digits, or R + 5 digits. Return only the ID string with no "
    "punctuation or whitespace. If you cannot read it with confidence, return "
    "exactly UNKNOWN."
)
PROMPT_VERSION = "v1"

MIN_LONG_SIDE_PX = 768

FORMAT_RE = re.compile(r"^(JBHZ\d{6}|JBHU\d{6}|R\d{5})$")
UNKNOWN_SENTINEL = "UNKNOWN"

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class QwenPortraitProcessor:
    _instance = None

    DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
    MAX_RETRIES = 3
    RETRY_BACKOFF_S = 2.0
    REQUEST_TIMEOUT_S = 90.0

    def __new__(cls, model_id: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model_id or cls.DEFAULT_MODEL)
        return cls._instance

    def _initialize(self, model_id: str):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        self._api_key = api_key
        self._model_id = model_id

        cache_env = os.environ.get("QWEN_CACHE_DIR")
        if cache_env:
            self._cache_dir: Optional[Path] = Path(cache_env)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

        self._cache_hits = 0
        self._cache_misses = 0
        self._format_gate_hits = 0
        self._format_gate_misses = 0
        self._unknown_responses = 0
        logger.info(
            f"QwenPortraitProcessor ready: model={model_id}, "
            f"cache={'off' if self._cache_dir is None else self._cache_dir}"
        )

    def _cache_key(self, png_bytes: bytes) -> str:
        h = hashlib.sha256()
        h.update(png_bytes)
        h.update(self._model_id.encode())
        h.update(PROMPT_VERSION.encode())
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[dict]:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_put(self, key: str, record: dict) -> None:
        if self._cache_dir is None:
            return
        try:
            (self._cache_dir / f"{key}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Qwen cache write failed: {e}")

    def _call_api(self, png_bytes: bytes) -> dict:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model_id,
            "temperature": 0,
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        last_err = None
        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                r = requests.post(
                    ENDPOINT,
                    headers=headers,
                    data=json.dumps(body),
                    timeout=self.REQUEST_TIMEOUT_S,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    logger.warning(f"Qwen attempt {attempt + 1} failed: {last_err}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_BACKOFF_S * (attempt + 1))
                    continue
                j = r.json()
                raw = (j["choices"][0]["message"]["content"] or "").strip()
                usage = j.get("usage", {})
                return {
                    "raw": raw,
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "attempts": attempt + 1,
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:140]}"
                logger.warning(f"Qwen attempt {attempt + 1} failed: {last_err}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_BACKOFF_S * (attempt + 1))
        raise RuntimeError(
            f"Qwen call failed after {self.MAX_RETRIES} attempts: {last_err}"
        )

    def process_image(self, image: Image.Image) -> Tuple[Optional[str], float]:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            long_side = max(image.size)
            if long_side < MIN_LONG_SIDE_PX:
                scale = MIN_LONG_SIDE_PX / long_side
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            key = self._cache_key(png_bytes)

            cached = self._cache_get(key)
            if cached is not None:
                self._cache_hits += 1
                record = cached
            else:
                self._cache_misses += 1
                record = {
                    "model": self._model_id,
                    "prompt_version": PROMPT_VERSION,
                    "crop_size": list(image.size),
                    **self._call_api(png_bytes),
                }
                self._cache_put(key, record)

            raw = (record.get("raw") or "").strip().upper().replace(" ", "")
            if not raw:
                self._format_gate_misses += 1
                return None, 0.0
            if raw == UNKNOWN_SENTINEL:
                self._unknown_responses += 1
                return None, 0.0
            if not FORMAT_RE.match(raw):
                self._format_gate_misses += 1
                logger.info(f"Qwen format-gate rejected: {raw!r}")
                return None, 0.0
            self._format_gate_hits += 1
            return raw, 1.0
        except Exception as e:
            logger.error(f"Qwen portrait failed: {e}")
            return None, 0.0
