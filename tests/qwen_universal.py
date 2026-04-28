"""
Qwen3-VL "universal" processor — no orientation hint in the prompt, no
format whitelist gate. Returns the model's raw answer (uppercased,
trimmed) or None for the literal UNKNOWN sentinel / empty response.

Used by tests/qwen_full_run.py for a fair head-to-head comparison
against EXP-25 (paddle + portrait-only Qwen fallback). Comparing
Qwen-on-everything vs paddle+cascade only makes sense if the Qwen call
isn't biased toward portrait OR horizontal reading order — hence the
omnidirectional prompt.

We preserve the raw Qwen output verbatim (in addition to gated) so
post-hoc analysis can apply different normalisation strategies without
re-billing the API.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from PIL import Image

import requests

logger = logging.getLogger(__name__)

PROMPT = (
    "The image shows a trailer ID plate. Read the characters in their "
    "natural reading order (left-to-right for horizontal plates, "
    "top-to-bottom for vertically-stacked plates). Return only the ID "
    "string, uppercase, with no punctuation or whitespace. If you "
    "cannot read it with confidence, return exactly UNKNOWN."
)
PROMPT_VERSION = "v1-universal"

MIN_LONG_SIDE_PX = 768
UNKNOWN_SENTINEL = "UNKNOWN"

CACHE_DIR = Path(__file__).parent / "results" / "qwen_universal_cache"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class QwenUniversalProcessor:
    _instance = None

    DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
    MAX_RETRIES = 3
    RETRY_BACKOFF_S = 2.0
    REQUEST_TIMEOUT_S = 90.0

    def __new__(cls, model_id: str | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model_id or cls.DEFAULT_MODEL)
        return cls._instance

    def _initialize(self, model_id: str):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Paste your key into .env at the project root."
            )
        self._api_key = api_key
        self._model_id = model_id
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_hits = 0
        self._cache_misses = 0
        self._unknown_responses = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _cache_key(self, png_bytes: bytes) -> str:
        h = hashlib.sha256()
        h.update(png_bytes)
        h.update(self._model_id.encode())
        h.update(PROMPT_VERSION.encode())
        return h.hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        path = CACHE_DIR / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_put(self, key: str, record: dict) -> None:
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

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
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        }
        last_err = None
        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                r = requests.post(ENDPOINT, headers=headers,
                                  data=json.dumps(body),
                                  timeout=self.REQUEST_TIMEOUT_S)
                latency_ms = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
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
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_BACKOFF_S * (attempt + 1))
        raise RuntimeError(f"Qwen call failed after {self.MAX_RETRIES} attempts: {last_err}")

    def process_image(self, image: Image.Image) -> dict:
        """Returns full record (raw + gated + meta), not just (text, conf)."""
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

        if record.get("input_tokens"):
            self._total_input_tokens += record["input_tokens"]
        if record.get("output_tokens"):
            self._total_output_tokens += record["output_tokens"]

        raw = (record.get("raw") or "").strip()
        normalized = raw.upper().replace(" ", "")
        is_unknown = normalized == UNKNOWN_SENTINEL
        if is_unknown:
            self._unknown_responses += 1

        return {
            "raw": raw,
            "normalized": "" if is_unknown or not normalized else normalized,
            "is_unknown": is_unknown,
            "from_cache": cached is not None,
            "latency_ms": record.get("latency_ms"),
            "crop_size": record.get("crop_size"),
        }

    def stats(self) -> dict:
        return {
            "model": self._model_id,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "unknown_responses": self._unknown_responses,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
