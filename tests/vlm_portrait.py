"""
EXP-18 — VLM fallback for portrait-bucket crops via Google Gemini.

Exposes a minimal client with the same ``process_image(pil) -> (text, conf)``
contract as OcrProcessor / TrocrProcessor, plus an on-disk response cache so
re-runs of the benchmark don't re-bill the API.

Prompt enforces a strict trailer-ID whitelist. A format gate rejects any
response that isn't JBHZ+6digits, JBHU+6digits, R+5digits, or the literal
"UNKNOWN" sentinel. Format-rejected responses return (None, 0.0).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

PROMPT = (
    "The image shows a trailer ID plate. Characters are upright and stacked "
    "vertically - read them top-to-bottom. Valid formats: JBHZ + 6 digits, "
    "JBHU + 6 digits, or R + 5 digits. Return only the ID string with no "
    "punctuation or whitespace. If you cannot read it with confidence, return "
    "exactly UNKNOWN."
)
PROMPT_VERSION = "v1"

MIN_LONG_SIDE_PX = 768  # Upscale tiny crops so Gemini's tile pipeline sees usable detail.

FORMAT_RE = re.compile(r"^(JBHZ\d{6}|JBHU\d{6}|R\d{5})$")
UNKNOWN_SENTINEL = "UNKNOWN"

CACHE_DIR = Path(__file__).parent / "results" / "vlm_cache"


class VlmPortraitProcessor:
    _instance = None

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_RETRIES = 3
    RETRY_BACKOFF_S = 2.0

    def __new__(cls, model_id: str | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model_id or cls.DEFAULT_MODEL)
        return cls._instance

    def _initialize(self, model_id: str):
        # Load .env so GEMINI_API_KEY is available without the user exporting it.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        from google import genai
        from google.genai import types as _types

        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY not set. Paste your key into .env at the project root."
            )

        logger.info(f"Initialising VlmPortraitProcessor ({model_id})...")
        self._client = genai.Client()
        self._types = _types
        self._model_id = model_id
        self._config = _types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=32,
            thinking_config=_types.ThinkingConfig(thinking_budget=0),
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        logger.info(f"VlmPortraitProcessor ready: model={model_id}, cache={CACHE_DIR}")

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
        types = self._types
        last_err = None
        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                r = self._client.models.generate_content(
                    model=self._model_id,
                    contents=[
                        types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                        PROMPT,
                    ],
                    config=self._config,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                raw = (r.text or "").strip()
                usage = r.usage_metadata
                return {
                    "raw": raw,
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": getattr(usage, "prompt_token_count", None),
                    "output_tokens": getattr(usage, "candidates_token_count", None),
                    "finish_reason": str(r.candidates[0].finish_reason) if r.candidates else None,
                    "attempts": attempt + 1,
                }
            except Exception as e:
                last_err = e
                msg = str(e)[:140]
                logger.warning(f"VLM call attempt {attempt + 1} failed: {type(e).__name__}: {msg}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_BACKOFF_S * (attempt + 1))
        raise RuntimeError(f"VLM call failed after {self.MAX_RETRIES} attempts: {last_err}")

    def process_image(self, image: Image.Image) -> tuple:
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
                call = self._call_api(png_bytes)
                record = {
                    "model": self._model_id,
                    "prompt_version": PROMPT_VERSION,
                    "crop_size": list(image.size),
                    **call,
                }
                self._cache_put(key, record)

            if record.get("input_tokens"):
                self._total_input_tokens += record["input_tokens"]
            if record.get("output_tokens"):
                self._total_output_tokens += record["output_tokens"]

            raw = (record.get("raw") or "").strip().upper().replace(" ", "")
            if not raw:
                return None, 0.0
            if raw == UNKNOWN_SENTINEL:
                return None, 0.0
            if not FORMAT_RE.match(raw):
                logger.info(f"VLM format-gate rejected: {raw!r}")
                return None, 0.0
            return raw, 1.0
        except Exception as e:
            logger.error(f"VLM portrait failed: {e}")
            return None, 0.0

    def stats(self) -> dict:
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
