"""
EXP-27 -- Qwen3-VL-8B local via Ollama, swap-in for OpenRouter (EXP-25).

Subclass of QwenPortraitProcessor that swaps the chat-completions endpoint
to a local Ollama server and drops the OpenRouter auth header. Same prompt,
same format gate, same 768-px LANCZOS upscale, same on-disk PNG-hash cache
pattern (different cache dir).

NOTE: qwen_portrait.py declares CACHE_DIR / ENDPOINT / PROMPT at module
level and references them directly inside _call_api / _cache_get /
_cache_put. To override cleanly without touching the parent we re-implement
those three methods here; everything else (process_image, _cache_key,
stats, format-gate logic) is reused via inheritance.

Tag is `qwen3-vl:8b` (Ollama default q4_K_M, ~5.86 GB) to match EXP-25's
OpenRouter `qwen/qwen3-vl-8b-instruct` as closely as possible.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

import requests

from qwen_portrait import PROMPT, QwenPortraitProcessor

logger = logging.getLogger(__name__)

LOCAL_CACHE_DIR = Path(__file__).parent / "results" / "qwen_local_cache"


class QwenLocalProcessor(QwenPortraitProcessor):
    _instance = None

    DEFAULT_MODEL = "qwen3-vl:8b"
    MAX_RETRIES = 1
    RETRY_BACKOFF_S = 0.0
    REQUEST_TIMEOUT_S = 300.0

    @staticmethod
    def _resolve_ollama_url() -> str:
        # OLLAMA_URL wins if set (full URL). Otherwise OLLAMA_HOST may be a
        # bare host (Ollama's own server bind var, e.g. "0.0.0.0" -- not
        # useful as a client URL); only honour it if it has a scheme.
        raw = os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST") or ""
        raw = raw.strip().rstrip("/")
        if raw.startswith(("http://", "https://")):
            return raw
        return "http://localhost:11434"

    def _initialize(self, model_id: str):
        self._endpoint = f"{self._resolve_ollama_url()}/v1/chat/completions"
        self._api_key = None
        self._model_id = model_id
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_dir = LOCAL_CACHE_DIR
        self._cache_hits = 0
        self._cache_misses = 0
        self._format_gate_hits = 0
        self._format_gate_misses = 0
        self._unknown_responses = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        logger.info(
            f"QwenLocalProcessor ready: model={model_id}, endpoint={self._endpoint}, "
            f"cache={self._cache_dir}"
        )

    def _cache_get(self, key: str) -> dict | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_put(self, key: str, record: dict) -> None:
        (self._cache_dir / f"{key}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    def _call_api(self, png_bytes: bytes) -> dict:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        headers = {"Content-Type": "application/json"}
        body = {
            "model": self._model_id,
            "temperature": 0,
            # qwen3-vl is a thinking model -- 32 tokens (EXP-25 OpenRouter
            # default) is consumed by reasoning, content returns empty.
            # 512 leaves headroom for thinking + answer.
            "max_tokens": 512,
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
                    self._endpoint,
                    headers=headers,
                    data=json.dumps(body),
                    timeout=self.REQUEST_TIMEOUT_S,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    logger.warning(f"Ollama call attempt {attempt + 1} failed: {last_err}")
                    continue
                j = r.json()
                raw = (j["choices"][0]["message"]["content"] or "").strip()
                usage = j.get("usage") or {}
                return {
                    "raw": raw,
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "attempts": attempt + 1,
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:140]}"
                logger.warning(f"Ollama call attempt {attempt + 1} failed: {last_err}")
        raise RuntimeError(f"Ollama call failed after {self.MAX_RETRIES} attempts: {last_err}")
