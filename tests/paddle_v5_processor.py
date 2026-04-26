"""
EXP-22 — PaddleOCR 3.x / PP-OCRv5 drop-in processor.

Mirrors the OcrProcessor.process_image(pil) -> (text, conf) contract so the
benchmark can swap engines via --rec-engine paddle-v5.

Notes:
- Lives in the .venv-paddle-v5 isolated environment. Does NOT replace the v4
  processor in app/ocr_processor.py.
- enable_mkldnn=False because paddlepaddle 3.3.1's PIR executor hits
  NotImplementedError on some oneDNN attribute conversions on Windows CPU.
- Uses PP-OCRv5_server_rec (the multilingual model that the v5 release notes
  associate with vertical-text capability), not the en_PP-OCRv5_mobile_rec
  default which is English-only and lighter-weight.
- Detection thresholds matched to EXP-09's tuning so the v4 vs v5 comparison
  isolates the model swap, not configuration drift.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Allow importing app.utils (postprocess_text) — same as v4 path.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PaddleV5Processor:
    _instance = None

    MIN_CONFIDENCE = 0.6  # match v4 OcrProcessor

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initialising PP-OCRv5 (paddleocr 3.x, server_rec, mkldnn off)...")
        from paddleocr import PaddleOCR

        # Argv isolation — same hack as v4 path. PaddleOCR's transitive deps
        # use argparse globally and choke on the benchmark's argv.
        _old_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            self._ocr = PaddleOCR(
                lang="en",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_textline_orientation=True,
                enable_mkldnn=False,
                text_det_thresh=0.2,
                text_det_box_thresh=0.3,
                text_det_unclip_ratio=2.0,
            )
        finally:
            sys.argv = _old_argv
        # Suppress PIL/Postprocess imports here (lazy) so import-time is cheap.
        from app.utils import postprocess_text as _postprocess
        self._postprocess = _postprocess
        logger.info("PP-OCRv5 ready.")

    def process_image(self, image: Image.Image) -> tuple:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            arr = np.array(image)
            result = self._ocr.predict(arr)
            if not result:
                return None, 0.0
            r0 = result[0]
            texts = r0.get("rec_texts") or []
            scores = r0.get("rec_scores") or []
            # Drop empty strings (per-char boxes the rec couldn't read).
            kept = [(t, s) for t, s in zip(texts, scores) if t]
            if not kept:
                return None, 0.0
            joined = " ".join(t for t, _ in kept)
            avg_conf = sum(s for _, s in kept) / len(kept)
            if avg_conf < self.MIN_CONFIDENCE:
                logger.info(f"Dropping low-conf v5 result '{joined}' ({avg_conf:.2f})")
                return None, 0.0
            joined = self._postprocess(joined)
            return joined, float(avg_conf)
        except Exception as e:
            logger.error(f"PP-OCRv5 failed: {e}")
            return None, 0.0
