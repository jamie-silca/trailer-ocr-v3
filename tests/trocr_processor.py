"""
EXP-20 — TrOCR as a drop-in replacement for PaddleOCR's recognition pipeline.

Mirrors the `OcrProcessor.process_image(pil) -> (text, conf)` contract so the
benchmark can swap engines via `--rec-engine trocr`.

TrOCR is end-to-end (ViT encoder + transformer decoder); it does not do
detection. For benchmark crops we already have the bbox, so feeding the whole
crop directly is the intended use.
"""
from __future__ import annotations

import logging
import math
from PIL import Image

logger = logging.getLogger(__name__)


class TrocrProcessor:
    _instance = None

    DEFAULT_MODEL = "microsoft/trocr-base-printed"
    MIN_CONFIDENCE = 0.3  # TrOCR geometric-mean token probabilities run lower than PP-OCR CRNN scores

    def __new__(cls, model_id: str | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model_id or cls.DEFAULT_MODEL)
        return cls._instance

    def _initialize(self, model_id: str):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        logger.info(f"Initialising TrOCR ({model_id})...")
        self._torch = torch
        self._processor = TrOCRProcessor.from_pretrained(model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self._model.eval()
        self._model_id = model_id
        logger.info(f"TrOCR ready: {model_id}")

    def process_image(self, image: Image.Image) -> tuple:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            torch = self._torch
            pv = self._processor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                out = self._model.generate(
                    pv,
                    max_new_tokens=20,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            ids = out.sequences
            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
            text = (text or "").strip().upper().replace(" ", "")

            # Geometric mean of per-token softmax max probabilities as a confidence proxy.
            conf = 1.0
            if out.scores:
                logp_sum = 0.0
                n = 0
                for step_logits in out.scores:
                    probs = torch.softmax(step_logits, dim=-1)
                    top_p = float(probs.max(dim=-1).values[0])
                    logp_sum += math.log(max(top_p, 1e-9))
                    n += 1
                if n:
                    conf = math.exp(logp_sum / n)

            if not text:
                return None, 0.0
            if conf < self.MIN_CONFIDENCE:
                logger.info(f"Dropping low-conf TrOCR result '{text}' ({conf:.2f} < {self.MIN_CONFIDENCE})")
                return None, 0.0
            return text, conf
        except Exception as e:
            logger.error(f"TrOCR failed: {e}")
            return None, 0.0
