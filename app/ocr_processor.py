import logging
import os

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from app.utils import pad_small, postprocess_text, sharpen, dilate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OcrProcessor:
    # Drop OCR results below this average confidence.
    # Validated post-hoc on EXP-09+10 output: +4.5pp precision for -0.15pp exact match.
    MIN_CONFIDENCE = 0.6

    _instance = None
    _ocr = None
    _qwen = None  # EXP-25: Qwen3-VL portrait fallback (initialised if OPENROUTER_API_KEY set)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OcrProcessor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing PaddleOCR 2.7.3...")
        try:
            # PaddleOCR's internal codebase uses argparse at a global level.
            # When running inside a Uvicorn/FastAPI wrapper, PaddleOCR tries to parse
            # the web server's CLI arguments as its own.
            import sys
            _old_argv = sys.argv
            sys.argv = [sys.argv[0]]  # Wipe the args temporarily

            # Optimized PaddleOCR parameters based on EXP-04 results
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False,
                det_db_thresh=0.2,
                det_db_box_thresh=0.3,
                det_db_unclip_ratio=2.0
            )

            sys.argv = _old_argv  # Restore them
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

        # EXP-25: portrait Qwen3-VL fallback. Off if no OPENROUTER_API_KEY —
        # service still runs as paddle-only in that case.
        if os.environ.get("OPENROUTER_API_KEY"):
            try:
                from app.qwen_portrait import QwenPortraitProcessor
                self._qwen = QwenPortraitProcessor()
                logger.info("Qwen portrait fallback enabled (EXP-25).")
            except Exception as e:
                logger.warning(f"Qwen portrait fallback disabled: {e}")
                self._qwen = None
        else:
            logger.info("OPENROUTER_API_KEY not set — Qwen portrait fallback disabled.")

    def process_image(self, image: Image.Image) -> tuple:
        """
        Run OCR on a PIL Image with Cascade Retry (EXP-10) and Qwen3-VL portrait
        fallback (EXP-25).
        Returns (text, confidence) or (None, 0.0)
        """
        try:
            # 1. Standard Pass (EXP-03: Pad small crops)
            processed_image = pad_small(image)
            text, conf = self._run_ocr(np.array(processed_image))

            # 2. EXP-10: Cascade Retry if no text found
            if not text:
                logger.info("First pass found no text. Attempting Cascade Retry (EXP-10)...")
                # Apply sharpen + dilate fallback preprocessing
                fallback_image = sharpen(image)
                fallback_image = dilate(fallback_image)
                # Pad the fallback image as well
                fallback_image = pad_small(fallback_image)

                text, conf = self._run_ocr(np.array(fallback_image))
                if text:
                    logger.info(f"Cascade Retry successful: '{text}'")

            # 3. EXP-25: Portrait Qwen3-VL fallback. Always fires on portrait
            # crops (h > 2w, i.e. stacked-vertical). Paddle's format-valid rate
            # on portrait is 0% in our dataset, so a paddle-format-valid skip
            # gate would never trigger. If Qwen returns format-valid text it
            # overwrites paddle; if Qwen returns UNKNOWN, paddle's output is
            # preserved (this is what salvages numeric portrait reads).
            if self._qwen is not None and image.height > 2 * image.width:
                qwen_text, qwen_conf = self._qwen.process_image(image)
                if qwen_text:
                    logger.info(
                        f"Qwen portrait fallback hit: '{qwen_text}' "
                        f"(paddle was: {text!r})"
                    )
                    text, conf = qwen_text, qwen_conf

            if not text:
                return None, 0.0

            if conf < self.MIN_CONFIDENCE:
                logger.info(f"Dropping low-confidence result: '{text}' ({conf:.2f} < {self.MIN_CONFIDENCE})")
                return None, 0.0

            # EXP-06: Domain-specific character substitution post-processing
            text = postprocess_text(text)

            logger.info(f"Final OCR Result: '{text}' ({conf:.2f})")
            return text, conf

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None, 0.0

    def _run_ocr(self, img_array: np.ndarray) -> tuple:
        """
        Internal helper to execute PaddleOCR on a numpy array.
        Returns (text, average_confidence)
        """
        # PaddleOCR 2.7.0 returns: [[box, (text, conf)], ...]
        result = self._ocr.ocr(img_array, cls=True)

        if not result or not result[0]:
            return None, 0.0

        detected_texts = []
        confidences = []

        for line in result[0]:
            if not line or len(line) < 2:
                continue
            text_info = line[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                detected_texts.append(str(text_info[0]))
                confidences.append(float(text_info[1]))

        if not detected_texts:
            return None, 0.0

        full_text = " ".join(detected_texts)
        avg_conf = sum(confidences) / len(confidences)
        return full_text, avg_conf
