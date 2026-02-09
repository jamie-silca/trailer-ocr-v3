import logging
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OcrProcessor:
    _instance = None
    _ocr = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OcrProcessor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing PaddleOCR 2.7.0...")
        try:
            # PaddleOCR 2.7.0 config - uses PP-OCRv4 by default
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False
            )
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def process_image(self, image: Image.Image) -> tuple:
        """
        Run OCR on a PIL Image.
        Returns (text, confidence) or (None, 0.0)
        """
        try:
            img_array = np.array(image)
            
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
            logger.info(f"OCR Result: '{full_text}' ({avg_conf:.2f})")
            return full_text, avg_conf
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None, 0.0
