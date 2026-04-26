"""
Verify EXP-22 §10's claim that v5 returns 6 per-char boxes on ann 14.

Re-runs PP-OCRv5 on ann 14 in three configurations:
  A) Bare PaddleOCR.predict — same as precompute_v5_detection.py
  B) PaddleV5Processor.process_image — same code path as paddle_v5_sanity.py
  C) PaddleOCR.predict with default thresholds (no EXP-09 tuning)

Run from .venv-paddle-v5/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260406"
ANN_FILE = DATASET / "annotations_2026-04-06_09-06_coco_with_text.json"

TARGET_ANNS = [14, 85, 227, 304, 138]  # 14 is the EXP-22 claim; others are mixed


def get_crop(coco, ann_id):
    by_id = {i["id"]: i for i in coco["images"]}
    for ann in coco["annotations"]:
        if ann["id"] == ann_id:
            img = Image.open(DATASET / by_id[ann["image_id"]]["file_name"]).convert("RGB")
            x, y, w, h = ann["bbox"]
            return img.crop((int(x), int(y), int(x+w), int(y+h))), (ann.get("text") or "").strip()
    return None, None


def main():
    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))

    from paddleocr import PaddleOCR

    print("=== Config A: EXP-09 tuning (text_det_thresh=0.2, box_thresh=0.3, unclip=2.0) ===")
    _argv = sys.argv; sys.argv = [sys.argv[0]]
    try:
        ocr_a = PaddleOCR(
            lang="en",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_textline_orientation=True,
            enable_mkldnn=False,
            text_det_thresh=0.2, text_det_box_thresh=0.3, text_det_unclip_ratio=2.0,
        )
    finally:
        sys.argv = _argv

    for ann_id in TARGET_ANNS:
        crop, gt = get_crop(coco, ann_id)
        if crop is None:
            print(f"  ann {ann_id}: NOT FOUND in annotations"); continue
        arr = np.array(crop)
        result = ocr_a.predict(arr)
        polys = (result[0].get("dt_polys") or []) if result else []
        rec_texts = (result[0].get("rec_texts") or []) if result else []
        rec_scores = (result[0].get("rec_scores") or []) if result else []
        print(f"  ann {ann_id}  gt={gt!r:>14}  crop={crop.size}  boxes={len(polys)}  "
              f"rec_texts={rec_texts}  scores={[round(s,2) for s in rec_scores]}")

    print()
    print("=== Config C: default detection thresholds (no tuning) ===")
    sys.argv = [sys.argv[0]]
    try:
        ocr_c = PaddleOCR(
            lang="en",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_textline_orientation=True,
            enable_mkldnn=False,
        )
    finally:
        sys.argv = _argv

    for ann_id in TARGET_ANNS:
        crop, gt = get_crop(coco, ann_id)
        if crop is None: continue
        arr = np.array(crop)
        result = ocr_c.predict(arr)
        polys = (result[0].get("dt_polys") or []) if result else []
        rec_texts = (result[0].get("rec_texts") or []) if result else []
        print(f"  ann {ann_id}  gt={gt!r:>14}  crop={crop.size}  boxes={len(polys)}  rec_texts={rec_texts}")


if __name__ == "__main__":
    main()
