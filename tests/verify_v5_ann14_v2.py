"""
Tight verify: ann 14 detection box-count under multiple configs.
Writes JSON to disk (avoids cp1252 console crash on CJK-hallucinated rec output).
"""
from __future__ import annotations
import json, sys, traceback
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260406"
ANN_FILE = DATASET / "annotations_2026-04-06_09-06_coco_with_text.json"
OUT = PROJECT_ROOT / "tests" / "results" / "verify_v5_ann14.json"


def get_crop(coco, ann_id):
    by_id = {i["id"]: i for i in coco["images"]}
    for ann in coco["annotations"]:
        if ann["id"] == ann_id:
            img = Image.open(DATASET / by_id[ann["image_id"]]["file_name"]).convert("RGB")
            x, y, w, h = ann["bbox"]
            return img.crop((int(x), int(y), int(x+w), int(y+h))), (ann.get("text") or "").strip()
    return None, None


def run(ocr, crop, label):
    out = {"label": label}
    try:
        # original
        arr = np.array(crop)
        r = ocr.predict(arr)
        polys = (r[0].get("dt_polys") or []) if r else []
        out["original"] = {
            "size": list(crop.size),
            "boxes": len(polys),
            "rec_texts_count": len((r[0].get("rec_texts") or []) if r else []),
        }
        # upscaled 4x
        big = crop.resize((crop.size[0]*4, crop.size[1]*4), Image.LANCZOS)
        arr = np.array(big)
        r = ocr.predict(arr)
        polys = (r[0].get("dt_polys") or []) if r else []
        out["upscaled_4x"] = {
            "size": list(big.size),
            "boxes": len(polys),
            "rec_texts_count": len((r[0].get("rec_texts") or []) if r else []),
        }
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    return out


def main():
    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    crop, gt = get_crop(coco, 14)

    from paddleocr import PaddleOCR
    results = {"ann": 14, "gt": gt, "crop_size": list(crop.size), "configs": {}}

    _argv = sys.argv

    # Config A — EXP-09 tuning (the documented config)
    sys.argv = [sys.argv[0]]
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
    results["configs"]["A_exp09_tuning"] = run(ocr_a, crop, "A")
    del ocr_a

    # Config C — defaults
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
    results["configs"]["C_defaults"] = run(ocr_c, crop, "C")
    del ocr_c

    # Config D — extreme low thresholds (more permissive than EXP-09)
    sys.argv = [sys.argv[0]]
    try:
        ocr_d = PaddleOCR(
            lang="en",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_textline_orientation=True,
            enable_mkldnn=False,
            text_det_thresh=0.1, text_det_box_thresh=0.15, text_det_unclip_ratio=2.5,
        )
    finally:
        sys.argv = _argv
    results["configs"]["D_extreme_permissive"] = run(ocr_d, crop, "D")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
