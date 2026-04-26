"""
EXP-17 precompute — Run PP-OCRv5 detection on every portrait-bucket crop and
dump per-character boxes to JSON.

Lives in .venv-paddle-v5 (paddleocr 3.5.0). The main benchmark loads the
JSON output and consumes the boxes through v4's recogniser, so we cross the
v4/v5 venv boundary exactly once per dataset.

Usage:
    .venv-paddle-v5/Scripts/python.exe tests/precompute_v5_detection.py

Output:
    tests/results/exp22_v5_detection.json
        {
          "<ann_id>": {
            "bbox": [x, y, w, h],
            "crop_size": [w, h],
            "gt": "JBHZ672061",
            "file": "...png",
            "dt_polys": [[[x,y],[x,y],[x,y],[x,y]], ...]   # crop-local coords
          },
          ...
        }
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))

DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260406"
ANN_FILE = DATASET / "annotations_2026-04-06_09-06_coco_with_text.json"
OUT_FILE = PROJECT_ROOT / "tests" / "results" / "exp22_v5_detection.json"


def select_portraits(coco):
    by_id = {i["id"]: i for i in coco["images"]}
    picked = []
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        if h <= 2 * w:
            continue
        picked.append({
            "ann": ann["id"],
            "gt": (ann.get("text") or "").strip(),
            "bbox": ann["bbox"],
            "file": by_id[ann["image_id"]]["file_name"],
        })
    return picked


def main():
    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    picks = select_portraits(coco)
    print(f"Found {len(picks)} portrait-bucket annotations.")

    from paddleocr import PaddleOCR
    print("Loading PP-OCRv5 (server_rec, mkldnn off, EXP-09 detector tuning)...")
    _old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        ocr = PaddleOCR(
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

    out = {}
    schema_logged = False
    start = time.perf_counter()
    for i, p in enumerate(picks, 1):
        img = Image.open(DATASET / p["file"]).convert("RGB")
        x, y, w, h = p["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        arr = np.array(crop)
        try:
            result = ocr.predict(arr)
        except Exception as e:
            print(f"  ann {p['ann']}: predict failed ({e})")
            out[str(p["ann"])] = {
                "bbox": p["bbox"], "crop_size": list(crop.size),
                "gt": p["gt"], "file": p["file"],
                "dt_polys": [], "error": str(e),
            }
            continue

        polys = []
        if result:
            r0 = result[0]
            if not schema_logged:
                try:
                    keys = list(r0.keys()) if hasattr(r0, "keys") else "no keys"
                except Exception:
                    keys = "introspection failed"
                print(f"  result[0] keys: {keys}")
                schema_logged = True
            polys_raw = None
            if hasattr(r0, "get"):
                polys_raw = r0.get("dt_polys")
                if polys_raw is None:
                    polys_raw = r0.get("det_polys")
            if polys_raw is not None:
                for poly in polys_raw:
                    polys.append([[float(pt[0]), float(pt[1])] for pt in poly])

        out[str(p["ann"])] = {
            "bbox": p["bbox"],
            "crop_size": list(crop.size),
            "gt": p["gt"],
            "file": p["file"],
            "dt_polys": polys,
        }
        if i % 20 == 0 or i == len(picks):
            elapsed = time.perf_counter() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(picks) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(picks)}  rate={rate:.2f}/s  eta={eta:.0f}s")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - start
    nonzero = sum(1 for v in out.values() if v.get("dt_polys"))
    box_counts = sorted(len(v.get("dt_polys") or []) for v in out.values())
    median_boxes = box_counts[len(box_counts) // 2] if box_counts else 0
    print(f"\nWrote {len(out)} entries to {OUT_FILE}")
    print(f"Wall: {elapsed:.1f}s  ({elapsed/len(out)*1000:.0f}ms/crop)")
    print(f"Boxes detected on {nonzero}/{len(out)} crops")
    print(f"Box-count distribution: min={box_counts[0]}, median={median_boxes}, max={box_counts[-1]}")


if __name__ == "__main__":
    main()
