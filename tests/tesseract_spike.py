"""
EXP-19 spike — Tesseract PSM 6 across the full portrait bucket of 20260423.

After the 20-crop sample produced 1 EXACT (ann 17), this widened run sweeps
all 119 portrait crops to see whether more cases land and whether they share
any structural property (image source, GT format).

Outputs:
  tests/results/tesseract_spike_psm6_20260423.json — per-crop rows + summary.
"""
from __future__ import annotations
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

TESSERACT_EXE = r"C:\Users\jamie.barker\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260423"
ANN_FILE = DATASET / "annotations_2026-04-23_11-24_coco_with_text.json"
OUT_FILE = PROJECT_ROOT / "tests" / "results" / "tesseract_spike_psm6_20260423.json"

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_SHORT_SIDE = 256
BORDER_PX = 20
PSM = 6


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def otsu_binary(g: Image.Image) -> Image.Image:
    arr = np.asarray(g)
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = arr.size
    sum_total = (np.arange(256) * hist).sum()
    sumB, wB, max_var, thresh = 0.0, 0.0, -1.0, 127
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var = wB * wF * (mB - mF) ** 2
        if var > max_var:
            max_var = var
            thresh = t
    return Image.fromarray((arr > thresh).astype(np.uint8) * 255, mode="L")


def preprocess(crop: Image.Image) -> Image.Image:
    g = crop.convert("L")
    sw, sh = g.size
    short = min(sw, sh)
    if short < MIN_SHORT_SIDE:
        scale = MIN_SHORT_SIDE / short
        g = g.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)
    return ImageOps.expand(otsu_binary(g), border=BORDER_PX, fill=255)


def gt_format(t: str) -> str:
    if re.match(r"^[A-Z]{4}\d{6}$", t):
        return "ALPHA4_DIGIT6"
    if re.match(r"^\d+$", t):
        return "NUMERIC"
    if re.match(r"^R\d+$", t):
        return "R_DIGITS"
    if t == "":
        return "EMPTY"
    return "OTHER"


def main():
    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    img_by_id = {i["id"]: i for i in coco["images"]}

    portraits = []
    for a in coco["annotations"]:
        bx = a["bbox"]
        w, h = bx[2], bx[3]
        if h <= 0 or w / h >= 0.5:
            continue
        portraits.append(a)
    portraits.sort(key=lambda a: a["id"])

    print(f"Tesseract: {pytesseract.get_tesseract_version()}")
    print(f"Portrait crops in 20260423: {len(portraits)}")
    print(f"Config: --oem 1 --psm {PSM} -c tessedit_char_whitelist={WHITELIST}")
    print()

    cfg = f"--oem 1 --psm {PSM} -c tessedit_char_whitelist={WHITELIST}"
    rows = []
    t_start = time.perf_counter()
    for i, ann in enumerate(portraits, 1):
        gt = (ann.get("text") or "").strip().upper()
        fmt = gt_format(gt)
        img_meta = img_by_id[ann["image_id"]]
        img = Image.open(DATASET / img_meta["file_name"]).convert("RGB")
        x, y, w, h = ann["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        pre = preprocess(crop)

        s = time.perf_counter()
        raw = pytesseract.image_to_string(pre, config=cfg)
        dt_ms = (time.perf_counter() - s) * 1000.0
        cleaned = re.sub(r"\s+", "", raw.strip().upper())
        ed = levenshtein(cleaned, gt) if gt else None

        rows.append({
            "ann_id": ann["id"],
            "image": img_meta["file_name"],
            "gt": gt,
            "gt_format": fmt,
            "tess_raw": cleaned,
            "edit_distance": ed,
            "exact": bool(gt) and cleaned == gt,
            "near": (ed is not None and ed <= 2),
            "crop_w": crop.width,
            "crop_h": crop.height,
            "ms": round(dt_ms, 1),
        })
        if i % 20 == 0 or i == len(portraits):
            elapsed = time.perf_counter() - t_start
            rate = i / elapsed
            eta = (len(portraits) - i) / rate
            print(f"  {i}/{len(portraits)}  rate={rate:.2f}/s  eta={eta:.0f}s")

    wall = time.perf_counter() - t_start

    by_fmt: dict[str, list] = {}
    for r in rows:
        by_fmt.setdefault(r["gt_format"], []).append(r)

    print(f"\nWall: {wall:.1f}s  ({len(rows)} crops, median {statistics.median(r['ms'] for r in rows):.0f}ms/crop)\n")
    print(f"{'gt_format':16s} {'n':>4s} {'exact':>6s} {'near':>5s} {'empty':>6s}")
    for fmt in sorted(by_fmt):
        items = by_fmt[fmt]
        n = len(items)
        exact = sum(1 for r in items if r["exact"])
        near = sum(1 for r in items if r["near"] and not r["exact"])
        empty = sum(1 for r in items if r["tess_raw"] == "")
        print(f"{fmt:16s} {n:>4d} {exact:>6d} {near:>5d} {empty:>6d}")

    total_exact = sum(1 for r in rows if r["exact"])
    total_near = sum(1 for r in rows if r["near"] and not r["exact"])
    print(f"\nOVERALL: EXACT {total_exact}/{len(rows)}, NEAR(<=2) {total_near}/{len(rows)}")

    print("\nEXACT hits:")
    for r in rows:
        if r["exact"]:
            print(f"  ann={r['ann_id']:6d}  gt={r['gt']:12s}  fmt={r['gt_format']}")

    print("\nNEAR hits (edit<=2, non-exact):")
    for r in rows:
        if r["near"] and not r["exact"]:
            print(f"  ann={r['ann_id']:6d}  gt={r['gt']:12s}  raw={r['tess_raw']!r:18s} edit={r['edit_distance']}  fmt={r['gt_format']}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "psm": PSM,
        "n": len(rows),
        "wall_s": round(wall, 1),
        "summary": {
            fmt: {
                "n": len(items),
                "exact": sum(1 for r in items if r["exact"]),
                "near": sum(1 for r in items if r["near"] and not r["exact"]),
                "empty": sum(1 for r in items if r["tess_raw"] == ""),
            } for fmt, items in by_fmt.items()
        },
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
