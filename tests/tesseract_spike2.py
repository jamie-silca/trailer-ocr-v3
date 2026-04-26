"""
EXP-19 spike2 — sanity variants before aborting.

Same 5 crops, but try:
  - PSM 5 grayscale-only (skip Otsu)
  - PSM 5 with inverted binary (in case polarity is wrong)
  - PSM 6 (single uniform block, horizontal)
  - PSM 11 (sparse text, no layout assumption)
  - Rotated 90 CW + PSM 6 (treat as horizontal)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

TESSERACT_EXE = r"C:\Users\jamie.barker\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260423"
ANN_FILE = DATASET / "annotations_2026-04-23_11-24_coco_with_text.json"
PICK_IDS = [17, 73, 165, 255, 356]
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_SHORT_SIDE = 256
BORDER_PX = 20


def upscale(g: Image.Image) -> Image.Image:
    sw, sh = g.size
    short = min(sw, sh)
    if short < MIN_SHORT_SIDE:
        scale = MIN_SHORT_SIDE / short
        g = g.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)
    return g


def otsu_binary(g: Image.Image, invert: bool = False) -> Image.Image:
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
    mask = (arr > thresh)
    if invert:
        mask = ~mask
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def run_tess(img: Image.Image, psm: int) -> str:
    cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
    raw = pytesseract.image_to_string(img, config=cfg)
    return re.sub(r"\s+", "", raw.strip().upper())


def main():
    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    img_by_id = {i["id"]: i for i in coco["images"]}
    ann_by_id = {a["id"]: a for a in coco["annotations"]}

    print(f"Tesseract: {pytesseract.get_tesseract_version()}\n")
    headers = ["psm5_gray", "psm5_otsu", "psm5_inv", "psm6_otsu", "psm11_otsu", "rot90_psm6"]
    print(f"{'ann':>5} {'gt':12} " + " ".join(f"{h:14}" for h in headers))

    for ann_id in PICK_IDS:
        ann = ann_by_id[ann_id]
        gt = (ann.get("text") or "").strip().upper()
        img_meta = img_by_id[ann["image_id"]]
        img = Image.open(DATASET / img_meta["file_name"]).convert("RGB")
        x, y, w, h = ann["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))

        gray = upscale(crop.convert("L"))
        gray_bordered = ImageOps.expand(gray, border=BORDER_PX, fill=255)
        otsu = ImageOps.expand(otsu_binary(gray, invert=False), border=BORDER_PX, fill=255)
        otsu_inv = ImageOps.expand(otsu_binary(gray, invert=True), border=BORDER_PX, fill=0)
        rotated = otsu.rotate(-90, expand=True)

        results = [
            run_tess(gray_bordered, 5),
            run_tess(otsu, 5),
            run_tess(otsu_inv, 5),
            run_tess(otsu, 6),
            run_tess(otsu, 11),
            run_tess(rotated, 6),
        ]
        cells = " ".join(f"{(r or '_'):14}" for r in results)
        print(f"{ann_id:>5} {gt:12} {cells}")


if __name__ == "__main__":
    main()
