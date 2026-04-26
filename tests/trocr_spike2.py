"""
EXP-20 spike 2 — sanity-check TrOCR on natural horizontal (wide-bucket) crops.

If TrOCR can't even read the font on a native horizontal crop, there's no point
wiring it in at all. If it does read wide crops, then its failure on stitched
portraits points to the stitch quality as the bottleneck, not the model.
"""
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260406"

SAMPLES = [
    # (ann_id, gt, file, bbox, bucket_tag)
    (1, "702524", "DJI_20260406091050_0147_V.jpeg",
     [2720.26, 191.99, 174.95, 70.69], "right"),
    (2, "702520", "DJI_20260406091050_0147_V.jpeg",
     [1538.24, 262.08, 181.54, 72.39], "right"),
    (8, "701488", "DJI_20260406091257_0215_V.jpeg",
     [462.25, 1444.39, 214.50, 90.65], "right"),
    (9, "ATLS03", "DJI_20260406091610_0311_V.jpeg",
     [3566.45, 1193.21, 205.64, 97.59], "wrong-pp"),
    (27, "JBHU 286646", "DJI_20260406091646_0330_V.jpeg",
     [3750.55, 1262.99, 247.54, 80.47], "wrong-pp"),
]


def crop_bbox(img, bbox):
    x, y, w, h = bbox
    return img.crop((int(x), int(y), int(x + w), int(y + h)))


def main(model_id):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    import time

    print(f"\n=== {model_id} ===")
    t0 = time.perf_counter()
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    for ann_id, gt, fn, bbox, tag in SAMPLES:
        img = Image.open(DATASET / fn).convert("RGB")
        crop = crop_bbox(img, bbox)
        t0 = time.perf_counter()
        pv = processor(images=crop, return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=20)
        out = processor.batch_decode(ids, skip_special_tokens=True)[0]
        dt = (time.perf_counter() - t0) * 1000
        match = "OK" if out.strip().upper().replace(" ", "") == gt.strip().upper().replace(" ", "") else "xx"
        print(f"  [{match}] ann{ann_id:3d} {tag:9s} gt={gt!r:14s} -> {out!r:25s} ({crop.size}, {dt:.0f} ms)")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["microsoft/trocr-small-printed"]:
        main(m)
