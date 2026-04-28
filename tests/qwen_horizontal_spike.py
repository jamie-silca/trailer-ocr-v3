"""
EXP-29 Stage 1 — paddle-fail spike on horizontal crops.

Loads the EXP-25-VERIFY baseline benchmark JSON, filters annotations where
aspect_ratio_bucket in {landscape, wide, very_wide} AND exact_match == False,
then calls QwenHorizontalProcessor on each. Reports per-crop rescue/regression
and aggregate rescue rate.

The crop is reproduced from the COCO annotation bbox the same way
benchmark_ocr.py crops it: open source image, slice [x, y, x+w, y+h]. No
preprocessing applied — Qwen's own 768-px upscale handles small crops.

Usage:
    python tests/qwen_horizontal_spike.py
        [--baseline tests/results/benchmark_EXP-25-VERIFY-NO-FMT-CHECK_20260427_133655.json]
        [--dataset 20260423]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from qwen_horizontal import QwenHorizontalProcessor  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen_horizontal_spike")
logger.setLevel(logging.INFO)

DATASETS = {
    "20260406": ("20260406", "annotations_2026-04-06_09-06_coco_with_text.json"),
    "20260423": ("20260423", "annotations_2026-04-23_11-24_coco_with_text.json"),
}

DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "tests"
    / "results"
    / "benchmark_EXP-25-VERIFY-NO-FMT-CHECK_20260427_133655.json"
)


def crop_absolute(image: Image.Image, bbox: list) -> Image.Image | None:
    x, y, w, h = bbox
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(image.width, int(x + w)), min(image.height, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--dataset", default="20260423", choices=list(DATASETS))
    ap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    results = baseline["annotation_results"]

    horiz_fails = [
        r for r in results
        if r["aspect_ratio_bucket"] in ("landscape", "wide", "very_wide")
        and r["exact_match"] is False
    ]
    logger.info(f"Baseline: {baseline_path.name}")
    logger.info(f"Horizontal paddle-fails: {len(horiz_fails)}")

    ds_dir, ds_ann = DATASETS[args.dataset]
    dataset_dir = PROJECT_ROOT / "tests" / "dataset" / ds_dir
    coco = json.loads((dataset_dir / ds_ann).read_text(encoding="utf-8"))
    anns_by_id = {a["id"]: a for a in coco["annotations"]}
    images_by_id = {img["id"]: img for img in coco["images"]}

    qwen = QwenHorizontalProcessor(model_id=args.model)

    rows = []
    image_cache: dict[int, Image.Image] = {}
    rescues = 0
    qwen_wrong_text = 0
    qwen_no_text = 0

    t_start = time.perf_counter()
    for i, baseline_row in enumerate(horiz_fails, 1):
        ann_id = baseline_row["annotation_id"]
        ann = anns_by_id[ann_id]
        gt = (baseline_row["ground_truth"] or "").strip().upper()
        bucket = baseline_row["aspect_ratio_bucket"]
        paddle_text = baseline_row["ocr_text"]
        image_id = baseline_row["image_id"]

        if image_id not in image_cache:
            img_meta = images_by_id[image_id]
            img_path = dataset_dir / img_meta["file_name"]
            image_cache[image_id] = Image.open(img_path).convert("RGB")
        crop = crop_absolute(image_cache[image_id], ann["bbox"])
        if crop is None:
            logger.warning(f"ann {ann_id}: invalid bbox, skipping")
            continue

        t0 = time.perf_counter()
        qwen_text, qwen_conf = qwen.process_image(crop)
        ms = (time.perf_counter() - t0) * 1000

        gated = qwen_text or ""
        is_rescue = bool(gated) and gated.upper() == gt
        is_qwen_wrong = bool(gated) and gated.upper() != gt

        if is_rescue:
            rescues += 1
        elif is_qwen_wrong:
            qwen_wrong_text += 1
        else:
            qwen_no_text += 1

        rows.append({
            "annotation_id": ann_id,
            "bucket": bucket,
            "ground_truth": gt,
            "paddle_text": paddle_text,
            "qwen_gated": gated,
            "is_rescue": is_rescue,
            "qwen_latency_ms": round(ms, 1),
            "crop_size": list(crop.size),
        })

        marker = "RESCUE" if is_rescue else ("WRONG " if is_qwen_wrong else "no-txt")
        logger.info(
            f"[{i:2d}/{len(horiz_fails)}] {marker} ann {ann_id:4d} {bucket:10s} "
            f"gt={gt!r:14s} paddle={(paddle_text or '')!r:18s} qwen={gated!r:14s} ({ms:.0f}ms)"
        )

    wall = time.perf_counter() - t_start

    n = len(horiz_fails)
    rescue_pct = (rescues / n * 100) if n else 0.0
    print()
    print("=" * 78)
    print(f"EXP-29 Stage 1 paddle-fail spike — {qwen._model_id}")
    print(f"baseline: {baseline_path.name}")
    print(f"n_paddle_fails: {n}  wall: {wall:.1f}s")
    print(f"rescues:       {rescues:3d} / {n} = {rescue_pct:.1f}%   (Qwen returns format-valid text matching GT)")
    print(f"qwen wrong:    {qwen_wrong_text:3d} / {n}              (Qwen returns format-valid text NOT matching GT)")
    print(f"qwen no-text:  {qwen_no_text:3d} / {n}              (Qwen UNKNOWN / format-rejected / empty)")
    print(f"qwen stats:    {qwen.stats()}")

    by_bucket: dict[str, dict] = {}
    for r in rows:
        b = r["bucket"]
        d = by_bucket.setdefault(b, {"total": 0, "rescue": 0, "wrong": 0, "none": 0})
        d["total"] += 1
        if r["is_rescue"]:
            d["rescue"] += 1
        elif r["qwen_gated"]:
            d["wrong"] += 1
        else:
            d["none"] += 1
    print("\nper-bucket:")
    for b, d in sorted(by_bucket.items()):
        rate = d["rescue"] / d["total"] * 100 if d["total"] else 0
        print(f"  {b:10s}: total={d['total']:3d} rescue={d['rescue']:3d} ({rate:.1f}%) wrong={d['wrong']:3d} none={d['none']:3d}")

    out = PROJECT_ROOT / "tests" / "results" / f"qwen_horizontal_spike_{args.dataset}.json"
    out.write_text(json.dumps({
        "experiment": "EXP-29-stage1",
        "model": qwen._model_id,
        "baseline": baseline_path.name,
        "dataset": args.dataset,
        "n_paddle_fails": n,
        "rescues": rescues,
        "rescue_rate": rescue_pct,
        "qwen_wrong_text": qwen_wrong_text,
        "qwen_no_text": qwen_no_text,
        "wall_s": round(wall, 2),
        "qwen_stats": qwen.stats(),
        "by_bucket": by_bucket,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nResult JSON: {out}")
    print("Stage-1 gate: >=30% rescue -> proceed to Stage 2; 15-30% -> manual; <15% -> shelve.")


if __name__ == "__main__":
    main()
