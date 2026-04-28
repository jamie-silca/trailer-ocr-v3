"""
Run Qwen3-VL-8B on EVERY annotation in the dataset (all buckets), saving
the raw + normalized output per crop. Then cross-reference vs the EXP-25
baseline benchmark JSON to answer:

  Q: "Does the VLM capture more than the current implementation, and
      does it capture things the current impl would fail at?"

Outputs:
  tests/results/qwen_full_run_<dataset>.json     — raw per-crop results
  tests/results/qwen_vs_exp25_<dataset>.json     — comparison summary

Cost: 419 OpenRouter calls ≈ $0.01. Cached on re-run.

Usage:
    python tests/qwen_full_run.py
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

from qwen_universal import QwenUniversalProcessor  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen_full_run")
logger.setLevel(logging.INFO)

DATASETS = {
    "20260406": ("20260406", "annotations_2026-04-06_09-06_coco_with_text.json"),
    "20260423": ("20260423", "annotations_2026-04-23_11-24_coco_with_text.json"),
}
DEFAULT_BASELINE = (
    PROJECT_ROOT / "tests" / "results"
    / "benchmark_EXP-25-VERIFY-NO-FMT-CHECK_20260427_133655.json"
)


def crop_absolute(image: Image.Image, bbox: list) -> Image.Image | None:
    x, y, w, h = bbox
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(image.width, int(x + w)), min(image.height, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def normalize_for_match(s: str | None) -> str:
    return (s or "").strip().upper().replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--dataset", default="20260423", choices=list(DATASETS))
    ap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap (0 = full).")
    args = ap.parse_args()

    ds_dir, ds_ann = DATASETS[args.dataset]
    dataset_dir = PROJECT_ROOT / "tests" / "dataset" / ds_dir
    coco = json.loads((dataset_dir / ds_ann).read_text(encoding="utf-8"))
    annotations = coco["annotations"]
    images_by_id = {img["id"]: img for img in coco["images"]}
    if args.limit:
        annotations = annotations[: args.limit]

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    exp25_by_ann = {r["annotation_id"]: r for r in baseline["annotation_results"]}

    qwen = QwenUniversalProcessor(model_id=args.model)
    image_cache: dict[int, Image.Image] = {}
    rows = []

    t_start = time.perf_counter()
    for i, ann in enumerate(annotations, 1):
        ann_id = ann["id"]
        gt = ann.get("text") or ""
        image_id = ann["image_id"]
        baseline_row = exp25_by_ann.get(ann_id)
        if baseline_row is None:
            logger.warning(f"ann {ann_id} missing from baseline, skipping")
            continue

        if image_id not in image_cache:
            img_meta = images_by_id[image_id]
            image_cache[image_id] = Image.open(dataset_dir / img_meta["file_name"]).convert("RGB")
        crop = crop_absolute(image_cache[image_id], ann["bbox"])
        if crop is None:
            logger.warning(f"ann {ann_id}: invalid bbox")
            continue

        rec = qwen.process_image(crop)
        gt_n = normalize_for_match(gt)
        qwen_n = normalize_for_match(rec["normalized"])
        qwen_exact = bool(qwen_n) and qwen_n == gt_n
        exp25_exact = bool(baseline_row.get("exact_match"))

        rows.append({
            "annotation_id": ann_id,
            "bucket": baseline_row["aspect_ratio_bucket"],
            "crop_size": baseline_row["crop_size"],
            "ground_truth": gt,
            "exp25_text": baseline_row["ocr_text"],
            "exp25_exact": exp25_exact,
            "qwen_raw": rec["raw"],
            "qwen_normalized": rec["normalized"],
            "qwen_is_unknown": rec["is_unknown"],
            "qwen_exact": qwen_exact,
            "qwen_from_cache": rec["from_cache"],
            "qwen_latency_ms": rec["latency_ms"],
        })

        marker = (
            "BOTH" if qwen_exact and exp25_exact else
            "QWEN" if qwen_exact else
            "E25 " if exp25_exact else
            "----"
        )
        logger.info(
            f"[{i:3d}/{len(annotations)}] {marker} ann {ann_id:4d} {baseline_row['aspect_ratio_bucket']:10s} "
            f"gt={gt!r:18s} exp25={(baseline_row['ocr_text'] or '')!r:18s} qwen={rec['normalized']!r:14s}"
        )

    wall = time.perf_counter() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(rows)
    qwen_correct = sum(1 for r in rows if r["qwen_exact"])
    exp25_correct = sum(1 for r in rows if r["exp25_exact"])
    both = sum(1 for r in rows if r["qwen_exact"] and r["exp25_exact"])
    qwen_only = [r for r in rows if r["qwen_exact"] and not r["exp25_exact"]]
    exp25_only = [r for r in rows if r["exp25_exact"] and not r["qwen_exact"]]
    neither = [r for r in rows if not r["qwen_exact"] and not r["exp25_exact"]]

    qwen_text_returned = sum(1 for r in rows if r["qwen_normalized"])
    qwen_wrong_text = qwen_text_returned - qwen_correct

    exp25_text_returned = sum(1 for r in rows if r["exp25_text"])
    exp25_wrong_text = exp25_text_returned - exp25_correct

    print()
    print("=" * 78)
    print(f"Qwen-only (universal prompt) vs EXP-25 cascade — n={n}")
    print(f"model: {qwen._model_id}  wall: {wall:.1f}s  cache_hits: {qwen._cache_hits}")
    print()
    print(f"{'metric':<32}{'EXP-25':>12}{'Qwen-only':>14}{'delta':>8}")
    print(f"{'EXACT':<32}{exp25_correct:>12}{qwen_correct:>14}{qwen_correct - exp25_correct:>+8}")
    print(f"{'wrong-text':<32}{exp25_wrong_text:>12}{qwen_wrong_text:>14}{qwen_wrong_text - exp25_wrong_text:>+8}")
    print(f"{'no-text':<32}{n - exp25_text_returned:>12}{n - qwen_text_returned:>14}")
    print()
    print("Overlap matrix:")
    print(f"  Both correct:     {both:4d}")
    print(f"  Qwen-only wins:   {len(qwen_only):4d}   <-- VLM captures, current would fail")
    print(f"  EXP-25-only wins: {len(exp25_only):4d}   <-- current captures, VLM would fail")
    print(f"  Both wrong:       {len(neither):4d}")

    # Per-bucket breakdown
    print("\nPer-bucket:")
    print(f"  {'bucket':<12}{'n':>4}{'exp25':>8}{'qwen':>8}{'qwen-only':>12}{'exp25-only':>12}")
    for b in ("portrait", "landscape", "wide", "very_wide"):
        bucket_rows = [r for r in rows if r["bucket"] == b]
        if not bucket_rows:
            continue
        e = sum(1 for r in bucket_rows if r["exp25_exact"])
        q = sum(1 for r in bucket_rows if r["qwen_exact"])
        qo = sum(1 for r in bucket_rows if r["qwen_exact"] and not r["exp25_exact"])
        eo = sum(1 for r in bucket_rows if r["exp25_exact"] and not r["qwen_exact"])
        print(f"  {b:<12}{len(bucket_rows):>4}{e:>8}{q:>8}{qo:>12}{eo:>12}")

    # Sample qwen-only wins (where VLM shines)
    print(f"\nSample Qwen-only wins (n={len(qwen_only)}, showing up to 20):")
    for r in qwen_only[:20]:
        print(f"  ann {r['annotation_id']:4d} {r['bucket']:10s} gt={r['ground_truth']!r:18s} "
              f"exp25={r['exp25_text']!r:18s} qwen={r['qwen_normalized']!r}")

    print(f"\nSample EXP-25-only wins (n={len(exp25_only)}, showing up to 20):")
    for r in exp25_only[:20]:
        print(f"  ann {r['annotation_id']:4d} {r['bucket']:10s} gt={r['ground_truth']!r:18s} "
              f"exp25={r['exp25_text']!r:18s} qwen={r['qwen_normalized']!r}")

    print(f"\nQwen stats: {qwen.stats()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_raw = PROJECT_ROOT / "tests" / "results" / f"qwen_full_run_{args.dataset}.json"
    out_raw.write_text(json.dumps({
        "experiment": "qwen-only-full-run",
        "model": qwen._model_id,
        "prompt_version": "v1-universal",
        "baseline": Path(args.baseline).name,
        "dataset": args.dataset,
        "n": n,
        "wall_s": round(wall, 2),
        "qwen_stats": qwen.stats(),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nRaw results: {out_raw}")

    out_cmp = PROJECT_ROOT / "tests" / "results" / f"qwen_vs_exp25_{args.dataset}.json"
    out_cmp.write_text(json.dumps({
        "n": n,
        "exp25_exact": exp25_correct,
        "qwen_exact": qwen_correct,
        "both_correct": both,
        "qwen_only_wins": [r["annotation_id"] for r in qwen_only],
        "exp25_only_wins": [r["annotation_id"] for r in exp25_only],
        "neither_correct": [r["annotation_id"] for r in neither],
        "by_bucket": {
            b: {
                "n": sum(1 for r in rows if r["bucket"] == b),
                "exp25_exact": sum(1 for r in rows if r["bucket"] == b and r["exp25_exact"]),
                "qwen_exact": sum(1 for r in rows if r["bucket"] == b and r["qwen_exact"]),
                "qwen_only_wins": sum(1 for r in rows if r["bucket"] == b and r["qwen_exact"] and not r["exp25_exact"]),
                "exp25_only_wins": sum(1 for r in rows if r["bucket"] == b and r["exp25_exact"] and not r["qwen_exact"]),
            } for b in ("portrait", "landscape", "wide", "very_wide")
        },
    }, indent=2), encoding="utf-8")
    print(f"Comparison: {out_cmp}")


if __name__ == "__main__":
    main()
