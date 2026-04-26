"""
v5 stratified-sample benchmark — runs PP-OCRv5 on a small varied-bucket sample
and compares accuracy + latency against the cached EXP-09 v4 baseline on the
same annotations.

Sample: min(20, bucket_count) per aspect_ratio_bucket, deterministic seed.
Run inside .venv-paddle-v5:
    .venv-paddle-v5/Scripts/python.exe tests/benchmark_v5_sample.py
"""
from __future__ import annotations
import json
import random
import statistics
import sys
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260406"
ANN_FILE = DATASET / "annotations_2026-04-06_09-06_coco_with_text.json"
EXP09_FILE = PROJECT_ROOT / "tests" / "results" / "benchmark_EXP-09_20260420_175919.json"
OUT_FILE = PROJECT_ROOT / "tests" / "results" / "benchmark_v5_sample.json"

PER_BUCKET = 20
SEED = 42


def aspect_bucket(w: int, h: int) -> str:
    if h == 0:
        return "invalid"
    r = w / h
    if r < 0.5: return "portrait"
    if r < 1.0: return "near_square"
    if r < 2.0: return "landscape"
    if r < 4.0: return "wide"
    return "very_wide"


def main():
    exp09 = json.loads(EXP09_FILE.read_text(encoding="utf-8"))
    by_ann_v4 = {a["annotation_id"]: a for a in exp09["annotation_results"]}

    rng = random.Random(SEED)
    by_bucket: dict[str, list] = {}
    for a in exp09["annotation_results"]:
        by_bucket.setdefault(a["aspect_ratio_bucket"], []).append(a)

    picks = []
    for bucket, items in sorted(by_bucket.items()):
        rng.shuffle(items)
        picks.extend(items[: min(PER_BUCKET, len(items))])
    print(f"Sampled {len(picks)} crops across {len(by_bucket)} buckets:")
    from collections import Counter
    for b, c in sorted(Counter(p["aspect_ratio_bucket"] for p in picks).items()):
        print(f"  {b:12s} {c}")

    from paddle_v5_processor import PaddleV5Processor
    proc = PaddleV5Processor()

    # warmup
    _img = Image.new("RGB", (200, 80), (255, 255, 255))
    proc.process_image(_img)

    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    img_by_id = {i["id"]: i for i in coco["images"]}
    ann_by_id = {a["id"]: a for a in coco["annotations"]}

    rows = []
    t0 = time.perf_counter()
    for i, p in enumerate(picks, 1):
        ann = ann_by_id[p["annotation_id"]]
        img_meta = img_by_id[ann["image_id"]]
        img = Image.open(DATASET / img_meta["file_name"]).convert("RGB")
        x, y, w, h = ann["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        gt = (ann.get("text") or "").strip()
        bucket = aspect_bucket(crop.width, crop.height)

        s = time.perf_counter()
        text, conf = proc.process_image(crop)
        dt_ms = (time.perf_counter() - s) * 1000.0

        v4 = by_ann_v4[p["annotation_id"]]
        rows.append({
            "annotation_id": p["annotation_id"],
            "bucket": bucket,
            "gt": gt,
            "v5_text": text,
            "v5_conf": conf,
            "v5_match": (text or "") == gt and bool(gt),
            "v5_ms": dt_ms,
            "v4_text": v4.get("ocr_text"),
            "v4_match": bool(v4.get("exact_match")),
            "v4_ms": v4.get("elapsed_ms"),
        })
        if i % 10 == 0 or i == len(picks):
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            eta = (len(picks) - i) / rate
            print(f"  {i}/{len(picks)}  rate={rate:.2f}/s  eta={eta:.0f}s")

    wall = time.perf_counter() - t0

    # Summaries
    def fmt(buckets):
        for b, items in sorted(buckets.items()):
            n = len(items)
            v4c = sum(1 for r in items if r["v4_match"])
            v5c = sum(1 for r in items if r["v5_match"])
            v4ms = statistics.median(r["v4_ms"] for r in items)
            v5ms = statistics.median(r["v5_ms"] for r in items)
            print(f"  {b:12s} n={n:3d} | v4 {v4c:3d}/{n} ({v4c/n*100:5.1f}%) {v4ms:7.1f}ms | "
                  f"v5 {v5c:3d}/{n} ({v5c/n*100:5.1f}%) {v5ms:7.1f}ms")

    print("\nPer-bucket (v4 EXP-09 cached vs v5 fresh):")
    bb: dict[str, list] = {}
    for r in rows:
        bb.setdefault(r["bucket"], []).append(r)
    fmt(bb)

    n = len(rows)
    v4_correct = sum(1 for r in rows if r["v4_match"])
    v5_correct = sum(1 for r in rows if r["v5_match"])
    print(f"\nOverall sample (n={n}):")
    print(f"  v4 EXP-09 : {v4_correct}/{n} = {v4_correct/n*100:.1f}%  | "
          f"median {statistics.median(r['v4_ms'] for r in rows):.1f}ms")
    print(f"  v5        : {v5_correct}/{n} = {v5_correct/n*100:.1f}%  | "
          f"median {statistics.median(r['v5_ms'] for r in rows):.1f}ms")
    print(f"  v5 wall   : {wall:.1f}s ({wall/n*1000:.0f}ms/crop incl. setup overhead)")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "seed": SEED,
        "per_bucket_target": PER_BUCKET,
        "n": n,
        "wall_s": wall,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
