"""
EXP-18 sanity — does upscaling + Gemini 2.5 Flash actually read portrait crops?

Picks N diverse portrait crops (different source images, varied GT IDs, varied
sizes) and feeds each through VlmPortraitProcessor. Tabulates hit rate so we
can decide whether to spend the full 156-call run (paid or tomorrow's free quota).

Usage:
    python tests/vlm_sanity.py           # default N=15
    python tests/vlm_sanity.py --n 5     # smoke test with 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))

DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260406"
ANN_FILE = DATASET / "annotations_2026-04-06_09-06_coco_with_text.json"


def pick_diverse_portraits(coco: dict, n: int, seed: int = 42) -> list[dict]:
    """Prefer unique source images, then unique GT IDs, then varied sizes."""
    images_by_id = {i["id"]: i for i in coco["images"]}
    candidates = []
    for ann in coco["annotations"]:
        gt = (ann.get("text") or "").strip()
        if not gt:
            continue
        x, y, w, h = ann["bbox"]
        if h <= 2 * w:  # portrait gate = benchmark's aspect_ratio_bucket=="portrait"
            continue
        candidates.append({
            "ann_id": ann["id"],
            "gt": gt,
            "bbox": ann["bbox"],
            "file": images_by_id[ann["image_id"]]["file_name"],
            "w": int(w), "h": int(h),
        })

    rng = random.Random(seed)
    rng.shuffle(candidates)

    picked: list[dict] = []
    seen_files: set[str] = set()
    seen_gts: set[str] = set()
    for c in candidates:
        if c["file"] in seen_files and c["gt"] in seen_gts:
            continue
        picked.append(c)
        seen_files.add(c["file"])
        seen_gts.add(c["gt"])
        if len(picked) >= n:
            break
    if len(picked) < n:
        for c in candidates:
            if c in picked:
                continue
            picked.append(c)
            if len(picked) >= n:
                break
    return picked


def crop_bbox(img: Image.Image, bbox: list) -> Image.Image:
    x, y, w, h = bbox
    return img.crop((int(x), int(y), int(x + w), int(y + h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    samples = pick_diverse_portraits(coco, args.n)

    from vlm_portrait import VlmPortraitProcessor
    vlm = VlmPortraitProcessor(args.model)

    print(f"\n=== EXP-18 sanity — {args.model}, N={len(samples)} ===\n")
    print(f"{'#':>2}  {'ann':>4}  {'orig':>9}  {'gt':>12}  {'vlm':>12}  {'match':>6}  {'latency':>9}")
    print("-" * 72)

    hits = 0
    rejected = 0  # format-gate rejected / UNKNOWN / empty
    errors = 0
    start = time.perf_counter()

    for i, s in enumerate(samples, 1):
        try:
            img = Image.open(DATASET / s["file"]).convert("RGB")
            crop = crop_bbox(img, s["bbox"])
            t0 = time.perf_counter()
            text, conf = vlm.process_image(crop)
            dt_ms = (time.perf_counter() - t0) * 1000

            orig = f"{s['w']}x{s['h']}"
            vlm_out = text if text else "-"
            if text is None:
                rejected += 1
                verdict = "miss"
            elif text == s["gt"].upper().replace(" ", ""):
                hits += 1
                verdict = "OK"
            else:
                verdict = "WRONG"
            print(f"{i:>2}  {s['ann_id']:>4}  {orig:>9}  {s['gt']:>12}  {vlm_out:>12}  {verdict:>6}  {dt_ms:>7.0f}ms")
        except Exception as e:
            errors += 1
            print(f"{i:>2}  {s['ann_id']:>4}  ERROR: {e}")

    elapsed = time.perf_counter() - start
    stats = vlm.stats()
    print("-" * 72)
    print(f"\nHits        : {hits} / {len(samples)}  ({100 * hits / len(samples):.1f}%)")
    print(f"Format-miss : {rejected}  (UNKNOWN / format-gate / empty)")
    print(f"Errors      : {errors}")
    print(f"Wall time   : {elapsed:.1f}s")
    print(f"Cache hits  : {stats['cache_hits']} / misses: {stats['cache_misses']}")
    print(f"Tokens      : in={stats['total_input_tokens']}  out={stats['total_output_tokens']}")

    # Decision rubric from the plan
    pct = hits / len(samples) if samples else 0
    print("\n=== Verdict ===")
    if pct >= 8 / 15:
        print(f"PROMISING ({hits}/{len(samples)} ≥ 8/15 rate) — enable billing, run full 156.")
    elif pct >= 3 / 15:
        print(f"MARGINAL ({hits}/{len(samples)}) — worth one more iteration before committing.")
    else:
        print(f"REJECT ({hits}/{len(samples)} < 3/15) — upscaling doesn't save EXP-18 with this model.")


if __name__ == "__main__":
    main()
