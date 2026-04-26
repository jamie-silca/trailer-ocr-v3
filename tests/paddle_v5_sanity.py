"""
EXP-22 sanity — can PP-OCRv5 read stacked-vertical portrait trailer IDs?

Mirrors tests/vlm_sanity.py: picks N diverse portrait crops and runs them
through PaddleV5Processor. Gates the full 672 run on >= 3/15 correct.

Usage:
    # From the v5 isolated venv:
    .venv-paddle-v5/Scripts/python.exe tests/paddle_v5_sanity.py --n 15
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))

DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260406"
ANN_FILE = DATASET / "annotations_2026-04-06_09-06_coco_with_text.json"


def pick_diverse_portraits(coco, n, seed=42):
    by_id = {i["id"]: i for i in coco["images"]}
    cands = []
    for ann in coco["annotations"]:
        gt = (ann.get("text") or "").strip()
        if not gt:
            continue
        x, y, w, h = ann["bbox"]
        if h <= 2 * w:
            continue
        cands.append({
            "ann": ann["id"], "gt": gt, "bbox": ann["bbox"],
            "file": by_id[ann["image_id"]]["file_name"],
            "w": int(w), "h": int(h),
        })
    rng = random.Random(seed); rng.shuffle(cands)
    picked, seen_files, seen_gts = [], set(), set()
    for c in cands:
        if c["file"] in seen_files and c["gt"] in seen_gts:
            continue
        picked.append(c); seen_files.add(c["file"]); seen_gts.add(c["gt"])
        if len(picked) >= n:
            break
    if len(picked) < n:
        for c in cands:
            if c in picked:
                continue
            picked.append(c)
            if len(picked) >= n:
                break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    samples = pick_diverse_portraits(coco, args.n)

    from paddle_v5_processor import PaddleV5Processor
    ocr = PaddleV5Processor()

    print(f"\n=== EXP-22 sanity — PP-OCRv5 on {len(samples)} portrait crops ===\n")
    print(f"{'#':>2}  {'ann':>4}  {'orig':>9}  {'gt':>14}  {'v5':>14}  {'match':>6}  {'latency':>9}")
    print("-" * 74)

    hits = 0
    start = time.perf_counter()
    for i, s in enumerate(samples, 1):
        img = Image.open(DATASET / s["file"]).convert("RGB")
        x, y, w, h = s["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        t0 = time.perf_counter()
        text, conf = ocr.process_image(crop)
        dt = (time.perf_counter() - t0) * 1000
        gt_norm = s["gt"].upper().replace(" ", "")
        got = (text or "").upper().replace(" ", "")
        match = "OK" if got == gt_norm else ("miss" if not got else "WRONG")
        if match == "OK":
            hits += 1
        print(f"{i:>2}  {s['ann']:>4}  {s['w']}x{s['h']:<5}  {s['gt']:>14}  {(text or '-'):>14}  {match:>6}  {dt:>7.0f}ms")

    elapsed = time.perf_counter() - start
    pct = hits / len(samples) if samples else 0
    print("-" * 74)
    print(f"\nHits: {hits}/{len(samples)} ({100*pct:.1f}%)  wall: {elapsed:.1f}s")
    print("\n=== Verdict ===")
    if hits >= 3:
        print(f"PROCEED to full 672-benchmark ({hits}/{len(samples)} >= 3/15).")
    else:
        print(f"REJECT ({hits}/{len(samples)} < 3/15) — v5 doesn't read stacked-vertical either. Move to EXP-17.")


if __name__ == "__main__":
    main()
