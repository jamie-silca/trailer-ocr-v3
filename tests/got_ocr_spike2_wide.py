"""
EXP-26 spike2 (wide bucket only) -- GOT-OCR-2.0-hf vs PaddleOCR baseline
on all 255 wide annotations of tests/dataset/20260423.

Per spike1 (got_ocr_spike.py): portrait gate failed (0/5 NEAR), wide gate
passed (5/5 EXACT on baseline-easy crops). The load-bearing question
remaining: does GOT-OCR rescue the 33 wide crops where PaddleOCR (EXP-23)
fails, while not regressing on the ~222 it passes?

Cross-references PaddleOCR pass/fail per ann_id from
tests/results/benchmark_EXP-23_20260425_153428.json.

Output: tests/results/got_ocr_spike2_wide_20260423.json with per-crop rows
+ summary (rescues, regressions, EXACT @ wide bucket overall).
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260423"
ANN_FILE = DATASET / "annotations_2026-04-23_11-24_coco_with_text.json"
EXP23_FILE = PROJECT_ROOT / "tests" / "results" / "benchmark_EXP-23_20260425_153428.json"
OUT_FILE = PROJECT_ROOT / "tests" / "results" / "got_ocr_spike2_wide_20260423.json"

ASSISTANT_MARKER = "assistant\n"


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


def decode_assistant(text: str) -> str:
    if ASSISTANT_MARKER in text:
        return text.split(ASSISTANT_MARKER, 1)[-1].strip()
    return text.strip()


def loop_dedupe(s: str) -> str:
    s = s.strip()
    n = len(s)
    if n < 8:
        return s
    for plen in range(4, min(n // 2 + 1, 13)):
        prefix = s[:plen]
        if s.startswith(prefix * 2):
            i = 0
            while i + plen <= n and s[i:i + plen] == prefix:
                i += plen
            if i == n or prefix.startswith(s[i:n]):
                return prefix
    return s


def main():
    print("Loading GOT-OCR-2.0-hf...")
    t0 = time.perf_counter()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "stepfun-ai/GOT-OCR-2.0-hf",
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    img_by_id = {i["id"]: i for i in coco["images"]}
    ann_by_id = {a["id"]: a for a in coco["annotations"]}

    exp23 = json.loads(EXP23_FILE.read_text(encoding="utf-8"))
    exp23_by_ann = {r["annotation_id"]: r for r in exp23["annotation_results"]}

    # All wide annotations, in stable order
    wides = []
    for a in coco["annotations"]:
        bx = a["bbox"]
        w, h = bx[2], bx[3]
        if h <= 0:
            continue
        if w / h < 2.0:
            continue
        wides.append(a)
    wides.sort(key=lambda a: a["id"])
    print(f"Wide crops: {len(wides)}")

    rows = []
    t_start = time.perf_counter()
    for i, ann in enumerate(wides, 1):
        gt = (ann.get("text") or "").strip().upper()
        img_meta = img_by_id[ann["image_id"]]
        img = Image.open(DATASET / img_meta["file_name"]).convert("RGB")
        x, y, w, h = ann["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))

        t0 = time.perf_counter()
        try:
            inputs = processor(images=crop, return_tensors="pt")
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            raw = processor.batch_decode(gen, skip_special_tokens=True)[0]
            err = None
        except Exception as e:
            raw = ""
            err = f"{type(e).__name__}: {str(e)[:140]}"
        dt_ms = round((time.perf_counter() - t0) * 1000, 1)

        decoded = decode_assistant(raw) if raw else ""
        cleaned = re.sub(r"\s+", "", decoded.upper())
        deduped = loop_dedupe(cleaned)
        ed = levenshtein(deduped, gt) if gt else None
        is_exact = bool(gt) and deduped == gt

        paddle_row = exp23_by_ann.get(ann["id"]) or {}
        paddle_text = (paddle_row.get("ocr_text") or "").upper()
        paddle_exact = bool(paddle_row.get("exact_match"))

        rows.append({
            "ann_id": ann["id"],
            "image": img_meta["file_name"],
            "gt": gt,
            "got_raw": deduped,
            "got_full_decoded": decoded[:200],
            "got_exact": is_exact,
            "got_ed": ed,
            "paddle_text": paddle_text,
            "paddle_exact": paddle_exact,
            "rescue": is_exact and not paddle_exact,
            "regression": paddle_exact and not is_exact,
            "ms": dt_ms,
            "error": err,
        })

        if i % 10 == 0 or i == len(wides):
            elapsed = time.perf_counter() - t_start
            rate = i / elapsed
            eta = (len(wides) - i) / rate if rate > 0 else 0
            got_exact_so_far = sum(1 for r in rows if r["got_exact"])
            rescues = sum(1 for r in rows if r["rescue"])
            regressions = sum(1 for r in rows if r["regression"])
            print(f"  {i}/{len(wides)}  got_exact={got_exact_so_far}  "
                  f"rescues={rescues}  regressions={regressions}  "
                  f"rate={rate:.2f}/s  eta={eta:.0f}s",
                  flush=True)

    wall = time.perf_counter() - t_start

    got_exact = sum(1 for r in rows if r["got_exact"])
    paddle_exact_n = sum(1 for r in rows if r["paddle_exact"])
    rescues = sum(1 for r in rows if r["rescue"])
    regressions = sum(1 for r in rows if r["regression"])
    both_correct = sum(1 for r in rows if r["got_exact"] and r["paddle_exact"])
    both_wrong = sum(1 for r in rows if not r["got_exact"] and not r["paddle_exact"])
    valid_ms = [r["ms"] for r in rows if r["ms"] is not None]

    print(f"\nWall: {wall:.1f}s ({len(rows)} crops, median {statistics.median(valid_ms):.0f}ms)")
    print(f"\nGOT-OCR EXACT:    {got_exact}/{len(rows)} ({got_exact/len(rows)*100:.1f}%)")
    print(f"PaddleOCR EXACT:  {paddle_exact_n}/{len(rows)} ({paddle_exact_n/len(rows)*100:.1f}%)")
    print(f"  Rescues   (GOT right, paddle wrong): {rescues}")
    print(f"  Regressions (paddle right, GOT wrong): {regressions}")
    print(f"  Both correct: {both_correct}")
    print(f"  Both wrong:   {both_wrong}")
    print(f"\nNet effect of FULL replacement: {got_exact - paddle_exact_n:+d} crops")
    print(f"Net effect of CASCADE (paddle->got fallback on paddle-fail): "
          f"{paddle_exact_n + rescues}/{len(rows)} = {(paddle_exact_n + rescues)/len(rows)*100:.1f}%")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "model": "stepfun-ai/GOT-OCR-2.0-hf",
        "n": len(rows),
        "wall_s": round(wall, 1),
        "median_ms": statistics.median(valid_ms) if valid_ms else 0,
        "got_exact": got_exact,
        "paddle_exact": paddle_exact_n,
        "rescues": rescues,
        "regressions": regressions,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
