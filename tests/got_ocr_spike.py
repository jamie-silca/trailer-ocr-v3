"""
EXP-26 spike1 -- GOT-OCR-2.0-hf on 5 portraits + 5 wides.

Pre-flight verification before any benchmark wiring (memory rule: re-run a
load-bearing spot-check before building on it).

Runs in .venv-got-ocr/. Uses the upstreamed transformers
GotOcr2ForConditionalGeneration via AutoModelForImageTextToText.generate().

Decision rule (per docs/ocr-performance-experiment-26.md):
  Portrait gate:  >=1/5 portrait JBHZ within edit-2 AND median <=30 s/crop
                  -> proceed to spike2.
  Horizontal gate: 5/5 wides EXACT after loop-dedupe.
                  Less than 3 -> wide A/B path is dead.

Loop-dedupe note: the model produces (X)(X)(X)... on clean horizontal
crops where X is the actual text. Post-process collapses immediate
repetitions of any 4+ char run.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260423"
ANN_FILE = DATASET / "annotations_2026-04-23_11-24_coco_with_text.json"
PORTRAIT_IDS = [17, 73, 165, 255, 356]   # same crops as EXP-19 / EXP-25 / EXP-27 spike1
# 5 wides where PaddleOCR EXP-23 succeeded -- baseline-friendly so a fail signals model failure not data.
WIDE_IDS_CANDIDATE = None  # picked at runtime; first 5 wides where PaddleOCR succeeded.

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
    """Strip the chat template prefix; assistant reply is after the last 'assistant\\n'."""
    if ASSISTANT_MARKER in text:
        return text.split(ASSISTANT_MARKER, 1)[-1].strip()
    return text.strip()


def loop_dedupe(s: str) -> str:
    """If s is XXXX... where X >=4 chars, return X. Else return s.

    Specifically: scan candidate prefix lengths 4-12; if s == prefix * k for k >=2
    (allowing partial last instance), return prefix.
    """
    s = s.strip()
    n = len(s)
    if n < 8:
        return s
    for plen in range(4, min(n // 2 + 1, 13)):
        prefix = s[:plen]
        # Check if prefix repeats at least twice
        if s.startswith(prefix * 2):
            # Find how much repeats; allow partial tail
            i = 0
            while i + plen <= n and s[i:i+plen] == prefix:
                i += plen
            # If the tail is a prefix-of-prefix, accept
            if i == n or prefix.startswith(s[i:n]):
                return prefix
    return s


def find_wide_ids(coco: dict, n: int = 5) -> list[int]:
    """Pick first n wide annotations (aspect ratio >= 2.0) sorted by ann_id."""
    img_by_id = {i["id"]: i for i in coco["images"]}
    wides = []
    for a in sorted(coco["annotations"], key=lambda x: x["id"]):
        bx = a["bbox"]
        w, h = bx[2], bx[3]
        if h <= 0:
            continue
        aspect = w / h
        if aspect < 2.0:
            continue
        gt = (a.get("text") or "").strip()
        if not gt:
            continue
        wides.append(a["id"])
        if len(wides) >= n:
            break
    return wides


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
    print(f"  loaded in {time.perf_counter()-t0:.1f}s, class={type(model).__name__}")

    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    img_by_id = {i["id"]: i for i in coco["images"]}

    wide_ids = find_wide_ids(coco, 5)
    print(f"\nportrait ann_ids: {PORTRAIT_IDS}")
    print(f"wide ann_ids:     {wide_ids}")

    def run_one(ann_id: int) -> tuple[Optional[str], int]:
        ann = ann_by_id[ann_id]
        img_meta = img_by_id[ann["image_id"]]
        img = Image.open(DATASET / img_meta["file_name"]).convert("RGB")
        x, y, w, h = ann["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        t0 = time.perf_counter()
        inputs = processor(images=crop, return_tensors="pt")
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw = processor.batch_decode(gen, skip_special_tokens=True)[0]
        dt_ms = int((time.perf_counter() - t0) * 1000)
        return decode_assistant(raw), dt_ms

    print(f"\n=== Portrait probes (5) ===")
    print(f"{'ann':>5} {'gt':12} {'got_raw':30} {'deduped':18} {'ed':>3} {'ms':>7}")
    print("-" * 86)
    p_near = 0; p_exact = 0; p_total_ms = 0
    for aid in PORTRAIT_IDS:
        gt = (ann_by_id[aid].get("text") or "").strip().upper()
        text, ms = run_one(aid)
        p_total_ms += ms
        deduped = loop_dedupe(re.sub(r"\s+", "", text.upper()))
        ed = levenshtein(deduped, gt) if gt else None
        is_exact = bool(gt) and deduped == gt
        is_near = ed is not None and ed <= 2
        if is_exact: p_exact += 1
        if is_near: p_near += 1
        print(f"{aid:>5} {gt:12} {text[:30]:30} {deduped[:18]:18} {ed if ed is not None else '-':>3} {ms:>7}")

    print(f"\n  PORTRAIT: EXACT {p_exact}/5, NEAR(<=2) {p_near}/5, avg {p_total_ms//5}ms/crop")

    print(f"\n=== Wide probes (5) ===")
    print(f"{'ann':>5} {'gt':14} {'got_raw':40} {'deduped':14} {'ed':>3} {'ms':>7}")
    print("-" * 90)
    w_exact = 0; w_total_ms = 0
    for aid in wide_ids:
        gt = (ann_by_id[aid].get("text") or "").strip().upper()
        text, ms = run_one(aid)
        w_total_ms += ms
        deduped = loop_dedupe(re.sub(r"\s+", "", text.upper()))
        ed = levenshtein(deduped, gt) if gt else None
        is_exact = bool(gt) and deduped == gt
        if is_exact: w_exact += 1
        print(f"{aid:>5} {gt:14} {text[:40]:40} {deduped[:14]:14} {ed if ed is not None else '-':>3} {ms:>7}")

    print(f"\n  WIDE:     EXACT {w_exact}/5, avg {w_total_ms//5}ms/crop")

    print(f"\n=== Verdict (per docs/ocr-performance-experiment-26.md) ===")
    portrait_pass = p_near >= 1 and (p_total_ms // 5) <= 30000
    wide_pass = w_exact >= 3
    print(f"  Portrait gate (>=1/5 NEAR AND median <=30s): {'PASS' if portrait_pass else 'FAIL'}")
    print(f"  Wide gate (>=3/5 EXACT after dedupe):        {'PASS' if wide_pass else 'FAIL'}")
    if not portrait_pass and not wide_pass:
        print(f"  -> abort EXP-26.")
    elif portrait_pass and wide_pass:
        print(f"  -> widen both buckets (spike2).")
    else:
        print(f"  -> widen the {'wide' if wide_pass else 'portrait'} bucket only.")


if __name__ == "__main__":
    main()
