"""
EXP-25 spike2 — Qwen3-VL-8B across the full 119 portrait crops of 20260423.

After the 5-crop spike returned 4/5 EXACT including 3 JBHZ hits, this widens
the run to characterise the full portrait bucket. The question: how many
unique wins vs PaddleOCR EXP-23, and how many of those are JBHZ (the
load-bearing 0/89 sub-bucket).

Outputs:
  tests/results/qwen_spike_8b_20260423.json  -- per-crop rows + summary.
"""
from __future__ import annotations
import base64
import io
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260423"
ANN_FILE = DATASET / "annotations_2026-04-23_11-24_coco_with_text.json"
_MODEL_SLUG = os.environ.get("QWEN_MODEL", "qwen/qwen3-vl-8b-instruct").split("/")[-1].replace(".", "").replace("-instruct", "")
OUT_FILE = PROJECT_ROOT / "tests" / "results" / f"qwen_spike_{_MODEL_SLUG}_20260423.json"

PROMPT = (
    "The image shows a trailer ID plate. Characters are upright and stacked "
    "vertically - read them top-to-bottom. Valid formats: JBHZ + 6 digits, "
    "JBHU + 6 digits, or R + 5 digits. Return only the ID string with no "
    "punctuation or whitespace. If you cannot read it with confidence, return "
    "exactly UNKNOWN."
)

MIN_LONG_SIDE_PX = 768
MODEL_ID = os.environ.get("QWEN_MODEL", "qwen/qwen3-vl-8b-instruct")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0


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


def encode_png_b64(img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    long_side = max(img.size)
    if long_side < MIN_LONG_SIDE_PX:
        scale = MIN_LONG_SIDE_PX / long_side
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_qwen(b64: str, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL_ID,
        "temperature": 0,
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            r = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=90)
            dt_ms = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (attempt + 1))
                continue
            j = r.json()
            raw = (j["choices"][0]["message"]["content"] or "").strip()
            usage = j.get("usage", {})
            return {
                "raw": raw,
                "ms": round(dt_ms, 1),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "attempts": attempt + 1,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    return {"error": last_err, "attempts": MAX_RETRIES}


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
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

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

    print(f"Model: {MODEL_ID}")
    print(f"Portrait crops in 20260423: {len(portraits)}")
    print()

    rows = []
    t_start = time.perf_counter()
    for i, ann in enumerate(portraits, 1):
        gt = (ann.get("text") or "").strip().upper()
        fmt = gt_format(gt)
        img_meta = img_by_id[ann["image_id"]]
        img = Image.open(DATASET / img_meta["file_name"]).convert("RGB")
        x, y, w, h = ann["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        b64 = encode_png_b64(crop)

        result = call_qwen(b64, api_key)
        if "error" in result:
            rows.append({
                "ann_id": ann["id"],
                "image": img_meta["file_name"],
                "gt": gt,
                "gt_format": fmt,
                "qwen_raw": "",
                "edit_distance": None,
                "exact": False,
                "near": False,
                "unknown": False,
                "error": result["error"],
                "ms": None,
            })
        else:
            raw_clean = re.sub(r"\s+", "", result["raw"].upper())
            is_unknown = (raw_clean == "UNKNOWN")
            ed = levenshtein(raw_clean, gt) if (gt and not is_unknown) else None
            rows.append({
                "ann_id": ann["id"],
                "image": img_meta["file_name"],
                "gt": gt,
                "gt_format": fmt,
                "qwen_raw": raw_clean,
                "edit_distance": ed,
                "exact": bool(gt) and not is_unknown and raw_clean == gt,
                "near": (ed is not None and ed <= 2),
                "unknown": is_unknown,
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "ms": result.get("ms"),
            })

        if i % 10 == 0 or i == len(portraits):
            elapsed = time.perf_counter() - t_start
            rate = i / elapsed
            eta = (len(portraits) - i) / rate if rate > 0 else 0
            done_exact = sum(1 for r in rows if r["exact"])
            print(f"  {i}/{len(portraits)}  exact={done_exact}  rate={rate:.2f}/s  eta={eta:.0f}s")

    wall = time.perf_counter() - t_start

    by_fmt = {}
    for r in rows:
        by_fmt.setdefault(r["gt_format"], []).append(r)

    valid_ms = [r["ms"] for r in rows if r.get("ms") is not None]
    median_ms = statistics.median(valid_ms) if valid_ms else 0
    print(f"\nWall: {wall:.1f}s  ({len(rows)} crops, median {median_ms:.0f}ms/crop)\n")
    print(f"{'gt_format':16s} {'n':>4s} {'exact':>6s} {'near':>5s} {'unk':>5s} {'err':>5s}")
    for fmt in sorted(by_fmt):
        items = by_fmt[fmt]
        n = len(items)
        exact = sum(1 for r in items if r["exact"])
        near = sum(1 for r in items if r["near"] and not r["exact"])
        unk = sum(1 for r in items if r["unknown"])
        err = sum(1 for r in items if r.get("error"))
        print(f"{fmt:16s} {n:>4d} {exact:>6d} {near:>5d} {unk:>5d} {err:>5d}")

    total_exact = sum(1 for r in rows if r["exact"])
    total_near = sum(1 for r in rows if r["near"] and not r["exact"])
    total_unk = sum(1 for r in rows if r["unknown"])
    total_err = sum(1 for r in rows if r.get("error"))
    print(f"\nOVERALL: EXACT {total_exact}/{len(rows)}, NEAR(<=2) {total_near}/{len(rows)}, "
          f"UNKNOWN {total_unk}, ERROR {total_err}")

    # JBHZ-specific
    jbhz_rows = [r for r in rows if r["gt"].startswith("JBHZ")]
    jbhu_rows = [r for r in rows if r["gt"].startswith("JBHU")]
    jbhz_exact = sum(1 for r in jbhz_rows if r["exact"])
    jbhu_exact = sum(1 for r in jbhu_rows if r["exact"])
    print(f"\nJBHZ sub-bucket: {jbhz_exact}/{len(jbhz_rows)} EXACT  (PaddleOCR EXP-23: 0/89-ish)")
    print(f"JBHU sub-bucket: {jbhu_exact}/{len(jbhu_rows)} EXACT")

    in_tok = sum((r.get("input_tokens") or 0) for r in rows)
    out_tok = sum((r.get("output_tokens") or 0) for r in rows)
    print(f"\nTokens: input {in_tok}, output {out_tok}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "model": MODEL_ID,
        "n": len(rows),
        "wall_s": round(wall, 1),
        "median_ms": median_ms,
        "summary": {
            fmt: {
                "n": len(items),
                "exact": sum(1 for r in items if r["exact"]),
                "near": sum(1 for r in items if r["near"] and not r["exact"]),
                "unknown": sum(1 for r in items if r["unknown"]),
                "error": sum(1 for r in items if r.get("error")),
            } for fmt, items in by_fmt.items()
        },
        "jbhz_exact": jbhz_exact,
        "jbhz_n": len(jbhz_rows),
        "jbhu_exact": jbhu_exact,
        "jbhu_n": len(jbhu_rows),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
