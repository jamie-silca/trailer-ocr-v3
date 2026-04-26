"""
EXP-27 spike2 -- Qwen3-VL-8B local via Ollama, full 119-crop portrait sweep.

Mirrors tests/qwen_spike2.py (EXP-25 OpenRouter widen) so the per-crop rows
diff cleanly against tests/results/qwen_spike_8b_20260423.json.

Bypasses qwen_local_processor.py and calls Ollama directly to capture *raw,
pre-format-gate* output -- the format gate is for production cascade, not
the spike measurement.

Output: tests/results/qwen_local_spike_qwen3vl8b_20260423.json with the
same row schema (ann_id, image, gt, gt_format, qwen_raw, edit_distance,
exact, near, unknown, ms, ...).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import statistics
import time
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260423"
ANN_FILE = DATASET / "annotations_2026-04-23_11-24_coco_with_text.json"
_MODEL_TAG = os.environ.get("QWEN_LOCAL_MODEL", "qwen3-vl:8b").replace(":", "").replace(".", "").replace("-", "")
OUT_FILE = PROJECT_ROOT / "tests" / "results" / f"qwen_local_spike_{_MODEL_TAG}_20260423.json"

PROMPT = (
    "The image shows a trailer ID plate. Characters are upright and stacked "
    "vertically - read them top-to-bottom. Valid formats: JBHZ + 6 digits, "
    "JBHU + 6 digits, or R + 5 digits. Return only the ID string with no "
    "punctuation or whitespace. If you cannot read it with confidence, return "
    "exactly UNKNOWN."
)

MIN_LONG_SIDE_PX = 768
MODEL_ID = os.environ.get("QWEN_LOCAL_MODEL", "qwen3-vl:8b")


def _resolve_ollama_url() -> str:
    raw = os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST") or ""
    raw = raw.strip().rstrip("/")
    if raw.startswith(("http://", "https://")):
        return raw
    return "http://localhost:11434"


ENDPOINT = f"{_resolve_ollama_url()}/v1/chat/completions"
REQUEST_TIMEOUT_S = 300.0


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


def call_ollama(b64: str) -> dict:
    headers = {"Content-Type": "application/json"}
    body = {
        "model": MODEL_ID,
        "temperature": 0,
        # qwen3-vl is a thinking model -- 32 tokens is consumed by reasoning.
        # 512 leaves headroom for ~120 thinking + 10 answer tokens.
        "max_tokens": 512,
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
    try:
        t0 = time.perf_counter()
        r = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=REQUEST_TIMEOUT_S)
        dt_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "ms": round(dt_ms, 1)}
        j = r.json()
        raw = (j["choices"][0]["message"]["content"] or "").strip()
        usage = j.get("usage") or {}
        return {
            "raw": raw,
            "ms": round(dt_ms, 1),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:140]}"}


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
    print(f"Endpoint: {ENDPOINT}")
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

        result = call_ollama(b64)
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
                "ms": result.get("ms"),
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

        if i % 5 == 0 or i == len(portraits):
            elapsed = time.perf_counter() - t_start
            rate = i / elapsed
            eta = (len(portraits) - i) / rate if rate > 0 else 0
            done_exact = sum(1 for r in rows if r["exact"])
            print(f"  {i}/{len(portraits)}  exact={done_exact}  rate={rate:.2f}/s  eta={eta:.0f}s",
                  flush=True)

    wall = time.perf_counter() - t_start

    by_fmt = {}
    for r in rows:
        by_fmt.setdefault(r["gt_format"], []).append(r)

    valid_ms = [r["ms"] for r in rows if r.get("ms") is not None]
    median_ms = statistics.median(valid_ms) if valid_ms else 0
    p95_ms = sorted(valid_ms)[int(len(valid_ms) * 0.95)] if len(valid_ms) >= 20 else None
    print(f"\nWall: {wall:.1f}s  ({len(rows)} crops, median {median_ms:.0f}ms, p95 {p95_ms}ms)\n")
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

    jbhz_rows = [r for r in rows if r["gt"].startswith("JBHZ")]
    jbhu_rows = [r for r in rows if r["gt"].startswith("JBHU")]
    jbhz_exact = sum(1 for r in jbhz_rows if r["exact"])
    jbhu_exact = sum(1 for r in jbhu_rows if r["exact"])
    print(f"\nJBHZ sub-bucket: {jbhz_exact}/{len(jbhz_rows)} EXACT  "
          f"(EXP-25 OpenRouter Qwen3-VL-8B: 47/92)")
    print(f"JBHU sub-bucket: {jbhu_exact}/{len(jbhu_rows)} EXACT")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "model": MODEL_ID,
        "endpoint": ENDPOINT,
        "n": len(rows),
        "wall_s": round(wall, 1),
        "median_ms": median_ms,
        "p95_ms": p95_ms,
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
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
