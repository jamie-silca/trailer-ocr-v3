"""
EXP-25 spike — Qwen2.5-VL-7B on the same 5 portrait crops EXP-19 used.

Pre-flight verification before any benchmark wiring (memory rule: re-run a
load-bearing spot-check before building on it).

Decision rule:
  - 0/5 within edit-distance 2 of GT  -> abort EXP-25, no wiring.
  - >=1/5 within edit-2                -> widen to 119 crops via qwen_spike2.py.

Outputs to stdout only; no JSON written for spike 1 (kept lightweight).
"""
from __future__ import annotations
import base64
import io
import json
import os
import re
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
PICK_IDS = [17, 73, 165, 255, 356]

# Same prompt text as tests/vlm_portrait.py (EXP-18) for apples-to-apples.
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
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    }
    t0 = time.perf_counter()
    r = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=60)
    dt_ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "ms": round(dt_ms, 0)}
    j = r.json()
    raw = (j["choices"][0]["message"]["content"] or "").strip()
    usage = j.get("usage", {})
    return {
        "raw": raw,
        "ms": round(dt_ms, 0),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env or environment.", file=sys.stderr)
        sys.exit(1)

    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    img_by_id = {i["id"]: i for i in coco["images"]}
    ann_by_id = {a["id"]: a for a in coco["annotations"]}

    print(f"Model: {MODEL_ID}")
    print(f"Prompt: {PROMPT[:80]}...")
    print()
    print(f"{'ann':>5} {'gt':12} {'qwen_raw':25} {'edit':>4} {'ms':>5} {'in_tok':>7} {'out_tok':>7}")
    print("-" * 75)

    near_count = 0
    exact_count = 0
    for ann_id in PICK_IDS:
        ann = ann_by_id[ann_id]
        gt = (ann.get("text") or "").strip().upper()
        img_meta = img_by_id[ann["image_id"]]
        img = Image.open(DATASET / img_meta["file_name"]).convert("RGB")
        x, y, w, h = ann["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        b64 = encode_png_b64(crop)

        result = call_qwen(b64, api_key)
        if "error" in result:
            print(f"{ann_id:>5} {gt:12} ERROR: {result['error']}")
            continue

        raw_clean = re.sub(r"\s+", "", result["raw"].upper())
        ed = levenshtein(raw_clean, gt) if gt else None
        is_exact = bool(gt) and raw_clean == gt
        is_near = ed is not None and ed <= 2

        if is_exact:
            exact_count += 1
        if is_near:
            near_count += 1

        print(f"{ann_id:>5} {gt:12} {raw_clean[:25]:25} {ed if ed is not None else '-':>4} "
              f"{result['ms']:>5} {result.get('input_tokens') or '-':>7} {result.get('output_tokens') or '-':>7}")

    print("-" * 75)
    print(f"EXACT: {exact_count}/{len(PICK_IDS)}, NEAR(<=2): {near_count}/{len(PICK_IDS)}")
    print()
    if near_count >= 1:
        print(f"PASS: >=1/5 within edit-2. Widen to 119 crops via qwen_spike2.py.")
    else:
        print(f"FAIL: 0/5 within edit-2. Consider abort or single fallback variant.")


if __name__ == "__main__":
    main()
