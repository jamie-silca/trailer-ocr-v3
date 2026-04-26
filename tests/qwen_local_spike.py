"""
EXP-27 spike1 -- Qwen3-VL-8B local via Ollama, same 5 crops as EXP-25 spike1.

Pre-flight verification before any benchmark wiring (memory rule: re-run a
load-bearing spot-check before building on it).

Bypasses qwen_local_processor.py and calls the Ollama chat-completions
endpoint directly so we capture the *raw, pre-format-gate* model output --
the format gate would discard 1-char-off responses that are still
informative for the spike decision (the gate is for production cascade,
not the spike).

Inline comparison vs the OpenRouter spike2 result
(tests/results/qwen_spike_8b_20260423.json) for the same ann_ids.

Decision rule:
  - 0/5 within edit-distance 2  -> abort EXP-27, no widen.
  - >=1/5 within edit-2          -> widen to 119 crops via qwen_local_spike2.py.

Outputs to stdout only; no JSON written for spike 1 (kept lightweight).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260423"
ANN_FILE = DATASET / "annotations_2026-04-23_11-24_coco_with_text.json"
PICK_IDS = [17, 73, 165, 255, 356]
OPENROUTER_SPIKE_FILE = PROJECT_ROOT / "tests" / "results" / "qwen_spike_8b_20260423.json"

# Identical wording to tests/qwen_portrait.py (EXP-25) so spike-vs-EXP-25
# numbers compare apples-to-apples on capable models. Smaller / weaker
# instruction-followers (moondream, granite-vision) misinterpret the long
# prompt as a description request and either narrate or refuse -- override
# via SPIKE_PROMPT env to "Read the text." for those probes.
DEFAULT_PROMPT = (
    "The image shows a trailer ID plate. Characters are upright and stacked "
    "vertically - read them top-to-bottom. Valid formats: JBHZ + 6 digits, "
    "JBHU + 6 digits, or R + 5 digits. Return only the ID string with no "
    "punctuation or whitespace. If you cannot read it with confidence, return "
    "exactly UNKNOWN."
)
PROMPT = os.environ.get("SPIKE_PROMPT") or DEFAULT_PROMPT

MIN_LONG_SIDE_PX = 768
MODEL_ID = os.environ.get("QWEN_LOCAL_MODEL", "qwen3-vl:8b")


def _resolve_ollama_url() -> str:
    # OLLAMA_URL wins if set (full URL). Otherwise OLLAMA_HOST may be a bare
    # host (Ollama's own server bind var, e.g. "0.0.0.0" -- not useful as a
    # client URL); only honour it if it has a scheme. Default to localhost.
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
        # qwen3-vl is a thinking model: it emits a long chain-of-thought
        # before the answer. With max_tokens=32 (EXP-25 default for the
        # non-thinking OpenRouter route) the budget is consumed entirely by
        # reasoning and `content` returns empty. 512 leaves comfortable
        # headroom for ~120 thinking + 10 answer tokens.
        "max_tokens": 512,
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
    r = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=REQUEST_TIMEOUT_S)
    dt_ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "ms": round(dt_ms, 0)}
    j = r.json()
    raw = (j["choices"][0]["message"]["content"] or "").strip()
    usage = j.get("usage") or {}
    return {
        "raw": raw,
        "ms": round(dt_ms, 0),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }


def load_openrouter_lookup() -> dict:
    if not OPENROUTER_SPIKE_FILE.exists():
        return {}
    data = json.loads(OPENROUTER_SPIKE_FILE.read_text(encoding="utf-8"))
    return {r["ann_id"]: r for r in data.get("rows", [])}


def main():
    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    img_by_id = {i["id"]: i for i in coco["images"]}
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    or_lookup = load_openrouter_lookup()

    print(f"Model: {MODEL_ID}")
    print(f"Endpoint: {ENDPOINT}")
    print(f"OpenRouter cross-ref: {'loaded' if or_lookup else 'not found'}")
    print()
    header = f"{'ann':>5} {'gt':12} {'local_raw':20} {'or_raw':14} {'l_ed':>4} {'or_ed':>5} {'ms':>7}"
    print(header)
    print("-" * len(header))

    near_count = 0
    exact_count = 0
    total_ms = 0.0
    n_called = 0
    for ann_id in PICK_IDS:
        ann = ann_by_id[ann_id]
        gt = (ann.get("text") or "").strip().upper()
        img_meta = img_by_id[ann["image_id"]]
        img = Image.open(DATASET / img_meta["file_name"]).convert("RGB")
        x, y, w, h = ann["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        b64 = encode_png_b64(crop)

        result = call_ollama(b64)
        if "error" in result:
            print(f"{ann_id:>5} {gt:12} ERROR: {result['error'][:80]}")
            continue
        total_ms += result["ms"]
        n_called += 1

        raw_clean = re.sub(r"\s+", "", result["raw"].upper())
        local_ed = levenshtein(raw_clean, gt) if gt else None
        is_exact = bool(gt) and raw_clean == gt
        is_near = local_ed is not None and local_ed <= 2
        if is_exact:
            exact_count += 1
        if is_near:
            near_count += 1

        or_row = or_lookup.get(ann_id)
        or_raw = (or_row.get("qwen_raw") or "") if or_row else ""
        or_ed = or_row.get("edit_distance") if or_row else None

        print(
            f"{ann_id:>5} {gt:12} {raw_clean[:20]:20} {or_raw[:14]:14} "
            f"{local_ed if local_ed is not None else '-':>4} "
            f"{or_ed if or_ed is not None else '-':>5} {result['ms']:>7.0f}"
        )

    print("-" * len(header))
    print(
        f"LOCAL: EXACT {exact_count}/{len(PICK_IDS)}, NEAR(<=2) {near_count}/{len(PICK_IDS)}, "
        f"avg {total_ms / max(n_called, 1):.0f} ms/crop"
    )
    print()
    if near_count >= 1:
        print(f"PASS: >=1/5 within edit-2. Widen via qwen_local_spike2.py.")
    else:
        print(f"FAIL: 0/5 within edit-2. Abort EXP-27 or check prompt/quant.")


if __name__ == "__main__":
    main()
