"""
EXP-18 plumbing check — does Gemini read EASY (wide-bucket) crops?

If yes: our system works, portraits are genuinely the hard case for Gemini.
If no:  our pipeline has a bug (encoding, prompt, gate) that's masking as a model failure.

Uses a generic prompt (not the portrait-specific one) and bypasses the format gate
so we see raw model output. Paces calls at ~15s apart to respect the 5 RPM cap.
"""
from __future__ import annotations
import io, json, random, sys, time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "dataset" / "20260406"
ANN_FILE = DATASET / "annotations_2026-04-06_09-06_coco_with_text.json"

N = 5
SLEEP_BETWEEN_CALLS_S = 15

GENERIC_PROMPT = (
    "Read the trailer ID printed on this plate. Return ONLY the ID string with "
    "no punctuation or whitespace. If you cannot read it, return exactly UNKNOWN."
)


def pick_wide(coco, n, seed=7):
    by_id = {i["id"]: i for i in coco["images"]}
    cands = []
    for ann in coco["annotations"]:
        gt = (ann.get("text") or "").strip()
        if not gt:
            continue
        x, y, w, h = ann["bbox"]
        if h == 0:
            continue
        ratio = w / h
        if 2.0 <= ratio < 4.0:  # "wide" bucket
            cands.append({"ann": ann["id"], "gt": gt, "bbox": ann["bbox"],
                          "file": by_id[ann["image_id"]]["file_name"],
                          "w": int(w), "h": int(h)})
    rng = random.Random(seed); rng.shuffle(cands)
    return cands[:n]


def main():
    from dotenv import load_dotenv
    load_dotenv()
    from google import genai
    from google.genai import types

    coco = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    samples = pick_wide(coco, N)

    client = genai.Client()
    cfg = types.GenerateContentConfig(
        temperature=0, max_output_tokens=32,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    print(f"\n=== EXP-18 plumbing check — Gemini 2.5 Flash on {N} WIDE crops ===\n")
    print(f"{'#':>2}  {'ann':>4}  {'orig':>9}  {'sent':>10}  {'gt':>14}  {'vlm-raw':>18}  {'match':>6}")
    print("-" * 80)

    hits = 0
    for i, s in enumerate(samples, 1):
        img = Image.open(DATASET / s["file"]).convert("RGB")
        x, y, w, h = s["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        # Upscale same as portrait path
        long_side = max(crop.size)
        if long_side < 768:
            scale = 768 / long_side
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        buf = io.BytesIO(); crop.save(buf, format="PNG")
        sent_size = f"{crop.size[0]}x{crop.size[1]}"

        try:
            r = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"), GENERIC_PROMPT],
                config=cfg,
            )
            raw = (r.text or "").strip()
            norm = raw.upper().replace(" ", "")
            gt_norm = s["gt"].upper().replace(" ", "")
            match = "OK" if norm == gt_norm else "WRONG"
            if match == "OK":
                hits += 1
        except Exception as e:
            raw = f"ERR: {type(e).__name__}"
            match = "ERR"
        print(f"{i:>2}  {s['ann']:>4}  {s['w']}x{s['h']:<5}  {sent_size:>10}  {s['gt']:>14}  {raw!r:>18}  {match:>6}")
        if i < len(samples):
            time.sleep(SLEEP_BETWEEN_CALLS_S)

    print("-" * 80)
    print(f"\nHits on WIDE bucket: {hits} / {N}")
    if hits >= 3:
        print("System plumbing OK — Gemini reads easy crops. Portrait failure is model-specific.")
    elif hits >= 1:
        print("System partially working — some plumbing issue OR Gemini Flash is just weak.")
    else:
        print("System-level failure — even easy crops fail. Investigate pipeline/prompt/encoding.")


if __name__ == "__main__":
    main()
