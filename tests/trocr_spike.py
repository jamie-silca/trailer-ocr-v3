"""
EXP-20 spike — quick TrOCR probe against EXP-13 debug images.

Ground truth for ann14 = JBHZ672061.

Inputs to test (all from tests/results/exp13_debug/):
  - ann14_crop.png          — raw portrait crop (stacked vertical)
  - ann14_rect_stitch.png   — EXP-13 perspective-rectified stitched horizontal strip
  - ann14_rectified.png     — rectified column only (no stitch)

If TrOCR can read the stitched strip, it's worth wiring into the benchmark.
If it produces gibberish on the clean stitched strip, abandon EXP-20 early.
"""
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DEBUG_DIR = PROJECT_ROOT / "tests" / "results" / "exp13_debug"

MODELS = [
    "microsoft/trocr-small-printed",
    "microsoft/trocr-base-printed",
]

GT = "JBHZ672061"
IMAGES = [
    "ann14_crop.png",
    "ann14_rect_stitch.png",
    "ann14_rectified.png",
]


def main(model_id: str) -> None:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch

    print(f"\n=== Loading {model_id} ===")
    t0 = time.perf_counter()
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    print(f"  GT = {GT}")
    for name in IMAGES:
        path = DEBUG_DIR / name
        if not path.exists():
            print(f"  [missing] {name}")
            continue
        img = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=20)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        dt = (time.perf_counter() - t0) * 1000
        match = "OK" if text.strip().upper().replace(" ", "") == GT else "xx"
        print(f"  [{match}] {name:28s} {img.size}  ->  {text!r:30s}  ({dt:.0f} ms)")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else MODELS
    for m in models:
        try:
            main(m)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED on {m}: {type(e).__name__}: {e}")
