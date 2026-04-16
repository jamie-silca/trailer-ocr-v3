# OCR Performance Experiment 01 — Baseline Benchmarking

## 1. Goal

Establish a repeatable, quantitative baseline for OCR performance on real-world trailer footage before making any changes. This gives us a reference point for every future experiment so we can objectively say whether a change made things better or worse.

Two dimensions are tracked every run:

| Dimension | What it measures |
|---|---|
| **Speed** | How long PaddleOCR takes per annotation crop (avg, median, percentiles) |
| **Accuracy** | Whether the text returned matches the ground truth label (% exact match) |

The app is a BE microservice that receives pre-cropped images from a YOLO detector, so the benchmark mirrors that — it feeds annotation bounding box crops directly into `OcrProcessor.process_image()`, bypassing the detector stage.

---

## 2. Methodology

### Dataset
- **Source:** `tests/dataset/20260406/`
- **Images:** 397 drone frames (DJI, 4000×3000 JPEG)
- **Annotations:** 672 bounding boxes in COCO format (`annotations_2026-04-06_09-06_coco_with_text.json`)
- **Category:** `trailer_id` (text regions on trailer plates)
- **Note:** Annotation JSON does not currently contain ground truth text despite the `_with_text` filename — this was confirmed by inspection. Accuracy scoring will activate automatically once a `text` field is populated per annotation.

### Crop Extraction
COCO bbox format is `[x, y, width, height]` in absolute pixels. Crops are extracted directly from the source image with no resizing or preprocessing. This is intentional — it matches what the production pipeline delivers after YOLO detection.

### OCR Engine
- **Library:** PaddleOCR 2.7.3 (PP-OCRv4 model)
- **Config:** `use_angle_cls=True`, `lang='en'`, `use_gpu=False`
- **Singleton:** Initialised once at startup, reused across all annotations (matches production behaviour)

### Timing
- **Warmup** (first `PaddleOCR()` init) is measured separately and excluded from per-annotation stats
- **Per-annotation time** is the wall clock for a single `OcrProcessor.process_image()` call, including numpy conversion and result parsing
- Image loading and crop extraction are not included in per-annotation time (in production these are handled upstream)

### Accuracy Scoring
When a `text` field is present in an annotation, the benchmark computes exact match (case-insensitive, strip whitespace). Results are stored per-annotation in the JSON output so partial/fuzzy matching can be computed from the raw data later.

### Output
Each run produces two files in `tests/results/`:
- `benchmark_YYYYMMDD_HHMMSS.log` — human-readable log with per-annotation lines and a summary block
- `benchmark_YYYYMMDD_HHMMSS.json` — machine-readable full results including every annotation's OCR output, confidence, elapsed time, and (when available) accuracy

### Reproducibility
```bash
# From project root
python tests/benchmark_ocr.py
```

---

## 3. Baseline Results (Run 01 — 2026-04-15)

> **PC note:** Run 01 was taken on default Windows power settings (balanced/power-saver). Run 02 below was taken with high-performance settings and is the canonical baseline.

### Run 01 — `20260415_113544` (Balanced power, warm GPU idle)

| Metric | Value |
|---|---|
| Annotations processed | 672 |
| OCR returned text | 294 (43.8%) |
| OCR returned nothing | 378 (56.2%) |
| **Avg per annotation** | **134.84 ms** |
| **Median per annotation** | **120.52 ms** |
| Std dev | 110.25 ms |
| Min | 10.32 ms |
| Max | 504.08 ms |
| p90 | 297.14 ms |
| p95 | 329.08 ms |
| p99 | 422.06 ms |
| Total wall time | 123.5 s |
| Throughput | 5.44 ann/s |
| Warmup (init) | 1.96 s |
| Accuracy | N/A — no ground truth |

### Run 02 — `20260415_140905` (High-performance power, canonical baseline, full ground truth)

| Metric | Value |
|---|---|
| Annotations processed | 672 / 672 |
| Ground truth available | 672 / 672 |
| **Correct (exact match)** | **150 (22.3%)** |
| Wrong text returned | 144 (21.4%) |
| No text returned (missed) | 378 (56.2%) |
| Precision when text returned | 150/294 (51.0%) |
| **Avg per annotation** | **144.88 ms** |
| **Median per annotation** | **123.56 ms** |
| Std dev | 118.49 ms |
| Min | 9.83 ms |
| Max | 527.94 ms |
| p90 | 308.58 ms |
| p95 | 360.68 ms |
| p99 | 467.87 ms |
| Total wall time | 125.8 s |
| Throughput | 5.34 ann/s |
| Warmup (init) | 2.03 s |

---

## 4. Key Observations from Run 01

### Detection rate is low (43.8% return text)
Over half of annotated crops came back empty. Breakdown:
- **154 fast-fails (<50 ms):** PaddleOCR's text-detection stage found no candidate regions. Likely very small or low-contrast crops.
- **224 slow-fails (≥50 ms):** Detection stage ran but recognition returned nothing meaningful.

### Vertical text is a clear weakness
| | Avg crop W | Avg crop H | Ratio |
|---|---|---|---|
| Success | 167 px | 89 px | ~1.9:1 landscape |
| Failure | 116 px | 134 px | ~0.9:1 portrait / vertical |

Crops that returned text are predominantly landscape-oriented (horizontal text). Failed crops skew square-to-portrait — consistent with the vertical trailer ID labels noted in the dataset description. `use_angle_cls=True` should handle angle correction, but the miss rate suggests it isn't enough on its own.

### Speed variance is high
Std dev of 110 ms against a median of 120 ms means runtime is highly unpredictable. The fastest calls (~10 ms) are likely quick-rejects with no detected regions; the slowest (~504 ms) involve larger crops with complex backgrounds.

---

## 5. Hypotheses for Future Experiments

The following are candidate improvements, evaluated against two constraints:
- Must not significantly increase cost (no GPU-intensive upscaling)
- Must not significantly degrade speed

| # | Hypothesis | Expected impact | Risk |
|---|---|---|---|
| H1 | Pre-rotate portrait-ratio crops 90° before OCR | Higher detection rate on vertical text | Small speed cost per crop |
| H2 | Grayscale + local contrast normalisation (CLAHE) | Better detection on faded/weathered plates | Negligible cost |
| H3 | Pad small crops to minimum dimension before OCR | Reduce fast-fail rate on tiny crops | Negligible cost |
| H4 | Two-pass: run portrait crops twice (0° and 90°) | Higher recall on vertical text | 2× OCR time for portrait crops only |
| H5 | Filter by aspect ratio to skip clearly invalid crops | Reduce wasted OCR calls | Risk of false negatives |

These will be tested in subsequent experiments, each with a benchmark run for before/after comparison.

---

## 6. Running the Benchmark

```bash
# Standard run (from project root)
python tests/benchmark_ocr.py

# Results land in:
tests/results/benchmark_YYYYMMDD_HHMMSS.log   # human log
tests/results/benchmark_YYYYMMDD_HHMMSS.json  # full data
```

To compare runs, load both JSON files and diff the `speed_ms` and accuracy blocks. The `annotation_results` array preserves per-crop OCR text for manual inspection.

---

## 7. Adding Ground Truth

To enable accuracy scoring, populate the `text` field in each COCO annotation:

```json
{
  "id": 1,
  "image_id": 142,
  "bbox": [2720.26, 192.0, 174.95, 70.69],
  "text": "ABCU1234567"
}
```

The benchmark will detect this automatically and report exact-match accuracy in both the log and JSON output.
