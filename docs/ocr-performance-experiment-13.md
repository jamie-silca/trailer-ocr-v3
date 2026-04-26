# OCR Performance Experiment 13 — Stacked-Vertical Portrait Decoder

## 1. Goal

Address the 23% of the dataset (156/672 portrait-ratio crops) that every prior experiment has left stuck at **1.3% accuracy**. Earlier attempts (EXP-01, 01B, 07) assumed the text was rotated and tried 90° rotations — they failed because, as the user subsequently clarified, the letters are **upright but stacked top-to-bottom** (one letter per row), not rotated.

The idea of EXP-13 is to reconstruct a synthetic horizontal text line from the stacked letters so PaddleOCR's recognition model sees the input distribution it was trained on.

Two variants were evaluated, both on top of the current best preprocessing stack (EXP-09 + EXP-04 + EXP-06):

| Variant | What it does |
|---|---|
| **EXP-13A — stitch** | Locate the text column; uniformly slice it into N ≈ height/32 horizontal bands; resize each to height 48 px; paste side-by-side into a synthetic horizontal strip; run full OCR on the strip. |
| **EXP-13B — per_letter** | Same column location + uniform slice; run OCR on each slice independently; concatenate top-to-bottom. |

Shape gate: only crops with `h > 2w` are routed through the decoder (matches the benchmark's `portrait` aspect-ratio bucket). All other crops flow through the standard pipeline unchanged.

---

## 2. Methodology

### Dataset
- **Source:** `tests/dataset/20260406/` (unchanged)
- **Images:** 397 drone frames
- **Annotations:** 672 bounding boxes in COCO format, all with ground-truth text
- **Portrait subset:** 156 crops (23.2%)

### Base configuration (same for both EXP-13A and EXP-13B, and the EXP-09 comparison row)

| Knob | Value | Source |
|---|---|---|
| `det_db_thresh` | 0.2 | EXP-04 |
| `det_db_box_thresh` | 0.3 | EXP-04 |
| `det_db_unclip_ratio` | 2.0 | EXP-04 |
| `det_limit_side_len` | 960 | default |
| Crop pre-pad (`pad`) | 64 px min side | EXP-03 |
| Bbox expansion | +10% per side | EXP-09 |
| Character substitution (`postprocess`) | on | EXP-06 |
| **Stacked-vertical decoder (`--stacked-vertical`)** | **`stitch` / `per_letter`** | **EXP-13** |

### EXP-13 decoder implementation

- Located in `tests/preprocessing.py::decode_stacked_vertical`.
- Text column bounds are obtained from PaddleOCR's internal `text_detector` (raw API). **Note:** the higher-level `ocr(det=True, rec=False)` API is unusable in PaddleOCR 2.7.3 — it raises `ValueError: The truth value of an array with more than one element is ambiguous` from an internal `if not dt_boxes:` check whenever boxes are actually found. This killed the v1 prototype; v2 calls `text_detector` directly.
- Letters are segmented by **uniform slicing** based on estimated char height (32 px). A projection-profile approach was evaluated and rejected during dev — gaps between stacked letters are irregular (some letters are tightly packed with no background rows between them), so the profile gives inconsistent segmentation.
- Stitched strip: each slice is resized to 48 px height preserving aspect ratio, pasted with a 4 px neutral-gray gap.
- Fallthrough: if `text_detector` finds no text, or estimated char count is < 3, the decoder returns `None` and the caller runs the standard pipeline.

### Reproducibility

```bash
# EXP-13A (stitch)
python tests/benchmark_ocr.py --exp-id EXP-13A \
    --stacked-vertical stitch \
    --preprocess pad,postprocess \
    --bbox-expand-ratio 0.1 \
    --det-db-thresh 0.2 --det-db-box-thresh 0.3 --det-db-unclip-ratio 2.0

# EXP-13B (per_letter)
python tests/benchmark_ocr.py --exp-id EXP-13B \
    --stacked-vertical per_letter \
    --preprocess pad,postprocess \
    --bbox-expand-ratio 0.1 \
    --det-db-thresh 0.2 --det-db-box-thresh 0.3 --det-db-unclip-ratio 2.0
```

---

## 3. Results

### Headline comparison (vs EXP-09 — current best config at time of writing)

| Metric | EXP-09 (baseline) | EXP-13A (stitch) | EXP-13B (per_letter) |
|---|---|---|---|
| **Exact match** | **257 / 672 (38.2%)** | **257 / 672 (38.2%)** | **255 / 672 (37.9%)** |
| Δ vs EXP-09 | — | **0.0 pp** | **−0.3 pp** |
| Wrong text | 189 (28.1%) | 165 (24.6%) | 219 (32.6%) |
| No text returned | 226 (33.6%) | 250 (37.2%) | 198 (29.5%) |
| **Precision (correct / returned)** | 57.6% | **60.9%** (+3.3 pp) | 53.8% (−3.8 pp) |
| Text returned | 446 (66.4%) | 422 (62.8%) | 474 (70.5%) |

### Speed

| Metric | EXP-09 | EXP-13A | EXP-13B |
|---|---|---|---|
| Avg (ms) | 128.5 | 240.3 | 218.8 |
| **Median (ms)** | **110.6** | **170.5** (+59.9) | **156.1** (+45.5) |
| Std dev (ms) | 94.8 | 255.4 | 224.5 |
| Min (ms) | 9.7 | 17.6 | 17.1 |
| Max (ms) | 420.8 | 1191.7 | 1051.0 |
| p90 (ms) | 256.3 | 711.9 | 583.6 |
| p95 (ms) | 301.0 | 828.0 | 729.1 |
| p99 (ms) | 378.9 | 965.1 | 932.4 |
| Wall time | 109.6 s | 187.6 s | 172.3 s |
| Throughput | 6.1 ann/s | 3.6 ann/s | 3.9 ann/s |
| Warmup | 2.0 s | 1.3 s | 1.1 s |

### Portrait-only subset (the target of this experiment)

| Metric | EXP-09 | EXP-13A | EXP-13B |
|---|---|---|---|
| **Portrait correct** | **2 / 156 (1.3%)** | **2 / 156 (1.3%)** | **0 / 156 (0.0%)** ▼ |
| Portrait text returned | 34 (21.8%) | 21 (13.5%) | 73 (46.8%) |
| Portrait median latency | ~110 ms | **660 ms** | **543 ms** |
| Portrait max latency | ~400 ms | 1191 ms | 1051 ms |

### Breakdown by aspect ratio (other buckets unchanged by the shape gate — confirmed)

| Bucket | Total | EXP-09 correct | EXP-13A correct | EXP-13B correct |
|---|---|---|---|---|
| portrait | 156 | 2 (1.3%) | 2 (1.3%) | **0 (0.0%)** |
| near_square | 4 | 0 | 0 | 0 |
| landscape | 54 | 33 (61.1%) | 33 (61.1%) | 33 (61.1%) |
| wide | 424 | 218 (51.4%) | 218 (51.4%) | 218 (51.4%) |
| very_wide | 34 | 4 (11.8%) | 4 (11.8%) | 4 (11.8%) |

Shape gate verified working — non-portrait buckets are bit-identical across the three runs.

---

## 4. Verdict

**Both variants: REJECTED as a portrait accuracy improvement.**

- EXP-13A (stitch): zero accuracy change on portraits. Only redeeming feature is a +3.3 pp precision bump overall because the decoder returns `None` for 13 portrait crops that the baseline would have scored as wrong text — i.e. it suppresses noise rather than decoding correctly.
- EXP-13B (per_letter): strictly worse — regresses from 2 to 0 correct portrait answers, pumps out 73 wrong portrait strings (vs 34 baseline), and drops precision by 3.8 pp overall.
- Both add 40–60 ms to median latency and ~60 s to total wall time.

Promoting neither variant. Revert the shape-gated branch in `benchmark_ocr.py` to `--stacked-vertical off` for any future combination run.

---

## 5. Key Observations — Why It Didn't Work

### Finding 1 — PaddleOCR's rec model cannot read reconstructed stacked text

Instrumented the decoder to feed the rec model directly (bypassing detection) on well-formed synthetic strips, including perspective-rectified versions of the text column.

Representative case — **Ann 14, ground truth `JBHZ672061`**:

| Input to rec model | Output | Confidence |
|---|---|---|
| Full crop (upright stacked) | `''` | 0.00 |
| Stitched strip (9 slices, 48 px tall) | `'3RH'` | 0.20 |
| Stitched strip (10 slices, no gap) | `'RSAHR'` | 0.21 |
| Perspective-rectified + 10-slice stitch | `'P1L575AA'` | 0.18 |
| Column rotated 90° CW | `'PONNNIWL'` | 0.76 (confidently wrong) |

Per-letter rec on the same crop (one character at a time):

| Slice | GT | Rec output | Confidence |
|---|---|---|---|
| 0 | J | `1` | 0.12 |
| 1 | B | `0` | 0.08 |
| 2 | H | `''` | 0.00 |
| 3 | Z | `L` | 0.78 |
| 4 | 6 | `4` | 0.10 |
| 5 | 7 | `7` | 0.16 ✓ |
| 6 | 2 | `5` | 0.08 |
| 7 | 0 | `F` | 0.09 |
| 8 | 6 | `1` | 0.08 |
| 9 | 1 | `P` | 0.06 |

The CRNN either hallucinates with high confidence (on rotated input) or flatlines near zero confidence (on stitched input). Single-letter rec hits 1 out of 10 characters. The trailer-yard print font (stencil-style, painted on metal, weathered) appears to be outside PaddleOCR PP-OCRv4's training distribution when presented without surrounding horizontal context.

### Finding 2 — PaddleOCR 2.7.x `ocr(det=True, rec=False)` is broken

The v1 decoder crashed on almost every portrait crop with `ValueError: truth value of array ambiguous`, traceable to an internal `if not dt_boxes:` check on a numpy array inside PaddleOCR. v2 works around this by calling `paddle_ocr.text_detector(img)` directly (which returns the raw boxes list and bypasses the buggy branch). Worth noting for any future experiment that wants detection-only access.

### Finding 3 — The detector sees stacked text as ONE region, not N letters

On portrait crops, `text_detector` returns a single tall quadrilateral covering the whole letter column, not one quad per letter. So any "detect letters → stitch them" pipeline has no choice but to slice the column uniformly — there are no per-letter boxes to re-use.

### Finding 4 — Inter-letter gaps are not separable via projection profiles

Row-wise std dev on Ann 14 shows some low-activity rows (around row 137–150, 199–207, 238, 267) but they are irregular — several adjacent letters have no visible gap between them. A projection-profile segmenter would split the column into the wrong number of pieces on most crops. Uniform slicing based on an expected character height is more reliable on this dataset, but also limits precision — a 32-px assumption that's off by 2–3 px per character drifts half a character by the bottom of a 10-letter ID.

### Finding 5 — Precision gain is real but narrow

EXP-13A's +3.3 pp precision gain is genuinely useful if the downstream system penalises wrong predictions more than missing ones — it suppresses 13 false portrait outputs. But that's a consolation prize for an experiment that was targeting accuracy, and it comes at ~+60 ms median latency.

---

## 6. Recommendations / Next Steps

Portrait stacked-vertical text appears **not solvable with the PaddleOCR PP-OCRv4 English model** via synthetic horizontal reconstruction. The rec model's learned character priors are too entangled with horizontal word context to extract from per-character inputs.

Realistic paths forward, ranked by expected cost/benefit:

| # | Approach | Est. gain | Cost |
|---|---|---|---|
| H1 | **Train/fine-tune a single-character recogniser** on a small labelled subset of the stacked-vertical crops (e.g. distil to a tiny CNN char classifier). Then use EXP-13 uniform slicing + this classifier instead of PaddleOCR rec. | Potentially unlocks the full 156-crop bucket (~+20 pp on portraits → ~+5 pp overall) | Medium — requires per-character labels; CPU-cheap at inference |
| H2 | **Swap the rec model for one that supports vertical text natively** (PaddleOCR PP-OCRv5 / PaddleOCR-VL, or a Chinese/Japanese model that handles vertical scripts). | Potentially equal to H1 | Medium — new model dependency; latency impact TBD |
| H3 | **Try Tesseract or EasyOCR as a portrait fallback**, since their rec models have different training distributions. Gate by shape, cascade after PaddleOCR fails. | Uncertain, probably modest | Low — try before committing to H1/H2 |
| H4 | Park portraits and focus Round-2 effort on **text-space temporal aggregation (EXP-14)** and **format-aware candidate rescoring (EXP-15)** from the original plan — they target the 30% wrong-text rate across all buckets and don't depend on solving portraits. | Estimated +3–6 pp overall (from EXP-11's upper bound) | Low — pure Python post-processing |

**Suggested next action:** H4 first (cheap, high upside, independent of the portrait problem), then H3 as a probe on portraits (one afternoon's work) before committing to H1 or H2.

---

## 7. Artefacts

- Benchmark JSON: `tests/results/benchmark_EXP-13A_20260424_155305.json`, `benchmark_EXP-13B_20260424_155629.json`
- Benchmark logs: same filenames with `.log` extension
- Debug images (Ann 14 diagnostic): `tests/results/exp13_debug/`
    - `ann14_crop.png` — original portrait crop
    - `ann14_rectified.png` — perspective-rectified column
    - `ann14_stitch.png` — synthetic horizontal strip
    - `ann14_slice_*.png` — per-letter slices
- Implementation: `tests/preprocessing.py::decode_stacked_vertical` (v2, uniform-slice)
- Benchmark wiring: `tests/benchmark_ocr.py` — `--stacked-vertical {off,stitch,per_letter}` flag
