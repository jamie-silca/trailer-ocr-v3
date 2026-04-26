# OCR Experiments Report

Generated: 2026-04-20 18:11 (Round 1: BASE through EXP-12)
Updated: 2026-04-24 (added EXP-13A, EXP-13B, EXP-15, EXP-14, EXP-16, EXP-20)
Updated: 2026-04-26 (added EXP-26 — GOT-OCR-2.0-hf REJECTED on portrait + wide; EXP-27 — local VLM ladder probe; qwen2.5vl:3b VIABLE as a no-cost local replacement for the EXP-25 OpenRouter cascade)
Dataset: Round 1-2 evaluated on 672 annotations / 397 frames (20260406). Round 3 evaluated on 419 annotations / 251 frames (20260423, cleaner lighting).
Engine: PaddleOCR 2.7.3 (PP-OCRv4), CPU-only. Round 3 adds Qwen3-VL-8B-Instruct via OpenRouter (EXP-25, shipped) and characterises a local Ollama qwen2.5vl:3b alternative (EXP-27, viable).

---

## Project Constraints

**Task:** Extract the trailer ID (alphanumeric text, typically ISO 6346 format) from aerial drone imagery. This is the sole OCR objective — no other text or scene content is relevant.

**Platform:** CPU-only backend service. No GPU available at inference time. All preprocessing and model choices must be viable on commodity cloud CPU instances.

**Cost sensitivity:** High. Per-crop latency directly affects API cost. Techniques that increase median latency by >50% require a proportional accuracy gain to justify adoption.

**Data source:** Moving aerial drone (DJI) capturing stationary trailers on ground. The camera moves significantly between frames — the same physical trailer can appear at entirely different pixel coordinates in consecutive frames. This rules out any pixel-space frame-to-frame tracker (IoU linking, SORT, ByteTrack) without GPS or visual stabilisation.

**Ground truth format:** Trailer IDs are pre-labelled bounding boxes (YOLO/COCO format). Evaluation is exact-match string accuracy against the labelled ID after normalisation.

---

## Quick Comparison

| Experiment | Verdict | Text Returned | Correct (Exact Match) | Exact Match % | Δ | Precision | Median ms | Δ Speed |
|---|---|---|---|---|---|---|---|---|
| BASE-01 | BASELINE | 294 (43.8%) | 150/672 | 22.3% | — | 51.0% | 123.56 | — |
| BASE-01-VERIFY | VERIFICATION | 294 (43.8%) | 150/672 | 22.3% | — | 51.0% | 99.97 | -23.6ms ▲ |
| EXP-01 | REJECTED | 292 (43.5%) | 150/672 | 22.3% | — | 51.4% | 90.04 | -33.5ms ▲ |
| EXP-01B | REJECTED | 282 (42.0%) | 149/672 | 22.2% | -0.1pp ▼ | 52.8% | 93.22 | -30.3ms ▲ |
| EXP-02 | REJECTED | 183 (27.2%) | 83/672 | 12.4% | -9.9pp ▼ | 45.4% | 97.96 | -25.6ms ▲ |
| EXP-03 | ACCEPTED | 400 (59.5%) | 204/672 | 30.4% | +8.1pp ▲ | 51.0% | 88.88 | -34.7ms ▲ |
| EXP-04 | ACCEPTED | 324 (48.2%) | 165/672 | 24.6% | +2.3pp ▲ | 50.9% | 123.7 | +0.1ms ▼ |
| EXP-03+04 | BEST CONFIG | 428 (63.7%) | 233/672 | 34.7% | +12.4pp ▲ | 54.4% | 97.62 | -25.9ms ▲ |
| EXP-03+04+06 | ACCEPTED (marginal) | 428 (63.7%) | 234/672 | 34.8% | +12.5pp ▲ | 54.7% | 85.63 | -37.9ms ▲ |
| EXP-05-320 | NEUTRAL | 298 (44.3%) | 150/672 | 22.3% | — | 50.3% | 110.22 | -13.3ms ▲ |
| EXP-05-480 | NEUTRAL | 298 (44.3%) | 150/672 | 22.3% | — | 50.3% | 103.99 | -19.6ms ▲ |
| EXP-05-640 | NEUTRAL | 298 (44.3%) | 150/672 | 22.3% | — | 50.3% | 103.69 | -19.9ms ▲ |
| EXP-07 | REJECTED | 307 (45.7%) | 150/672 | 22.3% | — | 48.9% | 144.4 | +20.8ms ▼ |
| EXP-08 | NEUTRAL | 428 (63.7%) | 233/672 | 34.7% | +12.4pp ▲ | 54.4% | 83.43 | -40.1ms ▲ |
| EXP-09 | NEW BEST CONFIG | 446 (66.4%) | 257/672 | 38.2% | +15.9pp ▲ | 57.6% | 110.55 | -13.0ms ▲ |
| EXP-10 | ACCEPTED (marginal) | 437 (65.0%) | 235/672 | 35.0% | +12.7pp ▲ | 53.8% | 103.69 | -19.9ms ▲ |
| EXP-09+10 | ACCEPTED (marginal) | 453 (67.4%) | 258/672 | 38.4% | +16.1pp ▲ | 57.0% | 141.9 | +18.3ms ▼ |
| EXP-11 | SPECIAL ANALYSIS | — | 156/414 tracks | 37.7% per-track | — | 53.2% | — (no re-OCR) | — |
| EXP-12 | INFEASIBLE | — | — | — | — | — | — | — |
| EXP-13A | REJECTED | — | — | — | — | — | — | — |
| EXP-13B | REJECTED | — | — | — | — | — | — | — |
| EXP-15 | ACCEPTED (marginal) | 418 (62.2%) | 259/672 | 38.5% | +0.3pp ▲ vs EXP-09 | 62.0% | ~138 | +1ms (noise) |
| EXP-14 | REJECTED (as planned) / gated variant marginal | 418 (62.2%) | 262/672 (best gated) | 39.0% (best gated) | +0.4pp ▲ vs EXP-15 gated; −2.1pp ▼ unrestricted | 62.7% gated | — (post-hoc) | — |
| EXP-16 | SKIPPED (duplicates EXP-13A) | — | — | — | — | — | — | — |
| EXP-20 | REJECTED (drop-in) / promising hybrid | 652 (97.0%) | 275/672 | 40.9% | +2.7pp ▲ vs EXP-09 | 42.2% (**-15.4pp ▼**) | 989 | +879ms ▼ |
| EXP-18 (Gemini Flash) | REJECTED (portrait) — scope broadened | — (sanity only) | 0/15 portrait sanity | 0.0% | **-1.3pp ▼** vs EXP-09 portrait (2/156) | — | ~5000 | +4890ms ▼ |
| EXP-22 (PP-OCRv5 drop-in) | REJECTED (drop-in); §10 hidden-finding **retracted** | — (sanity only) | 0/15 portrait sanity | 0.0% | **-1.3pp ▼** vs EXP-09 portrait | — | ~1650 | +1540ms ▼ |
| EXP-17 (per-char detect + unroll, v5-detector variant) | REJECTED — premise (v5 per-char detection) failed at execution | — | 0/156 (precompute) | — | — | — | — | — |

### Round 3 — evaluated on 20260423 dataset (419 annotations, cleaner lighting)

| Experiment | Verdict | Text Returned | Correct (Exact Match) | Exact Match % | Δ vs EXP-23 | Precision | Median ms | Δ Speed |
|---|---|---|---|---|---|---|---|---|
| EXP-23 (re-baseline, EXP-09 config on cleaner data) | NEW BASELINE | 328 (78.3%) | 262/419 | 62.5% | — | 79.9% | ~110 | — |
| EXP-19 (Tesseract PSM 6, portrait-only) | REJECTED | — | 4/119 portrait | 3.4% | — | — | ~140 | — |
| EXP-24 (Surya OCR, portrait-only) | PLANNED — deferred | — | — | — | — | — | — | — |
| **EXP-25 (Qwen3-VL-8B cascade fallback, portrait)** | **ACCEPTED — Round 3 headline** | **377 (90.0%)** | **314/419** | **74.9%** | **+12.4pp ▲** | **83.3% (+3.4pp ▲)** | **148** | **+38ms ▼** |
| EXP-26 (GOT-OCR-2.0-hf, portrait + wide A/B) | REJECTED — both gates failed | — | portrait 0/5; wide 148/265 | wide 55.8% (full repl); cascade +6 rescues / +0.7pp | catastrophic regression on full repl; cascade ROI negative @ 13s/crop | — | 13,400 | +13,290ms ▼ |
| EXP-27 (local VLM ladder — qwen2.5vl:3b via Ollama, portrait spike-only) | VIABLE — production wiring TBD | — (spike) | 60/119 portrait | 50.4% | matches EXP-25 portrait (-2.6pp); JBHZ 50/92 *beats* OR's 47/92 | — | 4109 (warm) | local-only; $0 / run |

---

## Detailed Experiment Results

### BASE-01 — Canonical baseline — all PaddleOCR defaults, no preprocessing

**Verdict: BASELINE**

> No changes applied — PaddleOCR PP-OCRv4 with all default parameters and no image preprocessing. This run establishes the canonical accuracy and speed baseline that all subsequent experiments are compared against.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **294 / 672 (43.8%)** | **—** |
| → Correct (exact match) | 150 (22.3%) | — |
| → Wrong text | 144 (21.4%) | — |
| No text returned | 378 (56.2%) | — |
| **Precision** (correct / returned) | **51.0%** | **—** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 1 | 0.6% | — | 32 (20.5%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 28 | 50.9% | — | 34 (61.8%) |
| wide | 422 | 120 | 28.4% | — | 204 (48.3%) |
| very_wide | 36 | 1 | 2.8% | — | 23 (63.9%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 144.88 ms | — |
| **Median** | **123.56 ms** | **—** |
| Std dev | 118.49 ms | — |
| Min | 9.83 ms | |
| Max | 527.94 ms | |
| p90 | 308.58 ms | — |
| p95 | 360.68 ms | — |
| p99 | 467.87 ms | — |
| Wall time | 125.791s | |
| Throughput | 5.34 ann/s | |

**Result Summary:** Establishing a baseline accuracy of 22.3%, this run highlights that nearly 60% of annotations (primarily small or vertical crops) are ignored by default.

---

### BASE-01-VERIFY — Baseline re-run with enhanced harness (verification)

**Verdict: VERIFICATION**

> Identical configuration to BASE-01, re-run using the enhanced benchmark harness (with --exp-id support, subset breakdowns, and config recording). The purpose is to verify that the new harness produces identical accuracy numbers to the original baseline, confirming the harness itself introduces no side effects.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **294 / 672 (43.8%)** | **—** |
| → Correct (exact match) | 150 (22.3%) | — |
| → Wrong text | 144 (21.4%) | — |
| No text returned | 378 (56.2%) | — |
| **Precision** (correct / returned) | **51.0%** | **—** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 1 | 0.6% | — | 32 (20.5%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 28 | 50.9% | — | 34 (61.8%) |
| wide | 422 | 120 | 28.4% | — | 204 (48.3%) |
| very_wide | 36 | 1 | 2.8% | — | 23 (63.9%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 115.53 ms | -29.3ms ▲ |
| **Median** | **99.97 ms** | **-23.6ms ▲** |
| Std dev | 94.47 ms | -24.0ms ▲ |
| Min | 8.89 ms | |
| Max | 486.79 ms | |
| p90 | 249.92 ms | -58.7ms ▲ |
| p95 | 279.38 ms | -81.3ms ▲ |
| p99 | 370.22 ms | -97.6ms ▲ |
| Wall time | 102.757s | |
| Throughput | 6.54 ann/s | |

**Result Summary:** Successfully validated the new benchmark harness consistency by reproducing identical accuracy results (22.3%), while noting minor speed variability likely due to environmental noise.

---

### EXP-01 — Auto-rotate portrait crops 90° CW

**Verdict: REJECTED**

> Portrait-oriented crops (width < height) are rotated 90 degrees clockwise before OCR using a lossless PIL transpose. The hypothesis was that ~156 portrait crops contain horizontal trailer ID text that appears vertical due to crop orientation, and PaddleOCR's angle classifier (which handles 0/180 degree flips) cannot correct 90-degree rotations. Expected a +10-15pp accuracy improvement on the portrait subset with negligible speed cost.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **292 / 672 (43.5%)** | **-0.3pp ▼** |
| → Correct (exact match) | 150 (22.3%) | — |
| → Wrong text | 142 (21.1%) | -0.3pp ▲ |
| No text returned | 380 (56.5%) | +0.3pp ▼ |
| **Precision** (correct / returned) | **51.4%** | **+0.4pp ▲** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 1 | 0.6% | — | 30 (19.2%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 28 | 50.9% | — | 34 (61.8%) |
| wide | 422 | 120 | 28.4% | — | 204 (48.3%) |
| very_wide | 36 | 1 | 2.8% | — | 23 (63.9%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 127.03 ms | -17.8ms ▲ |
| **Median** | **90.04 ms** | **-33.5ms ▲** |
| Std dev | 114.8 ms | -3.7ms ▲ |
| Min | 8.67 ms | |
| Max | 577.26 ms | |
| p90 | 302.55 ms | -6.0ms ▲ |
| p95 | 368.08 ms | +7.4ms ▼ |
| p99 | 423.17 ms | -44.7ms ▲ |
| Wall time | 110.767s | |
| Throughput | 6.07 ann/s | |

**Result Summary:** Proved that 90° rotation alone does not unlock portrait text, as the resulting horizontal images—often under 40px tall—remain below the recognition model's effective threshold.

---

### EXP-01B — Rotate portrait crops 90° CW + upscale to min 80px height

**Verdict: REJECTED**

> Extends EXP-01 by also upscaling portrait crops after rotation so the resulting image height reaches at least 80px, using PIL Lanczos interpolation. EXP-01 failed because rotated portrait crops were only 31-58px tall — too short for PaddleOCR's recognition stage to discriminate characters. The hypothesis was that bilinear upscaling to a minimum readable height would give the model enough pixel data to recognise the text.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **282 / 672 (42.0%)** | **-1.8pp ▼** |
| → Correct (exact match) | 149 (22.2%) | -0.1pp ▼ |
| → Wrong text | 133 (19.8%) | -1.6pp ▲ |
| No text returned | 390 (58.0%) | +1.8pp ▼ |
| **Precision** (correct / returned) | **52.8%** | **+1.8pp ▲** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 0 | 0.0% | -0.6pp ▼ | 20 (12.8%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 28 | 50.9% | — | 34 (61.8%) |
| wide | 422 | 120 | 28.4% | — | 204 (48.3%) |
| very_wide | 36 | 1 | 2.8% | — | 23 (63.9%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 136.24 ms | -8.6ms ▲ |
| **Median** | **93.22 ms** | **-30.3ms ▲** |
| Std dev | 128.68 ms | +10.2ms ▼ |
| Min | 8.08 ms | |
| Max | 569.34 ms | |
| p90 | 331.09 ms | +22.5ms ▼ |
| p95 | 383.99 ms | +23.3ms ▼ |
| p99 | 495.9 ms | +28.0ms ▼ |
| Wall time | 116.461s | |
| Throughput | 5.77 ann/s | |

**Result Summary:** Demonstrated that Lanczos upscaling of rotated crops is counterproductive; interpolation fails to reconstruct the fine stroke detail required for confident character discrimination.

---

### EXP-02 — CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=8×8)

**Verdict: REJECTED**

> Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast: the crop is converted to grayscale, CLAHE is applied (clipLimit=2.0, 8x8 tile grid), then converted back to 3-channel RGB before OCR. The hypothesis was that faded paint, shadows, and aerial haze reduce contrast between text and background, causing PaddleOCR's detection stage to miss text regions. CLAHE should have boosted local contrast enough for the DB detector to find previously-invisible text, expecting +3-8pp accuracy with minimal speed cost.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **183 / 672 (27.2%)** | **-16.6pp ▼** |
| → Correct (exact match) | 83 (12.4%) | -9.9pp ▼ |
| → Wrong text | 100 (14.9%) | -6.5pp ▲ |
| No text returned | 489 (72.8%) | +16.6pp ▼ |
| **Precision** (correct / returned) | **45.4%** | **-5.6pp ▼** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 2 | 1.3% | +0.7pp ▲ | 27 (17.3%) |
| near_square | 3 | 0 | 0.0% | — | 0 (0.0%) |
| landscape | 55 | 19 | 34.5% | -16.4pp ▼ | 24 (43.6%) |
| wide | 422 | 59 | 14.0% | -14.4pp ▼ | 108 (25.6%) |
| very_wide | 36 | 3 | 8.3% | +5.5pp ▲ | 24 (66.7%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 116.55 ms | -28.3ms ▲ |
| **Median** | **97.96 ms** | **-25.6ms ▲** |
| Std dev | 105.07 ms | -13.4ms ▲ |
| Min | 8.58 ms | |
| Max | 466.4 ms | |
| p90 | 257.15 ms | -51.4ms ▲ |
| p95 | 323.2 ms | -37.5ms ▲ |
| p99 | 406.86 ms | -61.0ms ▲ |
| Wall time | 109.558s | |
| Throughput | 6.13 ann/s | |

**Result Summary:** Catastrophic failure (-9.9pp accuracy) suggests that CLAHE-induced noise or boundary artifacts severely confuse the DB detector on drone-derived crops.

---

### EXP-03 — Pad small crops to min 64px per side (neutral grey border)

**Verdict: ACCEPTED**

> Crops with either dimension below 64px are padded with a neutral grey (128,128,128) border using PIL ImageOps.expand, bringing the minimum dimension to 64px. The hypothesis was that very small crops cause PaddleOCR's detection CNN to fast-fail because text features are too small after the model's internal resize step, and that providing additional spatial context via padding would allow the detector to find text regions it previously missed. Expected a modest +1-3pp improvement concentrated in the small crop bucket, with negligible speed cost.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **400 / 672 (59.5%)** | **+15.7pp ▲** |
| → Correct (exact match) | 204 (30.4%) | +8.1pp ▲ |
| → Wrong text | 196 (29.2%) | +7.8pp ▼ |
| No text returned | 272 (40.5%) | -15.7pp ▲ |
| **Precision** (correct / returned) | **51.0%** | **—** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 1 | 0.6% | — | 36 (23.1%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 30 | 54.5% | +3.6pp ▲ | 39 (70.9%) |
| wide | 422 | 170 | 40.3% | +11.9pp ▲ | 297 (70.4%) |
| very_wide | 36 | 3 | 8.3% | +5.5pp ▲ | 27 (75.0%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 132.85 ms | -12.0ms ▲ |
| **Median** | **88.88 ms** | **-34.7ms ▲** |
| Std dev | 109.39 ms | -9.1ms ▲ |
| Min | 9.44 ms | |
| Max | 467.29 ms | |
| p90 | 294.54 ms | -14.0ms ▲ |
| p95 | 342.73 ms | -17.9ms ▲ |
| p99 | 416.16 ms | -51.7ms ▲ |
| Wall time | 118.756s | |
| Throughput | 5.66 ann/s | |

**Result Summary:** Outstanding success (+8.1pp) proving that the "small crop" problem is primarily a context/padding issue for the detector rather than an inherent recognition limit.

---

### EXP-04 — Lower detection thresholds (db_thresh=0.2, box_thresh=0.3, unclip=2.0)

**Verdict: ACCEPTED**

> Lowers three PaddleOCR detection parameters from their defaults: det_db_thresh from 0.3 to 0.2 (pixel binarization sensitivity), det_db_box_thresh from 0.5 to 0.3 (minimum box confidence to keep a detection), and det_db_unclip_ratio from 1.5 to 2.0 (box expansion factor). The hypothesis was that PaddleOCR's default thresholds are calibrated for high-contrast document and scene text, and aerial drone imagery with lower contrast needs more lenient thresholds to detect faint text regions. Expected +3-7pp accuracy with a moderate speed increase (+10-30ms median) due to more candidate regions passing through the recognition stage.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **324 / 672 (48.2%)** | **+4.4pp ▲** |
| → Correct (exact match) | 165 (24.6%) | +2.3pp ▲ |
| → Wrong text | 159 (23.7%) | +2.3pp ▼ |
| No text returned | 348 (51.8%) | -4.4pp ▲ |
| **Precision** (correct / returned) | **50.9%** | **-0.1pp ▼** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 2 | 1.3% | +0.7pp ▲ | 40 (25.6%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 27 | 49.1% | -1.8pp ▼ | 36 (65.5%) |
| wide | 422 | 133 | 31.5% | +3.1pp ▲ | 221 (52.4%) |
| very_wide | 36 | 3 | 8.3% | +5.5pp ▲ | 26 (72.2%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 134.7 ms | -10.2ms ▲ |
| **Median** | **123.7 ms** | **+0.1ms ▼** |
| Std dev | 106.11 ms | -12.4ms ▲ |
| Min | 8.22 ms | |
| Max | 471.96 ms | |
| p90 | 283.19 ms | -25.4ms ▲ |
| p95 | 327.6 ms | -33.1ms ▲ |
| p99 | 391.36 ms | -76.5ms ▲ |
| Wall time | 117.199s | |
| Throughput | 5.73 ann/s | |

**Result Summary:** Successfully increased recall by +2.3pp, capturing "borderline" but correct detections that the default thresholds were aggressively discarding.

---

### EXP-03+04 — Pad + lower detection thresholds (combined)

**Verdict: BEST CONFIG**

> Combines EXP-03 (crop padding) and EXP-04 (lowered detection thresholds) in a single run to test whether their effects compound. The rationale is that padding gives the detector more spatial context while lowered thresholds make the detector more sensitive — addressing two different failure modes simultaneously. Expected at least additive improvement (+10pp or more) if the two techniques target non-overlapping failure populations.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **428 / 672 (63.7%)** | **+19.9pp ▲** |
| → Correct (exact match) | 233 (34.7%) | +12.4pp ▲ |
| → Wrong text | 195 (29.0%) | +7.6pp ▼ |
| No text returned | 244 (36.3%) | -19.9pp ▲ |
| **Precision** (correct / returned) | **54.4%** | **+3.4pp ▲** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 2 | 1.3% | +0.7pp ▲ | 41 (26.3%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 29 | 52.7% | +1.8pp ▲ | 41 (74.5%) |
| wide | 422 | 198 | 46.9% | +18.5pp ▲ | 315 (74.6%) |
| very_wide | 36 | 4 | 11.1% | +8.3pp ▲ | 30 (83.3%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 131.67 ms | -13.2ms ▲ |
| **Median** | **97.62 ms** | **-25.9ms ▲** |
| Std dev | 104.13 ms | -14.4ms ▲ |
| Min | 8.75 ms | |
| Max | 539.23 ms | |
| p90 | 274.1 ms | -34.5ms ▲ |
| p95 | 324.27 ms | -36.4ms ▲ |
| p99 | 415.74 ms | -52.1ms ▲ |
| Wall time | 116.293s | |
| Throughput | 5.78 ann/s | |

**Result Summary:** **Best Performance (Round 1):** Verified a highly effective additive relationship between padding and sensitivity, raising baseline accuracy by over 50% (from 22.3% to 34.7%).

---

### EXP-03+04+06 — Pad + lower thresholds + character substitution post-processing

**Verdict: ACCEPTED (marginal)**

> Adds EXP-06 post-processing character substitution on top of the EXP-03+04 combo: after OCR returns text, domain-specific rules correct known character confusions (O to 0, I to 1, S to 5 in numeric contexts; reverse in alpha contexts) for trailer ID formats like JBHU 235644. The hypothesis was that ~38 wrong-text results in the baseline are caused by systematic letter/digit confusion that can be corrected with conservative regex-based pattern matching. Expected +2-4pp marginal improvement on top of EXP-03+04, with zero speed cost since it is pure string post-processing.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **428 / 672 (63.7%)** | **+19.9pp ▲** |
| → Correct (exact match) | 234 (34.8%) | +12.5pp ▲ |
| → Wrong text | 194 (28.9%) | +7.5pp ▼ |
| No text returned | 244 (36.3%) | -19.9pp ▲ |
| **Precision** (correct / returned) | **54.7%** | **+3.7pp ▲** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 2 | 1.3% | +0.7pp ▲ | 41 (26.3%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 30 | 54.5% | +3.6pp ▲ | 41 (74.5%) |
| wide | 422 | 198 | 46.9% | +18.5pp ▲ | 315 (74.6%) |
| very_wide | 36 | 4 | 11.1% | +8.3pp ▲ | 30 (83.3%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 115.4 ms | -29.5ms ▲ |
| **Median** | **85.63 ms** | **-37.9ms ▲** |
| Std dev | 91.02 ms | -27.5ms ▲ |
| Min | 8.93 ms | |
| Max | 430.44 ms | |
| p90 | 240.71 ms | -67.9ms ▲ |
| p95 | 289.37 ms | -71.3ms ▲ |
| p99 | 369.42 ms | -98.4ms ▲ |
| Wall time | 101.268s | |
| Throughput | 6.64 ann/s | |

**Result Summary:** Adding character substitution provided a marginal +0.1pp gain, indicating that detection failures are a significantly higher priority than character-level confusion.

---

### EXP-05-320 — det_limit_side_len = 320

**Verdict: NEUTRAL**

> Changes PaddleOCR's det_limit_side_len from the default 960 to 320 — this controls the maximum side length the detection model resizes input to before running the DB text detector. The hypothesis was that the default 960 upscales small crops (~140x70px) by approximately 5.5x for detection, potentially introducing interpolation artefacts at an unexpected scale for the model. A smaller limit (320) would reduce the upscale factor to ~1.8x, which might better match the model's training distribution. Expected uncertain impact (+/- 2-5pp accuracy) with faster speed due to less computation in the detection CNN.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **298 / 672 (44.3%)** | **+0.5pp ▲** |
| → Correct (exact match) | 150 (22.3%) | — |
| → Wrong text | 148 (22.0%) | +0.6pp ▼ |
| No text returned | 374 (55.7%) | -0.5pp ▲ |
| **Precision** (correct / returned) | **50.3%** | **-0.7pp ▼** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 1 | 0.6% | — | 33 (21.2%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 28 | 50.9% | — | 34 (61.8%) |
| wide | 422 | 120 | 28.4% | — | 207 (49.1%) |
| very_wide | 36 | 1 | 2.8% | — | 23 (63.9%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 120.18 ms | -24.7ms ▲ |
| **Median** | **110.22 ms** | **-13.3ms ▲** |
| Std dev | 97.88 ms | -20.6ms ▲ |
| Min | 8.84 ms | |
| Max | 510.56 ms | |
| p90 | 263.81 ms | -44.8ms ▲ |
| p95 | 295.09 ms | -65.6ms ▲ |
| p99 | 377.29 ms | -90.6ms ▲ |
| Wall time | 104.558s | |
| Throughput | 6.43 ann/s | |

**Result Summary:** Neutral results across all resize limits indicate that PaddleOCR's internal upscaling to 960px is robust and not a source of performance degradation for small crops.

---

### EXP-05-480 — det_limit_side_len = 480

**Verdict: NEUTRAL**

> Same as EXP-05-320 but with det_limit_side_len set to 480, testing a middle ground between 320 and the default 960. The hypothesis was that 480 might be the sweet spot: enough resolution for the detection model to find text features without the excessive upscaling of 960. Expected to perform similarly to or slightly better than 320, with moderate speed improvement.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **298 / 672 (44.3%)** | **+0.5pp ▲** |
| → Correct (exact match) | 150 (22.3%) | — |
| → Wrong text | 148 (22.0%) | +0.6pp ▼ |
| No text returned | 374 (55.7%) | -0.5pp ▲ |
| **Precision** (correct / returned) | **50.3%** | **-0.7pp ▼** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 1 | 0.6% | — | 33 (21.2%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 28 | 50.9% | — | 34 (61.8%) |
| wide | 422 | 120 | 28.4% | — | 207 (49.1%) |
| very_wide | 36 | 1 | 2.8% | — | 23 (63.9%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 113.34 ms | -31.5ms ▲ |
| **Median** | **103.99 ms** | **-19.6ms ▲** |
| Std dev | 91.44 ms | -27.0ms ▲ |
| Min | 7.64 ms | |
| Max | 390.22 ms | |
| p90 | 240.71 ms | -67.9ms ▲ |
| p95 | 284.76 ms | -75.9ms ▲ |
| p99 | 349.26 ms | -118.6ms ▲ |
| Wall time | 100.271s | |
| Throughput | 6.7 ann/s | |

**Result Summary:** Neutral results across all resize limits indicate that PaddleOCR's internal upscaling to 960px is robust and not a source of performance degradation for small crops.

---

### EXP-05-640 — det_limit_side_len = 640

**Verdict: NEUTRAL**

> Same as EXP-05-320 but with det_limit_side_len set to 640, the highest of the three test values. The hypothesis was that 640 provides higher resolution for detection while still avoiding the full 5.5x upscale of the default 960, potentially balancing text feature clarity against interpolation noise. Expected similar accuracy to other EXP-05 variants with speed between 320 and 960.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **298 / 672 (44.3%)** | **+0.5pp ▲** |
| → Correct (exact match) | 150 (22.3%) | — |
| → Wrong text | 148 (22.0%) | +0.6pp ▼ |
| No text returned | 374 (55.7%) | -0.5pp ▲ |
| **Precision** (correct / returned) | **50.3%** | **-0.7pp ▼** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 1 | 0.6% | — | 33 (21.2%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 28 | 50.9% | — | 34 (61.8%) |
| wide | 422 | 120 | 28.4% | — | 207 (49.1%) |
| very_wide | 36 | 1 | 2.8% | — | 23 (63.9%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 114.58 ms | -30.3ms ▲ |
| **Median** | **103.69 ms** | **-19.9ms ▲** |
| Std dev | 92.44 ms | -26.0ms ▲ |
| Min | 7.85 ms | |
| Max | 400.74 ms | |
| p90 | 242.62 ms | -66.0ms ▲ |
| p95 | 289.66 ms | -71.0ms ▲ |
| p99 | 358.23 ms | -109.6ms ▲ |
| Wall time | 101.343s | |
| Throughput | 6.63 ann/s | |

**Result Summary:** Neutral results across all resize limits indicate that PaddleOCR's internal upscaling to 960px is robust and not a source of performance degradation for small crops.

---

### EXP-07 — Two-pass OCR for portrait crops (original + 90cw + 90ccw, best confidence)

**Verdict: REJECTED**

> For portrait-oriented crops, runs OCR three times — at the original orientation, rotated 90 degrees clockwise, and rotated 90 degrees counter-clockwise — and returns whichever result has the highest confidence score. The hypothesis was that some portrait crops may contain text oriented differently (rotated CW vs CCW), and a multi-orientation approach would be more robust than EXP-01's single fixed rotation by covering all possible orientations. Expected +1-3pp beyond EXP-01 on the portrait subset, at the cost of up to 3x slower processing for the 156 portrait crops.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **307 / 672 (45.7%)** | **+1.9pp ▲** |
| → Correct (exact match) | 150 (22.3%) | — |
| → Wrong text | 157 (23.4%) | +2.0pp ▼ |
| No text returned | 365 (54.3%) | -1.9pp ▲ |
| **Precision** (correct / returned) | **48.9%** | **-2.1pp ▼** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 1 | 0.6% | — | 45 (28.8%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 28 | 50.9% | — | 34 (61.8%) |
| wide | 422 | 120 | 28.4% | — | 204 (48.3%) |
| very_wide | 36 | 1 | 2.8% | — | 23 (63.9%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 206.59 ms | +61.7ms ▼ |
| **Median** | **144.4 ms** | **+20.8ms ▼** |
| Std dev | 225.92 ms | +107.4ms ▼ |
| Min | 7.8 ms | |
| Max | 1102.79 ms | |
| p90 | 547.08 ms | +238.5ms ▼ |
| p95 | 740.09 ms | +379.4ms ▼ |
| p99 | 929.93 ms | +462.1ms ▼ |
| Wall time | 163.495s | |
| Throughput | 4.11 ann/s | |

**Result Summary:** Confirmed that even a comprehensive 3-pass multi-orientation approach cannot overcome the resolution bottleneck for portraits, while adding an unacceptable 16% speed penalty.

---

### EXP-08 — Improved positional postprocessing (digit-in-prefix fix, H→8 in digit context)

**Verdict: NEUTRAL**

> Adds an improved positional character substitution pass on top of EXP-03+04+06: digits in the 4-character alpha prefix are forced to their nearest letter (0->O, 1->I, 5->S, 6->G, 8->B), and H is added to the digit-context substitution table (H->8). OcrProcessor already applies EXP-06 character substitution internally; this experiment runs an additional pass with the expanded ruleset to catch cases the conservative EXP-06 rules miss. Expected +0.5-2pp marginal improvement with zero speed cost.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **428 / 672 (63.7%)** | **+19.9pp ▲** |
| → Correct (exact match) | 233 (34.7%) | +12.4pp ▲ |
| → Wrong text | 195 (29.0%) | +7.6pp ▼ |
| No text returned | 244 (36.3%) | -19.9pp ▲ |
| **Precision** (correct / returned) | **54.4%** | **+3.4pp ▲** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 2 | 1.3% | +0.7pp ▲ | 41 (26.3%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 30 | 54.5% | +3.6pp ▲ | 41 (74.5%) |
| wide | 422 | 198 | 46.9% | +18.5pp ▲ | 315 (74.6%) |
| very_wide | 36 | 3 | 8.3% | +5.5pp ▲ | 30 (83.3%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 115.0 ms | -29.9ms ▲ |
| **Median** | **83.43 ms** | **-40.1ms ▲** |
| Std dev | 90.92 ms | -27.6ms ▲ |
| Min | 9.1 ms | |
| Max | 436.34 ms | |
| p90 | 234.61 ms | -74.0ms ▲ |
| p95 | 289.19 ms | -71.5ms ▲ |
| p99 | 370.66 ms | -97.2ms ▲ |
| Wall time | 101.676s | |
| Throughput | 6.61 ann/s | |

**Result Summary:** Running postprocess_v2 on top of OcrProcessor's already-applied EXP-06 rules provides no net gain; double-postprocessing is idempotent for well-matched text and marginally harmful where the expanded ruleset applies conflicting substitutions. The improved rules need to replace, not stack on, the existing substitution.

---

### EXP-09 — 10% YOLO bbox expansion before crop (pixel context from real image edges)

**Verdict: NEW BEST CONFIG**

> Each YOLO bounding box is expanded by 10% on each side before cropping, so the crop includes a ring of real image pixels around the annotated region instead of the neutral-grey padding added by EXP-03. The hypothesis was that EXP-03's neutral-grey padding provides spatial context that prevents the detection stage from fast-failing, but real surrounding pixels should provide richer features — especially for trailer IDs that bleed close to the bbox edge. Expected +2-5pp improvement on the wide and landscape subsets, with a slight speed increase from larger crops.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **446 / 672 (66.4%)** | **+22.6pp ▲** |
| → Correct (exact match) | 257 (38.2%) | +15.9pp ▲ |
| → Wrong text | 189 (28.1%) | +6.7pp ▼ |
| No text returned | 226 (33.6%) | -22.6pp ▲ |
| **Precision** (correct / returned) | **57.6%** | **+6.6pp ▲** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 2 | 1.3% | +0.7pp ▲ | 34 (21.8%) |
| near_square | 4 | 0 | 0.0% | — | 1 (25.0%) |
| landscape | 54 | 33 | 61.1% | +10.2pp ▲ | 39 (72.2%) |
| wide | 424 | 218 | 51.4% | +23.0pp ▲ | 342 (80.7%) |
| very_wide | 34 | 4 | 11.8% | +9.0pp ▲ | 30 (88.2%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 128.52 ms | -16.4ms ▲ |
| **Median** | **110.55 ms** | **-13.0ms ▲** |
| Std dev | 94.75 ms | -23.7ms ▲ |
| Min | 9.67 ms | |
| Max | 420.82 ms | |
| p90 | 256.27 ms | -52.3ms ▲ |
| p95 | 300.97 ms | -59.7ms ▲ |
| p99 | 378.87 ms | -89.0ms ▲ |
| Wall time | 109.583s | |
| Throughput | 6.13 ann/s | |

**Result Summary:** **New Best Config:** Providing real image context via 10% bbox expansion is more effective than neutral-grey padding (EXP-03). The wider crops give the DB detector richer surrounding features, boosting wide-format accuracy from 46.9% to 51.4% and delivering a net +3.4pp gain with no precision loss.

---

### EXP-10 — Cascade retry: on no-text, retry with sharpen+dilate fallback preprocessing

**Verdict: ACCEPTED (marginal)**

> Implements a two-pass cascade: the first pass runs the standard EXP-03+04+06 pipeline; if no text is detected, a fallback retry applies PIL UnsharpMask sharpening and morphological dilation (2x2 kernel, 1 iteration) to thicken and sharpen strokes before retrying OCR. The hypothesis was that the 244 no-text crops contain faint or blurry text that sharpening and stroke thickening could make legible, and that restricting the expensive fallback to the failing 36% of crops keeps the average latency acceptable. Expected +1-4pp accuracy from rescued crops, at a cost of ~20-40ms average latency increase.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **437 / 672 (65.0%)** | **+21.2pp ▲** |
| → Correct (exact match) | 235 (35.0%) | +12.7pp ▲ |
| → Wrong text | 202 (30.1%) | +8.7pp ▼ |
| No text returned | 235 (35.0%) | -21.2pp ▲ |
| **Precision** (correct / returned) | **53.8%** | **+2.8pp ▲** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 2 | 1.3% | +0.7pp ▲ | 43 (27.6%) |
| near_square | 3 | 0 | 0.0% | — | 1 (33.3%) |
| landscape | 55 | 30 | 54.5% | +3.6pp ▲ | 41 (74.5%) |
| wide | 422 | 199 | 47.2% | +18.8pp ▲ | 322 (76.3%) |
| very_wide | 36 | 4 | 11.1% | +8.3pp ▲ | 30 (83.3%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 139.64 ms | -5.2ms ▲ |
| **Median** | **103.69 ms** | **-19.9ms ▲** |
| Std dev | 114.92 ms | -3.6ms ▲ |
| Min | 17.98 ms | |
| Max | 545.17 ms | |
| p90 | 302.85 ms | -5.7ms ▲ |
| p95 | 374.47 ms | +13.8ms ▼ |
| p99 | 454.3 ms | -13.6ms ▲ |
| Wall time | 117.309s | |
| Throughput | 5.73 ann/s | |

**Result Summary:** Cascade retry rescued 9 previously-blank crops, but 8 of the 9 were wrong predictions — a 11% success rate on the hardest failures. Sharpen+dilate finds signals that the model cannot reliably decode, adding noise more than signal. The speed overhead (+18ms median) is also unattractive for only +0.2pp accuracy at the cost of -0.9pp precision.

---

### EXP-09+10 — Bbox expansion + cascade retry (combined)

**Verdict: ACCEPTED (marginal)**

> Combines EXP-09 (10% bbox expansion) and EXP-10 (cascade sharpen+dilate retry) to test whether their effects compound. The rationale is that expanded crops reduce the no-text rate, so the cascade operates on a smaller pool of harder failures — potentially making its signal-to-noise better than EXP-10 alone. Expected at least additive improvement (+3.5-6pp) if the two techniques target different sub-populations of failing crops.

#### Accuracy

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| **Text returned** | **453 / 672 (67.4%)** | **+23.6pp ▲** |
| → Correct (exact match) | 258 (38.4%) | +16.1pp ▲ |
| → Wrong text | 195 (29.0%) | +7.6pp ▼ |
| No text returned | 219 (32.6%) | -23.6pp ▲ |
| **Precision** (correct / returned) | **57.0%** | **+6.0pp ▲** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |
|---|---|---|---|---|---|
| portrait | 156 | 2 | 1.3% | +0.7pp ▲ | 36 (23.1%) |
| near_square | 4 | 0 | 0.0% | — | 1 (25.0%) |
| landscape | 54 | 33 | 61.1% | +10.2pp ▲ | 39 (72.2%) |
| wide | 424 | 219 | 51.7% | +23.3pp ▲ | 347 (81.8%) |
| very_wide | 34 | 4 | 11.8% | +9.0pp ▲ | 30 (88.2%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs BASE-01 |
|---|---|---|
| Average | 154.82 ms | +9.9ms ▼ |
| **Median** | **141.9 ms** | **+18.3ms ▼** |
| Std dev | 116.93 ms | -1.6ms ▲ |
| Min | 18.7 ms | |
| Max | 617.58 ms | |
| p90 | 328.1 ms | +19.5ms ▼ |
| p95 | 374.95 ms | +14.3ms ▼ |
| p99 | 484.33 ms | +16.5ms ▼ |
| Wall time | 127.781s | |
| Throughput | 5.26 ann/s | |

**Result Summary:** Combining bbox expansion with cascade retry compounded to +3.6pp accuracy (vs +3.4pp for EXP-09 alone), adding exactly 1 more correct prediction. However, the cascade's drag on precision (57.0% vs EXP-09's 57.6%) and +31ms median latency penalty make EXP-09 alone the recommended production config unless raw recall is the priority.

---

## EXP-11 — Temporal Aggregation Analysis (post-hoc, no re-OCR)

**Verdict: SPECIAL ANALYSIS**

> Groups all annotations that share the same ground-truth trailer ID (i.e. the same physical trailer appearing in multiple consecutive drone frames) and applies confidence-weighted majority voting to select one prediction per trailer. No additional OCR is needed — this runs post-hoc on the EXP-03+04+06 result JSON. The ground-truth grouping gives an upper bound on what a real visual tracker could achieve; in production, tracks would be formed by IoU-based bbox linking across adjacent frames.

#### Track Summary

| Metric | Value |
|---|---|
| Total unique trailers (tracks) | 414 |
| Single-frame tracks | 206 (no temporal benefit possible) |
| Multi-frame tracks | 208 (>= 2 frames) |

#### Per-Track Accuracy (confidence-weighted majority vote)

| Scope | Correct | Total | Accuracy | Precision |
|---|---|---|---|---|
| All tracks | 156 | 414 | 37.7% | 53.2% |
| Multi-frame tracks only | 100 | 208 | 48.1% | 62.9% |
| Multi-frame per-annotation baseline | — | — | 38.2% | — |

**Temporal gain on multi-frame trailers: +9.9pp** (from 38.2% per-annotation to 48.1% per-track)

#### Confidence Threshold Sweep (per annotation, no re-OCR)

Dropping low-confidence results trades a small recall loss for significant precision gain.

| Min Confidence | Results Kept | Exact Match % | Precision |
|---|---|---|---|
| >= 0.3 | 428 (63.7%) | 34.8% | 54.7% |
| >= 0.4 | 428 (63.7%) | 34.8% | 54.7% |
| >= 0.5 | 428 (63.7%) | 34.8% | 54.7% |
| >= 0.6 | 377 (56.1%) | 34.5% | 61.5% |
| >= 0.7 | 364 (54.2%) | 34.4% | 63.5% |
| >= 0.8 | 330 (49.1%) | 33.9% | 69.1% |
| >= 0.9 | 260 (38.7%) | 30.7% | 79.2% |

**Result Summary:** Confidence-weighted majority voting raises per-track accuracy from 38.2% to 48.1% on multi-frame trailers (+9.9pp). For 50% of trailers that appear in only one frame, temporal aggregation provides no benefit. A confidence threshold of 0.6 boosts precision from 54.7% to 61.5% at a cost of only -0.3pp exact-match accuracy — a favourable trade for applications where wrong IDs are more costly than missed ones.

---

## EXP-12 — IoU-based Frame-to-Frame Tracker (Option A)

**Verdict: INFEASIBLE FOR THIS DATASET**

> Attempted to link YOLO detections across consecutive drone frames using bbox IoU (greedy max-IoU matching, union-find chaining). The plan was to replicate EXP-11's temporal aggregation with a realistic tracker instead of ground-truth grouping.

#### Root Cause: Moving Camera, No Shared Coordinate System

The fundamental assumption of IoU-based tracking is that the same object occupies **similar pixel coordinates** in adjacent frames. This holds for fixed-mount cameras (CCTV, traffic cams). It does **not** hold for an aerial drone flying over a yard — the drone moves significantly between frames (~2–3s interval), so the same physical trailer lands at completely different pixel positions each frame.

**Diagnostic evidence:**

| Metric | Value |
|---|---|
| Same-GT cross-frame pairs (gap=1) | 249 |
| IoU >= 0.3 among those pairs | 0 / 249 (0%) |
| IoU >= 0.1 among those pairs | 0 / 249 (0%) |
| Tracker output: total tracks | 666 (ideal: 414) |
| Multi-frame links formed | 6 (all contaminated — different GTs merged) |
| Oversplit vs ideal | +252 tracks |

The tracker produced 660 singleton tracks (99% single-frame) because no bboxes overlapped across frames. The 6 multi-frame links it did form were all contaminated (different GT trailers merged), yielding 16.7% accuracy on those 6 tracks vs 38.2% per-annotation baseline — **worse than no tracker at all**.

#### Why Option B (ByteTrack/SORT) Would Have the Same Problem

ByteTrack and SORT are also pixel-space IoU trackers with a Kalman velocity model on top. Since cross-frame IoU is 0.0 across the board, a velocity model cannot bridge the gap. Option B would produce the same oversplit result.

#### What Would Actually Work

To link the same trailer across frames in moving-camera aerial footage, one of these approaches is required:

| Approach | Description | Complexity |
|---|---|---|
| **GPS/GSD geo-registration** | Convert each bbox to real-world GPS coordinates using drone telemetry + altitude. Trailers cluster spatially regardless of frame. | Medium — needs telemetry in image EXIF |
| **Homography stabilisation** | Estimate frame-to-frame homography from feature matches (ORB/SIFT), warp to common reference, then apply IoU. | Medium — CPU-feasible with OpenCV |
| **Re-ID appearance matching** | Embed each crop with a lightweight CNN (e.g. OSNet) and match by cosine similarity across frames. | High — requires a trained re-ID model |
| **Ground-truth grouping (EXP-11)** | Group by known trailer ID. Ideal upper bound — only useful as an analysis/benchmark tool. | Done |

**Result Summary:** IoU-based temporal linking is not viable for this dataset. The drone's movement between frames makes pixel-space bbox overlap effectively zero, causing the tracker to output 666 singletons vs 414 ideal tracks. No accuracy benefit is obtainable from this approach without first geo-registering frames to a common coordinate system or using appearance-based re-identification.

---

## EXP-13A — Stacked-Vertical Portrait Decoder (stitch variant)

**Verdict: REJECTED**

> Portrait crops (h > 2w, 156 of 672) contain **upright stacked letters**, not rotated text — earlier rotation experiments (EXP-01/01B/07) failed because they addressed the wrong failure mode. EXP-13A locates the text column via PaddleOCR's raw `text_detector`, uniformly slices it into bands of ~32 px height, resizes each band to 48 px height, and pastes them side-by-side into a synthetic horizontal strip before running full OCR. Layered on top of the EXP-09 config (10% bbox expansion + EXP-03 pad + EXP-04 thresholds + EXP-06 postprocess). Shape gate: only applied to portrait crops; all other buckets flow through the standard pipeline unchanged.

#### Accuracy (full dataset)

| Metric | Value | Δ vs EXP-09 |
|---|---|---|
| **Text returned** | **422 / 672 (62.8%)** | **-3.6pp ▼** |
| → Correct (exact match) | 257 (38.2%) | 0.0pp |
| → Wrong text | 165 (24.6%) | -3.5pp ▲ |
| No text returned | 250 (37.2%) | +3.6pp ▼ |
| **Precision** (correct / returned) | **60.9%** | **+3.3pp ▲** |

#### Breakdown by Aspect Ratio (portrait is the target bucket)

| Bucket | Total | Correct | Δ vs EXP-09 | Text Returned |
|---|---|---|---|---|
| portrait | 156 | 2 (1.3%) | — | 21 (13.5%) |
| near_square | 4 | 0 | — | 1 (25.0%) |
| landscape | 54 | 33 (61.1%) | — | 39 (72.2%) |
| wide | 424 | 218 (51.4%) | — | 342 (80.7%) |
| very_wide | 34 | 4 (11.8%) | — | 30 (88.2%) |

Shape gate verified working: non-portrait buckets are bit-identical to EXP-09.

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs EXP-09 |
|---|---|---|
| Average | 240.3 ms | +111.8ms ▼ |
| **Median** | **170.5 ms** | **+59.9ms ▼** |
| Std dev | 255.4 ms | +160.6ms ▼ |
| Max | 1191.7 ms | +770.9ms ▼ |
| p95 | 828.0 ms | +527.0ms ▼ |
| Wall time | 187.6s | |
| Throughput | 3.6 ann/s | |

**Result Summary:** Zero accuracy gain on the target bucket (portrait stays at 2/156 = 1.3%). The slice-and-paste strip is too noisy for the recognition model to decode — synthetic concatenation introduces boundary artefacts at every letter junction, and uniform slicing misaligns letters whose spacing isn't uniform. Overall exact-match is flat while median latency +55% (110 ms → 170 ms, portrait itself 660 ms median). Full write-up: [docs/ocr-performance-experiment-13.md](ocr-performance-experiment-13.md).

---

## EXP-13B — Stacked-Vertical Portrait Decoder (per_letter variant)

**Verdict: REJECTED**

> Same column-locate and uniform-slice procedure as EXP-13A, but instead of stitching the slices into a single strip, each slice is OCR'd independently and the recognised characters concatenated top-to-bottom. The hypothesis was that isolating each letter gives the recognition model cleaner input than a synthetic strip with boundary artefacts. Shape gate identical to EXP-13A.

#### Accuracy (full dataset)

| Metric | Value | Δ vs EXP-09 |
|---|---|---|
| **Text returned** | **474 / 672 (70.5%)** | **+4.1pp ▲** |
| → Correct (exact match) | 255 (37.9%) | -0.3pp ▼ |
| → Wrong text | 219 (32.6%) | +4.5pp ▼ |
| No text returned | 198 (29.5%) | -4.1pp ▲ |
| **Precision** (correct / returned) | **53.8%** | **-3.8pp ▼** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Δ vs EXP-09 | Text Returned |
|---|---|---|---|---|
| portrait | 156 | 0 (0.0%) | **-1.3pp ▼** | 73 (46.8%) |
| near_square | 4 | 0 | — | 1 (25.0%) |
| landscape | 54 | 33 (61.1%) | — | 39 (72.2%) |
| wide | 424 | 218 (51.4%) | — | 342 (80.7%) |
| very_wide | 34 | 4 (11.8%) | — | 30 (88.2%) |

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs EXP-09 |
|---|---|---|
| Average | 218.8 ms | +90.3ms ▼ |
| **Median** | **156.1 ms** | **+45.5ms ▼** |
| Max | 1051.0 ms | +630.2ms ▼ |
| p95 | 729.1 ms | +428.2ms ▼ |
| Wall time | 172.3s | |

**Result Summary:** The per-letter variant goes **backwards on the target bucket** — portrait accuracy drops from 2/156 to 0/156 because per-slice OCR recognises something on 73 portrait crops (3.4x more text returned) but gets every single one wrong. Per-letter context is too narrow; the recognition model hallucinates single characters from noisy 1-letter tiles. Precision drops -3.8pp overall. Full write-up: [docs/ocr-performance-experiment-13.md](ocr-performance-experiment-13.md).

---

## EXP-15 — Format-Aware Candidate Rescoring

**Verdict: ACCEPTED (marginal)**

> Post-processing rescorer applied after `postprocess_text()`. Enumerates OCR output variants via a character-confusion table (0↔O/D/Q, 1↔I/L, 2↔Z, 5↔S, 6↔G, 8↔B, H→8), BFS depth ≤ 2, cap 64 candidates. Each candidate is scored against a strict whitelist of trailer-ID formats: `^JBHZ\d{6}$`, `^JBHU\d{6}$`, `^R\d{5}$` (all score 1.0). A rewrite is accepted only if the candidate scores ≥ raw + 0.5 AND Hamming distance ≤ 2. Conservative by construction — requires jumping from score 0.0 to a strict format match. Added on top of the EXP-09 stack (bbox expansion done upstream, EXP-03+04+06 in-service). An earlier v1 included a generic `^[A-Z]{3,4}\d{5,7}$` pattern at score 0.5; it regressed −0.8pp by rewriting real leading-digit IDs like `1RZ96854 → IRZ96854`. The generic pattern was removed.

#### Accuracy

| Metric | Value | Δ vs EXP-09 |
|---|---|---|
| **Text returned** | **418 / 672 (62.2%)** | **-4.2pp ▼** |
| → Correct (exact match) | 259 (38.5%) | +0.3pp ▲ |
| → Wrong text | 159 (23.7%) | -4.4pp ▲ |
| No text returned | 254 (37.8%) | +4.2pp ▼ |
| **Precision** (correct / returned) | **62.0%** | **+4.4pp ▲** |

> Note: text-returned delta vs EXP-09 reflects PaddleOCR run-to-run variance on borderline crops, not a rescorer effect (the rescorer only rewrites existing text — it cannot change the text-returned rate). EXP-15-BASELINE (rescorer off, identical CLI otherwise) reproduced 257/672 = 38.2% / 61.5% precision — within noise of the EXP-09 row.

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Δ vs EXP-09 | Text Returned |
|---|---|---|---|---|
| portrait | 156 | 2 (1.3%) | — | 17 (10.9%) |
| near_square | 4 | 0 | — | 0 (0.0%) |
| landscape | 54 | 33 (61.1%) | — | 39 (72.2%) |
| wide | 424 | 220 (51.9%) | **+0.5pp ▲** | 333 (78.5%) |
| very_wide | 34 | 4 (11.8%) | — | 29 (85.3%) |

Gains concentrated in the `wide` bucket as expected — this is where `R\d{5}` trailer IDs live and where PaddleOCR's character-level errors are most fixable.

#### Rescore tally

| Outcome | Count | Examples |
|---|---|---|
| **Helpful** (wrong → correct) | **2** | `RS0408 → R50408` (gt `R50408`), `R26D42 → R26042` (gt `R26042`) |
| Harmful (correct → wrong) | 0 | — |
| Neutral (wrong → wrong) | 2 | `RZ0833 → R20833` (gt `1RZ08337`), `RS0421 → R50421` (gt `R90421`) |

Only 4 rescores fire across 672 annotations — very conservative.

#### Speed (per annotation, OCR call only)

| Metric | Value | Δ vs EXP-09 |
|---|---|---|
| **Median** | **~138 ms** | **+1ms (noise)** |
| p95 | ~370 ms | noise |

**Result Summary:** Produced 2 clean wins, 0 regressions across 672 annotations. Below the plan's +1-3pp target, but precision improved +0.5pp, latency is noise-level (<1ms), and the rescorer is composable with future experiments. The low rescore count indicates most wrong-text errors in the EXP-09 baseline are **not** simple 1-2 character confusions against the known formats — they're either genuinely different ID shapes or have more than 2 character errors. This bounds the ceiling of pure format rescoring. Currently benchmark-only; production wiring deferred. Full write-up: [docs/ocr-performance-experiment-15.md](ocr-performance-experiment-15.md).

---

## EXP-14 — Text-Space Temporal Voting

**Verdict: REJECTED (as planned) / gated variant marginal**

> Post-hoc analysis over the EXP-15 benchmark JSON (259/672 = 38.5%). Sort annotations by DJI frame-seq number, slide a window of W consecutive seqs, and cluster detections where `edit_distance(text_i, text_j) ≤ max_edit` AND `|aspect_ratio_i - aspect_ratio_j| ≤ ar_tol`. Confidence-weighted vote per cluster; tie-break by known-format match then single-detection confidence. Rewrite every cluster member with the winner. No re-OCR. Premise: EXP-11 showed voting across GT-grouped frames adds +9.9pp on multi-frame trailers — EXP-14 tests whether edit-distance clustering can recover that gain without a geometric tracker (which EXP-12 proved infeasible).

#### Parameter Sweep (no gating, as the plan specified)

| W | max_edit | Touched | Helpful | Harmful | Δ correct | Δ precision |
|---|---|---|---|---|---|---|
| 3 | 1 | 46 | 10 | 9 | +1 (+0.1pp) | +0.2pp |
| 3 | 2 | 90 | 12 | 20 | **−8 (−1.2pp)** | −1.9pp |
| 5 | 1 | 47 | 10 | 10 | 0 | 0 |
| **5** | **2** (plan default) | **102** | **12** | **26** | **−14 (−2.1pp)** | **−3.3pp** |
| 7 | 1 | 48 | 10 | 11 | −1 | −0.2pp |
| 7 | 2 | 116 | 12 | 32 | **−20 (−3.0pp)** | −4.8pp |

The plan's default (W=5, max_edit=2) regresses −2.1pp. Harmful rewrites scale much faster than helpful ones as window and distance loosen.

#### Gated Variants

| Variant | Touched | Helpful | Harmful | Δ correct | Δ precision |
|---|---|---|---|---|---|
| W=3, E=1, `--min-agreement 2` | 16 | 1 | 4 | −3 (−0.4pp) | −0.7pp |
| W=3, E=2, `--format-only --min-agreement 2` | 3 | 0 | 1 | −1 | −0.2pp |
| **W=3, E=1, `--format-only` (best)** | **10** | **4** | **1** | **+3 (+0.4pp)** | **+0.7pp** |
| W=2, E=1 (tightest window, no gate) | 44 | 10 | 9 | +1 | +0.2pp |

#### Accuracy (best gated config: W=3, E=1, `--format-only`)

| Metric | Value | Δ vs EXP-15 baseline |
|---|---|---|
| **Text returned** | **418 / 672 (62.2%)** | — |
| → Correct (exact match) | 262 (39.0%) | +0.4pp ▲ |
| → Wrong text | 156 (23.2%) | -0.4pp ▲ |
| **Precision** (correct / returned) | **62.7%** | **+0.7pp ▲** |

#### Breakdown by Aspect Ratio (best gated config)

| Bucket | Total | Correct | Δ vs EXP-15 |
|---|---|---|---|
| wide | 424 | 223 | **+3** |
| landscape | 54 | 33 | — |
| portrait | 156 | 2 | — |
| very_wide | 34 | 4 | — |
| near_square | 4 | 0 | — |

All gains in `wide`. Other buckets untouched because voting only fires where OCR returned format-matching text.

#### Why the Unrestricted Plan Fails

The yard contains clusters of physically-adjacent trailers with near-sequential numeric IDs. Edit-distance clustering merges different trailers whose IDs differ by 1-2 chars:

```
702520 ↔ 702524   (distance 1, both real GT)
702522 ↔ 702632   (distance 2, both real GT)
676052 ↔ 676065   (distance 2, both real GT)
717    ↔ 714      (distance 1, both real GT)
SFU100885 ↔ SIFU1008858   (different container codes, merged)
```

Aspect-ratio filtering (`ar_tol=0.3`) doesn't discriminate because same-style trailers have identical aspect ratios. The only safe operating point restricts voting to the sparse `^R\d{5}$` sub-population, where within-window edit-distance collisions between different trailers are rare.

**Result Summary:** The plan's ideal of +3–6pp is not reachable via text-space clustering on this dataset. Unrestricted voting regresses −2.1pp. The best safely-gated variant produces +0.4pp / +0.7pp precision — similar magnitude to EXP-15. **Fundamental limit:** text-space edit distance is not a correspondence signal when the ID population is structured (numeric-yard IDs cluster within 1–2 edits of each other). Any meaningful gain toward EXP-11's +9.9pp ceiling requires a real correspondence signal (homography, geo-registration, or re-ID), not a voter. Benchmark-only; not wired to production. Full write-up: [docs/ocr-performance-experiment-14.md](ocr-performance-experiment-14.md).

---

### EXP-16 — Unroll-to-horizontal composer for portrait

**Verdict: SKIPPED — duplicates EXP-13A.**

> Planned as a virtual-horizontalisation pass for stacked-vertical portrait crops. On review of the existing codebase, the pipeline is identical to EXP-13A's `--stacked-vertical stitch` variant (`_stitch_letters` in [tests/preprocessing.py](../tests/preprocessing.py#L349)), which already produced 2/156 portrait correct and no additional lift over EXP-09.

EXP-13 Finding 1 (rec-model gibberish on stitched strips) identified the **recogniser**, not the stitch, as the bottleneck. EXP-16 would have produced the same result. Decision: skip, move to EXP-20 (swap the recogniser). Full write-up: [docs/ocr-performance-experiment-16.md](ocr-performance-experiment-16.md).

---

### EXP-20 — TrOCR as full-pipeline recogniser

**Verdict: REJECTED as drop-in replacement. Hybrid fallback path still open.**

> Swapped PaddleOCR PP-OCRv4 for Microsoft TrOCR (`trocr-base-printed`, ViT encoder + transformer decoder) as the end-to-end recogniser. Motivated by two spikes: (1) both TrOCR-small and TrOCR-base produced gibberish on EXP-13's stitched portrait strip, confirming the stitch itself is the bottleneck for portrait; but (2) TrOCR-base read 4/5 natural horizontal wide crops correctly, including `ATLS03` where PP-OCR returns `ITESOS`. That second finding justified a full-pipeline A/B on 672.

#### Accuracy

| Metric | Value | Δ vs EXP-09 |
|---|---|---|
| **Text returned** | **652 / 672 (97.0%)** | +30.6pp ▼ (TrOCR almost never abstains) |
| → Correct (exact match) | 275 (40.9%) | +2.7pp ▲ |
| → Wrong text | 377 (56.1%) | +28.0pp ▼ |
| No text returned | 20 (3.0%) | -30.6pp ▼ |
| **Precision** (correct / returned) | **42.2%** | **-15.4pp ▼** |

#### Breakdown by Aspect Ratio

| Bucket | Total | Correct | Δ vs EXP-09 |
|---|---|---|---|
| portrait | 156 | **0** | **-2** |
| near_square | 3 | 0 | — |
| landscape | 55 | 33 | ~flat |
| wide | 422 | **235** | **+17** |
| very_wide | 36 | 7 | +3 |

#### Speed

Median **989 ms** (p90 1547 ms, p95 1776 ms). **≈9× slower** than EXP-09's 110 ms median.

#### Result Summary

Real +2.7pp exact-match gain concentrated in `wide` (+17) and `very_wide` (+3). But TrOCR is a language model — it fabricates plausible text on crops PP-OCR correctly rejects (no-text rate collapsed 33.6% → 3.0%, wrong-text rose 189 → 377), so **precision drops 15 pp**. Portrait is *worse*, not better: TrOCR cannot read stacked-vertical either, so the project-critical blocker is untouched. Latency ~9× rules out a straight swap on its own merits.

**Follow-ups that remain viable:**
1. **Disagreement fallback.** Run PP-OCR first, then TrOCR only when PP-OCR's output fails the format gate. Keeps baseline precision, captures wide-bucket gains on ambiguous crops, and pays the latency only where PP-OCR already failed.
2. **TrOCR as a low-confidence tie-breaker.** Similar envelope, more integration work.

Neither helps portrait. EXP-18 (VLM fallback) and EXP-17 (per-character detection + unroll) both attempted to attack the recogniser on its portrait failure mode in Round 2; both were rejected (see EXP-18 and EXP-17 sections below, and Round 2 Close-Out for the full list). Full write-up: [docs/ocr-performance-experiment-20.md](ocr-performance-experiment-20.md).

---

### EXP-18 — VLM portrait fallback (Gemini 2.5 Flash)

**Verdict: REJECTED for Gemini 2.5 Flash. Scope broadened, experiment continues under EXP-22.**

> Routed portrait-bucket crops (aspect < 0.5, 156/672) to Google Gemini 2.5 Flash via a format-gated prompt (`JBHZ`+6digits / `JBHU`+6digits / `R`+5digits / `UNKNOWN`), with on-disk response cache and Lanczos upscaling to a 768-px long side. The plan's a-priori target was 120/156 (≈77%) on the portrait bucket. First pass was run via `--portrait-strategy vlm` on the full portrait subset; second pass via a 15-crop sanity script after upscaling was added.

#### Accuracy

| Run | Correct | Notes |
|---|---|---|
| Scope 1 — 119/156 crops, no upscaling | **0/119** | Model emitted format-valid `JBHZ######` strings with fabricated digits. Cut short at the daily 20-RPD free-tier cap. |
| Scope 2 — 15 diverse crops, with 768-px upscaling | **0/15** | 12/15 returned `UNKNOWN` or empty (format gate working); 3/15 fabricated. Upscaling shifted Gemini from confident fabrication to abstention, not accuracy. |

#### Plumbing calibration (wide bucket, 5 crops)

To rule out pipeline bugs masking as model failure, ran Gemini on 5 wide-bucket crops with a generic OCR prompt (no format gate):

| GT | Gemini raw | Outcome |
|---|---|---|
| `ATLS03` | `'ATIS03'` | L↔I confusion |
| `ATLS03` | `'ATIS03'` | L↔I confusion |
| `R44045` | `'R4045'` | missing one `4` |
| `13208` | `'13208'` | ✔ exact |
| `1393` | `'1393'` | ✔ exact |

**2/5 exact, 5/5 legibly reading pixels.** Pipeline works; portrait failure is model-specific.

#### Speed

~3–10 s per call (free-tier hosted API, 5–20 RPM caps). Not production-viable even if accuracy were good.

#### Result Summary

Gemini 2.5 Flash **cannot read stacked-vertical trailer IDs regardless of input resolution** — it either fabricates digits or abstains. This is a catastrophic layout-specific failure, not a pixel-reading failure (the plumbing calibration confirms the model reads ordinary wide crops at ~40% exact / 100% near-miss). The original EXP-18 plan was too narrowly scoped to "VLM fallback" — vertical / stacked text is a mature OCR subdomain (ISO-6346 container codes, CJK newspapers) with specialized solutions that were not considered. Full write-up: [docs/ocr-performance-experiment-18.md](ocr-performance-experiment-18.md).

**Follow-ups in priority order (Round 3 — see "Round 2 close-out" below for status):**
1. **Train a dedicated per-character detector** on ~50 hand-labelled portrait crops (YOLO/PaddleDetection fine-tune). EXP-22 and EXP-17 between them ruled out the cheap "use a different OCR model's detector" path; the remaining route is to label and train. Multi-day effort, real engineering.
2. **PaddleOCR-VL (0.9B)** — Baidu's VLM successor to PP-OCR, purpose-built for vertical text, pre-trained weights on HuggingFace.
3. **Qwen2.5-VL-7B-Instruct** (Apache 2.0, local) — benchmark-topping multi-orientation recogniser. Heavier infra.
4. **Claude Haiku 4.5** — only hosted VLM with a credible shot at arbitrary layouts. ~$0.05 for a 15-crop sanity.

Cost of this experiment: **$0** (burned two free-tier keys' daily quotas, no billable usage).

---

### EXP-22 — PP-OCRv5 drop-in replacement for PP-OCRv4

**Verdict: REJECTED as drop-in. Earlier "v5 detector enables EXP-17" finding RETRACTED — see corrected section below.**

> Installed `paddleocr 3.5.0` + `paddlepaddle 3.3.1` (CPU, MKLDNN disabled) into an isolated `.venv-paddle-v5/` to avoid dep-conflict risk on the existing v4 stack. All EXP-09 detection-tuning parameters translated 1:1 to the v3.x API (e.g. `det_db_thresh` → `text_det_thresh`), so the comparison is a clean isolation of the model swap. Used `PP-OCRv5_server_rec` — the multilingual variant that the v5 release notes attribute vertical-text capability to — rather than the English-only mobile default.

#### 15-crop portrait sanity

| Hits | Latency (median) | Notes |
|---|---|---|
| **0/15** | ~1650 ms/crop | 14/15 returned empty (detector found per-char boxes but rec couldn't read isolated upright chars); 1/15 returned a partial wrong answer on a non-stacked GT |

#### Spot-check vs v4 EXP-09

| Annotation | Bucket | GT | v4 (EXP-09) | v5 (EXP-22) |
|---|---|---|---|---|
| ann 672 | landscape | `13208` | `'13208'` ✔ | `'3208'` (drops leading `1`) |
| ann 14 | portrait | `JBHZ672061` | `''` (0 detection boxes) | `''` (~~6 boxes~~ → corrected: **0 boxes** on re-verification, see below) |

#### Speed

Median ~**1650 ms/crop** with MKLDNN disabled (a paddlepaddle 3.3.1 + Windows CPU bug — `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]` — forced this off). EXP-09 is 110 ms. **~15× slower than v4** even before considering accuracy.

#### Hidden Finding — RETRACTED (2026-04-25)

The previous revision of this section claimed v5's detector returned **6 per-character boxes** on `ann 14` and proposed EXP-17 build on top of that. **The claim does not reproduce.** Verified at the start of EXP-17 execution under three threshold configurations × two input scales (six combinations total) — all return **zero boxes**. The full 156-portrait-crop precompute reinforces the result: 131/156 crops have zero boxes, max 2 boxes on any crop, never the per-char count the plan required.

| Verification | Result |
|---|---|
| 156-crop precompute (EXP-09 tuning) | 25/156 with any boxes; min=0, **median=0, max=2** |
| `ann 14` × EXP-09 tuning × 58×326 original | **0 boxes** |
| `ann 14` × EXP-09 tuning × 232×1304 (4× upscale) | 0 boxes |
| `ann 14` × default thresholds × original | 0 boxes |
| `ann 14` × default thresholds × upscaled | 0 boxes |
| `ann 14` × extreme-permissive (`thresh=0.1`, `box_thresh=0.15`) × original | 0 boxes |
| `ann 14` × extreme-permissive × upscaled | 0 boxes |

Raw outputs: [tests/results/exp22_v5_detection.json](../tests/results/exp22_v5_detection.json), [tests/results/verify_v5_ann14.json](../tests/results/verify_v5_ann14.json). The original "6 boxes" claim could not be reproduced and most likely came from misreading the result schema (e.g. counting `rec_polys` corners or dictionary keys rather than `len(dt_polys)`); the previous session's exact code path is not preserved.

#### Result Summary

v5 is **not viable as a drop-in replacement** for v4 — slower, no portrait improvement, mild landscape regression. With the hidden-finding retraction, v5's detector is also not a workable base for an EXP-17 per-char-detection pipeline on this dataset. The full 672-annotation run was skipped per the plan's §8 decision rubric (< 10/156 → REJECT).

The v5 install lives only in `.venv-paddle-v5/`; production deps and the deployed Cloud Run image are unchanged. Full write-up: [docs/ocr-performance-experiment-22.md](ocr-performance-experiment-22.md).

---

### EXP-17 — Per-character detection + unroll (v5-detector variant)

**Verdict: REJECTED — premise invalidated at execution. Round 2 closes here.**

> The revised EXP-17 plan banked on EXP-22 §10's "v5 returns per-char boxes natively". Verification before any benchmark run (see retraction above) showed that claim does not reproduce: zero boxes on `ann 14` across six config × scale combinations, median zero boxes across the full 156-crop portrait set, max 2 boxes on any crop. With no per-char detection available, both Strategy A (stitch into horizontal strip) and Strategy B (per-char rec) die at the precompute step before reaching the benchmark. The original v4-retune fallback hypothesis (extreme DB thresholds + larger `det_limit_side_len`) was not exercised in this round.

Full write-up: [docs/ocr-performance-experiment-17.md](ocr-performance-experiment-17.md).

---

## Round 2 Close-Out

**Round 2 ends at EXP-09's 38.2 % exact-match / 57.6 % precision** as the production configuration. EXP-15 and EXP-09+10 added marginal lifts (+0.3pp / +0.2pp) but neither survived precision-cost analysis as a clear win. Portrait-bucket accuracy stays at **2/156 (1.3 %)** as a known limitation.

**What was tried for portrait in Round 2 (all rejected):**

| Experiment | Approach | Outcome |
|---|---|---|
| EXP-13A/B | Stitch upright stacked letters into a horizontal strip via uniform slicing | Rejected — stitched strips fail to read |
| EXP-16 | Same as EXP-13 with finer slicing tuning | Skipped (duplicates EXP-13A) |
| EXP-18 | VLM fallback (Gemini 2.5 Flash) on portrait crops | Rejected — 0/15 hits, hallucinates format-valid IDs |
| EXP-22 | PP-OCRv5 drop-in (recogniser + detector swap) | Rejected — 0/15 hits, ~15× slower |
| EXP-17 | v5 detector → v4 recogniser bridge via per-char stitch | Rejected — premise (v5 per-char detection) does not reproduce |

**Round 3 candidate levers (deferred):**

1. **Train a dedicated per-character detector** on ~50 hand-labelled portrait crops (YOLO or PaddleDetection fine-tune). The remaining route after off-the-shelf detectors were ruled out.
2. **PaddleOCR-VL (0.9B)** — Baidu's purpose-built vertical-text VLM successor to PP-OCR.
3. **Qwen2.5-VL-7B-Instruct** (Apache 2.0, local) — benchmark-topping multi-orientation recogniser. Heavier infra.
4. **Claude Haiku 4.5** — hosted VLM. ~$0.05 for a 15-crop sanity to settle whether any general-purpose VLM can read these crops.

**Production state at Round 2 close:**

- `app/ocr_processor.py`: PaddleOCR PP-OCRv4 with EXP-03 padding + EXP-04 detector tuning + EXP-06 postprocessing (unchanged from Round 1's accepted config; bbox expansion is now upstream in the main app).
- `requirements.txt` / Dockerfile / Cloud Run image: unchanged.
- `.venv-paddle-v5/` exists as benchmark-only scaffolding for future v5 work; not on the deployment path.
- Verification artefacts retained: `tests/precompute_v5_detection.py`, `tests/verify_v5_ann14_v2.py`, plus their JSON outputs under `tests/results/`.

---

## Round 3

Round 3 opens with a re-baseline on a cleaner dataset (EXP-23), runs through one classical-OCR probe (EXP-19) and one layout-aware-OCR plan that was deferred (EXP-24), and lands on the first experiment that **actually moves the JBHZ portrait bottleneck**: a hosted-VLM cascade fallback (EXP-25, ACCEPTED).

### EXP-23 — Re-baseline on cleaner dataset (20260423)

**Verdict: ACCEPTED as new Round 3 baseline.**

> Re-ran the EXP-09 production config on a fresh 251-image / 419-annotation set (`tests/dataset/20260423`) shot under better lighting than the original 20260406 set. Same code, same params — pure dataset-quality A/B.

| Metric | EXP-09 on 20260406 | EXP-23 on 20260423 |
|---|---|---|
| Overall EXACT | 257 / 672 (38.2 %) | **262 / 419 (62.5 %)** |
| Precision | 57.6 % | **79.9 %** |
| Portrait | 2 / 156 (1.3 %) | ≈ 9 / 119 (7.6 %) |
| JBHZ alpha-prefix vertical | 0 / 95 | 0 / 89 |

**Image quality lifted overall by ~24 pp / ~22 pp precision but moved the JBHZ portrait sub-bucket by zero.** That makes JBHZ the architectural bottleneck — not a pixel problem, a layout problem — and frames the Round 3 candidate space.

20260423 becomes the canonical Round 3 evaluation set for everything below.

---

### EXP-19 — Tesseract PSM 5/6 for portrait crops

**Verdict: REJECTED — marginal value, fails to move JBHZ.**

> Hypothesised that Tesseract's PSM 5 ("single uniform block of vertically aligned text") would beat PaddleOCR's horizontal-line assumption on stacked vertical Latin glyphs. Tesseract 5.5.0 (UB Mannheim build, OEM 1, A-Z 0-9 whitelist) on all 119 portraits in 20260423 produced **4/119 EXACT and 10/119 within edit-distance 2**, with only **3 unique correct hits** vs PaddleOCR. Pre-flight discovered PSM 5 returns empty/garbage on 5/5 sample crops (Tesseract's layout analyser doesn't engage PSM 5 on Latin stacked-vertical input); PSM 6 ("uniform block, default Latin") was the only mode that engaged the LSTM at all.

| GT format | n | EXACT | NEAR (≤2) | empty |
|---|---:|---:|---:|---:|
| ALPHA4_DIGIT6 | 95 | 2 | 4 | 26 |
| NUMERIC | 17 | 1 | 6 | 5 |
| OTHER | 7 | 1 | 0 | 1 |
| **TOTAL** | 119 | **4** | **10** | 32 |

**All 3 strict unique wins are JBHU prefix; zero JBHZ hits across either engine.** The load-bearing sub-bucket (~89/95 alpha4 portraits) is unaffected. Median latency ~140 ms/crop on CPU.

Cost: ~45 minutes total (Tesseract install + 3 spike scripts + 119-crop sweep + write-up). Cost of avoided full benchmark wiring: ~half a day. The "verify priors before building" memory rule earned its keep — a 1/20 sample looked tempting but the widened 4/119 picture was the real signal. Full write-up: [docs/ocr-performance-experiment-19.md](ocr-performance-experiment-19.md).

---

### EXP-24 — Surya OCR for portrait crops

**Verdict: PLANNED — deferred behind EXP-25.**

Layout-aware transformer-based OCR (Apache 2.0). Plan written; not run. EXP-25's strong result reduced its priority — Surya remains a fallback option only if EXP-25 hits a production blocker (cost, latency, or vendor reliance). Plan: [docs/ocr-performance-experiment-24.md](ocr-performance-experiment-24.md).

---

### EXP-25 — Qwen3-VL-8B portrait cascade fallback (via OpenRouter)

**Verdict: ACCEPTED — Round 3 headline. JBHZ bottleneck broken; benchmark-only wiring shipped, production rollout is a separate plan.**

> Hosted-VLM cascade: PaddleOCR runs first; if a portrait crop returns no text **OR** text that doesn't match `JBHZ\d{6} / JBHU\d{6} / R\d{5}`, fall back to Qwen3-VL-8B-Instruct via OpenRouter's chat-completions API. Strict format gate + `UNKNOWN` self-rejection sentinel preserve precision; `ocr_text` is only overwritten when Qwen returns format-valid output, so PaddleOCR's numeric-portrait wins are kept by construction. On-disk PNG-hash response cache makes re-runs zero-cost.
>
> Initially planned as `qwen-2.5-vl-7b-instruct`; OpenRouter has retired the 2.5 line at this size, swapped to `qwen-3-vl-8b-instruct` (newer generation, same tier). 32B-Instruct comparison run on the same 119 portraits scored materially worse (27/119 vs 8B's 63/119) — provider-specific fine-tune differences, not a model-size effect. **8B is the winner.**

#### Pre-flight (5 crops → 119 crops)

| Stage | Result |
|---|---|
| 5-crop spike (same crops as EXP-19) | **4/5 EXACT** including 3 JBHZ; 1/5 cleanly self-rejected as UNKNOWN |
| Widened 119-crop sweep | **63/119 EXACT (53 %)**, 41 UNKNOWN (clean reject), 0 errors |
| 32B escalation | 27/119 — REJECTED |

JBHZ alpha-prefix sub-bucket: **47/92 EXACT (51 %)** vs prior 0/89 across every other engine attempted.

#### Cascade vs spike — gate design lesson

| Run | Portrait correct (n=119) | Precision |
|---|---|---|
| Spike (full Qwen replacement) | 63 / 119 (53 %) | — |
| Run A — strict no-text gate | 48 / 119 (40 %) | 57 % |
| Run B — no-text **or** format-miss gate | **59 / 119 (50 %)** | **70 %** |

Strict no-text gate left 15 wins on the table — PaddleOCR returned hallucinated horizontal-text on portrait crops with confidence > 0.6, so the no-text trigger never fired. Upgrading to no-text-or-format-miss recovers most of those. The remaining 4-case gap vs the spike is the format gate doing its job (rejecting non-whitelisted Qwen outputs like `7322039P` against malformed-GT crops). The cascade is *more precise* than the spike by design.

#### Full 419 — Round 3 headline

`python tests/benchmark_ocr.py --exp-id EXP-25-qwen-full --dataset 20260423 --portrait-strategy qwen --format-rescore on`.

| Metric | EXP-23 baseline | EXP-25 cascade | Δ |
|---|---|---|---|
| **Overall EXACT** | 262 / 419 (62.5 %) | **314 / 419 (74.9 %)** | **+52 (+12.4 pp)** |
| **Precision** | 79.9 % | **83.3 %** | **+3.4 pp** |
| Portrait | ≈ 9 / 119 | 59 / 119 (49.6 %) | +50 |
| JBHZ valid-format | 0 / 89 | 46 / 88 | +46 |
| Wide | 218 / 255 | 222 / 255 (87.1 %) | within noise |
| Landscape | 33 / 35 | 30 / 35 (85.7 %) | within noise |
| Median latency | ~110 ms | 148 ms | +38 ms |

Qwen invoked 119 times across the 419-crop run (every portrait that triggered the gate), 62 format-valid hits, 57 misses (UNKNOWN or non-whitelisted). 119 cache files on disk after the run; subsequent runs are byte-identical at near-zero API cost.

#### Cost

- 5-crop spike: ~$0.001
- 119-crop widened sweep (8B): ~$0.0024
- 32B escalation comparison: ~$0.02
- Full wiring + 419-crop benchmark + iteration: ~$0.01 cumulative
- **Total experiment cost: ~$0.05 in API spend, ~3 hours engineering**

Per-call: ~$0.00004 at 8B Parasail pricing on OpenRouter (≈ 174 input + 9 output tokens). Production-scale: 100 K calls/year ≈ $4.

#### Result Summary

EXP-25 is the highest-ROI experiment in the project's history. JBHZ alpha-prefix portrait — the bottleneck Round 1 + Round 2 + the early Round 3 probes (EXP-18 Gemini, EXP-19 Tesseract, EXP-20 TrOCR) all failed to move — drops from 0/89 to 46/88 (52 %). Overall metric jumps **62.5 % → 74.9 %**, precision *also improves* by 3.4 pp (the format gate makes the cascade strictly safer than running PaddleOCR alone on portraits). Median latency hit is +38 ms.

Implementation:
- [tests/qwen_portrait.py](../tests/qwen_portrait.py) — singleton, OpenRouter HTTP client, 90 s timeout, 3-retry exponential backoff, on-disk PNG-hash response cache at `tests/results/qwen_cache/`.
- [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — `--portrait-strategy qwen` + `--qwen-model` flags; cascade block sits between the standard pipeline output and the EXP-10 sharpen+dilate retry; only overwrites `ocr_text` on Qwen format-valid hit.
- Production wiring (`app/ocr_processor.py`) intentionally untouched — separate rollout plan needed (key plumbing, request budget, timeout handling, OpenRouter provider pinning for determinism).

Full write-up: [docs/ocr-performance-experiment-25.md](ocr-performance-experiment-25.md).

#### Open follow-ups (not blockers for ship)

1. **Z↔N confusion** in `app/utils.py:_CONFUSIONS` — recovers 1-2 more edit-1 misses post-rescore (~5 min change).
2. **Production wiring plan** — separate document, includes OpenRouter key, cost budget, timeout / retry policy, fallback when API is unavailable.
3. **Provider pinning** — at the OpenRouter level via `provider:` block; eliminates the 1-2 cases of cross-provider non-determinism observed in the cascade-vs-spike comparison.

---

### EXP-26 — GOT-OCR-2.0-hf on portrait + wide

**Verdict: REJECTED — both gates failed in characterisation. Spike scripts retained, no production wrapper or benchmark wiring added.**

> Probe a 580 M-param end-to-end OCR transformer ([stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf), Apache-style license) as an alternative to PaddleOCR on the wide bucket (PaddleOCR ceiling 222/255 = 87.1 % EXP-23) and as a portrait-cascade alternative to EXP-25 / EXP-27. Loaded via the upstreamed `GotOcr2ForConditionalGeneration` in transformers 4.54.1, torch 2.4.1+cpu, isolated `.venv-got-ocr/`.

#### Spike1 (5 portraits + 5 wides, 2026-04-26)

`.venv-got-ocr/Scripts/python.exe tests/got_ocr_spike.py`

| Bucket | EXACT | NEAR(≤2) | Median ms | Notes |
|---|---|---|---|---|
| Portrait (5) — same crops as EXP-19/EXP-25/EXP-27 | 0/5 | 0/5 | 13,565 | Same JBHZ wall as Tesseract / TrOCR. Output examples: `BHU235644J20000…`, `BHN6702088J88…`. Char fragments visible but no consistent start anchor; loops on filler. |
| Wide (5) — PaddleOCR-easy crops, ann_ids 3-7 | 5/5 | — | 13,193 | Loop pattern `XXXX…` correctly read once → loop-dedupe extracts clean GT match. Three of five cleanly read once and stopped. |

**Portrait gate (≥ 1/5 NEAR AND median ≤ 30 s): FAIL.** Drop the portrait cascade path; EXP-25 / EXP-27 remain the portrait answer.

**Wide gate (≥ 3/5 EXACT after dedupe): PASS at 5/5.** But the 5 chosen wides are PaddleOCR-easy — the load-bearing question is whether GOT rescues the wides where PaddleOCR fails. Spike2 widens to all 265 wides.

#### Spike2 wide — full 265 wides, cross-referenced vs PaddleOCR EXP-23 baseline

`.venv-got-ocr/Scripts/python.exe tests/got_ocr_spike2_wide.py`

Wall: ~60 min (rate 0.07 crop/s on CPU). Median 13,400 ms/crop. Result file: [tests/results/got_ocr_spike2_wide_20260423.json](../tests/results/got_ocr_spike2_wide_20260423.json).

| Metric | Count | Rate |
|---|---|---|
| GOT-OCR EXACT (full replacement) | 148/265 | 55.8 % |
| PaddleOCR EXACT (EXP-23 baseline) | 225/265 | 84.9 % |
| Cascade (paddle → GOT fallback on paddle-fail) | 231/265 | 87.2 % |

**Per-crop cross-tab vs PaddleOCR:**

| Outcome | Count |
|---|---|
| Both correct | 142 |
| Both wrong   | 34  |
| **GOT rescues** (GOT ✓, paddle ✗) | **6** (15 % rescue rate on 40 paddle failures) |
| **GOT regressions** (GOT ✗, paddle ✓) | **83** (37 % regression rate on 225 paddle successes) |

#### Why both deployment paths fail

1. **Full replacement** (`--horizontal-engine got_ocr`) — net **−77 crops** vs PaddleOCR. 30+ pp regression. Catastrophic.
2. **Cascade fallback** (paddle first → GOT on paddle-fail) — only **+6 wins** = +0.7 pp on the 419-crop run, gated on a model 100× slower than PaddleOCR (13.4 s vs 110 ms). Negative ROI even with non-zero gain.

**Failure-mode analysis:** GOT-OCR is **correlated** with PaddleOCR, not complementary. It reads cleanly-printed wides (where paddle also reads them) and fails on the same hard wides (motion blur, occlusion, embossing). Same difficulty distribution as paddle → no rescue signal where it matters. Wider OCR-corpus pretraining (charts, formulas, music) didn't help on this image distribution.

#### Cost

- Venv setup (~1.5 GB on disk: torch 2.4.1+cpu wheel + transformers + accelerate + huggingface_hub).
- Model checkpoint pull from HF (1.14 GB, one-time).
- Spike1 (5+5): ~2 min wall.
- Spike2 (265 wides): ~60 min wall.
- Engineering time: ~1.5 hr.
- API cost: $0 (fully local).

#### Result Summary

GOT-OCR-2.0 was an open candidate the team had not yet characterised; this experiment closes it definitively for both portrait and wide on the 20260423 dataset. The 5+5 spike1 framing (portrait + horizontal A/B) was load-bearing — the wide-bucket result on baseline-easy crops looked promising on its own (5/5 EXACT) and would have been a misleading green light without the full 265-crop widen. EXP-19's lesson ("widen the spike before declaring abort") applies here in reverse: widen before declaring acceptance.

Implementation:
- [.venv-got-ocr/](../.venv-got-ocr/) + [requirements-got-ocr.txt](../requirements-got-ocr.txt) — isolated env, retained for any future re-test (different quant, fine-tune, etc.).
- [tests/got_ocr_spike.py](../tests/got_ocr_spike.py) — 5+5 pre-flight.
- [tests/got_ocr_spike2_wide.py](../tests/got_ocr_spike2_wide.py) — 265-crop cross-reference.
- No `tests/got_ocr_processor.py` written; no `--horizontal-engine` flag added to `tests/benchmark_ocr.py`.

Full write-up: [docs/ocr-performance-experiment-26.md](ocr-performance-experiment-26.md).

---

### EXP-27 — Local VLM ladder probe — qwen2.5vl:3b via Ollama

**Verdict: VIABLE — characterised in spike, production wiring is a separate plan.**

> Decoupling probe: can a CPU/GPU-local VLM match EXP-25's hosted Qwen3-VL-8B accuracy and replace the OpenRouter dependency? Initial attempt (`qwen3-vl:8b` via Ollama) failed — the model is thinking-mode-enabled and burnt the entire `max_tokens=32` budget on chain-of-thought before emitting any answer (0/5 EXACT). After bumping to `max_tokens=512`, recovered to 24/80 EXACT but with a 27.5 pp gap vs OpenRouter — q4_K_M quantisation hits OCR-fine character recognition harder than expected. Switched track to non-thinking small-VLM ladder.

#### Ladder summary (5-crop pre-flight gate)

| Rung | Model | Params (Q) | EXACT @ 5 | Latency | Verdict |
|---|---|---:|:---:|---:|---|
| 1 | `qwen3-vl:8b` (q4_K_M) | 8.8B | 0/5† / 3/5‡ | 14-50 s | REJECTED — thinking + quant gap (24/80 vs 46/80, killed at 80) |
| 2.1 | `moondream:1.8b-v2-q4_0` | 1.8B (q4) | 0/5 | 0.5 s | REJECTED — image captions only, can't follow OCR prompts |
| 2.2 | `granite3.2-vision:2b` | 2.5B (q4) | 0/5* | 1.0 s | REJECTED — refusals + JOHNSON hallucinations + partial reads |
| 2.3 | **`qwen2.5vl:3b`** | 3.8B (q4_K_M) | **3/5** | 4 s | **VIABLE — proceed to spike2** |

† at max_tokens=32. ‡ at max_tokens=512. * with simple "Read the text." prompt; the EXP-25 detailed prompt triggered "Unfortunately, I am unable…" refusals.

#### Headline — qwen2.5vl:3b spike2 (full 119 portrait crops)

`QWEN_LOCAL_MODEL=qwen2.5vl:3b python tests/qwen_local_spike2.py`

| Metric | EXP-25 OpenRouter Qwen3-VL-8B | qwen2.5vl:3b local | Δ |
|---|---|---|---|
| Portrait EXACT (n=119) | 63 (53 %) | **60 (50.4 %)** | -2.6 pp |
| **JBHZ EXACT (n=92)** | **47 (51 %)** | **50 (54 %)** | **+3 ⬆** |
| JBHU EXACT (n=8) | _(not previously broken out)_ | 6 (75 %) | — |
| NUMERIC EXACT (n=17) | _(not previously broken out)_ | 4 (24 %) | — |
| Median ms / crop | ~1,144 ms | 4,109 ms | +2,965 ms |
| p95 ms / crop | _(unknown)_ | 7,996 ms | — |
| Wall, 119 crops | ~150 s | 594 s | +444 s |
| Projected 419-cascade wall | ~140 s | ~450 s | +310 s |
| Cost per 419 run | ~$0.05 | $0 | $0 |
| Hardware | n/a (vendor) | local Ollama on dev RTX | — |

**The critical finding: local qwen2.5vl:3b *exceeds* the hosted Qwen3-VL-8B on the load-bearing JBHZ stacked-vertical sub-bucket** (50/92 vs 47/92), at zero recurring cost. ~3.5× slower per crop than OpenRouter but the projected 419-cascade wall (~7.5 min) is well inside the doc's 30-min threshold.

#### Per-crop diff vs OpenRouter on the same 119 ann_ids

| Outcome | Count |
|---|---:|
| Both correct | 44 |
| Both wrong   | 40 |
| Local-only wins (OR returned UNKNOWN / wrong) | **16** (13 JBHZ, 3 NUMERIC) |
| OR-only wins (local UNKNOWN'd / wrong) | **19** (8 NUMERIC, 9 ALPHA4 near-misses, 2 OTHER) |

The two models are partially complementary. Local is more confident on JBHZ alpha-prefix vertical (the EXP-25 bottleneck); OpenRouter is more confident on numerics. Theoretical local-first → OR-fallback cascade ceiling: 60 + 19 = 79 / 119 (66 %). Decision lives in the production-wiring plan.

#### Operational findings worth keeping

- **Ollama auto-uses GPU when CUDA is present.** Initial plan called this a "CPU spike"; Task Manager showed RTX at 75 % utilisation while CPU sat at 21 %. Any *true* CPU run requires `OLLAMA_NUM_GPU=0` set in env.
- **Thinking-mode VLMs need a 512+ token budget.** EXP-25's `max_tokens=32` works for non-thinking models but produces empty content on thinking models — the budget is consumed by the `reasoning` field before any `content` is emitted.
- **Prompt tolerance varies by model size.** The EXP-25 detailed prompt works on capable models (Qwen2.5-VL-3B, Qwen3-VL-8B, hosted) but is misinterpreted as a description request by smaller models (moondream narrates; granite refuses). Spike scripts now accept `SPIKE_PROMPT` env override for ladder probes.

#### Cost

- 4 model pulls (qwen3-vl:8b 6.1 GB, moondream 1.7 GB, granite 2.4 GB, qwen2.5vl:3b 3.2 GB), 3 of which were `ollama rm`'d after rejection.
- Engineering time: ~3 hr (initial qwen3-vl probe + ladder probe + spike2 + write-up).
- API cost: $0 (this experiment is fully local).
- Disk peak: ~6 GB during qwen3-vl:8b run (now removed); current footprint qwen2.5vl:3b 3.2 GB.

#### Result Summary

EXP-27 produces a viable **local-Ollama drop-in candidate** for EXP-25's OpenRouter dependency. `qwen2.5vl:3b` matches EXP-25's portrait accuracy within 2.6 pp overall and *beats* it on JBHZ specifically. ~3.5× per-crop latency but well inside acceptable; zero recurring cost. The four-rung ladder also produced two operational findings (Ollama-auto-GPU, thinking-mode budget) worth carrying forward.

Implementation:
- [tests/qwen_local_processor.py](../tests/qwen_local_processor.py) — singleton subclass of `QwenPortraitProcessor`; overrides endpoint (Ollama OpenAI-compat at `/v1/chat/completions`), drops auth, swaps cache dir, raises `max_tokens` to 512, raises timeout to 300 s.
- [tests/qwen_local_spike.py](../tests/qwen_local_spike.py) + [tests/qwen_local_spike2.py](../tests/qwen_local_spike2.py) — bypass the processor to capture raw pre-format-gate output for honest edit-distance.
- [tests/results/qwen_local_spike_qwen25vl3b_20260423.json](../tests/results/qwen_local_spike_qwen25vl3b_20260423.json) — full 119-crop result, schema matches the EXP-25 OpenRouter spike for one-diff comparison.

Full write-up: [docs/ocr-performance-experiment-27.md](ocr-performance-experiment-27.md).

#### Open follow-ups

1. **Production wiring plan (EXP-28)** — adapt `qwen_local_processor.py`, add `--portrait-strategy qwen_local` to `tests/benchmark_ocr.py`, run full 419-crop cascade benchmark, propagate to `app/ocr_processor.py`. Open question: pure local vs local-first / OR-fallback hybrid.
2. **Quantisation ladder** — if the production benchmark falls short, retry q5_K_M / q8_0 / fp16 (~10-15 GB pull, ~1 hr re-run). q4_K_M was the spike point.
3. **True CPU characterisation** — for production hardware planning. Set `OLLAMA_NUM_GPU=0`. Currently deferred (the GPU result is what matters for the dev-machine scale test).

---
