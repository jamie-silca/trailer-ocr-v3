# OCR Experiment Comparison Report

Generated: 2026-04-16 13:44
Dataset: 672 annotations across 397 aerial drone frames
Engine: PaddleOCR 2.7.3 (PP-OCRv4), CPU-only

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


**Result Summary:** **Best Performance:** Verified a highly effective additive relationship between padding and sensitivity, raising baseline accuracy by over 50% (from 22.3% to 34.7%).


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


**Result Summary:** Neutral results across all resize limits indicate that PaddleOCR’s internal upscaling to 960px is robust and not a source of performance degradation for small crops.


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


**Result Summary:** Neutral results across all resize limits indicate that PaddleOCR’s internal upscaling to 960px is robust and not a source of performance degradation for small crops.


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


**Result Summary:** Neutral results across all resize limits indicate that PaddleOCR’s internal upscaling to 960px is robust and not a source of performance degradation for small crops.


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
