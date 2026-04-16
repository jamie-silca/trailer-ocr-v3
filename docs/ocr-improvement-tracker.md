# OCR Improvement Tracker

Iterative improvement log for the trailer OCR microservice. Inspired by the [autoresearch](https://github.com/karpathy/autoresearch) methodology: one variable at a time, measured against a canonical baseline, no repeating dead ends.

---

## Current Best Configuration

> **Last updated: 2026-04-16**

| Metric | Baseline (BASE-01) | Best (EXP-03+04) | Delta |
|---|---|---|---|
| Exact match accuracy | 22.3% (150/672) | **34.7% (233/672)** | **+12.4pp** |
| Precision (when text returned) | 51.0% | **54.4%** | +3.4pp |
| Text returned | 294 (43.8%) | 428 (63.7%) | +134 crops |
| No text returned | 378 (56.2%) | 244 (36.3%) | -134 crops |
| Median per-annotation | 123.56 ms | **97.62 ms** | **-26ms (faster)** |

**Run command:**
```bash
python tests/benchmark_ocr.py --exp-id EXP-03-04-COMBO --preprocess pad \
  --det-db-thresh 0.2 --det-db-box-thresh 0.3 --det-db-unclip-ratio 2.0
```

**What changed:**
- `--preprocess pad` — crops below 64px on either dimension are padded with neutral grey (PIL `ImageOps.expand`)
- `det_db_thresh`: 0.3 → 0.2 — lower pixel-level binarization threshold
- `det_db_box_thresh`: 0.5 → 0.3 — lower box confidence threshold (more detections)
- `det_db_unclip_ratio`: 1.5 → 2.0 — wider box expansion (catches more context)

**Status:** Validated in benchmark. Ready to promote to `app/ocr_processor.py` when approved.

---

## 1. Project Overview

The service acts as a BE microservice receiving **pre-cropped images** from an upstream YOLO detector and returning OCR text + confidence. All experiments operate on these crops — not full frames.

**Dataset:** 397 aerial drone frames (DJI, 4000×3000 JPEG), 672 COCO bounding box annotations with ground truth text.

**Stack:** PaddleOCR 2.7.3 (PP-OCRv4), `paddlepaddle==2.6.2`, CPU-only, FastAPI.

**Related docs:**
- [ocr-performance-experiment-01.md](ocr-performance-experiment-01.md) — baseline run methodology and results
- [pddleocr-experimentation-01-report](pddleocr-experimentation-01-report) — PaddleOCR version hell post-mortem

---

## 2. Constraints & Restrictions

| Constraint | Rationale |
|---|---|
| No GPU / no AI upscaling | Cloud Run deployment, cost-sensitive |
| No fixed-dimension resize before OCR | PaddleOCR does this internally — duplicating it causes conflicts |
| No mean/std normalization before OCR | PaddleOCR applies its own normalization internally |
| No hard binarization (B&W) | PaddleOCR expects colour/grayscale input |
| Speed: median must stay under ~200ms | Production SLA; degrade only with proportional accuracy gain |
| Pinned versions: paddleocr==2.7.3, paddlepaddle==2.6.2 | Version changes require full dependency overhaul — high risk per post-mortem |

**Safe preprocessing** (won't conflict with PaddleOCR internals):
- CLAHE contrast enhancement
- Spatial rotation / transpose
- Border padding
- Grayscale conversion (PaddleOCR handles both RGB and grayscale)

---

## 3. Canonical Baseline

**Run ID:** `BASE-01` → file `benchmark_20260415_140905`
**Date:** 2026-04-15 (high-performance power settings, Run 02)

### OCR Configuration (all PaddleOCR defaults)

| Parameter | Value |
|---|---|
| `use_angle_cls` | True |
| `lang` | en |
| `use_gpu` | False |
| `show_log` | False |
| `det_db_thresh` | 0.3 (default) |
| `det_db_box_thresh` | 0.5 (default) |
| `det_db_unclip_ratio` | 1.5 (default) |
| `det_limit_side_len` | 960 (default) |
| `drop_score` | 0.5 (default) |
| `cls_thresh` | 0.9 (default) |
| `rec_image_shape` | [3, 48, 320] (default) |
| Preprocessing | None |

### Library Versions

| Library | Version |
|---|---|
| paddleocr | 2.7.3 |
| paddlepaddle | 2.6.2 |
| numpy | 1.26.4 |
| pillow | 10.2.0 |
| opencv-python-headless | ≥4.8.0 |

### Results

| Metric | Value |
|---|---|
| Annotations processed | 672 / 672 |
| **Exact match accuracy** | **22.3% (150/672)** |
| Wrong text returned | 21.4% (144/672) |
| No text returned | 56.2% (378/672) |
| Precision when text returned | 51.0% (150/294) |
| Avg per annotation | 144.88 ms |
| **Median per annotation** | **123.56 ms** |
| Std dev | 118.49 ms |
| p90 | 308.58 ms |
| p95 | 360.68 ms |
| p99 | 467.87 ms |
| Throughput | 5.34 ann/s |
| Warmup (init) | 2.03 s |

### Key Failure Breakdown

| Failure type | Count | Notes |
|---|---|---|
| Portrait crops (ratio < 0.5) | ~147 | Near-zero accuracy; `use_angle_cls` doesn't help 90° rotation |
| Fast fails < 50ms | ~154 | Detection stage found no candidate regions (too small / low contrast) |
| Slow fails ≥ 50ms | ~224 | Detection ran but recognition returned nothing |
| Character confusion (0↔O, 1↔I, 2↔N, H↔O) | ~38 | Systematic substitution errors in wrong-text cases |

**Confidence reliability:** Below 0.9 confidence, success rate is 6-12%. At 1.0 confidence, still 16% wrong text.

**Aspect ratio sweet spot for success:** ~2.48:1 landscape (avg 168×71 px). Failures skew portrait or very wide.

---

## 4. Experiment Log

All deltas are vs. BASE-01.

| Exp ID | Hypothesis | Status | Accuracy | Δ Accuracy | Median ms | Δ Speed | Date | Notes |
|---|---|---|---|---|---|---|---|---|
| BASE-01 | Canonical baseline | COMPLETE | 22.3% (150/672) | — | 123.56 ms | — | 2026-04-15 | All PaddleOCR defaults, no preprocessing |
| EXP-01 | Auto-rotate portrait crops | REJECTED | 22.3% (150/672) | 0pp | 90.04 ms | -33ms | 2026-04-15 | Portrait accuracy unchanged (0.6%). Root cause: crops are extreme strips (30-80px wide, 170-360px tall). After rotation, height is only 31-58px — too short for character recognition. See EXP-01B. |
| EXP-01B | Rotate + upscale portrait crops | REJECTED | 22.2% (149/672) | -0.1pp | 93.22 ms | -30ms | 2026-04-15 | Portrait accuracy 0.0% — worse than baseline. Upscaling introduced Lanczos blur that hurt previously-correct crops (e.g. Ann 562: 352024→30N024). Root cause: portrait crops are edge-on views of side-wall text from overhead drone. No preprocessing can recover missing pixel information. Structural limitation. |
| EXP-02 | CLAHE contrast enhancement | REJECTED | 12.4% (83/672) | -9.9pp | 97.96 ms | -26ms | 2026-04-15 | Significant regression. Text returns dropped from 294→183. Wide bucket 28.4%→14.0%. CLAHE converts to grayscale, losing colour discriminability PaddleOCR's DB detector relies on. Noise amplification on small tiles harmful. Do not use. |
| EXP-03 | Pad small crops to min 64px | **ACCEPTED** | **30.4% (204/672)** | **+8.1pp** | 88.88 ms | -35ms | 2026-04-15 | Major win. Wide bucket 28.4%→40.3%, small bucket 9.0%→24.4%. Text returns 294→400. Speed improved. |
| EXP-04 | Lower detection thresholds (db_thresh=0.2, box_thresh=0.3, unclip=2.0) | ACCEPTED | 24.6% (165/672) | +2.3pp | 123.70 ms | +0ms | 2026-04-15 | Modest standalone win. Compounds well with EXP-03. |
| EXP-03+04 COMBO | Pad + lower detection thresholds | **ACCEPTED** | **34.7% (233/672)** | **+12.4pp** | 97.62 ms | -26ms | 2026-04-15 | Best config so far. Wide bucket 46.9%, precision 54.4% (up from 51.0%). 428 text returns vs 294. Faster than baseline. |
| EXP-03+04+06 COMBO | Pad + thresholds + character substitution | ACCEPTED | 34.8% (234/672) | +12.5pp | 85.63 ms | -38ms | 2026-04-15 | Marginal gain from post-processing (+1 annotation). Conservative rules safe to include. |
| EXP-05 | `det_limit_side_len` at 320, 480, 640 | NEUTRAL | 22.3% (150/672) | 0pp | ~100ms | -23ms | 2026-04-16 | All three values identical to baseline. Crops are smaller than all limits — no resize occurs. Keep default 960. |
| EXP-07 | Two-pass OCR for portrait crops | REJECTED | 22.3% (150/672) | 0pp | 144.40 ms | +21ms | 2026-04-16 | No accuracy gain. 3 passes per portrait crop drives p90 to 547ms, max to 1102ms. Precision dropped (48.9%). Picks most-confidently-wrong answer. Portrait crops are structurally unreadable — confirmed definitively. |

---

## 5. Experiment Queue

Ordered by expected impact vs effort. Highest impact, lowest effort first.

---

### EXP-01 — Auto-Rotate Portrait Crops

**Hypothesis:** Portrait-oriented crops (w < h) are vertical trailer ID labels. PaddleOCR's `use_angle_cls=True` corrects 0°/180° but not 90° rotations. Rotating portrait crops 90° CW before OCR should recover most of the ~147 near-zero-accuracy portrait failures.

**Preprocessing:** `rotate_portrait` — if `image.width < image.height`, apply `image.transpose(Image.Transpose.ROTATE_270)` (lossless, < 1ms)

**PaddleOCR config:** Unchanged (BASE-01 defaults)

**Expected impact:**
- Accuracy: +10–15pp (recovering even 40% of 147 portrait crops ≈ +59 correct → ~32% total)
- Speed: Negligible (< 1ms PIL transpose per portrait crop)

**Risk:** None — spatial transform before PaddleOCR input. 0.6% current accuracy on portraits means almost nothing to lose.

**Success criteria:** Overall accuracy ≥ 28%; portrait-subset accuracy improvement visible; landscape subset unaffected.

---

### EXP-01B — Rotate + Upscale Portrait Crops

**Hypothesis:** EXP-01 showed that rotating portrait crops to landscape doesn't help because after rotation the image height is only 31-58px — too short for PaddleOCR's recognition stage to discriminate characters. A simple bilinear upscale (PIL `resize`, not AI upscaling) to a minimum height after rotation will give the model enough pixels.

**Note on "no AI upscaling" constraint:** The constraint refers to GPU-intensive AI super-resolution (ESRGAN etc.). PIL bilinear/bicubic interpolation is CPU-trivial (<1ms), uses no model, no GPU. It is safe and appropriate here.

**Preprocessing:** `rotate_and_scale` — after detecting portrait crop and rotating 90° CW:
- If resulting height < `min_height` (target: 80px), scale up proportionally using `Image.LANCZOS`
- e.g. 326×50 rotated → scale 1.6× → 522×80

**PaddleOCR config:** Unchanged (BASE-01 defaults)

**Expected impact:**
- Accuracy: +5–15pp improvement on portrait subset (156 crops currently at 0.6%)
- Speed: ~1-2ms per portrait crop for PIL resize

**Risk:** Low. PaddleOCR receives a larger image and handles it with its own internal resize. No conflict.

**Success criteria:** Portrait-subset accuracy > 5% (up from 0.6%); overall accuracy > 22.3%.

---

### EXP-02 — CLAHE Contrast Enhancement

**Hypothesis:** Low-contrast crops (faded paint, shadow, aerial haze) cause PaddleOCR's detection stage to find no text regions. CLAHE applied to the grayscale channel will improve local contrast without conflicting with PaddleOCR's internal normalization.

**Preprocessing:** `clahe` — convert to grayscale, apply `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`, convert back to 3-channel RGB

**PaddleOCR config:** Unchanged (BASE-01 defaults)

**Expected impact:**
- Accuracy: +3–8pp (recovering some of the ~222 slow-fail crops)
- Speed: +1–3ms per crop

**Risk:** Low. Monitor for false positives: CLAHE can make texture/noise look like text.

**Success criteria:** "OCR returned text" count ≥ 320 (up from 294); precision stays ≥ 45%.

---

### EXP-03 — Pad Small Crops

**Hypothesis:** Very small crops (< 64px on either dimension) fast-fail because PaddleOCR's internal resize leaves text features too small to detect. Adding a neutral gray border gives more spatial context.

**Preprocessing:** `pad` — if `w < 64` or `h < 64`, expand with `ImageOps.expand(border=(pad_w, pad_h), fill=(128,128,128))`

**PaddleOCR config:** Unchanged (BASE-01 defaults)

**Expected impact:**
- Accuracy: +1–3pp (targeting ~78 tiny crops)
- Speed: Negligible

**Risk:** None — equivalent to a larger image.

**Success criteria:** Exact match improvement specifically for crops where `area < 5000px`.

---

### EXP-04 — Lower Detection Thresholds

**Hypothesis:** PaddleOCR's default `det_db_box_thresh=0.5` is calibrated for documents/scene text with high contrast. Aerial drone imagery has lower contrast. Lowering thresholds will detect more text regions.

**Preprocessing:** None

**PaddleOCR config changes:**
- `det_db_box_thresh`: 0.5 → 0.3
- `det_db_thresh`: 0.3 → 0.2
- `det_db_unclip_ratio`: 1.5 → 2.0

**Expected impact:**
- Accuracy: +3–7pp
- Speed: +10–30ms median (more candidate regions processed through recognition)

**Risk:** None (official parameters). If precision drops below 40%, thresholds are too loose — reject.

**Success criteria:** "OCR returned text" count ≥ 350; precision ≥ 40%; median ≤ 180ms.

---

### EXP-05 — `det_limit_side_len` Tuning for Crop Input

**Hypothesis:** Default `det_limit_side_len=960` was designed for full-size images. On small crops (median ~139×68px), it upscales ~5.5× for detection — potentially introducing noise at unexpected scale. Testing 320–480 may be better for crop-sized input and also faster.

**Preprocessing:** None

**PaddleOCR config changes:** Test `det_limit_side_len` at 320, 480, 640 in separate sub-runs.

**Expected impact:**
- Accuracy: Uncertain (±2–5pp, empirical)
- Speed: −10–20ms median (less computation)

**Risk:** None (official parameter). May help or hurt — empirical question.

**Success criteria:** Any value matching or beating BASE-01 accuracy while being faster is a win.

---

### EXP-06 — Post-Processing Character Substitution

**Hypothesis:** Known confusion pairs (0↔O, 1↔I, 2↔N, S↔5) follow context: in numeric-majority strings, `O→0, I→1, S→5`; in alpha-majority contexts, reverse. Trailer IDs follow patterns like `[A-Z]{4}\s?\d{6}`.

**Preprocessing:** None

**PaddleOCR config:** Unchanged (BASE-01 defaults)

**Post-processing:** Add substitution rules after OCR returns. Implemented in `tests/preprocessing.py` (not in production code until proven).

**Expected impact:**
- Accuracy: +2–4pp (targeting ~38 character-confusion wrong-text cases)
- Speed: Negligible (regex, < 1ms)

**Risk:** Zero — pure post-processing. Must be conservative to avoid breaking correct results.

**Success criteria:** Wrong-text count ≤ 120 (down from 144); no correct results broken.

---

### EXP-07 — Two-Pass OCR for Portrait Crops

**Hypothesis:** EXP-01's single-rotation approach may harm portrait crops that read correctly top-to-bottom. Two-pass OCR (original + 90° CW, take best confidence) is more robust.

**Prerequisite:** Run after EXP-01 — only worthwhile if EXP-01 leaves significant accuracy on the table for portraits.

**Preprocessing:** `two_pass_portrait` — run OCR on original; if portrait and confidence < 0.7, run again at 90° CW and 90° CCW, return highest confidence result.

**Expected impact:**
- Accuracy: +1–3pp beyond EXP-01
- Speed: Up to 3× slower on portrait crops only (~147 crops × 2 extra passes)

**Risk:** None.

**Success criteria:** Marginal gain ≥ 2pp over EXP-01 to justify the speed cost.

---

### EXP-COMBO-01 — Stack All Winners

After EXP-01 through EXP-06, run a combined experiment stacking all passing experiments to measure compound effects. Compare vs. BASE-01 and vs. each individual experiment.

---

## 6. Methodology

- **One variable per experiment** (or a tightly coupled group, explicitly noted)
- **Benchmark-first isolation:** All experiments are implemented in `tests/benchmark_ocr.py` + `tests/preprocessing.py`. Production `app/ocr_processor.py` is only modified when an experiment is **proven and explicitly promoted**
- **Compare to BASE-01 always** — not to the previous experiment, to avoid drift
- **Same conditions:** High-performance power settings, same machine, same dataset (all 672 annotations)
- **Run command:**
  ```bash
  python tests/benchmark_ocr.py --exp-id EXP-01 --preprocess rotate
  ```
- **Output files:** `tests/results/benchmark_EXP01_YYYYMMDD_HHMMSS.json/.log` — each JSON is fully self-documenting (OCR config, library versions, preprocessing applied, per-annotation results)
- **Success criteria (minimum bar):** Exact match accuracy improvement without precision dropping below 40%

---

## 7. Tried & Rejected

_Nothing yet. Entries added here as experiments fail, with evidence linking to the result JSON._

| Exp ID | What was tried | Why rejected | Evidence |
|---|---|---|---|
| EXP-01 | Auto-rotate portrait crops 90° CW before OCR | Portrait accuracy unchanged (0.6%). Root cause: portrait crops are extreme thin strips (30-80px wide, 170-360px tall). After rotating to landscape, height is only 31-58px — too short for PaddleOCR recognition to discriminate characters. Rotation alone is not the fix. | `benchmark_EXP-01_20260415_155642.json` |
| EXP-01B | Rotate portrait crops 90° CW + upscale to min 80px height | Portrait accuracy 0.0% — worse than baseline. Upscaling introduced blur that broke previously-correct crops. Fundamental diagnosis: portrait crops are **edge-on views of side-wall trailer text** from overhead drone — only the top edge of characters visible. No preprocessing can recover absent pixel data. Consider excluding portrait crops from accuracy targets. | `benchmark_EXP-01B_20260415_161242.json` |
| EXP-02 | CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=8×8) | Major regression: 12.4% vs 22.3% baseline. Conversion to grayscale stripped colour discriminability that PaddleOCR DB detector relies on. Noise amplification on tiny tiles (21×9px for 168×71 crops) broke detection. Future: try contrast enhancement in colour space only (e.g. CLAHE on L channel of LAB, keep colour). | `benchmark_EXP-02_20260415_164715.json` |
| EXP-05 | `det_limit_side_len` at 320, 480, 640 | All three identical to baseline. These crops are already smaller than all three limits, so PaddleOCR's detection stage doesn't resize them regardless of the setting. Not a useful knob for this input type. | `benchmark_EXP-05-320/480/640_*.json` |
| EXP-07 | Two-pass OCR for portrait crops (original + 90cw + 90ccw, best confidence) | Same accuracy as baseline (22.3%). Three passes inflate portrait crop time to 500-1100ms. Precision dropped to 48.9% — selecting the most confidently wrong answer from three bad options. Portrait crops are **definitively unreadable** from overhead drone at this angle. Do not revisit with preprocessing. | `benchmark_EXP-07_20260416_013420.json` |

---

## 8. PP-OCRv5 Migration Assessment

**Decision: Defer — do not attempt PP-OCRv5 migration until EXP-06 is complete.**

Reasons:
1. The [PaddleOCR post-mortem](pddleocr-experimentation-01-report) documents that the current working stack required 5+ hacks just to run (LD_PRELOAD, argv isolation, OpenCV cleanup, Debian pinning, model baking). PP-OCRv5 requires PaddleOCR 2.9+ and PaddlePaddle 3.x — a full dependency overhaul with the same potential for version hell.
2. The seven experiments above target the **dominant failure modes** (aspect ratio, contrast, thresholds) — input/config issues, not model architecture. PP-OCRv5 would not fix these.
3. The primary bottleneck is the detection stage failing on 56.2% of crops — likely caused by unusual input characteristics (small pre-cropped aerial imagery), not model generation.
4. If EXP-01–EXP-06 plateau below target accuracy, revisit with a full cost/benefit assessment.

**Alternative if migration is required:** Google Vision API or AWS Textract for serverless; long-running VM with Conda environment for self-hosted PaddleOCR.
