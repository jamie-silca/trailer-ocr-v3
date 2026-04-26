# OCR Performance Experiment 19 — Tesseract PSM 5 for Portrait

**Status:** REJECTED (2026-04-25) — Tesseract 5.5.0 (UB Mannheim build, PSM 6, OEM 1, A-Z 0-9 whitelist) on all 119 portraits in `tests/dataset/20260423` produced **4/119 EXACT and 10/119 within edit-distance 2**, with only **3 unique correct hits** that PaddleOCR misses. Marginal value, fails to move the load-bearing JBHZ alpha-prefix vertical sub-bucket (0 hits across both engines).

## Spike outcome

Two pre-flight runs were performed before any benchmark wiring (per memory rule: re-run a load-bearing spot-check before building on it).

**Run 1 — PSM 5 on 5 crops** ([tests/tesseract_spike2.py](../tests/tesseract_spike2.py) for the cross-tab):

| ann | gt           | psm5_otsu          | psm5_inv           | psm6_otsu        | rot90_psm6     |
|-----|--------------|--------------------|--------------------|------------------|----------------|
|  17 | JBHU235644   | `7EDNMONOTT`       | `VEDNMNORT`        | **`JBHU235644`** | `DANUIOWAYCIOKW` |
|  73 | JBHZ676208   | _(empty)_          | _(empty)_          | `HZ6708`         | `MOWAAWOIDE`   |
| 165 | JBHZ672066   | _(empty)_          | _(empty)_          | _(empty)_        | `GRE`          |
| 255 | JBHZ660789   | _(empty)_          | _(empty)_          | _(empty)_        | _(empty)_      |
| 356 | JBHZ672637   | _(empty)_          | _(empty)_          | `Y67`            | `ALSATONIS`    |

**PSM 5 (the originally hypothesised mode) was strictly empty/garbage on 5/5.** The mode advertised as "single uniform block of vertically aligned text" does not engage the LSTM in any useful way on these tall thin crops. PSM 6 happened to read `ann 17` exactly — interesting but not yet evidence of a working pipeline.

**Run 2 — PSM 6 on the full 119 portrait crops** ([tests/tesseract_spike.py](../tests/tesseract_spike.py), output [tests/results/tesseract_spike_psm6_20260423.json](../tests/results/tesseract_spike_psm6_20260423.json)):

| GT format     |  n  | EXACT | NEAR (≤2) | empty |
|---------------|----:|------:|----------:|------:|
| ALPHA4_DIGIT6 |  95 |   2   |    4      |   26  |
| NUMERIC       |  17 |   1   |    6      |    5  |
| OTHER         |   7 |   1   |    0      |    1  |
| **TOTAL**     | 119 | **4** |  **10**   |   32  |

Median latency 140 ms/crop on CPU.

### Cross-referenced against PaddleOCR EXP-23 results

The relevant question isn't "how many can Tesseract read" but "how many can Tesseract read **that PaddleOCR can't**":

| ann_id | gt           | paddle_v4   | tess_psm6   | tess+rescore (edit≤2) | unique to Tess |
|-------:|--------------|-------------|-------------|------------------------|:--------------:|
|     17 | JBHU235644   | _(empty)_   | JBHU235644  | JBHU235644             |      ✓         |
|    113 | JBHU280944   | _(empty)_   | JBHU280944  | JBHU280944             |      ✓         |
|    329 | 7322039P     | 73220392    | 7322039P    | 7322039P               |      ✓         |
|    297 | JBHU235644   | _(empty)_   | JBHJ232644  | JBHJ232644 (edit 2)    | ✓ if rescored  |
|    362 | JBHU284775   | _(empty)_   | JBHU2847    | JBHU2847 (edit 2)      | ✓ if rescored  |
|      1 | 352827       | 352827      | 35282T      | 352827                 |      —         |
|      2 | 353267       | 353267      | 3532G7      | 353267                 |      —         |
|     61 | 352337       | 352337      | 352337      | 352337                 |      —         |
|    138 | 352337       | 352337      | 3S233       | (no fix)               |      —         |
|    288 | JBHU286632   | 83280032    | JB286632    | JB286632 (edit 2)      |      —         |
|    313 | JBHU257263   | 83257263    | JBHU22263   | JBHU22263 (edit 2)     |      —         |

**Unique wins: 3 strict, 5 if EXP-15 `format_rescore` upgrades edit-2 cases.** All unique alpha4 wins are **JBHU prefix** — no JBHZ portrait crop reads correctly under either engine. Numeric portraits gain nothing (PaddleOCR already handles them).

### Why this isn't enough

The load-bearing question for Round 3 is the JBHZ alpha-prefix vertical sub-bucket (~89 of 95 alpha4 portraits, the dominant operational case). Tesseract is **0/89-ish** there — same wall as PaddleOCR. The 3-5 unique wins come from JBHU + a malformed-GT "other" case, not from the bottleneck.

Overall-benchmark math: +3-5 correct on 419 lifts overall from 62.5% → 63.2-63.7% (+0.5-1.2pp). Cost: pytesseract dep, ~50MB Tesseract binary in the deployment image, ~140 ms added median latency on portrait crops, and the `tesseract_portrait.py` + benchmark wiring (~half a day). Marginal ROI, and the win is concentrated in a sub-sub-bucket (JBHU portraits) that's not the bottleneck.

### Why PSM 5 doesn't fire

PSM 5 ("single uniform block of vertically aligned text") is documented for CJK vertical writing where each line of glyphs is read top-to-bottom. On Latin-alphabet stacked-vertical crops, Tesseract's layout analyser apparently never assigns the input as a valid PSM-5 region — it returns empty rather than a parse. PSM 6 (single uniform block, default Latin layout) at least engages the LSTM, but the recogniser still produces garbage on most crops because the binarised tall-thin input falls outside its training distribution.

## Why PSM 5 doesn't fire

PSM 5 is documented as "Assume a single uniform block of vertically aligned text". The mode exists for CJK vertical writing where each line of glyphs is read top-to-bottom. On a Latin-alphabet stacked-vertical crop, Tesseract's layout analyser apparently never assigns the input as a valid PSM-5 region — it returns nothing rather than a parse. PSM 6 (single uniform block, default Latin layout) will at least try, but the LSTM recogniser then produces garbage because each glyph has been pre-binarised on a tall thin crop with significant inter-character noise that Tesseract's text-line model isn't trained to handle.

Either way, the load-bearing hypothesis ("a layout-aware engine with weaker character recognition beats PP-OCRv4 on stacked vertical text") is empirically false at production scale on this dataset. PP-OCRv4 currently sits at 0/95 on this sub-bucket; Tesseract sits at 1/20 ≈ 5/95 projected.

## Decision

- No `tesseract_portrait.py` or `--portrait-strategy tesseract` wiring is added.
- Tesseract 5.5 is **not** added as a system/Docker dependency.
- The spike scripts are retained for reproducibility:
  - [tests/tesseract_spike.py](../tests/tesseract_spike.py) — full 119-crop PSM 6 sweep.
  - [tests/tesseract_spike2.py](../tests/tesseract_spike2.py) — 5-crop cross-PSM matrix.
- Round 3 candidate space narrows to: stronger VLMs (Claude Haiku 4.5 / Gemini 2.5 Pro / Qwen2.5-VL — Gemini Flash already failed in EXP-18) or a custom char detector + tiny CNN classifier (EXP-21, blocked on hand-labelling ~50 portrait crops). Both should target **JBHZ alpha-prefix vertical** specifically.

**Cost of the experiment:** ~45 minutes (Tesseract install + 3 spike scripts + 119-crop sweep + cross-reference + write-up). Cost of the avoided full wiring: ~half a day. Memory rule "verify priors before building" caught a hypothesis that looked productive on first read but didn't survive widening the sample.

---

## Original plan (retained for archive)

**Status:** PLANNED. Portrait-bucket-only. Cheap alternative-engine probe.

## 1. Goal

Route portrait crops to **Tesseract with page-segmentation mode 5**
("single uniform block of vertically aligned text"). This is the only widely
available OCR engine with a first-class mode for exactly this layout. Measure
whether a layout-aware engine with weaker character recognition beats PP-OCRv4
(strong recognition, wrong layout assumption) on stacked vertical text.

Target: **portrait bucket ≥ 20/156**. If Tesseract dominates here, it becomes
a low-cost, on-prem alternative to EXP-18's VLM route.

## 2. Hypothesis

Tesseract's LSTM recogniser is generally weaker than PP-OCRv4 on character
accuracy, but PSM 5 gives it the correct structural prior for stacked vertical
text. PP-OCRv4 has no equivalent mode — it's horizontal-sequence by design.
On a layout-first problem, layout-correct beats recogniser-strong.

## 3. Methodology

### Pipeline

- Aspect < 0.5 → Tesseract path.
- Aspect ≥ 0.5 → existing EXP-09+15 path (unchanged).

### Tesseract configuration

- **Binary:** Tesseract 5.x (pip `pytesseract` as wrapper).
- **PSM:** 5 (vertical block). Also A/B against PSM 6 (uniform block) as control.
- **OEM:** 1 (LSTM only).
- **Language:** `eng`.
- **Character whitelist:** `tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`.
- **Preprocessing:** upscale to min 256-px short side, binarise (Otsu), add
  ~20 px white border (Tesseract likes context padding).

### Validation gate

Output stripped, upper-cased, whitespace removed. Accept only if matches the
format whitelist (`^JBHZ\d{6}$ | ^JBHU\d{6}$ | ^R\d{5}$`) after running through
EXP-15's `format_rescore`. Otherwise fall back to existing portrait path.

## 4. Critical Files

- [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — add
  `--portrait-strategy tesseract` and `--tesseract-psm {5,6}` flags.
- `tests/tesseract_portrait.py` (new) — thin `read_portrait_tesseract(image, psm) -> (text, confidence)` wrapper.
- System dependency: Tesseract 5.x binary. Document install steps in the post-mortem.

## 5. Implementation Steps

1. Install Tesseract 5.x locally; verify `tesseract --version`.
2. Implement `tests/tesseract_portrait.py` with preprocessing (upscale + Otsu
   + border) and PSM 5 config.
3. Wire `--portrait-strategy tesseract` into benchmark loop.
4. Run portrait-only subset with PSM 5 and PSM 6 to compare.
5. Run full 672 benchmark with winning config.
6. Post-mortem: portrait delta, characteristic failure modes (Tesseract tends
   to split/merge adjacent glyphs differently from PP-OCR), latency.

## 6. Verification

- **Primary:** portrait bucket ≥ 20/156.
- **Guardrail 1:** non-portrait buckets unchanged (Tesseract never runs on them).
- **Guardrail 2:** precision unchanged — format-gate + rescore catches
  Tesseract's character-level noise.
- **Latency:** Tesseract on CPU is typically 50–150 ms per small image;
  budget < 200 ms median overhead on portrait crops.

## 7. Risks

- **Character-level accuracy floor:** Tesseract is often 5–15pp behind PP-OCR
  on per-char accuracy for clean printed text. Format gate + EXP-15 rescore
  are the defence; if they can't lift Tesseract's output to the strict format,
  this experiment will underperform EXP-16.
- **New binary dependency:** adds a system-level dep to the Docker image.
  Acceptable for a portrait-only route, but document the cost.
- **PSM 5 is sensitive to border padding and image scale.** Expect to spend
  some time tuning the preprocess before the headline run.

## 8. Out of Scope

- Training a Tesseract model (requires ground-truth dataset and substantial effort).
- Using Tesseract on non-portrait crops (PP-OCR already dominates there).
- Comparing every PSM — 5 vs 6 is sufficient to characterise.
