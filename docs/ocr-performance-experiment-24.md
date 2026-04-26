# OCR Performance Experiment 24 — Surya OCR for Portrait Crops

**Status:** PLANNED. Portrait-bucket-only. Cheap layout-aware-engine probe.

## Context

EXP-19 ruled out Tesseract PSM 5/6 for stacked vertical (4/119 EXACT, 0/89 on
the load-bearing JBHZ alpha-prefix sub-bucket). EXP-20 ruled out TrOCR as
recogniser (0/156 on portrait, identical layout failure to PaddleOCR).
EXP-18 ruled out Gemini 2.5 Flash (0/15 portrait spike).

The remaining categorical alternatives — before falling back to stronger VLMs
or the hand-labelled char-detector path (EXP-21) — are layout-aware OCR engines
*other* than Tesseract. **Surya** ([github.com/VikParuchuri/surya](https://github.com/VikParuchuri/surya))
is the most credible candidate: Apache 2.0, transformer-based detector with
explicit reading-order and layout-analysis stages, and a recogniser that is
**not** a horizontal-sequence CRNN. Whether its layout analysis fires on
Latin stacked-vertical crops is the open empirical question.

This is the cheapest unrun probe. ~30-60 minutes of spike work to know if
it's worth wiring.

## 1. Goal

Probe whether Surya's layout-aware detection + recognition stack reads
JBHZ/JBHU stacked-vertical portrait crops where PaddleOCR + Tesseract +
TrOCR all sit at 0–5/95.

Target on JBHZ alpha-prefix sub-bucket: **≥ 5/89** justifies wiring;
**≥ 20/89** would reset Round 3 priorities.

## 2. Hypothesis

Surya's detector predicts line-level boxes and a reading order, and the
recogniser sees pre-cropped lines. If Surya's layout stage classifies a
stacked-vertical Latin crop as a single line (in any orientation it can
handle), the recogniser may read it; if the layout stage forces a
horizontal-line assumption like PaddleOCR, this fails identically.

Realistic prior: **~30% it materially helps**. Surya is targeted at
documents/books, not embossed ID plates with stacked Latin glyphs. Worth
the ~30 min spike specifically because it's the last layout-aware engine
on the table before committing to VLM or EXP-21.

## 3. Methodology

### Pipeline

- Aspect < 0.5 → Surya path.
- Aspect ≥ 0.5 → existing EXP-09 path (unchanged).

### Surya configuration

- `pip install surya-ocr` into the project venv. CPU mode for the spike;
  GPU later only if a positive result justifies it.
- Use Surya's combined predictor: `RecognitionPredictor()(images, langs, DetectionPredictor()(images))`.
- `langs=["en"]` only. Strip → upper → drop whitespace → format-gate
  identical to VLM path: `^(JBHZ\d{6}|JBHU\d{6}|R\d{5})$`. Run through
  EXP-15 `format_rescore` before gating.
- No preprocessing in the spike (Surya does its own). If raw fails on all
  5 spike crops, try the EXP-19 preprocessing chain (≥256 px short side
  + Otsu + 20 px white border) as a single sanity variant.

### Validation gate

Same strict gate as EXP-18 / EXP-19 plan. Format hit → return `(text, 1.0)`.
Format miss → fall through to standard PP-OCRv4 pipeline so numeric portraits
and malformed-GT cases aren't penalised.

## 4. Pre-flight verification (LOAD-BEARING — do this first)

Memory rule: verify load-bearing priors before building wiring on them.
Tesseract spike caught a hypothesis that looked productive on first read; do
the same here.

1. `pip install surya-ocr` into the venv. Confirm the import works.
2. Write `tests/surya_spike.py` — a ~50-line throwaway: pick the **same 5
   alpha4+6digit portrait crops** EXP-19 used (ann ids 17, 73, 165, 255, 356
   from 20260423). Run Surya on each raw crop, print `(ann_id, gt, surya_out)`.
3. **Decision rule:**
   - 0/5 within edit-2 of GT → run a single fallback variant (EXP-19's
     preprocess chain). If still 0/5 → **abort EXP-24**, document the
     outcome, do not wire further.
   - ≥1/5 reads correctly or near-correctly → widen to all 119 portrait
     crops via `tests/surya_spike2.py` (the EXP-19 spike1 pattern). If the
     119-crop sweep shows ≥ 5/89 EXACT on JBHZ — or ≥ 5 unique wins vs
     PaddleOCR EXP-23 — proceed to wiring.
4. If the wider sweep is still marginal (< 5 unique wins, < 5 JBHZ EXACT),
   close the experiment same way EXP-19 closed: document, retain spike
   scripts, no wiring.

This pattern saved ~half a day on EXP-19. Same logic applies here.

## 5. Implementation Steps (only if pre-flight passes)

1. `tests/surya_portrait.py` (new) — modelled on
   [tests/vlm_portrait.py](../tests/vlm_portrait.py): singleton
   `SuryaPortraitProcessor`, `process_image(pil) -> (text, conf)`, format
   gate identical to VLM path.
2. [tests/benchmark_ocr.py:103](../tests/benchmark_ocr.py#L103) — extend
   `--portrait-strategy` choices from `["off", "vlm"]` to
   `["off", "vlm", "surya"]`. Reuse the existing portrait-gate at
   lines 477-485.
3. Add a `surya` branch parallel to the `vlm` branch. Falls through to
   standard pipeline on format-gate miss.
4. Run order:
   - Portrait-only on 20260423: `--exp-id EXP-24-surya --dataset 20260423 --only-bucket portrait --portrait-strategy surya`. n=119.
   - Full 419 on 20260423 with the winning config to confirm guardrails.

## 6. Critical Files

- `tests/surya_spike.py` (new, throwaway after spike) — 5-crop pre-flight.
- `tests/surya_spike2.py` (new, throwaway) — 119-crop sweep, only if spike 1 passes.
- `tests/surya_portrait.py` (new) — `SuryaPortraitProcessor`, mirrors
  [tests/vlm_portrait.py](../tests/vlm_portrait.py).
- [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — extend `--portrait-strategy`.
- `docs/ocr-performance-experiment-24.md` — flip from PLANNED to RUN with results.

## 7. Verification

- **Spike pass:** ≥ 1/5 within edit-2 on raw OR preprocessed variant.
- **Widened-spike pass:** ≥ 5 unique wins vs PaddleOCR EXP-23 on 119 crops,
  with ≥ 3 of those on the JBHZ sub-bucket. (JBHU-only wins like Tesseract's
  3 are insufficient — that's the EXP-19 trap.)
- **Primary metric (post-wiring):** JBHZ alpha4+6digit portrait sub-bucket
  EXACT on 20260423 (currently 0/89). Target ≥ 5/89.
- **Guardrail 1:** non-portrait buckets unchanged (Surya never runs there).
- **Guardrail 2:** precision unchanged or improved — strict format gate +
  rescore is the precision defence.
- **Latency:** Surya on CPU is reportedly slow (transformer detector +
  recogniser); budget < 1.5 s median overhead on portraits. If it blows past
  this on CPU, treat as a GPU-only path and re-evaluate deployment cost.

## 8. Risks

- **Layout assumption mismatch.** If Surya's detector forces horizontal-line
  reading order, this fails identically to PaddleOCR — the layout-correctness
  hypothesis is what's being tested.
- **CPU latency.** Likely worse than Tesseract's 140 ms. If GPU is required,
  the deployment cost (CUDA in Docker) is significant — only pay it if the
  accuracy win is large.
- **Dependency conflicts.** Surya pulls torch + transformers; same risk
  surface as TrOCR (EXP-20). Validate against the existing PaddleOCR install
  before wiring.
- **Sample-size noise.** 5-crop spike is intentionally cheap; widen before
  trusting either positive or negative signal (EXP-19 lesson — 1/20 looked
  like noise but 4/119 was the real picture).

## 9. Out of Scope

- Fine-tuning Surya on trailer-ID data (no labelled corpus).
- Surya on non-portrait crops (PaddleOCR adequate; this is portrait-targeted).
- Cross-engine voting / ensembling — only after at least one engine is
  meaningfully strong on JBHZ alpha4+6digit.
- Production wiring. Trigger is a positive result, then a separate rollout
  plan including any Surya runtime deps in the deployment image.

## 10. Cost estimate

- Pre-flight (install + 5-crop spike + decide): ~30-60 min.
- Widened sweep (119 crops + cross-reference): ~1-2 hr if it triggers.
- Full wiring + benchmark + write-up: ~half day if pre-flight passes.

Total worst case ~half-day, total best case (early abort) ~45 min.
Same shape as EXP-19. Same memory rule applies: verify before building.
