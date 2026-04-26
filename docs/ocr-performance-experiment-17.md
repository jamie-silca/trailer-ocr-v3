# OCR Performance Experiment 17 — Character-Level Detection + Unroll

**Status:** REJECTED (2026-04-25) — premise invalidated at execution time. The revised plan below relied on EXP-22's claim that PP-OCRv5's detector returns per-character boxes on stacked-vertical portrait crops. Verification at the start of execution showed:

- Full 156-crop precompute: **131/156 crops have zero boxes**, max 2 boxes on any crop, never the per-char count required.
- Targeted re-run on `ann 14` (the EXP-22-cited example, GT `JBHZ672061`): **0 boxes** under EXP-09 tuning, 0 under defaults, 0 under extreme-permissive (`thresh=0.1`, `box_thresh=0.15`), at both 58×326 original and 232×1304 4×-upscaled — six combinations, zero boxes in all six.
- Outputs: [tests/results/exp22_v5_detection.json](../tests/results/exp22_v5_detection.json), [tests/results/verify_v5_ann14.json](../tests/results/verify_v5_ann14.json).

EXP-22's "hidden finding" §10 has been retracted and corrected; see [ocr-performance-experiment-22.md §10](ocr-performance-experiment-22.md). With no per-char detection available from v5, the stitch and per-char-rec strategies both die at the precompute step. The original v4-retune fallback hypothesis (extreme DB thresholds, `det_limit_side_len ≥ 1280`) was never benchmarked in this run; if revived later, document as a new experiment.

**Round 2 closes at EXP-09's 38.2 % exact-match / 57.6 % precision** with the portrait bucket marked as a known limitation pending Round 3 work (candidate: train a dedicated per-character detector on ~50 hand-labelled portrait crops).

---

## Original plan (retained for archive)

## 1. Goal

Replace EXP-16's uniform vertical slicing with **per-character DB detection** so
that composite horizontal strips are built from true glyph boundaries rather
than height/N heuristics. Target the failure modes EXP-13 exposed (variable
char widths — `1` vs `0` — scratches breaking uniform spacing).

Target: on top of EXP-16's portrait gain, add **+5–10 additional portrait
correct** (i.e. portrait 35–45/156). Overall **+1–2pp** over EXP-16.

## 2. Hypothesis (revised after EXP-22)

PP-OCRv5's detector returns per-character boxes natively on stacked-vertical
portrait crops (verified in EXP-22 §10: 6 per-char boxes on `ann 14`, GT
`JBHZ672061`, with EXP-09 tuning unchanged — vs 0 boxes from v4 on the same
input). Sort those boxes by Y-centre, crop each with a small margin, compose
into a horizontal strip, and run **v4's recogniser** on the strip. v4's rec
handles horizontal text well (51% on `wide` bucket); the layout problem moves
from "rec can't read vertical" to "stitch produces a horizontal strip the rec
already handles".

Alternative path if the stitch strip fails the format gate: **read each
per-char box independently** through v4 rec at a small scale-up, concatenate
top-to-bottom. Slower per-crop but doesn't depend on stitch quality.

The original v4-retune hypothesis is preserved as a fallback if the v5-venv
crossing turns out to be operationally clumsy.

## 3. Methodology (revised after EXP-22)

### Pipeline

1. Run **PP-OCRv5 detection only** on the portrait crop (via `.venv-paddle-v5/`).
   Use EXP-09-equivalent tuning (`text_det_thresh=0.2`,
   `text_det_box_thresh=0.3`, `text_det_unclip_ratio=2.0`). No upscaling
   needed — EXP-22 confirmed v5 returns per-char boxes on the original 47–58 px
   wide portrait crops.
2. Extract `dt_polys` from the v5 result. Each is a 4-point polygon around
   one character.
3. Filter boxes: keep those whose centre lies inside the main column (reject
   stray detections on adjacent trailers, edges, or yard markings).
4. Sort remaining boxes by Y-centre. This gives char order top-to-bottom.
5. **Strategy A — stitch:** for each box, crop with a small margin, normalise
   to H=48, paste left-to-right into a synthetic horizontal strip. Feed strip
   to v4's recogniser.
6. **Strategy B — per-char rec:** for each box, crop independently and feed
   to v4 rec. Concatenate non-empty results in Y-order.
7. Validate against the EXP-15 format whitelist. If both strategies fail
   the gate, fall through to v4's standard portrait path (currently 2/156
   correct, so no regression risk).

### Cross-venv operational shape

EXP-22 set up `.venv-paddle-v5/` as an isolated venv. The benchmark cannot
run both v4 (in the main env) and v5 (in `.venv-paddle-v5/`) inside the same
Python process — they pin incompatible `paddlepaddle` majors. Two options:

- **A (preferred):** Pre-compute v5 detection results once for all 156
  portrait crops via a one-shot script in `.venv-paddle-v5/`, dump
  `{annotation_id: dt_polys}` to JSON in `tests/results/exp22_v5_detection.json`,
  then run the main benchmark in the v4 env which loads that JSON to get
  per-char boxes. Adds zero per-crop latency to v4, isolates the v5 dependency
  to a precomputation step. Best for repeated benchmark iteration.
- **B (alternative):** Run the entire EXP-17 sanity inside `.venv-paddle-v5/`
  using both v5 detection and v5 recognition (skip v4 rec). Cleaner code but
  carries v5's 15× latency penalty into the result.

Plan defaults to **A**.

### No detector parameter sweep

EXP-22 already showed v5 produces good per-char boxes with EXP-09's tuning.
No sweep needed. If portrait gain is < 5/156 on the first run, *then* sweep.

## 4. Critical Files

- `tests/precompute_v5_detection.py` (new) — one-shot script run from `.venv-paddle-v5/`. Iterates over portrait-bucket crops, extracts v5 `dt_polys`, dumps `tests/results/exp22_v5_detection.json` keyed by annotation id.
- [tests/preprocessing.py](../tests/preprocessing.py) — extend `decode_stacked_vertical` (currently the EXP-13 stitch helper) to accept a `box_source="v5_precomputed"` mode that loads boxes from the JSON instead of running v4 detection.
- [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — add `--portrait-strategy v5_per_char` option. Reuses the existing `--stacked-vertical stitch` glue, just changes where boxes come from.
- [app/ocr_processor.py](../app/ocr_processor.py) — no change.

## 5. Implementation Steps

1. Write `tests/precompute_v5_detection.py`. Loops over the 156 portrait-bucket
   annotations, instantiates `PaddleV5Processor` (or directly `PaddleOCR`),
   stores `result[0]['dt_polys']` per annotation. ~5 minutes of wall time
   given EXP-22's 1.5 s/crop.
2. Spot-check by overlaying boxes on 5 portrait crops to confirm column-membership
   filter is working. Adjust the membership filter if v5 finds noise boxes
   outside the main column.
3. Implement Strategy A (stitch) first since it reuses EXP-13's
   `_stitch_letters` helper. Run portrait-only via
   `python tests/benchmark_ocr.py --exp-id EXP-17-stitch-v5 --portrait-strategy v5_per_char --only-bucket portrait`.
4. If Strategy A < 5/156: implement Strategy B (per-char rec) as a fallback.
5. Run full 672 with the winning strategy. Compare against EXP-09 via
   `tests/compare_experiments.py`.
6. Post-mortem: portrait delta, where stitched strip succeeded vs failed,
   sample of boxes from `dt_polys` that the column filter rejected (debug aid).

## 6. Verification

- **Primary:** portrait bucket. Target ≥ EXP-16 + 5.
- **Guardrail 1:** non-portrait buckets unchanged.
- **Guardrail 2:** added latency < 80 ms median on portrait crops (extra DB pass
  on an upscaled image — budget it).
- **Guardrail 3:** precision unchanged or improved.

## 7. Risks

- **Cross-venv friction.** Pre-computing v5 detection means the EXP-17 pipeline
  has a one-shot setup step every time the dataset or annotations change.
  Mitigation: precompute once, version-control the JSON output, regenerate
  only on dataset changes. Adds zero latency to the benchmark itself.
- **Box-count mismatch with GT length.** v5 might return 6 boxes for a
  10-char ID (under-detection of dim or partly-occluded glyphs). The
  per-strategy decision rule should reject any reconstruction that doesn't
  match a known format length, then fall through to v4's standard portrait
  path — same precision-protection logic as EXP-15.
- **Stitch quality.** EXP-13 already saw stitched portrait strips fail to
  read despite tight glyph alignment (gibberish from both v4 rec and TrOCR).
  Strategy B (per-char rec) is the contingency.
- **Per-char rec quality unknown.** v4's recogniser was trained on text-strings,
  not single chars. A 1-char input may return empty or garbage. Need spot
  checks on 5 manually cropped chars before committing to Strategy B.
- **Production wiring.** If EXP-17 ships, production needs access to v5
  detection — adds `paddleocr 3.x` + `paddlepaddle 3.x` to the deployed
  image. That's a non-trivial dep upgrade. Out of scope for this experiment;
  the trigger is a positive result, then a separate migration plan.

## 8. Out of Scope

- Training / fine-tuning a dedicated char detector (future work).
- Alternative engines.
- Production wiring.
