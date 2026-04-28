# OCR Performance Experiment 29 — Qwen3-VL Cascade for Horizontal Crops

**Status:** Stage 1 RUN — **BORDERLINE (2026-04-28).** Rescue rate
**7/45 = 15.6%**, inside the 15-30% manual-decision band per the
pre-registered gate. Recommendation: do **not** auto-proceed to Stage 2;
re-check the rescue list manually and decide whether the 7 wins justify
a 419-crop A/B given the 9 wrong-text introductions seen on the same
spike.

## Context

Production sits at 314/419 = 74.9% after EXP-25 wired the Qwen3-VL
portrait cascade. Horizontal buckets remain the next ceiling:

| bucket     | aspect      | n   | paddle EXACT | gap |
|------------|-------------|-----|--------------|-----|
| landscape  | 1.0 ≤ w/h<2 | 35  | 30 (85.7%)   | 5   |
| wide       | 2.0 ≤ w/h<4 | 255 | 222 (87.1%)  | 33  |
| very_wide  | w/h ≥ 4.0   | 10  | 3 (30.0%)    | 7   |
| **total**  |             | 300 | 255 (85.0%)  | **45** |

GOT-OCR-2.0-hf was tested on this bucket (EXP-26) and rejected: 15%
rescue rate, 37% regression rate — *correlated* with paddle, not
complementary. **Qwen3-VL had not been tried on horizontal**; different
model family, different training distribution, GOT's rejection doesn't
transfer automatically.

Stage 1 was a paddle-fail spike to answer the load-bearing question:
**is Qwen3-VL complementary to paddle on horizontal, or correlated?**

## Stage 1 — paddle-fail spike

**Run command:**
```bash
python tests/qwen_horizontal_spike.py \
  --baseline tests/results/benchmark_EXP-25-VERIFY-NO-FMT-CHECK_20260427_133655.json \
  --dataset 20260423
```

**Result file:** [tests/results/qwen_horizontal_spike_20260423.json](../tests/results/qwen_horizontal_spike_20260423.json)

**Wall:** 65.5 s (45 OpenRouter calls, 13.4 K input + 340 output tokens
≈ ~$0.001 estimated).

### Aggregate

| outcome                                                   | count | rate  |
|-----------------------------------------------------------|------:|------:|
| **Rescue** (Qwen format-valid AND matches GT)             |     7 | 15.6% |
| Qwen wrong-text (format-valid but != GT)                  |     9 | 20.0% |
| Qwen no-text (UNKNOWN / format-rejected / empty)          |    29 | 64.4% |
| Total paddle-fails                                        |    45 |       |

### Per-bucket

| bucket    | n  | rescue | rescue rate |
|-----------|---:|-------:|------------:|
| landscape |  5 |      1 | 20.0%       |
| wide      | 33 |      6 | 18.2%       |
| very_wide |  7 |      0 | 0.0%        |

**very_wide is dead.** All 7 paddle-fails in this bucket also fail
Qwen — these are container plates (`CXTU1156285`, `UTCU 4941900`,
`TIFU 1723667`, `RLTU3006285`) that Qwen reads as a slightly different
non-format-valid string and gets format-gated to no-text. Even relaxing
the gate would only swap one wrong text for another.

### The 7 rescues

| ann | bucket    | GT        | paddle              | qwen      |
|----:|-----------|-----------|---------------------|-----------|
|  10 | wide      | `R89649`  | `R89649 MAT`        | `R89649`  |
|  81 | landscape | `70851`   | `0851`              | `70851`   |
| 216 | wide      | `R45777`  | `R457TZ`            | `R45777`  |
| 260 | wide      | `77233`   | (no-text)           | `77233`   |
| 349 | wide      | `R31997`  | `MAL R31997`        | `R31997`  |
| 370 | wide      | `701599`  | (no-text)           | `701599`  |
| 404 | wide      | `701679`  | `101679`            | `701679`  |

Pattern: clean reads where paddle either bolted on extra characters
(`R89649 MAT`, `MAL R31997`), substituted a leading digit (`101679`
for `701679`, `0851` for `70851`), or returned no-text. Qwen reads
these correctly because the underlying glyphs are clean — paddle's
detector is the failure point, not the recognizer.

### The 9 wrong-text introductions

| ann | bucket    | GT          | paddle      | qwen        | impact in cascade |
|----:|-----------|-------------|-------------|-------------|-------------------|
|  39 | wide      | `MAT R31997`| `R31997`    | `R31997`    | wrong→wrong (same) |
| 127 | wide      | `70863`     | `70663`     | `708663`    | wrong→wrong (different) |
| 132 | wide      | `702130`    | `702150`    | `702180`    | wrong→wrong (different) |
| 141 | landscape | `51 1842`   | `511842`    | `511842`    | wrong→wrong (same; GT has space) |
| 301 | landscape | `51 1842`   | `511842`    | `511842`    | wrong→wrong (same; GT has space) |
| 331 | wide      | `R46302P`   | `R46302`    | `R46302`    | wrong→wrong (same) |
| 371 | wide      | `70296`     | `70298`     | `70298`     | wrong→wrong (same) |
| 389 | wide      | `7230`      | (no-text)   | `17350`     | **none→wrong (precision regression)** |
| 405 | wide      | `70863`     | `70663`     | `766663`    | wrong→wrong (different) |

Three sub-patterns:
- **5 cases (39, 141, 301, 331, 371): Qwen agrees with paddle.** Both
  produce the same wrong text; cascade replaces wrong with identical
  wrong. No behavioural change.
- **3 cases (127, 132, 405): Qwen and paddle both wrong, differently.**
  Cascade swaps the wrong text. Net wrong-text count unchanged.
- **1 case (389): paddle no-text, Qwen wrong.** This is the only true
  precision regression — cascade introduces a wrong-text where there
  was a clean miss.

Two cases (141, 301) have ground-truth labels (`51 1842`) that contain
a space; both engines read `511842`. This may be a label-quality issue
rather than a model failure.

## Decision

**Stage 1 falls in the 15–30% borderline band.** Per the
pre-registered gate, this requires manual decision rather than auto-
proceeding to Stage 2.

**Arguments for proceeding to Stage 2:**

- Net +7 EXACT on the 419-crop run = 314 → 321 (76.6%, +1.7 pp).
- Only 1 true precision regression (ann 389: none→wrong).
- 5 of 9 "wrong-text" cases are wrong→identical-wrong (no observable
  change).
- The Qwen3-VL rescues are *complementary* in nature — they're paddle
  detector failures (boxes too small / boxes too greedy), not paddle
  recognizer failures. This is the opposite signal from GOT-OCR
  (EXP-26), where regressions outnumbered rescues 14:1.

**Arguments against proceeding to Stage 2:**

- 15.6% rescue rate is statistically close to GOT-OCR's 15% — the
  pattern that justified GOT-OCR's rejection.
- A full Stage 2 run fires Qwen on **all** horizontal paddle-fails
  *and* all paddle outputs that fail the relaxed format gate. That's
  more crops than the 45 here, and the wrong-text rate on those
  additional crops is unknown — it may be higher than 20%.
- very_wide rescue rate is 0%; landscape n=5 too small to extrapolate;
  the +7 wins are entirely from the wide bucket (6/33 = 18.2%).

**Recommendation:** proceed to Stage 2 *with one tightening*: skip the
very_wide bucket entirely (0% rescue rate, no upside). Run Stage 2
on `landscape ∪ wide` only, using `--horizontal-strategy qwen` gated to
those two buckets.

## Out of scope (Stage 1)

- Stage 2 wiring (`--horizontal-strategy` flag, full 290-crop A/B).
- Production wiring (`app/ocr_processor.py`).
- Aspect-gate widening to include very_wide once the bucket has any
  signal.
- Label-quality review of ann 141 / 301 (`51 1842` GT spacing).

## Reproducibility

- Model: `qwen/qwen3-vl-8b-instruct` via OpenRouter.
- Prompt: `tests/qwen_horizontal.py` `PROMPT_VERSION = "v1-horizontal"`.
- Format regex: `^(JBHZ\d{6}|JBHU\d{6}|R\d{5}|\d{5,6})$`.
- Dataset: `tests/dataset/20260423`,
  `annotations_2026-04-23_11-24_coco_with_text.json`.
- Baseline: [benchmark_EXP-25-VERIFY-NO-FMT-CHECK_20260427_133655.json](../tests/results/benchmark_EXP-25-VERIFY-NO-FMT-CHECK_20260427_133655.json).
- Cache: `tests/results/qwen_horizontal_cache/` (re-runs deterministic).
