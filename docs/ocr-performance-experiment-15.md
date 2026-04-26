# OCR Performance Experiment 15 — Format-Aware Candidate Rescoring

## 1. Goal

Reduce wrong-text errors by rewriting OCR output strings that are one or two character swaps away from a known trailer-ID format. Pure post-processing — no model calls, no preprocessing changes, no latency beyond microseconds of regex/string work.

Target: **+1-3pp exact-match accuracy** and **+2-5pp precision** on top of the EXP-09 baseline (bbox expansion done upstream in the main app + EXP-03+04+06 inside this service), with zero regression on non-matching crops.

---

## 2. Methodology

### Baseline for this round

EXP-09-equivalent config:
- Upstream (main app): 10% bbox expansion before crop
- This service: `pad` to 64px min dim + lowered detection thresholds (`det_db_thresh=0.2`, `box_thresh=0.3`, `unclip=2.0`) + EXP-06 postprocess

Baseline (reproduced as `EXP-15-BASELINE`): **257/672 = 38.2%, precision 61.5%**.

### The rescorer

Given the post-processed OCR string `s` (upper-cased, stripped, spaces removed):

1. Score `s` against a whitelist of strict trailer-ID formats:
   - `^JBHZ\d{6}$` → 1.0
   - `^JBHU\d{6}$` → 1.0
   - `^R\d{5}$` → 1.0
2. If `s` already scores 1.0, return unchanged.
3. Else enumerate candidate strings by applying character-confusion substitutions at each position (BFS, depth ≤ 2, cap 64 candidates). Confusions: `0↔O/D/Q`, `1↔I/L`, `2↔Z`, `5↔S`, `6↔G`, `8↔B`, `H→8`.
4. Keep only candidates that (a) score ≥ `raw + 0.5`, and (b) differ from `s` by at most 2 characters (Hamming).
5. Pick the highest-scoring candidate; tie-break by smallest edit distance. Return it, or `s` unchanged if no candidate qualifies.

### First attempt — what went wrong

Initial version included a generic fallback pattern `^[A-Z]{3,4}\d{5,7}$` at score 0.5. This caused **8 correct-to-wrong rewrites** on IDs that genuinely start with digit `1`, e.g.:

```
'1RZ96854' -> 'IRZ96854'   (gt='1RZ96854')   # WRONG
'1RG21086' -> 'LRG21086'   (gt='1RG21086')   # WRONG
```

The dataset contains a whole family of `1RZ*` and `1RG*` trailer IDs whose actual first character is a digit. The generic pattern was a fabricated prior, not a real format in the data, so it biased the rescorer toward rewriting real leading digits into letters.

**Fix:** removed the 0.5 generic pattern entirely. Only the three strict patterns remain. This means a rescore requires going from raw score 0.0 → candidate score 1.0.

### Safeguards

- Never operate on empty/missing text.
- Never operate on strings shorter than 4 chars.
- Never rewrite a string that already matches a strict format.
- Cap accepted substitutions at Hamming distance 2.

---

## 3. Results

Run ID: `EXP-15_20260424_162508`. Comparison against `EXP-15-BASELINE_20260424_161921`.

### Headline

| Metric | Baseline | EXP-15 | Δ |
|---|---|---|---|
| Text returned | 418 (62.2%) | 418 (62.2%) | — |
| **Correct (exact match)** | **257 (38.2%)** | **259 (38.5%)** | **+0.3pp ▲** |
| Wrong text | 161 (24.0%) | 159 (23.7%) | -0.3pp ▲ |
| **Precision** | **61.5%** | **62.0%** | **+0.5pp ▲** |
| Median latency | 137.13 ms | 138.44 ms | +1.3 ms (noise) |
| p95 latency | 365.07 ms | ~370 ms | noise |

### Rescore tally

4 rescores fired across the 672 annotations:

| Outcome | Count | Examples |
|---|---|---|
| **Helpful** (wrong → correct) | **2** | `RS0408 → R50408` (gt `R50408`), `R26D42 → R26042` (gt `R26042`) |
| Harmful (correct → wrong) | 0 | — |
| Neutral (wrong → wrong) | 2 | `RZ0833 → R20833` (gt `1RZ08337`), `RS0421 → R50421` (gt `R90421`) |

### Per-bucket impact

| Bucket | Baseline correct | EXP-15 correct | Δ |
|---|---|---|---|
| wide | 218 / 424 | 220 / 424 | +2 ▲ |
| landscape | 33 / 54 | 33 / 54 | — |
| portrait | 2 / 156 | 2 / 156 | — |
| very_wide | 4 / 34 | 4 / 34 | — |
| near_square | 0 / 4 | 0 / 4 | — |

Gains are concentrated in the `wide` bucket, as expected — this is where the `R\d{5}` trailer IDs live and where PaddleOCR's character-level errors are most fixable.

---

## 4. Verdict

**ACCEPTED (marginal).**

The rescorer produced 2 clean wins and 0 regressions across 672 annotations. The absolute gain is below the plan's +1-3pp target, but:

- Precision improved (+0.5pp) — no false positives introduced.
- Latency is noise-level (<1ms overhead).
- The rescorer fires only 4 times; it's extremely conservative and composable.
- Future experiments can layer on top without interaction concerns.

The low rescore count indicates that most wrong-text errors in the EXP-09 baseline are **not** simple 1-2 character confusions against the known formats — they're either genuinely different strings (non-`R`/non-`JBH*` formats) or have more than 2 character errors. This bounds the ceiling of pure format rescoring.

### Recommendations

1. Wire `format_rescore()` into `OcrProcessor.process_image()` after `postprocess_text()`. Low risk, small durable win.
2. Keep the confusion table and patterns in one place (`app/utils.py`) so Round 2 experiments that add more patterns (e.g. learning from production traffic) can extend it cleanly.
3. Move on to **EXP-14 (temporal aggregation)** — likely a larger lever since many wrong-text cases are one-of-a-kind frame errors that a multi-frame vote could correct.

---

## 5. Reproducing the Run

```bash
# Baseline sanity (must match EXP-09 at 38.2%)
python tests/benchmark_ocr.py --exp-id EXP-15-baseline \
  --preprocess pad,postprocess --bbox-expand-ratio 0.1 \
  --det-db-thresh 0.2 --det-db-box-thresh 0.3 --det-db-unclip-ratio 2.0 \
  --format-rescore off

# EXP-15
python tests/benchmark_ocr.py --exp-id EXP-15 \
  --preprocess pad,postprocess --bbox-expand-ratio 0.1 \
  --det-db-thresh 0.2 --det-db-box-thresh 0.3 --det-db-unclip-ratio 2.0 \
  --format-rescore on
```

Implementation: `format_rescore()` in [app/utils.py](../app/utils.py). Benchmark flag wiring in [tests/benchmark_ocr.py](../tests/benchmark_ocr.py).

---

## 6. Lessons

1. **Fabricated priors bite hard.** The generic `^[A-Z]{3,4}\d{5,7}$` pattern seemed safe — it was weaker (score 0.5) and targeted "plausible ISO-6346 shapes". In practice it biased the rescorer against the real data distribution, costing 8 correct predictions. Encoding only **observed** format patterns is safer than encoding **assumed** ones.
2. **Rescoring is a small lever on this dataset.** Most wrong-text errors aren't 1-2 char fixes against the known formats. Bigger gains will come from temporal voting, not string surgery.
3. **Character-confusion tables work cleanly when gated on strict format matches.** Zero harm when the candidate must jump to a score-1.0 format; harm concentrates when a partial-credit bar is used.
