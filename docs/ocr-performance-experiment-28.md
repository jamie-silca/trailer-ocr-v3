# OCR Performance Experiment 28 — Qwen-first for Portrait Crops

**Status:** RUN — **NEGATIVE (2026-04-27).** Hypothesis rejected. Qwen-first
(skip paddle entirely on portrait) is strictly worse than the EXP-25 cascade
on accuracy: **52/119 (43.7%)** vs **59/119 (49.6%)** — a loss of 7 portrait
crops with 0 gains. The EXP-25 cascade remains the correct production strategy.

## Context

EXP-25 cascade (portrait: run paddle → Qwen if paddle format-misses) lands at
59/119 portrait correct and 25/119 wrong-text. In production, portrait crops
returning wrong text raised a question: **would calling Qwen unconditionally on
portrait crops — skipping paddle and returning None when Qwen rejects — improve
portrait accuracy and/or eliminate the wrong-text failures?**

The preliminary analysis (from cross-referencing the EXP-25 spike vs the
cascade) predicted +4 accuracy and -9 wrong-text. This experiment tests that
directly.

## Hypothesis

Calling Qwen3-VL-8B as the primary (not fallback) engine for portrait crops
(aspect < 0.5) and returning `None` on UNKNOWN instead of falling back to
paddle's output will:

1. Increase portrait exact-match accuracy over EXP-25 cascade.
2. Eliminate the 9 "paddle garbage leak" wrong-text cases visible in EXP-25.

## Implementation

Added `--portrait-strategy qwen-first` flag to
[tests/benchmark_ocr.py](../tests/benchmark_ocr.py). When set, portrait crops
skip the standard PaddleOCR path entirely; Qwen is called directly, and the
result is `None` (no-text) on UNKNOWN rather than falling back to paddle output.
Non-portrait crops use the standard pipeline unchanged.

## Results (2026-04-27)

Run command:
```bash
python tests/benchmark_ocr.py --exp-id EXP-28-QWEN-FIRST \
  --dataset 20260423 --portrait-strategy qwen-first --format-rescore on
```

Output: `tests/results/benchmark_EXP-28-QWEN-FIRST_20260427_130634.json`

### Full-dataset comparison

| Metric | EXP-25 cascade | EXP-28 qwen-first | Δ |
|---|---|---|---|
| **Overall EXACT** | **314 / 419 (74.9%)** | 307 / 419 (73.3%) | **-7 (-1.6 pp)** |
| **Precision** | 83.3% | **86.5%** | **+3.2 pp** |
| Wrong-text | 63 (15.0%) | **48 (11.5%)** | **-15** |
| No-text | 42 (10.0%) | 64 (15.3%) | +22 |
| Portrait EXACT | 59 / 119 (49.6%) | 52 / 119 (43.7%) | -7 (-5.9 pp) |
| Portrait text returned | 84 / 119 (70.6%) | 62 / 119 (52.1%) | -22 |
| Wide EXACT | 222 / 255 (87.1%) | 222 / 255 (87.1%) | flat |
| Landscape EXACT | 30 / 35 (85.7%) | 30 / 35 (85.7%) | flat |
| Wall time (cached) | 77.6 s | 56.6 s | -21 s |

### Portrait crop divergences

**EXP-25 cascade correct, EXP-28 qwen-first wrong (7):**

| ann | GT | cascade | qwen-first | root cause |
|-----|-----|-----|-----|-----|
| 1 | `352827` | `352827` | None | numeric — Qwen format gate rejects |
| 2 | `353267` | `353267` | None | numeric — Qwen format gate rejects |
| 8 | `23283` | `23283` | None | numeric — Qwen format gate rejects |
| 58 | `352088` | `352088` | None | numeric — Qwen format gate rejects |
| 61 | `352337` | `352337` | None | numeric — Qwen format gate rejects |
| 129 | `353267` | `353267` | None | numeric — Qwen format gate rejects |
| 138 | `352337` | `352337` | None | numeric — Qwen format gate rejects |

**EXP-28 qwen-first correct, EXP-25 cascade wrong (0):** None.

## Verdict

**Hypothesis rejected on both counts.**

1. **Accuracy:** Qwen-first loses 7 portrait crops and gains 0. All 7 losses
   are numeric portrait IDs (e.g. `352827`, `23283`) where Qwen's format
   whitelist (`^(JBHZ\d{6}|JBHU\d{6}|R\d{5})$`) correctly returns UNKNOWN
   and the cascade falls back to paddle's correct read. Qwen-first has no
   fallback, so these return no-text.

2. **Wrong-text elimination:** Wrong-text does drop (63 → 48, -15), but at
   the cost of -7 correct answers. Net is a loss.

3. **Precision improvement is real (+3.2 pp):** The tradeoff is cleaner
   failures (no-text instead of garbage) in exchange for fewer correct answers.
   Whether this tradeoff is worth it depends on the downstream use case. For
   a trailer-gate system where a false read is more harmful than a miss, the
   higher-precision qwen-first variant could be justified — but the current
   deployment does not warrant the regression.

### Why the pre-experiment prediction was wrong

The spike predicted +4 correct crops (63 vs 59). The benchmark shows -7. Three
sources of error:

1. **Spike didn't apply the Qwen format gate.** `tests/qwen_spike2.py` returned
   Qwen's raw output and matched it against GT directly. `qwen_portrait.py`
   (used by the benchmark) applies `^(JBHZ\d{6}|JBHU\d{6}|R\d{5})$` — so
   numeric IDs that Qwen reads correctly are still rejected and returned as
   UNKNOWN. The spike's "63 correct" included these numeric reads; the
   benchmark doesn't.

2. **Cache misses from preprocessing difference.** The spike passed raw crops;
   the benchmark applies `pad_small` before the Qwen call. Different image →
   different sha256 cache key → fresh OpenRouter API call → non-deterministic
   result. Five JBHZ crops that Qwen read correctly in the spike returned
   UNKNOWN on fresh calls in EXP-28.

3. **Cross-run non-determinism.** OpenRouter routes between providers; the same
   crop can return a different token sequence on a fresh call (noted in EXP-25
   docs). The cache was not re-usable between the spike and EXP-28.

## Recommendation

**Keep EXP-25 cascade as the production and benchmark strategy.** Qwen-first
offers no accuracy benefit and regresses numeric portrait crops.

The 9 "paddle garbage leak" wrong-text cases from EXP-25 (where Qwen returns
UNKNOWN and paddle's garbage is preserved) remain an open quality issue. A
targeted fix — null paddle's output only when it contains non-alphanumeric
characters or exceeds plausible trailer-ID length — could eliminate those 9
cases without touching the numeric portrait wins. This is not a separate
experiment; it is a production postprocess heuristic.

## Out of Scope

- Any change to `app/ocr_processor.py` — this experiment was benchmark-only.
- Testing qwen-first with a different (wider or narrower) aspect-ratio gate.
- Ensemble approaches (Qwen + paddle voting).
