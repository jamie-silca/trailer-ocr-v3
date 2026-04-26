# OCR Performance Experiment 20 — TrOCR on Unrolled Horizontal Strip

**Status:** RUN — **REJECTED as drop-in replacement.** Scope pivoted mid-experiment
from "TrOCR on stitched portrait strip" (Scope B) to "TrOCR as full-pipeline
recogniser" (Scope A) after spike 1 confirmed TrOCR cannot read EXP-13's
stitched portrait strip (see §9). Scope A ran; results below.

## TL;DR

| Metric | EXP-09 baseline | EXP-20 (trocr-base, full swap) | Δ |
|---|---|---|---|
| Overall correct | 257 / 672 (38.2%) | **275 / 672 (40.9%)** | +18 (+2.7 pp) |
| Precision | **57.6%** | 42.2% | **-15.4 pp** |
| Portrait | 2 / 156 (1.3%) | 0 / 156 (0.0%) | -2 |
| Wide | 218 / 424 (51.4%) | **235 / 422 (55.7%)** | +17 |
| Very wide | 4 / 34 (11.8%) | 7 / 36 (19.4%) | +3 |
| Landscape | 33 / 54 (61.1%) | 33 / 55 (60.0%) | ~flat |
| Median latency | 110 ms | **989 ms** | +9× |

TrOCR lifts exact match by +2.7 pp — real gain concentrated in `wide` — but
collapses precision by 15 pp because it hallucinates text on crops that PP-OCR
correctly rejects (`no_text` dropped 226 → 20, `wrong_text` rose 189 → 377).
It also fails portrait entirely and runs 9× slower.

**Not a drop-in replacement.** Viable only as an *additive* component:
disagreement fallback or format-gated second pass. See §10.


## 1. Goal

Replace PP-OCRv4's CRNN recogniser with Microsoft's **TrOCR**
(`microsoft/trocr-base-printed`) for portrait-bucket crops. TrOCR is a
transformer encoder-decoder trained extensively on clean printed text —
often 5–15pp stronger than CRNNs on short ID strings. Run it on the
horizontal strip produced by EXP-16's unroll step.

Target: on top of EXP-16, **+10–20 additional portrait correct**
(portrait bucket 35–55/156). Overall **+2–3pp** on top of EXP-16.

## 2. Hypothesis

EXP-16 will already have solved the *layout* problem (stacked vertical →
horizontal strip). The remaining ceiling is set by the **recogniser's ability
to read that strip**. PP-OCRv4 rec is good, but printed trailer IDs against
unusual backgrounds are a known weak spot for CRNNs. TrOCR's ViT encoder sees
the whole strip at once, has seen orders-of-magnitude more printed-text pretraining,
and handles the short-string regime well.

## 3. Methodology

### Pipeline

1. Run EXP-16's unroll composer on portrait crops → horizontal strip.
2. Feed the strip to TrOCR (`microsoft/trocr-base-printed`).
3. Post-process identically to PP-OCR path (upper, strip, EXP-15 rescore).
4. Format-gate + fallback to EXP-16 output on failure.

### Model variants to A/B

- `microsoft/trocr-base-printed` (334M params, ~300 ms CPU).
- `microsoft/trocr-small-printed` (62M params, ~100 ms CPU) — pick if latency-bound.
- Optionally: `microsoft/trocr-large-printed` (558M) for a ceiling read.

### Infrastructure

- `transformers` + `torch` CPU inference. Model weights ~1.3 GB (base);
  confirm this is acceptable for the Docker image before committing.
- Load the model once at startup as a singleton (same pattern as the PaddleOCR
  instance).

## 4. Critical Files

- [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — add `--portrait-strategy trocr --trocr-model {small,base,large}`.
- `tests/trocr_portrait.py` (new) — `read_strip_trocr(strip_image, model) -> (text, confidence)`.
- [app/ocr_processor.py](../app/ocr_processor.py) — no change during benchmark.
- `requirements.txt` — add `transformers`, `torch` (CPU wheel) as optional benchmark deps.

## 5. Implementation Steps

1. Spike: load `trocr-small-printed` in a Python REPL, run on 5 hand-crafted
   stacked-vertical composites, confirm it outputs the expected strings. Bail
   early if TrOCR struggles on this layout without fine-tuning.
2. Implement `tests/trocr_portrait.py`.
3. Wire `--portrait-strategy trocr` into the benchmark loop. Reuse EXP-16's
   unroll step to produce the strip.
4. Run portrait-only subset with small / base / large; pick winning
   accuracy × latency tier.
5. Run full 672 benchmark with winning config.
6. Post-mortem: portrait delta vs EXP-16, latency distribution, model-size
   trade-off, cases where TrOCR beat PP-OCR on the same composite strip.

## 6. Verification

- **Primary:** portrait bucket ≥ EXP-16 + 10.
- **Guardrail 1:** non-portrait buckets unchanged.
- **Guardrail 2:** precision unchanged or improved.
- **Latency:** TrOCR-small ~100 ms, base ~300 ms, large ~700 ms CPU. Pick the
  tier that fits the per-portrait-crop budget.

## 7. Risks

- **Model-size bloat:** base model adds ~1.3 GB to the image. For a
  portrait-only, low-volume path this may be unjustified. If small/base lose
  to PP-OCR on the strip, abandon; don't ship large.
- **CPU latency:** transformer decoding on CPU is slow. If the production SLA
  is tight, TrOCR may be off the table despite accuracy wins.
- **Torch dependency conflicts:** PaddleOCR pins specific numpy/opencv
  versions. Validate that adding torch doesn't break PaddleOCR's install.
- **Novel failure modes:** TrOCR occasionally emits grammatical English when
  confused (it's a language model). Format gate catches these but loses
  fallback precision. Log explicitly.

## 9. Results (executed 2026-04-24)

### Spike 1 — TrOCR on EXP-13 stitched portrait strip

Run: `tests/trocr_spike.py`, inputs from `tests/results/exp13_debug/`, GT `JBHZ672061`.

| Input | trocr-small output | trocr-base output |
|---|---|---|
| Raw portrait crop | gibberish | gibberish |
| EXP-13 rect-stitch | `'REF6555829'` | `'***65655688'` |
| Rectified column | gibberish | gibberish |

Confirms EXP-13 Finding 1 from a second recogniser: stitched portrait strip
is unreadable regardless of the recogniser. Scope B abandoned here.

### Spike 2 — TrOCR on natural horizontal wide crops

Run: `tests/trocr_spike2.py`, 5 wide-bucket crops, 2 known-wrong for PP-OCR.

- trocr-small: 2 / 5
- **trocr-base: 4 / 5** — including correctly reading `ATLS03` where PP-OCR
  returns `ITESOS`.

This motivated the Scope A pivot: TrOCR may be a stronger recogniser than
PP-OCR CRNN on non-portrait buckets.

### Scope A — TrOCR-base as full-pipeline recogniser (full 672)

Run: `python tests/benchmark_ocr.py --exp-id EXP-20-full-trocr-base --rec-engine trocr --trocr-model microsoft/trocr-base-printed`.

- **Overall:** 275/672 correct (40.92%), 42.18% precision, median 989 ms.
- **Gain vs EXP-09:** +18 correct (+2.7 pp), but -15.4 pp precision.
- **Portrait:** 0/156 — TrOCR fails stacked-vertical identically to spike 1.
- **Wide:** 235/422 (55.7%) vs baseline 218/424 (51.4%), **+17 correct**.
- **No-text rate:** 20/672 (3.0%) vs baseline 226/672 (33.6%) — TrOCR
  essentially never returns empty, so every unreadable crop becomes a
  wrong-text answer. This is the root cause of the precision collapse.
- **Latency:** median 989 ms, p90 1547 ms, p95 1776 ms. ~9× slower than PP-OCR.

### Verdict

**REJECTED as drop-in replacement.** Scope A's precision regression (-15 pp)
and latency blow-up (9×) outweigh its +2.7 pp exact-match gain. Portrait (the
project-critical blocker) is *worse*, not better.

## 10. Follow-up opportunities

The `wide`-bucket gain is real. Two ways to keep it without the downsides:

1. **Disagreement fallback.** Run PP-OCR first. If format-rescore accepts it,
   ship it. Otherwise run TrOCR and accept its answer only if format-rescore
   accepts it. Keeps PP-OCR's precision, borrows TrOCR's wide-bucket reads
   on crops PP-OCR struggles with. Cost: ~989 ms extra on ambiguous crops only.
2. **TrOCR as a rescoring tie-breaker** on crops where PP-OCR confidence is
   low. Similar latency envelope, harder to implement cleanly.

Neither helps portrait. For portrait, proceed to EXP-18 (VLM) or EXP-17
(per-char detection) — both attack the right bottleneck.

## 11. Out of Scope

- Fine-tuning TrOCR on trailer-ID data (future work if base accuracy is close).
- TrOCR on non-portrait buckets — PP-OCR already adequate.
- GPU inference — evaluating a CPU-only deployment.
