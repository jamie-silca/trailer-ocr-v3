# OCR Performance Experiment 22 — PP-OCRv5 Migration

**Status:** RUN — **REJECTED as a drop-in replacement.** 0/15 on the portrait
sanity gate, ~15× slower than v4, modest regression on a sampled landscape
crop. Full 672 run skipped per decision-rubric §8 (< 10/156 → REJECT).

> **Earlier "hidden finding" retracted (2026-04-25):** §10 originally claimed
> v5's detector returned 6 per-character boxes on `ann 14` and motivated a
> revised EXP-17. Verification at full-dataset scale (156 portrait crops) and
> targeted re-run on `ann 14` across three threshold configurations × two
> input scales returned **zero boxes in every combination**. The claim does
> not reproduce; see §10 for the corrected data and the JSON dumps under
> `tests/results/`. **EXP-17 has been rejected as a consequence.**

## TL;DR

We are on `paddleocr 2.7.3` + `paddlepaddle 2.6.2` (PP-OCRv4). PP-OCRv5 shipped
in mid-2025 and its release notes explicitly claim **"substantial improvements
in detection and recognition for challenging scenarios, including vertical text
layouts."** Portrait is the project-critical blocker (EXP-09 baseline: 2/156 = 1.3%).
Every experiment that attacked portrait via preprocessing, stitching, or recogniser
swap has either regressed or held flat. **The recogniser itself being upgraded to a
version that natively handles vertical text has not yet been tried.**

Previous agent deferred this in `ocr-improvement-tracker.md §8` on the grounds
that (a) the stack required 5+ hacks to run and v5 needs PaddlePaddle 3.x
(dep-hell risk), and (b) earlier experiments targeted preprocessing, not model
architecture. Both conditions have since changed: (a) we now have ~20
experiments, so the version-hell tradeoff is re-scoped — it's one experiment
against many unknowns vs. the last untried in-family lever; (b) portrait is
now conclusively an architecture problem (EXP-13, EXP-16, EXP-20, EXP-18 all
failed on it).

## 1. Goal

Test whether a `paddleocr 3.x` install with PP-OCRv5 weights, used as a drop-in
replacement for the current PP-OCRv4 pipeline, improves portrait-bucket
accuracy. Secondary: verify no regression on wide / landscape / near_square.

**Target:** portrait ≥ 30/156 (20%). Overall exact-match ≥ EXP-09's 38.2%.
**Stretch:** portrait ≥ 60/156 (38%), overall ≥ 45%.

(Targets are lower than EXP-18's 77% because v5's vertical-text claim is
generic — it was trained on CJK vertical layouts, not English stacked-vertical
industrial IDs. Any meaningful lift is valuable.)

## Baseline for comparison

**Primary baseline = EXP-09** (current tuned v4 pipeline = `paddleocr 2.7.3` +
bbox expansion + EXP-03+04+06 preprocessing + cascade retry + `MIN_CONFIDENCE=0.6`):

| Metric | EXP-09 value |
|---|---|
| Exact match (overall) | 38.2% (257/672) |
| Precision | 57.6% |
| Median latency | 110.6 ms |
| Text returned | 66.4% (446/672) |
| Portrait correct | 2/156 (1.3%) |
| Wide correct | 218/424 (51.4%) |
| Landscape correct | 33/54 (61.1%) |

BASE-01 (raw v4, no tuning; 22.3% / 51.0% / 123.6 ms) is a secondary reference
for architecture-only comparison but is not the production-relevant number.

### Config translation note

PP-OCRv5 uses `paddleocr 3.x`, whose Python API has renamed and partially
removed some v4 tuning knobs we currently rely on (`det_db_thresh`,
`det_db_box_thresh`, `det_db_unclip_ratio`). Exact 1:1 config matching may not
be possible. Plan: run v5 with defaults first (**EXP-22a**, pure architecture
swap); then, if any of EXP-09's tuning translates cleanly to the 3.x API,
re-run with matched params (**EXP-22b**, production-relevant comparison). If
all params translate, EXP-22a and EXP-22b collapse into one run. Any param
that does not translate will be called out explicitly in the results writeup.

## 2. Hypothesis

PP-OCRv5's detection model (PP-DocLayout-plus / enhanced DB) was retrained on
vertically-oriented layouts, so the detector will no longer collapse stacked
letters into a single elongated box. The v5 recogniser was similarly trained
on vertical text, so it should read the per-character boxes in their natural
reading order. The stacked-vertical trailer-ID layout is structurally identical
to Japanese vertical newspaper columns (characters upright, stacked top-to-bottom),
which v5 targets explicitly.

Risks to this hypothesis: v5's vertical-text training is heavily CJK-weighted;
performance on Latin alphanumerics in a vertical layout is not a guarantee. A
15-sample portrait sanity test will catch this early and save the full 672 run
if the hit rate is < 10%.

## 3. Methodology

### Install strategy

Install PP-OCRv5 into a **separate Python virtualenv** (`.venv-paddle-v5/`) —
don't touch the current `paddleocr 2.7.3` install that runs everything else.
This eliminates dependency-conflict risk entirely: if v5 breaks, the existing
v4 benchmarks and the production path in `app/ocr_processor.py` continue to
work untouched.

```
python -m venv .venv-paddle-v5
.venv-paddle-v5/Scripts/activate      # Windows
pip install "paddleocr>=3.0" "paddlepaddle>=3.0"   # CPU wheels
```

If the v5 install fails to import / initialise / read a trivial image, abort
and move to EXP-17 (per-char detection). Do not spend cycles on dep-resolution.

### Engine wiring

Add a new processor `tests/paddle_v5_processor.py` mirroring the
`OcrProcessor` / `TrocrProcessor` / `VlmPortraitProcessor` contract:

```python
class PaddleV5Processor:
    _instance = None
    def __new__(cls, model_dir=None): ...
    def process_image(self, image: PIL.Image) -> tuple[str|None, float]: ...
```

Add `--rec-engine paddle-v5` to `tests/benchmark_ocr.py` alongside the
existing `paddle` and `trocr` choices.

### Experiment order

1. **Import / init sanity** — in the v5 venv, load the model and OCR a single
   known-good landscape crop (ann id 672, GT `13208`). Must succeed before
   proceeding. Time budget: 15 minutes.
2. **15-crop portrait sanity** — reuse `tests/vlm_sanity.py` pattern; new
   script `tests/paddle_v5_sanity.py` using the same diverse-portrait picker.
   Gate: ≥ 3/15 correct to proceed to full run. Below that: reject and move
   to EXP-17.
3. **Full 672 benchmark** — `python tests/benchmark_ocr.py --exp-id EXP-22-paddle-v5 --rec-engine paddle-v5`.
   Same preprocessing, thresholds, and post-processing as EXP-09 so the delta
   is isolated to the v4 → v5 swap.
4. **Portrait-only re-run with EXP-13A stitch disabled / enabled** —
   confirm that v5's native vertical support makes stitching unnecessary (or
   that it still helps).

## 4. Critical Files

- `tests/paddle_v5_processor.py` (new) — v5 singleton, drop-in `process_image(pil) -> (text, conf)`.
- `tests/paddle_v5_sanity.py` (new) — 15-crop portrait gate script, parallel to [tests/vlm_sanity.py](../tests/vlm_sanity.py).
- [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — extend `--rec-engine` choices with `paddle-v5`, branch init and per-crop dispatch.
- [app/ocr_processor.py](../app/ocr_processor.py) — **no change during benchmark.** Only touched if verdict is ACCEPT.
- [requirements.txt](../requirements.txt) — unchanged. v5 deps live in `.venv-paddle-v5/` only for the duration of the experiment.

## 5. Implementation Steps

1. Create `.venv-paddle-v5/`, install `paddleocr>=3.0` + `paddlepaddle>=3.0` CPU wheels.
2. Run a one-liner import + OCR check on a known-good crop. Bail if it fails.
3. Implement `tests/paddle_v5_processor.py`. Mirror `app/ocr_processor.py`'s
   current thresholds (`det_db_thresh=0.2`, `det_db_box_thresh=0.3`,
   `det_db_unclip_ratio=2.0`) so the comparison is apples-to-apples against
   EXP-09. These names may have changed in the v3.x API — translate as needed.
4. Wire `--rec-engine paddle-v5` into the benchmark. Run the 15-crop sanity
   (`python tests/paddle_v5_sanity.py`).
5. Gate decision on sanity:
   - ≥ 3/15: proceed to full 672.
   - < 3/15: stop. Document result, move to EXP-17.
6. Full 672 run (`--exp-id EXP-22-paddle-v5`). Compare against EXP-09 via
   `tests/compare_experiments.py`.
7. If ACCEPT verdict (portrait ≥ 30/156 AND no bucket regression): promote to
   `app/ocr_processor.py`. Note: production migration involves bumping the
   main `requirements.txt` and re-validating deployment on Cloud Run. That's a
   **separate** follow-up — the benchmark-side experiment stops at "proven to
   work locally".

## 6. Verification

- **Primary:** portrait bucket correct count. Target ≥ 30/156.
- **Guardrail 1:** no bucket regresses by more than 2 correct-counts vs EXP-09.
- **Guardrail 2:** precision (correct / text-returned) within 3pp of EXP-09's 57.6%.
- **Guardrail 3:** median latency within 2× of EXP-09's 110 ms. v5 is somewhat
  heavier than v4 on CPU; a modest slowdown is acceptable, a 10× blow-up (as
  with TrOCR / Gemini) is not.
- **Sanity 1:** import and init in the clean venv succeed without Windows DLL,
  OpenMP, or argv-parsing hacks.

## 7. Risks

- **Install / DLL hell on Windows.** The previous agent's top concern. Mitigation:
  isolated venv, hard 1-hour time budget. If dep resolution takes longer, bail
  and move to EXP-17 — that's the ultimate fallback anyway.
- **PaddleOCR 3.x API is not backward-compatible with 2.x.** The constructor
  signature, result-tuple shape, and parameter names have changed. Expect to
  rewrite the `_run_ocr` helper from scratch, not port. Allocate 1–2 hours for
  API translation.
- **PP-OCRv5's vertical-text training is CJK-heavy.** Latin alphanumerics in
  a vertical layout may still be out-of-distribution. The 15-crop sanity gate
  catches this at minimum cost.
- **Weight-download size.** v5 models are ~100–300 MB larger than v4. Not a
  blocker for a local benchmark, but noteworthy for the eventual Cloud Run image.
- **Existing v4 install is not invalidated.** No production risk — v4 remains
  the deployed path until the benchmark verdict is in.

## 8. Decision Rubric

| Portrait result | Verdict | Next step |
|---|---|---|
| ≥ 60/156 (≥ 38%) | **STRONG ACCEPT** | Plan v4 → v5 migration in `app/ocr_processor.py` + `requirements.txt`. Re-validate Cloud Run deployment. Close EXP-17 as unnecessary. |
| 30–59/156 | **ACCEPT** | Promote, but keep EXP-17 open as a complementary layer. |
| 10–29/156 | **MARGINAL** | Don't promote yet. Run EXP-17 on top of v5 detection (v5 detector may give better per-char boxes even if v5 recogniser is mediocre). |
| < 10/156 | **REJECT** | v5 isn't the answer for this layout. Close this and execute EXP-17 on the existing v4 stack. |

## 9. Results (executed 2026-04-25)

### Install

| Component | v4 (current) | v5 (this experiment) |
|---|---|---|
| `paddleocr` | 2.7.3 | **3.5.0** |
| `paddlepaddle` | 2.6.2 | **3.3.1** |
| Detection model | PP-OCRv4 server det | PP-OCRv5_server_det |
| Recognition model | PP-OCRv4 mobile rec | **PP-OCRv5_server_rec** (multilingual; the model the v5 release notes attribute vertical-text capability to) |
| OneDNN/MKLDNN | Enabled | **Disabled** (paddle 3.3.1 PIR executor hits `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]` on Windows CPU when MKLDNN is on) |

Install completed cleanly into `.venv-paddle-v5/` in ~5 minutes. Initial
prediction call hit the MKLDNN bug above; resolved with `enable_mkldnn=False`.
No other deps-hell or DLL issues — the previous agent's primary concern (cited
in `ocr-improvement-tracker.md §8`) did not materialise on this stack.

### Config translation

All EXP-09 detection-tuning parameters translated 1:1 to the v3.x API:

| v4 param (paddleocr 2.7.3) | v5 equivalent (paddleocr 3.5.0) | Same value used |
|---|---|---|
| `det_db_thresh=0.2` | `text_det_thresh=0.2` | ✔ |
| `det_db_box_thresh=0.3` | `text_det_box_thresh=0.3` | ✔ |
| `det_db_unclip_ratio=2.0` | `text_det_unclip_ratio=2.0` | ✔ |
| `det_limit_side_len=960` | `text_det_limit_side_len=960` | ✔ (default) |
| `use_angle_cls=True` | `use_textline_orientation=True` | ✔ |

So EXP-22 is a **clean isolation of the model swap** with no drift in tuning.

### Spot checks before sanity run

Two crops, same tuned config, comparing v4 (production) vs v5:

| Annotation | Bucket | GT | v4 (EXP-09) | v5 (this) |
|---|---|---|---|---|
| ann 672 | landscape | `13208` | `'13208'` ✔ | `'3208'` (drops leading `1`) |
| ann 14 | portrait | `JBHZ672061` | `''` (empty) | `''` (empty) — but **detector returns 6 per-char boxes**, recogniser fails on each |

The portrait result is qualitatively different from v4: v4's detector returns
zero boxes, so the rec stage never runs. v5's detector returns six per-character
boxes, but the rec stage returns empty strings (or single chars like `'1'`)
for each — the recogniser is trained on text-strings, not isolated upright
characters in vertical context.

### 15-crop portrait sanity (`tests/paddle_v5_sanity.py --n 15`)

| # | ann | orig size | GT | v5 result | Match | Latency |
|---|---|---|---|---|---|---|
| 1 | 503 | 47×346 | `JBHZ669398` | empty | miss | 1891 ms |
| 2 | 212 | 52×326 | `JBHZ 669398` | empty | miss | 1634 ms |
| 3 | 136 | 57×191 | `663021` | empty | miss | 1476 ms |
| 4 | 63 | 57×329 | `JBHU252965` | empty | miss | 1654 ms |
| 5 | 628 | 45×328 | `JBHZ6685094` | empty | miss | 1748 ms |
| 6 | 290 | 47×335 | `JBHZ679578` | empty | miss | 1704 ms |
| 7 | 356 | 61×320 | `JBHZ677797` | empty | miss | 1612 ms |
| 8 | 528 | 46×342 | `JBHZ668184` | empty | miss | 1754 ms |
| 9 | 563 | 48×341 | `JBHZ677246` | empty | miss | 1717 ms |
| 10 | 448 | 38×157 | `43478` | empty | miss | 1488 ms |
| 11 | 586 | 58×350 | `JBHZ668184` | empty | miss | 1658 ms |
| 12 | 105 | 24×196 | `667469` | empty | miss | 1774 ms |
| 13 | 340 | 42×179 | `49484` | empty | miss | 1520 ms |
| 14 | 62 | 57×329 | `JBHU 252965` | empty | miss | 1634 ms |
| 15 | 470 | 69×264 | `7322039P` | `'322039'` | wrong | 2061 ms |

**0/15 hits. Wall: 26.6 s. Median ~1.65 s/crop.**

### Verdict per §8 decision rubric

`< 10/156` extrapolated from 0/15 sanity → **REJECT** as drop-in replacement.
Skip the full 672 run; the conclusion would not change for the rejection
question. The full 672 might be useful later as a baseline for EXP-17 with
v5 detection, but standalone it does not pay for the 17-minute wall time
when the verdict is foregone.

## 10. Hidden Finding — RETRACTED

> **Retraction (2026-04-25):** an earlier revision of this section claimed
> v5's detector returned 6 per-character boxes on `ann 14` and proposed
> EXP-17 build on that. **The claim does not reproduce.** Verified at the
> start of EXP-17 execution; results below.
>
> The original §10 was the basis for the revised EXP-17 plan
> ([docs/ocr-performance-experiment-17.md](ocr-performance-experiment-17.md)).
> EXP-17 has been marked REJECTED as a consequence.

### What the original §10 claimed

That on `ann 14` (portrait, 58×326, GT `JBHZ672061`), with EXP-09 detection
tuning, PP-OCRv5's detector returned **6 per-character boxes** at original
resolution and 7 at 4× upscale, while v4 returned 0. The implication drawn
was that v5 detection enables a per-char → horizontal-strip pipeline cheaply.

### What actually happens — full-dataset precompute

Ran `tests/precompute_v5_detection.py` over all 156 portrait-bucket
annotations with the documented EXP-09 tuning
(`text_det_thresh=0.2`, `text_det_box_thresh=0.3`, `text_det_unclip_ratio=2.0`):

| Metric | Result |
|---|---|
| Crops with at least one detection box | **25 / 156** (16 %) |
| Box-count distribution | **min=0, median=0, max=2** |
| Boxes on `ann 14` specifically | **0** (vs claimed 6) |
| Total wall time | 301 s (~1.93 s/crop) |

Output dumped to [tests/results/exp22_v5_detection.json](../tests/results/exp22_v5_detection.json).

### Targeted reproduction on `ann 14` across configs

`tests/verify_v5_ann14_v2.py` ran v5 detection on `ann 14` under three
detection-threshold configurations × two input scales:

| Config | 58×326 original | 232×1304 (4× upscale) |
|---|---|---|
| A — EXP-09 tuning (the documented config) | **0 boxes** | 0 boxes |
| C — defaults (no tuning) | 0 boxes | 0 boxes |
| D — extreme-permissive (`thresh=0.1`, `box_thresh=0.15`, `unclip=2.5`) | 0 boxes | 0 boxes |

All six combinations return zero boxes. Output dumped to
[tests/results/verify_v5_ann14.json](../tests/results/verify_v5_ann14.json).

### Likely cause of the retracted claim

Cannot be reconstructed from the prior session. Most plausible explanations,
in rough order of probability:

- The previous run inspected `rec_polys` (4 corner points × 1 box ≈ "6 numbers")
  or counted dictionary keys on `result[0]` rather than `len(dt_polys)`.
- The previous run used a transient PaddleOCR build / a non-default
  intermediate model that has since been replaced.
- The previous run's number was synthesised from notes, not the actual output.

In all cases the doc's number is wrong and the implication drawn from it does
not hold.

### Impact on Round 2

EXP-17 (revised) was the only Round 2 plan still on the queue that depended on
this finding. With the premise gone, no other Round 2 lever for the portrait
bucket remains in scope. **Round 2 closes at EXP-09's 38.2 % / 57.6 %**
([docs/ocr-experiments-report.md](ocr-experiments-report.md)).

## 11. Speed regression

Median ~1.65 s/crop on portrait, ~1.85 s on landscape (two-sample mean from
the spot checks). EXP-09's median is 110 ms. **v5 is ~15× slower than v4** on
this CPU configuration with MKLDNN disabled. Re-enabling MKLDNN would likely
recover some of the latency (it can't be benchmarked on Windows until the
PIR-executor bug is patched upstream), but probably not all 15×. This alone
is a strong reason against a production swap even if accuracy were on par.

## 12. Production-impact assessment

- **No.** Do not promote v5 to `app/ocr_processor.py`.
- The v5 install lives only in `.venv-paddle-v5/` and does not interfere with
  the deployed v4 path or the existing `paddleocr 2.7.3` install in the
  primary Python environment.
- Production dependency surface (`requirements.txt`, Dockerfile, Cloud Run
  image) is unchanged.
- `tests/paddle_v5_processor.py` and `tests/paddle_v5_sanity.py` remain as
  benchmark-only utilities. If EXP-17 reuses v5's detector, the benchmark
  pipeline will pick them up via the v5 venv; production wiring is a
  separate decision tied to whether EXP-17 itself ships.

## 13. Out of Scope

- **Production migration.** Success here justifies planning a migration, not
  executing one. Cloud Run redeploy + requirements pinning + Docker image
  re-validation is a separate task with its own risk budget.
- **PP-OCRv5 fine-tuning on trailer IDs.** Future work if out-of-box v5 is
  close but not quite (e.g. 20–30/156). Needs labelled training data.
- **PaddleOCR-VL (the 0.9B VLM).** Separate experiment. v5 first because it's
  a drop-in replacement in the pipeline we already have; PaddleOCR-VL is a
  different architecture (VLM-style encoder-decoder) that would need new
  wiring regardless of outcome.
- **Benchmarks on non-portrait buckets.** Measured, but not optimised for —
  v4 is already strong there.
