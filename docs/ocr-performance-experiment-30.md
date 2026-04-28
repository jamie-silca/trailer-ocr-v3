# OCR Performance Experiment 30 — Qwen3-VL-only vs EXP-25 cascade (head-to-head)

**Status:** RUN — **MIXED (2026-04-28).** Direct answer to *"can the
VLM capture more than the current implementation, and capture things
the current impl fails at?"* — **Yes for some buckets, no overall.**

- VLM-only: **306/419 (73.0%)** — net **-8 EXACT** vs EXP-25 cascade
  (**314/419 (74.9%)**).
- VLM-only **uniquely wins 23 crops** EXP-25 misses.
- EXP-25 **uniquely wins 31 crops** VLM-only misses.
- The two engines are **partially complementary** — total reachable
  ceiling if perfectly oracled per crop is **283 + 23 + 31 = 337/419
  (80.4%)**, +5.5 pp above either alone.

The headline number is "VLM-only loses overall," but the per-bucket and
unique-win breakdown shows the VLM has clear strengths the cascade
isn't currently tapping — specifically on `very_wide` container plates
and on paddle no-text portrait numerics.

## Setup

- **Model:** `qwen/qwen3-vl-8b-instruct` via OpenRouter.
- **Prompt:** "universal" — describes both reading orders so the model
  isn't biased to vertical or horizontal. PROMPT_VERSION =
  `v1-universal`. See [tests/qwen_universal.py](../tests/qwen_universal.py).
- **No format gate.** Raw model output is uppercased, whitespace
  stripped, compared to GT (also uppercased + whitespace stripped). The
  literal `UNKNOWN` sentinel is mapped to no-text.
- **Baseline:** [tests/results/benchmark_EXP-25-VERIFY-NO-FMT-CHECK_20260427_133655.json](../tests/results/benchmark_EXP-25-VERIFY-NO-FMT-CHECK_20260427_133655.json).
- **Run command:**
  ```bash
  python tests/qwen_full_run.py --dataset 20260423
  ```
- **Wall:** 597 s for 419 fresh API calls (~$0.011), 43 s on cached
  re-run.

**Outputs:**
- [tests/results/qwen_full_run_20260423.json](../tests/results/qwen_full_run_20260423.json)
  — per-crop raw + normalized Qwen output, both engines' exact_match.
- [tests/results/qwen_vs_exp25_20260423.json](../tests/results/qwen_vs_exp25_20260423.json)
  — comparison summary with ann_id lists for each overlap quadrant.
- [tests/results/qwen_universal_cache/](../tests/results/qwen_universal_cache/)
  — sha256-keyed JSON cache; re-runs use it deterministically.

## Headline numbers

| metric                                      | EXP-25 cascade | Qwen-only | Δ    |
|---------------------------------------------|---------------:|----------:|-----:|
| **EXACT (419)**                             | **314 (74.9%)**| **306 (73.0%)** | **-8** |
| Wrong-text                                  | 63             | 81        | +18  |
| No-text                                     | 42             | 32        | -10  |
| Both correct                                | 283            | 283       | —    |
| **Qwen-only wins (VLM uniquely captures)**  | —              | **23**    |      |
| **EXP-25-only wins (VLM uniquely misses)**  | **31**         | —         |      |
| Both wrong                                  | 82             | 82        | —    |

### Per-bucket

| bucket    | n   | EXP-25 | Qwen-only | Qwen-only wins | EXP-25-only wins | Net Δ |
|-----------|----:|-------:|----------:|---------------:|-----------------:|------:|
| portrait  | 119 | 59     | 46        | 5              | **18**           | **-13** |
| landscape |  35 | 30     | 30        | 3              | 3                |  0    |
| wide      | 255 | 222    | 223       | **10**         | 9                | +1    |
| very_wide |  10 |  3     |  **7**    | **5**          | 1                | **+4** |

## Where the VLM shines (23 unique wins)

### 1. very_wide container plates — strongest signal (+5 unique wins of 10 crops)

VLM reads multi-segment container codes that paddle splits or shortens:

| ann | GT                 | EXP-25                | Qwen-only           |
|----:|--------------------|-----------------------|---------------------|
|  98 | `CXTU1156285`      | `CXTU_115628 5`       | `CXTU1156285`       |
| 105 | `KTU1156285`       | `KTU 115628 5`        | `KTU1156285`        |
| 140 | `TIFU 1723667`     | `TIFU 172366`         | `TIFU1723667`       |
| 325 | `UTCU 4941900`     | `UTC04941900`         | `UTCU4941900`       |
| 377 | `RLTU3006285`      | `RLT03006285`         | `RLTU3006285`       |

Pattern: paddle splits the container into separate detection boxes and
either drops the trailing digit, replaces `U` with `0`, or reorders.
The VLM treats the plate as one read.

### 2. Paddle no-text rescues on portrait numerics + clean wides (+5 unique wins)

Where paddle returned nothing at all and Qwen read the plate:

| ann | bucket    | GT       | Qwen-only |
|----:|-----------|----------|-----------|
| 246 | wide      | `L906`   | `L906`    |
| 260 | wide      | `77233`  | `77233`   |
| 308 | wide      | `L906`   | `L906`    |
| 340 | portrait  | `39723`  | `39723`   |
| 341 | portrait  | `40389`  | `40389`   |

These are easy cascade wins — a "fire Qwen on paddle no-text" trigger
catches all 5 with zero risk of overwriting a correct paddle read.

### 3. Paddle detector cropping/bolt-on errors (+8 unique wins)

Paddle returns a *partial* or *augmented* read; Qwen reads the full
plate cleanly:

| ann | bucket    | GT             | EXP-25            | Qwen-only      |
|----:|-----------|----------------|-------------------|----------------|
|  13 | wide      | `1RZ95517`     | `1R295517`        | `1RZ95517`     |
|  39 | wide      | `MAT R31997`   | `R31997`          | `MATR31997`    |
|  48 | wide      | `1RZ08417`     | `1RZ08`           | `1RZ08417`     |
|  81 | landscape | `70851`        | `0851`            | `70851`        |
| 135 | wide      | `UTCU 494896 ` | `494896 UTCU`     | `UTCU494896`   |
| 216 | wide      | `R45777`       | `R457TZ`          | `R45777`       |
| 257 | portrait  | `7322039P`     | `73220332`        | `7322039P`     |
| 329 | portrait  | `7322039P`     | `73220392`        | `7322039P`     |

These cases are **hardest to convert to cascade wins** — paddle has
non-empty wrong-text output, so a "fire on no-text" trigger misses
them. Capturing them would require either a confidence/length heuristic
on paddle output, or always-fire-Qwen with a vote.

## Where the VLM hurts (31 unique losses)

### 1. JBHZ stacked-vertical character drop (12 of 18 portrait losses)

Qwen drops the `B` from `JBHZ` or substitutes `Z→N`:

| ann | GT             | Qwen          | failure mode |
|----:|----------------|---------------|--------------|
|  19 | `JBHZ667652`   | `JHZ667652`   | dropped B   |
|  24 | `JBHZ092214`   | `JHZ092214`   | dropped B   |
|  26 | `JBHZ676595`   | `JHZ676595`   | dropped B   |
| 161 | `JBHZ675116`   | `JBHN675116`  | Z→N         |
| 164 | `JBHZ667496`   | `JBHN667496`  | Z→N         |
| 166 | `JBHZ672066`   | `JBHN672066`  | Z→N         |
| 165 | `JBHZ672066`   | `BH2672066`   | dropped J + Z→2 |
| 204 | `JBHZ668161`   | `JHZ668161`   | dropped B   |

This is exactly the failure mode the EXP-25 portrait cascade was
designed to *avoid* — paddle's correct JBHZ reads are why we keep it.

### 2. Single-character substitutions on clean wide numerics (5 unique losses)

| ann | bucket    | GT           | Qwen        |
|----:|-----------|--------------|-------------|
|  20 | wide      | `6669`       | `66669`     |
|  29 | wide      | `11-1260`    | `111260`    |
|  49 | wide      | `8268`       | `B268`      |
| 149 | wide      | `701619`     | `701679`    |
| 284 | landscape | `8713`       | `3713`      |
| 259 | wide      | `R46302`     | `R46302D`   |

Hyphen drops (`11-1260`), digit doubling (`6669`→`66669`), and 1-char
substitutions are common Qwen failure patterns on already-clean wide
crops.

### 3. Hallucinations / extra characters (2 cases)

| ann | bucket    | GT            | Qwen                |
|----:|-----------|---------------|---------------------|
| 102 | landscape | `4-26 4-24`   | `426VITUC424P`      |
| 242 | very_wide | `FLNU 500569` | `FLNU500569C`       |

## Direct answer to the question

> "Is VLM able to capture more than current implementation and/or
>  capture things current implementation would fail at?"

**Yes — 23 unique wins, concentrated where you'd expect:**

1. **very_wide container plates (5 wins / 10 crops, +50% of the bucket)** —
   the VLM is genuinely better here. Paddle's detector is the bottleneck
   on multi-segment plates.
2. **Paddle no-text rescues (5 wins on landscape/wide/portrait)** — VLM
   reads where paddle's detector found nothing.
3. **Detector cropping recoveries (8 wins on wide/landscape)** — VLM
   reads the full plate when paddle returned a truncated or
   bolted-on output.

**But VLM-only is a net regression** because it costs **31 wins on
JBHZ stacked-vertical reads + clean wide numerics** that paddle gets
right and the VLM mangles with character-substitution hallucinations.

**Implication for production:** the right design is **not** VLM-only,
and it is **not** the current portrait-only cascade either. The
asymmetric strength patterns suggest:

- **Definitely worth wiring:** `very_wide` Qwen cascade (+4 net,
  highest-confidence bucket-level win).
- **Worth wiring with care:** `paddle no-text` trigger across all
  horizontal buckets (+5 wins, 0 risk).
- **Marginal / not worth wiring:** wide bucket "always-fire" cascade —
  +10 wins offset by 9 losses ≈ wash, and the spike in EXP-29 already
  showed wide is borderline at 18% rescue rate.
- **Keep current behaviour:** portrait cascade as-is, `landscape`
  paddle-only.

## Out of scope

- Wiring any of the implications above (separate plan if approved).
- Multi-vote / confidence-blended outputs (paddle + Qwen voting). The
  no-format-gate Qwen output already has the data needed to test this
  post-hoc, but it's a different experiment.
- Other VLMs / local Qwen. EXP-26 and EXP-27 cover those; EXP-30
  reuses the EXP-25 OpenRouter Qwen3-VL-8B as the reference VLM.
