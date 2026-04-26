# OCR Performance Experiment 27 — Local VLM Spike Ladder (via Ollama)

**Status:** COMPLETE (2026-04-26). After rejecting `qwen3-vl:8b`
(thinking-mode + q4 quant gap, §1-11), a small-VLM ladder probe
(§12) landed on **`qwen2.5vl:3b`** as a viable local replacement for
the EXP-25 OpenRouter cascade — **60/119 EXACT (50.4%)** at 4 s/crop
on local Ollama, with a 50/92 JBHZ sub-bucket result that *exceeds*
the OpenRouter baseline (47/92). Recommendation: proceed to a
production-wiring plan. Ladder rejections preserved in §12.1 / §12.2.

**Headline:** Local Ollama qwen3-vl:8b q4_K_M (running on the dev
machine's NVIDIA RTX, *not* CPU as planned — Ollama auto-uses CUDA when
present) gets **24/80 EXACT (30%)** on the same portrait crops where
EXP-25's OpenRouter qwen/qwen3-vl-8b-instruct gets **46/80 EXACT (58%)**
— a stable **27.5 pp gap**. Median latency 13.8 s/crop on spike1,
~50 s/crop on spike2 once the harder JBHZ portraits land in the queue.

## Context

EXP-25 cascade depends on OpenRouter — ~$0.05 per 419-crop benchmark
plus a vendor key, network dependency, and per-call latency floor that
all carry over to production. Eliminating those is worth a focused
experiment **before** the production-wiring plan, because the answer
shapes that plan: local viable → no vendor at all; local viable with
trade-offs → cascade local with API fallback; not viable → keep the
OpenRouter dependency and design accordingly.

The plan was a CPU spike. **It turned out to be a GPU spike** — the dev
machine has an NVIDIA RTX (Ryzen 7 5800H + RTX, ~28 GB RAM total) and
Ollama auto-detected CUDA. Task Manager showed GPU 1 at 75% utilisation
/ 92 °C during inference; CPU sat at 21% (background load). True CPU
characterisation is a follow-up (`OLLAMA_NUM_GPU=0`); the GPU result
already invalidates the model class for this hardware regardless of CPU
behaviour, so the CPU run is deferred.

## 1. Goal (revised after GPU discovery)

Originally: characterise CPU latency vs OpenRouter accuracy on the
119-portrait set. As-run: characterise **GPU q4_K_M accuracy** vs the
same OpenRouter baseline. The accuracy gap is the load-bearing finding;
hardware is a secondary concern because if the model can't match
OpenRouter at full GPU speed, CPU won't help.

## 2. Hypothesis vs Outcome

**Hypothesis (planning):** Qwen3-VL-8B locally would be within ~10 pp of
the OpenRouter version's accuracy because it's the same model family /
weights / prompt; the quantisation step from full-precision (OpenRouter
hosting probably fp16 or bf16) to q4_K_M (Ollama default) was assumed
not to materially affect OCR-style structured-output tasks.

**Outcome:** Hypothesis falsified. The q4_K_M quant loses ~27.5 pp on
this stacked-vertical-text task. Direction is consistent across all
checkpoints (50, 60, 70, 80 crops) — the gap is not noise.

## 3. Method (as run)

### Setup

- Ollama 0.21.2 on Windows. `ollama pull qwen3-vl:8b` (~5.86 GB, q4_K_M
  default). Model digest `901cae732162`. Listed by `ollama show`:
  architecture qwen3vl, 8.8B params, context 262144, capabilities
  include `vision` and `thinking`.
- Hardware: AMD Ryzen 7 5800H (8 cores) + NVIDIA RTX (16 logical
  cores, 27.9 GB RAM). Ollama auto-uses CUDA.
- Endpoint: `http://localhost:11434/v1/chat/completions` (OpenAI-compat).
  Same prompt as EXP-25
  ([tests/qwen_portrait.py:35-41](../tests/qwen_portrait.py#L35-L41)).
  Same 768-px LANCZOS upscale.
- `temperature=0`, `max_tokens=512` (raised from EXP-25's 32; see §4).
- 119-portrait set from `tests/dataset/20260423`, ordered by ann_id —
  identical iteration order to EXP-25's
  [tests/qwen_spike2.py](../tests/qwen_spike2.py), so each progress
  checkpoint cross-references one-for-one against the OpenRouter rows
  in
  [tests/results/qwen_spike_8b_20260423.json](../tests/results/qwen_spike_8b_20260423.json).

### Files

- **[tests/qwen_local_processor.py](../tests/qwen_local_processor.py)** —
  subclass of QwenPortraitProcessor; overrides `_initialize`, `_call_api`,
  `_cache_get`, `_cache_put` to swap endpoint + auth + cache dir.
  Reuses prompt, format gate, 768-px upscale, sha256 PNG cache key.
- **[tests/qwen_local_spike.py](../tests/qwen_local_spike.py)** — 5-crop
  pre-flight, same ann_ids as
  [tests/qwen_spike.py](../tests/qwen_spike.py). Bypasses processor to
  capture *raw, pre-format-gate* output (the gate is for production
  cascade, not the spike measurement).
- **[tests/qwen_local_spike2.py](../tests/qwen_local_spike2.py)** —
  119-crop widen. Same shape as
  [tests/qwen_spike2.py](../tests/qwen_spike2.py). Was killed at 80/119;
  partial stdout preserved at
  [tests/results/qwen_local_spike2_partial_50of119_20260426.log](../tests/results/qwen_local_spike2_partial_50of119_20260426.log).

## 4. The thinking-mode trap

`qwen3-vl:8b` is a **thinking-mode** model. Setting `max_tokens=32`
(matching EXP-25's OpenRouter default) yields **empty `content` on every
crop** — the model burns the full 32-token budget on chain-of-thought
reasoning before emitting any answer.

Probe (single crop, ann_id 73, GT `JBHZ676208`):

```
content: ''
reasoning: "So, let's look at the image. The text is vertical. Let's
            read each character. The first part is J B H Z, then 6"
finish_reason: length
```

The reasoning *correctly identified* the prefix as JBHZ + leading 6 —
the model can read the image — but ran out of tokens before producing
`content`. Spike1's first run (with max_tokens=32) returned 0/5 EXACT;
re-run with max_tokens=512 returned 3/5 EXACT in 9-19 s/crop.

**`think: false` does not disable thinking** for this model in Ollama
0.21.2 — it just renames the streamed reasoning field from
`reasoning` (OpenAI-compat) to `thinking` (`/api/chat`). The reasoning
still consumes tokens.

The fix in all three new files: `max_tokens=512`. Comment added at each
call site so a future reader doesn't try to "optimise" it back down.

## 5. Spike1 — 5 portraits (PASS gate)

| ann | gt           | local_raw    | OR_raw       | local_ed | OR_ed | ms     |
|----:|--------------|--------------|--------------|---------:|------:|-------:|
|  17 | JBHU235644   | JBHU235644   | JBHU235644   |        0 |     0 | 11,111 |
|  73 | JBHZ676208   | JBHZ676208   | JBHZ676208   |        0 |     0 |  9,599 |
| 165 | JBHZ672066   | _(empty)_    | JBHZ672066   |       10 |     0 | 19,547 |
| 255 | JBHZ660789   | _(empty)_    | UNKNOWN      |       10 |     - | 19,583 |
| 356 | JBHZ672637   | JBHZ672637   | JBHZ672637   |        0 |     0 | 13,771 |

LOCAL: **3/5 EXACT (60%)**, avg 14.7 s/crop.
OpenRouter on same 5: 4/5 EXACT (80%).

Both empty-content failures (ann 165, 255) timed out at exactly 19.6 s
— the model burned the full 512-token budget without producing
content. This is the same failure mode as max_tokens=32 just at a
higher ceiling — for some crops, the reasoning never converges.

## 6. Spike2 (partial, killed at 80/119)

Per-crop progress checkpoints, with EXP-25 OpenRouter cross-reference
on the same ann_ids:

| n   | local EXACT | local %  | OR EXACT | OR %     | gap (pp) |
|----:|------------:|---------:|---------:|---------:|---------:|
|  50 |       14    | 28.0%    |       29 | 58.0%    |    -30.0 |
|  60 |       17    | 28.3%    |       35 | 58.3%    |    -30.0 |
|  70 |       21    | 30.0%    |       40 | 57.1%    |    -27.1 |
|  80 |       24    | 30.0%    |       46 | 57.5%    |    -27.5 |

The gap stabilises at ~27-30 pp from crop 50 onward. Extrapolating to
n=119: local would land at ~36/119 (30%) vs EXP-25's confirmed 63/119
(53%). Killed at 80/119 per user instruction once the gap was
demonstrably stable.

Latency: rate dropped from 0.05 crop/s on the first 5 crops (early
mostly-numeric annotations) to 0.02 crop/s by crop 30+ (JBHZ-heavy).
Mean ~50 s/crop on the harder JBHZ runs — the harder the crop, the
deeper the model thinks before giving up at 512 tokens. Projected
119-crop wall: ~70 min. Projected 419-crop cascade wall (~110 portraits
fire): ~92 min. **Outside** the doc's "≤ 30 min" recommendation
threshold *even if* accuracy had matched.

## 7. Comparison table (EXP-25 vs EXP-27, partial)

| metric                                | EXP-25 (OpenRouter Qwen3-VL-8B) | EXP-27 (Ollama qwen3-vl:8b q4_K_M, GPU) |
|---------------------------------------|---------------------------------|------------------------------------------|
| portrait EXACT (extrapolated to 119)  | 63/119 (53%)                    | ~36/119 (~30%)                           |
| portrait EXACT @ 80 (measured)        | 46/80 (58%)                     | 24/80 (30%)                              |
| accuracy gap (pp)                     | —                               | -27.5 pp                                 |
| median ms / crop                      | ~1,144 ms                       | ~14,000 ms (spike1) / ~50,000 (spike2)   |
| projected wall, 119 crops             | ~150 s                          | ~4,200 s (~70 min)                       |
| projected wall, 419 cascade           | ~140 s                          | ~5,500 s (~92 min)                       |
| cost per 419 run                      | ~$0.05                          | $0                                       |
| hardware                              | n/a (vendor)                    | NVIDIA RTX, 75% utilisation              |

**Decision rule from the plan:** *"If accuracy within ~10 pp of EXP-25
AND wall projection ≤ 30 min on 419, recommend scale test."* Neither
condition met. **Recommend not pursuing.**

## 8. Why the gap exists (hypothesis)

Three plausible contributors, in order of likelihood:

1. **q4_K_M quantisation cost.** OpenRouter likely serves the model at
   bf16 / fp16 / q8_0. q4_K_M is aggressive for fine-grained
   character-recognition tasks where each glyph's representation is
   load-bearing. Quantisation noise on the vision encoder's spatial
   tokens probably degrades the JBHZ-character-shape distinctions the
   most.
2. **Thinking-mode budget cap.** With max_tokens=512, ~10-15% of crops
   exhaust the budget on reasoning without converging. OpenRouter's
   model variant (`qwen/qwen3-vl-8b-instruct`) appears to behave more
   directly — possibly a different system prompt, possibly the hosted
   variant has thinking off by default.
3. **Vision encoder image-size handling.** Ollama may resize / patch
   images differently than OpenRouter's serving stack. Untested.

## 9. Recommendation

- **Don't wire `qwen3-vl:8b` q4_K_M as a local replacement for the
  EXP-25 cascade.** The accuracy gap is too large (27.5 pp) and the
  latency is too long (~92 min projected per 419 run vs OpenRouter's
  ~140 s).
- **Keep EXP-25 as the production path.** Its $0.05/run cost is
  trivially cheaper than the operational cost of dealing with this
  level of accuracy regression.
- **Open follow-ups (NOT running this round):**
  1. **Quantisation lever:** retry with q5_K_M or q8_0 (Ollama can pull
     these as separate tags if available, e.g.
     `qwen3-vl:8b-instruct-q8_0`). Closes the gap if the quant is the
     bottleneck. Cost: ~10-15 GB pull, ~1 hr re-run.
  2. **Different model:** Qwen2.5-VL-7B (older but no thinking mode) or
     MiniCPM-V 2.6 (OCR-focused). Lower accuracy ceiling but faster
     wall + simpler output. Worth a 5-crop spike before widening.
  3. **True CPU characterisation** (`OLLAMA_NUM_GPU=0`). Only
     interesting if the GPU result had been positive — currently it's
     not, so deferred.

## 10. Out of Scope

- Wiring `qwen_local` as a `--portrait-strategy` choice. Not justified
  by the data.
- Quantisation tuning beyond q4_K_M. Listed above as a follow-up but
  not part of this experiment.
- Running on different hardware. The dev machine's RTX is what we have;
  production CPU cost (the original question) is moot if the model
  class doesn't clear the accuracy bar at any cost.

## 11. Reproducibility

- Ollama version: `0.21.2`
- Model tag: `qwen3-vl:8b`
- Model digest: `901cae732162`
- Quantisation: `Q4_K_M`
- CPU: AMD Ryzen 7 5800H (8 cores / 16 threads, 3.20 GHz base)
- GPU (auto-used): NVIDIA RTX (75% utilisation observed during inference)
- RAM: 27.9 GB total, ~20.4 GB used during run
- Dataset: `tests/dataset/20260423/annotations_2026-04-23_11-24_coco_with_text.json`,
  119 portrait crops (aspect < 0.5)
- max_tokens: 512 (deviates from EXP-25's 32 — see §4)
- Prompt: identical to
  [tests/qwen_portrait.py:35-41](../tests/qwen_portrait.py#L35-L41)

Re-run: `python tests/qwen_local_spike2.py` (assumes `ollama pull
qwen3-vl:8b` already done; cache at `tests/results/qwen_local_cache/`
short-circuits previously-seen crops).

---

## 12. Follow-up — small-VLM probe ladder

After §1-11 closed the qwen3-vl:8b probe negative, the load-bearing
question shifted: was it the *thinking-mode* (model class) or the
*q4_K_M quant* that broke accuracy? The clean test is to spike
**non-thinking small VLMs** at a size ladder (smallest first; abort
each rung that doesn't clear a 5-crop sanity gate).

The spike scripts ([tests/qwen_local_spike.py](../tests/qwen_local_spike.py))
are model-agnostic via the `QWEN_LOCAL_MODEL` env var; the ladder
runs reuse the same prompt, dataset, ann_ids, processor, and cache
infrastructure as the qwen3-vl probe. The format gate
(`^(JBHZ\d{6}|JBHU\d{6}|R\d{5})$`) lives downstream of the spike, so
spike output is raw model text and edit-distance is computed honestly
even when the gate would have rejected.

### 12.1 — moondream:1.8b-v2-q4_0 — REJECTED (2026-04-26)

Pulled `moondream:1.8b-v2-q4_0` (1.7 GB). Architecture: Phi-2 1B base
+ 454M CLIP projector. No thinking mode. Spike1 on the same 5
ann_ids as EXP-25 spike1 + qwen3-vl spike1.

| ann | gt           | local_raw                     | OR_raw       | l_ed | ms     |
|----:|--------------|-------------------------------|--------------|-----:|-------:|
|  17 | JBHU235644   | THEIMAGEFEATURESAGRA…         | JBHU235644   |  233 | 34,245 |
|  73 | JBHZ676208   | THEIMAGEFEATURESATAL…         | JBHZ676208   |  148 |    458 |
| 165 | JBHZ672066   | THEIMAGEFEATURESATAL…         | JBHZ672066   |  208 |    539 |
| 255 | JBHZ660789   | THEIMAGEFEATURESALON…         | UNKNOWN      |  316 |    648 |
| 356 | JBHZ672637   | THEIMAGESHOWSATALLME…         | JBHZ672637   |  177 |    452 |

**LOCAL: 0/5 EXACT, 0/5 NEAR(≤2)**, avg 7.3 s/crop (skewed by 34 s
cold-load on ann 17; warm crops were 450-650 ms each — moondream is
fast, just wrong).

**Failure mode:** moondream returns image *captions*, not OCR. Across
all 5 crops it produced narrative descriptions ("a tall, thin pole
with black writing"). Probe of 4 alternative prompt phrasings on a
single crop:

| prompt                                                    | output                                                       |
|-----------------------------------------------------------|--------------------------------------------------------------|
| "Read the text."                                          | "…The numbers '67' and '8' are clearly visible on the pole…" |
| "What text is in this image?"                             | "…the numbers 'JHZ' written on it…"                          |
| "Read all text in the image. Return only the text."       | "…The numbers '67' and '8' are clearly visible…"             |
| "OCR this image. Return only the characters you read."    | "…The numbers '67' and '68' are clearly visible…"            |

Every prompt produced narrative wrapper; the model glimpses fragments
("JHZ", "67", "8") but cannot follow a "return only" instruction and
cannot assemble the full ID. Verdict: **moondream is undersized for
both instruction-following and dense-text recognition** — the 1B Phi-2
base is too small to suppress its visual-chat training prior.

`ollama rm moondream:1.8b-v2-q4_0` to reclaim 1.7 GB. Cache dir
unchanged (sha256 cache key includes the model_id, so the moondream
entries are inert and can be left in place).

### 12.2 — granite3.2-vision:2b — REJECTED (2026-04-26)

Pulled `granite3.2-vision:2b` (2.4 GB, IBM Granite Vision 3.2,
Granite 2.5B base + 442M CLIP, q4_K_M, 16K context, no thinking,
explicitly OCR/document-tuned per the model card).

**Initial spike with the EXP-25 prompt: 0/5 EXACT, all crops returned
refusals** (`UNFORTUNATELY, I AM UNABLE...`, `UNFORTUNATELY, I CAN'T...`).
The long instruction triggers a "describe the image / no visible
content" refusal pattern, not OCR. Probed 6 alternative prompts on
ann 73 (JBHZ676208) — simple prompts unlock OCR behaviour:

| prompt                                            | output                  |
|---------------------------------------------------|-------------------------|
| EXP-25 detailed prompt                            | "Unfortunately, I am unable to provide…" |
| "Read the text in this image."                    | `J B H Z 6 2 0 8`       |
| "What characters are written on this sign?"       | "unanswerable"          |
| "Transcribe the text."                            | `J B H Z 6 2 0 8`       |
| "Extract the text from this image."               | `J B H Z 6 2 0 8`       |
| "List all characters visible…"                    | "unanswerable"          |

Re-ran spike1 with `SPIKE_PROMPT="Read the text in this image."` (and
again with `"Transcribe the text."` to control for prompt sensitivity):

| ann | gt           | "Read…" output                | "Transcribe…" output           | l_ed | OR_raw       |
|----:|--------------|-------------------------------|--------------------------------|-----:|--------------|
|  17 | JBHU235644   | `JOHNSON`                     | `JOHNSON`                      |    8 | JBHU235644   |
|  73 | JBHZ676208   | `JBHZ6208`                    | `JBHZ6208`                     |    2 | JBHZ676208   |
| 165 | JBHZ672066   | `UNANSWERABLE`                | `THEIMAGEPROVIDEDAPPE…`        |12-199| JBHZ672066   |
| 255 | JBHZ660789   | `I'MSORRY,BUTICAN'TAS…`       | `I'MSORRY,BUTICAN'TAS…`        | 74-80| UNKNOWN      |
| 356 | JBHZ672637   | `58`                          | `58`                           |   10 | JBHZ672637   |

**LOCAL: 0/5 EXACT, 1/5 NEAR (≤ed-2)** under both simple prompts. The
1 NEAR (ann 73, edit-2) and the same `JOHNSON`/`58` hallucinations
appear under both — failure mix is **crop-dependent, not
prompt-dependent**. Latency excellent: ~1.0 s/crop average (4× faster
than qwen3-vl:8b on the same hardware).

The technical pass against the spike rule (≥1/5 NEAR) is real but
qualitatively weak: 1 partial read out of 5, with hallucinations
(`JOHNSON`, `58`) and refusals on the rest. Even the one NEAR is
truncated (`JBHZ6208` instead of `JBHZ676208`, missing two interior
digits). Extrapolating to 119 crops, an exact-match rate above ~10%
seems implausible — well below EXP-25's 53% bar and well below
EXP-25's 47/92 JBHZ floor.

**Verdict:** the 2B Granite model handles clean printed documents
reasonably (per its training distribution) but cannot read embossed
small-text stacked-vertical trailer plates. OCR-tuning at this scale
is a different problem from OCR on this image distribution.
**Skipping the 119-crop widen** — the 5-crop signal is unambiguous.

`ollama rm granite3.2-vision:2b` to reclaim 2.4 GB.

### 12.3 — qwen2.5vl:3b — **VIABLE** (2026-04-26)

Pulled `qwen2.5vl:3b` (3.2 GB, Qwen 2.5-VL 3B base + CLIP, q4_K_M, 128K
context, **no thinking**). The Apache-licensed predecessor of qwen3-vl,
< half the size of the qwen3-vl:8b probe in §1-11.

**Spike1 (5 crops, EXP-25 prompt unchanged):** 3/5 EXACT, 3/5 NEAR(≤2).
Avg 19.3 s/crop — but that's skewed by an 80 s cold-load on ann 17;
warm crops were 4 s each. Same 3 EXACTs as OpenRouter on the same
crops, same UNKNOWN on ann 255. Only difference: ann 165 returned
`672066` (digits without prefix) where OpenRouter got `JBHZ672066`.

**Spike2 (full 119 portrait crops):**

| metric                        | EXP-25 OpenRouter Qwen3-VL-8B | qwen2.5vl:3b local |
|-------------------------------|-------------------------------|-----------------------------|
| portrait EXACT                | 63/119 (53%)                  | **60/119 (50.4%)**          |
| **JBHZ EXACT**                | **47/92 (51%)**               | **50/92 (54%) ⬆**           |
| JBHU EXACT                    | _(not broken out)_            | 6/8 (75%)                   |
| NUMERIC EXACT                 | _(not broken out)_            | 4/17 (24%)                  |
| OTHER EXACT                   | _(not broken out)_            | 1/7 (14%)                   |
| median ms / crop              | ~1,144 ms                     | 4,109 ms                    |
| p95 ms / crop                 | _(unknown)_                   | 7,996 ms                    |
| wall, 119 crops               | ~150 s                        | 594 s (~10 min)             |
| projected wall, 419 cascade   | ~140 s                        | ~450 s (~7.5 min)           |
| cost per 419 run              | ~$0.05                        | $0                          |
| hardware                      | n/a (vendor)                  | local Ollama, NVIDIA RTX    |

**Verdict:** clears every gate. **Local wins on the load-bearing JBHZ
sub-bucket** (50/92 vs 47/92) at zero recurring cost; ~3.5× slower per
crop than OpenRouter but the projected 419-cascade wall is ~7.5 min,
well under the 30-min threshold. Result file:
[tests/results/qwen_local_spike_qwen25vl3b_20260423.json](../tests/results/qwen_local_spike_qwen25vl3b_20260423.json).

**Per-crop diff vs OpenRouter:**

- Both correct: 44/119
- Both wrong:   40/119
- **Unique local wins (16):** all 16 are crops where OpenRouter
  returned `UNKNOWN` (or one near-miss: `JBHN672066` vs GT
  `JBHZ672066`). Local has 13 of these wins on the JBHZ alpha-prefix
  sub-bucket — exactly where the OpenRouter cascade left value on the
  table — plus 3 numeric crops (`23283`, `666613`, `40389`).
- **Unique OR wins (19):** local UNKNOWN'd on 9 NUMERIC crops where
  OpenRouter read them, plus 2 `7322039P` (OTHER format), plus 7
  ALPHA4 near-misses (1-char errors or truncations like
  `JBHZ67263` vs `JBHZ672637`, `JBHZ667A96` vs `JBHZ667496`).

**Implication for production wiring:** the two models are *partially
complementary*. A local-first → OpenRouter-fallback cascade gated on
UNKNOWN would theoretically reach 79/119 (66%) on this dataset. Pure
local replacement at 60/119 keeps cost at $0 but loses the numeric
sub-bucket vs OpenRouter. The decision lives in a follow-up
production-wiring plan; **this experiment's recommendation is to
proceed to that plan**.

### 12.4 — Ladder summary

| rung |  model                   | params  | EXACT@5 | EXACT@119 | latency      | verdict             |
|-----:|--------------------------|--------:|--------:|----------:|-------------:|---------------------|
|   1  | qwen3-vl:8b (q4)         |    8.8B |   0/5† / 3/5‡ |    24/80 (30%, killed) | 14-50 s    | REJECT (thinking)   |
|  2.1 | moondream:1.8b-v2-q4_0   |    1.8B |    0/5  |        — | 0.5 s        | REJECT (no instr.)  |
|  2.2 | granite3.2-vision:2b     |    2.5B |    0/5* |        — | 1.0 s        | REJECT (refusals)   |
|  2.3 | **qwen2.5vl:3b**         |    3.8B |    3/5  | **60/119 (50%)** | 4 s | **VIABLE**          |

† at max_tokens=32. ‡ at max_tokens=512. * with simple prompt; refused
on EXP-25 prompt.

**Chosen model:** `qwen2.5vl:3b` (Apache 2.0, ~3.2 GB, ~4 s/crop on
the dev RTX, accuracy parity with EXP-25 OpenRouter on the
load-bearing JBHZ sub-bucket).
