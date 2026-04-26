# OCR Performance Experiment 26 — GOT-OCR-2.0-hf on Portrait + Horizontal

**Status:** REJECTED (2026-04-26). Both gates failed in characterisation:
- **Portrait spike1: 0/5 EXACT, 0/5 NEAR(≤2)** — same JBHZ wall as
  Tesseract / TrOCR.
- **Wide spike2 on all 265 wides:** GOT 148/265 (55.8%) vs PaddleOCR
  225/265 (84.9%). Cascade (paddle→GOT fallback) only adds 6 rescues
  (+0.7 pp); 83 regressions show GOT-OCR is *correlated* with
  PaddleOCR, not complementary.

GOT-OCR is REJECTED for both portrait and wide deployment paths on
this dataset. Median latency 13.4 s/crop on CPU sealed the negative
ROI even before accuracy was characterised.

## 0. Spike1 result (5 portraits + 5 wides, 2026-04-26)

`.venv-got-ocr/Scripts/python.exe tests/got_ocr_spike.py`

GOT-OCR-2.0-hf loaded via `AutoModelForImageTextToText` (transformers
4.54.1, CPU). Model class `GotOcr2ForConditionalGeneration`, 580M
parameters. Generation: `max_new_tokens=64, do_sample=False`,
post-processing strips chat template prefix (`...assistant\n`) and
loop-dedupes immediate `XXX...` repetitions of 4-12 char prefixes.

**Portrait (5 same crops as EXP-19/EXP-25/EXP-27 spike1):**

| ann | gt           | raw                           | deduped              |  ed |    ms |
|----:|--------------|-------------------------------|----------------------|----:|------:|
|  17 | JBHU235644   | `BHU235644J20000…`            | `BHU235644J20000…`   |  29 | 13,788 |
|  73 | JBHZ676208   | `BHN6702088J88…`              | `BHN6702088J88…`     |  31 | 13,258 |
| 165 | JBHZ672066   | `670007670000E0…`             | `670007670000E0…`    |  29 | 13,440 |
| 255 | JBHZ660789   | `60047004JN40040JN40040…`     | `60047004JN40040JN…` |  49 | 13,408 |
| 356 | JBHZ672637   | `BHN67203377NN67203377…`      | `BHN67203377NN67203…`|  34 | 13,931 |

**PORTRAIT: 0/5 EXACT, 0/5 NEAR(≤2), avg 13,565 ms/crop.**

GOT-OCR cannot read stacked-vertical Latin trailer plates. The model
emits character fragments — `BHU235644J` on ann 17 is suggestive of a
*rotational* read of `JBHU235644` (J wraps to the end) — but the
"start anchor" is unstable across crops, and the recognition then
loops. Same wall as Tesseract (EXP-19) and TrOCR (EXP-20). The wider
training distribution didn't help.

**Wide (5 PaddleOCR-easy crops, ann_ids 3-7):**

| ann | gt     | raw                                       | deduped  | ed |    ms |
|----:|--------|-------------------------------------------|----------|---:|------:|
|   3 | 701901 | `7019017019017019017019017019…`           | `701901` |  0 | 13,335 |
|   4 | 702526 | `702526`                                  | `702526` |  0 | 13,193 |
|   5 | 09231  | `09231`                                   | `09231`  |  0 | 13,091 |
|   6 | 702830 | `702830`                                  | `702830` |  0 | 13,230 |
|   7 | 703548 | `7035487035487035487035487035487035487035` | `703548`|  0 | 13,120 |

**WIDE: 5/5 EXACT after loop-dedupe, avg 13,193 ms/crop.**

The 5 wides chosen are PaddleOCR-easy (baseline succeeds on them) —
this only confirms GOT-OCR ≥ PaddleOCR on **easy** wides, not on the
**33** wide crops where PaddleOCR currently fails. **That's the
load-bearing question for spike2.**

**Verdict (per §4 gates):**

- **Portrait gate** (≥ 1/5 NEAR AND median ≤ 30 s): **FAIL**. Drop the
  portrait cascade path for GOT-OCR. EXP-25's qwen cascade /
  EXP-27's local qwen2.5vl:3b remain the portrait answer.
- **Horizontal gate** (≥ 3/5 EXACT after dedupe): **PASS** at 5/5 on
  PaddleOCR-easy wides. Proceeding to spike2 — but the load-bearing
  question is whether GOT rescues the wides where PaddleOCR fails.

## 0.5 Spike2 wide — REJECTED (2026-04-26)

`.venv-got-ocr/Scripts/python.exe tests/got_ocr_spike2_wide.py`

Ran GOT-OCR-2.0 on **all 265 wide annotations** in
`tests/dataset/20260423` (aspect ratio ≥ 2.0, slightly more than the
255 the EXP-23 baseline aggregated due to a marginal aspect-ratio
boundary difference). Wall: ~60 min, median 13.4 s/crop. Cross-ref
against [tests/results/benchmark_EXP-23_20260425_153428.json](../tests/results/benchmark_EXP-23_20260425_153428.json)
for PaddleOCR pass/fail per ann_id.

Result file:
[tests/results/got_ocr_spike2_wide_20260423.json](../tests/results/got_ocr_spike2_wide_20260423.json).

**Headline:**

| metric                                    | count   | rate   |
|-------------------------------------------|--------:|-------:|
| GOT-OCR EXACT (full replacement)          | 148/265 | 55.8 % |
| PaddleOCR EXACT (baseline)                | 225/265 | 84.9 % |
| Cascade (paddle→GOT fallback on paddle-fail) | 231/265 | 87.2 % |

**Cross-tab vs PaddleOCR per crop:**

| outcome                                | count |
|----------------------------------------|------:|
| Both correct                           |  142  |
| Both wrong                             |   34  |
| **GOT rescues** (GOT ✓, paddle ✗)      |  **6** (15 % rescue rate of 40 paddle failures) |
| **GOT regressions** (GOT ✗, paddle ✓)  | **83** (37 % regression rate of 225 paddle successes) |

**Why both deployment paths fail:**

1. **Full replacement** (`--horizontal-engine got_ocr`): -77 crops
   net (148 vs 225). 30+ pp regression vs PaddleOCR. Catastrophic.
2. **Cascade fallback** (paddle first → GOT on paddle-fail): +6 wins,
   +0.7 pp on the 419-crop run, gated on a 13 s/crop CPU model that's
   now 100× slower than the paddle baseline (~110 ms). Negative ROI
   even with non-zero gain.

**Failure-mode analysis:** GOT-OCR is **correlated** with PaddleOCR,
not complementary. It reads cleanly-printed wides (where paddle also
reads them) and fails on the same hard wides (motion blur, partial
occlusion, embossing). Both engines have the same difficulty
distribution on this dataset; GOT-OCR has no rescue signal where it
matters.

**Decision:** REJECT both deployment paths. No `tests/got_ocr_processor.py`
production wrapper, no `--horizontal-engine` flag added to
[tests/benchmark_ocr.py](../tests/benchmark_ocr.py). Spike scripts
retained for reproducibility:

- [tests/got_ocr_spike.py](../tests/got_ocr_spike.py) — 5+5 pre-flight.
- [tests/got_ocr_spike2_wide.py](../tests/got_ocr_spike2_wide.py) — 265-crop
  cross-reference vs PaddleOCR baseline.

`.venv-got-ocr/` retained for any future GOT-OCR re-test (different
quant, fine-tune, etc.); HF model cache `~/.cache/huggingface/` keeps
the 1.14 GB checkpoint. To reclaim disk: `rm -rf .venv-got-ocr` and
`huggingface-cli delete-cache`.

**Cost of the experiment:**

- Venv setup + transformers/torch/accelerate (~1.5 GB on disk).
- Model checkpoint pull (1.14 GB, one-time).
- Spike1 (5+5) — 2 min wall.
- Spike2 (265 wides) — 60 min wall.
- Engineering time: ~1.5 hr total (including this write-up).

Worth it: a definitive empirical rejection on a 580 M end-to-end OCR
transformer that the team had open as a candidate for ~6 weeks.

---

## Original plan (retained for archive)

## Context

EXP-25 closed at **314/419 = 74.9%** on `tests/dataset/20260423`. Two
known gaps remain on the cleaner dataset:

- Portrait JBHZ alpha-prefix sub-bucket is at **47/92 (51%)** with the
  Qwen3-VL-8B OpenRouter cascade — the cascade is paid + network-bound,
  so a self-hosted alternative would be operationally cheaper.
- Wide bucket caps at **222/255 = 87.1%** with PaddleOCR — the
  strongest non-portrait accuracy we have, but 33 errors remain.

A colleague flagged
[stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
— 1.14 GB safetensors, ~580M-param end-to-end OCR transformer
(model SHA `d3017ef` on HF main as of 2026-04-26). Apache-style license,
upstreamed into `transformers>=4.49.0` as
`GotOcr2ForConditionalGeneration`. Pretrained on diverse text types
(printed, handwritten, formulas, charts, sheet music) which is a wider
training distribution than TrOCR-base-printed (rejected in EXP-20).

## 1. Goal

Probe two questions in one experiment:

1. **Portrait JBHZ:** does GOT-OCR read stacked-vertical Latin trailer
   IDs that PaddleOCR misses? Target ≥ 5 unique JBHZ EXACT wins vs
   PaddleOCR EXP-23 to justify wiring as a `--portrait-strategy got_ocr`
   cascade alternative to EXP-25's OpenRouter dependency.
2. **Wide bucket:** does GOT-OCR match or beat PaddleOCR on horizontal
   text? Target ≥ 87% EXACT on a 50-wide sample (matches PaddleOCR
   floor) to justify a full-replacement A/B test on the 255-wide bucket.

Either result on its own is wireable; both is the strongest case.

## 2. Hypothesis

GOT-OCR's training distribution covers diverse layouts (the model card
emphasises charts, formulas, music), which is a strictly wider corpus
than TrOCR's printed-text set. The end-to-end ViT encoder may handle
stacked vertical Latin where TrOCR (EXP-20) failed because it learned
broader spatial-token-order priors during pretraining.

Realistic prior: **~25% it materially helps on JBHZ, ~40% it matches
PaddleOCR on wide.** End-to-end OCR transformers without explicit
layout-aware detection have failed before (EXP-20 TrOCR, EXP-19
Tesseract); the wider GOT pretraining corpus is the only reason this
isn't a dupe of EXP-20.

## 3. Methodology

### Isolated venv

GOT-OCR-2.0 needs `transformers>=4.49.0`, `torch` (CPU wheels),
`huggingface_hub`, `accelerate`. The main venv pins `paddleocr==2.7.3`
/ `paddlepaddle==2.6.2`; mixing modern transformers risks numpy /
pillow conflicts (EXP-22 hit this with paddle v5). Mirror the
[.venv-paddle-v5](../.venv-paddle-v5/) isolation pattern:

- **`.venv-got-ocr/`** + **`requirements-got-ocr.txt`** (new) —
  transformers, torch CPU wheels, huggingface_hub, accelerate, pillow,
  requests. No paddle deps.
- Document install + activation steps inline below in §10.

### Inference config

- `AutoModel.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf",
  torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)`
- `AutoTokenizer.from_pretrained(...)`.
- Inference call: `model.chat(tokenizer, image_path, ocr_type='ocr')`
  — plain-text mode. The HF model card uses path-based input; pass the
  PIL crop via `tempfile.NamedTemporaryFile`. Verify in spike1 that the
  in-memory round-trip costs < 10 ms.
- No preprocessing in the spike (GOT does its own). If raw fails, try
  EXP-19's preprocess chain (≥ 256 px short side + Otsu + 20 px white
  border) as a single sanity variant.

### Validation gate

- **Portrait path:** strip → upper → drop whitespace → match
  `^(JBHZ\d{6}|JBHU\d{6}|R\d{5})$` (identical to
  [tests/qwen_portrait.py](../tests/qwen_portrait.py)). Run through
  EXP-15 `format_rescore` before gating. Format hit → `(text, 1.0)`.
  Format miss → fall through to standard PP-OCRv4.
- **Horizontal path:** no format gate (free-text output is the
  measurement). EXACT match required against GT.

## 4. Pre-flight verification (LOAD-BEARING — do this first)

Memory rule: verify load-bearing priors before building wiring on them.
EXP-19 spike caught a hypothesis that looked productive on first read;
same logic here.

1. `pip install -r requirements-got-ocr.txt` in `.venv-got-ocr/`.
   Verify `python -c "from transformers import AutoModel;
   AutoModel.from_pretrained('stepfun-ai/GOT-OCR-2.0-hf')"` succeeds
   (one-time ~1.14 GB pull from HF, ~30-60 s model load).
2. Write `tests/got_ocr_spike.py` — 5 portraits + 5 wides. Hard-coded
   ann_ids. Use the **same 5 portrait crops** EXP-19 used (ann ids
   17, 73, 165, 255, 356 from 20260423) so this is a literal
   apples-to-apples comparison. 5 wides: pick the EXP-23 wide bucket
   crops where PaddleOCR succeeded (a baseline gate — if GOT can't
   even match PaddleOCR on easy wides, kill).
3. **Decision rule:**
   - 0/5 portrait JBHZ correct AND median > 30 s/crop CPU → **abort
     EXP-26**, document, do not wire.
   - ≥ 1/5 portrait reads correctly OR within edit-2 → widen to
     119-portrait + 50-wide spike (`tests/got_ocr_spike2.py`).
   - 5/5 wides match GT → wide-bucket hypothesis still viable; widen
     wides too. < 3/5 wides match → wide A/B path is dead, narrow to
     portrait-only.
4. **Spike2 per-bucket gates:**
   - Portrait cascade: ≥ 5 unique JBHZ EXACT wins vs PaddleOCR EXP-23
     justifies wiring as `--portrait-strategy got_ocr`.
   - Horizontal A/B: ≥ 87% EXACT on the 50-wide sample (matches
     PaddleOCR floor) justifies a full-replacement A/B as
     `--horizontal-engine got_ocr`.
5. If both gates fail, close the experiment same way EXP-19 closed:
   document, retain spike scripts, no wiring.

## 5. Implementation Steps (only if a pre-flight gate passes)

1. **`tests/got_ocr_processor.py`** (new) — modelled on
   [tests/qwen_portrait.py](../tests/qwen_portrait.py):
   - Singleton `GotOcrProcessor` with `__new__` cache.
   - Lazy model + tokenizer load on first `process_image` call. Log
     load time once.
   - On-disk cache at `tests/results/got_ocr_cache/`, keyed by
     `sha256(png + model_revision + ocr_type)`.
   - `process_image(pil) -> (text, conf)` — same contract as
     OcrProcessor / QwenPortraitProcessor.
   - `stats()` (cache hits/misses, format-gate hits/misses, total
     inference seconds).
2. **[tests/benchmark_ocr.py:103](../tests/benchmark_ocr.py#L103)** —
   extend `--portrait-strategy` choices to add `got_ocr`. Cascade
   insertion mirrors the EXP-25 pattern at
   [tests/benchmark_ocr.py:518](../tests/benchmark_ocr.py#L518): runs
   only when `not ocr_text` and `processed_crop.height > 2 *
   processed_crop.width`.
3. **(Conditional, only if wide gate passes)** Add
   `--horizontal-engine {paddle,got_ocr}` flag, default `paddle`. When
   `got_ocr`, replace PaddleOCR call entirely on `aspect ≥ 0.5` crops.
4. Run order:
   - Portrait-only: `--exp-id EXP-26-got-portrait --dataset 20260423
     --only-bucket portrait --portrait-strategy got_ocr`. n=119.
   - (Conditional) Wide-only: `--exp-id EXP-26-got-wide --dataset
     20260423 --only-bucket wide --horizontal-engine got_ocr`. n=255.
   - Full 419 with winning config(s) to confirm guardrails.

## 6. Critical Files

- **`.venv-got-ocr/`** + **`requirements-got-ocr.txt`** (new) —
  isolated env. transformers version pin in the requirements file.
- **`tests/got_ocr_spike.py`** (new, throwaway) — 5+5 pre-flight.
- **`tests/got_ocr_spike2.py`** (new, throwaway) — 119+50 widened.
- **`tests/got_ocr_processor.py`** (new) — `GotOcrProcessor`,
  mirrors [tests/qwen_portrait.py](../tests/qwen_portrait.py).
- **[tests/benchmark_ocr.py](../tests/benchmark_ocr.py)** — extend
  `--portrait-strategy`, conditionally add `--horizontal-engine`.
- `docs/ocr-performance-experiment-26.md` — flip PLANNED → RUN with
  results.

### Reused, not rewritten

- Format gate from
  [tests/qwen_portrait.py:46-47](../tests/qwen_portrait.py#L46).
- Cache pattern from
  [tests/qwen_portrait.py:94-113](../tests/qwen_portrait.py#L94).
- Spike row schema + summary stats from
  [tests/qwen_spike2.py](../tests/qwen_spike2.py) — drop in `got_raw`
  for `qwen_raw`.
- EXP-25 cascade insertion at
  [tests/benchmark_ocr.py:518](../tests/benchmark_ocr.py#L518) — copy
  pattern, swap processor.

## 7. Verification

- **Spike1 sanity:** model loads < 60 s, non-empty output on ≥ 3/5
  wides. Anything less and the model isn't engaging.
- **Portrait JBHZ gate:** spike2 ≥ 5 JBHZ EXACT (Qwen3-VL-8B is at
  47/92 for reference; GOT need not match, just clear PaddleOCR's 0).
- **Horizontal gate:** spike2 wide bucket EXACT rate ≥ 87% on the
  50-wide sample.
- **Latency budget:** median CPU inference ≤ 30 s/crop in spike1,
  ≤ 10 s/crop median across spike2.
- **Primary metric (post-wiring):** JBHZ alpha4+6digit portrait
  sub-bucket EXACT on 20260423 — currently 0/92 with PaddleOCR alone,
  47/92 with EXP-25 cascade. Target ≥ 5/92 to justify GOT cascade as
  alternative path.
- **Guardrail 1:** non-target buckets unchanged when only one strategy
  is enabled.
- **Guardrail 2:** precision unchanged — strict format gate + EXP-15
  rescore on the portrait path.
- **Reproducibility:** pin `transformers` version in
  `requirements-got-ocr.txt`, model SHA `d3017ef`, dataset annotation
  file hash in §10.

## 8. Risks

- **Layout assumption mismatch.** End-to-end OCR transformers without
  explicit layout-aware detection failed twice (EXP-19 Tesseract,
  EXP-20 TrOCR). GOT's wider training distribution is the only reason
  to expect a different outcome — empirically untested on Latin
  stacked-vertical.
- **CPU latency.** TrOCR was 989 ms/crop; GOT is larger (~580M vs
  ~334M) and may be slower. If > 30 s/crop in spike1, treat as a
  GPU-only path and re-evaluate deployment cost.
- **Dependency conflicts.** transformers + torch share a surface with
  paddleocr's numpy / pillow pins. Isolated venv is the defence.
- **Sample-size noise.** 5+5 spike is cheap by design; widen before
  trusting either signal (EXP-19 lesson — 1/20 looked like noise but
  4/119 was the real picture).
- **HF model card uses path-only API.** If `model.chat` requires a
  filesystem path and the tempfile round-trip costs more than ~50 ms,
  the latency budget tightens. Spike1 will measure.

## 9. Out of Scope

- Production wiring (`app/ocr_processor.py`). Trigger is a positive
  419-crop benchmark; that's a separate plan with deployment-image
  changes (transformers, torch, GOT model checkpoint baked in).
- GOT-OCR's other modes (`format`, `fine-grained`, `multi-crop`).
  Plain `ocr_type='ocr'` only — trailer IDs are flat strings.
- Fine-tuning GOT-OCR on stacked-vertical crops. Spike answers
  whether out-of-the-box GOT recognises these glyphs at all; FT is a
  much larger separate plan.
- GPU acceleration. CPU is the realistic deployment target.
- Cross-engine voting (GOT + PaddleOCR + Qwen). Premature.

## 10. Cost estimate + reproducibility

- Venv setup + spike1: ~45 min (~1.14 GB HF pull + 5+5 crops).
- Spike2 (119+50): ~1-2 hr depending on CPU latency.
- Wiring + 419-crop benchmark + write-up: ~half day if a gate passes.

Total worst case ~half-day, best case (early abort) ~45 min.

**Reproducibility checklist** (filled in at flip-from-PLANNED time):
- `transformers` version: _(pin)_
- `torch` version + index URL: _(pin)_
- Model SHA: `d3017ef` (verify at run-time)
- Dataset annotation file hash:
  `sha256(tests/dataset/20260423/annotations_2026-04-23_11-24_coco_with_text.json)`
- CPU model: `python -c "import platform; print(platform.processor())"`
