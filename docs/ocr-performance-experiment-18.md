# OCR Performance Experiment 18 — VLM Fallback for Portrait Crops

**Status:** RUN — **Gemini 2.5 Flash REJECTED** (0/15 on portraits with upscaling;
0/119 on first pass without). Pipeline plumbing verified working via wide-bucket
calibration run. Experiment continues: next candidates are open-source self-hosted
(Qwen2.5-VL, PaddleOCR-VL) and stronger hosted APIs (Claude Haiku 4.5, Gemini 2.5
Pro). See §9–10 below.

## 1. Goal

Route portrait-bucket crops (aspect < 0.5) to a vision-language model with a
strict format-constrained prompt. Accept only outputs matching the trailer-ID
whitelist. Measure the accuracy ceiling a commercial VLM can achieve on this
subset, and decide whether the cost / latency trade-off is acceptable for
production.

Target: **portrait bucket ≥ 120/156 (~77%)**. Combined with EXP-09+15 on the
other buckets, overall should reach **≥ 55%** exact match.

## 2. Hypothesis

Stacked vertical trailer IDs with upright characters are a trivial perception
task for a modern VLM (Claude Haiku 4.5, Gemini 2.5 Flash, GPT-4o-mini). The
models have seen arbitrary text layouts during pre-training and don't need the
horizontal-sequence inductive bias that breaks PP-OCRv4 on this bucket.

Portrait volume is small (156/672 annotations, ~23%). Per-request latency of
500–2000 ms is acceptable if the alternative is shipping a product that can't
read portrait IDs.

## 3. Methodology

### Routing

- Aspect < 0.5 → VLM path.
- Aspect ≥ 0.5 → existing EXP-09+15 path (unchanged).

### VLM call

- **Candidate models:** `claude-haiku-4-5-20251001`, `gemini-2.5-flash`,
  `gpt-4o-mini`. Benchmark all three; pick winner on accuracy × cost × latency.
- **Prompt (draft):**
  > *"The image shows a trailer ID plate. Characters are upright and stacked
  > vertically — read them top-to-bottom. Valid formats: `JBHZ` + 6 digits,
  > `JBHU` + 6 digits, or `R` + 5 digits. Return only the ID string with no
  > punctuation or whitespace. If you cannot read it with confidence, return
  > exactly `UNKNOWN`."*
- **Image encoding:** send the raw portrait crop, optionally upscaled to a
  min 512-px long side.
- **Temperature:** 0. **Max tokens:** 20.

### Validation gate

Reject any response that does not match `^(JBHZ\d{6}|JBHU\d{6}|R\d{5}|UNKNOWN)$`.
On `UNKNOWN` or a rejected response, fall back to the current portrait path.

### Cost / latency accounting

Measure per-call wall clock, input/output tokens, and per-crop USD. Total cost
for a 156-crop run is expected to be **< $0.50** on any of the three candidates.

## 4. Critical Files

- [app/ocr_processor.py](../app/ocr_processor.py) — no change during benchmark.
- [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — add
  `--portrait-strategy vlm --vlm-provider {anthropic,google,openai} --vlm-model <id>`
  flag set. Inject API call in portrait branch.
- `tests/vlm_portrait.py` (new) — provider-agnostic `read_portrait(image, model) -> (text, note, cost_usd, latency_ms)` wrapper.
- [tests/results/](../tests/results/) — cache VLM responses by crop hash to make re-runs free.

## 5. Implementation Steps

1. Implement `tests/vlm_portrait.py` with an API-key-driven multi-provider client.
   Cache responses on disk (JSON keyed by SHA256 of crop bytes) so re-runs of
   the benchmark don't re-bill.
2. Wire `--portrait-strategy vlm` into the benchmark loop.
3. Run the 156-crop portrait-only subset against each candidate model. Tabulate
   accuracy, latency, cost.
4. Run full 672-annotation benchmark with the winning model.
5. Post-mortem: portrait bucket delta, failure-mode breakdown (did the VLM
   misread, refuse, or hallucinate?), total cost, latency distribution, and
   per-production-request economics at projected daily volume.

## 6. Verification

- **Primary:** portrait bucket accuracy. Target ≥ 120/156.
- **Guardrail 1:** non-portrait buckets unchanged (VLM never runs on them).
- **Guardrail 2:** precision must not drop — the format gate rejects malformed
  responses, so any wrong VLM answer only lands in the output if it matches a
  real format pattern. That's a real risk: track `harmful` cases explicitly.
- **Cost check:** per-production-request cost must be tractable at expected
  daily volume (needs stakeholder input on volume).

## 7. Risks

- **Format-matching hallucinations:** the VLM could confidently return a
  well-formed but wrong ID. Unlike local OCR errors, these can't be caught by
  a downstream format check — they already pass it. Mitigation: confidence
  proxy via self-consistency (call twice, require match) or require the VLM
  to also return a confidence flag in a structured response.
- **Privacy / egress:** portrait crops are sent to a third-party API. Confirm
  this is acceptable before running against production data.
- **Vendor lock-in / availability:** mitigate by benchmarking 3 providers.
- **Latency:** 500–2000 ms per call. If the production SLA is sub-second,
  portrait path needs async handling.

## 8. Out of Scope

- Self-hosted VLMs (Qwen-VL, LLaVA, Idefics) — separate experiment if cost or
  privacy rules out hosted APIs.
- VLM for non-portrait buckets — current local OCR already works there.
- Fine-tuning a hosted VLM.

## 9. Results — Gemini 2.5 Flash (executed 2026-04-24 → 2026-04-25)

### Scope 1 — full benchmark, no upscaling

Command:
```
python tests/benchmark_ocr.py \
  --exp-id EXP-18-portrait-gemini-flash \
  --portrait-strategy vlm --vlm-model gemini-2.5-flash --only-bucket portrait
```

Result (partial, 119/156 processed before hitting free-tier RPD cap): **0/119
correct.** Every response was format-valid (`JBHZ` + 6 digits) but fabricated —
no two answers alike. The model pattern-matched to the prompt's format spec
without reading pixels.

### Scope 2 — 15-crop sanity with 768px upscaling

Added Lanczos upscale to a min 768-px long side (matches Gemini's tile size),
cleared the stale cache, and re-ran via `tests/vlm_sanity.py --n 15`:

- **0/15 correct.**
- 12/15 returned `UNKNOWN` / empty (format gate working).
- 3/15 returned format-valid fabrications.

Upscaling did not help. The model shifts from "confidently fabricate" to "abstain",
but it still cannot read the stacked-vertical glyphs.

### Scope 3 — plumbing calibration on wide-bucket crops

Hypothesis: 0/15 is so bad it could indicate a pipeline bug (encoding, prompt,
gate). Ran `tests/vlm_sanity_nonportrait.py` on 5 wide-bucket crops with a
generic OCR prompt (no format gate):

| # | Orig size | GT | Gemini raw | Outcome |
|---|---|---|---|---|
| 1 | 205x97 | `ATLS03` | `'ATIS03'` | L↔I confusion |
| 2 | 205x97 | `ATLS03` | `'ATIS03'` | L↔I confusion |
| 3 | 147x47 | `R44045` | `'R4045'` | missing duplicated `4` |
| 4 | 147x53 | `13208` | `'13208'` | ✔ exact |
| 5 | 123x56 | `1393` | `'1393'` | ✔ exact |

**2/5 exact, 5/5 legibly reading pixels.** Pipeline is fine. Portrait failure
is model-specific, not a setup bug.

### Verdict (Gemini 2.5 Flash)

**REJECTED for portrait.** Not a plumbing issue — the model catastrophically
fails at the stacked-vertical layout specifically. Also shows ordinary OCR
weaknesses on small horizontal text (L↔I confusion, character elision).

Cost: **$0** (free tier). Quota: daily limit exhausted on two separate API keys
during testing.

## 10. Next steps — broader scope

Stacked vertical trailer text is a known subdomain of OCR (license plates,
containers, industrial asset tags, CJK newspaper scan). It is not a VLM-only
problem. The follow-up experiment will evaluate a broader shortlist than the
original plan's three hosted VLMs — specifically including specialized
vertical-text and scene-text recognisers. Shortlist to be captured in EXP-22.
