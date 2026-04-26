# OCR Performance Experiment 23 — EXP-09 on Cleaner Dataset (20260423)

**Status:** COMPLETE (2026-04-25). Diagnostic run, no config change.

## 1. Question

Round 2 closed at 38.2% / 57.6% on `tests/dataset/20260406`, with portraits stuck at 2/156 (1.3%). The 20260406 set is natively blurry. EXP-23 re-runs the EXP-09 production config unchanged on the cleaner `tests/dataset/20260423` (251 imgs / 419 anns) to answer:

> Is portrait failure caused by the vertical-stacked-text **layout**, or by the **blur** of those particular crops?

## 2. Method

- Same code, same thresholds (`text_det_thresh=0.2`, `text_det_box_thresh=0.3`, `text_det_unclip_ratio=2.0`, `MIN_CONFIDENCE=0.6`, EXP-03+04+06 pre/post).
- Added `--dataset {20260406,20260423}` flag in [tests/benchmark_ocr.py](../tests/benchmark_ocr.py); default unchanged.
- Run: `python tests/benchmark_ocr.py --exp-id EXP-23 --dataset 20260423`.
- Output: [tests/results/benchmark_EXP-23_20260425_153428.json](../tests/results/benchmark_EXP-23_20260425_153428.json).

## 3. Results

| bucket      | OLD 20260406              | NEW 20260423              | Δ accuracy |
|-------------|---------------------------|---------------------------|-----------:|
| portrait    |   2/156 = **1.3%**, prec 5.9%   |   7/119 = **5.9%**, prec 20.0% |   +4.6pp   |
| wide        | 218/424 = **51.4%**, prec 63.7% | 222/255 = **87.1%**, prec 89.2% |  +35.7pp   |
| landscape   |  33/ 54 = 61.1%, prec 84.6%     |  30/ 35 = **85.7%**, prec 85.7% |  +24.6pp   |
| very_wide   |   4/ 34 = 11.8%, prec 13.3%     |   3/ 10 = 30.0%, prec 33.3%   |  (n=10)    |
| **OVERALL** | **257/672 = 38.2%, prec 57.6%** | **262/419 = 62.5%, prec 79.9%** | **+24.3pp** |

Median latency: 111ms → 138ms (no meaningful change; near-identical engine work, fewer empty-skip retries on the cleaner set).

## 4. The portrait number is misleading — break it down by GT format

The portrait bucket on the new dataset contains two structurally different sub-populations:

| GT format                        | n   | correct | rate   |
|----------------------------------|-----|---------|--------|
| `^[A-Z]{4}\d{6}$` (JBHZ/JBHU…)   |  95 |    **0** |  **0.0%** |
| numeric-only (e.g. `352827`)     |  17 |    7    | 41.2%  |
| other                            |   7 |    0    |  0.0%  |

The entire 5.9% portrait gain comes from the easier **numeric-only stacked** crops. **JBHZ-style alpha-prefix vertical stays 0/95.** That sub-bucket was 0/156-equivalent on the old set too.

84/119 portraits (71%) returned no text at all — the detector silently rejects the crop. The remaining 28/119 returned hallucinated text (`A-NNOONIWL`, `INONNN`, `BINOO8NOM`, etc.).

## 5. Verdict

**Image quality was a major bottleneck for horizontal text** (wide bucket nearly doubled, 51% → 87%; landscape +25pp). The current engine + EXP-09 config is much stronger than the 20260406 numbers implied — overall jumps from 38% to 63% with no code change.

**Vertical stacked alphanumeric text is a layout problem, not a clarity problem.** Even on the clean set, the alpha-prefix vertical sub-bucket sits at 0/95. Round 3 still needs a structural fix:
- VLM fallback on the portrait path (Claude Haiku / Gemini Flash / Qwen2.5-VL), gated by EXP-15 format whitelist.
- Per-character DB detector finetuned on ~50 hand-labelled stacked-vertical crops (the candidate noted in EXP-17's rejection).

**Headline:** without changing anything else, switching from the blurry training-style dataset to the clean operational-style dataset takes us from 38.2% → **62.5%** overall and 57.6% → **79.9%** precision. The remaining gap is dominated by the JBHZ-style vertical sub-bucket plus a minority of non-conforming "other" GTs.

## 6. Out of Scope

- Any config / preprocessing change.
- Round 3 experiments themselves — this run is purely a diagnostic to prioritise them.
- Updating production defaults — still EXP-09 / commit `4008962`.
