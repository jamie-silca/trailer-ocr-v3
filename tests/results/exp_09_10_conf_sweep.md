# Confidence Threshold Sweep — EXP-09-10

Input: `benchmark_EXP-09-10_20260420_180525.json`

Parity check: reported exact_match_pct=38.39 precision_pct=56.95 text_returned=453 | recomputed exact_match_pct=38.39 precision_pct=56.95 text_returned=453

## Summary

| Min Conf | Text Returned | Correct | Exact Match % | Precision % | Δ ExactMatch | Δ Precision |
|---|---|---|---|---|---|---|
| ungated | 453 (67.4%) | 258 | 38.39% | 56.95% | +0.00pp | +0.00pp |
| ≥0.50 | 453 (67.4%) | 258 | 38.39% | 56.95% | +0.00pp | +0.00pp |
| ≥0.60 | 418 (62.2%) | 257 | 38.24% | 61.48% | -0.15pp | +4.53pp |
| ≥0.70 | 391 (58.2%) | 256 | 38.10% | 65.47% | -0.30pp | +8.52pp |
| ≥0.80 | 351 (52.2%) | 251 | 37.35% | 71.51% | -1.04pp | +14.56pp |
| ≥0.90 | 297 (44.2%) | 234 | 34.82% | 78.79% | -3.57pp | +21.83pp |

## Per-aspect-ratio bucket (correct / total)

| Bucket | ungated | ≥0.50 | ≥0.60 | ≥0.70 | ≥0.80 | ≥0.90 |
|---|---|---|---|---|---|---|
| landscape | 33/54 (61.1%) | 33/54 (61.1%) | 33/54 (61.1%) | 33/54 (61.1%) | 33/54 (61.1%) | 33/54 (61.1%) |
| near_square | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) |
| portrait | 2/156 (1.3%) | 2/156 (1.3%) | 2/156 (1.3%) | 2/156 (1.3%) | 2/156 (1.3%) | 2/156 (1.3%) |
| very_wide | 4/34 (11.8%) | 4/34 (11.8%) | 4/34 (11.8%) | 4/34 (11.8%) | 4/34 (11.8%) | 3/34 (8.8%) |
| wide | 219/424 (51.7%) | 219/424 (51.7%) | 218/424 (51.4%) | 217/424 (51.2%) | 212/424 (50.0%) | 196/424 (46.2%) |

## Verdicts

- **ungated**: Baseline — no gating.
- **≥0.50**: Δ exact match +0.00pp, Δ precision +0.00pp — dropped 0 low-conf results.
- **≥0.60**: Δ exact match -0.15pp, Δ precision +4.53pp — dropped 35 low-conf results.
- **≥0.70**: Δ exact match -0.30pp, Δ precision +8.52pp — dropped 62 low-conf results.
- **≥0.80**: Δ exact match -1.04pp, Δ precision +14.56pp — dropped 102 low-conf results.
- **≥0.90**: Δ exact match -3.57pp, Δ precision +21.83pp — dropped 156 low-conf results.
