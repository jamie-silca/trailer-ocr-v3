# OCR Performance Experiment 14 — Text-Space Temporal Voting

## 1. Goal

EXP-11 proved confidence-weighted voting across ground-truth-grouped frames
adds **+9.9pp** on multi-frame trailers (38.2% → 48.1%). EXP-12 proved pixel-
space IoU tracking is infeasible on this moving-drone dataset (cross-frame
IoU ~= 0). The EXP-14 plan: skip geometric tracking and instead cluster
detections by **OCR-string edit distance** within a sliding window of
consecutive frames, then vote.

Plan target: **+3–6pp overall exact-match**, no precision regression.

---

## 2. Methodology

Post-hoc analysis over [benchmark_EXP-15_20260424_162508.json](../tests/results/benchmark_EXP-15_20260424_162508.json)
(current best: 259/672 = 38.5%, precision 62.0%). No re-OCR — pure string
post-processing.

**Procedure** ([tests/text_space_voter.py](../tests/text_space_voter.py)):

1. Parse frame seq number from `DJI_<ts>_<seq>_V.jpeg`. Sort annotations by seq.
2. Slide a window of W consecutive seqs. Collect all detections inside.
3. Agglomerative-cluster by: `edit_distance(text_i, text_j) <= max_edit` **AND**
   `|aspect_ratio_i - aspect_ratio_j| <= ar_tol`.
4. Confidence-weighted vote per cluster. Tie-break: prefer candidates matching
   `^JBHZ\d{6}$ | ^JBHU\d{6}$ | ^R\d{5}$`, then highest single-detection conf.
5. Rewrite every cluster member's text with the winner.

Optional gates tested: `--min-agreement k` (winner needs ≥k distinct voters),
`--format-only` (only vote in clusters containing at least one known-format
match).

---

## 3. Results

### Parameter sweep (no gating, as originally planned)

| W | max_edit | Touched | Helpful | Harmful | Neutral | Δ correct | Δ precision |
|---|---|---|---|---|---|---|---|
| 3 | 1 | 46 | 10 | 9 | 1 | **+1 (+0.1pp)** | +0.2pp |
| 3 | 2 | 90 | 12 | 20 | 8 | **−8 (−1.2pp)** | −1.9pp |
| 5 | 1 | 47 | 10 | 10 | 1 | 0 | 0 |
| 5 | 2 | 102 | 12 | 26 | 8 | **−14 (−2.1pp)** | −3.3pp |
| 7 | 1 | 48 | 10 | 11 | 1 | −1 | −0.2pp |
| 7 | 2 | 116 | 12 | 32 | 8 | **−20 (−3.0pp)** | −4.8pp |

**The plan's default (W=5, max_edit=2) regresses −2.1pp.** The wider the window
and looser the edit-distance threshold, the worse it gets — harmful rewrites
scale much faster than helpful ones.

### Gated variants

| Variant | Touched | Helpful | Harmful | Δ correct | Δ precision |
|---|---|---|---|---|---|
| W=3, E=1, `--min-agreement 2` | 16 | 1 | 4 | −3 (−0.4pp) | −0.7pp |
| W=3, E=2, `--format-only --min-agreement 2` | 3 | 0 | 1 | −1 | −0.2pp |
| **W=3, E=1, `--format-only`** | **10** | **4** | **1** | **+3 (+0.4pp)** | **+0.7pp** |
| W=2, E=1 (tightest window, no gate) | 44 | 10 | 9 | +1 | +0.2pp |

**Best config:** `--window 3 --max-edit 1 --format-only` →
**262/672 = 39.0%, precision 62.7%** (vs EXP-15 baseline 38.5% / 62.0%).

### Best-config voting log (10 clusters fired)

```
R50421 vs R90421 -> R90421      (gt R90421 — helpful)
R47376 vs R42376 -> R42376      (gt R42376 — helpful)
R50322 vs R50622 -> R50622      (gt R50622 — helpful)
R45354 vs R45384 -> R45384      (gt R45384 — helpful, 2 clusters)
R84524 vs R54524 -> R84524      (gt R54524 — HARMFUL, 2 clusters)
```

Four distinct trailer-ID corrections. One harmful rewrite (`R54524 → R84524`)
where the confidence-weighted vote picked the wrong candidate because the
minority reading had higher confidence.

### Per-bucket impact (best config)

All gains in `wide`. Other buckets untouched because voting only fires where
OCR already returned format-matching text.

| Bucket | Baseline | EXP-14 | Δ |
|---|---|---|---|
| wide | 220/424 | 223/424 | +3 |
| landscape, portrait, very_wide, near_square | — | — | 0 |

---

## 4. Why the Unrestricted Plan Fails

The plan assumed that within a short window of consecutive drone frames, OCR
outputs separated by small edit distance belong to the same trailer. **This is
false for this dataset.** The yard contains physically-adjacent trailers with
near-sequential IDs:

```
Seq 163: 702632 (gt 702632) + Seq 166: 702522 (gt 702522)
       edit distance 2, aspect ratio identical, same window → merged → vote picks 702632 → harmful

Seq 155: 737 (gt 737) + nearby: 72, 714, 717
       short numeric IDs, edit distance 1 — catastrophically merge
```

Examples of harmful merges observed at W=5, E=2:
- `702520 ↔ 702524` (distance 1, both real GT)
- `702522 ↔ 702632` (distance 2, both real GT)
- `676052 ↔ 676065` (distance 2, both real GT)
- `717 ↔ 714` (distance 1, both real GT)
- `SFU100885 ↔ SIFU1008858` (different container codes, merged)

Aspect-ratio filtering (`ar_tol=0.3`) doesn't discriminate because same-style
trailers have identical aspect ratios.

**Root cause:** text-space edit distance is not a correspondence signal on this
dataset — the trailers themselves are similar strings. Without spatial
verification (which EXP-12 showed is infeasible), the clustering merges
different objects.

The only safe operating point is the heavily gated one: **format-only +
E=1 + W=3**, which restricts voting to the `R\d{5}` population (where small
edit-distance clusters are genuinely same-trailer re-reads) and ignores the
numeric-ID trailers entirely.

---

## 5. Verdict

**REJECTED as a broad strategy. Accept only the heavily gated variant.**

The planned config (+3 to +6pp target) regresses −2.1pp. The best safely-gated
variant produces +0.4pp (+3 correct / 672, precision +0.7pp) — below the
plan's target and comparable in magnitude to EXP-15.

**Fundamental limit:** text-space clustering cannot distinguish same-trailer
re-reads from adjacent-trailer-with-similar-ID reads on this dataset. Any gain
has to come from *either* (a) a spatial correspondence signal (homography /
geo-registration, per EXP-17), or (b) gating to a sub-population where
ID collisions are rare (here: the `R\d{5}` trailers, which is what the
`--format-only` gate does).

### Recommendations

1. **Do not wire EXP-14 into production as currently conceived.** The dataset's
   adjacent-trailer pattern would cause ongoing harmful merges.
2. **If pursuing a voter, restrict it to `^R\d{5}$`-matching outputs only**, with
   edit distance 1 and window 3. Expected gain: ~+0.4pp, precision +0.7pp.
   Still marginal.
3. **Revisit the premise for EXP-17 (homography tracker).** EXP-14's failure
   mode suggests the *real* blocker is the lack of a correspondence signal,
   not the lack of a voter. If EXP-17 can provide that signal, a combined
   EXP-11-style vote could recover more of the original +9.9pp ceiling.
4. **EXP-11's ideal +9.9pp is a ceiling only achievable with perfect
   correspondence.** Text-space clustering captures almost none of it on
   this dataset.

---

## 6. Reproducing

```bash
# Unrestricted plan default (regresses)
python tests/text_space_voter.py --window 5 --max-edit 2 --ar-tol 0.3

# Best safe config (marginal positive)
python tests/text_space_voter.py --window 3 --max-edit 1 --format-only

# Source: benchmark_EXP-15_20260424_162508.json (EXP-15 at 259/672 = 38.5%)
```

Outputs `tests/results/benchmark_EXP-14_<timestamp>.json` with the cluster log
and per-annotation final texts.

---

## 7. Lessons

1. **Edit distance is not a correspondence signal on datasets with structured
   IDs.** When the population shares a narrow ID shape (numeric-yard IDs
   here), many real distinct objects sit within 1–2 edits of each other.
   Clustering merges them.
2. **EXP-11's upper bound was a trap.** +9.9pp assumes perfect track
   formation. The 9.9pp isn't "unlocked by voting" — it's "unlocked by
   correspondence, then voting." Voting without correspondence does near
   zero on this dataset.
3. **Format-gating is the only defense without spatial information.** The
   `R\d{5}` sub-population is sparse enough inside any given window that
   edit-distance clustering rarely merges different trailers. Everything
   else (`\d{3-7}` numeric IDs, short alphanumerics) is too dense.
4. **Gating that shrinks activity to ~10 rewrites — the same order as EXP-15
   — produces similar-magnitude gains.** We're at the floor of what
   post-processing can do without new information.
