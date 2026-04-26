# OCR Improvements — Round 2 Plan (EXP-14 onward)

## Context

Round 1 squeezed ~+16 pp from padding, detection-threshold tuning, bbox expansion, and conservative character substitution. Current production-capable best (EXP-09 + EXP-04 + EXP-06): **38.2% exact match, 57.6% precision, ~110 ms median** on 672 aerial drone trailer-ID crops.

**EXP-13 (stacked-vertical portrait decoder)** was evaluated and rejected — see [docs/ocr-performance-experiment-13.md](ocr-performance-experiment-13.md). Headline finding: PaddleOCR PP-OCRv4's English rec model cannot read reconstructed stacked text (returns 0.18–0.20-confidence hallucinations). The stencil-style painted trailer font is outside its training distribution without horizontal word context. Portraits remain stuck at 1.3%.

Round 2 deliberately **steps around** the portrait problem — it targets the 28% wrong-text rate and exploits cross-frame signal that every prior experiment has left on the table. Portrait fix is deferred to a model-swap or fine-tune effort (see §5).

Constraints (same as Round 1):
- CPU-only, no GPU upscaling
- Accuracy-first: moderate latency growth OK if the accuracy payoff is clear
- No model training in-scope for Round 2 (defer to Round 3)

**Target:** combined config reaches **≥ 45% exact match, ≥ 62% precision** without portraits contributing — i.e. the gains come from wrong-text reduction and temporal voting.

---

## Proposed Experiments

Ranked by expected impact × implementation cost. All are preprocessing or post-hoc — no retraining, no GPU.

### ⭐ EXP-14 — Text-space temporal aggregation

**Problem.** EXP-11 already proved confidence-weighted voting across frames adds **+9.9 pp on multi-frame trailers** (per-trailer accuracy 38.2% → 48.1%). EXP-12 showed IoU-based pixel-space tracking is infeasible — the moving drone makes cross-frame IoU effectively zero. So the gain sits there, locked behind a tracker problem.

**Insight.** You don't need a geometric *tracker* to vote. If within a sliding window of N consecutive frames the OCR outputs are `{"JBHU235644", "JBH0235644", "JBHU235644"}`, you can cluster them by **edit distance on the OCR string itself** (Levenshtein ≤ 2) and vote — without ever solving the correspondence problem in pixel space.

**Approach.**
1. Sort frames by timestamp (filename order in this dataset).
2. Sliding window of W frames (start with W = 5).
3. Within each window, collect `(frame_id, bbox_shape, ocr_text, confidence)` tuples.
4. Cluster tuples where `edit_distance(text_a, text_b) ≤ 2` **AND** `|aspect_ratio_a - aspect_ratio_b| < 0.3` (weak shape prior — prevents merging two unrelated trailers that happened to OCR similarly).
5. Within each cluster of size ≥ 2, replace each member's text with confidence-weighted majority vote.
6. Optional refinement: among tied or near-tied candidates, prefer the one matching a known format prefix (`^JBHZ\d{6}$`, `^JBHU\d{6}$`, `^R\d{5}$`).

**Expected impact.** Recover a meaningful portion of EXP-11's +9.9 pp ceiling. Realistic target: **+3–6 pp overall exact match.** Skips the 156 portrait crops (nothing to vote on if no text was returned) so doesn't need a portrait fix.

**Cost.** Post-hoc / zero OCR re-runs. Pure Python string work, sub-ms per frame.

**Files to touch.**
- New: `app/temporal_voter.py` (production-ready cluster+vote logic)
- Modified: `tests/benchmark_ocr.py` — `--temporal-vote W` flag; post-processing pass over `annotation_results` after the main loop completes
- Consider extending [tests/temporal_aggregation.py](../tests/temporal_aggregation.py) instead of a new file if the existing module fits
- Eventually: a batch endpoint in [app/main.py](../app/main.py) — out of scope for the experiment itself

**Verification.**
- Primary: overall exact-match accuracy, per-aspect-ratio breakdown.
- Guardrail: precision must not drop (if cluster voting merges a correct + wrong pair, it could).
- Ablation: W = 3, 5, 7 sweep. Edit-distance threshold 1 vs 2.
- Cross-check with EXP-11's ideal ceiling (48.1% per-track on multi-frame): how much of the gap does text-space voting close?

---

### ⭐ EXP-15 — Format-aware candidate rescoring

**Problem.** 28–30% of OCR outputs are wrong-text (returned a string, but it doesn't match ground truth). EXP-06 applies one conservative substitution pass (`O↔0`, `I↔1`, etc.) based on fixed positional rules. It leaves most ambiguities on the table because it only tries one substitution combination.

**Approach.**
1. Given OCR output `s`, enumerate bounded candidate variants by applying the confusion table (`O↔0, I↔1, S↔5, B↔8, G↔6, Z↔2, Q↔0, H↔8`) at every position where a character is in a confusable class. Cap total candidates at ~64 to bound cost.
2. Score each candidate by:
    ```
    score = confidence_prior × format_match_score(candidate)
    ```
    where `format_match_score` is high if the candidate matches one of the known patterns:
    - `^JBHZ\d{6}$` (common, always vertical but also appears horizontally)
    - `^JBHU\d{6}$` (less common, always vertical)
    - `^R\d{5}$`
    - `^[A-Z]{4}\d{6}$` (generic trailer-ID format)
    - `^\d{5,7}$` (pure numeric)
3. Pick the highest-scoring candidate. If none match any pattern, fall back to the conservative EXP-06 output.

**Expected impact.** **+1–3 pp overall.** Lower-bound because most wrong-text cases are probably detection-level failures, not single-character confusions — but it's nearly free.

**Cost.** `O(2^k)` where k ≈ number of confusable chars in the output (≤ 10 for a trailer ID). Single-digit microseconds per result. Zero OCR re-runs.

**Files to touch.**
- Modified: [app/utils.py](../app/utils.py) — extend `postprocess_text()` into a candidate-enumeration rescorer
- Modified: [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — new `--rescore` flag (mutually exclusive with the existing `postprocess` flag, or stacks after it)

**Verification.**
- Primary: wrong-text count reduction, exact-match gain.
- Guardrail: **false corrections** — cases where EXP-06 got it right but the rescorer picks a wrong variant. Track this explicitly in the subset stats.
- Ablation: with/without format prior (pure confidence weighting alone as a control).

---

### EXP-16 — Blur-gated adaptive sharpening

**Problem.** Motion blur from the moving drone is a known failure mode. EXP-10 applied sharpen+dilate as a retry fallback and barely helped — because it added noise on crops that didn't need it.

**Approach.**
1. Compute variance of Laplacian per crop (one OpenCV call, ~1 ms).
2. If variance < threshold (calibrate against dataset; expect ~80), apply PIL UnsharpMask **before** the first OCR pass (not as retry).
3. Else pass through untouched.

**Expected impact.** **+1–3 pp** on the blurry subset. Modest but roughly free.

**Cost.** ~1–2 ms per crop for the blur metric; sharpening only on blurry crops.

**Files to touch.**
- New function in [tests/preprocessing.py](../tests/preprocessing.py) — `blur_gated_sharpen(image, threshold=80)`
- Modified [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — `--blur-gated-sharpen` flag

**Verification.**
- Primary: exact-match gain on blurry subset (define via the Laplacian metric; produce a separate bucket in the stats).
- Guardrail: non-blurry bucket accuracy unchanged.

---

### EXP-17 — Homography-stabilised tracker (complement, not alternative, to EXP-14)

**Problem.** EXP-14 can only vote when OCR actually returned text. For the 33% of crops where OCR returned nothing, there is no string to cluster. Homography-based tracking can link those silent crops to their voted neighbours.

**Approach.**
1. For each consecutive frame pair, compute ORB features + match + RANSAC homography (OpenCV).
2. Warp each frame's bboxes into the next frame's coordinate system.
3. Apply IoU on warped boxes → standard EXP-11-style track formation.
4. Propagate the voted text from any multi-frame track onto its silent members.

**Expected impact.** Incremental **+1–3 pp** over EXP-14 by rescuing silent crops. Only worth doing if EXP-14 shows a meaningful gain first.

**Cost.** ~50–100 ms per frame pair for ORB + RANSAC. Amortises across all bboxes in the frame.

**Files to touch.**
- New: `tests/homography_tracker.py`
- Modified [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — `--homography-vote` flag (layered on top of `--temporal-vote`)

**Priority.** Implement only after EXP-14 is landed and measured — the two approaches are complementary but text-space voting is the cheaper and strictly lower-risk of the two.

---

### EXP-18 — Multi-scale ensemble for low-confidence crops (gated)

**Approach.** For first-pass crops where confidence < 0.6, re-run OCR at 1.5× scale (plain PIL resize — no AI upscaling). Take the higher-confidence result.

**Expected impact.** **+0.5–2 pp.** Marginal but applies to the hardest ~35% of crops.

**Cost.** +40–80 ms on ~35% of crops → ~+15–25 ms average.

**Priority.** Low. Include only after the cheaper experiments are exhausted and if the latency budget permits.

---

## Recommended Execution Order

| # | Experiment | Why now |
|---|---|---|
| 1 | **EXP-15** (format-aware rescoring) | Fastest to land, near-zero risk. Sanity check on the pipeline plumbing for Round 2 and a quick easy win. |
| 2 | **EXP-14** (text-space temporal voting) | Biggest untapped ceiling. Unlocks EXP-11's proven +9.9 pp. Independent of EXP-15 — run both in the final combined config. |
| 3 | **EXP-16** (blur-gated sharpen) | Fast to try, modest gain, low coupling to EXP-14/15. |
| 4 | **EXP-17** (homography tracker) | Only if EXP-14 leaves obvious misses (silent crops that should have been linked). |
| 5 | **EXP-18** (multi-scale ensemble) | Only if latency budget permits and the earlier experiments haven't closed the gap to the ~45% target. |

Each experiment gets its own combined-config benchmark run on top of the current best, then a combined EXP-14+15(+16) run for the final report.

---

## Out of Scope for Round 2

- **Portrait fix** — deferred to Round 3. Paths: fine-tune a tiny CPU char-classifier on a small labelled portrait subset, or swap PaddleOCR for a model with native vertical-text support (PP-OCRv5 / PaddleOCR-VL / Chinese-trained model).
- Multi-engine ensemble (EasyOCR, Tesseract) — failure modes overlap, compute doubles. May revisit as a targeted portrait-only fallback under Round 3.
- GPS / telemetry-based frame registration — only consider if homography stabilisation (EXP-17) proves insufficient.
- Any model training or GPU-dependent preprocessing.

---

## Verification Plan (common to all Round 2 experiments)

1. Add CLI flag + branch in [tests/benchmark_ocr.py](../tests/benchmark_ocr.py).
2. Run on top of the current best config (`--preprocess pad,postprocess --bbox-expand-ratio 0.1 --det-db-thresh 0.2 --det-db-box-thresh 0.3 --det-db-unclip-ratio 2.0`).
3. Regenerate [docs/experiment-comparison-report.md](experiment-comparison-report.md) with the new row via [tests/compare_experiments.py](../tests/compare_experiments.py).
4. Inspect per-aspect-ratio and per-area breakdowns — any experiment that gains overall but regresses a bucket is suspect.
5. Check p95 latency, not just median.
6. Wire winners into [app/ocr_processor.py](../app/ocr_processor.py) and [app/main.py](../app/main.py) only after benchmark confirms a net gain with acceptable latency.

---

## Critical Files

- [app/ocr_processor.py](../app/ocr_processor.py) — `OcrProcessor.process_image()` singleton entry point
- [app/utils.py](../app/utils.py) — target for EXP-15 extension (`postprocess_text` → candidate rescorer)
- [app/main.py](../app/main.py) — FastAPI endpoints; may need a batch/temporal endpoint for EXP-14 production wiring
- [tests/preprocessing.py](../tests/preprocessing.py) — target for EXP-16 (`blur_gated_sharpen`)
- [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — harness, CLI flags, per-experiment branches
- [tests/temporal_aggregation.py](../tests/temporal_aggregation.py) — EXP-11 voting logic; extend for EXP-14's sliding-window text-space clustering
- [tests/compare_experiments.py](../tests/compare_experiments.py) — regenerates the comparison report
- [docs/experiment-comparison-report.md](experiment-comparison-report.md) — per-experiment rows
- [docs/ocr-performance-experiment-13.md](ocr-performance-experiment-13.md) — EXP-13 post-mortem, context for why portraits are deferred
