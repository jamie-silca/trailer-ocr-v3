"""
Temporal Aggregation Analysis (EXP-11)
=======================================
Groups annotations by ground-truth identity (same trailer appearing across
multiple frames), applies confidence-weighted majority voting, and reports
per-track accuracy — the metric that matters for production (one answer per
unique trailer).

No re-OCR is needed: runs post-hoc on any existing benchmark JSON.

Usage:
    python tests/temporal_aggregation.py
    python tests/temporal_aggregation.py --input tests/results/benchmark_EXP-03-04-06-COMBO_20260415_173040.json
    python tests/temporal_aggregation.py --output tests/results/benchmark_EXP-11_<timestamp>.json
"""

import argparse
import json
import platform
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"

DEFAULT_INPUT = RESULTS_DIR / "benchmark_EXP-03-04-06-COMBO_20260415_173040.json"


def confidence_weighted_vote(group: list[dict]) -> tuple[str | None, float]:
    """
    Return the majority OCR text weighted by confidence, and the mean confidence
    of all votes that agree with the winner.
    """
    vote_weights: dict[str, float] = defaultdict(float)
    for r in group:
        text = r.get("ocr_text")
        conf = r.get("ocr_confidence") or 0.0
        if text:
            vote_weights[text] += conf
    if not vote_weights:
        return None, 0.0
    winner = max(vote_weights, key=vote_weights.get)
    winner_confs = [r.get("ocr_confidence") or 0.0 for r in group
                    if r.get("ocr_text") == winner]
    avg_conf = sum(winner_confs) / len(winner_confs) if winner_confs else 0.0
    return winner, avg_conf


def run(input_path: Path, output_path: Path | None):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    source_exp = data.get("exp_id", "UNKNOWN")
    results = data["annotation_results"]
    total_anns = len(results)

    # ── Per-annotation baseline ───────────────────────────────────────────────
    per_ann_correct = sum(1 for r in results if r.get("exact_match") is True)
    per_ann_returned = sum(1 for r in results if r.get("ocr_text"))
    per_ann_precision = (per_ann_correct / per_ann_returned * 100) if per_ann_returned else 0

    # ── Build ground-truth tracks ─────────────────────────────────────────────
    # Group all annotations that share the same normalised ground-truth string.
    # In production you'd use a visual tracker; here ground truth gives us the
    # ideal upper-bound of what temporal aggregation can achieve.
    gt_groups: dict[str, list[dict]] = defaultdict(list)
    no_gt = 0
    for r in results:
        gt = r.get("ground_truth")
        if gt:
            gt_groups[gt.strip().upper()].append(r)
        else:
            no_gt += 1

    total_tracks = len(gt_groups)
    single_frame_tracks = sum(1 for v in gt_groups.values() if len(v) == 1)
    multi_frame_tracks = total_tracks - single_frame_tracks

    # ── Per-track majority vote ───────────────────────────────────────────────
    track_results = []
    for gt, group in gt_groups.items():
        predicted, pred_conf = confidence_weighted_vote(group)
        exact = (predicted or "").strip().upper() == gt if predicted else False

        # Per-annotation stats for this track
        ann_correct = sum(1 for r in group if r.get("exact_match") is True)
        ann_returned = sum(1 for r in group if r.get("ocr_text"))  # noqa: F841
        all_predictions = [r.get("ocr_text") for r in group if r.get("ocr_text")]

        track_results.append({
            "ground_truth": gt,
            "frame_count": len(group),
            "annotation_ids": [r["annotation_id"] for r in group],
            "image_files": list({r["image_file"] for r in group}),
            "per_ann_correct": ann_correct,
            "per_ann_returned": ann_returned,
            "all_predictions": all_predictions,
            "voted_text": predicted,
            "voted_conf": round(pred_conf, 4) if pred_conf else None,
            "exact_match": exact,
        })

    track_correct = sum(1 for t in track_results if t["exact_match"])
    track_returned = sum(1 for t in track_results if t["voted_text"])
    track_precision = (track_correct / track_returned * 100) if track_returned else 0

    # Multi-frame tracks only
    multi_tracks = [t for t in track_results if t["frame_count"] >= 2]
    multi_correct = sum(1 for t in multi_tracks if t["exact_match"])
    multi_returned = sum(1 for t in multi_tracks if t["voted_text"])
    multi_precision = (multi_correct / multi_returned * 100) if multi_returned else 0

    # Per-annotation accuracy on multi-frame tracks only (for baseline comparison)
    multi_ann_results = [r for r in results
                         if r.get("ground_truth") and
                         len(gt_groups[r["ground_truth"].strip().upper()]) >= 2]
    multi_ann_correct = sum(1 for r in multi_ann_results if r.get("exact_match") is True)
    multi_ann_returned = sum(1 for r in multi_ann_results if r.get("ocr_text"))

    # ── Confidence threshold sweep (on raw per-annotation results) ────────────
    conf_sweep = []
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        kept = [r for r in results if (r.get("ocr_confidence") or 0) >= thresh]
        correct_at_thresh = sum(1 for r in kept if r.get("exact_match") is True)
        conf_sweep.append({
            "threshold": thresh,
            "kept": len(kept),
            "kept_pct": round(100 * len(kept) / total_anns, 1),
            "correct": correct_at_thresh,
            "correct_pct": round(100 * correct_at_thresh / total_anns, 1),
            "precision_pct": round(100 * correct_at_thresh / len(kept), 1) if kept else 0,
        })

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("TEMPORAL AGGREGATION ANALYSIS (EXP-11)")
    print(f"Source:  {input_path.name}  (EXP: {source_exp})")
    print(f"Run at:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()
    print("PER-ANNOTATION BASELINE (unchanged from source experiment):")
    print(f"  Correct       : {per_ann_correct} / {total_anns} = {per_ann_correct/total_anns*100:.1f}%")
    print(f"  Text returned : {per_ann_returned} / {total_anns} = {per_ann_returned/total_anns*100:.1f}%")
    print(f"  Precision     : {per_ann_precision:.1f}%")
    print()
    print(f"TRACK SUMMARY:  {total_tracks} unique trailers in {total_anns} annotations")
    print(f"  Single-frame  : {single_frame_tracks} trailers (no temporal benefit possible)")
    print(f"  Multi-frame   : {multi_frame_tracks} trailers ({len(multi_ann_results)} annotations)")
    print()
    print("PER-TRACK ACCURACY (confidence-weighted majority vote):")
    print(f"  All tracks    : {track_correct} / {total_tracks} = {track_correct/total_tracks*100:.1f}%  (precision {track_precision:.1f}%)")
    print(f"  Multi-frame   : {multi_correct} / {len(multi_tracks)} = {multi_correct/len(multi_tracks)*100:.1f}%  (precision {multi_precision:.1f}%)")
    print()
    print(f"  Multi-frame per-annotation baseline: {multi_ann_correct}/{len(multi_ann_results)} = {multi_ann_correct/len(multi_ann_results)*100:.1f}%")
    print(f"  -> Temporal gain on multi-frame trailers: "
          f"+{multi_correct/len(multi_tracks)*100 - multi_ann_correct/len(multi_ann_results)*100:.1f}pp")
    print()
    print("CONFIDENCE THRESHOLD SWEEP (per annotation, no re-OCR):")
    print(f"  {'Thresh':>6}  {'Kept':>5}  {'%':>5}  {'Correct':>8}  {'Exact%':>7}  {'Precision':>10}")
    for s in conf_sweep:
        print(f"  {s['threshold']:>6.1f}  {s['kept']:>5}  {s['kept_pct']:>5.1f}  "
              f"{s['correct']:>8}  {s['correct_pct']:>7.1f}  {s['precision_pct']:>10.1f}")
    print()
    print("=" * 70)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"benchmark_EXP-11_{ts}.json"

    summary = {
        "exp_id": "EXP-11",
        "analysis_type": "temporal_aggregation",
        "source_exp_id": source_exp,
        "source_file": str(input_path),
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "per_annotation_baseline": {
            "total": total_anns,
            "correct": per_ann_correct,
            "correct_pct": round(per_ann_correct / total_anns * 100, 1),
            "text_returned": per_ann_returned,
            "precision_pct": round(per_ann_precision, 1),
        },
        "track_summary": {
            "total_tracks": total_tracks,
            "single_frame": single_frame_tracks,
            "multi_frame": multi_frame_tracks,
        },
        "per_track_accuracy": {
            "all_tracks": {
                "correct": track_correct,
                "total": total_tracks,
                "correct_pct": round(track_correct / total_tracks * 100, 1),
                "returned": track_returned,
                "precision_pct": round(track_precision, 1),
            },
            "multi_frame_only": {
                "correct": multi_correct,
                "total": len(multi_tracks),
                "correct_pct": round(multi_correct / len(multi_tracks) * 100, 1) if multi_tracks else 0,
                "returned": multi_returned,
                "precision_pct": round(multi_precision, 1),
                "per_ann_baseline_correct_pct": round(multi_ann_correct / len(multi_ann_results) * 100, 1) if multi_ann_results else 0,
            },
        },
        "confidence_sweep": conf_sweep,
        "track_results": track_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal aggregation analysis (EXP-11)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Source benchmark JSON to analyse")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: auto-named in tests/results/)")
    args = parser.parse_args()
    run(Path(args.input), Path(args.output) if args.output else None)
