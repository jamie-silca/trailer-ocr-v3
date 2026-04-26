"""
IoU-based Frame-to-Frame Tracker (EXP-12)
==========================================
Links YOLO detections across consecutive drone frames using bbox IoU, then
applies confidence-weighted plurality voting per track. Unlike EXP-11 (which
grouped by ground truth as an ideal upper bound), this is a realistic
production-style tracker — its output is what we'd actually ship.

Filename convention: DJI_<timestamp>_<seq>_V.jpeg
    We sort frames by <seq> and link each detection in frame N to the
    highest-IoU detection in frame N+1 (if IoU > threshold). Chains form
    via union-find.

Usage:
    python tests/iou_tracker.py
    python tests/iou_tracker.py --input tests/results/benchmark_EXP-09_20260420_175919.json
    python tests/iou_tracker.py --iou-thresh 0.3 --max-gap 1
"""

import argparse
import json
import platform
import re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
DEFAULT_INPUT = RESULTS_DIR / "benchmark_EXP-09_20260420_175919.json"

SEQ_RE = re.compile(r"DJI_\d+_(\d+)_", re.IGNORECASE)


def parse_seq(image_file: str) -> int | None:
    m = SEQ_RE.search(image_file)
    return int(m.group(1)) if m else None


def iou(a: list[float], b: list[float]) -> float:
    """IoU for COCO bboxes [x, y, w, h]."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def link_tracks(results: list[dict], iou_thresh: float, max_gap: int) -> dict[int, int]:
    """
    Assign a track_id to every annotation via greedy IoU linking across frames.
    Returns: {annotation_id: track_id}
    """
    by_seq: dict[int, list[dict]] = defaultdict(list)
    unlinked = []
    for r in results:
        seq = parse_seq(r["image_file"])
        if seq is None:
            unlinked.append(r)
        else:
            by_seq[seq].append(r)

    uf = UnionFind()
    for r in results:
        uf.find(r["annotation_id"])

    sorted_seqs = sorted(by_seq.keys())
    for i, seq in enumerate(sorted_seqs):
        cur = by_seq[seq]
        for j in range(i + 1, min(i + 1 + max_gap, len(sorted_seqs))):
            nxt_seq = sorted_seqs[j]
            if nxt_seq - seq > max_gap:
                break
            nxt = by_seq[nxt_seq]
            pairs = []
            for a in cur:
                for b in nxt:
                    s = iou(a["bbox"], b["bbox"])
                    if s >= iou_thresh:
                        pairs.append((s, a["annotation_id"], b["annotation_id"]))
            pairs.sort(reverse=True)
            used_a, used_b = set(), set()
            for s, aid, bid in pairs:
                if aid in used_a or bid in used_b:
                    continue
                uf.union(aid, bid)
                used_a.add(aid)
                used_b.add(bid)

    track_map = {r["annotation_id"]: uf.find(r["annotation_id"]) for r in results if parse_seq(r["image_file"]) is not None}
    next_singleton = max(track_map.values(), default=0) + 1
    for r in unlinked:
        track_map[r["annotation_id"]] = next_singleton
        next_singleton += 1
    return track_map


def confidence_weighted_vote(group: list[dict]) -> tuple[str | None, float]:
    weights: dict[str, float] = defaultdict(float)
    for r in group:
        t, c = r.get("ocr_text"), r.get("ocr_confidence") or 0.0
        if t:
            weights[t] += c
    if not weights:
        return None, 0.0
    winner = max(weights, key=weights.get)
    confs = [r.get("ocr_confidence") or 0.0 for r in group if r.get("ocr_text") == winner]
    return winner, (sum(confs) / len(confs)) if confs else 0.0


def dominant_gt(group: list[dict]) -> str | None:
    gts = [r["ground_truth"].strip().upper() for r in group if r.get("ground_truth")]
    if not gts:
        return None
    return Counter(gts).most_common(1)[0][0]


def run(input_path: Path, iou_thresh: float, max_gap: int, output_path: Path | None):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    source_exp = data.get("exp_id", "UNKNOWN")
    results = data["annotation_results"]
    total_anns = len(results)

    per_ann_correct = sum(1 for r in results if r.get("exact_match"))
    per_ann_returned = sum(1 for r in results if r.get("ocr_text"))

    track_map = link_tracks(results, iou_thresh, max_gap)
    tracks: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        tracks[track_map[r["annotation_id"]]].append(r)

    gt_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        gt = r.get("ground_truth")
        if gt:
            gt_groups[gt.strip().upper()].append(r)
    ideal_tracks = len(gt_groups)

    track_rows = []
    pure_tracks = 0
    contaminated = 0
    for tid, group in tracks.items():
        gts_in_track = {r["ground_truth"].strip().upper() for r in group if r.get("ground_truth")}
        if len(gts_in_track) == 1:
            pure_tracks += 1
        elif len(gts_in_track) > 1:
            contaminated += 1
        dom = dominant_gt(group)
        voted, conf = confidence_weighted_vote(group)
        exact = (voted or "").strip().upper() == dom if (voted and dom) else False
        track_rows.append({
            "track_id": tid,
            "frame_count": len(group),
            "annotation_ids": [r["annotation_id"] for r in group],
            "distinct_gts": sorted(gts_in_track),
            "dominant_gt": dom,
            "voted_text": voted,
            "voted_conf": round(conf, 4) if conf else None,
            "exact_match": exact,
        })

    total_tracks = len(tracks)
    single_frame = sum(1 for t in track_rows if t["frame_count"] == 1)
    multi_frame = total_tracks - single_frame
    track_correct = sum(1 for t in track_rows if t["exact_match"])
    track_returned = sum(1 for t in track_rows if t["voted_text"])
    track_precision = (track_correct / track_returned * 100) if track_returned else 0

    multi = [t for t in track_rows if t["frame_count"] >= 2]
    multi_correct = sum(1 for t in multi if t["exact_match"])
    multi_returned = sum(1 for t in multi if t["voted_text"])
    multi_precision = (multi_correct / multi_returned * 100) if multi_returned else 0

    print("=" * 72)
    print("IoU TRACKER -- TEMPORAL AGGREGATION (EXP-12)")
    print(f"Source:  {input_path.name}  (EXP: {source_exp})")
    print(f"Params:  iou_thresh={iou_thresh}  max_gap={max_gap} frames")
    print(f"Run at:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)
    print()
    print("PER-ANNOTATION BASELINE:")
    print(f"  Correct       : {per_ann_correct} / {total_anns} = {per_ann_correct/total_anns*100:.1f}%")
    print(f"  Text returned : {per_ann_returned} / {total_anns} = {per_ann_returned/total_anns*100:.1f}%")
    print()
    print("TRACK FORMATION:")
    print(f"  Tracker produced : {total_tracks} tracks  (ideal by GT: {ideal_tracks})")
    print(f"  Single-frame     : {single_frame}")
    print(f"  Multi-frame      : {multi_frame}")
    print(f"  Pure tracks      : {pure_tracks}  (one GT identity)")
    print(f"  Contaminated     : {contaminated}  (multiple GTs merged -- tracker error)")
    print(f"  Over/undersplit  : {total_tracks - ideal_tracks:+d} vs ideal")
    print()
    print("PER-TRACK ACCURACY (confidence-weighted vote, vs dominant GT):")
    print(f"  All tracks   : {track_correct} / {total_tracks} = {track_correct/total_tracks*100:.1f}%  (precision {track_precision:.1f}%)")
    if multi:
        print(f"  Multi-frame  : {multi_correct} / {len(multi)} = {multi_correct/len(multi)*100:.1f}%  (precision {multi_precision:.1f}%)")
    print()
    print("=" * 72)

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"benchmark_EXP-12_{ts}.json"

    summary = {
        "exp_id": "EXP-12",
        "analysis_type": "iou_tracker_temporal_aggregation",
        "source_exp_id": source_exp,
        "source_file": str(input_path),
        "params": {"iou_thresh": iou_thresh, "max_gap": max_gap},
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "system": {"platform": platform.platform(), "python": platform.python_version()},
        "per_annotation_baseline": {
            "total": total_anns,
            "correct": per_ann_correct,
            "correct_pct": round(per_ann_correct / total_anns * 100, 1),
        },
        "track_formation": {
            "total_tracks": total_tracks,
            "ideal_tracks_by_gt": ideal_tracks,
            "single_frame": single_frame,
            "multi_frame": multi_frame,
            "pure_tracks": pure_tracks,
            "contaminated_tracks": contaminated,
        },
        "per_track_accuracy": {
            "all_tracks": {
                "correct": track_correct, "total": total_tracks,
                "correct_pct": round(track_correct / total_tracks * 100, 1),
                "returned": track_returned,
                "precision_pct": round(track_precision, 1),
            },
            "multi_frame_only": {
                "correct": multi_correct, "total": len(multi),
                "correct_pct": round(multi_correct / len(multi) * 100, 1) if multi else 0,
                "returned": multi_returned,
                "precision_pct": round(multi_precision, 1),
            } if multi else None,
        },
        "track_results": track_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="IoU tracker + temporal aggregation (EXP-12)")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--iou-thresh", type=float, default=0.3)
    p.add_argument("--max-gap", type=int, default=1,
                   help="Max frame-seq gap to link across (1 = consecutive frames only)")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    run(Path(args.input), args.iou_thresh, args.max_gap,
        Path(args.output) if args.output else None)
