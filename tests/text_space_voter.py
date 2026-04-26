"""
Text-Space Temporal Voting (EXP-14)
====================================
Post-hoc analysis: given an existing benchmark JSON (per-annotation OCR results
with ground truth), cluster detections within a sliding window of consecutive
frames by OCR-string edit distance + aspect-ratio similarity, apply confidence-
weighted vote within each cluster, and rewrite each member's text with the
winner.

Rationale: EXP-12 proved pixel-space IoU tracking is infeasible on this moving-
drone dataset (cross-frame IoU ~= 0). But we don't need a geometric tracker to
vote — if three nearby frames emit {"JBHU235644","JBH0235644","JBHU235644"}, an
edit-distance cluster can vote without ever solving correspondence in pixels.

Usage:
    python tests/text_space_voter.py
    python tests/text_space_voter.py --input tests/results/benchmark_EXP-15_20260424_162508.json
    python tests/text_space_voter.py --window 5 --max-edit 2 --ar-tol 0.3
"""

import argparse
import json
import platform
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
DEFAULT_INPUT = RESULTS_DIR / "benchmark_EXP-15_20260424_162508.json"

SEQ_RE = re.compile(r"DJI_\d+_(\d+)_", re.IGNORECASE)

_FORMAT_RES = [
    re.compile(r"^JBHZ\d{6}$"),
    re.compile(r"^JBHU\d{6}$"),
    re.compile(r"^R\d{5}$"),
]


def parse_seq(image_file: str) -> int | None:
    m = SEQ_RE.search(image_file)
    return int(m.group(1)) if m else None


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Bounded Levenshtein. Returns `cap` if distance >= cap."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) >= cap:
        return cap
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min >= cap:
            return cap
        prev = cur
    return min(prev[lb], cap)


def format_match(s: str) -> bool:
    return any(p.match(s) for p in _FORMAT_RES)


def cluster_window(items: list[dict], max_edit: int, ar_tol: float) -> list[list[int]]:
    """
    Greedy agglomerative clustering within a window. `items` is a list of dicts
    with keys: norm_text, aspect_ratio. Returns list of index-lists.
    """
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        ti, ari = items[i]["norm_text"], items[i]["aspect_ratio"]
        if not ti:
            continue
        for j in range(i + 1, n):
            tj, arj = items[j]["norm_text"], items[j]["aspect_ratio"]
            if not tj:
                continue
            if ari is not None and arj is not None and abs(ari - arj) > ar_tol:
                continue
            d = edit_distance(ti, tj, cap=max_edit + 1)
            if d <= max_edit:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if items[i]["norm_text"]:
            groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) >= 2]


def vote(members: list[dict], format_tiebreak: bool = True,
         min_agreement: int = 1) -> tuple[str | None, float]:
    """Confidence-weighted vote. Returns (None, 0) if no winner has `min_agreement` distinct supporters."""
    weights: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    max_conf_per_text: dict[str, float] = defaultdict(float)
    for m in members:
        t = m["norm_text"]
        c = m["conf"] or 0.0
        weights[t] += c
        counts[t] += 1
        if c > max_conf_per_text[t]:
            max_conf_per_text[t] = c
    top = max(weights.values())
    candidates = [t for t, w in weights.items() if w == top]
    if format_tiebreak and len(candidates) > 1:
        fm = [t for t in candidates if format_match(t)]
        if fm:
            candidates = fm
    winner = max(candidates, key=lambda t: max_conf_per_text[t])
    if counts[winner] < min_agreement:
        return None, 0.0
    return winner, weights[winner]


def run(input_path: Path, window: int, max_edit: int, ar_tol: float,
        min_agreement: int, format_only: bool,
        output_path: Path | None):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    source_exp = data.get("exp_id", "UNKNOWN")
    results = data["annotation_results"]
    total = len(results)

    # Baseline stats
    base_correct = sum(1 for r in results if r.get("exact_match") is True)
    base_returned = sum(1 for r in results if r.get("ocr_text"))

    # Attach normalised fields; group by frame seq
    enriched = []
    for r in results:
        seq = parse_seq(r["image_file"])
        txt = (r.get("ocr_text") or "").strip().upper().replace(" ", "")
        enriched.append({
            "r": r,
            "seq": seq,
            "norm_text": txt,
            "conf": r.get("ocr_confidence"),
            "aspect_ratio": r.get("aspect_ratio"),
        })

    # Sort by seq (None → stable, treated as isolated)
    by_seq: dict[int, list[int]] = defaultdict(list)
    no_seq_idx: list[int] = []
    for idx, e in enumerate(enriched):
        if e["seq"] is None:
            no_seq_idx.append(idx)
        else:
            by_seq[e["seq"]].append(idx)
    sorted_seqs = sorted(by_seq.keys())

    # Slide a window of size `window` across consecutive seqs; cluster+vote.
    # Each annotation may appear in multiple windows; we apply votes
    # greedily and mark members as voted so later windows don't re-vote.
    voted: dict[int, tuple[str, str]] = {}  # idx -> (new_text, cluster_sig)
    cluster_log = []

    for start_i in range(len(sorted_seqs)):
        seqs_in_win = sorted_seqs[start_i:start_i + window]
        if len(seqs_in_win) < 2:
            break
        idxs = [i for s in seqs_in_win for i in by_seq[s]]
        if len(idxs) < 2:
            continue
        items = [enriched[i] for i in idxs]
        clusters = cluster_window(items, max_edit=max_edit, ar_tol=ar_tol)
        for cluster in clusters:
            members = [items[k] for k in cluster]
            if format_only and not any(format_match(m["norm_text"]) for m in members):
                continue
            texts_in = {m["norm_text"] for m in members}
            if len(texts_in) == 1:
                continue
            winner, score = vote(members, min_agreement=min_agreement)
            if winner is None:
                continue
            sig = f"win{start_i}:{sorted_seqs[start_i]}-{seqs_in_win[-1]}"
            for k in cluster:
                gidx = idxs[k]
                if gidx not in voted:
                    voted[gidx] = (winner, sig)
            cluster_log.append({
                "window_start_seq": sorted_seqs[start_i],
                "window_end_seq": seqs_in_win[-1],
                "member_texts": [m["norm_text"] for m in members],
                "member_confs": [round(m["conf"] or 0, 3) for m in members],
                "winner": winner,
                "score": round(score, 3),
            })

    # Rewrite + score
    unchanged_correct = 0
    helpful = 0  # wrong -> correct
    harmful = 0  # correct -> wrong
    neutral_ww = 0  # wrong -> wrong (different)
    neutral_same = 0  # wrong -> same wrong (text unchanged)
    final_correct = 0
    final_returned = 0
    per_ann_final = []

    for idx, e in enumerate(enriched):
        r = e["r"]
        gt = (r.get("ground_truth") or "").strip().upper() or None
        orig = (r.get("ocr_text") or "").strip().upper() or None
        if idx in voted:
            new_text = voted[idx][0]
        else:
            new_text = orig

        new_match = (new_text == gt) if (new_text and gt) else False
        orig_match = (orig == gt) if (orig and gt) else False

        if new_text:
            final_returned += 1
        if new_match:
            final_correct += 1

        if idx in voted and new_text != orig:
            if orig_match and not new_match:
                harmful += 1
            elif not orig_match and new_match:
                helpful += 1
            else:
                neutral_ww += 1
        elif idx in voted and new_text == orig:
            neutral_same += 1
        if new_match and orig_match:
            unchanged_correct += 1

        per_ann_final.append({
            "annotation_id": r["annotation_id"],
            "image_file": r["image_file"],
            "ground_truth": r.get("ground_truth"),
            "orig_text": r.get("ocr_text"),
            "voted_text": new_text if idx in voted else None,
            "final_text": new_text,
            "orig_correct": orig_match,
            "final_correct": new_match,
        })

    final_precision = 100 * final_correct / final_returned if final_returned else 0
    base_precision = 100 * base_correct / base_returned if base_returned else 0

    # Subset: per-aspect-ratio bucket delta
    by_bucket_delta: dict[str, dict] = defaultdict(lambda: {"total": 0, "base": 0, "final": 0})
    for e, pa in zip(enriched, per_ann_final):
        b = e["r"].get("aspect_ratio_bucket", "unknown")
        by_bucket_delta[b]["total"] += 1
        if pa["orig_correct"]:
            by_bucket_delta[b]["base"] += 1
        if pa["final_correct"]:
            by_bucket_delta[b]["final"] += 1

    # Print summary
    print("=" * 72)
    print("TEXT-SPACE TEMPORAL VOTING (EXP-14)")
    print(f"Source:  {input_path.name}  (EXP: {source_exp})")
    print(f"Params:  window={window}  max_edit={max_edit}  ar_tol={ar_tol}")
    print(f"Run at:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)
    print()
    print("PER-ANNOTATION BASELINE (from source):")
    print(f"  Correct       : {base_correct} / {total} = {base_correct/total*100:.1f}%")
    print(f"  Text returned : {base_returned} / {total} = {base_returned/total*100:.1f}%")
    print(f"  Precision     : {base_precision:.1f}%")
    print()
    print("VOTING ACTIVITY:")
    print(f"  Annotations touched by a vote : {len(voted)} / {total}")
    print(f"  Cluster votes fired (distinct): {len(cluster_log)}")
    print(f"  Rewrite outcomes (text changed):")
    print(f"    helpful  (wrong  -> correct) : {helpful}")
    print(f"    harmful  (correct-> wrong)   : {harmful}")
    print(f"    neutral  (wrong  -> wrong)   : {neutral_ww}")
    print(f"    no-op    (text unchanged)    : {neutral_same}")
    print()
    print("AFTER VOTING:")
    print(f"  Correct       : {final_correct} / {total} = {final_correct/total*100:.1f}%  "
          f"(d {final_correct-base_correct:+d}, {final_correct/total*100 - base_correct/total*100:+.1f} pp)")
    print(f"  Text returned : {final_returned} / {total} = {final_returned/total*100:.1f}%")
    print(f"  Precision     : {final_precision:.1f}%  (d {final_precision-base_precision:+.1f} pp)")
    print()
    print("PER-BUCKET DELTA:")
    for b, s in sorted(by_bucket_delta.items()):
        d = s["final"] - s["base"]
        print(f"  {b:12s} : {s['total']:4d} total | base {s['base']:3d}  -> final {s['final']:3d}  ({d:+d})")
    print()
    print("=" * 72)

    # Save JSON
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"benchmark_EXP-14_{ts}.json"

    summary = {
        "exp_id": "EXP-14",
        "analysis_type": "text_space_temporal_voting",
        "source_exp_id": source_exp,
        "source_file": str(input_path),
        "params": {"window": window, "max_edit": max_edit, "ar_tol": ar_tol},
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "system": {"platform": platform.platform(), "python": platform.python_version()},
        "baseline": {
            "total": total,
            "correct": base_correct,
            "correct_pct": round(base_correct / total * 100, 2),
            "returned": base_returned,
            "precision_pct": round(base_precision, 2),
        },
        "voting_activity": {
            "annotations_touched": len(voted),
            "cluster_votes": len(cluster_log),
            "helpful": helpful,
            "harmful": harmful,
            "neutral_diff": neutral_ww,
            "noop": neutral_same,
        },
        "after_voting": {
            "total": total,
            "correct": final_correct,
            "correct_pct": round(final_correct / total * 100, 2),
            "delta_correct": final_correct - base_correct,
            "delta_pp": round(final_correct / total * 100 - base_correct / total * 100, 2),
            "returned": final_returned,
            "precision_pct": round(final_precision, 2),
            "delta_precision_pp": round(final_precision - base_precision, 2),
        },
        "per_bucket_delta": {b: {**s, "delta": s["final"] - s["base"]} for b, s in by_bucket_delta.items()},
        "cluster_log": cluster_log,
        "per_annotation_final": per_ann_final,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Text-space temporal voting (EXP-14)")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--window", type=int, default=5,
                   help="Sliding window size in frame-seq units (default 5)")
    p.add_argument("--max-edit", type=int, default=2,
                   help="Max Levenshtein distance to merge two strings (default 2)")
    p.add_argument("--ar-tol", type=float, default=0.3,
                   help="Max |aspect_ratio_a - aspect_ratio_b| to merge (default 0.3)")
    p.add_argument("--min-agreement", type=int, default=1,
                   help="Min number of distinct votes the winner must have (default 1=plurality)")
    p.add_argument("--format-only", action="store_true",
                   help="Only vote within clusters where at least one member matches a known format")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    run(Path(args.input), args.window, args.max_edit, args.ar_tol,
        args.min_agreement, args.format_only,
        Path(args.output) if args.output else None)
