"""
Post-hoc confidence-threshold sweep on a benchmark JSON.

Applies a min-confidence gate to per-annotation OCR results (no re-OCR) and
reports how accuracy / precision / text-returned shift across thresholds. Used
to evaluate whether adding --min-conf to the deployed pipeline would be a
favourable trade.

Usage:
    python tests/apply_conf_threshold.py
    python tests/apply_conf_threshold.py --input tests/results/benchmark_EXP-09-10_20260420_180525.json
"""

import argparse
import copy
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 and can't encode Δ / ≥ in our markdown.
# Re-open stdout as utf-8 with replacement so printing never fails.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "tests" / "results" / "benchmark_EXP-09-10_20260420_180525.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "results" / "exp_09_10_conf_sweep.md"
THRESHOLDS = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90]


def apply_threshold(results: list[dict], threshold: float) -> list[dict]:
    """Return a deep-copied results list with ocr_text/exact_match zeroed where conf < threshold."""
    out = []
    for r in results:
        r2 = copy.deepcopy(r)
        conf = r2.get("ocr_confidence")
        if conf is None or conf < threshold:
            if r2.get("ocr_text"):
                r2["ocr_text"] = None
                r2["ocr_confidence"] = 0.0
                # exact_match is only meaningful when ground truth exists
                if r2.get("ground_truth") is not None:
                    r2["exact_match"] = False
        out.append(r2)
    return out


def compute_stats(results: list[dict]) -> dict:
    total = len(results)
    text_returned = sum(1 for r in results if r.get("ocr_text"))
    correct = sum(1 for r in results if r.get("exact_match") is True)
    wrong = sum(1 for r in results if r.get("exact_match") is False and r.get("ocr_text"))
    no_text = total - text_returned

    buckets: dict[str, dict] = {}
    for r in results:
        b = r.get("aspect_ratio_bucket", "unknown")
        s = buckets.setdefault(b, {"total": 0, "correct": 0, "text_returned": 0})
        s["total"] += 1
        if r.get("ocr_text"):
            s["text_returned"] += 1
        if r.get("exact_match") is True:
            s["correct"] += 1

    return {
        "total": total,
        "text_returned": text_returned,
        "correct": correct,
        "wrong": wrong,
        "no_text": no_text,
        "exact_match_pct": 100 * correct / total if total else 0.0,
        "precision_pct": 100 * correct / text_returned if text_returned else 0.0,
        "buckets": buckets,
    }


def format_table(rows: list[tuple[float, dict]]) -> str:
    lines = []
    lines.append("| Min Conf | Text Returned | Correct | Exact Match % | Precision % | Δ ExactMatch | Δ Precision |")
    lines.append("|---|---|---|---|---|---|---|")
    baseline_em = rows[0][1]["exact_match_pct"]
    baseline_pr = rows[0][1]["precision_pct"]
    for th, s in rows:
        th_label = "ungated" if th == 0.0 else f"≥{th:.2f}"
        d_em = s["exact_match_pct"] - baseline_em
        d_pr = s["precision_pct"] - baseline_pr
        lines.append(
            f"| {th_label} | {s['text_returned']} ({100*s['text_returned']/s['total']:.1f}%) "
            f"| {s['correct']} | {s['exact_match_pct']:.2f}% | {s['precision_pct']:.2f}% "
            f"| {d_em:+.2f}pp | {d_pr:+.2f}pp |"
        )
    return "\n".join(lines)


def format_bucket_table(rows: list[tuple[float, dict]]) -> str:
    bucket_names = sorted({b for _, s in rows for b in s["buckets"]})
    lines = []
    header = ["Bucket"] + [("ungated" if th == 0.0 else f"≥{th:.2f}") for th, _ in rows]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for bn in bucket_names:
        row = [bn]
        for _, s in rows:
            b = s["buckets"].get(bn, {"total": 0, "correct": 0})
            total = b["total"]
            correct = b["correct"]
            pct = 100 * correct / total if total else 0.0
            row.append(f"{correct}/{total} ({pct:.1f}%)")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def verdict(th: float, s: dict, baseline: dict) -> str:
    if th == 0.0:
        return "Baseline — no gating."
    d_em = s["exact_match_pct"] - baseline["exact_match_pct"]
    d_pr = s["precision_pct"] - baseline["precision_pct"]
    return f"Δ exact match {d_em:+.2f}pp, Δ precision {d_pr:+.2f}pp — dropped {baseline['text_returned']-s['text_returned']} low-conf results."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    results = data["annotation_results"]
    exp_id = data.get("exp_id", "UNKNOWN")

    rows: list[tuple[float, dict]] = []
    for th in THRESHOLDS:
        filtered = apply_threshold(results, th)
        rows.append((th, compute_stats(filtered)))

    baseline_stats = rows[0][1]
    reported = data.get("accuracy", {})
    parity_note = (
        f"Parity check: reported exact_match_pct={reported.get('exact_match_pct')} "
        f"precision_pct={reported.get('precision_pct')} text_returned={data.get('ocr_returned_text')} | "
        f"recomputed exact_match_pct={baseline_stats['exact_match_pct']:.2f} "
        f"precision_pct={baseline_stats['precision_pct']:.2f} text_returned={baseline_stats['text_returned']}"
    )

    summary_table = format_table(rows)
    bucket_table = format_bucket_table(rows)

    out_lines = [
        f"# Confidence Threshold Sweep — {exp_id}",
        "",
        f"Input: `{Path(args.input).name}`",
        "",
        parity_note,
        "",
        "## Summary",
        "",
        summary_table,
        "",
        "## Per-aspect-ratio bucket (correct / total)",
        "",
        bucket_table,
        "",
        "## Verdicts",
        "",
    ]
    for th, s in rows:
        th_label = "ungated" if th == 0.0 else f"≥{th:.2f}"
        out_lines.append(f"- **{th_label}**: {verdict(th, s, baseline_stats)}")

    out_md = "\n".join(out_lines) + "\n"

    print(out_md)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(out_md, encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
