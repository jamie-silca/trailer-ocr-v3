"""
Experiment Comparison Report Generator
=======================================
Reads all benchmark JSON result files and generates a single human-readable
comparison report with detailed stats for every experiment.

Usage:
    python tests/compare_experiments.py
    python tests/compare_experiments.py --output docs/experiment-comparison-report.md
"""

import argparse
import json
import statistics
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"

# Canonical result files for each experiment, in logical order
EXPERIMENTS = {
    "BASE-01": "benchmark_20260415_140905.json",
    "BASE-01-VERIFY": "benchmark_BASE-01-VERIFY_20260415_154909.json",
    "EXP-01": "benchmark_EXP-01_20260415_155642.json",
    "EXP-01B": "benchmark_EXP-01B_20260415_161242.json",
    "EXP-02": "benchmark_EXP-02_20260415_164715.json",
    "EXP-03": "benchmark_EXP-03_20260415_170620.json",
    "EXP-04": "benchmark_EXP-04_20260415_172240.json",
    "EXP-03+04": "benchmark_EXP-03-04-COMBO_20260415_172550.json",
    "EXP-03+04+06": "benchmark_EXP-03-04-06-COMBO_20260415_173040.json",
    "EXP-05-320": "benchmark_EXP-05-320_20260416_010336.json",
    "EXP-05-480": "benchmark_EXP-05-480_20260416_011759.json",
    "EXP-05-640": "benchmark_EXP-05-640_20260416_011951.json",
    "EXP-07": "benchmark_EXP-07_20260416_013420.json",
    "EXP-08": "benchmark_EXP-08_20260420_175409.json",
    "EXP-09": "benchmark_EXP-09_20260420_175919.json",
    "EXP-10": "benchmark_EXP-10_20260420_180153.json",
    "EXP-09+10": "benchmark_EXP-09-10_20260420_180525.json",
}

EXPERIMENT_DESCRIPTIONS = {
    "BASE-01": "Canonical baseline — all PaddleOCR defaults, no preprocessing",
    "BASE-01-VERIFY": "Baseline re-run with enhanced harness (verification)",
    "EXP-01": "Auto-rotate portrait crops 90° CW",
    "EXP-01B": "Rotate portrait crops 90° CW + upscale to min 80px height",
    "EXP-02": "CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=8×8)",
    "EXP-03": "Pad small crops to min 64px per side (neutral grey border)",
    "EXP-04": "Lower detection thresholds (db_thresh=0.2, box_thresh=0.3, unclip=2.0)",
    "EXP-03+04": "Pad + lower detection thresholds (combined)",
    "EXP-03+04+06": "Pad + lower thresholds + character substitution post-processing",
    "EXP-05-320": "det_limit_side_len = 320",
    "EXP-05-480": "det_limit_side_len = 480",
    "EXP-05-640": "det_limit_side_len = 640",
    "EXP-07": "Two-pass OCR for portrait crops (original + 90cw + 90ccw, best confidence)",
    "EXP-08": "Improved positional postprocessing (digit-in-prefix fix, H→8 in digit context)",
    "EXP-09": "10% YOLO bbox expansion before crop (pixel context from real image edges)",
    "EXP-10": "Cascade retry: on no-text, retry with sharpen+dilate fallback preprocessing",
    "EXP-09+10": "Bbox expansion + cascade retry (combined)",
}

# 2-3 sentences per experiment: what changed, why, and expected effect.
EXPERIMENT_HYPOTHESES = {
    "BASE-01": (
        "No changes applied — PaddleOCR PP-OCRv4 with all default parameters and no image preprocessing. "
        "This run establishes the canonical accuracy and speed baseline that all subsequent experiments are compared against."
    ),
    "BASE-01-VERIFY": (
        "Identical configuration to BASE-01, re-run using the enhanced benchmark harness (with --exp-id support, subset breakdowns, and config recording). "
        "The purpose is to verify that the new harness produces identical accuracy numbers to the original baseline, confirming the harness itself introduces no side effects."
    ),
    "EXP-01": (
        "Portrait-oriented crops (width < height) are rotated 90 degrees clockwise before OCR using a lossless PIL transpose. "
        "The hypothesis was that ~156 portrait crops contain horizontal trailer ID text that appears vertical due to crop orientation, and PaddleOCR's angle classifier (which handles 0/180 degree flips) cannot correct 90-degree rotations. "
        "Expected a +10-15pp accuracy improvement on the portrait subset with negligible speed cost."
    ),
    "EXP-01B": (
        "Extends EXP-01 by also upscaling portrait crops after rotation so the resulting image height reaches at least 80px, using PIL Lanczos interpolation. "
        "EXP-01 failed because rotated portrait crops were only 31-58px tall — too short for PaddleOCR's recognition stage to discriminate characters. "
        "The hypothesis was that bilinear upscaling to a minimum readable height would give the model enough pixel data to recognise the text."
    ),
    "EXP-02": (
        "Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast: the crop is converted to grayscale, CLAHE is applied (clipLimit=2.0, 8x8 tile grid), then converted back to 3-channel RGB before OCR. "
        "The hypothesis was that faded paint, shadows, and aerial haze reduce contrast between text and background, causing PaddleOCR's detection stage to miss text regions. "
        "CLAHE should have boosted local contrast enough for the DB detector to find previously-invisible text, expecting +3-8pp accuracy with minimal speed cost."
    ),
    "EXP-03": (
        "Crops with either dimension below 64px are padded with a neutral grey (128,128,128) border using PIL ImageOps.expand, bringing the minimum dimension to 64px. "
        "The hypothesis was that very small crops cause PaddleOCR's detection CNN to fast-fail because text features are too small after the model's internal resize step, and that providing additional spatial context via padding would allow the detector to find text regions it previously missed. "
        "Expected a modest +1-3pp improvement concentrated in the small crop bucket, with negligible speed cost."
    ),
    "EXP-04": (
        "Lowers three PaddleOCR detection parameters from their defaults: det_db_thresh from 0.3 to 0.2 (pixel binarization sensitivity), det_db_box_thresh from 0.5 to 0.3 (minimum box confidence to keep a detection), and det_db_unclip_ratio from 1.5 to 2.0 (box expansion factor). "
        "The hypothesis was that PaddleOCR's default thresholds are calibrated for high-contrast document and scene text, and aerial drone imagery with lower contrast needs more lenient thresholds to detect faint text regions. "
        "Expected +3-7pp accuracy with a moderate speed increase (+10-30ms median) due to more candidate regions passing through the recognition stage."
    ),
    "EXP-03+04": (
        "Combines EXP-03 (crop padding) and EXP-04 (lowered detection thresholds) in a single run to test whether their effects compound. "
        "The rationale is that padding gives the detector more spatial context while lowered thresholds make the detector more sensitive — addressing two different failure modes simultaneously. "
        "Expected at least additive improvement (+10pp or more) if the two techniques target non-overlapping failure populations."
    ),
    "EXP-03+04+06": (
        "Adds EXP-06 post-processing character substitution on top of the EXP-03+04 combo: after OCR returns text, domain-specific rules correct known character confusions (O to 0, I to 1, S to 5 in numeric contexts; reverse in alpha contexts) for trailer ID formats like JBHU 235644. "
        "The hypothesis was that ~38 wrong-text results in the baseline are caused by systematic letter/digit confusion that can be corrected with conservative regex-based pattern matching. "
        "Expected +2-4pp marginal improvement on top of EXP-03+04, with zero speed cost since it is pure string post-processing."
    ),
    "EXP-05-320": (
        "Changes PaddleOCR's det_limit_side_len from the default 960 to 320 — this controls the maximum side length the detection model resizes input to before running the DB text detector. "
        "The hypothesis was that the default 960 upscales small crops (~140x70px) by approximately 5.5x for detection, potentially introducing interpolation artefacts at an unexpected scale for the model. A smaller limit (320) would reduce the upscale factor to ~1.8x, which might better match the model's training distribution. "
        "Expected uncertain impact (+/- 2-5pp accuracy) with faster speed due to less computation in the detection CNN."
    ),
    "EXP-05-480": (
        "Same as EXP-05-320 but with det_limit_side_len set to 480, testing a middle ground between 320 and the default 960. "
        "The hypothesis was that 480 might be the sweet spot: enough resolution for the detection model to find text features without the excessive upscaling of 960. "
        "Expected to perform similarly to or slightly better than 320, with moderate speed improvement."
    ),
    "EXP-05-640": (
        "Same as EXP-05-320 but with det_limit_side_len set to 640, the highest of the three test values. "
        "The hypothesis was that 640 provides higher resolution for detection while still avoiding the full 5.5x upscale of the default 960, potentially balancing text feature clarity against interpolation noise. "
        "Expected similar accuracy to other EXP-05 variants with speed between 320 and 960."
    ),
    "EXP-07": (
        "For portrait-oriented crops, runs OCR three times — at the original orientation, rotated 90 degrees clockwise, and rotated 90 degrees counter-clockwise — and returns whichever result has the highest confidence score. "
        "The hypothesis was that some portrait crops may contain text oriented differently (rotated CW vs CCW), and a multi-orientation approach would be more robust than EXP-01's single fixed rotation by covering all possible orientations. "
        "Expected +1-3pp beyond EXP-01 on the portrait subset, at the cost of up to 3x slower processing for the 156 portrait crops."
    ),
    "EXP-08": (
        "Adds an improved positional character substitution pass on top of EXP-03+04+06: digits in the 4-character alpha prefix are forced to their nearest letter (0->O, 1->I, 5->S, 6->G, 8->B), and H is added to the digit-context substitution table (H->8). "
        "OcrProcessor already applies EXP-06 character substitution internally; this experiment runs an additional pass with the expanded ruleset to catch cases the conservative EXP-06 rules miss. "
        "Expected +0.5-2pp marginal improvement with zero speed cost."
    ),
    "EXP-09": (
        "Each YOLO bounding box is expanded by 10% on each side before cropping, so the crop includes a ring of real image pixels around the annotated region instead of the neutral-grey padding added by EXP-03. "
        "The hypothesis was that EXP-03's neutral-grey padding provides spatial context that prevents the detection stage from fast-failing, but real surrounding pixels should provide richer features — especially for trailer IDs that bleed close to the bbox edge. "
        "Expected +2-5pp improvement on the wide and landscape subsets, with a slight speed increase from larger crops."
    ),
    "EXP-10": (
        "Implements a two-pass cascade: the first pass runs the standard EXP-03+04+06 pipeline; if no text is detected, a fallback retry applies PIL UnsharpMask sharpening and morphological dilation (2x2 kernel, 1 iteration) to thicken and sharpen strokes before retrying OCR. "
        "The hypothesis was that the 244 no-text crops contain faint or blurry text that sharpening and stroke thickening could make legible, and that restricting the expensive fallback to the failing 36% of crops keeps the average latency acceptable. "
        "Expected +1-4pp accuracy from rescued crops, at a cost of ~20-40ms average latency increase."
    ),
    "EXP-09+10": (
        "Combines EXP-09 (10% bbox expansion) and EXP-10 (cascade sharpen+dilate retry) to test whether their effects compound. "
        "The rationale is that expanded crops reduce the no-text rate, so the cascade operates on a smaller pool of harder failures — potentially making its signal-to-noise better than EXP-10 alone. "
        "Expected at least additive improvement (+3.5-6pp) if the two techniques target different sub-populations of failing crops."
    ),
}

EXPERIMENT_VERDICTS = {
    "BASE-01": "BASELINE",
    "BASE-01-VERIFY": "VERIFICATION",
    "EXP-01": "REJECTED",
    "EXP-01B": "REJECTED",
    "EXP-02": "REJECTED",
    "EXP-03": "ACCEPTED",
    "EXP-04": "ACCEPTED",
    "EXP-03+04": "BEST CONFIG",
    "EXP-03+04+06": "ACCEPTED (marginal)",
    "EXP-05-320": "NEUTRAL",
    "EXP-05-480": "NEUTRAL",
    "EXP-05-640": "NEUTRAL",
    "EXP-07": "REJECTED",
    "EXP-08": "NEUTRAL",
    "EXP-09": "NEW BEST CONFIG",
    "EXP-10": "ACCEPTED (marginal)",
    "EXP-09+10": "ACCEPTED (marginal)",
}

EXPERIMENT_RESULT_SUMMARIES = {
    "BASE-01": "Establishing a baseline accuracy of 22.3%, this run highlights that nearly 60% of annotations (primarily small or vertical crops) are ignored by default.",
    "BASE-01-VERIFY": "Successfully validated the new benchmark harness consistency by reproducing identical accuracy results (22.3%), while noting minor speed variability likely due to environmental noise.",
    "EXP-01": "Proved that 90° rotation alone does not unlock portrait text, as the resulting horizontal images—often under 40px tall—remain below the recognition model's effective threshold.",
    "EXP-01B": "Demonstrated that Lanczos upscaling of rotated crops is counterproductive; interpolation fails to reconstruct the fine stroke detail required for confident character discrimination.",
    "EXP-02": "Catastrophic failure (-9.9pp accuracy) suggests that CLAHE-induced noise or boundary artifacts severely confuse the DB detector on drone-derived crops.",
    "EXP-03": "Outstanding success (+8.1pp) proving that the \"small crop\" problem is primarily a context/padding issue for the detector rather than an inherent recognition limit.",
    "EXP-04": "Successfully increased recall by +2.3pp, capturing \"borderline\" but correct detections that the default thresholds were aggressively discarding.",
    "EXP-03+04": "**Best Performance (Round 1):** Verified a highly effective additive relationship between padding and sensitivity, raising baseline accuracy by over 50% (from 22.3% to 34.7%).",
    "EXP-03+04+06": "Adding character substitution provided a marginal +0.1pp gain, indicating that detection failures are a significantly higher priority than character-level confusion.",
    "EXP-05-320": "Neutral results across all resize limits indicate that PaddleOCR's internal upscaling to 960px is robust and not a source of performance degradation for small crops.",
    "EXP-05-480": "Neutral results across all resize limits indicate that PaddleOCR's internal upscaling to 960px is robust and not a source of performance degradation for small crops.",
    "EXP-05-640": "Neutral results across all resize limits indicate that PaddleOCR's internal upscaling to 960px is robust and not a source of performance degradation for small crops.",
    "EXP-07": "Confirmed that even a comprehensive 3-pass multi-orientation approach cannot overcome the resolution bottleneck for portraits, while adding an unacceptable 16% speed penalty.",
    "EXP-08": "Running postprocess_v2 on top of OcrProcessor's already-applied EXP-06 rules provides no net gain; double-postprocessing is idempotent for well-matched text and marginally harmful where the expanded ruleset applies conflicting substitutions. The improved rules need to replace, not stack on, the existing substitution.",
    "EXP-09": "**New Best Config:** Providing real image context via 10% bbox expansion is more effective than neutral-grey padding (EXP-03). The wider crops give the DB detector richer surrounding features, boosting wide-format accuracy from 46.9% to 51.4% and delivering a net +3.4pp gain with no precision loss.",
    "EXP-10": "Cascade retry rescued 9 previously-blank crops, but 8 of the 9 were wrong predictions — a 11% success rate on the hardest failures. Sharpen+dilate finds signals that the model cannot reliably decode, adding noise more than signal. The speed overhead (+18ms median) is also unattractive for only +0.2pp accuracy at the cost of -0.9pp precision.",
    "EXP-09+10": "Combining bbox expansion with cascade retry compounded to +3.6pp accuracy (vs +3.4pp for EXP-09 alone), adding exactly 1 more correct prediction. However, the cascade's drag on precision (57.0% vs EXP-09's 57.6%) and +31ms median latency penalty make EXP-09 alone the recommended production config unless raw recall is the priority.",
}


def aspect_ratio_bucket(w: int, h: int) -> str:
    if h == 0:
        return "invalid"
    ratio = w / h
    if ratio < 0.5:
        return "portrait"
    elif ratio < 1.0:
        return "near_square"
    elif ratio < 2.0:
        return "landscape"
    elif ratio < 4.0:
        return "wide"
    else:
        return "very_wide"


def compute_stats(data: dict) -> dict:
    """Compute uniform stats from any result JSON (old or new format)."""
    results = data["annotation_results"]
    total = len(results)

    # Speed
    times = [r["elapsed_ms"] for r in results]
    sorted_times = sorted(times)
    speed = {
        "average": round(statistics.mean(times), 2),
        "median": round(statistics.median(times), 2),
        "stdev": round(statistics.stdev(times), 2) if len(times) > 1 else 0,
        "min": round(min(times), 2),
        "max": round(max(times), 2),
        "p90": round(sorted_times[int(len(sorted_times) * 0.90)], 2),
        "p95": round(sorted_times[int(len(sorted_times) * 0.95)], 2),
        "p99": round(sorted_times[int(len(sorted_times) * 0.99)], 2),
    }

    # Accuracy
    text_returned = [r for r in results if r.get("ocr_text")]
    no_text = [r for r in results if not r.get("ocr_text")]
    correct = [r for r in results if r.get("exact_match") is True]
    wrong = [r for r in results if r.get("exact_match") is False and r.get("ocr_text")]

    accuracy = {
        "total": total,
        "text_returned": len(text_returned),
        "text_returned_pct": round(100 * len(text_returned) / total, 1),
        "no_text": len(no_text),
        "no_text_pct": round(100 * len(no_text) / total, 1),
        "correct": len(correct),
        "correct_pct": round(100 * len(correct) / total, 1),
        "wrong": len(wrong),
        "wrong_pct": round(100 * len(wrong) / total, 1),
        "precision_pct": round(100 * len(correct) / len(text_returned), 1) if text_returned else 0,
    }

    # Subset stats by aspect ratio
    ar_buckets: dict[str, dict] = {}
    for r in results:
        crop_w, crop_h = r.get("crop_size", [0, 0])
        bucket = r.get("aspect_ratio_bucket", aspect_ratio_bucket(crop_w, crop_h))
        if bucket not in ar_buckets:
            ar_buckets[bucket] = {"total": 0, "correct": 0, "text_returned": 0}
        ar_buckets[bucket]["total"] += 1
        if r.get("ocr_text"):
            ar_buckets[bucket]["text_returned"] += 1
        if r.get("exact_match") is True:
            ar_buckets[bucket]["correct"] += 1

    subsets = {}
    for k, v in sorted(ar_buckets.items()):
        t = v["total"]
        subsets[k] = {
            "total": t,
            "correct": v["correct"],
            "correct_pct": round(100 * v["correct"] / t, 1) if t else 0,
            "text_returned": v["text_returned"],
            "text_returned_pct": round(100 * v["text_returned"] / t, 1) if t else 0,
        }

    return {
        "speed": speed,
        "accuracy": accuracy,
        "subsets": subsets,
        "wall_time_s": data.get("total_wall_time_s", round(sum(times) / 1000, 1)),
        "throughput": data.get("throughput_ann_per_s", round(total / (sum(times) / 1000), 2)),
        "warmup_s": data.get("warmup_time_s", "N/A"),
    }


def delta_str(val: float, base: float, unit: str = "", higher_is_better: bool = True) -> str:
    diff = val - base
    if abs(diff) < 0.05:
        return "—"
    sign = "+" if diff > 0 else ""
    indicator = ""
    if higher_is_better:
        indicator = " ▲" if diff > 0 else " ▼"
    else:
        indicator = " ▼" if diff > 0 else " ▲"
    return f"{sign}{diff:.1f}{unit}{indicator}"


def generate_report(output_path: str | None = None):
    all_stats = {}
    for exp_id, filename in EXPERIMENTS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {exp_id}")
            continue
        with open(path) as f:
            data = json.load(f)
        all_stats[exp_id] = compute_stats(data)

    base = all_stats.get("BASE-01")
    if not base:
        print("ERROR: BASE-01 not found")
        return

    # Load EXP-11 temporal aggregation results if available
    exp11_data = None
    exp11_files = sorted(RESULTS_DIR.glob("benchmark_EXP-11_*.json"), reverse=True)
    if exp11_files:
        with open(exp11_files[0]) as f:
            exp11_data = json.load(f)

    lines: list[str] = []
    def w(s: str = ""):
        lines.append(s)

    w("# OCR Experiment Comparison Report")
    w()
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"Dataset: 672 annotations across 397 aerial drone frames")
    w(f"Engine: PaddleOCR 2.7.3 (PP-OCRv4), CPU-only")
    w()

    # ── Quick comparison table ──
    w("---")
    w()
    w("## Quick Comparison")
    w()
    w("| Experiment | Verdict | Text Returned | Correct (Exact Match) | Exact Match % | Δ | Precision | Median ms | Δ Speed |")
    w("|---|---|---|---|---|---|---|---|---|")
    for exp_id in EXPERIMENTS:
        if exp_id not in all_stats:
            continue
        s = all_stats[exp_id]
        a = s["accuracy"]
        sp = s["speed"]
        verdict = EXPERIMENT_VERDICTS.get(exp_id, "")
        d_acc = delta_str(a["correct_pct"], base["accuracy"]["correct_pct"], "pp")
        d_spd = delta_str(sp["median"], base["speed"]["median"], "ms", higher_is_better=False)
        w(f"| {exp_id} | {verdict} | {a['text_returned']} ({a['text_returned_pct']}%) | {a['correct']}/{a['total']} | {a['correct_pct']}% | {d_acc} | {a['precision_pct']}% | {sp['median']} | {d_spd} |")
    # EXP-11 is a different metric (per-track, not per-annotation) — add as a note row
    if exp11_data:
        pt = exp11_data["per_track_accuracy"]["all_tracks"]
        w(f"| EXP-11 | SPECIAL ANALYSIS | — | {pt['correct']}/{pt['total']} tracks | {pt['correct_pct']}% per-track | — | {pt['precision_pct']}% | — (no re-OCR) | — |")
    w()

    # ── Detailed per-experiment sections ──
    w("---")
    w()
    w("## Detailed Experiment Results")
    w()

    for exp_id in EXPERIMENTS:
        if exp_id not in all_stats:
            continue
        s = all_stats[exp_id]
        a = s["accuracy"]
        sp = s["speed"]
        desc = EXPERIMENT_DESCRIPTIONS.get(exp_id, "")
        verdict = EXPERIMENT_VERDICTS.get(exp_id, "")

        w(f"### {exp_id} — {desc}")
        w()
        w(f"**Verdict: {verdict}**")
        w()
        hypothesis = EXPERIMENT_HYPOTHESES.get(exp_id, "")
        if hypothesis:
            w(f"> {hypothesis}")
            w()

        # Accuracy (before speed, matching original report format)
        w("#### Accuracy")
        w()
        w("| Metric | Value | Δ vs BASE-01 |")
        w("|---|---|---|")
        w(f"| **Text returned** | **{a['text_returned']} / {a['total']} ({a['text_returned_pct']}%)** | **{delta_str(a['text_returned_pct'], base['accuracy']['text_returned_pct'], 'pp')}** |")
        w(f"| → Correct (exact match) | {a['correct']} ({a['correct_pct']}%) | {delta_str(a['correct_pct'], base['accuracy']['correct_pct'], 'pp')} |")
        w(f"| → Wrong text | {a['wrong']} ({a['wrong_pct']}%) | {delta_str(a['wrong_pct'], base['accuracy']['wrong_pct'], 'pp', False)} |")
        w(f"| No text returned | {a['no_text']} ({a['no_text_pct']}%) | {delta_str(a['no_text_pct'], base['accuracy']['no_text_pct'], 'pp', False)} |")
        w(f"| **Precision** (correct / returned) | **{a['precision_pct']}%** | **{delta_str(a['precision_pct'], base['accuracy']['precision_pct'], 'pp')}** |")
        w()

        # Subset breakdown
        w("#### Breakdown by Aspect Ratio")
        w()
        w("| Bucket | Total | Correct | Accuracy | Δ vs BASE-01 | Text Returned |")
        w("|---|---|---|---|---|---|")
        for bucket in ["portrait", "near_square", "landscape", "wide", "very_wide"]:
            sub = s["subsets"].get(bucket, {"total": 0, "correct": 0, "correct_pct": 0, "text_returned": 0, "text_returned_pct": 0})
            base_sub = base["subsets"].get(bucket, {"correct_pct": 0})
            d = delta_str(sub["correct_pct"], base_sub["correct_pct"], "pp")
            w(f"| {bucket} | {sub['total']} | {sub['correct']} | {sub['correct_pct']}% | {d} | {sub['text_returned']} ({sub['text_returned_pct']}%) |")
        w()

        # Speed
        w("#### Speed (per annotation, OCR call only)")
        w()
        w("| Metric | Value | Δ vs BASE-01 |")
        w("|---|---|---|")
        w(f"| Average | {sp['average']} ms | {delta_str(sp['average'], base['speed']['average'], 'ms', False)} |")
        w(f"| **Median** | **{sp['median']} ms** | **{delta_str(sp['median'], base['speed']['median'], 'ms', False)}** |")
        w(f"| Std dev | {sp['stdev']} ms | {delta_str(sp['stdev'], base['speed']['stdev'], 'ms', False)} |")
        w(f"| Min | {sp['min']} ms | |")
        w(f"| Max | {sp['max']} ms | |")
        w(f"| p90 | {sp['p90']} ms | {delta_str(sp['p90'], base['speed']['p90'], 'ms', False)} |")
        w(f"| p95 | {sp['p95']} ms | {delta_str(sp['p95'], base['speed']['p95'], 'ms', False)} |")
        w(f"| p99 | {sp['p99']} ms | {delta_str(sp['p99'], base['speed']['p99'], 'ms', False)} |")
        w(f"| Wall time | {s['wall_time_s']}s | |")
        w(f"| Throughput | {s['throughput']} ann/s | |")
        w()

        summary = EXPERIMENT_RESULT_SUMMARIES.get(exp_id, "")
        if summary:
            w(f"**Result Summary:** {summary}")
            w()

        w("---")
        w()

    # ── EXP-11: Temporal aggregation special section ──────────────────────────
    if exp11_data:
        w("## EXP-11 — Temporal Aggregation Analysis (post-hoc, no re-OCR)")
        w()
        w("**Verdict: SPECIAL ANALYSIS**")
        w()
        w("> Groups all annotations that share the same ground-truth trailer ID (i.e. the same physical trailer appearing in multiple consecutive drone frames) and applies confidence-weighted majority voting to select one prediction per trailer. "
          "No additional OCR is needed — this runs post-hoc on the EXP-03+04+06 result JSON. "
          "The ground-truth grouping gives an upper bound on what a real visual tracker could achieve; in production, tracks would be formed by IoU-based bbox linking across adjacent frames.")
        w()

        pt_all = exp11_data["per_track_accuracy"]["all_tracks"]
        pt_multi = exp11_data["per_track_accuracy"]["multi_frame_only"]
        ts = exp11_data["track_summary"]

        w("#### Track Summary")
        w()
        w("| Metric | Value |")
        w("|---|---|")
        w(f"| Total unique trailers (tracks) | {ts['total_tracks']} |")
        w(f"| Single-frame tracks | {ts['single_frame']} (no temporal benefit possible) |")
        w(f"| Multi-frame tracks | {ts['multi_frame']} (>= 2 frames) |")
        w()

        w("#### Per-Track Accuracy (confidence-weighted majority vote)")
        w()
        w("| Scope | Correct | Total | Accuracy | Precision |")
        w("|---|---|---|---|---|")
        w(f"| All tracks | {pt_all['correct']} | {pt_all['total']} | {pt_all['correct_pct']}% | {pt_all['precision_pct']}% |")
        w(f"| Multi-frame tracks only | {pt_multi['correct']} | {pt_multi['total']} | {pt_multi['correct_pct']}% | {pt_multi['precision_pct']}% |")
        w(f"| Multi-frame per-annotation baseline | — | — | {pt_multi['per_ann_baseline_correct_pct']}% | — |")
        w()
        w(f"**Temporal gain on multi-frame trailers: +{pt_multi['correct_pct'] - pt_multi['per_ann_baseline_correct_pct']:.1f}pp** (from {pt_multi['per_ann_baseline_correct_pct']}% per-annotation to {pt_multi['correct_pct']}% per-track)")
        w()

        # Confidence sweep
        w("#### Confidence Threshold Sweep (per annotation, no re-OCR)")
        w()
        w("Dropping low-confidence results trades a small recall loss for significant precision gain.")
        w()
        w("| Min Confidence | Results Kept | Exact Match % | Precision |")
        w("|---|---|---|---|")
        for s in exp11_data["confidence_sweep"]:
            w(f"| >= {s['threshold']:.1f} | {s['kept']} ({s['kept_pct']}%) | {s['correct_pct']}% | {s['precision_pct']}% |")
        w()
        w("**Result Summary:** Confidence-weighted majority voting raises per-track accuracy from 38.2% to 48.1% on multi-frame trailers (+9.9pp). "
          "For 50% of trailers that appear in only one frame, temporal aggregation provides no benefit. "
          "A confidence threshold of 0.6 boosts precision from 54.7% to 61.5% at a cost of only -0.3pp exact-match accuracy — a favourable trade for applications where wrong IDs are more costly than missed ones.")
        w()
        w("---")
        w()

    report = "\n".join(lines)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Report written to: {out}")
    else:
        print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment comparison report")
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: print to stdout)")
    args = parser.parse_args()
    generate_report(args.output)
