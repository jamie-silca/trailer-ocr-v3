"""
OCR Benchmark Script
====================
Runs PaddleOCR against all annotated bounding boxes in the dataset.
Measures speed and accuracy, and supports experiment preprocessing flags.

Dataset: tests/dataset/20260406
Annotation format: COCO - bbox = [x, y, width, height] in absolute pixels

Usage:
    # Baseline run
    python tests/benchmark_ocr.py

    # Named experiment with preprocessing
    python tests/benchmark_ocr.py --exp-id EXP-01 --preprocess rotate
    python tests/benchmark_ocr.py --exp-id EXP-02 --preprocess rotate,clahe
    python tests/benchmark_ocr.py --exp-id EXP-03 --preprocess rotate,clahe,pad

    # Multiple preprocessing flags (comma-separated, applied in order)
    python tests/benchmark_ocr.py --exp-id EXP-COMBO-01 --preprocess rotate,clahe,pad

Output files (in tests/results/):
    benchmark_{EXP_ID}_{YYYYMMDD_HHMMSS}.log   -- human-readable log
    benchmark_{EXP_ID}_{YYYYMMDD_HHMMSS}.json  -- machine-readable full results
"""

import argparse
import json
import sys
import time
import statistics
import logging
import platform
from pathlib import Path
from datetime import datetime
from PIL import Image

# Allow running from project root
PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
# Insert tests dir explicitly so `preprocessing` is found locally, not from
# any installed package named `tests` (e.g. ultralytics installs one)
sys.path.insert(0, str(TESTS_DIR))

# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR Benchmark")
    parser.add_argument(
        "--exp-id",
        default="BASE",
        help="Experiment ID (e.g. EXP-01). Included in output filenames.",
    )
    parser.add_argument(
        "--preprocess",
        default="",
        help="Comma-separated preprocessing flags to apply before OCR. "
             "Supported: rotate, rotate_scale, clahe, pad. E.g. --preprocess rotate,clahe",
    )
    # PaddleOCR detection threshold overrides (EXP-04 / EXP-05)
    parser.add_argument("--det-db-thresh", type=float, default=0.3,
                        help="PaddleOCR det_db_thresh (default 0.3)")
    parser.add_argument("--det-db-box-thresh", type=float, default=0.5,
                        help="PaddleOCR det_db_box_thresh (default 0.5)")
    parser.add_argument("--det-db-unclip-ratio", type=float, default=1.5,
                        help="PaddleOCR det_db_unclip_ratio (default 1.5)")
    parser.add_argument("--det-limit-side-len", type=int, default=960,
                        help="PaddleOCR det_limit_side_len (default 960)")
    parser.add_argument("--two-pass", action="store_true",
                        help="EXP-07: for portrait crops, run OCR at original + 90cw + 90ccw orientations "
                             "and return the highest-confidence result.")
    # argparse will only see the benchmark's own args; sys.argv isolation is
    # handled in OcrProcessor for PaddleOCR. We parse known args only so that
    # running via `python tests/benchmark_ocr.py` (no args) still works.
    args, _ = parser.parse_known_args()
    return args


# ── Config ────────────────────────────────────────────────────────────────────

ARGS = _parse_args()

DATASET_DIR = PROJECT_ROOT / "tests" / "dataset" / "20260406"
ANNOTATION_FILE = DATASET_DIR / "annotations_2026-04-06_09-06_coco_with_text.json"
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_ID = ARGS.exp_id.upper().replace(" ", "-")
_file_prefix = f"benchmark_{EXP_ID}_{RUN_TIMESTAMP}"
LOG_FILE = RESULTS_DIR / f"{_file_prefix}.log"
RESULTS_JSON = RESULTS_DIR / f"{_file_prefix}.json"

PREPROCESS_FLAGS: list[str] = [f.strip() for f in ARGS.preprocess.split(",") if f.strip()]

# ── Logging ───────────────────────────────────────────────────────────────────

def _configure_logging():
    """
    Apply log handlers to root logger.
    Called at startup AND again after OcrProcessor warm-up, because PaddleOCR's
    lazy model init resets root-logger handlers and level when it first loads.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    root.handlers.clear()
    # errors='replace' prevents UnicodeEncodeError on Windows cp1252 consoles
    sh = logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)
    )
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


_configure_logging()
logger = logging.getLogger("ocr_benchmark")

# Imports after logging setup (OcrProcessor import triggers PaddleOCR import)
from app.ocr_processor import OcrProcessor  # noqa: E402
from preprocessing import apply_preprocessing, postprocess_text, get_portrait_rotations  # noqa: E402

# ── OCR config patching for experiments that change PaddleOCR params ─────────
# If non-default detection params are passed, patch OcrProcessor._initialize
# so the singleton is created with the experiment's params.
# This keeps production ocr_processor.py untouched.

_DEFAULT_OCR_PARAMS = {
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.5,
    "det_db_unclip_ratio": 1.5,
    "det_limit_side_len": 960,
}

_ACTIVE_OCR_PARAMS = {
    "det_db_thresh": ARGS.det_db_thresh,
    "det_db_box_thresh": ARGS.det_db_box_thresh,
    "det_db_unclip_ratio": ARGS.det_db_unclip_ratio,
    "det_limit_side_len": ARGS.det_limit_side_len,
}

def _maybe_patch_ocr_processor():
    if _ACTIVE_OCR_PARAMS == _DEFAULT_OCR_PARAMS:
        return  # no patch needed
    from paddleocr import PaddleOCR as _PaddleOCR
    params = _ACTIVE_OCR_PARAMS

    def _patched_initialize(self):
        import sys as _sys
        _old_argv = _sys.argv
        _sys.argv = [_sys.argv[0]]
        self._ocr = _PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False,
            det_db_thresh=params["det_db_thresh"],
            det_db_box_thresh=params["det_db_box_thresh"],
            det_db_unclip_ratio=params["det_db_unclip_ratio"],
            det_limit_side_len=params["det_limit_side_len"],
        )
        _sys.argv = _old_argv

    OcrProcessor._initialize = _patched_initialize

_maybe_patch_ocr_processor()

# ── Helpers ───────────────────────────────────────────────────────────────────

def crop_absolute(image: Image.Image, bbox_abs: list) -> Image.Image | None:
    """
    Crop using COCO absolute bbox [x, y, width, height].
    Returns cropped PIL image or None if invalid.
    """
    x, y, w, h = bbox_abs
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    img_w, img_h = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def aspect_ratio_bucket(w: int, h: int) -> str:
    if h == 0:
        return "invalid"
    ratio = w / h
    if ratio < 0.5:
        return "portrait"       # very tall
    elif ratio < 1.0:
        return "near_square"    # slightly taller than wide
    elif ratio < 2.0:
        return "landscape"      # normal landscape
    elif ratio < 4.0:
        return "wide"           # wide landscape
    else:
        return "very_wide"      # extreme


def area_bucket(w: int, h: int) -> str:
    area = w * h
    if area < 2000:
        return "tiny"           # < 2000px²
    elif area < 5000:
        return "small"          # 2000–5000px²
    elif area < 15000:
        return "medium"         # 5000–15000px²
    else:
        return "large"          # > 15000px²


def _get_library_versions() -> dict:
    """Collect installed library versions for self-documenting JSON output."""
    versions = {}
    for lib in ("paddleocr", "paddlepaddle", "numpy", "PIL", "cv2"):
        try:
            if lib == "PIL":
                import PIL
                versions["pillow"] = PIL.__version__
            elif lib == "cv2":
                import cv2
                versions["opencv"] = cv2.__version__
            elif lib == "paddleocr":
                import paddleocr
                versions["paddleocr"] = getattr(paddleocr, "__version__", "unknown")
            elif lib == "paddlepaddle":
                import paddle
                versions["paddlepaddle"] = paddle.__version__
            elif lib == "numpy":
                import numpy
                versions["numpy"] = numpy.__version__
        except Exception:
            versions[lib] = "not_installed"
    return versions


def _get_ocr_config() -> dict:
    """
    Document the active PaddleOCR configuration for this run.
    Reflects any param overrides passed via CLI flags.
    """
    return {
        "engine": "PaddleOCR PP-OCRv4",
        "use_angle_cls": True,
        "lang": "en",
        "use_gpu": False,
        "show_log": False,
        "det_db_thresh": _ACTIVE_OCR_PARAMS["det_db_thresh"],
        "det_db_box_thresh": _ACTIVE_OCR_PARAMS["det_db_box_thresh"],
        "det_db_unclip_ratio": _ACTIVE_OCR_PARAMS["det_db_unclip_ratio"],
        "det_limit_side_len": _ACTIVE_OCR_PARAMS["det_limit_side_len"],
        "drop_score": 0.5,              # default
        "cls_thresh": 0.9,              # default
        "rec_image_shape": "3,48,320",  # default
        "note": "All values are PP-OCRv4 defaults; none are overridden in ocr_processor.py",
    }


# ── Subset stat helpers ───────────────────────────────────────────────────────

def _subset_stats(results: list[dict], label_key: str) -> dict:
    """
    Given annotation results, group by label_key and compute accuracy + count
    per bucket. Returns {bucket_value: {total, correct, pct, text_returned}}.
    """
    buckets: dict[str, dict] = {}
    for r in results:
        key = r.get(label_key, "unknown")
        if key not in buckets:
            buckets[key] = {"total": 0, "correct": 0, "text_returned": 0}
        buckets[key]["total"] += 1
        if r["ocr_text"]:
            buckets[key]["text_returned"] += 1
        if r.get("exact_match") is True:
            buckets[key]["correct"] += 1

    out = {}
    for k, v in sorted(buckets.items()):
        total = v["total"]
        correct = v["correct"]
        text_ret = v["text_returned"]
        out[k] = {
            "total": total,
            "correct": correct,
            "accuracy_pct": round(100 * correct / total, 1) if total else 0,
            "text_returned": text_ret,
            "text_returned_pct": round(100 * text_ret / total, 1) if total else 0,
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark():
    logger.info("=" * 70)
    logger.info("OCR BENCHMARK")
    logger.info(f"Experiment ID : {EXP_ID}")
    logger.info(f"Timestamp     : {RUN_TIMESTAMP}")
    logger.info(f"Preprocessing : {PREPROCESS_FLAGS if PREPROCESS_FLAGS else 'none'}")
    logger.info(f"Dataset       : {DATASET_DIR}")
    logger.info(f"Annotation    : {ANNOTATION_FILE.name}")
    logger.info(f"Log file      : {LOG_FILE}")
    logger.info(f"Results JSON  : {RESULTS_JSON}")
    logger.info("=" * 70)

    # Load annotations
    with open(ANNOTATION_FILE, encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]
    logger.info(f"Loaded {len(coco['images'])} images, {len(annotations)} annotations")

    has_ground_truth = any(ann.get("text") for ann in annotations)
    if has_ground_truth:
        logger.info("Ground truth text found — accuracy scoring enabled.")
    else:
        logger.warning(
            "No 'text' field in annotations — accuracy scoring SKIPPED. "
            "Results logged for manual review."
        )

    # Init OCR (warm-up happens here)
    logger.info("Initialising OcrProcessor (PaddleOCR warm-up)...")
    warmup_start = time.perf_counter()
    ocr = OcrProcessor()
    warmup_elapsed = time.perf_counter() - warmup_start
    # PaddleOCR model init resets root-logger — restore ours
    _configure_logging()
    logger.info(f"OcrProcessor ready in {warmup_elapsed:.2f}s")

    # Group annotations by image for efficient loading
    anns_by_image: dict[int, list] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Run OCR pass
    logger.info("-" * 70)
    logger.info("Starting OCR pass...")
    benchmark_start = time.perf_counter()

    annotation_results = []
    skipped_images = 0
    skipped_crops = 0

    for image_id, anns in anns_by_image.items():
        img_meta = images_by_id.get(image_id)
        if not img_meta:
            logger.warning(f"Image ID {image_id} not found in metadata, skipping {len(anns)} annotations")
            skipped_images += 1
            continue

        img_path = DATASET_DIR / img_meta["file_name"]
        if not img_path.exists():
            logger.warning(f"Image file not found: {img_path.name}, skipping {len(anns)} annotations")
            skipped_images += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open {img_path.name}: {e}")
            skipped_images += 1
            continue

        for ann in anns:
            ann_id = ann["id"]
            bbox = ann["bbox"]
            ground_truth = ann.get("text")

            crop = crop_absolute(image, bbox)
            if crop is None:
                logger.warning(f"  Ann {ann_id}: invalid crop bbox {bbox}, skipping")
                skipped_crops += 1
                continue

            orig_w, orig_h = crop.size

            # Apply experiment preprocessing
            processed_crop, preprocessing_applied = apply_preprocessing(crop, PREPROCESS_FLAGS)
            proc_w, proc_h = processed_crop.size

            # Time the OCR call (EXP-07: two-pass for portrait crops)
            t0 = time.perf_counter()
            if ARGS.two_pass and processed_crop.width < processed_crop.height:
                # Try original + 90cw + 90ccw; pick highest confidence
                candidates = get_portrait_rotations(processed_crop)
                best_text, best_conf, best_label = None, 0.0, "original"
                for cand_img, cand_label in candidates:
                    t, c = ocr.process_image(cand_img)
                    if c > best_conf:
                        best_text, best_conf, best_label = t, c, cand_label
                ocr_text, ocr_conf = best_text, best_conf
                if best_label != "original":
                    preprocessing_applied.append(f"two_pass_best:{best_label}")
            else:
                ocr_text, ocr_conf = ocr.process_image(processed_crop)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # EXP-06: post-processing character substitution
            if "postprocess" in PREPROCESS_FLAGS and ocr_text:
                ocr_text_raw = ocr_text
                ocr_text = postprocess_text(ocr_text)
                if ocr_text != ocr_text_raw:
                    preprocessing_applied.append(f"postprocessed:{repr(ocr_text_raw)}->{repr(ocr_text)}")

            # Accuracy
            exact_match = None
            if ground_truth is not None:
                exact_match = (ocr_text or "").strip().upper() == ground_truth.strip().upper()

            ar_bucket = aspect_ratio_bucket(orig_w, orig_h)
            a_bucket = area_bucket(orig_w, orig_h)

            result = {
                "annotation_id": ann_id,
                "image_id": image_id,
                "image_file": img_meta["file_name"],
                "bbox": bbox,
                "crop_size": [orig_w, orig_h],
                "aspect_ratio": round(orig_w / orig_h, 3) if orig_h else None,
                "aspect_ratio_bucket": ar_bucket,
                "area_bucket": a_bucket,
                "processed_size": [proc_w, proc_h],
                "preprocessing": preprocessing_applied,
                "ocr_text": ocr_text,
                "ocr_confidence": round(ocr_conf, 4) if ocr_conf else None,
                "elapsed_ms": round(elapsed_ms, 2),
                "ground_truth": ground_truth,
                "exact_match": exact_match,
            }
            annotation_results.append(result)

            status = "OK" if exact_match else ("XX" if exact_match is False else "--")
            conf_display = f"{ocr_conf:.2f}" if ocr_conf else "0.00"
            logger.info(
                f"  [{status}] Ann {ann_id:4d} | {img_meta['file_name']:40s} | "
                f"crop {orig_w:4d}x{orig_h:3d} ({ar_bucket:10s}) | "
                f"{elapsed_ms:7.1f}ms | "
                f"text={repr(ocr_text)[:40]:42s} conf={conf_display}"
            )

    total_elapsed = time.perf_counter() - benchmark_start

    # ── Statistics ─────────────────────────────────────────────────────────────
    if not annotation_results:
        logger.error("No annotations processed!")
        return

    times = [r["elapsed_ms"] for r in annotation_results]
    successful = [r for r in annotation_results if r["ocr_text"]]
    failed = [r for r in annotation_results if not r["ocr_text"]]

    avg_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    stdev_ms = statistics.stdev(times) if len(times) > 1 else 0
    min_ms = min(times)
    max_ms = max(times)
    sorted_times = sorted(times)
    p90 = sorted_times[int(len(sorted_times) * 0.90)]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]

    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Experiment ID               : {EXP_ID}")
    logger.info(f"  Preprocessing               : {PREPROCESS_FLAGS if PREPROCESS_FLAGS else 'none'}")
    logger.info(f"  Total annotations processed : {len(annotation_results)}")
    logger.info(f"  Skipped images              : {skipped_images}")
    logger.info(f"  Skipped crops (invalid bbox): {skipped_crops}")
    logger.info(f"  OCR returned text           : {len(successful)} ({100*len(successful)/len(annotation_results):.1f}%)")
    logger.info(f"  OCR returned no text        : {len(failed)} ({100*len(failed)/len(annotation_results):.1f}%)")
    logger.info("")
    logger.info("  SPEED - per annotation (OCR call only, excludes image load)")
    logger.info(f"    Average  : {avg_ms:8.2f} ms")
    logger.info(f"    Median   : {median_ms:8.2f} ms")
    logger.info(f"    Std dev  : {stdev_ms:8.2f} ms")
    logger.info(f"    Min      : {min_ms:8.2f} ms")
    logger.info(f"    Max      : {max_ms:8.2f} ms")
    logger.info(f"    p90      : {p90:8.2f} ms")
    logger.info(f"    p95      : {p95:8.2f} ms")
    logger.info(f"    p99      : {p99:8.2f} ms")
    logger.info("")
    logger.info(f"  OVERALL wall time           : {total_elapsed:.2f}s")
    logger.info(f"  Throughput                  : {len(annotation_results)/total_elapsed:.1f} annotations/s")
    logger.info("")

    # Accuracy
    if has_ground_truth:
        exact_matches = [r for r in annotation_results if r["exact_match"] is True]
        wrong = [r for r in annotation_results if r["exact_match"] is False and r["ocr_text"]]
        precision = len(exact_matches) / len(successful) * 100 if successful else 0
        logger.info("  ACCURACY (exact match, case-insensitive)")
        logger.info(f"    Correct   : {len(exact_matches)} / {len(annotation_results)} ({100*len(exact_matches)/len(annotation_results):.1f}%)")
        logger.info(f"    Wrong     : {len(wrong)} ({100*len(wrong)/len(annotation_results):.1f}%)")
        logger.info(f"    No text   : {len(failed)} ({100*len(failed)/len(annotation_results):.1f}%)")
        logger.info(f"    Precision : {precision:.1f}% (correct / text_returned)")
        logger.info("")
    else:
        logger.info("  ACCURACY: N/A — no ground truth text in annotations")
        logger.info("")

    # Per-subset breakdowns
    logger.info("  BREAKDOWN BY ASPECT RATIO BUCKET")
    ar_stats = _subset_stats(annotation_results, "aspect_ratio_bucket")
    for bucket, s in ar_stats.items():
        logger.info(
            f"    {bucket:12s} : {s['total']:4d} total | "
            f"{s['correct']:3d} correct ({s['accuracy_pct']:5.1f}%) | "
            f"{s['text_returned']:3d} returned text ({s['text_returned_pct']:5.1f}%)"
        )
    logger.info("")
    logger.info("  BREAKDOWN BY CROP AREA BUCKET")
    area_stats = _subset_stats(annotation_results, "area_bucket")
    for bucket, s in area_stats.items():
        logger.info(
            f"    {bucket:10s} : {s['total']:4d} total | "
            f"{s['correct']:3d} correct ({s['accuracy_pct']:5.1f}%) | "
            f"{s['text_returned']:3d} returned text ({s['text_returned_pct']:5.1f}%)"
        )
    logger.info("")
    logger.info("=" * 70)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    summary = {
        "exp_id": EXP_ID,
        "run_timestamp": RUN_TIMESTAMP,
        "dataset": str(DATASET_DIR),
        "annotation_file": ANNOTATION_FILE.name,
        # Full self-documenting config — so results are reproducible without
        # checking source code. The baseline only had a summary string.
        "ocr_config": _get_ocr_config(),
        "library_versions": _get_library_versions(),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "preprocessing_flags": PREPROCESS_FLAGS,
        "warmup_time_s": round(warmup_elapsed, 3),
        "total_wall_time_s": round(total_elapsed, 3),
        "total_processed": len(annotation_results),
        "skipped_images": skipped_images,
        "skipped_crops": skipped_crops,
        "ocr_returned_text": len(successful),
        "ocr_no_text": len(failed),
        "speed_ms": {
            "average": round(avg_ms, 2),
            "median": round(median_ms, 2),
            "stdev": round(stdev_ms, 2),
            "min": round(min_ms, 2),
            "max": round(max_ms, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
        },
        "throughput_ann_per_s": round(len(annotation_results) / total_elapsed, 2),
        "has_ground_truth": has_ground_truth,
    }

    if has_ground_truth:
        exact_matches = [r for r in annotation_results if r["exact_match"] is True]
        wrong = [r for r in annotation_results if r["exact_match"] is False and r["ocr_text"]]
        summary["accuracy"] = {
            "correct": len(exact_matches),
            "wrong_text": len(wrong),
            "no_text": len(failed),
            "total": len(annotation_results),
            "exact_match_pct": round(100 * len(exact_matches) / len(annotation_results), 2),
            "precision_pct": round(100 * len(exact_matches) / len(successful), 2) if successful else 0,
        }

    summary["subset_stats"] = {
        "by_aspect_ratio_bucket": ar_stats,
        "by_area_bucket": area_stats,
    }

    summary["annotation_results"] = annotation_results

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Full results saved to: {RESULTS_JSON}")
    logger.info(f"Log saved to         : {LOG_FILE}")


if __name__ == "__main__":
    run_benchmark()
