"""
eval_against_gt.py
------------------
Runs the current trailer-ocr-v3 PaddleOCR configuration against a COCO-annotated
dataset and compares predictions to ground-truth text labels.

Usage:
    python tests/eval_against_gt.py \
        --images_dir "C:/Users/jamie.barker/Downloads/20260421/download-1776783672892" \
        --gt_json   "C:/Users/jamie.barker/Downloads/20260421/download-1776783672892/annotations_2026-04-20_12-50_coco_with_text.json" \
        --output_dir "tests/eval_results"

The script mirrors the production pipeline:
  - EXP-03: pad_small (min 64px)
  - EXP-04: det_db_thresh=0.2, det_db_box_thresh=0.3, det_db_unclip_ratio=2.0
  - EXP-06: postprocess_text (char substitution)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Mirror of production utils (no import from app package needed)
# ---------------------------------------------------------------------------

def pad_small(image: Image.Image, min_dim: int = 64, fill=(128, 128, 128)) -> Image.Image:
    w, h = image.size
    if w >= min_dim and h >= min_dim:
        return image
    pad_w = max(0, (min_dim - w) // 2)
    pad_h = max(0, (min_dim - h) // 2)
    return ImageOps.expand(image.convert("RGB"), border=(pad_w, pad_h, pad_w, pad_h), fill=fill)


import re
_CONTAINER_CODE_RE = re.compile(r'^([A-Z0-9]{4})\s?(\d{5,7})$', re.IGNORECASE)
_PURE_DIGITS_RE    = re.compile(r'^[\dOIlSBG ]{5,8}$', re.IGNORECASE)
_ALPHA_TO_DIGIT    = str.maketrans("OIlSBG", "011582")

def postprocess_text(text):
    if not text:
        return text
    stripped = text.strip()
    m = _CONTAINER_CODE_RE.match(stripped)
    if m:
        prefix = m.group(1).upper()
        suffix = m.group(2).translate(_ALPHA_TO_DIGIT)
        space  = " " if " " in stripped else ""
        return f"{prefix}{space}{suffix}"
    if _PURE_DIGITS_RE.match(stripped):
        candidate = stripped.replace(" ", "").translate(_ALPHA_TO_DIGIT)
        if candidate.isdigit():
            return candidate
    return stripped


def crop_from_bbox(image: Image.Image, bbox) -> Image.Image:
    """
    bbox is [x, y, width, height] in absolute pixel coords (COCO format).
    Returns cropped PIL Image.
    """
    x, y, w, h = bbox
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    iw, ih = image.size
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(iw, x2); y2 = min(ih, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def exact_match(pred: str, gt: str) -> bool:
    return pred.strip().upper() == gt.strip().upper()


def char_accuracy(pred: str, gt: str) -> float:
    """Per-character accuracy using simple alignment (min-length match)."""
    if not gt:
        return 1.0 if not pred else 0.0
    p = pred.strip().upper()
    g = gt.strip().upper()
    matches = sum(a == b for a, b in zip(p, g))
    return matches / max(len(g), len(p))


def levenshtein(s1: str, s2: str) -> int:
    s1, s2 = s1.upper(), s2.upper()
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j] + (c1 != c2), curr[j] + 1, prev[j + 1] + 1))
        prev = curr
    return prev[-1]


def normalized_edit_distance(pred: str, gt: str) -> float:
    if not gt and not pred:
        return 0.0
    dist = levenshtein(pred, gt)
    return dist / max(len(gt), len(pred), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eval current OCR config vs ground truth")
    parser.add_argument("--images_dir", required=True,
                        help="Directory containing the source JPEG images")
    parser.add_argument("--gt_json", required=True,
                        help="COCO JSON file with ground-truth text annotations")
    parser.add_argument("--output_dir", default="tests/eval_results",
                        help="Directory to write results CSV and report")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit evaluation to N images (for quick tests)")
    parser.add_argument("--conf_threshold", type=float, default=0.0,
                        help="Skip OCR results below this confidence")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    log = logging.getLogger("eval")

    # ---- Load GT ----
    log.info(f"Loading ground truth from: {args.gt_json}")
    with open(args.gt_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_info   = {img["id"]: img for img in coco["images"]}
    annotations   = coco["annotations"]

    # Filter to annotations that have a 'text' field
    text_annots = [a for a in annotations if a.get("text")]
    log.info(f"Total annotated regions with text: {len(text_annots)}")

    # Group by image
    from collections import defaultdict
    by_image = defaultdict(list)
    for a in text_annots:
        by_image[a["image_id"]].append(a)

    image_ids = sorted(by_image.keys())
    if args.max_images:
        image_ids = image_ids[: args.max_images]
        log.info(f"Limiting to {args.max_images} images → {len(image_ids)} images")

    # ---- Initialize PaddleOCR (EXP-04 config) ----
    log.info("Initializing PaddleOCR (EXP-04 config)...")
    _old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=False,
        show_log=False,
        det_db_thresh=0.2,
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=2.0
    )
    sys.argv = _old_argv
    log.info("PaddleOCR ready.")

    # ---- Evaluate ----
    results = []
    images_dir = Path(args.images_dir)

    total_annots   = 0
    exact_hits     = 0
    total_ned      = 0.0
    no_pred_count  = 0
    total_time_s   = 0.0

    for img_id in image_ids:
        img_info = images_info.get(img_id)
        if not img_info:
            log.warning(f"image_id {img_id} not found in images list, skipping.")
            continue

        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            log.warning(f"Image file not found: {img_path}, skipping.")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.error(f"Could not open {img_path}: {e}")
            continue

        log.info(f"Processing image [{img_id}]: {img_info['file_name']} "
                 f"({len(by_image[img_id])} annotations)")

        for annot in by_image[img_id]:
            gt_text  = annot["text"].strip()
            bbox     = annot["bbox"]   # [x, y, w, h] absolute pixels
            annot_id = annot["id"]

            crop = crop_from_bbox(image, bbox)
            if crop is None:
                log.warning(f"  annot {annot_id}: invalid crop, skipping.")
                continue

            # EXP-03: pad small crops
            crop = pad_small(crop)
            img_array = np.array(crop)

            t0 = time.perf_counter()
            try:
                raw = ocr.ocr(img_array, cls=True)
            except Exception as e:
                log.error(f"  annot {annot_id}: OCR error: {e}")
                raw = None
            elapsed = time.perf_counter() - t0
            total_time_s += elapsed

            # Parse PaddleOCR output
            pred_text = None
            ocr_conf  = 0.0
            if raw and raw[0]:
                texts = []
                confs = []
                for line in raw[0]:
                    if line and len(line) >= 2:
                        ti = line[1]
                        if isinstance(ti, (list, tuple)) and len(ti) >= 2:
                            texts.append(str(ti[0]))
                            confs.append(float(ti[1]))
                if texts:
                    pred_raw  = " ".join(texts)
                    ocr_conf  = sum(confs) / len(confs)
                    # EXP-06: postprocess
                    pred_text = postprocess_text(pred_raw)

            # Apply confidence threshold
            if pred_text and ocr_conf < args.conf_threshold:
                pred_text = None

            # Metrics
            if pred_text is None:
                no_pred_count += 1
                em  = False
                ned = 1.0
                ca  = 0.0
            else:
                em  = exact_match(pred_text, gt_text)
                ned = normalized_edit_distance(pred_text, gt_text)
                ca  = char_accuracy(pred_text, gt_text)

            if em:
                exact_hits += 1
            total_ned += ned
            total_annots += 1

            results.append({
                "annot_id":    annot_id,
                "image_id":    img_id,
                "filename":    img_info["file_name"],
                "gt_text":     gt_text,
                "pred_text":   pred_text if pred_text else "",
                "ocr_conf":    round(ocr_conf, 4),
                "exact_match": em,
                "ned":         round(ned, 4),
                "char_acc":    round(ca, 4),
                "time_ms":     round(elapsed * 1000, 1),
                "bbox_x":      bbox[0],
                "bbox_y":      bbox[1],
                "bbox_w":      bbox[2],
                "bbox_h":      bbox[3],
            })

            status = "✓" if em else "✗"
            log.info(f"  [{status}] annot {annot_id}: GT='{gt_text}' → PRED='{pred_text}' "
                     f"(conf={ocr_conf:.2f}, NED={ned:.3f})")

    # ---- Summary ----
    em_pct  = (exact_hits / total_annots * 100) if total_annots else 0
    avg_ned = (total_ned / total_annots) if total_annots else 0
    avg_ms  = (total_time_s / total_annots * 1000) if total_annots else 0

    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total annotations evaluated : {total_annots}")
    log.info(f"  No prediction (empty OCR)   : {no_pred_count}")
    log.info(f"  Exact match (case-insensitive): {exact_hits} / {total_annots}  ({em_pct:.1f}%)")
    log.info(f"  Avg Normalized Edit Distance : {avg_ned:.4f}")
    log.info(f"  Avg OCR time per crop        : {avg_ms:.1f} ms")
    log.info("=" * 60)

    # ---- Write outputs ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    import csv
    csv_path = out_dir / "eval_results.csv"
    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Results CSV written to: {csv_path}")

    # Summary JSON
    summary = {
        "config": {
            "det_db_thresh": 0.2,
            "det_db_box_thresh": 0.3,
            "det_db_unclip_ratio": 2.0,
            "use_angle_cls": True,
            "lang": "en",
            "pad_small_min_dim": 64,
            "postprocess_exp06": True,
        },
        "dataset": {
            "gt_json": args.gt_json,
            "images_dir": args.images_dir,
            "images_evaluated": len(image_ids),
            "total_annotations": total_annots,
        },
        "metrics": {
            "exact_match_count": exact_hits,
            "exact_match_pct": round(em_pct, 2),
            "no_prediction_count": no_pred_count,
            "avg_normalized_edit_distance": round(avg_ned, 4),
            "avg_time_per_crop_ms": round(avg_ms, 1),
        }
    }
    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary JSON written to: {summary_path}")

    # Failure cases (non-exact matches)
    failures = [r for r in results if not r["exact_match"]]
    fail_path = out_dir / "failures.csv"
    if failures:
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
        log.info(f"Failure cases ({len(failures)}) written to: {fail_path}")

    print("\n--- QUICK SUMMARY ---")
    print(f"Exact Match:  {exact_hits}/{total_annots}  ({em_pct:.1f}%)")
    print(f"Avg NED:      {avg_ned:.4f}")
    print(f"No Pred:      {no_pred_count}")
    print(f"Avg Crop ms:  {avg_ms:.1f}")
    print(f"\nFull results: {csv_path}")
    print(f"Summary:      {summary_path}")


if __name__ == "__main__":
    main()
