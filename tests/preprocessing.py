"""
OCR Experiment Preprocessing Functions
=======================================
All preprocessing transforms used in benchmark experiments.

These functions operate on PIL Images BEFORE passing to OcrProcessor.
They are kept here (not in app/ocr_processor.py) so that:
  - Production code stays unchanged during experimentation
  - Transforms are promoted to production only after proven

Benchmark usage:
    from tests.preprocessing import apply_preprocessing
    preprocessed = apply_preprocessing(crop, flags=["rotate", "clahe", "pad"])

Each function is safe to compose:
    image -> rotate -> clahe -> pad -> OcrProcessor.process_image()

IMPORTANT — what NOT to do here:
  - Do NOT resize to fixed dimensions (PaddleOCR does this internally)
  - Do NOT apply mean/std normalization (PaddleOCR normalizes internally)
  - Do NOT hard-binarize to pure B&W (PaddleOCR expects colour/grayscale)
"""

from __future__ import annotations

import re
from typing import Optional
import numpy as np
from PIL import Image, ImageOps, ImageFilter

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ── EXP-01: Auto-rotate portrait crops ───────────────────────────────────────

def rotate_portrait(image: Image.Image) -> tuple[Image.Image, bool]:
    """
    If the image is portrait-oriented (width < height), rotate 90° clockwise.
    Returns (image, was_rotated).

    Rationale: Trailer ID labels on the side of trailers are often vertical
    (portrait) in the crop. PaddleOCR's use_angle_cls handles 0°/180° but
    not 90° rotations. Rotating to landscape before OCR recovers ~147 crops
    that currently score near 0% accuracy.

    PIL ROTATE_270 = 90° clockwise (rotate CCW by 270 = CW by 90).
    This is a lossless pixel transpose — no interpolation.
    """
    if image.width < image.height:
        return image.transpose(Image.Transpose.ROTATE_270), True
    return image, False


# ── EXP-01B: Rotate + upscale portrait crops ─────────────────────────────────

def rotate_and_scale_portrait(
    image: Image.Image,
    min_height: int = 80,
) -> tuple[Image.Image, list[str]]:
    """
    For portrait crops (w < h): rotate 90° CW then upscale proportionally so
    the height reaches min_height pixels. Uses PIL LANCZOS (high-quality
    downsampling / upsampling, CPU-only, no model).

    This differs from EXP-01 (rotation only) in that it addresses the root
    cause discovered in EXP-01's failure: portrait crops are extreme thin
    strips (30-80px wide, 170-360px tall). After rotation, the height is only
    31-58px — too short for PaddleOCR's recognition stage to discriminate
    characters. Upscaling to min_height gives the model enough pixels.

    NOT "AI upscaling" — this is simple bilinear/lanczos interpolation,
    CPU-trivial and constraint-safe.
    """
    applied: list[str] = []
    if image.width >= image.height:
        return image, applied

    # Rotate 90° CW
    rotated = image.transpose(Image.Transpose.ROTATE_270)
    applied.append("rotated_90cw")

    # Upscale if height is below min_height
    rw, rh = rotated.size
    if rh < min_height:
        scale = min_height / rh
        new_w = int(rw * scale)
        new_h = min_height
        rotated = rotated.resize((new_w, new_h), Image.LANCZOS)
        applied.append(f"scaled_{rw}x{rh}_to_{new_w}x{new_h}")

    return rotated, applied


# ── EXP-02: CLAHE contrast enhancement ───────────────────────────────────────

def clahe(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    grayscale channel, then convert back to 3-channel RGB.

    Safe: CLAHE is enhancement, not normalization. PaddleOCR's internal
    pipeline applies mean/std normalization AFTER this, which is complementary.

    Requires: opencv-python-headless (already in requirements.txt).
    """
    if not _CV2_AVAILABLE:
        raise ImportError("opencv is required for CLAHE preprocessing. Install opencv-python-headless.")

    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe_obj.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


# ── EXP-03: Pad small crops ───────────────────────────────────────────────────

def pad_small(
    image: Image.Image,
    min_dim: int = 64,
    fill: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """
    If either dimension is below min_dim, pad symmetrically with a neutral
    gray border to reach min_dim.

    Rationale: Very small crops fast-fail because PaddleOCR's detection CNN
    can't find features after its internal resize. Padding gives more spatial
    context and prevents extreme downscaling artefacts.

    Neutral gray (128,128,128) is used rather than white/black to avoid
    introducing strong edges at the border that could fool the detector.
    """
    w, h = image.size
    if w >= min_dim and h >= min_dim:
        return image

    pad_w = max(0, (min_dim - w) // 2)
    pad_h = max(0, (min_dim - h) // 2)
    return ImageOps.expand(image.convert("RGB"), border=(pad_w, pad_h, pad_w, pad_h), fill=fill)


# ── EXP-06: Post-processing character substitution ───────────────────────────

# Trailer ID formats observed in dataset:
#   [A-Z]{4} \d{6}   e.g. "JBHU 235644", "JBHZ 092249"
#   \d{6}            e.g. "702524", "534600"
#   Mixed short      e.g. "NK523", "3RG23186"
#
# Character confusion observed in baseline (144 wrong-text cases):
#   0 <-> O  (11x)
#   2 <-> N  (11x)
#   3 <-> space (11x)
#   H <-> O  (8x)
#   1 <-> I  (8x)

_CONTAINER_CODE_RE = re.compile(r'^([A-Z0-9]{4})\s?(\d{5,7})$', re.IGNORECASE)
_PURE_DIGITS_RE = re.compile(r'^[\dOIlSBG ]{5,8}$', re.IGNORECASE)

# In a numeric context: substitute look-alike letters → digits
_ALPHA_TO_DIGIT = str.maketrans("OIlSBG", "011582")

# In an alpha context (4-char prefix): substitute look-alike digits → letters
_DIGIT_TO_ALPHA = str.maketrans("01", "OI")


def postprocess_text(text: Optional[str]) -> Optional[str]:
    """
    Apply conservative domain-specific character substitution to OCR output.

    Rules:
    1. If text matches container code format ([A-Z]{4} \d{5-7}):
       - Keep alpha prefix as-is (PaddleOCR rarely confuses alpha in alpha context)
       - In the numeric suffix: O→0, I→1, l→1, S→5, B→8, G→6
    2. If text looks like a pure numeric code (6-8 chars, mostly digits with
       common look-alikes): apply numeric substitution across whole string
    3. Otherwise: return unchanged (too risky to apply rules to unknown formats)

    Deliberately conservative: only apply to clearly identifiable formats.
    Unknown / short / ambiguous text is returned unchanged.
    """
    if not text:
        return text

    stripped = text.strip()

    # Rule 1: Container code format e.g. "JBHU 235644" or "JBHU235644"
    m = _CONTAINER_CODE_RE.match(stripped)
    if m:
        prefix = m.group(1).upper()
        suffix = m.group(2).translate(_ALPHA_TO_DIGIT)
        # Reconstruct with original spacing
        space = " " if " " in stripped else ""
        return f"{prefix}{space}{suffix}"

    # Rule 2: Looks like a pure numeric code with look-alike contamination
    if _PURE_DIGITS_RE.match(stripped):
        # Apply digit substitution only if result is plausibly all-numeric
        candidate = stripped.replace(" ", "").translate(_ALPHA_TO_DIGIT)
        if candidate.isdigit():
            return candidate

    return stripped


# ── EXP-08: Unsharp mask sharpening ──────────────────────────────────────────

def sharpen(image: Image.Image) -> Image.Image:
    """
    Apply PIL UnsharpMask to enhance edge contrast on faded/blurry text.
    radius=1 targets character-edge frequencies; percent=150 is a moderate
    boost; threshold=3 suppresses flat noise regions.

    Safer than CLAHE (EXP-02): operates on edges only, not full histogram,
    so it doesn't amplify background noise the way CLAHE did.
    """
    return image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))


# ── EXP-08: Morphological dilation ────────────────────────────────────────────

def dilate(image: Image.Image) -> Image.Image:
    """
    Apply a single iteration of 2×2 morphological dilation to thicken text
    strokes. Useful for faded paint where PaddleOCR's DB detector misses
    thin/broken strokes.

    Operates on each RGB channel independently — equivalent to a per-channel
    max-pool. Requires opencv; falls back to a PIL MaxFilter if unavailable.
    """
    if _CV2_AVAILABLE:
        arr = np.array(image.convert("RGB"))
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(arr, kernel, iterations=1)
        return Image.fromarray(dilated)
    else:
        return image.filter(ImageFilter.MaxFilter(3))


# ── EXP-08: Improved positional postprocessing ────────────────────────────────

# Expanded digit-in-alpha context substitution table.
# Original _DIGIT_TO_ALPHA only covered 0→O, 1→I.
# These visually-similar pairings cover common confusions in the prefix field.
_DIGIT_TO_ALPHA_V2 = str.maketrans("015689", "OISGBG")

# Expanded letter-in-digit context substitution table.
# Original covered OIlSBG; adding H (looks like 8 in aerial imagery).
_ALPHA_TO_DIGIT_V2 = str.maketrans("OIlSBGH", "0115826")

# Container/trailer code pattern: 4 alphanumeric chars then 5–7 digits.
_CONTAINER_CODE_RE_V2 = re.compile(r'^([A-Z0-9]{4})\s?([A-Z0-9]{5,7})$', re.IGNORECASE)
_PURE_DIGITS_RE_V2 = re.compile(r'^[\dOIlSBGH ]{5,8}$', re.IGNORECASE)


def postprocess_v2(text: Optional[str]) -> Optional[str]:
    """
    Improved positional character substitution (EXP-08).

    Enhancements over postprocess_text (EXP-06):
    1. Prefix (positions 1–4): substitute digits that look like letters
       (0→O, 1→I, 5→S, 6→G, 8→B, 9→G). EXP-06 left digits in prefix untouched.
    2. Suffix (positions 5+): also substitute H→8 (H looks like 8 in aerial
       imagery). EXP-06 did not include H.
    3. The 4-char pattern allows mixed alphanum in suffix for initial matching,
       then enforces positional substitution strictly.

    Conservative: only triggers on clearly-matched patterns.
    """
    if not text:
        return text

    stripped = text.strip()

    # Rule 1: Container/trailer code e.g. "JBHU 235644" or "JBH0235644"
    m = _CONTAINER_CODE_RE_V2.match(stripped)
    if m:
        # Prefix: letters where digits snuck in
        prefix = m.group(1).upper().translate(_DIGIT_TO_ALPHA_V2)
        # Suffix: digits where letters snuck in
        suffix = m.group(2).upper().translate(_ALPHA_TO_DIGIT_V2)
        space = " " if " " in stripped else ""
        return f"{prefix}{space}{suffix}"

    # Rule 2: Looks like a pure numeric code (with look-alike contamination)
    if _PURE_DIGITS_RE_V2.match(stripped):
        candidate = stripped.replace(" ", "").translate(_ALPHA_TO_DIGIT_V2)
        if candidate.isdigit():
            return candidate

    return stripped


# ── EXP-07: Two-pass OCR for portrait crops ───────────────────────────────────

def get_portrait_rotations(image: Image.Image) -> list[tuple[Image.Image, str]]:
    """
    For portrait crops, return candidate images to try in a two-pass OCR run.
    Returns list of (image, label) tuples.

    Only relevant if the crop is portrait-oriented (width < height).
    The caller runs OCR on each, then picks the result with highest confidence.
    """
    if image.width >= image.height:
        return [(image, "original")]

    return [
        (image, "original"),
        (image.transpose(Image.Transpose.ROTATE_270), "rotated_90cw"),
        (image.transpose(Image.Transpose.ROTATE_90), "rotated_90ccw"),
    ]


# ── EXP-13: Stacked-vertical portrait decoder ────────────────────────────────

# The "portrait" aspect-ratio bucket in benchmark_ocr.py is defined as w/h < 0.5,
# i.e. h > 2w. EXP-13 targets exactly that bucket: crops where trailer-ID letters
# are upright but stacked top-to-bottom (one letter per "line"). Previous
# portrait attempts (EXP-01/01B/07) wrongly assumed the text was rotated 90°;
# rotating upright stacked letters only makes things worse.
#
# Dataset format priors (from user):
#   JBHZ + 6 digits — always vertical (common)
#   JBHU + 6 digits — always vertical (less common)
#   R + 5 digits    — may be vertical
#   Plus other/random patterns.

def _quad_y_center(quad) -> float:
    """quad is an iterable of 4 (x, y) points (PaddleOCR detection output)."""
    return sum(float(pt[1]) for pt in quad) / 4.0


def _quad_bbox(quad) -> tuple[int, int, int, int]:
    """Axis-aligned bounding rectangle of a quadrilateral."""
    xs = [float(pt[0]) for pt in quad]
    ys = [float(pt[1]) for pt in quad]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _stitch_letters(
    letter_imgs: list[Image.Image],
    target_height: int = 48,
    gap_px: int = 4,
    fill: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """
    Normalise each letter image to target_height preserving aspect ratio, then
    paste side-by-side with a small neutral-gray gap. Produces the horizontal
    text-line shape that PaddleOCR's rec model was trained on.
    """
    resized: list[Image.Image] = []
    for li in letter_imgs:
        w, h = li.size
        if h <= 0 or w <= 0:
            continue
        new_h = target_height
        new_w = max(1, int(round(w * (target_height / h))))
        resized.append(li.convert("RGB").resize((new_w, new_h), Image.LANCZOS))

    if not resized:
        return Image.new("RGB", (target_height, target_height), fill)

    total_w = sum(img.width for img in resized) + gap_px * (len(resized) - 1)
    strip = Image.new("RGB", (total_w, target_height), fill)
    x = 0
    for img in resized:
        strip.paste(img, (x, 0))
        x += img.width + gap_px
    return strip


def _get_text_column_bounds(
    paddle_ocr,
    img_array: np.ndarray,
) -> Optional[tuple[int, int, int, int]]:
    """
    Use PaddleOCR's text_detector (a raw internal API that works reliably in
    2.7.x, unlike `ocr(det=True, rec=False)` which has a numpy-array bug) to
    find the tight axis-aligned bounding box of ALL detected text regions in
    the image. Returns (x1, y1, x2, y2) or None if detection fails / nothing
    found.

    For stacked-vertical portrait crops the detector typically returns ONE
    tall parallelogram covering the whole letter column, or a few. Unioning
    them gives us the text span, which we then uniformly slice.
    """
    try:
        dt_boxes, _ = paddle_ocr.text_detector(img_array)
    except Exception:  # noqa: BLE001
        return None
    if dt_boxes is None or len(dt_boxes) == 0:
        return None
    xs, ys = [], []
    for quad in dt_boxes:
        for pt in quad:
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
    if not xs:
        return None
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def decode_stacked_vertical(
    image: Image.Image,
    ocr_processor,
    variant: str = "stitch",
    min_chars: int = 3,
    char_height_px: int = 32,
    target_height: int = 48,
    gap_px: int = 4,
    side_padding: int = 3,
) -> tuple[Optional[str], float, list[str]]:
    """
    EXP-13 — Decode a portrait crop where upright letters are stacked
    top-to-bottom (one letter per row).

    Strategy:
      1. Use PaddleOCR's internal `text_detector` to find the vertical bounds
         of the text column (skipping background above/below the letters).
         We bypass the higher-level `ocr(det=True, rec=False)` API which has
         a known bug in 2.7.x (`ValueError: truth value of array ambiguous`).
      2. UNIFORMLY slice the text column into N ≈ height / char_height_px
         horizontal bands. We avoid projection-profile segmentation because
         inter-letter gaps are irregular — some letters are tightly packed
         with no background rows between them.
      3. Either:
           variant="stitch" — resize each band to target_height, paste side
             by side into a synthetic horizontal strip, run the full OCR
             pipeline on the strip. The rec model sees the horizontal text
             line shape it was trained on.
           variant="per_letter" — run OCR on each band independently and
             concatenate outputs top-to-bottom.

    Fallback: if the text column can't be located OR fewer than min_chars
    bands would be produced, returns (None, 0.0, ops) so the caller falls
    through to the standard pipeline.

    Parameters
    ----------
    image : PIL.Image
        Portrait-oriented crop (expected h > 2w; caller should gate).
    ocr_processor : OcrProcessor
        Singleton wrapper; accesses `._ocr` for the internal text_detector
        and uses `process_image()` for recognition.
    variant : "stitch" | "per_letter"
    char_height_px : int
        Expected height of a single character in the original crop. 32 is
        tuned for the trailer dataset (10-11 char IDs in ~320px crops).
    """
    paddle = ocr_processor._ocr
    img_rgb = image.convert("RGB")
    img_w, img_h = img_rgb.size
    img_array = np.array(img_rgb)

    # 1. Find text column bounds via text_detector (avoids PaddleOCR 2.7.x bug)
    bounds = _get_text_column_bounds(paddle, img_array)
    if bounds is None:
        # No text detected at all — give up, let caller fall through.
        return None, 0.0, ["stacked_vertical:no_text_detected"]

    tx1, ty1, tx2, ty2 = bounds
    # Clamp to image and add a small margin on the x-axis for letters near edges
    tx1 = max(0, tx1 - side_padding)
    tx2 = min(img_w, tx2 + side_padding)
    ty1 = max(0, ty1)
    ty2 = min(img_h, ty2)

    text_h = ty2 - ty1
    text_w = tx2 - tx1
    if text_h <= 0 or text_w <= 0:
        return None, 0.0, ["stacked_vertical:empty_text_region"]

    # 2. Estimate number of characters from text span / expected char height.
    n_chars = max(1, int(round(text_h / char_height_px)))
    if n_chars < min_chars:
        return None, 0.0, [f"stacked_vertical:only_{n_chars}_estimated_chars(<{min_chars})"]

    # 3. Uniform slicing — avoid projection profile (gaps are irregular)
    letter_imgs: list[Image.Image] = []
    for k in range(n_chars):
        y1 = ty1 + int(round(k * text_h / n_chars))
        y2 = ty1 + int(round((k + 1) * text_h / n_chars))
        if y2 <= y1:
            continue
        letter_imgs.append(img_rgb.crop((tx1, y1, tx2, y2)))

    if len(letter_imgs) < min_chars:
        return None, 0.0, [f"stacked_vertical:slice_failure({len(letter_imgs)})"]

    ops = [
        f"stacked_vertical_{variant}:col_{tx1},{ty1}-{tx2},{ty2}",
        f"{len(letter_imgs)}_slices@{char_height_px}px",
    ]

    if variant == "stitch":
        strip = _stitch_letters(letter_imgs, target_height=target_height, gap_px=gap_px)
        text, conf = ocr_processor.process_image(strip)
        ops.append(f"stitch_strip:{strip.size[0]}x{strip.size[1]}")
        return text, conf, ops

    if variant == "per_letter":
        chars: list[str] = []
        confs: list[float] = []
        for li in letter_imgs:
            t, c = ocr_processor.process_image(li)
            if t:
                stripped = t.strip().replace(" ", "")
                if stripped:
                    chars.append(stripped[0])
                    confs.append(float(c) if c else 0.0)
        if not chars:
            return None, 0.0, ops + ["per_letter:no_recognitions"]
        text = "".join(chars)
        conf = sum(confs) / len(confs) if confs else 0.0
        return text, conf, ops

    return None, 0.0, [f"stacked_vertical:unknown_variant_{variant}"]


# ── Compose: apply_preprocessing ─────────────────────────────────────────────

AVAILABLE_FLAGS = {"rotate", "rotate_scale", "clahe", "pad", "sharpen", "dilate", "postprocess", "postprocess_v2"}

def apply_preprocessing(
    image: Image.Image,
    flags: list[str],
) -> tuple[Image.Image, list[str]]:
    """
    Apply a list of named preprocessing transforms in order.
    Returns (transformed_image, list_of_applied_transforms).

    Supported flags (applied in this order if present):
      "rotate"        — EXP-01: rotate portrait crops 90° CW (no upscale)
      "rotate_scale"  — EXP-01B: rotate portrait crops 90° CW + upscale to min 80px height
      "clahe"         — EXP-02: CLAHE contrast enhancement
      "pad"           — EXP-03: pad small crops to min 64px per side
      "sharpen"       — EXP-08: PIL UnsharpMask edge sharpening
      "dilate"        — EXP-08: morphological dilation (thicken strokes)

    Note: "two_pass" (EXP-07), "postprocess" (EXP-06), and "postprocess_v2"
    (EXP-08) are handled separately in the benchmark because they require
    special logic around the OCR call itself.

    Unknown flags are silently ignored (logged by the benchmark).
    """
    applied: list[str] = []
    img = image.convert("RGB")

    if "rotate_scale" in flags:
        img, ops = rotate_and_scale_portrait(img)
        applied.extend(ops)
    elif "rotate" in flags:
        img, was_rotated = rotate_portrait(img)
        if was_rotated:
            applied.append("rotated_90cw")

    if "clahe" in flags:
        img = clahe(img)
        applied.append("clahe")

    if "pad" in flags:
        before_size = img.size
        img = pad_small(img)
        if img.size != before_size:
            applied.append(f"padded_{before_size[0]}x{before_size[1]}_to_{img.size[0]}x{img.size[1]}")

    if "sharpen" in flags:
        img = sharpen(img)
        applied.append("sharpen")

    if "dilate" in flags:
        img = dilate(img)
        applied.append("dilate")

    return img, applied
