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
from PIL import Image, ImageOps

try:
    import cv2
    import numpy as np
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


# ── Compose: apply_preprocessing ─────────────────────────────────────────────

AVAILABLE_FLAGS = {"rotate", "rotate_scale", "clahe", "pad"}

def apply_preprocessing(
    image: Image.Image,
    flags: list[str],
) -> tuple[Image.Image, list[str]]:
    """
    Apply a list of named preprocessing transforms in order.
    Returns (transformed_image, list_of_applied_transforms).

    Supported flags (applied in this order if present):
      "rotate"       — EXP-01: rotate portrait crops 90° CW (no upscale)
      "rotate_scale" — EXP-01B: rotate portrait crops 90° CW + upscale to min 80px height
      "clahe"        — EXP-02: CLAHE contrast enhancement
      "pad"          — EXP-03: pad small crops to min 64px per side

    Note: "two_pass" (EXP-07) and "postprocess" (EXP-06) are handled
    separately in the benchmark because they require special logic around
    the OCR call itself.

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

    return img, applied
