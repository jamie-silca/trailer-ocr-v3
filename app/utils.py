import io
import re
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from datetime import datetime, timezone
from typing import Optional

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

def crop_image(image: Image.Image, bbox_norm: dict) -> Image.Image:
    """
    Crop image using normalized bounding box coordinates.
    bbox_norm should have x_min, y_min, x_max, y_max in 0-1 range.
    """
    width, height = image.size
    
    x1 = int(bbox_norm['x_min'] * width)
    y1 = int(bbox_norm['y_min'] * height)
    x2 = int(bbox_norm['x_max'] * width)
    y2 = int(bbox_norm['y_max'] * height)
    
    # Ensure usage of valid coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None # Invalid crop
        
    return image.crop((x1, y1, x2, y2))

def get_current_timestamp() -> datetime:
    return datetime.now(timezone.utc)

def pad_small(
    image: Image.Image,
    min_dim: int = 64,
    fill: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """
    If either dimension is below min_dim, pad symmetrically with a neutral
    gray border to reach min_dim. (From EXP-03)
    """
    w, h = image.size
    if w >= min_dim and h >= min_dim:
        return image

    pad_w = max(0, (min_dim - w) // 2)
    pad_h = max(0, (min_dim - h) // 2)
    return ImageOps.expand(image.convert("RGB"), border=(pad_w, pad_h, pad_w, pad_h), fill=fill)

# Character substitution rules from EXP-06
_CONTAINER_CODE_RE = re.compile(r'^([A-Z0-9]{4})\s?(\d{5,7})$', re.IGNORECASE)
_PURE_DIGITS_RE = re.compile(r'^[\dOIlSBG ]{5,8}$', re.IGNORECASE)
_ALPHA_TO_DIGIT = str.maketrans("OIlSBG", "011582")

def sharpen(image: Image.Image) -> Image.Image:
    """
    Apply PIL UnsharpMask to enhance edge contrast on faded/blurry text.
    (From EXP-08)
    """
    return image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

def dilate(image: Image.Image) -> Image.Image:
    """
    Apply a single iteration of 2x2 morphological dilation to thicken text
    strokes. (From EXP-08)
    """
    if _CV2_AVAILABLE:
        arr = np.array(image.convert("RGB"))
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(arr, kernel, iterations=1)
        return Image.fromarray(dilated)
    else:
        # Fallback to PIL MaxFilter if cv2 is not available
        return image.filter(ImageFilter.MaxFilter(3))

def postprocess_text(text: Optional[str]) -> Optional[str]:
    """
    Apply conservative domain-specific character substitution to OCR output.
    (From EXP-06)
    """
    if not text:
        return text

    stripped = text.strip()

    # Rule 1: Container code format e.g. "JBHU 235644"
    m = _CONTAINER_CODE_RE.match(stripped)
    if m:
        prefix = m.group(1).upper()
        suffix = m.group(2).translate(_ALPHA_TO_DIGIT)
        space = " " if " " in stripped else ""
        return f"{prefix}{space}{suffix}"

    # Rule 2: Looks like a pure numeric code
    if _PURE_DIGITS_RE.match(stripped):
        candidate = stripped.replace(" ", "").translate(_ALPHA_TO_DIGIT)
        if candidate.isdigit():
            return candidate

    return stripped


# ── EXP-15: format-aware candidate rescoring ─────────────────────────────────

_FORMAT_PATTERNS = [
    (re.compile(r'^JBHZ\d{6}$'), 1.0),
    (re.compile(r'^JBHU\d{6}$'), 1.0),
    (re.compile(r'^R\d{5}$'), 1.0),
]

_CONFUSIONS = {
    "0": ["O", "D", "Q"],
    "O": ["0"], "D": ["0"], "Q": ["0"],
    "1": ["I", "L"],
    "I": ["1"], "L": ["1"],
    "2": ["Z"], "Z": ["2"],
    "5": ["S"], "S": ["5"],
    "6": ["G"], "G": ["6"],
    "8": ["B"], "B": ["8"],
    "H": ["8"],
}


def _format_score(s: str) -> float:
    for pat, score in _FORMAT_PATTERNS:
        if pat.match(s):
            return score
    return 0.0


def _enumerate_candidates(s: str, max_edits: int = 2, cap: int = 64) -> set[str]:
    seen = {s}
    frontier = [s]
    for _ in range(max_edits):
        new_frontier = []
        for cand in frontier:
            for i, ch in enumerate(cand):
                for sub in _CONFUSIONS.get(ch, []):
                    nc = cand[:i] + sub + cand[i + 1:]
                    if nc not in seen:
                        seen.add(nc)
                        new_frontier.append(nc)
                        if len(seen) >= cap:
                            return seen
        frontier = new_frontier
    return seen


def format_rescore(text: Optional[str]) -> tuple[Optional[str], str]:
    """
    EXP-15: Format-aware candidate rescoring.

    Given post-processed OCR text, enumerate substitution variants under known
    character confusions and pick the candidate that best matches known
    trailer-ID formats (JBHZ+6d, JBHU+6d, R+5d). Conservative: only upgrades
    when a candidate matches a stronger format AND is within 2 char swaps.

    Returns (final_text, note) where note is one of:
        "empty", "too_short", "already_matches", "rescored", "unchanged".
    """
    if not text:
        return text, "empty"
    s = text.strip().upper().replace(" ", "")
    if len(s) < 4:
        return text, "too_short"
    raw_score = _format_score(s)
    if raw_score >= 1.0:
        return text, "already_matches"

    best_cand: Optional[str] = None
    best_score = raw_score
    best_dist = 99
    for c in _enumerate_candidates(s, max_edits=2, cap=64):
        if c == s:
            continue
        sc = _format_score(c)
        if sc < raw_score + 0.5:
            continue
        # Hamming distance (same length by construction)
        d = sum(1 for a, b in zip(c, s) if a != b)
        if d > 2:
            continue
        if sc > best_score or (sc == best_score and d < best_dist):
            best_cand, best_score, best_dist = c, sc, d

    if best_cand is not None:
        return best_cand, "rescored"
    return text, "unchanged"
