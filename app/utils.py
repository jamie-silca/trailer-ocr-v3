import io
import re
from PIL import Image, ImageOps
from datetime import datetime, timezone
from typing import Optional

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
