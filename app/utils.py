import io
from PIL import Image
from datetime import datetime, timezone

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
