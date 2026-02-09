from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    # New fields
    text: Optional[str] = None
    ocr_confidence: Optional[float] = None

class ImageMetadata(BaseModel):
    filename: str
    width: int
    height: int

class DetectionResponse(BaseModel):
    id: Optional[str] = None
    success: bool
    timestamp: datetime
    processing_time_ms: float
    image_metadata: ImageMetadata
    detections: List[Detection]
    detection_count: int
    error: Optional[str] = None

class OcrRequest(BaseModel):
    # If we want to request specifically just OCR on an image + bbox list
    pass

class BatchDetectionResponse(BaseModel):
    success: bool
    timestamp: datetime
    total_processing_time_ms: float
    results: List[DetectionResponse]
    total_detections: int
    error: Optional[str] = None

class UrlDetectionRequest(BaseModel):
    image_url: str
    id: Optional[str] = None
    confidence_threshold: float = 0.0

class UrlBatchRecord(BaseModel):
    image_url: str
    id: Optional[str] = None

class UrlBatchDetectionRequest(BaseModel):
    images: List[UrlBatchRecord]
    confidence_threshold: float = 0.0
