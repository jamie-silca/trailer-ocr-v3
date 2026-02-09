from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from typing import List, Optional
import os
import json
import logging
import time
import httpx
from contextlib import asynccontextmanager

from app.models import DetectionResponse, Detection, BatchDetectionResponse, UrlDetectionRequest, UrlBatchDetectionRequest
from app.ocr_processor import OcrProcessor
from app.utils import bytes_to_image, crop_image, get_current_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DETECTOR_URL = os.getenv("DETECTOR_URL", "http://trailer-detector:8000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize OCR
    OcrProcessor()
    yield
    # Shutdown

app = FastAPI(
    title="Trailer OCR Service",
    description="Microservice for OCR on trailer IDs, integrated with YOLO detection",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trailer-ocr", "paddle_status": "initialized"}

@app.post("/detect-and-ocr", response_model=DetectionResponse)
async def detect_and_ocr(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0)
):
    """
    Full pipeline: 
    1. Send image to Detector Service
    2. Get Bounding Boxes
    3. Crop and Run OCR
    4. Return Enriched Results
    """
    try:
        # 1. Read Image
        logger.info(f"Processing file: {file.filename}")
        file_content = await file.read()
        image = bytes_to_image(file_content)
        
        # 2. Call Detector Service
        # We need to send the FILE to the detector
        async with httpx.AsyncClient() as client:
            # Prepare multipart upload
            files = {'file': (file.filename, file_content, file.content_type)}
            params = {'confidence_threshold': confidence_threshold}
            
            logger.info(f"Calling detector at {DETECTOR_URL}/detect")
            resp = await client.post(
                f"{DETECTOR_URL}/detect", 
                files=files, 
                params=params,
                timeout=120.0
            )
            
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Detector failed: {resp.text}")
                
            detection_data = resp.json()
            
        # Parse into model
        detection_response = DetectionResponse(**detection_data)
        
        if not detection_response.success or not detection_response.detections:
            logger.info("No detections found or detector reported failure.")
            return detection_response

        # 3. Pipeline: Crop & OCR
        ocr_processor = OcrProcessor()
        
        for detection in detection_response.detections:
            # Crop
            bbox = detection.bbox
            # bbox object to dict for helper
            bbox_dict = {
                'x_min': bbox.x_min,
                'y_min': bbox.y_min,
                'x_max': bbox.x_max,
                'y_max': bbox.y_max
            }
            
            cropped_img = crop_image(image, bbox_dict)
            
            if cropped_img:
                # Run OCR
                text, conf = ocr_processor.process_image(cropped_img)
                if text:
                    detection.text = text
                    detection.ocr_confidence = conf
                    logger.info(f"OCR Result: {text} ({conf:.2f})")
                else:
                    logger.info("OCR found no text in crop.")
            else:
                logger.warning("Invalid crop dimensions, skipping OCR.")

        # Update processing time to include OCR time? 
        # The current processing_time_ms is just detection time.
        # Maybe we should add a total_time field? 
        # For now, let's just return the enriched response. 
        # The user asked to "update and return json".
        
        return detection_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect-and-ocr pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-and-ocr/batch", response_model=BatchDetectionResponse)
async def detect_and_ocr_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0)
):
    """
    Batch Detect + OCR Pipeline
    """
    start_time_total = time.perf_counter()
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # 1. Read all images into memory & prepare for detector
        images_map = {} # filename -> image_object
        upload_files_payload = []

        for file in files:
            content = await file.read()
            images_map[file.filename] = bytes_to_image(content)
            # Re-pack for httpx
            # Note: detector expects 'files' list
            upload_files_payload.append(('files', (file.filename, content, file.content_type)))

        # 2. Call Detector Batch
        async with httpx.AsyncClient(timeout=300.0) as client:
            params = {'confidence_threshold': confidence_threshold}
            logger.info(f"Calling detector batch at {DETECTOR_URL}/detect/batch")
            
            resp = await client.post(
                f"{DETECTOR_URL}/detect/batch", 
                files=upload_files_payload, 
                params=params,
                timeout=300.0
            )
            
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Detector failed: {resp.text}")
                
            batch_data = resp.json()

        # 3. Parse Response
        batch_response = BatchDetectionResponse(**batch_data)

        # 4. Run OCR on results
        ocr_processor = OcrProcessor()
        
        for result in batch_response.results:
            filename = result.image_metadata.filename
            if filename not in images_map:
                continue
                
            original_image = images_map[filename]
            
            if result.detections:
                for detection in result.detections:
                    # Crop
                    bbox = detection.bbox
                    bbox_dict = {
                        'x_min': bbox.x_min,
                        'y_min': bbox.y_min,
                        'x_max': bbox.x_max,
                        'y_max': bbox.y_max
                    }
                    
                    cropped_img = crop_image(original_image, bbox_dict)
                    
                    if cropped_img:
                        text, conf = ocr_processor.process_image(cropped_img)
                        if text:
                            detection.text = text
                            detection.ocr_confidence = conf
                            logger.info(f"[{filename}] OCR: {text} ({conf:.2f})")

        # Update total time to include OCR time
        batch_response.total_processing_time_ms = (time.perf_counter() - start_time_total) * 1000
        
        return batch_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch detect-and-ocr: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-and-ocr/url", response_model=DetectionResponse)
async def detect_and_ocr_url(request: UrlDetectionRequest):
    try:
        logger.info(f"Processing URL: {request.image_url}")
        
        # 1. Download Image
        async with httpx.AsyncClient() as client:
            resp = await client.get(request.image_url, timeout=30.0)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image: {resp.status_code}")
            file_content = resp.content
            
        filename = request.id or request.image_url.split('/')[-1]
        image = bytes_to_image(file_content)
        
        # 2. Call Detector Service
        async with httpx.AsyncClient() as client:
            files = {'file': (filename, file_content, "image/jpeg")}
            params = {'confidence_threshold': request.confidence_threshold}
            
            logger.info(f"Calling detector at {DETECTOR_URL}/detect")
            resp = await client.post(
                f"{DETECTOR_URL}/detect", 
                files=files, 
                params=params,
                timeout=120.0
            )
            
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Detector failed: {resp.text}")
                
            detection_data = resp.json()
            
        # Parse into model
        detection_response = DetectionResponse(**detection_data)
        if request.id:
            detection_response.id = request.id
        
        if not detection_response.success or not detection_response.detections:
            return detection_response

        # 3. Pipeline: Crop & OCR
        ocr_processor = OcrProcessor()
        
        for detection in detection_response.detections:
            bbox = detection.bbox
            bbox_dict = {
                'x_min': bbox.x_min,
                'y_min': bbox.y_min,
                'x_max': bbox.x_max,
                'y_max': bbox.y_max
            }
            
            cropped_img = crop_image(image, bbox_dict)
            
            if cropped_img:
                text, conf = ocr_processor.process_image(cropped_img)
                if text:
                    detection.text = text
                    detection.ocr_confidence = conf
                    logger.info(f"OCR Result: {text} ({conf:.2f})")
        
        return detection_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect-and-ocr url: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-and-ocr/url/batch", response_model=BatchDetectionResponse)
async def detect_and_ocr_url_batch(request: UrlBatchDetectionRequest):
    start_time_total = time.perf_counter()
    
    if not request.images:
        raise HTTPException(status_code=400, detail="No images provided")

    try:
        # 1. Download all images
        images_map = {} 
        upload_files_payload = []
        
        async with httpx.AsyncClient() as client:
            for img_req in request.images:
                try:
                    resp = await client.get(img_req.image_url, timeout=30.0)
                    if resp.status_code == 200:
                        content = resp.content
                        filename = img_req.id or img_req.image_url.split('/')[-1]
                        
                        images_map[filename] = bytes_to_image(content)
                        upload_files_payload.append(('files', (filename, content, "image/jpeg")))
                except Exception as e:
                    logger.error(f"Failed to download {img_req.image_url}: {e}")

        if not upload_files_payload:
             raise HTTPException(status_code=400, detail="No valid images could be downloaded")

        # 2. Call Detector Batch
        async with httpx.AsyncClient(timeout=300.0) as client:
            params = {'confidence_threshold': request.confidence_threshold}
            
            resp = await client.post(
                f"{DETECTOR_URL}/detect/batch", 
                files=upload_files_payload, 
                params=params,
                timeout=300.0
            )
            
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Detector failed: {resp.text}")
                
            batch_data = resp.json()

        # 3. Parse and OCR
        batch_response = BatchDetectionResponse(**batch_data)
        ocr_processor = OcrProcessor()
        
        for result in batch_response.results:
            filename = result.image_metadata.filename
            if filename in images_map:
                original_image = images_map[filename]
                
                if result.detections:
                    for detection in result.detections:
                        bbox = detection.bbox
                        bbox_dict = {
                            'x_min': bbox.x_min,
                            'y_min': bbox.y_min,
                            'x_max': bbox.x_max,
                            'y_max': bbox.y_max
                        }
                        
                        cropped_img = crop_image(original_image, bbox_dict)
                        if cropped_img:
                            text, conf = ocr_processor.process_image(cropped_img)
                            if text:
                                detection.text = text
                                detection.ocr_confidence = conf

        batch_response.total_processing_time_ms = (time.perf_counter() - start_time_total) * 1000
        return batch_response

    except Exception as e:
        logger.error(f"Error in batch detect-and-ocr url: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr", response_model=Detection)
async def ocr_direct(file: UploadFile = File(...)):
    """
    Direct OCR on an uploaded image. Returns text and confidence.
    """
    try:
        content = await file.read()
        image = bytes_to_image(content)
        
        ocr_processor = OcrProcessor()
        text, conf = ocr_processor.process_image(image)
        
        
        from app.models import BoundingBox
        
        return Detection(
            class_id=-1,
            class_name="ocr_only",
            confidence=1.0,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1, width=1, height=1),
            text=text,
            ocr_confidence=conf
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
