# Trailer OCR Service v3

A microservice that orchestrates the detection and OCR pipeline for Trailer IDs.

It works by:
1.  Receiving an image.
2.  sending it to the **Trailer Detector Service** (running on port 8000).
3.  Receiving bounding boxes.
4.  Cropping the image locally.
5.  Running PaddleOCR on the crops.
6.  Returning the combined results.

## Prerequisites

- **Trailer Detector Service** must be running on port **8000**.
- This service expects to be able to reach the detector at `http://host.docker.internal:8000` (default for Docker Desktop).

## Running the Service

Build and start the service:

```bash
docker-compose up --build -d
```

The service will start on port **8001**.

## API Documentation

- **Swagger UI**: [http://localhost:8001/docs](http://localhost:8001/docs)

## Key Endpoints

### 1. Detect and OCR (Pipeline)

Run the full pipeline on a single image.

```bash
POST /detect-and-ocr
```

**Example**:
```bash
curl -X POST "http://localhost:8001/detect-and-ocr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

### 2. QC Only (Direct)

Run OCR directly on an image (assumes the image *is* the text crop).

```bash
POST /ocr
```
