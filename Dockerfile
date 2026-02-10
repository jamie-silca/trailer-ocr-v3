FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies for PaddleOCR and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

# Create a non-root user and set up model directory
ENV HOME=/app

RUN mkdir -p /app/.paddleocr && chmod -R 777 /app/.paddleocr

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# Explicitly bind to port 8080, ignoring potential env var conflicts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
