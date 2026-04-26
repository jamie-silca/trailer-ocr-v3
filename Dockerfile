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
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
RUN pip install --no-cache-dir -r requirements.txt

# Apply the OpenMP hack identified in the post-mortem
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1

# Create separate directory for downloading paddle models
RUN mkdir -p /root/.paddleocr

COPY app /app/app

# Create a non-root user? 
# Paddle often tries to download models to home dir. 
# Let's run as root for simplicity in download or configure home correctly.
# Ideally use non-root, so let's set HOME.
ENV HOME=/app

RUN mkdir -p /app/.paddleocr && chmod -R 777 /app/.paddleocr

RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Pre-download models to bake them into the image
# COPY --chown=appuser:appuser download_models.py .
# RUN python download_models.py

EXPOSE 8080

# Required runtime env vars (set in Cloud Run service config):
#   DETECTOR_URL          - URL of the trailer-detector service.
#   OPENROUTER_API_KEY    - enables EXP-25 Qwen3-VL portrait fallback.
#                           Without it the service runs paddle-only.

# Explicitly bind to port 8080, ignoring potential env var conflicts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
