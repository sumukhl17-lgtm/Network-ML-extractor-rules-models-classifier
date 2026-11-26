FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIDEO_MODEL_SOURCE=pretrained \
    VIDEO_MODEL_CACHE=/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch, keeping CUDA packages out of this service.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

# Install the Video branch dependencies after CPU-only PyTorch.
COPY backend/video-requirements.txt /app/backend/video-requirements.txt
RUN python -m pip install -r /app/backend/video-requirements.txt \
    && python -m pip install \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "python-multipart>=0.0.9"

COPY backend/video/ /app/backend/video/
COPY video_service/main.py /app/main.py

RUN mkdir -p /models

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8003/classify/video/health', timeout=4).close()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
