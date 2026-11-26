FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY audio_service/requirements.txt /app/audio_service/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/audio_service/requirements.txt

COPY audio_service/ /app/audio_service/

EXPOSE 8001 8002

CMD ["python", "-m", "audio_service"]
