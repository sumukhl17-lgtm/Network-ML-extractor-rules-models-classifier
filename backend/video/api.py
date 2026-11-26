"""FastAPI routes for video classification."""
from fastapi import APIRouter, File, HTTPException, UploadFile
from . import config, model_info, reload_model
from .service import VideoServiceError, classify_video

router = APIRouter(prefix="/classify/video", tags=["video"])

@router.post("")
def classify(file: UploadFile = File(...)):
    data = file.file.read(config.max_upload_bytes() + 1)
    try:
        return classify_video(data, file.filename or "")
    except VideoServiceError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc

@router.get("/health")
def health():
    return {"status": "ok", "model": model_info(), "decoder": "ffmpeg"}

@router.get("/model")
def current_model():
    return model_info()

@router.post("/model/reload")
def reload():
    return reload_model()
