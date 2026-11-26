"""Framework-independent video upload validation and classification service."""
from __future__ import annotations
from . import config
from .extract import VideoExtractionError, UnsupportedVideoType, extract_frames
from .model import classify_frames

class VideoServiceError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def classify_video(file_bytes: bytes, filename: str) -> dict:
    if not file_bytes:
        raise VideoServiceError(400, "Empty video file")
    if len(file_bytes) > config.max_upload_bytes():
        raise VideoServiceError(413, f"Video exceeds {config.max_upload_bytes() // (1024 * 1024)} MB upload limit")
    try:
        extracted = extract_frames(file_bytes, filename)
    except UnsupportedVideoType as exc:
        raise VideoServiceError(415, str(exc)) from exc
    except VideoExtractionError as exc:
        raise VideoServiceError(422, str(exc)) from exc
    try:
        prediction = classify_frames(extracted.frames)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise VideoServiceError(503, f"Video model unavailable: {exc}") from exc
    return {
        "filename": filename or "",
        **prediction,
        "extractor": extracted.extractor,
        "file_type": "video/" + (filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"),
        "duration_seconds": round(extracted.duration_seconds, 3),
        "frame_count": extracted.sampled_frames,
        "sampling_interval_seconds": round(extracted.sampling_interval_seconds, 3),
        "audio_detected": extracted.has_audio,
        "audio_assisted": False,
        "real_data_validated": False,
    }
