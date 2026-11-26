"""Video classification branch for the unified multimodal platform."""
from .model import classify_frames, model_info, reload_model
from .service import classify_video

__all__ = ["classify_video", "classify_frames", "model_info", "reload_model"]
