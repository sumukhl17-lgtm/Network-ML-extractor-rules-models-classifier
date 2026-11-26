"""Video branch settings; environment variables are read at call time."""
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


def max_upload_bytes() -> int: return int(os.getenv("VIDEO_MAX_BYTES", str(200 * 1024 * 1024)))
def max_duration_seconds() -> float: return float(os.getenv("VIDEO_MAX_DURATION_SECONDS", "600"))
def max_frames() -> int: return int(os.getenv("VIDEO_MAX_FRAMES", "12"))
def sample_interval_seconds() -> float: return float(os.getenv("VIDEO_SAMPLE_INTERVAL_SECONDS", "2"))
def frame_size() -> int: return int(os.getenv("VIDEO_FRAME_SIZE", "224"))
def unknown_threshold() -> float: return float(os.getenv("VIDEO_UNKNOWN_THRESHOLD", "0.22"))
def model_id() -> str: return os.getenv("VIDEO_CLIP_MODEL", "openai/clip-vit-base-patch32")
def cache_dir() -> Path: return Path(os.getenv("VIDEO_MODEL_CACHE", str(PKG_DIR / "model_cache")))
def model_source() -> str: return os.getenv("VIDEO_MODEL_SOURCE", "auto").lower()
def registered_model_name() -> str: return os.getenv("VIDEO_MODEL_NAME", "video-classifier")
def model_alias() -> str: return os.getenv("MODEL_ALIAS", "champion")
def ffmpeg_binary() -> str: return os.getenv("FFMPEG_BINARY", "ffmpeg")
def ffprobe_binary() -> str: return os.getenv("FFPROBE_BINARY", "ffprobe")

CATEGORIES = [
    "People and conversations",
    "Sports and physical activity",
    "Nature and outdoor scenes",
    "Animals",
    "Animation",
    "Screen recordings and presentations",
    "Vehicles and transport",
]
PROMPTS = [
    "Live-action footage of people speaking, talking to the camera, or interacting with each other.",
    "Live-action footage of sports, exercise, athletes, games, or physical activity.",
    "Live-action outdoor scenery showing landscapes, skies, clouds, forests, mountains, rivers, or beaches.",
    "Live-action footage of real animals, pets, birds, or wildlife.",
    "Cartoon scenes, animated characters, anime, or computer-generated animation.",
    "A computer or mobile screen recording showing software, a user interface, slides, a presentation, or a demonstration.",
    "Live-action footage of cars, buses, trains, aircraft, roads, or other transportation.",
]
