"""Bounded video metadata and frame extraction using the FFmpeg CLI."""
from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from . import config

class VideoExtractionError(ValueError): pass
class UnsupportedVideoType(VideoExtractionError): pass

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@dataclass
class VideoFrames:
    frames: list
    duration_seconds: float
    sampled_frames: int
    sampling_interval_seconds: float
    extractor: str = "ffmpeg"
    has_audio: bool = False


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)
    except FileNotFoundError as exc:
        raise VideoExtractionError("FFmpeg/FFprobe is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise VideoExtractionError((exc.stderr or "Video decoding failed")[-1000:]) from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoExtractionError("Video processing timed out") from exc


def extract_frames(data: bytes, filename: str) -> VideoFrames:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedVideoType(f"Unsupported video extension '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    if not data:
        raise VideoExtractionError("Empty video file")
    with tempfile.TemporaryDirectory(prefix="video-classify-") as tmp:
        source = Path(tmp) / ("upload" + ext)
        source.write_bytes(data)
        probe = _run([config.ffprobe_binary(), "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(source)])
        try:
            info = json.loads(probe.stdout)
            duration = float(info.get("format", {}).get("duration", 0))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise VideoExtractionError("Could not read video metadata") from exc
        if duration <= 0:
            raise VideoExtractionError("Video has no valid duration")
        if duration > config.max_duration_seconds():
            raise VideoExtractionError(f"Video exceeds the {config.max_duration_seconds():g}-second duration limit")
        has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        interval = max(config.sample_interval_seconds(), duration / max(config.max_frames(), 1))
        # Sample at interval midpoints to avoid repeatedly selecting the opening frame.
        timestamps = []
        t = min(interval / 2.0, duration / 2.0)
        while t < duration and len(timestamps) < config.max_frames():
            timestamps.append(t)
            t += interval
        if not timestamps:
            timestamps = [duration / 2.0]
        frames = []
        for idx, timestamp in enumerate(timestamps):
            out = Path(tmp) / f"frame-{idx:03d}.jpg"
            _run([config.ffmpeg_binary(), "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(source), "-frames:v", "1", "-vf", f"scale={config.frame_size()}:{config.frame_size()}:force_original_aspect_ratio=decrease,pad={config.frame_size()}:{config.frame_size()}:(ow-iw)/2:(oh-ih)/2", "-q:v", "3", str(out)])
            if out.exists() and out.stat().st_size:
                frames.append(out.read_bytes())
        if not frames:
            raise VideoExtractionError("No frames could be extracted from this video")
        return VideoFrames(frames, duration, len(frames), interval, has_audio=has_audio)
