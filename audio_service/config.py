import math
import os
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    service_token: str | None = None
    max_upload_bytes: int = 20 * 1024 * 1024
    host: str = "0.0.0.0"
    port: int = 8001
    max_audio_duration_seconds: float = 120.0
    max_audio_sample_rate: int = 192_000
    max_audio_channels: int = 8
    max_decoded_samples: int = 20_000_000
    worker_port: int = 8002
    worker_url: str = "http://audio-worker:8002"
    decode_timeout_seconds: float = 30.0
    inference_timeout_seconds: float = 120.0
    max_concurrent_decode_jobs: int = 1
    child_cpu_limit_seconds: int | None = 25
    child_address_space_limit_bytes: int | None = None
    model_source: str = "bundle"
    model_bundle_dir: str | None = None

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be greater than zero")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not 1 <= self.worker_port <= 65535:
            raise ValueError("worker_port must be between 1 and 65535")
        worker_url = urlsplit(self.worker_url)
        if (
            worker_url.scheme not in {"http", "https"}
            or not worker_url.hostname
            or worker_url.username is not None
            or worker_url.password is not None
            or worker_url.query
            or worker_url.fragment
        ):
            raise ValueError("worker_url must be an HTTP(S) URL without credentials")
        if (
            not math.isfinite(self.max_audio_duration_seconds)
            or self.max_audio_duration_seconds <= 0
        ):
            raise ValueError("max_audio_duration_seconds must be finite and positive")
        if not math.isfinite(self.decode_timeout_seconds) or self.decode_timeout_seconds <= 0:
            raise ValueError("decode_timeout_seconds must be finite and positive")
        if (
            not math.isfinite(self.inference_timeout_seconds)
            or self.inference_timeout_seconds <= 0
        ):
            raise ValueError("inference_timeout_seconds must be finite and positive")
        if self.max_concurrent_decode_jobs <= 0:
            raise ValueError("max_concurrent_decode_jobs must be greater than zero")
        if (
            self.child_cpu_limit_seconds is not None
            and self.child_cpu_limit_seconds <= 0
        ):
            raise ValueError("child_cpu_limit_seconds must be positive when configured")
        if (
            self.child_address_space_limit_bytes is not None
            and self.child_address_space_limit_bytes <= 0
        ):
            raise ValueError(
                "child_address_space_limit_bytes must be positive when configured"
            )
        if self.max_audio_sample_rate <= 0:
            raise ValueError("max_audio_sample_rate must be greater than zero")
        if self.max_audio_channels <= 0:
            raise ValueError("max_audio_channels must be greater than zero")
        if self.max_decoded_samples <= 0:
            raise ValueError("max_decoded_samples must be greater than zero")
        if self.model_source not in {"bundle", "registry"}:
            raise ValueError("AUDIO_MODEL_SOURCE must be 'bundle' or 'registry'")

    @classmethod
    def from_environment(cls) -> "Settings":
        return load_settings()


def load_settings() -> Settings:
    raw_max_upload_bytes = os.getenv("AUDIO_MAX_UPLOAD_BYTES", "20971520")
    raw_port = os.getenv("AUDIO_SERVICE_PORT", "8001")
    raw_max_duration = os.getenv("AUDIO_MAX_DURATION_SECONDS", "120")
    raw_max_sample_rate = os.getenv("AUDIO_MAX_SAMPLE_RATE", "192000")
    raw_max_channels = os.getenv("AUDIO_MAX_CHANNELS", "8")
    raw_max_decoded_samples = os.getenv("AUDIO_MAX_DECODED_SAMPLES", "20000000")
    raw_worker_port = os.getenv("AUDIO_WORKER_PORT", "8002")
    raw_decode_timeout = os.getenv("AUDIO_DECODE_TIMEOUT_SECONDS", "30")
    raw_inference_timeout = os.getenv("AUDIO_INFERENCE_TIMEOUT_SECONDS", "120")
    raw_max_concurrent_jobs = os.getenv("AUDIO_MAX_CONCURRENT_DECODE_JOBS", "1")
    raw_child_cpu_limit = os.getenv("AUDIO_CHILD_CPU_LIMIT_SECONDS", "25")
    raw_child_as_limit = os.getenv("AUDIO_CHILD_ADDRESS_SPACE_LIMIT_BYTES", "")

    try:
        max_upload_bytes = int(raw_max_upload_bytes)
        port = int(raw_port)
        max_audio_duration_seconds = float(raw_max_duration)
        max_audio_sample_rate = int(raw_max_sample_rate)
        max_audio_channels = int(raw_max_channels)
        max_decoded_samples = int(raw_max_decoded_samples)
        worker_port = int(raw_worker_port)
        decode_timeout_seconds = float(raw_decode_timeout)
        inference_timeout_seconds = float(raw_inference_timeout)
        max_concurrent_decode_jobs = int(raw_max_concurrent_jobs)
        child_cpu_limit_seconds = (
            int(raw_child_cpu_limit) if raw_child_cpu_limit.strip() else None
        )
        child_address_space_limit_bytes = (
            int(raw_child_as_limit) if raw_child_as_limit.strip() else None
        )
    except ValueError as exc:
        raise ValueError("Audio service limits and port must be numeric") from exc

    return Settings(
        service_token=os.getenv("AUDIO_SERVICE_TOKEN") or None,
        max_upload_bytes=max_upload_bytes,
        host=os.getenv("AUDIO_SERVICE_HOST", "0.0.0.0"),
        port=port,
        max_audio_duration_seconds=max_audio_duration_seconds,
        max_audio_sample_rate=max_audio_sample_rate,
        max_audio_channels=max_audio_channels,
        max_decoded_samples=max_decoded_samples,
        worker_port=worker_port,
        worker_url=os.getenv("AUDIO_WORKER_URL", "http://audio-worker:8002"),
        decode_timeout_seconds=decode_timeout_seconds,
        inference_timeout_seconds=inference_timeout_seconds,
        max_concurrent_decode_jobs=max_concurrent_decode_jobs,
        child_cpu_limit_seconds=child_cpu_limit_seconds,
        child_address_space_limit_bytes=child_address_space_limit_bytes,
        model_source=os.getenv("AUDIO_MODEL_SOURCE", "bundle").strip().lower(),
        model_bundle_dir=os.getenv("AUDIO_MODEL_BUNDLE_DIR") or None,
    )
