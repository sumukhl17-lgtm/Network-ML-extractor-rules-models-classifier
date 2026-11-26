from __future__ import annotations

import math

import av
import numpy as np

from .config import Settings
from .decoder import DecodedAudio
from .service import AudioServiceError

TARGET_SAMPLE_RATE = 16_000
MIN_MODEL_INPUT_SAMPLES = 15_600


def _invalid_audio() -> AudioServiceError:
    return AudioServiceError(
        400,
        "invalid_audio",
        "The decoded audio cannot be preprocessed.",
    )


def _decoded_sample_limit() -> AudioServiceError:
    return AudioServiceError(
        413,
        "decoded_sample_limit_exceeded",
        "The decoded audio exceeds the sample limit.",
    )


def preprocess_audio(
    decoded: DecodedAudio,
    settings: Settings | None = None,
) -> np.ndarray:
    """Convert decoded frames to a bounded mono, 16 kHz, float32 waveform."""
    limits = settings or Settings()
    if not isinstance(decoded, DecodedAudio):
        raise _invalid_audio()
    if (
        not isinstance(decoded.duration_seconds, (int, float))
        or isinstance(decoded.duration_seconds, bool)
        or not math.isfinite(decoded.duration_seconds)
        or decoded.duration_seconds <= 0
    ):
        raise _invalid_audio()
    if decoded.duration_seconds > limits.max_audio_duration_seconds:
        raise AudioServiceError(
            413,
            "duration_limit_exceeded",
            "The audio duration exceeds the limit.",
        )
    if (
        not isinstance(decoded.sample_rate, int)
        or isinstance(decoded.sample_rate, bool)
        or decoded.sample_rate <= 0
    ):
        raise _invalid_audio()
    if decoded.sample_rate > limits.max_audio_sample_rate:
        raise AudioServiceError(
            415,
            "sample_rate_limit_exceeded",
            "The audio sample rate exceeds the limit.",
        )
    if (
        not isinstance(decoded.channels, int)
        or isinstance(decoded.channels, bool)
        or decoded.channels <= 0
    ):
        raise _invalid_audio()
    if decoded.channels > limits.max_audio_channels:
        raise AudioServiceError(
            415,
            "channel_limit_exceeded",
            "The audio channel count exceeds the limit.",
        )
    if not decoded.frames:
        raise _invalid_audio()

    resampler = av.AudioResampler(
        format="fltp",
        layout="mono",
        rate=TARGET_SAMPLE_RATE,
    )
    chunks: list[np.ndarray] = []
    decoded_samples = 0
    output_samples = 0
    max_output_samples = (
        math.ceil(limits.max_audio_duration_seconds * TARGET_SAMPLE_RATE) + 1
    )

    def collect(frame: av.AudioFrame) -> None:
        nonlocal output_samples

        try:
            samples = np.asarray(frame.to_ndarray())
            sample_rate = frame.sample_rate
            channels = frame.layout.nb_channels
        except (AttributeError, TypeError, ValueError, av.error.FFmpegError):
            raise _invalid_audio() from None
        if (
            sample_rate != TARGET_SAMPLE_RATE
            or channels != 1
            or samples.ndim != 2
            or samples.shape[0] != 1
            or samples.dtype.kind != "f"
        ):
            raise _invalid_audio()
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        if not np.isfinite(samples).all():
            raise _invalid_audio()

        chunk = samples.reshape(-1)
        if chunk.size == 0:
            return
        output_samples += int(chunk.size)
        if output_samples > max_output_samples:
            raise _decoded_sample_limit()
        chunks.append(chunk)

    try:
        for frame in decoded.frames:
            try:
                frame_samples = frame.samples
                frame_rate = frame.sample_rate
                frame_channels = frame.layout.nb_channels
            except (AttributeError, TypeError, ValueError, av.error.FFmpegError):
                raise _invalid_audio() from None
            if (
                not isinstance(frame_samples, int)
                or frame_samples <= 0
                or frame_rate != decoded.sample_rate
                or frame_channels != decoded.channels
            ):
                raise _invalid_audio()
            decoded_samples += frame_samples * decoded.channels
            if decoded_samples > limits.max_decoded_samples:
                raise _decoded_sample_limit()
            for output_frame in resampler.resample(frame):
                collect(output_frame)
        for output_frame in resampler.resample(None):
            collect(output_frame)
    except AudioServiceError:
        raise
    except (av.error.FFmpegError, AttributeError, TypeError, ValueError, OverflowError):
        raise _invalid_audio() from None

    if not chunks:
        raise _invalid_audio()

    waveform = np.concatenate(chunks)
    if not np.isfinite(waveform).all():
        raise _invalid_audio()
    np.clip(waveform, -1.0, 1.0, out=waveform)
    if waveform.size < MIN_MODEL_INPUT_SAMPLES:
        raise AudioServiceError(
            422,
            "audio_too_short",
            "Audio must be at least 0.975 seconds long for reliable classification.",
        )
    return waveform
