from __future__ import annotations

from fractions import Fraction

import av
import numpy as np
import pytest

from audio_service.config import Settings
from audio_service.decoder import DecodedAudio
from audio_service.preprocessing import (
    MIN_MODEL_INPUT_SAMPLES,
    TARGET_SAMPLE_RATE,
    preprocess_audio,
)
from audio_service.service import AudioServiceError


def make_decoded_audio(
    *,
    sample_rate: int = 16_000,
    sample_format: str = "fltp",
    channels: int = 1,
    duration_seconds: float = 1.0,
    amplitude: float = 0.5,
    channel_amplitudes: tuple[float, ...] | None = None,
) -> DecodedAudio:
    samples_per_channel = round(sample_rate * duration_seconds)
    layout = "mono" if channels == 1 else "stereo"
    frame = av.AudioFrame(
        format=sample_format,
        layout=layout,
        samples=samples_per_channel,
    )
    frame.sample_rate = sample_rate
    frame.pts = 0
    frame.time_base = Fraction(1, sample_rate)

    amplitudes = channel_amplitudes or (amplitude,) * channels
    phase = np.arange(samples_per_channel, dtype=np.float64) / sample_rate
    channel_data = [
        amplitudes[channel]
        * np.sin(2 * np.pi * 440 * phase)
        for channel in range(channels)
    ]

    if sample_format in {"s16", "s16p"}:
        encoded = [
            np.rint(samples * 32767).astype("<i2").tobytes()
            for samples in channel_data
        ]
    elif sample_format in {"flt", "fltp"}:
        encoded = [
            np.asarray(samples, dtype="<f4").tobytes()
            for samples in channel_data
        ]
    else:
        raise ValueError(f"Unsupported test sample format: {sample_format}")

    if sample_format.endswith("p"):
        for plane, channel_bytes in zip(frame.planes, encoded):
            plane.update(channel_bytes + bytes(plane.buffer_size - len(channel_bytes)))
    else:
        interleaved = (
            np.stack(
                [
                    np.frombuffer(channel_bytes, dtype="<i2" if sample_format == "s16" else "<f4")
                    for channel_bytes in encoded
                ],
                axis=1,
            )
            .reshape(-1)
            .tobytes()
        )
        frame.planes[0].update(
            interleaved
            + bytes(frame.planes[0].buffer_size - len(interleaved))
        )

    return DecodedAudio(
        frames=(frame,),
        duration_seconds=samples_per_channel / sample_rate,
        sample_rate=sample_rate,
        channels=channels,
        codec_name="test",
        format_name="test",
    )


def assert_audio_error(
    decoded: DecodedAudio,
    expected_status: int,
    expected_code: str,
    settings: Settings | None = None,
) -> None:
    with pytest.raises(AudioServiceError) as raised:
        preprocess_audio(decoded, settings)
    assert raised.value.status_code == expected_status
    assert raised.value.code == expected_code


@pytest.mark.parametrize(
    ("sample_rate", "sample_format", "channels"),
    [
        (8_000, "s16", 1),
        (22_050, "fltp", 1),
        (44_100, "s16p", 2),
        (48_000, "flt", 2),
    ],
)
def test_preprocesses_sample_rates_formats_and_channel_counts(
    sample_rate: int,
    sample_format: str,
    channels: int,
) -> None:
    waveform = preprocess_audio(
        make_decoded_audio(
            sample_rate=sample_rate,
            sample_format=sample_format,
            channels=channels,
        )
    )

    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert abs(waveform.size - TARGET_SAMPLE_RATE) <= 1
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 0.1
    assert waveform.min() >= -1.0
    assert waveform.max() <= 1.0
    spectrum = np.abs(np.fft.rfft(waveform))
    frequencies = np.fft.rfftfreq(waveform.size, d=1 / TARGET_SAMPLE_RATE)
    dominant_frequency = frequencies[1 + int(np.argmax(spectrum[1:]))]
    assert dominant_frequency == pytest.approx(440, abs=2)


def test_stereo_channels_are_mixed_to_mono_without_amplification() -> None:
    waveform = preprocess_audio(
        make_decoded_audio(
            sample_format="fltp",
            channels=2,
            channel_amplitudes=(0.2, 0.6),
        )
    )

    assert waveform.ndim == 1
    assert np.sqrt(np.mean(np.square(waveform))) == pytest.approx(
        0.4,
        abs=0.02,
    )


def test_resampler_preserves_low_amplitude_signal() -> None:
    waveform = preprocess_audio(
        make_decoded_audio(
            sample_rate=44_100,
            sample_format="fltp",
            amplitude=1e-4,
        )
    )

    assert 5e-5 < float(np.max(np.abs(waveform))) < 2e-4


def test_silence_remains_silent() -> None:
    decoded = make_decoded_audio(amplitude=0.0)

    waveform = preprocess_audio(decoded)

    assert waveform.size >= MIN_MODEL_INPUT_SAMPLES
    assert np.count_nonzero(waveform) == 0


def test_clips_out_of_range_float_samples_without_peak_normalizing() -> None:
    decoded = make_decoded_audio(amplitude=1.5)

    waveform = preprocess_audio(decoded)

    assert waveform.max() == 1.0
    assert waveform.min() >= -1.0


def test_rejects_empty_decoded_audio() -> None:
    assert_audio_error(
        DecodedAudio((), 0.0, 16_000, 1, None, None),
        400,
        "invalid_audio",
    )


def test_rejects_audio_too_short_for_yamnet_patch() -> None:
    assert_audio_error(
        make_decoded_audio(duration_seconds=0.25),
        422,
        "audio_too_short",
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_rejects_non_finite_samples(bad_value: float) -> None:
    decoded = make_decoded_audio()
    frame = decoded.frames[0]
    samples = np.full(frame.samples, bad_value, dtype="<f4")
    frame.planes[0].update(samples.tobytes())

    assert_audio_error(decoded, 400, "invalid_audio")


def test_rejects_frames_with_inconsistent_sample_rate() -> None:
    decoded = make_decoded_audio()
    frame = decoded.frames[0]
    frame.sample_rate = 8_000

    assert_audio_error(decoded, 400, "invalid_audio")


def test_enforces_existing_duration_and_decoded_sample_limits() -> None:
    decoded = make_decoded_audio(duration_seconds=1.0)
    assert_audio_error(
        decoded,
        413,
        "duration_limit_exceeded",
        Settings(max_audio_duration_seconds=0.5),
    )

    assert_audio_error(
        decoded,
        413,
        "decoded_sample_limit_exceeded",
        Settings(max_decoded_samples=100),
    )
