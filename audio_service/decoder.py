from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
import math
from typing import TYPE_CHECKING, Any

from .config import Settings
from .service import AudioServiceError

if TYPE_CHECKING:
    import av


@dataclass(frozen=True)
class DecodedAudio:
    frames: tuple[av.AudioFrame, ...]
    duration_seconds: float
    sample_rate: int
    channels: int
    codec_name: str | None
    format_name: str | None


def _invalid_audio() -> AudioServiceError:
    return AudioServiceError(
        400,
        "invalid_audio",
        "The uploaded content is not a complete, decodable audio file.",
    )


def _unsupported_audio() -> AudioServiceError:
    return AudioServiceError(
        415,
        "unsupported_audio",
        "The uploaded media does not contain a supported audio stream.",
    )


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


def _stream_channels(stream: Any) -> int | None:
    try:
        layout = stream.codec_context.layout
        return _positive_int(layout.nb_channels)
    except (AttributeError, TypeError, ValueError):
        return None


def _frame_channels(frame: Any) -> int | None:
    try:
        return _positive_int(frame.layout.nb_channels)
    except (AttributeError, TypeError, ValueError):
        return None


def _stream_duration_seconds(stream: Any) -> Fraction | None:
    try:
        duration = stream.duration
        time_base = stream.time_base
        if duration is None or time_base is None:
            return None
        seconds = Fraction(duration) * Fraction(time_base)
    except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return seconds if seconds >= 0 else None


def _positive_time_base(value: Any) -> Fraction | None:
    if value is None:
        return None
    try:
        time_base = Fraction(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return time_base if time_base > 0 else None


def _timestamp_seconds(pts: Any, time_base: Any) -> Fraction | None:
    if pts is None:
        return None
    positive_time_base = _positive_time_base(time_base)
    if positive_time_base is None:
        raise _invalid_audio()
    try:
        seconds = Fraction(pts) * positive_time_base
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        raise _invalid_audio() from None
    return seconds


def _packet_duration_seconds(packet: Any, stream_time_base: Any) -> Fraction | None:
    duration = getattr(packet, "duration", None)
    if not isinstance(duration, int) or duration <= 0:
        return None
    time_base = getattr(packet, "time_base", None) or stream_time_base
    positive_time_base = _positive_time_base(time_base)
    if positive_time_base is None:
        return None
    return Fraction(duration) * positive_time_base


def _preflight_packet(
    packet: Any,
    *,
    stream_time_base: Any,
    sample_rate: int | None,
    channels: int | None,
    decoded_samples: int,
    samples_per_channel: int,
    settings: Settings,
) -> tuple[Fraction | None, int | None]:
    duration_seconds = _packet_duration_seconds(packet, stream_time_base)
    if duration_seconds is None:
        return None, None

    rate_bound = sample_rate or settings.max_audio_sample_rate
    estimated_samples = math.ceil(duration_seconds * rate_bound)
    channel_bound = channels or settings.max_audio_channels
    if decoded_samples + estimated_samples * channel_bound > settings.max_decoded_samples:
        raise AudioServiceError(
            413,
            "decoded_sample_limit_exceeded",
            "The decoded audio exceeds the sample limit.",
        )
    max_duration = Fraction(str(settings.max_audio_duration_seconds))
    if samples_per_channel and sample_rate is not None:
        estimated_total_duration = Fraction(
            samples_per_channel, sample_rate
        ) + duration_seconds
    else:
        estimated_total_duration = duration_seconds
    if estimated_total_duration > max_duration:
        raise AudioServiceError(
            413, "duration_limit_exceeded", "The audio duration exceeds the limit."
        )
    return duration_seconds, estimated_samples


def decode_audio(audio_bytes: bytes, settings: Settings) -> DecodedAudio:
    if not audio_bytes:
        raise AudioServiceError(400, "empty_audio", "The uploaded audio file is empty.")
    if len(audio_bytes) > settings.max_upload_bytes:
        raise AudioServiceError(
            413, "file_too_large", "The uploaded file exceeds the size limit."
        )

    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to decode audio; install audio_service/requirements.txt."
        ) from exc

    try:
        with av.open(BytesIO(audio_bytes), mode="r") as container:
            audio_streams = container.streams.audio
            if not audio_streams:
                raise _unsupported_audio()

            stream = audio_streams[0]
            codec_context = stream.codec_context
            sample_rate = _positive_int(getattr(codec_context, "sample_rate", None))
            channels = _stream_channels(stream)
            format_name = getattr(container.format, "name", None)
            codec_name = getattr(codec_context, "name", None)
            stream_time_base = getattr(stream, "time_base", None)
            advertised_duration = _stream_duration_seconds(stream)
            max_duration = Fraction(str(settings.max_audio_duration_seconds))

            if sample_rate is not None and sample_rate > settings.max_audio_sample_rate:
                raise AudioServiceError(
                    415, "sample_rate_limit_exceeded", "The audio sample rate exceeds the limit."
                )
            if channels is not None and channels > settings.max_audio_channels:
                raise AudioServiceError(
                    415, "channel_limit_exceeded", "The audio channel count exceeds the limit."
                )
            if (
                advertised_duration is not None
                and advertised_duration > settings.max_audio_duration_seconds
            ):
                raise AudioServiceError(
                    413, "duration_limit_exceeded", "The audio duration exceeds the limit."
                )

            frames: list[av.AudioFrame] = []
            samples_per_channel = 0
            decoded_samples = 0
            packet_duration_without_timestamps = Fraction(0)
            first_frame_timestamp: Fraction | None = None
            previous_frame_start: Fraction | None = None
            previous_frame_end: Fraction | None = None
            packet_timestamp_origin: Fraction | None = None
            previous_packet_start: Fraction | None = None
            previous_packet_end: Fraction | None = None
            saw_data_packet = False
            saw_flush_packet = False

            def append_frame(frame: av.AudioFrame) -> None:
                nonlocal sample_rate
                nonlocal channels
                nonlocal samples_per_channel
                nonlocal decoded_samples
                nonlocal first_frame_timestamp
                nonlocal previous_frame_start
                nonlocal previous_frame_end

                frame_samples = _positive_int(getattr(frame, "samples", None))
                frame_rate = _positive_int(getattr(frame, "sample_rate", None))
                frame_channels = _frame_channels(frame)
                if frame_samples is None:
                    raise _invalid_audio()
                if sample_rate is None:
                    sample_rate = frame_rate
                elif frame_rate is not None and frame_rate != sample_rate:
                    raise _invalid_audio()
                if channels is None:
                    channels = frame_channels
                elif frame_channels is not None and frame_channels != channels:
                    raise _invalid_audio()
                if sample_rate is None or channels is None:
                    raise _invalid_audio()
                if sample_rate > settings.max_audio_sample_rate:
                    raise AudioServiceError(
                        415,
                        "sample_rate_limit_exceeded",
                        "The audio sample rate exceeds the limit.",
                    )
                if channels > settings.max_audio_channels:
                    raise AudioServiceError(
                        415,
                        "channel_limit_exceeded",
                        "The audio channel count exceeds the limit.",
                    )

                next_samples_per_channel = samples_per_channel + frame_samples
                next_decoded_samples = decoded_samples + frame_samples * channels
                signal_duration = Fraction(next_samples_per_channel, sample_rate)

                frame_pts = getattr(frame, "pts", None)
                frame_time_base = (
                    getattr(frame, "time_base", None) or stream_time_base
                )
                frame_timestamp = _timestamp_seconds(frame_pts, frame_time_base)
                timestamp_duration = Fraction(0)
                frame_duration = Fraction(frame_samples, sample_rate)
                if frame_timestamp is not None and first_frame_timestamp is None:
                    first_frame_timestamp = frame_timestamp - Fraction(
                        samples_per_channel, sample_rate
                    )

                if frame_timestamp is not None:
                    frame_start = frame_timestamp
                elif previous_frame_end is not None:
                    frame_start = previous_frame_end
                elif first_frame_timestamp is not None:
                    frame_start = first_frame_timestamp + Fraction(
                        samples_per_channel, sample_rate
                    )
                else:
                    frame_start = None

                if frame_start is not None:
                    tolerance = max(
                        _positive_time_base(frame_time_base) or Fraction(0),
                        Fraction(1, sample_rate),
                    )
                    if previous_frame_start is not None and (
                        frame_start < previous_frame_start - tolerance
                        or frame_start < (previous_frame_end or Fraction(0)) - tolerance
                    ):
                        raise _invalid_audio()
                    frame_end = frame_start + frame_duration
                    previous_frame_start = frame_start
                    previous_frame_end = frame_end
                    if first_frame_timestamp is None:
                        raise _invalid_audio()
                    timestamp_duration = frame_end - first_frame_timestamp

                    metadata_duration = getattr(frame, "duration", None)
                    positive_time_base = _positive_time_base(frame_time_base)
                    if (
                        isinstance(metadata_duration, int)
                        and metadata_duration > 0
                        and positive_time_base is not None
                    ):
                        frame_duration_metadata = (
                            Fraction(metadata_duration) * positive_time_base
                        )
                        if abs(frame_duration_metadata - frame_duration) > tolerance:
                            raise _invalid_audio()

                duration = max(signal_duration, timestamp_duration)
                if duration > max_duration:
                    raise AudioServiceError(
                        413,
                        "duration_limit_exceeded",
                        "The audio duration exceeds the limit.",
                    )
                if next_decoded_samples > settings.max_decoded_samples:
                    raise AudioServiceError(
                        413,
                        "decoded_sample_limit_exceeded",
                        "The decoded audio exceeds the sample limit.",
                    )

                frames.append(frame)
                samples_per_channel = next_samples_per_channel
                decoded_samples = next_decoded_samples

            for packet in container.demux(stream):
                if packet.is_corrupt:
                    raise _invalid_audio()
                if packet.size > 0:
                    saw_data_packet = True
                    packet_duration, _ = _preflight_packet(
                        packet,
                        stream_time_base=stream_time_base,
                        sample_rate=sample_rate,
                        channels=channels,
                        decoded_samples=decoded_samples,
                        samples_per_channel=samples_per_channel,
                        settings=settings,
                    )
                    packet_time_base = (
                        getattr(packet, "time_base", None) or stream_time_base
                    )
                    packet_start = _timestamp_seconds(
                        getattr(packet, "pts", None), packet_time_base
                    )
                    if packet_duration is not None and packet_start is None:
                        packet_duration_without_timestamps += packet_duration
                        if packet_duration_without_timestamps > max_duration:
                            raise AudioServiceError(
                                413,
                                "duration_limit_exceeded",
                                "The audio duration exceeds the limit.",
                            )
                    if packet_start is not None:
                        if packet_timestamp_origin is None:
                            packet_timestamp_origin = packet_start
                        elif previous_packet_start is not None:
                            tolerance = max(
                                _positive_time_base(packet_time_base) or Fraction(0),
                                Fraction(
                                    1,
                                    sample_rate or settings.max_audio_sample_rate,
                                ),
                            )
                            if packet_start < previous_packet_start - tolerance:
                                raise _invalid_audio()
                        previous_packet_start = packet_start
                        packet_end = packet_start + (packet_duration or Fraction(0))
                        previous_packet_end = packet_end
                        if packet_end - packet_timestamp_origin > max_duration:
                            raise AudioServiceError(
                                413,
                                "duration_limit_exceeded",
                                "The audio duration exceeds the limit.",
                            )
                elif packet.pts is None and packet.duration in (None, 0):
                    saw_flush_packet = True

                for frame in packet.decode():
                    append_frame(frame)

            if not saw_flush_packet:
                try:
                    for frame in codec_context.decode(None):
                        append_frame(frame)
                except av.error.EOFError:
                    pass

            if (
                not frames
                or not saw_data_packet
                or samples_per_channel <= 0
                or sample_rate is None
                or channels is None
            ):
                raise _invalid_audio()

            frame_timestamp_duration = (
                previous_frame_end - first_frame_timestamp
                if previous_frame_end is not None and first_frame_timestamp is not None
                else Fraction(0)
            )
            packet_timestamp_duration = (
                previous_packet_end - packet_timestamp_origin
                if previous_packet_end is not None
                and packet_timestamp_origin is not None
                else Fraction(0)
            )
            packet_duration_estimate = (
                packet_timestamp_duration + packet_duration_without_timestamps
            )
            duration_seconds = float(
                max(
                    Fraction(samples_per_channel, sample_rate),
                    frame_timestamp_duration,
                    packet_duration_estimate,
                )
            )
            duration_tolerance = max(
                Fraction(1, sample_rate),
                _positive_time_base(stream_time_base) or Fraction(0),
            )
            if (
                advertised_duration is not None
                and advertised_duration
                > Fraction(str(duration_seconds)) + duration_tolerance
            ):
                raise _invalid_audio()
            return DecodedAudio(
                frames=tuple(frames),
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
                codec_name=codec_name,
                format_name=format_name,
            )
    except AudioServiceError:
        raise
    except av.error.FFmpegError as exc:
        if type(exc).__name__ in {
            "DecoderNotFoundError",
            "DemuxerNotFoundError",
            "UnknownCodecError",
        }:
            raise _unsupported_audio() from None
        raise _invalid_audio() from None
    except (EOFError, OSError, ValueError):
        raise _invalid_audio() from None
