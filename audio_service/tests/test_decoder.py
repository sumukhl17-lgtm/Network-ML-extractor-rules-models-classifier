from hashlib import sha256
from fractions import Fraction
from io import BytesIO
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import wave

import pytest

from audio_service.config import Settings
from audio_service.decoder import DecodedAudio, _preflight_packet, decode_audio
from audio_service.service import AudioServiceError


def make_wav(
    *,
    sample_rate: int = 8_000,
    channels: int = 1,
    samples_per_channel: int = 8_000,
) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        pcm_data = bytearray()
        for sample_index in range(samples_per_channel):
            sample = round(
                12_000 * math.sin(2 * math.pi * 440 * sample_index / sample_rate)
            )
            for _ in range(channels):
                pcm_data.extend(struct.pack("<h", sample))
        wav_file.writeframes(pcm_data)
    return output.getvalue()


def make_pyav_encoded_audio(
    pyav: object,
    *,
    container_format: str,
    codec_name: str,
    sample_rate: int,
    sample_format: str,
    samples_per_frame: int,
    frame_count: int,
) -> tuple[bytes, float]:
    output = BytesIO()
    with pyav.open(output, mode="w", format=container_format) as container:
        stream = container.add_stream(codec_name, rate=sample_rate)
        stream.layout = "mono"
        stream.format = sample_format
        for frame_index in range(frame_count):
            first_sample = frame_index * samples_per_frame
            frame = pyav.AudioFrame(
                format=sample_format,
                layout="mono",
                samples=samples_per_frame,
            )
            frame.sample_rate = sample_rate
            frame.pts = first_sample
            frame.time_base = Fraction(1, sample_rate)

            if sample_format in {"s16", "s16p"}:
                samples = (
                    round(
                        12_000
                        * math.sin(
                            2
                            * math.pi
                            * 440
                            * (first_sample + index)
                            / sample_rate
                        )
                    )
                    for index in range(samples_per_frame)
                )
                sample_bytes = struct.pack(
                    "<" + "h" * samples_per_frame,
                    *samples,
                )
            elif sample_format == "fltp":
                samples = (
                    0.5
                    * math.sin(
                        2
                        * math.pi
                        * 440
                        * (first_sample + index)
                        / sample_rate
                    )
                    for index in range(samples_per_frame)
                )
                sample_bytes = struct.pack(
                    "<" + "f" * samples_per_frame,
                    *samples,
                )
            else:
                raise ValueError(f"Unsupported test sample format: {sample_format}")

            for plane in frame.planes:
                plane.update(
                    sample_bytes + bytes(plane.buffer_size - len(sample_bytes))
                )
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)

    return output.getvalue(), samples_per_frame * frame_count / sample_rate


def make_extensible_wav() -> bytes:
    standard_wav = make_wav()
    fmt_chunk = struct.pack(
        "<HHIIHHHHI",
        0xFFFE,
        1,
        8_000,
        16_000,
        2,
        16,
        22,
        16,
        4,
    )
    pcm_subformat_guid = bytes.fromhex("0100000000001000800000aa00389b71")
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_chunk) + len(pcm_subformat_guid))
        + fmt_chunk
        + pcm_subformat_guid
        + standard_wav[36:]
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def make_matroska_flac(pyav: object, timestamps: tuple[int, ...]) -> bytes:
    output = BytesIO()
    with pyav.open(output, mode="w", format="matroska") as container:
        stream = container.add_stream("flac", rate=8_000)
        stream.layout = "mono"
        stream.format = "s16"
        for pts in timestamps:
            frame = pyav.AudioFrame(format="s16", layout="mono", samples=512)
            frame.sample_rate = 8_000
            frame.pts = pts
            frame.time_base = Fraction(1, 8_000)
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


def make_malformed_matroska_packet(pyav: object) -> bytes:
    output = BytesIO()
    with pyav.open(output, mode="w", format="matroska") as container:
        stream = container.add_stream("flac", rate=8_000)
        stream.layout = "mono"
        packet = pyav.Packet(b"malformed")
        packet.stream = stream
        packet.pts = 0
        packet.dts = 0
        packet.duration = 8_000
        packet.time_base = Fraction(1, 8_000)
        container.mux(packet)
    return output.getvalue()


@pytest.fixture
def pyav() -> object:
    return pytest.importorskip("av")


def assert_audio_error(
    audio_bytes: bytes,
    settings: Settings,
    expected_status: int,
    expected_code: str,
) -> None:
    with pytest.raises(AudioServiceError) as raised:
        decode_audio(audio_bytes, settings)
    assert raised.value.status_code == expected_status
    assert raised.value.code == expected_code


def has_nonzero_decoded_samples(decoded: DecodedAudio) -> bool:
    return any(
        any(bytes(plane))
        for frame in decoded.frames
        for plane in frame.planes
    )


def test_decodes_generated_wav_and_reports_audio_metadata(pyav: object) -> None:
    audio_bytes, expected_duration = make_pyav_encoded_audio(
        pyav,
        container_format="wav",
        codec_name="pcm_s16le",
        sample_rate=8_000,
        sample_format="s16",
        samples_per_frame=1_000,
        frame_count=8,
    )
    decoded = decode_audio(audio_bytes, Settings())

    assert decoded.frames
    assert sum(frame.samples for frame in decoded.frames) > 0
    assert has_nonzero_decoded_samples(decoded)
    assert decoded.sample_rate == 8_000
    assert decoded.channels == 1
    assert decoded.duration_seconds == expected_duration
    assert decoded.codec_name == "pcm_s16le"
    assert decoded.format_name == "wav"


def test_decodes_pcm_wave_format_extensible(pyav: object) -> None:
    decoded = decode_audio(make_extensible_wav(), Settings())

    assert decoded.frames
    assert sum(frame.samples for frame in decoded.frames) > 0
    assert has_nonzero_decoded_samples(decoded)
    assert decoded.sample_rate == 8_000
    assert decoded.channels == 1
    assert decoded.duration_seconds == 1.0
    assert decoded.codec_name == "pcm_s16le"
    assert decoded.format_name == "wav"


@pytest.mark.parametrize(
    (
        "container_format",
        "codec_name",
        "sample_rate",
        "sample_format",
        "samples_per_frame",
        "frame_count",
        "expected_format",
        "expected_codec",
        "duration_tolerance",
    ),
    [
        pytest.param(
            "mp3",
            "libmp3lame",
            16_000,
            "s16p",
            1_152,
            8,
            "mp3",
            "mp3",
            0.12,
            id="mp3",
        ),
        pytest.param(
            "flac",
            "flac",
            16_000,
            "s16",
            1_024,
            8,
            "flac",
            "flac",
            0.01,
            id="standalone-flac",
        ),
        pytest.param(
            "ogg",
            "libopus",
            48_000,
            "s16",
            960,
            10,
            "ogg",
            "opus",
            0.04,
            id="ogg-opus",
        ),
        pytest.param(
            "ipod",
            "aac",
            44_100,
            "fltp",
            1_024,
            12,
            "m4a",
            "aac",
            0.08,
            id="aac-m4a",
        ),
    ],
)
def test_decodes_pyav_encoded_compatibility_formats(
    pyav: object,
    container_format: str,
    codec_name: str,
    sample_rate: int,
    sample_format: str,
    samples_per_frame: int,
    frame_count: int,
    expected_format: str,
    expected_codec: str,
    duration_tolerance: float,
) -> None:
    audio_bytes, expected_duration = make_pyav_encoded_audio(
        pyav,
        container_format=container_format,
        codec_name=codec_name,
        sample_rate=sample_rate,
        sample_format=sample_format,
        samples_per_frame=samples_per_frame,
        frame_count=frame_count,
    )

    decoded = decode_audio(audio_bytes, Settings())

    assert decoded.frames
    assert decoded.format_name is not None
    assert expected_format in decoded.format_name.split(",")
    assert decoded.codec_name is not None
    assert expected_codec in decoded.codec_name
    assert decoded.sample_rate == sample_rate
    assert decoded.channels == 1
    assert sum(frame.samples for frame in decoded.frames) > 0
    assert abs(decoded.duration_seconds - expected_duration) <= duration_tolerance
    assert has_nonzero_decoded_samples(decoded)


def test_decodes_vetted_ogg_vorbis_fixture(pyav: object) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sample-5s.ogg"
    audio_bytes = fixture_path.read_bytes()
    assert (
        sha256(audio_bytes).hexdigest()
        == "5e193a7f3c41318e01b4931ba4b81072c2f3b4c36b97243ec48b9d61c1089ead"
    )

    with pyav.open(BytesIO(audio_bytes), mode="r") as container:
        assert container.format.name == "ogg"
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "vorbis"
        assert stream.codec_context.sample_rate == 44_100
        assert stream.codec_context.layout.nb_channels == 2

    decoded = decode_audio(audio_bytes, Settings())

    assert decoded.frames
    assert decoded.format_name == "ogg"
    assert decoded.codec_name == "vorbis"
    assert decoded.sample_rate == 44_100
    assert decoded.channels == 2
    assert sum(frame.samples for frame in decoded.frames) > 0
    assert abs(decoded.duration_seconds - 5.0) <= 0.05
    assert has_nonzero_decoded_samples(decoded)


def test_rejects_empty_audio_payload() -> None:
    assert_audio_error(b"", Settings(), 400, "empty_audio")


def test_rejects_input_over_configured_upload_limit() -> None:
    assert_audio_error(
        b"12345",
        Settings(max_upload_bytes=4),
        413,
        "file_too_large",
    )


def test_rejects_random_bytes(pyav: object) -> None:
    assert_audio_error(b"not an audio file", Settings(), 400, "invalid_audio")


def test_rejects_valid_media_without_audio_stream(pyav: object) -> None:
    ppm_image = b"P6\n1 1\n255\n\x00\x00\x00"

    assert_audio_error(ppm_image, Settings(), 415, "unsupported_audio")


def test_enforces_duration_limit_during_decode(pyav: object) -> None:
    settings = Settings(max_audio_duration_seconds=0.5)

    assert_audio_error(
        make_wav(samples_per_channel=8_000),
        settings,
        413,
        "duration_limit_exceeded",
    )


def test_enforces_sample_rate_limit(pyav: object) -> None:
    settings = Settings(max_audio_sample_rate=4_000)

    assert_audio_error(
        make_wav(sample_rate=8_000),
        settings,
        415,
        "sample_rate_limit_exceeded",
    )


def test_enforces_channel_limit(pyav: object) -> None:
    settings = Settings(max_audio_channels=1)

    assert_audio_error(
        make_wav(channels=2),
        settings,
        415,
        "channel_limit_exceeded",
    )


def test_enforces_decoded_sample_limit_across_channels(pyav: object) -> None:
    settings = Settings(max_decoded_samples=15_999)

    assert_audio_error(
        make_wav(channels=2),
        settings,
        413,
        "decoded_sample_limit_exceeded",
    )


def test_rejects_truncated_wav(pyav: object) -> None:
    truncated = make_wav()[:-8]

    assert_audio_error(truncated, Settings(), 400, "invalid_audio")


def test_unknown_stream_duration_uses_decoded_frame_timestamps(pyav: object) -> None:
    audio_bytes = make_matroska_flac(pyav, (0, 8_000))
    with pyav.open(BytesIO(audio_bytes)) as container:
        assert container.streams.audio[0].duration is None

    decoded = decode_audio(audio_bytes, Settings())

    sample_duration = sum(frame.samples for frame in decoded.frames) / decoded.sample_rate
    assert decoded.duration_seconds > sample_duration


def test_timestamp_gap_counts_toward_duration_limit(pyav: object) -> None:
    audio_bytes = make_matroska_flac(pyav, (0, 8_000))

    assert_audio_error(
        audio_bytes,
        Settings(max_audio_duration_seconds=0.5),
        413,
        "duration_limit_exceeded",
    )


def test_packet_sample_budget_is_checked_before_decode(pyav: object) -> None:
    malformed_packet = make_malformed_matroska_packet(pyav)

    assert_audio_error(
        malformed_packet,
        Settings(max_decoded_samples=100),
        413,
        "decoded_sample_limit_exceeded",
    )


def test_packet_duration_budget_is_checked_before_decode(pyav: object) -> None:
    malformed_packet = make_malformed_matroska_packet(pyav)

    assert_audio_error(
        malformed_packet,
        Settings(max_audio_duration_seconds=0.1),
        413,
        "duration_limit_exceeded",
    )


def test_preflight_allows_missing_packet_duration() -> None:
    duration, estimated_samples = _preflight_packet(
        SimpleNamespace(duration=None, time_base=None),
        stream_time_base=None,
        sample_rate=8_000,
        channels=1,
        decoded_samples=0,
        samples_per_channel=0,
        settings=Settings(),
    )

    assert duration is None
    assert estimated_samples is None


def test_preflight_bounds_packets_when_sample_rate_metadata_is_missing() -> None:
    duration, estimated_samples = _preflight_packet(
        SimpleNamespace(duration=1_000, time_base=Fraction(1, 8_000)),
        stream_time_base=None,
        sample_rate=None,
        channels=None,
        decoded_samples=0,
        samples_per_channel=0,
        settings=Settings(),
    )

    assert duration == Fraction(1, 8)
    assert estimated_samples == 24_000


def test_rejects_malformed_packet_without_exposing_decoder_error(
    pyav: object,
) -> None:
    malformed_packet = make_malformed_matroska_packet(pyav)

    assert_audio_error(malformed_packet, Settings(), 400, "invalid_audio")


@pytest.mark.parametrize(
    "settings",
    [
        {"max_audio_duration_seconds": 0},
        {"max_audio_duration_seconds": float("inf")},
        {"max_audio_sample_rate": 0},
        {"max_audio_channels": 0},
        {"max_decoded_samples": 0},
    ],
)
def test_rejects_invalid_audio_limit_settings(settings: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        Settings(**settings)
