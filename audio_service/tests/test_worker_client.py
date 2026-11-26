import asyncio

import httpx
import pytest

from audio_service.config import Settings
from audio_service.schemas import DecodeMetadataResponse
from audio_service.service import AudioServiceError
from audio_service.worker_client import HttpAudioWorkerClient
from audio_service.worker_api import create_worker_app


def test_worker_unavailable_is_mapped_to_safe_service_error() -> None:
    def fail_request(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private connection details")

    client = HttpAudioWorkerClient(
        Settings(service_token="test-token"),
        transport=httpx.MockTransport(fail_request),
    )

    async def run() -> None:
        with pytest.raises(AudioServiceError) as raised:
            await client.decode(b"audio")
        assert raised.value.status_code == 503
        assert raised.value.code == "worker_unavailable"
        assert "private connection details" not in raised.value.message

    asyncio.run(run())


def test_malformed_worker_response_is_rejected_without_leaking_body() -> None:
    def malformed_response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"decoder traceback with local paths")

    client = HttpAudioWorkerClient(
        Settings(service_token="test-token"),
        transport=httpx.MockTransport(malformed_response),
    )

    async def run() -> None:
        with pytest.raises(AudioServiceError) as raised:
            await client.decode(b"audio")
        assert raised.value.status_code == 502
        assert raised.value.code == "invalid_worker_result"
        assert "traceback" not in raised.value.message

    asyncio.run(run())


def test_worker_error_body_is_mapped_to_fixed_safe_message() -> None:
    def error_response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "invalid_audio",
                    "message": "raw ffmpeg exception and local path",
                }
            },
        )

    client = HttpAudioWorkerClient(
        Settings(service_token="test-token"),
        transport=httpx.MockTransport(error_response),
    )

    async def run() -> None:
        with pytest.raises(AudioServiceError) as raised:
            await client.decode(b"audio")
        assert raised.value.code == "invalid_audio"
        assert "raw ffmpeg" not in raised.value.message

    asyncio.run(run())


def test_http_client_receives_only_bounded_decode_metadata() -> None:
    class FakeSupervisor:
        async def decode(
            self, _: bytes, __: Settings
        ) -> DecodeMetadataResponse:
            return DecodeMetadataResponse(
                duration_seconds=1,
                sample_rate=8_000,
                channels=1,
                codec_name="pcm_s16le",
                format_name="wav",
            )

    app = create_worker_app(
        settings=Settings(service_token="test-token"),
        supervisor=FakeSupervisor(),
    )
    client = HttpAudioWorkerClient(
        Settings(
            service_token="test-token",
            worker_url="http://audio-worker",
        ),
        transport=httpx.ASGITransport(app=app),
    )

    async def run() -> None:
        result = await client.decode(b"bounded upload")
        assert result.duration_seconds == 1
        assert result.sample_rate == 8_000
        assert result.channels == 1

    asyncio.run(run())


def test_oversized_worker_response_is_rejected() -> None:
    def oversized_response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (16 * 1024 + 1))

    client = HttpAudioWorkerClient(
        Settings(service_token="test-token"),
        transport=httpx.MockTransport(oversized_response),
    )

    async def run() -> None:
        with pytest.raises(AudioServiceError) as raised:
            await client.decode(b"audio")
        assert raised.value.code == "invalid_worker_response"

    asyncio.run(run())
