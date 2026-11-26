import asyncio
from dataclasses import dataclass, field
from io import BytesIO

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from audio_service.api import create_app
from audio_service.config import Settings
from audio_service.schemas import (
    ClassificationResponse,
    ModelStatusResponse,
    ReadinessResponse,
)
from audio_service.service import AudioServiceError, read_limited_upload

TOKEN = "test-audio-service-token"


@dataclass
class FakeWorkerClient:
    result: ClassificationResponse | None = None
    error: Exception | None = None
    readiness_result: ReadinessResponse = field(
        default_factory=lambda: ReadinessResponse(
            status="ready",
            model=ModelStatusResponse(loaded=True, version="test-model-v1"),
        )
    )
    readiness_error: AudioServiceError | None = None
    calls: int = 0
    readiness_calls: int = 0
    received_audio: bytes | None = None
    received_filename: str | None = None
    received_content_type: str | None = None

    async def classify(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None,
    ) -> ClassificationResponse:
        self.calls += 1
        self.received_audio = audio_bytes
        self.received_filename = filename
        self.received_content_type = content_type

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError("The test worker result was not configured.")

        return self.result

    async def readiness(self) -> ReadinessResponse:
        self.readiness_calls += 1

        if self.readiness_error is not None:
            raise self.readiness_error

        return self.readiness_result


def make_response() -> ClassificationResponse:
    return ClassificationResponse(
        filename="recording.wav",
        category="dog",
        sound_group="animals",
        confidence=0.93,
        best_guess="dog",
        top_k=[{"category": "dog", "confidence": 0.93}],
        reason="A mock inference result for API contract testing.",
        source="esc50",
        model_version="test-model-v1",
        duration_seconds=2.5,
        sample_rate=16000,
        channels=1,
    )


def make_ready_response() -> ReadinessResponse:
    return ReadinessResponse(
        status="ready",
        model=ModelStatusResponse(
            loaded=True,
            version="test-model-v1",
        ),
    )


def make_not_ready_response() -> ReadinessResponse:
    return ReadinessResponse(
        status="not_ready",
        model=ModelStatusResponse(
            loaded=False,
            version=None,
        ),
    )


def make_client(
    *,
    worker_client: FakeWorkerClient | None = None,
    token: str | None = TOKEN,
    max_upload_bytes: int = 1024,
    max_concurrent_decode_jobs: int = 1,
) -> TestClient:
    app = create_app(
        settings=Settings(
            service_token=token,
            max_upload_bytes=max_upload_bytes,
            max_concurrent_decode_jobs=max_concurrent_decode_jobs,
        ),
        worker_client=worker_client
        or FakeWorkerClient(result=make_response()),
    )
    return TestClient(app)


class TrackingUpload:
    def __init__(self, content: bytes) -> None:
        self.filename = "recording.wav"
        self.content_type = "audio/wav"
        self.read_sizes: list[int] = []
        self.stream = BytesIO(content)
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.stream.read(size)

    async def close(self) -> None:
        self.closed = True


def invoke_classify(
    app: FastAPI,
    upload: TrackingUpload,
    token: str | None,
    *,
    receive=None,
) -> ClassificationResponse:
    route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == "/internal/v1/classify"
    )

    async def never_receive():
        await asyncio.Future()

    request = Request(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "method": "POST",
            "path": "/internal/v1/classify",
        },
        receive=receive or never_receive,
    )

    return asyncio.run(
        route.endpoint(
            request=request,
            file=upload,
            x_audio_service_token=token,
        )
    )


def test_health_succeeds_without_loaded_model() -> None:
    with make_client(
        worker_client=FakeWorkerClient(
            result=make_response(),
            readiness_result=make_not_ready_response(),
        )
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_is_503_when_worker_model_is_unavailable() -> None:
    worker = FakeWorkerClient(
        result=make_response(),
        readiness_result=make_not_ready_response(),
    )

    with make_client(worker_client=worker) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "model": {"loaded": False, "version": None},
    }
    assert worker.readiness_calls == 1


def test_readiness_is_200_when_worker_model_is_ready() -> None:
    worker = FakeWorkerClient(
        result=make_response(),
        readiness_result=make_ready_response(),
    )

    with make_client(worker_client=worker) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": {"loaded": True, "version": "test-model-v1"},
    }
    assert worker.readiness_calls == 1


def test_readiness_returns_503_when_worker_is_unavailable() -> None:
    worker = FakeWorkerClient(
        result=make_response(),
        readiness_error=AudioServiceError(
            503,
            "worker_unavailable",
            "The audio worker is unavailable.",
        ),
    )

    with make_client(worker_client=worker) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "model": {"loaded": False, "version": None},
    }


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Audio-Service-Token": "wrong-token"}],
)
def test_classification_rejects_missing_or_invalid_authentication(
    headers: dict[str, str],
) -> None:
    worker = FakeWorkerClient(result=make_response())

    with make_client(worker_client=worker) as client:
        response = client.post(
            "/internal/v1/classify",
            files={"file": ("recording.wav", b"audio bytes", "audio/wav")},
            headers=headers,
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert TOKEN not in response.text
    assert worker.calls == 0


def test_classification_fails_closed_when_token_is_not_configured() -> None:
    worker = FakeWorkerClient(result=make_response())

    with make_client(worker_client=worker, token=None) as client:
        response = client.post(
            "/internal/v1/classify",
            files={"file": ("recording.wav", b"audio bytes", "audio/wav")},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "authentication_not_configured"
    assert worker.calls == 0


@pytest.mark.parametrize("token", [None, ""])
def test_production_startup_requires_nonempty_token(
    monkeypatch: pytest.MonkeyPatch,
    token: str | None,
) -> None:
    if token is None:
        monkeypatch.delenv("AUDIO_SERVICE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("AUDIO_SERVICE_TOKEN", token)

    app = create_app()

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(RuntimeError, match="AUDIO_SERVICE_TOKEN"):
        asyncio.run(run_lifespan())


def test_empty_upload_is_rejected() -> None:
    worker = FakeWorkerClient(result=make_response())

    with make_client(worker_client=worker) as client:
        response = client.post(
            "/internal/v1/classify",
            files={"file": ("recording.wav", b"", "audio/wav")},
            headers={"X-Audio-Service-Token": TOKEN},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"
    assert worker.calls == 0


def test_oversized_upload_is_rejected() -> None:
    worker = FakeWorkerClient(result=make_response())

    with make_client(
        worker_client=worker,
        max_upload_bytes=4,
    ) as client:
        response = client.post(
            "/internal/v1/classify",
            files={"file": ("recording.wav", b"12345", "audio/wav")},
            headers={"X-Audio-Service-Token": TOKEN},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    assert worker.calls == 0


def test_limited_upload_reads_no_more_than_limit_plus_one() -> None:
    class TrackingReadUpload:
        def __init__(self) -> None:
            self.read_sizes: list[int] = []
            self.stream = BytesIO(b"0123456789")

        async def read(self, size: int = -1) -> bytes:
            self.read_sizes.append(size)
            return self.stream.read(size)

    upload = TrackingReadUpload()

    with pytest.raises(AudioServiceError) as raised:
        asyncio.run(read_limited_upload(upload, max_bytes=4))

    assert raised.value.status_code == 413
    assert upload.read_sizes == [5]


def test_classification_returns_safe_error_when_worker_is_unavailable() -> None:
    worker = FakeWorkerClient(
        error=AudioServiceError(
            503,
            "worker_unavailable",
            "The audio worker is unavailable.",
        )
    )

    with make_client(worker_client=worker) as client:
        response = client.post(
            "/internal/v1/classify",
            files={"file": ("recording.wav", b"audio bytes", "audio/wav")},
            headers={"X-Audio-Service-Token": TOKEN},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "worker_unavailable"
    assert response.json()["error"]["message"] == "The audio worker is unavailable."
    assert worker.calls == 1


def test_worker_classification_returns_schema_valid_response() -> None:
    worker = FakeWorkerClient(result=make_response())

    with make_client(worker_client=worker) as client:
        response = client.post(
            "/internal/v1/classify",
            files={
                "file": (
                    "recording.wav",
                    b"mock audio fixture",
                    "audio/wav",
                )
            },
            headers={"X-Audio-Service-Token": TOKEN},
        )

    assert response.status_code == 200
    assert response.json() == make_response().model_dump()
    assert worker.calls == 1
    assert worker.received_audio == b"mock audio fixture"
    assert worker.received_filename == "recording.wav"
    assert worker.received_content_type == "audio/wav"


def test_api_rejects_excess_worker_request_without_queueing() -> None:
    async def run() -> None:
        class BlockingWorker:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.calls = 0

            async def classify(
                self,
                audio_bytes: bytes,
                filename: str,
                content_type: str | None,
            ) -> ClassificationResponse:
                assert audio_bytes
                assert filename == "recording.wav"
                assert content_type == "audio/wav"
                self.calls += 1
                self.started.set()
                await self.release.wait()
                return make_response()

            async def readiness(self) -> ReadinessResponse:
                return make_ready_response()

        worker = BlockingWorker()
        app = create_app(
            settings=Settings(
                service_token=TOKEN,
                max_concurrent_decode_jobs=1,
            ),
            worker_client=worker,
        )

        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://audio-api",
        ) as client:
            first_request = asyncio.create_task(
                client.post(
                    "/internal/v1/classify",
                    headers={"X-Audio-Service-Token": TOKEN},
                    files={"file": ("recording.wav", b"first", "audio/wav")},
                )
            )

            await worker.started.wait()

            second_response = await client.post(
                "/internal/v1/classify",
                headers={"X-Audio-Service-Token": TOKEN},
                files={"file": ("recording.wav", b"second", "audio/wav")},
            )

            assert second_response.status_code == 429
            assert second_response.json()["error"]["code"] == "worker_overloaded"
            assert worker.calls == 1

            worker.release.set()
            first_response = await first_request

            assert first_response.status_code == 200
            assert first_response.json() == make_response().model_dump()

    asyncio.run(run())


def test_gateway_disconnect_cancels_worker_and_releases_capacity() -> None:
    async def run() -> None:
        class CancellableWorker:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def classify(
                self,
                _: bytes,
                filename: str,
                content_type: str | None,
            ) -> ClassificationResponse:
                assert filename == "recording.wav"
                assert content_type == "audio/wav"
                self.started.set()

                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise

            async def readiness(self) -> ReadinessResponse:
                return make_ready_response()

        worker = CancellableWorker()
        app = create_app(
            settings=Settings(
                service_token=TOKEN,
                max_concurrent_decode_jobs=1,
            ),
            worker_client=worker,
        )

        disconnected = asyncio.Event()

        async def receive_disconnect():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        upload = TrackingUpload(b"audio bytes")

        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/internal/v1/classify"
        )

        request = Request(
            {
                "type": "http",
                "app": app,
                "headers": [],
                "method": "POST",
                "path": "/internal/v1/classify",
            },
            receive=receive_disconnect,
        )

        task = asyncio.create_task(
            endpoint(
                request=request,
                file=upload,
                x_audio_service_token=TOKEN,
            )
        )

        await worker.started.wait()
        disconnected.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert worker.cancelled.is_set()
        assert upload.closed
        assert await app.state.decode_capacity.try_acquire()
        await app.state.decode_capacity.release()

    asyncio.run(run())


def test_gateway_task_cancellation_propagates_and_releases_capacity() -> None:
    async def run() -> None:
        class CancellableWorker:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def classify(
                self,
                _: bytes,
                filename: str,
                content_type: str | None,
            ) -> ClassificationResponse:
                assert filename == "recording.wav"
                assert content_type == "audio/wav"
                self.started.set()

                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise

            async def readiness(self) -> ReadinessResponse:
                return make_ready_response()

        worker = CancellableWorker()
        app = create_app(
            settings=Settings(
                service_token=TOKEN,
                max_concurrent_decode_jobs=1,
            ),
            worker_client=worker,
        )

        upload = TrackingUpload(b"audio bytes")

        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/internal/v1/classify"
        )

        async def never_receive():
            await asyncio.Future()

        task = asyncio.create_task(
            endpoint(
                request=Request(
                    {
                        "type": "http",
                        "app": app,
                        "headers": [],
                        "method": "POST",
                        "path": "/internal/v1/classify",
                    },
                    receive=never_receive,
                ),
                file=upload,
                x_audio_service_token=TOKEN,
            )
        )

        await worker.started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert worker.cancelled.is_set()
        assert upload.closed
        assert await app.state.decode_capacity.try_acquire()
        await app.state.decode_capacity.release()

    asyncio.run(run())


def test_upload_is_closed_after_successful_classification() -> None:
    app = create_app(
        settings=Settings(
            service_token=TOKEN,
            max_upload_bytes=1024,
        ),
        worker_client=FakeWorkerClient(result=make_response()),
    )

    upload = TrackingUpload(b"audio bytes")
    response = invoke_classify(app, upload, TOKEN)

    assert response == make_response()
    assert upload.closed


@pytest.mark.parametrize(
    ("content", "token", "max_bytes", "worker_error", "error_code"),
    [
        (b"audio bytes", "wrong-token", 1024, None, "unauthorized"),
        (b"", TOKEN, 1024, None, "empty_file"),
        (b"12345", TOKEN, 4, None, "file_too_large"),
        (
            b"audio bytes",
            TOKEN,
            1024,
            AudioServiceError(
                503,
                "worker_unavailable",
                "The audio worker is unavailable.",
            ),
            "worker_unavailable",
        ),
    ],
)
def test_upload_is_closed_after_classification_failure(
    content: bytes,
    token: str,
    max_bytes: int,
    worker_error: Exception | None,
    error_code: str,
) -> None:
    app = create_app(
        settings=Settings(
            service_token=TOKEN,
            max_upload_bytes=max_bytes,
        ),
        worker_client=FakeWorkerClient(
            result=make_response(),
            error=worker_error,
        ),
    )

    upload = TrackingUpload(content)

    with pytest.raises(AudioServiceError) as raised:
        invoke_classify(app, upload, token)

    assert raised.value.code == error_code
    assert upload.closed


def test_upload_is_closed_after_unexpected_worker_exception() -> None:
    app = create_app(
        settings=Settings(service_token=TOKEN),
        worker_client=FakeWorkerClient(
            error=RuntimeError("unexpected worker failure"),
        ),
    )

    upload = TrackingUpload(b"audio bytes")

    with pytest.raises(RuntimeError, match="unexpected worker failure"):
        invoke_classify(app, upload, TOKEN)

    assert upload.closed
