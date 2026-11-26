import asyncio
from io import BytesIO
import multiprocessing
import os
import signal
import sys
import threading
import wave
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from audio_service.config import Settings
from audio_service.schemas import DecodeMetadataResponse
from audio_service.service import AudioServiceError
from audio_service.worker_api import create_worker_app
from audio_service.worker_runtime import (
    ProcessDecodeSupervisor,
    _apply_child_resource_limits,
)

TOKEN = "test-audio-service-token"


def make_wav() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(b"\x00\x00" * 8_000)
    return output.getvalue()


def metadata() -> DecodeMetadataResponse:
    return DecodeMetadataResponse(
        duration_seconds=1.0,
        sample_rate=8_000,
        channels=1,
        codec_name="pcm_s16le",
        format_name="wav",
    )


def _blocking_child(
    connection: object,
    _: bytes,
    __: Settings,
    started: object,
    release: object,
    pid_queue: object,
) -> None:
    pid_queue.put(os.getpid())
    started.set()
    release.wait()


def _crashing_child(connection: object, _: bytes, __: Settings) -> None:
    os._exit(17)


def _metadata_child(connection: object, _: bytes, __: Settings) -> None:
    connection.send(
        {
            "kind": "ok",
            "result": metadata().model_dump(),
        }
    )
    connection.close()


def _sigterm_ignoring_child(
    connection: object,
    _: bytes,
    __: Settings,
    started: object,
    pid_queue: object,
) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pid_queue.put(os.getpid())
    started.set()
    while True:
        signal.pause()


def _malformed_result_child(connection: object, _: bytes, __: Settings) -> None:
    connection.send({"kind": "ok", "result": {"duration_seconds": 1.0}})
    connection.close()


def _missing_result_child(connection: object, _: bytes, __: Settings) -> None:
    connection.close()


def _serialization_failure_child(
    connection: object, _: bytes, __: Settings
) -> None:
    try:
        connection.send({"kind": "ok", "result": lambda: None})
    except Exception:
        pass
    finally:
        connection.close()


def assert_process_reaped(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_supervisor_decodes_in_a_spawned_child_process() -> None:
    pytest.importorskip("av")

    async def run() -> DecodeMetadataResponse:
        return await ProcessDecodeSupervisor().decode(
            make_wav(),
            Settings(
                child_cpu_limit_seconds=None,
                child_address_space_limit_bytes=None,
            ),
        )

    result = asyncio.run(run())
    assert result == metadata()


def test_timeout_terminates_and_reaps_the_decode_child() -> None:
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    release = context.Event()
    pid_queue = context.Queue()
    supervisor = ProcessDecodeSupervisor(
        child_target=_blocking_child,
        child_target_args=(started, release, pid_queue),
        terminate_grace_seconds=0.05,
    )
    settings = Settings(
        decode_timeout_seconds=0.5,
        child_cpu_limit_seconds=None,
        child_address_space_limit_bytes=None,
    )

    async def run() -> None:
        task = asyncio.create_task(supervisor.decode(b"bounded", settings))
        pid = await asyncio.to_thread(pid_queue.get, True, 5)
        assert started.is_set()
        with pytest.raises(AudioServiceError) as raised:
            await task
        assert raised.value.status_code == 504
        assert raised.value.code == "decode_timeout"
        assert_process_reaped(pid)

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "pause"),
    reason="requires POSIX SIGTERM ignore behavior",
)
def test_timeout_escalates_to_kill_for_sigterm_ignoring_child() -> None:
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    pid_queue = context.Queue()

    class EscalationTrackingSupervisor(ProcessDecodeSupervisor):
        def __init__(self) -> None:
            super().__init__(
                child_target=_sigterm_ignoring_child,
                child_target_args=(started, pid_queue),
                terminate_grace_seconds=0.05,
            )
            self.kill_calls = 0

        def _kill(self, process: multiprocessing.Process) -> None:
            self.kill_calls += 1
            super()._kill(process)

    supervisor = EscalationTrackingSupervisor()

    async def run() -> None:
        task = asyncio.create_task(
            supervisor.decode(
                b"bounded",
                Settings(
                    decode_timeout_seconds=0.5,
                    child_cpu_limit_seconds=None,
                    child_address_space_limit_bytes=None,
                ),
            )
        )
        pid = await asyncio.to_thread(pid_queue.get, True, 5)
        assert started.is_set()
        with pytest.raises(AudioServiceError) as raised:
            await task
        assert raised.value.code == "decode_timeout"
        assert supervisor.kill_calls == 1
        assert_process_reaped(pid)

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()


def test_cancellation_terminates_and_reaps_the_decode_child() -> None:
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    release = context.Event()
    pid_queue = context.Queue()
    supervisor = ProcessDecodeSupervisor(
        child_target=_blocking_child,
        child_target_args=(started, release, pid_queue),
        terminate_grace_seconds=0.05,
    )
    settings = Settings(
        decode_timeout_seconds=30,
        child_cpu_limit_seconds=None,
        child_address_space_limit_bytes=None,
    )

    async def run() -> None:
        task = asyncio.create_task(supervisor.decode(b"bounded", settings))
        pid = await asyncio.to_thread(pid_queue.get, True, 5)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert_process_reaped(pid)

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()


def test_health_and_readiness_stay_responsive_during_delayed_startup() -> None:
    async def run() -> None:
        start_entered = threading.Event()
        release_start = threading.Event()

        def delayed_starter(process: multiprocessing.Process) -> None:
            start_entered.set()
            if not release_start.wait(timeout=10):
                raise TimeoutError("Test did not release delayed process startup.")
            process.start()

        class ReadyInferenceSupervisor:
            is_running = True
            model_version = "test-model-v1"

            async def start(self) -> None:
                return None

            async def aclose(self) -> None:
                return None

        supervisor = ProcessDecodeSupervisor(
            child_target=_metadata_child,
            process_starter=delayed_starter,
        )
        app = create_worker_app(
            settings=Settings(
                service_token=TOKEN,
                child_cpu_limit_seconds=None,
                child_address_space_limit_bytes=None,
            ),
            supervisor=supervisor,
            inference_supervisor=ReadyInferenceSupervisor(),
        )

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://audio-worker",
            ) as client:
                decode_task = asyncio.create_task(
                    client.post(
                        "/internal/v1/decode",
                        headers={"X-Audio-Service-Token": TOKEN},
                        files={
                            "file": (
                                "file.bin",
                                b"bounded",
                                "application/octet-stream",
                            )
                        },
                    )
                )
                assert await asyncio.to_thread(start_entered.wait, 5)

                health = await client.get("/health")
                readiness = await client.get("/ready")

                assert health.status_code == 200
                assert readiness.status_code == 200
                assert readiness.json()["model"]["version"] == "test-model-v1"

                release_start.set()
                response = await decode_task
                assert response.status_code == 200
                assert response.json() == metadata().model_dump()
        finally:
            # Always release the deliberately blocked startup thread.
            release_start.set()

    asyncio.run(run())


def test_cancellation_during_startup_waits_for_child_cleanup() -> None:
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    release_child = context.Event()
    pid_queue = context.Queue()
    start_entered = threading.Event()
    release_start = threading.Event()

    def delayed_starter(process: multiprocessing.Process) -> None:
        start_entered.set()
        release_start.wait()
        process.start()
        pid_queue.put(process.pid)

    cleanup_scheduled = asyncio.Event()

    class CleanupTrackingSupervisor(ProcessDecodeSupervisor):
        def _schedule_start_cleanup(self, *args):
            cleanup_scheduled.set()
            return super()._schedule_start_cleanup(*args)

    supervisor = CleanupTrackingSupervisor(
        child_target=_blocking_child,
        child_target_args=(started, release_child, pid_queue),
        process_starter=delayed_starter,
        terminate_grace_seconds=0.05,
    )
    settings = Settings(
        decode_timeout_seconds=30,
        child_cpu_limit_seconds=None,
        child_address_space_limit_bytes=None,
    )

    async def run() -> None:
        task = asyncio.create_task(supervisor.decode(b"bounded", settings))
        assert await asyncio.to_thread(start_entered.wait, 5)
        task.cancel()
        await cleanup_scheduled.wait()
        assert not task.done()
        release_start.set()
        pid = await asyncio.to_thread(pid_queue.get, True, 5)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert_process_reaped(pid)

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()


def test_process_start_failure_returns_safe_error() -> None:
    def fail_start(_: multiprocessing.Process) -> None:
        raise OSError("private startup failure")

    supervisor = ProcessDecodeSupervisor(process_starter=fail_start)

    async def run() -> None:
        with pytest.raises(AudioServiceError) as raised:
            await supervisor.decode(
                b"bounded",
                Settings(
                    child_cpu_limit_seconds=None,
                    child_address_space_limit_bytes=None,
                ),
            )
        assert raised.value.code == "decode_worker_failed"
        assert "private startup failure" not in raised.value.message

    asyncio.run(run())


def test_child_crash_returns_safe_error_and_is_reaped() -> None:
    supervisor = ProcessDecodeSupervisor(child_target=_crashing_child)

    async def run() -> None:
        with pytest.raises(AudioServiceError) as raised:
            await supervisor.decode(
                b"bounded",
                Settings(
                    child_cpu_limit_seconds=None,
                    child_address_space_limit_bytes=None,
                ),
            )
        assert raised.value.status_code == 503
        assert raised.value.code == "decode_worker_failed"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("target", "expected_code"),
    [
        (_malformed_result_child, "invalid_worker_result"),
        (_missing_result_child, "decode_worker_failed"),
        (_serialization_failure_child, "decode_worker_failed"),
    ],
)
def test_malformed_or_missing_child_result_is_rejected(
    target: object,
    expected_code: str,
) -> None:
    supervisor = ProcessDecodeSupervisor(child_target=target)

    async def run() -> None:
        with pytest.raises(AudioServiceError) as raised:
            await supervisor.decode(
                b"bounded",
                Settings(
                    child_cpu_limit_seconds=None,
                    child_address_space_limit_bytes=None,
                ),
            )
        assert raised.value.code == expected_code

    asyncio.run(run())


def test_resource_limits_are_not_applied_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    _apply_child_resource_limits(Settings())


def test_linux_resource_limits_are_applied_to_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []
    fake_resource = SimpleNamespace(
        RLIMIT_CPU=1,
        RLIMIT_AS=2,
        setrlimit=lambda resource_id, limits: calls.append((resource_id, limits)),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    _apply_child_resource_limits(
        Settings(
            child_cpu_limit_seconds=7,
            child_address_space_limit_bytes=11_000,
        )
    )

    assert calls == [(1, (7, 8)), (2, (11_000, 11_000))]


class TrackingUpload:
    filename = "untrusted-name.bin"
    content_type = "application/octet-stream"

    def __init__(self, content: bytes) -> None:
        self._stream = BytesIO(content)
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    async def close(self) -> None:
        self.closed = True


class ConnectedRequest(Request):
    async def is_disconnected(self) -> bool:
        return False


def route_endpoint(app: FastAPI):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/internal/v1/decode"
    )


def make_request(app: FastAPI) -> ConnectedRequest:
    return ConnectedRequest(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "method": "POST",
            "path": "/internal/v1/decode",
        }
    )


class ImmediateSupervisor:
    def __init__(self, error: AudioServiceError | None = None) -> None:
        self.error = error

    async def decode(
        self, _: bytes, __: Settings
    ) -> DecodeMetadataResponse:
        if self.error is not None:
            raise self.error
        return metadata()


@pytest.mark.parametrize(
    "supervisor",
    [
        ImmediateSupervisor(),
        ImmediateSupervisor(
            AudioServiceError(504, "decode_timeout", "Audio decoding timed out.")
        ),
    ],
)
def test_worker_closes_upload_after_success_or_timeout(
    supervisor: ImmediateSupervisor,
) -> None:
    app = create_worker_app(
        settings=Settings(service_token=TOKEN),
        supervisor=supervisor,
    )
    upload = TrackingUpload(b"audio")
    endpoint = route_endpoint(app)

    async def run() -> None:
        if supervisor.error is None:
            result = await endpoint(
                request=make_request(app),
                file=upload,
                x_audio_service_token=TOKEN,
            )
            assert result == metadata()
        else:
            with pytest.raises(AudioServiceError) as raised:
                await endpoint(
                    request=make_request(app),
                    file=upload,
                    x_audio_service_token=TOKEN,
                )
            assert raised.value.code == "decode_timeout"

    asyncio.run(run())
    assert upload.closed


def test_worker_rejects_missing_or_invalid_authentication() -> None:
    app = create_worker_app(
        settings=Settings(service_token=TOKEN),
        supervisor=ImmediateSupervisor(),
    )
    endpoint = route_endpoint(app)

    async def run(token: str | None) -> None:
        upload = TrackingUpload(b"audio")
        with pytest.raises(AudioServiceError) as raised:
            await endpoint(
                request=make_request(app),
                file=upload,
                x_audio_service_token=token,
            )
        assert raised.value.code == "unauthorized"
        assert upload.closed

    asyncio.run(run(None))
    asyncio.run(run("incorrect"))


def test_worker_startup_requires_a_nonempty_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDIO_SERVICE_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="AUDIO_SERVICE_TOKEN"):
        with TestClient(create_worker_app()):
            pass


@pytest.mark.parametrize(
    "settings",
    [
        {"decode_timeout_seconds": 0},
        {"decode_timeout_seconds": float("inf")},
        {"max_concurrent_decode_jobs": 0},
        {"child_cpu_limit_seconds": 0},
        {"child_address_space_limit_bytes": 0},
        {"worker_url": "file:///tmp/worker"},
        {"worker_url": "http://user:secret@audio-worker:8002"},
    ],
)
def test_invalid_worker_resource_settings_are_rejected(
    settings: dict[str, int | float | str],
) -> None:
    with pytest.raises(ValueError):
        Settings(**settings)


def test_worker_rejects_oversized_upload() -> None:
    app = create_worker_app(
        settings=Settings(service_token=TOKEN, max_upload_bytes=4),
        supervisor=ImmediateSupervisor(),
    )
    endpoint = route_endpoint(app)
    upload = TrackingUpload(b"12345")

    async def run() -> None:
        with pytest.raises(AudioServiceError) as raised:
            await endpoint(
                request=make_request(app),
                file=upload,
                x_audio_service_token=TOKEN,
            )
        assert raised.value.code == "file_too_large"

    asyncio.run(run())
    assert upload.closed


def test_worker_cancellation_closes_upload_and_reaps_child() -> None:
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    release = context.Event()
    pid_queue = context.Queue()
    supervisor = ProcessDecodeSupervisor(
        child_target=_blocking_child,
        child_target_args=(started, release, pid_queue),
        terminate_grace_seconds=0.05,
    )

    class DisconnectWhenStartedRequest(ConnectedRequest):
        async def is_disconnected(self) -> bool:
            return started.is_set()

    app = create_worker_app(
        settings=Settings(
            service_token=TOKEN,
            decode_timeout_seconds=30,
            child_cpu_limit_seconds=None,
            child_address_space_limit_bytes=None,
        ),
        supervisor=supervisor,
    )
    upload = TrackingUpload(b"audio")
    endpoint = route_endpoint(app)

    async def run() -> int:
        task = asyncio.create_task(
            endpoint(
                request=DisconnectWhenStartedRequest(
                    {
                        "type": "http",
                        "app": app,
                        "headers": [],
                        "method": "POST",
                        "path": "/internal/v1/decode",
                    }
                ),
                file=upload,
                x_audio_service_token=TOKEN,
            )
        )
        pid = await asyncio.to_thread(pid_queue.get, True, 5)
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid

    try:
        pid = asyncio.run(run())
        assert upload.closed
        assert_process_reaped(pid)
    finally:
        pid_queue.close()
        pid_queue.join_thread()


def test_worker_rejects_excess_request_without_queueing() -> None:
    async def run() -> None:
        class BlockingSupervisor:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def decode(
                self, _: bytes, __: Settings
            ) -> DecodeMetadataResponse:
                self.started.set()
                await self.release.wait()
                return metadata()

        supervisor = BlockingSupervisor()
        app = create_worker_app(
            settings=Settings(
                service_token=TOKEN,
                max_concurrent_decode_jobs=1,
                child_cpu_limit_seconds=None,
                child_address_space_limit_bytes=None,
            ),
            supervisor=supervisor,
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://audio-worker",
        ) as client:
            first_request = asyncio.create_task(
                client.post(
                    "/internal/v1/decode",
                    headers={"X-Audio-Service-Token": TOKEN},
                    files={"file": ("file.bin", b"first", "application/octet-stream")},
                )
            )
            await supervisor.started.wait()
            second_response = await client.post(
                "/internal/v1/decode",
                headers={"X-Audio-Service-Token": TOKEN},
                files={"file": ("file.bin", b"second", "application/octet-stream")},
            )
            assert second_response.status_code == 429
            assert second_response.json()["error"]["code"] == "worker_overloaded"

            supervisor.release.set()
            first_response = await first_request
            assert first_response.status_code == 200
            assert first_response.json() == metadata().model_dump()

    asyncio.run(run())
