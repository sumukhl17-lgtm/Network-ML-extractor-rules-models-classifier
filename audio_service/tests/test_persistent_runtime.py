
import asyncio
import multiprocessing
import os
import time

import pytest

from audio_service.service import AudioServiceError
from audio_service.persistent_runtime import PersistentProcessSupervisor


def _initialize_state() -> str:
    return "initialized"


def _failing_initializer() -> None:
    raise RuntimeError("private initializer failure")


def _handle_request(
    payload: object,
    state: str,
    pid_queue: object,
) -> tuple[int, object, str]:
    pid_queue.put(os.getpid())

    if payload == "fail":
        raise RuntimeError("private handler failure")

    if payload == "invalid":
        raise AudioServiceError(
            400,
            "invalid_audio",
            "The uploaded audio is invalid.",
        )

    if payload == "unsupported":
        raise AudioServiceError(
            415,
            "unsupported_audio",
            "The uploaded media format is unsupported.",
        )

    if payload == "slow":
        time.sleep(5)

    return os.getpid(), payload, state


def assert_process_reaped(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def make_supervisor(pid_queue: object) -> PersistentProcessSupervisor:
    return PersistentProcessSupervisor(
        initializer=_initialize_state,
        handler=_handle_request,
        handler_args=(pid_queue,),
        startup_timeout_seconds=10,
        terminate_grace_seconds=0.05,
    )


def test_requests_reuse_one_persistent_child() -> None:
    context = multiprocessing.get_context("spawn")
    pid_queue = context.Queue()
    supervisor = make_supervisor(pid_queue)

    async def run() -> None:
        try:
            first = await supervisor.request("first", timeout_seconds=5)
            second = await supervisor.request("second", timeout_seconds=5)

            assert first[1:] == ("first", "initialized")
            assert second[1:] == ("second", "initialized")
            assert first[0] == second[0]
            assert first[0] != os.getpid()
            assert pid_queue.get(timeout=5) == first[0]
            assert pid_queue.get(timeout=5) == first[0]
            assert supervisor.is_running
        finally:
            await supervisor.aclose()

        assert not supervisor.is_running

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()


def test_initializer_failure_returns_safe_startup_error() -> None:
    supervisor = PersistentProcessSupervisor(
        initializer=_failing_initializer,
        handler=_handle_request,
        startup_timeout_seconds=5,
    )

    async def run() -> None:
        try:
            with pytest.raises(AudioServiceError) as raised:
                await supervisor.request("payload", timeout_seconds=5)

            assert raised.value.status_code == 503
            assert raised.value.code == "inference_worker_startup_failed"
            assert "private initializer failure" not in raised.value.message
        finally:
            await supervisor.aclose()

    asyncio.run(run())


def test_handler_failure_stops_the_child_and_returns_safe_error() -> None:
    context = multiprocessing.get_context("spawn")
    pid_queue = context.Queue()
    supervisor = make_supervisor(pid_queue)

    async def run() -> None:
        try:
            with pytest.raises(AudioServiceError) as raised:
                await supervisor.request("fail", timeout_seconds=5)

            assert raised.value.status_code == 503
            assert raised.value.code == "inference_worker_failed"
            assert "private handler failure" not in raised.value.message
            pid = pid_queue.get(timeout=5)
            assert_process_reaped(pid)
            assert not supervisor.is_running
        finally:
            await supervisor.aclose()

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()


def test_application_error_is_preserved_and_child_stays_running() -> None:
    context = multiprocessing.get_context("spawn")
    pid_queue = context.Queue()
    supervisor = make_supervisor(pid_queue)

    async def run() -> None:
        try:
            with pytest.raises(AudioServiceError) as raised:
                await supervisor.request("invalid", timeout_seconds=5)

            assert raised.value.status_code == 400
            assert raised.value.code == "invalid_audio"
            assert raised.value.message == "The uploaded audio is invalid."
            assert supervisor.is_running

            result = await supervisor.request("valid", timeout_seconds=5)

            assert result[1:] == ("valid", "initialized")
            assert result[0] == pid_queue.get(timeout=5)
            assert result[0] == pid_queue.get(timeout=5)
            assert supervisor.is_running
        finally:
            await supervisor.aclose()

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()


def test_unsupported_audio_error_is_preserved() -> None:
    context = multiprocessing.get_context("spawn")
    pid_queue = context.Queue()
    supervisor = make_supervisor(pid_queue)

    async def run() -> None:
        try:
            with pytest.raises(AudioServiceError) as raised:
                await supervisor.request("unsupported", timeout_seconds=5)

            assert raised.value.status_code == 415
            assert raised.value.code == "unsupported_audio"
            assert raised.value.message == (
                "The uploaded media format is unsupported."
            )
            assert supervisor.is_running
        finally:
            await supervisor.aclose()

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()


def test_timeout_reaps_child_and_next_request_restarts_it() -> None:
    context = multiprocessing.get_context("spawn")
    pid_queue = context.Queue()
    supervisor = make_supervisor(pid_queue)

    async def run() -> None:
        try:
            with pytest.raises(AudioServiceError) as raised:
                await supervisor.request("slow", timeout_seconds=0.3)

            assert raised.value.status_code == 504
            assert raised.value.code == "inference_timeout"

            old_pid = pid_queue.get(timeout=5)
            assert_process_reaped(old_pid)
            assert not supervisor.is_running

            result = await supervisor.request("restarted", timeout_seconds=5)
            new_pid = result[0]

            assert result[1:] == ("restarted", "initialized")
            assert new_pid != old_pid
            assert pid_queue.get(timeout=5) == new_pid
            assert supervisor.is_running
        finally:
            await supervisor.aclose()

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()


def test_cancellation_reaps_persistent_child() -> None:
    context = multiprocessing.get_context("spawn")
    pid_queue = context.Queue()
    supervisor = make_supervisor(pid_queue)

    async def run() -> None:
        try:
            task = asyncio.create_task(
                supervisor.request("slow", timeout_seconds=30)
            )

            pid = await asyncio.to_thread(pid_queue.get, True, 5)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert_process_reaped(pid)
            assert not supervisor.is_running
        finally:
            await supervisor.aclose()

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()



def test_explicit_start_initializes_child_before_first_request() -> None:
    context = multiprocessing.get_context("spawn")
    pid_queue = context.Queue()
    supervisor = make_supervisor(pid_queue)

    async def run() -> None:
        try:
            await supervisor.start()

            assert supervisor.is_running
            startup_pid = supervisor.last_child_pid
            assert startup_pid is not None
            assert startup_pid != os.getpid()

            result = await supervisor.request("after-start", timeout_seconds=5)

            assert result == (startup_pid, "after-start", "initialized")
            assert pid_queue.get(timeout=5) == startup_pid
            assert supervisor.is_running
            assert supervisor.last_child_pid == startup_pid
        finally:
            await supervisor.aclose()

        assert not supervisor.is_running

    try:
        asyncio.run(run())
    finally:
        pid_queue.close()
        pid_queue.join_thread()
