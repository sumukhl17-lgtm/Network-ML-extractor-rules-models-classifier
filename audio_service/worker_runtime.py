from __future__ import annotations

import asyncio
from collections.abc import Callable
import multiprocessing
import os
import sys
from typing import Any

from pydantic import ValidationError

from .config import Settings
from .schemas import DecodeMetadataResponse
from .service import AudioServiceError

ChildTarget = Callable[..., None]
ProcessStarter = Callable[[multiprocessing.Process], None]


def _start_multiprocessing_process(process: multiprocessing.Process) -> None:
    process.start()


def _apply_child_resource_limits(settings: Settings) -> None:
    if not sys.platform.startswith("linux"):
        return

    import resource

    if settings.child_cpu_limit_seconds is not None:
        cpu_limit = settings.child_cpu_limit_seconds
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    if settings.child_address_space_limit_bytes is not None:
        memory_limit = settings.child_address_space_limit_bytes
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


def _send_result(connection: Any, message: dict[str, Any]) -> None:
    try:
        connection.send(message)
    except Exception:
        pass


def _decode_child(
    connection: Any,
    audio_bytes: bytes,
    settings: Settings,
) -> None:
    try:
        _apply_child_resource_limits(settings)
        from .decoder import decode_audio

        decoded = decode_audio(audio_bytes, settings)
        _send_result(
            connection,
            {
                "kind": "ok",
                "result": {
                    "duration_seconds": decoded.duration_seconds,
                    "sample_rate": decoded.sample_rate,
                    "channels": decoded.channels,
                    "codec_name": decoded.codec_name,
                    "format_name": decoded.format_name,
                },
            },
        )
    except AudioServiceError as exc:
        _send_result(
            connection,
            {
                "kind": "audio_error",
                "status_code": exc.status_code,
                "code": exc.code,
            },
        )
    except Exception:
        _send_result(connection, {"kind": "failure"})
    finally:
        connection.close()


class ProcessDecodeSupervisor:
    def __init__(
        self,
        *,
        child_target: ChildTarget = _decode_child,
        child_target_args: tuple[Any, ...] = (),
        terminate_grace_seconds: float = 0.2,
        process_starter: ProcessStarter = _start_multiprocessing_process,
    ) -> None:
        self._child_target = child_target
        self._child_target_args = child_target_args
        self._terminate_grace_seconds = terminate_grace_seconds
        self._process_starter = process_starter
        self._deferred_cleanups: set[asyncio.Task[None]] = set()
        self.last_child_pid: int | None = None

    async def aclose(self) -> None:
        while self._deferred_cleanups:
            await asyncio.gather(
                *tuple(self._deferred_cleanups),
                return_exceptions=True,
            )

    async def decode(
        self, audio_bytes: bytes, settings: Settings
    ) -> DecodeMetadataResponse:
        deadline = asyncio.get_running_loop().time() + settings.decode_timeout_seconds
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        child_settings = Settings(
            max_upload_bytes=settings.max_upload_bytes,
            max_audio_duration_seconds=settings.max_audio_duration_seconds,
            max_audio_sample_rate=settings.max_audio_sample_rate,
            max_audio_channels=settings.max_audio_channels,
            max_decoded_samples=settings.max_decoded_samples,
            child_cpu_limit_seconds=settings.child_cpu_limit_seconds,
            child_address_space_limit_bytes=settings.child_address_space_limit_bytes,
        )
        process = context.Process(
            target=self._child_target,
            args=(sender, audio_bytes, child_settings, *self._child_target_args),
            daemon=False,
        )
        started = False
        deferred_cleanup = False

        try:
            start_task = asyncio.create_task(
                asyncio.to_thread(self._process_starter, process)
            )
            try:
                remaining = max(
                    0.0, deadline - asyncio.get_running_loop().time()
                )
                await asyncio.wait_for(
                    asyncio.shield(start_task),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                deferred_cleanup = True
                cleanup_task = self._schedule_start_cleanup(
                    start_task, process, receiver, sender
                )
                await self._await_cleanup(cleanup_task)
                raise AudioServiceError(
                    504, "decode_timeout", "Audio decoding exceeded its time limit."
                ) from None
            except asyncio.CancelledError:
                deferred_cleanup = True
                cleanup_task = self._schedule_start_cleanup(
                    start_task, process, receiver, sender
                )
                await self._await_cleanup(cleanup_task)
                raise
            except Exception:
                deferred_cleanup = True
                cleanup_task = self._schedule_start_cleanup(
                    start_task, process, receiver, sender
                )
                await self._await_cleanup(cleanup_task)
                raise self._child_failed() from None

            started = True
            self.last_child_pid = process.pid
            sender.close()

            while True:
                if receiver.poll():
                    try:
                        message = receiver.recv()
                    except Exception:
                        raise self._child_failed() from None
                    await self._reap(process)
                    return self._validate_message(message)

                if not process.is_alive():
                    process.join()
                    if receiver.poll():
                        try:
                            message = receiver.recv()
                        except Exception:
                            raise self._child_failed() from None
                        return self._validate_message(message)
                    raise self._child_failed()

                if asyncio.get_running_loop().time() >= deadline:
                    await self._stop_process(process)
                    raise AudioServiceError(
                        504, "decode_timeout", "Audio decoding exceeded its time limit."
                    )

                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            if started:
                await self._stop_process(process)
            raise
        except AudioServiceError:
            raise
        except Exception:
            raise self._child_failed() from None
        finally:
            if started:
                sender.close()
                receiver.close()
                if process.is_alive():
                    await self._stop_process(process)
                process.join()
                process.close()
            elif not deferred_cleanup:
                sender.close()
                receiver.close()
            self.last_child_pid = None

    def _schedule_start_cleanup(
        self,
        start_task: asyncio.Task[None],
        process: multiprocessing.Process,
        receiver: Any,
        sender: Any,
    ) -> asyncio.Task[None]:
        cleanup_task = asyncio.create_task(
            self._cleanup_after_start(start_task, process, receiver, sender)
        )
        self._deferred_cleanups.add(cleanup_task)
        cleanup_task.add_done_callback(self._deferred_cleanups.discard)
        return cleanup_task

    async def _cleanup_after_start(
        self,
        start_task: asyncio.Task[None],
        process: multiprocessing.Process,
        receiver: Any,
        sender: Any,
    ) -> None:
        try:
            try:
                await start_task
            except BaseException:
                pass
            if process.pid is not None:
                if process.is_alive():
                    await self._stop_process(process)
                process.join()
                process.close()
        finally:
            sender.close()
            receiver.close()

    @staticmethod
    async def _await_cleanup(cleanup_task: asyncio.Task[None]) -> None:
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        cleanup_task.result()

    async def _reap(self, process: multiprocessing.Process) -> None:
        process.join(timeout=self._terminate_grace_seconds)
        if process.is_alive():
            await self._stop_process(process)

    async def _stop_process(self, process: multiprocessing.Process) -> None:
        if process.is_alive():
            try:
                process.terminate()
            except OSError:
                pass
            process.join(timeout=self._terminate_grace_seconds)
        if process.is_alive():
            try:
                self._kill(process)
            except OSError:
                pass
            process.join()

    @staticmethod
    def _kill(process: multiprocessing.Process) -> None:
        process.kill()

    @staticmethod
    def _child_failed() -> AudioServiceError:
        return AudioServiceError(
            503, "decode_worker_failed", "The audio decode worker failed."
        )

    def _validate_message(self, message: object) -> DecodeMetadataResponse:
        if not isinstance(message, dict):
            raise self._child_failed()

        if message.get("kind") == "audio_error":
            code = message.get("code")
            errors = {
                "empty_audio": (400, "The uploaded audio file is empty."),
                "file_too_large": (413, "The uploaded file exceeds the size limit."),
                "invalid_audio": (
                    400,
                    "The uploaded content is not a complete, decodable audio file.",
                ),
                "unsupported_audio": (
                    415,
                    "The uploaded media does not contain a supported audio stream.",
                ),
                "duration_limit_exceeded": (
                    413,
                    "The audio duration exceeds the limit.",
                ),
                "sample_rate_limit_exceeded": (
                    415,
                    "The audio sample rate exceeds the limit.",
                ),
                "channel_limit_exceeded": (
                    415,
                    "The audio channel count exceeds the limit.",
                ),
                "decoded_sample_limit_exceeded": (
                    413,
                    "The decoded audio exceeds the sample limit.",
                ),
                "audio_too_short": (
                    422,
                    "Audio must be at least 0.975 seconds long for reliable classification.",
                ),
            }
            if not isinstance(code, str) or code not in errors:
                raise self._child_failed()
            status_code, message_text = errors[code]
            raise AudioServiceError(status_code, code, message_text)

        if message.get("kind") == "failure":
            raise self._child_failed()
        if message.get("kind") != "ok":
            raise self._child_failed()

        try:
            return DecodeMetadataResponse.model_validate(message.get("result"))
        except (ValidationError, TypeError, ValueError):
            raise AudioServiceError(
                502,
                "invalid_worker_result",
                "The audio worker returned an invalid result.",
            ) from None
