from __future__ import annotations

import asyncio
from collections.abc import Callable
import multiprocessing
from multiprocessing.connection import Connection
from typing import Any

from .service import AudioServiceError


Initializer = Callable[..., Any]
Handler = Callable[..., Any]
ProcessStarter = Callable[[multiprocessing.Process], None]


class _RemoteAudioServiceError(AudioServiceError):
    """A safe application error returned by the persistent child."""


def _start_process(process: multiprocessing.Process) -> None:
    process.start()


def _send(connection: Connection, message: dict[str, Any]) -> None:
    try:
        connection.send(message)
    except Exception:
        pass


def _persistent_child(
    connection: Connection,
    initializer: Initializer,
    handler: Handler,
    initializer_args: tuple[Any, ...],
    handler_args: tuple[Any, ...],
) -> None:
    try:
        state = initializer(*initializer_args)

        ready_message: dict[str, Any] = {"kind": "ready"}
        model_version = getattr(state, "model_version", None)

        if isinstance(model_version, str) and model_version.strip():
            ready_message["model_version"] = model_version

        connection.send(ready_message)

        while True:
            message = connection.recv()

            if not isinstance(message, dict):
                break

            if message.get("kind") == "shutdown":
                break

            if message.get("kind") != "request":
                break

            request_id = message.get("request_id")

            try:
                result = handler(
                    message.get("payload"),
                    state,
                    *handler_args,
                )

                connection.send({
                    "kind": "ok",
                    "request_id": request_id,
                    "result": result,
                })

            except AudioServiceError as exc:
                _send(
                    connection,
                    {
                        "kind": "application_error",
                        "request_id": request_id,
                        "status_code": exc.status_code,
                        "code": exc.code,
                        "message": exc.message,
                    },
                )

            except Exception:
                _send(
                    connection,
                    {
                        "kind": "failure",
                        "request_id": request_id,
                    },
                )

    except Exception:
        _send(connection, {"kind": "startup_failed"})

    finally:
        connection.close()


class PersistentProcessSupervisor:
    """Own one persistent child and serialize requests sent to it."""

    def __init__(
        self,
        *,
        initializer: Initializer,
        handler: Handler,
        initializer_args: tuple[Any, ...] = (),
        handler_args: tuple[Any, ...] = (),
        startup_timeout_seconds: float = 120.0,
        terminate_grace_seconds: float = 0.2,
        process_starter: ProcessStarter = _start_process,
    ) -> None:
        if startup_timeout_seconds <= 0:
            raise ValueError("startup timeout must be positive")

        if terminate_grace_seconds < 0:
            raise ValueError("terminate grace period cannot be negative")

        self._initializer = initializer
        self._handler = handler
        self._initializer_args = initializer_args
        self._handler_args = handler_args
        self._startup_timeout = startup_timeout_seconds
        self._terminate_grace = terminate_grace_seconds
        self._process_starter = process_starter

        self._lock = asyncio.Lock()
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._request_id = 0
        self._deferred_cleanups: set[asyncio.Task[None]] = set()

        self.last_child_pid: int | None = None
        self.model_version: str | None = None

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.is_alive()
            and self._connection is not None
        )

    async def start(self) -> None:
        """Start the persistent child and wait until its initializer is ready."""
        async with self._lock:
            await self._wait_for_deferred_cleanups()
            await self._ensure_started()

    async def request(self, payload: Any, *, timeout_seconds: float) -> Any:
        if timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")

        async with self._lock:
            await self._wait_for_deferred_cleanups()
            await self._ensure_started()

            process = self._process
            connection = self._connection

            if process is None or connection is None:
                raise self._worker_failed()

            self._request_id += 1
            request_id = self._request_id

            deadline = (
                asyncio.get_running_loop().time() + timeout_seconds
            )

            try:
                remaining = (
                    deadline - asyncio.get_running_loop().time()
                )

                if remaining <= 0:
                    raise asyncio.TimeoutError

                await self._send_until(
                    connection,
                    {
                        "kind": "request",
                        "request_id": request_id,
                        "payload": payload,
                    },
                    deadline,
                    process,
                )

                message = await self._receive_until(
                    deadline,
                    process,
                    connection,
                )

                if (
                    not isinstance(message, dict)
                    or message.get("request_id") != request_id
                ):
                    raise self._worker_failed()

                if message.get("kind") == "ok":
                    return message.get("result")

                if message.get("kind") == "application_error":
                    status_code = message.get("status_code")
                    code = message.get("code")
                    error_message = message.get("message")

                    if (
                        not isinstance(status_code, int)
                        or isinstance(status_code, bool)
                        or not 400 <= status_code <= 599
                        or not isinstance(code, str)
                        or not code
                        or len(code) > 128
                        or not isinstance(error_message, str)
                        or not error_message
                        or len(error_message) > 512
                    ):
                        raise self._worker_failed()

                    raise _RemoteAudioServiceError(
                        status_code,
                        code,
                        error_message,
                    )

                raise self._worker_failed()

            except asyncio.TimeoutError:
                await self._stop_locked()

                raise AudioServiceError(
                    504,
                    "inference_timeout",
                    "Audio inference exceeded its time limit.",
                ) from None

            except asyncio.CancelledError:
                await self._stop_locked()
                raise

            except _RemoteAudioServiceError:
                raise

            except AudioServiceError:
                await self._stop_locked()
                raise

            except Exception:
                await self._stop_locked()
                raise self._worker_failed() from None

    async def _ensure_started(self) -> None:
        if self.is_running:
            return

        await self._stop_locked()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout

        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(
            duplex=True
        )

        process = context.Process(
            target=_persistent_child,
            args=(
                child_connection,
                self._initializer,
                self._handler,
                self._initializer_args,
                self._handler_args,
            ),
            daemon=False,
        )

        start_task = asyncio.create_task(
            asyncio.to_thread(
                self._process_starter,
                process,
            )
        )

        try:
            await self._wait_for_start(start_task, deadline)

        except asyncio.TimeoutError:
            self._schedule_start_cleanup(
                start_task,
                process,
                parent_connection,
                child_connection,
            )
            raise self._startup_failed() from None

        except asyncio.CancelledError:
            self._schedule_start_cleanup(
                start_task,
                process,
                parent_connection,
                child_connection,
            )
            raise

        except Exception:
            self._schedule_start_cleanup(
                start_task,
                process,
                parent_connection,
                child_connection,
            )
            raise self._startup_failed() from None

        child_connection.close()

        self._process = process
        self._connection = parent_connection
        self.last_child_pid = process.pid

        try:
            message = await self._receive_until(
                deadline,
                process,
                parent_connection,
            )

            if (
                not isinstance(message, dict)
                or message.get("kind") != "ready"
            ):
                raise self._startup_failed()

            model_version = message.get("model_version")

            if model_version is not None and (
                not isinstance(model_version, str)
                or not model_version.strip()
            ):
                raise self._startup_failed()

            self.model_version = model_version

        except asyncio.TimeoutError:
            await self._stop_locked()
            raise self._startup_failed() from None

        except BaseException:
            await self._stop_locked()
            raise

    @staticmethod
    async def _wait_for_start(
        start_task: asyncio.Task[None],
        deadline: float,
    ) -> None:
        remaining = (
            deadline - asyncio.get_running_loop().time()
        )

        if remaining <= 0:
            raise asyncio.TimeoutError

        await asyncio.wait_for(
            asyncio.shield(start_task),
            timeout=remaining,
        )

    async def _send_until(
        self,
        connection: Connection,
        message: dict[str, Any],
        deadline: float,
        process: multiprocessing.Process,
    ) -> None:
        remaining = (
            deadline - asyncio.get_running_loop().time()
        )

        if remaining <= 0:
            raise asyncio.TimeoutError

        send_task = asyncio.create_task(
            asyncio.to_thread(
                connection.send,
                message,
            )
        )

        try:
            await asyncio.wait_for(
                asyncio.shield(send_task),
                timeout=remaining,
            )

        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
        ):
            try:
                if process.is_alive():
                    process.kill()

            except (OSError, ValueError):
                try:
                    if process.is_alive():
                        process.terminate()
                except (OSError, ValueError):
                    pass

            await asyncio.to_thread(process.join)

            while not send_task.done():
                try:
                    await asyncio.shield(send_task)

                except asyncio.CancelledError:
                    continue

                except BaseException:
                    break

            if send_task.done():
                try:
                    send_task.result()
                except BaseException:
                    pass

            raise

    async def _receive_until(
        self,
        deadline: float,
        process: multiprocessing.Process,
        connection: Connection,
    ) -> Any:
        loop = asyncio.get_running_loop()

        while loop.time() < deadline:
            if connection.poll():
                return await asyncio.to_thread(
                    connection.recv
                )

            if not process.is_alive():
                if connection.poll():
                    return await asyncio.to_thread(
                        connection.recv
                    )

                raise self._worker_failed()

            await asyncio.sleep(0.01)

        raise asyncio.TimeoutError

    def _schedule_start_cleanup(
        self,
        start_task: asyncio.Task[None],
        process: multiprocessing.Process,
        parent_connection: Connection,
        child_connection: Connection,
    ) -> None:
        task = asyncio.create_task(
            self._cleanup_after_start(
                start_task,
                process,
                parent_connection,
                child_connection,
            )
        )

        self._deferred_cleanups.add(task)
        task.add_done_callback(
            self._deferred_cleanups.discard
        )

    async def _cleanup_after_start(
        self,
        start_task: asyncio.Task[None],
        process: multiprocessing.Process,
        parent_connection: Connection,
        child_connection: Connection,
    ) -> None:
        try:
            try:
                await start_task
            except BaseException:
                pass

            if process.pid is not None:
                await self._terminate_process(process)
            else:
                process.close()

        finally:
            parent_connection.close()
            child_connection.close()

    async def _wait_for_deferred_cleanups(self) -> None:
        while self._deferred_cleanups:
            await asyncio.gather(
                *tuple(self._deferred_cleanups),
                return_exceptions=True,
            )

    async def aclose(self) -> None:
        async with self._lock:
            await self._stop_locked()
            await self._wait_for_deferred_cleanups()

    async def _stop_locked(self) -> None:
        process = self._process
        connection = self._connection

        self._process = None
        self._connection = None
        self.last_child_pid = None
        self.model_version = None

        if connection is not None:
            if process is not None and process.is_alive():
                try:
                    connection.send({"kind": "shutdown"})
                except Exception:
                    pass

            connection.close()

        if process is not None and process.pid is not None:
            await self._terminate_process(process)

    async def _terminate_process(
        self,
        process: multiprocessing.Process,
    ) -> None:
        try:
            await asyncio.to_thread(
                process.join,
                self._terminate_grace,
            )

            if process.is_alive():
                process.terminate()

                await asyncio.to_thread(
                    process.join,
                    self._terminate_grace,
                )

            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join)

        finally:
            if not process.is_alive():
                process.close()

    @staticmethod
    def _startup_failed() -> AudioServiceError:
        return AudioServiceError(
            503,
            "inference_worker_startup_failed",
            "The audio inference worker could not start.",
        )

    @staticmethod
    def _worker_failed() -> AudioServiceError:
        return AudioServiceError(
            503,
            "inference_worker_failed",
            "The audio inference worker failed.",
        )
