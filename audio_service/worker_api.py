from __future__ import annotations

import asyncio
import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .capacity import DecodeCapacity
from .config import Settings, load_settings
from .inference_runtime import create_inference_supervisor
from .schemas import (
    ClassificationResponse,
    DecodeMetadataResponse,
    ErrorResponse,
    HealthResponse,
    ModelStatusResponse,
    ReadinessResponse,
)
from .service import AudioServiceError, read_limited_upload
from .worker_runtime import ProcessDecodeSupervisor


async def _decode_while_connected(
    request: Request,
    supervisor: ProcessDecodeSupervisor,
    payload: bytes,
    settings: Settings,
) -> DecodeMetadataResponse:
    task = asyncio.create_task(supervisor.decode(payload, settings))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.05)
            if not task.done() and await request.is_disconnected():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise asyncio.CancelledError
        return task.result()
    except BaseException:
        if not task.done():
            task.cancel()
        try:
            await asyncio.shield(task)
        except BaseException:
            pass
        raise


async def _inference_while_connected(
    request: Request,
    supervisor: object,
    payload: dict[str, object],
    timeout_seconds: float,
) -> ClassificationResponse:
    task = asyncio.create_task(
        supervisor.request(payload, timeout_seconds=timeout_seconds)
    )
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.05)
            if not task.done() and await request.is_disconnected():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise asyncio.CancelledError

        result = task.result()
        try:
            return ClassificationResponse.model_validate(result)
        except (ValidationError, ValueError, TypeError):
            raise AudioServiceError(
                502,
                "invalid_worker_result",
                "The audio worker returned an invalid result.",
            ) from None
    except BaseException:
        if not task.done():
            task.cancel()
        try:
            await asyncio.shield(task)
        except BaseException:
            pass
        raise


def create_worker_app(
    *,
    settings: Settings | None = None,
    supervisor: ProcessDecodeSupervisor | None = None,
    inference_supervisor: object | None = None,
) -> FastAPI:
    validate_startup_auth = settings is None
    worker_settings = settings if settings is not None else load_settings()
    decode_supervisor = supervisor or ProcessDecodeSupervisor()
    model_supervisor = (
        inference_supervisor
        if inference_supervisor is not None
        else create_inference_supervisor(worker_settings)
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        token = worker_settings.service_token
        try:
            if validate_startup_auth and (not token or not token.strip()):
                raise RuntimeError(
                    "AUDIO_SERVICE_TOKEN must be set to a non-empty value before "
                    "starting the Audio worker."
                )

            await model_supervisor.start()
            yield
        finally:
            close_inference = getattr(model_supervisor, "aclose", None)
            if close_inference is not None:
                await close_inference()

            close_decode = getattr(decode_supervisor, "aclose", None)
            if close_decode is not None:
                await close_decode()

    app = FastAPI(title="Audio Inference Worker", lifespan=lifespan)
    app.state.settings = worker_settings
    app.state.supervisor = decode_supervisor
    app.state.inference_supervisor = model_supervisor
    app.state.decode_capacity = DecodeCapacity(
        worker_settings.max_concurrent_decode_jobs
    )

    @app.exception_handler(AudioServiceError)
    async def handle_audio_service_error(
        _: Request, exc: AudioServiceError
    ) -> JSONResponse:
        body = ErrorResponse(
            error={"code": exc.code, "message": exc.message}
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadinessResponse)
    async def readiness(request: Request) -> JSONResponse | ReadinessResponse:
        inference = request.app.state.inference_supervisor
        loaded = bool(
            inference.is_running
            and isinstance(inference.model_version, str)
            and inference.model_version.strip()
        )
        response = ReadinessResponse(
            status="ready" if loaded else "not_ready",
            model=ModelStatusResponse(
                loaded=loaded,
                version=inference.model_version if loaded else None,
            ),
        )
        if not loaded:
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @app.post(
        "/internal/v1/decode",
        response_model=DecodeMetadataResponse,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def decode(
        request: Request,
        file: UploadFile = File(...),
        x_audio_service_token: str | None = Header(default=None),
    ) -> DecodeMetadataResponse:
        acquired = False
        try:
            configured_token = request.app.state.settings.service_token
            if not configured_token or not configured_token.strip():
                raise AudioServiceError(
                    503,
                    "authentication_not_configured",
                    "Internal service authentication is not configured.",
                )
            if x_audio_service_token is None or not hmac.compare_digest(
                x_audio_service_token.encode("utf-8"),
                configured_token.encode("utf-8"),
            ):
                raise AudioServiceError(
                    401, "unauthorized", "Authentication is required."
                )

            acquired = await request.app.state.decode_capacity.try_acquire()
            if not acquired:
                raise AudioServiceError(
                    429,
                    "worker_overloaded",
                    "The audio worker is at capacity. Retry later.",
                )

            payload = await read_limited_upload(
                file, request.app.state.settings.max_upload_bytes
            )
            return await _decode_while_connected(
                request,
                request.app.state.supervisor,
                payload,
                request.app.state.settings,
            )
        finally:
            try:
                await file.close()
            finally:
                if acquired:
                    await request.app.state.decode_capacity.release()

    @app.post(
        "/internal/v1/classify",
        response_model=ClassificationResponse,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def classify(
        request: Request,
        file: UploadFile = File(...),
        x_audio_service_token: str | None = Header(default=None),
    ) -> ClassificationResponse:
        acquired = False
        try:
            configured_token = request.app.state.settings.service_token
            if not configured_token or not configured_token.strip():
                raise AudioServiceError(
                    503,
                    "authentication_not_configured",
                    "Internal service authentication is not configured.",
                )
            if x_audio_service_token is None or not hmac.compare_digest(
                x_audio_service_token.encode("utf-8"),
                configured_token.encode("utf-8"),
            ):
                raise AudioServiceError(
                    401, "unauthorized", "Authentication is required."
                )

            acquired = await request.app.state.decode_capacity.try_acquire()
            if not acquired:
                raise AudioServiceError(
                    429,
                    "worker_overloaded",
                    "The audio worker is at capacity. Retry later.",
                )

            payload = await read_limited_upload(
                file, request.app.state.settings.max_upload_bytes
            )
            return await _inference_while_connected(
                request,
                request.app.state.inference_supervisor,
                {
                    "audio_bytes": payload,
                    "filename": file.filename or "",
                    "content_type": file.content_type,
                },
                request.app.state.settings.inference_timeout_seconds,
            )
        finally:
            try:
                await file.close()
            finally:
                if acquired:
                    await request.app.state.decode_capacity.release()

    return app


app = create_worker_app()
