import asyncio
import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse

from .capacity import DecodeCapacity
from .config import Settings, load_settings
from .schemas import (
    ClassificationResponse,
    ErrorResponse,
    HealthResponse,
    ModelStatusResponse,
    ReadinessResponse,
)
from .service import AudioServiceError, read_limited_upload
from .worker_client import HttpAudioWorkerClient


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _classify_until_disconnected(
    request: Request,
    worker_client: object,
    payload: bytes,
    filename: str,
    content_type: str | None,
) -> ClassificationResponse:
    worker_task = asyncio.create_task(
        worker_client.classify(payload, filename, content_type)
    )
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))

    try:
        done, _ = await asyncio.wait(
            {worker_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if disconnect_task in done:
            disconnect_task.result()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            raise asyncio.CancelledError

        result = worker_task.result()
        return ClassificationResponse.model_validate(result)
    except BaseException:
        if not worker_task.done():
            worker_task.cancel()
        try:
            await worker_task
        except BaseException:
            pass
        raise
    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
        await asyncio.gather(disconnect_task, return_exceptions=True)


def create_app(
    *,
    settings: Settings | None = None,
    worker_client: object | None = None,
) -> FastAPI:
    validate_startup_auth = settings is None
    service_settings = settings if settings is not None else load_settings()
    audio_worker = (
        worker_client
        if worker_client is not None
        else HttpAudioWorkerClient(service_settings)
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        token = service_settings.service_token
        if validate_startup_auth and (not token or not token.strip()):
            raise RuntimeError(
                "AUDIO_SERVICE_TOKEN must be set to a non-empty value before "
                "starting the Audio service."
            )
        yield

    app = FastAPI(title="Audio Inference Service", lifespan=lifespan)
    app.state.settings = service_settings
    app.state.worker_client = audio_worker
    app.state.decode_capacity = DecodeCapacity(
        service_settings.max_concurrent_decode_jobs
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
        try:
            response = await request.app.state.worker_client.readiness()
        except AudioServiceError:
            response = ReadinessResponse(
                status="not_ready",
                model=ModelStatusResponse(loaded=False, version=None),
            )

        if response.status != "ready" or not response.model.loaded:
            return JSONResponse(
                status_code=503,
                content=response.model_dump(),
            )
        return response

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

            return await _classify_until_disconnected(
                request,
                request.app.state.worker_client,
                payload,
                file.filename or "",
                file.content_type,
            )
        finally:
            try:
                await file.close()
            finally:
                if acquired:
                    await request.app.state.decode_capacity.release()

    return app


app = create_app()
