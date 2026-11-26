"""Proxy Audio classification requests to the isolated Audio service."""
from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ..core.errors import AppError, error_body

log = logging.getLogger("app.audio")

router = APIRouter(prefix="/classify/audio", tags=["audio"])

DEFAULT_MAX_BYTES = 20 * 1024 * 1024
ALLOWED_UPSTREAM_STATUSES = {400, 413, 415, 422, 429, 502, 503, 504}


def _service_url() -> str:
    return os.getenv("AUDIO_SERVICE_URL", "http://audio-api:8001").rstrip("/")


def _service_token() -> str | None:
    token = os.getenv("AUDIO_SERVICE_TOKEN")
    return token if token and token.strip() else None


def _max_upload_bytes() -> int:
    return int(os.getenv("AUDIO_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_BYTES)))


def _upstream_error(response: httpx.Response) -> JSONResponse:
    """Normalize Audio's error schema into the platform API error format."""
    status = response.status_code

    # A 401 from the internal service means the backend and Audio service
    # have mismatched/missing configuration. Never expose that as a browser
    # authentication challenge.
    if status == 401:
        return JSONResponse(
            status_code=503,
            content=error_body(
                "AUDIO_SERVICE_AUTH_FAILED",
                "Audio service authentication is misconfigured",
            ),
        )

    if status not in ALLOWED_UPSTREAM_STATUSES:
        status = 502

    try:
        payload = response.json()
    except ValueError:
        payload = None

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        upstream_code = error.get("code")
    else:
        message = None
        upstream_code = None

    code_by_status = {
        400: "AUDIO_BAD_REQUEST",
        413: "FILE_TOO_LARGE",
        415: "UNSUPPORTED_FILE_TYPE",
        422: "AUDIO_UNPROCESSABLE",
        429: "AUDIO_WORKER_OVERLOADED",
        502: "AUDIO_WORKER_ERROR",
        503: "AUDIO_SERVICE_UNAVAILABLE",
        504: "AUDIO_INFERENCE_TIMEOUT",
    }

    # Use a stable platform-owned code rather than forwarding arbitrary
    # internal error identifiers.
    code = code_by_status.get(status, "AUDIO_UPSTREAM_ERROR")
    if not isinstance(message, str) or not message.strip():
        message = "Audio classification could not be completed"

    return JSONResponse(
        status_code=status,
        content=error_body(code, message),
    )


@router.post("")
async def classify_audio_upload(file: UploadFile = File(...)):
    """Forward a bounded upload to the authenticated Audio service."""
    try:
        token = _service_token()
        if token is None:
            raise AppError(
                503,
                "AUDIO_SERVICE_NOT_CONFIGURED",
                "Audio classification is not configured on this server",
            )

        max_bytes = _max_upload_bytes()
        if max_bytes <= 0:
            raise AppError(
                503,
                "AUDIO_SERVICE_MISCONFIGURED",
                "Audio upload limits are incorrectly configured",
            )

        data = await file.read(max_bytes + 1)

        if not data:
            raise AppError(400, "EMPTY_FILE", "The uploaded audio file is empty")

        if len(data) > max_bytes:
            raise AppError(
                413,
                "FILE_TOO_LARGE",
                f"Audio exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
            )

        timeout = httpx.Timeout(150.0, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{_service_url()}/internal/v1/classify",
                headers={"X-Audio-Service-Token": token},
                files={
                    "file": (
                        file.filename or "upload",
                        data,
                        file.content_type or "application/octet-stream",
                    )
                },
            )

        if response.status_code >= 400:
            return _upstream_error(response)

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=502,
                content=error_body(
                    "INVALID_AUDIO_RESPONSE",
                    "The Audio service returned an invalid response",
                ),
            )

        return JSONResponse(status_code=response.status_code, content=payload)

    except AppError as exc:
        return JSONResponse(status_code=exc.status, content=exc.body())

    except httpx.TimeoutException:
        log.warning("Audio service request timed out")
        return JSONResponse(
            status_code=503,
            content=error_body(
                "AUDIO_SERVICE_TIMEOUT",
                "Audio classification timed out. Please try again.",
            ),
        )

    except httpx.RequestError:
        log.exception("Audio service is unreachable")
        return JSONResponse(
            status_code=503,
            content=error_body(
                "AUDIO_SERVICE_UNAVAILABLE",
                "The Audio classification service is unavailable",
            ),
        )

    except (ValueError, TypeError):
        log.exception("Audio service configuration is invalid")
        return JSONResponse(
            status_code=503,
            content=error_body(
                "AUDIO_SERVICE_MISCONFIGURED",
                "Audio classification is incorrectly configured",
            ),
        )

    finally:
        await file.close()
