"""Proxy Video classification requests to the isolated Video service."""
from __future__ import annotations

import os
import logging

import httpx
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ..core.errors import AppError, code_for_status, error_body

log = logging.getLogger("app.video")

router = APIRouter(prefix="/classify/video", tags=["video"])

DEFAULT_MAX_BYTES = 200 * 1024 * 1024


def _service_url() -> str:
    return os.getenv("VIDEO_SERVICE_URL", "http://video-api:8003").rstrip("/")


def _max_upload_bytes() -> int:
    return int(os.getenv("VIDEO_MAX_BYTES", str(DEFAULT_MAX_BYTES)))


@router.post("")
async def classify_video_upload(file: UploadFile = File(...)):
    """Forward a bounded upload to the isolated Video classifier."""
    max_bytes = _max_upload_bytes()

    try:
        data = await file.read(max_bytes + 1)

        if not data:
            raise AppError(400, "EMPTY_FILE", "The uploaded video is empty")

        if len(data) > max_bytes:
            raise AppError(
                413,
                "FILE_TOO_LARGE",
                f"Video exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
            )

        timeout = httpx.Timeout(660.0, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{_service_url()}/classify/video",
                files={
                    "file": (
                        file.filename or "upload.mp4",
                        data,
                        file.content_type or "application/octet-stream",
                    )
                },
            )

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code >= 400:
            detail = payload.get("detail") if isinstance(payload, dict) else None

            if isinstance(detail, dict):
                message = str(detail.get("message", "Video classification failed"))
                details = detail.get("details", {})
                code = str(detail.get("code", code_for_status(response.status_code)))
            else:
                message = str(detail or "Video classification failed")
                details = {}
                code = code_for_status(response.status_code)

            return JSONResponse(
                status_code=response.status_code,
                content=error_body(code, message, details),
            )

        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=502,
                content=error_body(
                    "INVALID_VIDEO_RESPONSE",
                    "The Video service returned an invalid response",
                ),
            )

        return JSONResponse(status_code=response.status_code, content=payload)

    except AppError as exc:
        return JSONResponse(status_code=exc.status, content=exc.body())

    except httpx.TimeoutException:
        log.warning("Video service request timed out")
        return JSONResponse(
            status_code=503,
            content=error_body(
                "VIDEO_SERVICE_TIMEOUT",
                "Video classification timed out. Please try a shorter video.",
            ),
        )

    except httpx.RequestError:
        log.exception("Video service is unreachable")
        return JSONResponse(
            status_code=503,
            content=error_body(
                "VIDEO_SERVICE_UNAVAILABLE",
                "The Video classification service is unavailable",
            ),
        )

    finally:
        await file.close()
