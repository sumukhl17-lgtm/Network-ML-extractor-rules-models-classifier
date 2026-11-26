from __future__ import annotations

import httpx
from pydantic import ValidationError

from .config import Settings
from .schemas import (
    ClassificationResponse,
    DecodeMetadataResponse,
    ErrorResponse,
    ReadinessResponse,
)
from .service import AudioServiceError


_SAFE_WORKER_ERRORS = {
    "empty_file": (400, "The uploaded file is empty."),
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
    "duration_limit_exceeded": (413, "The audio duration exceeds the limit."),
    "sample_rate_limit_exceeded": (415, "The audio sample rate exceeds the limit."),
    "channel_limit_exceeded": (415, "The audio channel count exceeds the limit."),
    "decoded_sample_limit_exceeded": (
        413,
        "The decoded audio exceeds the sample limit.",
    ),
    "audio_too_short": (
        422,
        "Audio must be at least 0.975 seconds long for reliable classification.",
    ),
    "worker_overloaded": (429, "The audio worker is at capacity. Retry later."),
    "worker_unavailable": (503, "The audio worker is unavailable."),
    "worker_failed": (503, "The audio worker failed."),
    "decode_timeout": (504, "Audio decoding exceeded its time limit."),
    "decode_worker_failed": (503, "The audio decode worker failed."),
    "inference_timeout": (504, "Audio inference exceeded its time limit."),
    "inference_worker_failed": (503, "The audio inference worker failed."),
    "model_unavailable": (503, "The inference model is not ready."),
    "authentication_not_configured": (
        503,
        "Internal service authentication is not configured.",
    ),
    "unauthorized": (401, "Authentication is required."),
    "invalid_worker_result": (502, "The audio worker returned an invalid result."),
}
_MAX_WORKER_RESPONSE_BYTES = 16 * 1024


class HttpAudioWorkerClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def _request(
        self,
        *,
        method: str,
        path: str,
        timeout: float,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> tuple[int, bytearray]:
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self._transport,
            ) as client:
                async with client.stream(
                    method,
                    f"{self._settings.worker_url.rstrip('/')}{path}",
                    headers={
                        "X-Audio-Service-Token": self._settings.service_token or ""
                    },
                    files=files,
                ) as response:
                    content = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if len(content) + len(chunk) > _MAX_WORKER_RESPONSE_BYTES:
                            raise AudioServiceError(
                                502,
                                "invalid_worker_response",
                                "The audio worker returned an invalid response.",
                            )
                        content.extend(chunk)
                    return response.status_code, content
        except httpx.TimeoutException:
            raise AudioServiceError(
                504,
                "worker_timeout",
                "The audio worker did not respond within its time limit.",
            ) from None
        except httpx.RequestError:
            raise AudioServiceError(
                503,
                "worker_unavailable",
                "The audio worker is unavailable.",
            ) from None

    @staticmethod
    def _raise_worker_error(status_code: int, content: bytearray) -> None:
        try:
            error = ErrorResponse.model_validate_json(content)
        except (ValidationError, ValueError, TypeError):
            raise AudioServiceError(
                502,
                "invalid_worker_response",
                "The audio worker returned an invalid response.",
            ) from None

        safe_error = _SAFE_WORKER_ERRORS.get(error.error.code)
        if safe_error is None:
            if status_code >= 500:
                raise AudioServiceError(
                    503,
                    "worker_failed",
                    "The audio worker failed.",
                )
            raise AudioServiceError(
                502,
                "invalid_worker_response",
                "The audio worker returned an invalid response.",
            )

        mapped_status, message = safe_error
        raise AudioServiceError(mapped_status, error.error.code, message)

    async def decode(self, audio_bytes: bytes) -> DecodeMetadataResponse:
        status_code, content = await self._request(
            method="POST",
            path="/internal/v1/decode",
            timeout=self._settings.decode_timeout_seconds + 5.0,
            files={
                "file": (
                    "upload.bin",
                    audio_bytes,
                    "application/octet-stream",
                )
            },
        )

        if status_code == 200:
            try:
                return DecodeMetadataResponse.model_validate_json(content)
            except (ValidationError, ValueError, TypeError):
                raise AudioServiceError(
                    502,
                    "invalid_worker_result",
                    "The audio worker returned an invalid result.",
                ) from None

        self._raise_worker_error(status_code, content)
        raise AssertionError("unreachable")

    async def classify(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None,
    ) -> ClassificationResponse:
        status_code, content = await self._request(
            method="POST",
            path="/internal/v1/classify",
            timeout=self._settings.inference_timeout_seconds + 5.0,
            files={
                "file": (
                    filename,
                    audio_bytes,
                    content_type or "application/octet-stream",
                )
            },
        )

        if status_code == 200:
            try:
                return ClassificationResponse.model_validate_json(content)
            except (ValidationError, ValueError, TypeError):
                raise AudioServiceError(
                    502,
                    "invalid_worker_result",
                    "The audio worker returned an invalid result.",
                ) from None

        self._raise_worker_error(status_code, content)
        raise AssertionError("unreachable")

    async def readiness(self) -> ReadinessResponse:
        status_code, content = await self._request(
            method="GET",
            path="/ready",
            timeout=5.0,
        )

        try:
            response = ReadinessResponse.model_validate_json(content)
        except (ValidationError, ValueError, TypeError):
            if status_code >= 500:
                raise AudioServiceError(
                    503,
                    "worker_unavailable",
                    "The audio worker is unavailable.",
                ) from None
            raise AudioServiceError(
                502,
                "invalid_worker_response",
                "The audio worker returned an invalid response.",
            ) from None

        if status_code not in (200, 503):
            raise AudioServiceError(
                502,
                "invalid_worker_response",
                "The audio worker returned an invalid response.",
            )

        return response
