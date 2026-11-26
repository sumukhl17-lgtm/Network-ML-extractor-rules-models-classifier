from dataclasses import dataclass
from typing import Protocol

from .schemas import ClassificationResponse, DecodeMetadataResponse


class AudioServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class ModelUnavailable(AudioServiceError):
    def __init__(self) -> None:
        super().__init__(503, "model_unavailable", "The inference model is not ready.")


@dataclass(frozen=True)
class ModelStatus:
    loaded: bool
    version: str | None = None


class InferenceService(Protocol):
    def model_status(self) -> ModelStatus: ...

    def classify(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None,
    ) -> ClassificationResponse: ...


class DecodeWorkerClient(Protocol):
    async def decode(self, audio_bytes: bytes) -> DecodeMetadataResponse: ...


class UnavailableInferenceService:
    def model_status(self) -> ModelStatus:
        return ModelStatus(loaded=False)

    def classify(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None,
    ) -> ClassificationResponse:
        raise ModelUnavailable()


class ReadableUpload(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


async def read_limited_upload(upload: ReadableUpload, max_bytes: int) -> bytes:
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise AudioServiceError(413, "file_too_large", "The uploaded file exceeds the size limit.")
    if not data:
        raise AudioServiceError(400, "empty_file", "The uploaded file is empty.")
    return data
