from typing import Literal

from pydantic import BaseModel, Field


class TopKPrediction(BaseModel):
    category: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ClassificationResponse(BaseModel):
    filename: str
    category: str = Field(min_length=1)
    sound_group: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    best_guess: str | None
    top_k: list[TopKPrediction]
    reason: str
    source: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)


class DecodeMetadataResponse(BaseModel):
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    codec_name: str | None = Field(default=None, max_length=64)
    format_name: str | None = Field(default=None, max_length=128)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ModelStatusResponse(BaseModel):
    loaded: bool
    version: str | None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model: ModelStatusResponse


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
