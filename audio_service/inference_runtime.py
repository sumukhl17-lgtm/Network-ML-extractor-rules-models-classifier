from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from .classifier import classify_embedding, validate_classifier
from .config import Settings
from .decoder import decode_audio
from .model_bundle import load_bundle
from .persistent_runtime import PersistentProcessSupervisor
from .preprocessing import preprocess_audio
from .registry import load_champion
from .service import AudioServiceError, ModelUnavailable
from .yamnet import YAMNetEmbedder


@dataclass
class InferenceState:
    """Models and configuration initialized once inside the persistent child."""

    settings: Settings
    embedder: YAMNetEmbedder
    classifier: object
    metadata: dict[str, Any]
    model_version: str
    source: str = "bundle"


def initialize_inference(settings: Settings) -> InferenceState:
    """Load and validate both models once when the child process starts."""

    if not isinstance(settings, Settings):
        raise ModelUnavailable()

    # This must be set before TensorFlow is imported.
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    # Do not pass the service authentication token into the child.
    child_settings = replace(settings, service_token=None)

    if settings.model_source == "bundle":
        classifier, metadata, model_version = load_bundle(
            settings.model_bundle_dir
        )
        source = "bundle"
    elif settings.model_source == "registry":
        classifier, metadata, model_version = load_champion()
        source = "mlflow"
    else:
        raise ModelUnavailable()

    if (
        not isinstance(metadata, dict)
        or not isinstance(model_version, str)
        or not model_version.strip()
    ):
        raise ModelUnavailable()

    validate_classifier(classifier)

    embedder = YAMNetEmbedder()
    embedder.load()

    return InferenceState(
        settings=child_settings,
        embedder=embedder,
        classifier=classifier,
        metadata=metadata,
        model_version=model_version,
        source=source,
    )


def handle_inference(
    payload: object,
    state: InferenceState,
) -> dict[str, Any]:
    """Decode and classify one audio upload inside the persistent child."""

    if not isinstance(state, InferenceState):
        raise ModelUnavailable()

    if not isinstance(payload, dict):
        raise AudioServiceError(
            400,
            "invalid_request",
            "The inference request is invalid.",
        )

    if not {"audio_bytes", "filename"}.issubset(payload):
        raise AudioServiceError(
            400,
            "invalid_request",
            "The inference request is missing required fields.",
        )

    if set(payload) - {"audio_bytes", "filename", "content_type"}:
        raise AudioServiceError(
            400,
            "invalid_request",
            "The inference request contains unsupported fields.",
        )

    audio_bytes = payload["audio_bytes"]
    filename = payload["filename"]
    content_type = payload.get("content_type")

    if (
        not isinstance(audio_bytes, bytes)
        or not audio_bytes
        or len(audio_bytes) > state.settings.max_upload_bytes
    ):
        raise AudioServiceError(
            400 if not isinstance(audio_bytes, bytes) or not audio_bytes else 413,
            "invalid_audio" if not isinstance(audio_bytes, bytes) or not audio_bytes
            else "file_too_large",
            "The uploaded audio file is invalid."
            if not isinstance(audio_bytes, bytes) or not audio_bytes
            else "The uploaded file exceeds the size limit.",
        )

    if (
        not isinstance(filename, str)
        or not filename.strip()
        or len(filename) > 255
    ):
        raise AudioServiceError(
            400,
            "invalid_filename",
            "The uploaded filename is invalid.",
        )

    if (
        content_type is not None
        and (
            not isinstance(content_type, str)
            or len(content_type) > 255
        )
    ):
        raise AudioServiceError(
            400,
            "invalid_content_type",
            "The uploaded content type is invalid.",
        )

    decoded = decode_audio(audio_bytes, state.settings)
    waveform = preprocess_audio(decoded, state.settings)
    yamnet_output = state.embedder.embed(waveform)

    response = classify_embedding(
        model=state.classifier,
        embedding=yamnet_output.embedding,
        yamnet_output=yamnet_output,
        filename=filename,
        duration_seconds=decoded.duration_seconds,
        model_version=state.model_version,
        source=state.source,
    )

    return response.model_dump(mode="json")


def create_inference_supervisor(
    settings: Settings,
    *,
    startup_timeout_seconds: float = 120.0,
    terminate_grace_seconds: float = 0.2,
) -> PersistentProcessSupervisor:
    """Create a supervisor for the persistent audio inference child."""

    if not isinstance(settings, Settings):
        raise TypeError("settings must be a Settings instance")

    child_settings = replace(settings, service_token=None)

    return PersistentProcessSupervisor(
        initializer=initialize_inference,
        handler=handle_inference,
        initializer_args=(child_settings,),
        startup_timeout_seconds=startup_timeout_seconds,
        terminate_grace_seconds=terminate_grace_seconds,
    )
