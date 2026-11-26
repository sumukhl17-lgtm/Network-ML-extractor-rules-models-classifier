from __future__ import annotations

import numpy as np

from .schemas import ClassificationResponse
from .service import AudioServiceError, ModelUnavailable
from .taxonomy import ESC50_CLASSES, ESC50_TO_SOUND_GROUP
from .yamnet import YAMNetOutput, YAMNET_EMBEDDING_SIZE


def _invalid_model_output() -> AudioServiceError:
    return AudioServiceError(
        502,
        "invalid_model_output",
        "The audio classifier returned an invalid result.",
    )


def validate_classifier(model: object) -> tuple[str, ...]:
    try:
        classes = tuple(str(label) for label in model.classes_)
        predict_proba = model.predict_proba
    except (AttributeError, TypeError):
        raise ModelUnavailable() from None
    if (
        not callable(predict_proba)
        or len(classes) != len(ESC50_CLASSES)
        or set(classes) != set(ESC50_CLASSES)
    ):
        raise ModelUnavailable()
    return classes


def classify_embedding(
    *,
    model: object,
    embedding: np.ndarray,
    yamnet_output: YAMNetOutput,
    filename: str,
    duration_seconds: float,
    model_version: str,
    source: str = "bundle",
) -> ClassificationResponse:
    classes = validate_classifier(model)
    if (
        embedding.shape != (YAMNET_EMBEDDING_SIZE,)
        or embedding.dtype != np.float32
        or not np.isfinite(embedding).all()
    ):
        raise _invalid_model_output()
    try:
        probabilities = np.asarray(
            model.predict_proba(embedding.reshape(1, -1)),
            dtype=np.float64,
        )
    except (TypeError, ValueError, AttributeError):
        raise _invalid_model_output() from None
    if (
        probabilities.shape != (1, len(classes))
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5)
    ):
        raise _invalid_model_output()

    scores = probabilities[0]
    ranked_indices = np.argsort(-scores, kind="stable")[:3]
    best_label = classes[int(ranked_indices[0])]
    top_k = [
        {
            "category": classes[int(index)],
            "confidence": float(scores[index]),
        }
        for index in ranked_indices
    ]
    return ClassificationResponse(
        filename=filename,
        category=best_label,
        sound_group=ESC50_TO_SOUND_GROUP[best_label],
        confidence=float(scores[ranked_indices[0]]),
        best_guess=best_label,
        top_k=top_k,
        reason=(
            "Fine-grained ESC-50 label from the classifier; confidence scores "
            "are not calibrated."
        ),
        source=source,
        model_version=model_version,
        duration_seconds=duration_seconds,
        sample_rate=16_000,
        channels=1,
    )
