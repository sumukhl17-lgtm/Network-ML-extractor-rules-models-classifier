from __future__ import annotations

import numpy as np
import pytest

from audio_service.preprocessing import MIN_MODEL_INPUT_SAMPLES
from audio_service.service import AudioServiceError, ModelUnavailable
from audio_service.yamnet import (
    YAMNET_CLASS_COUNT,
    YAMNET_EMBEDDING_SIZE,
    YAMNetEmbedder,
)


class FakeModel:
    def __init__(self, scores: np.ndarray, embeddings: np.ndarray) -> None:
        self.scores = scores
        self.embeddings = embeddings
        self.calls = 0

    def __call__(self, _: np.ndarray) -> tuple[np.ndarray, np.ndarray, None]:
        self.calls += 1
        return self.scores, self.embeddings, None


def make_embedder(
    model: FakeModel | None = None,
    *,
    scores: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
) -> tuple[YAMNetEmbedder, FakeModel, list[int]]:
    frame_scores = (
        np.full((2, YAMNET_CLASS_COUNT), 0.25, dtype=np.float32)
        if scores is None
        else scores
    )
    frame_embeddings = (
        np.arange(2 * YAMNET_EMBEDDING_SIZE, dtype=np.float32).reshape(
            2,
            YAMNET_EMBEDDING_SIZE,
        )
        if embeddings is None
        else embeddings
    )
    model = model or FakeModel(frame_scores, frame_embeddings)
    loader_calls: list[int] = []

    def loader() -> tuple[FakeModel, tuple[str, ...], str]:
        loader_calls.append(1)
        return (
            model,
            tuple(f"AudioSet class {index}" for index in range(YAMNET_CLASS_COUNT)),
            "a" * 64,
        )

    return YAMNetEmbedder(loader), model, loader_calls


def test_mean_pools_frame_scores_and_embeddings_and_caches_model() -> None:
    scores = np.zeros((2, YAMNET_CLASS_COUNT), dtype=np.float32)
    scores[0, 0] = 0.2
    scores[1, 0] = 0.8
    embeddings = np.vstack(
        (
            np.zeros(YAMNET_EMBEDDING_SIZE, dtype=np.float32),
            np.full(YAMNET_EMBEDDING_SIZE, 2.0, dtype=np.float32),
        )
    )
    embedder, model, loader_calls = make_embedder(
        scores=scores,
        embeddings=embeddings,
    )
    waveform = np.zeros(MIN_MODEL_INPUT_SAMPLES, dtype=np.float32)

    first = embedder.embed(waveform)
    second = embedder.embed(waveform)

    assert first.class_scores.shape == (YAMNET_CLASS_COUNT,)
    assert first.class_scores[0] == pytest.approx(0.5)
    assert first.embedding.shape == (YAMNET_EMBEDDING_SIZE,)
    assert np.all(first.embedding == 1.0)
    assert first.class_names[0] == "AudioSet class 0"
    assert first.model_version == "google/yamnet/1"
    assert first.model_sha256 == "a" * 64
    assert np.array_equal(first.embedding, second.embedding)
    assert loader_calls == [1]
    assert model.calls == 2


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros((1, MIN_MODEL_INPUT_SAMPLES), dtype=np.float32),
        np.zeros(MIN_MODEL_INPUT_SAMPLES - 1, dtype=np.float32),
        np.zeros(MIN_MODEL_INPUT_SAMPLES, dtype=np.float64),
        np.full(MIN_MODEL_INPUT_SAMPLES, np.nan, dtype=np.float32),
        np.full(MIN_MODEL_INPUT_SAMPLES, np.inf, dtype=np.float32),
        np.full(MIN_MODEL_INPUT_SAMPLES, 1.1, dtype=np.float32),
    ],
)
def test_rejects_invalid_waveforms(waveform: np.ndarray) -> None:
    embedder, _, _ = make_embedder()

    with pytest.raises(AudioServiceError) as raised:
        embedder.embed(waveform)

    assert raised.value.code == "invalid_audio"


@pytest.mark.parametrize(
    ("scores", "embeddings"),
    [
        (
            np.zeros((2, YAMNET_CLASS_COUNT - 1), dtype=np.float32),
            np.zeros((2, YAMNET_EMBEDDING_SIZE), dtype=np.float32),
        ),
        (
            np.zeros((2, YAMNET_CLASS_COUNT), dtype=np.float32),
            np.zeros((1, YAMNET_EMBEDDING_SIZE), dtype=np.float32),
        ),
        (
            np.full((2, YAMNET_CLASS_COUNT), np.nan, dtype=np.float32),
            np.zeros((2, YAMNET_EMBEDDING_SIZE), dtype=np.float32),
        ),
        (
            np.zeros((2, YAMNET_CLASS_COUNT), dtype=np.float32),
            np.full((2, YAMNET_EMBEDDING_SIZE), np.inf, dtype=np.float32),
        ),
    ],
)
def test_rejects_malformed_or_non_finite_model_output(
    scores: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    embedder, _, _ = make_embedder(scores=scores, embeddings=embeddings)

    with pytest.raises(AudioServiceError) as raised:
        embedder.embed(np.zeros(MIN_MODEL_INPUT_SAMPLES, dtype=np.float32))

    assert raised.value.code == "invalid_model_output"


def test_reports_model_unavailable_when_loader_fails() -> None:
    def loader():
        raise ModelUnavailable()

    embedder = YAMNetEmbedder(loader)

    with pytest.raises(ModelUnavailable):
        embedder.embed(np.zeros(MIN_MODEL_INPUT_SAMPLES, dtype=np.float32))
