from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from audio_service import calibration
from audio_service.taxonomy import ESC50_CLASSES
from audio_service.yamnet import YAMNET_EMBEDDING_SIZE


def make_calibration_data():
    labels = np.asarray(
        [
            label
            for label in ESC50_CLASSES
            for _ in range(40)
        ],
        dtype=np.str_,
    )
    folds = np.asarray(
        [
            fold
            for _label in ESC50_CLASSES
            for fold in range(1, 6)
            for _ in range(8)
        ],
        dtype=np.int64,
    )
    embeddings = np.zeros(
        (2000, YAMNET_EMBEDDING_SIZE),
        dtype=np.float32,
    )
    return embeddings, labels, folds


def test_probability_metrics_are_finite_and_bounded():
    labels = np.asarray(
        [ESC50_CLASSES[0], ESC50_CLASSES[1]],
        dtype=np.str_,
    )
    probabilities = np.zeros((2, len(ESC50_CLASSES)), dtype=np.float64)
    probabilities[0, 0] = 1.0
    probabilities[1, 1] = 1.0

    metrics = calibration._probability_metrics(labels, probabilities)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] >= 0.0
    assert metrics["log_loss"] >= 0.0
    assert metrics["brier_score"] == 0.0
    assert metrics["top_label_ece"] == 0.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_probability_metrics_penalize_wrong_confident_predictions():
    labels = np.asarray([ESC50_CLASSES[0]], dtype=np.str_)
    probabilities = np.zeros((1, len(ESC50_CLASSES)), dtype=np.float64)
    probabilities[0, 1] = 1.0

    metrics = calibration._probability_metrics(labels, probabilities)

    assert metrics["accuracy"] == 0.0
    assert metrics["brier_score"] == 2.0
    assert metrics["top_label_ece"] == 1.0


def test_calibration_rejects_invalid_embedding_shape():
    with pytest.raises(ValueError, match="invalid shape or value"):
        calibration.evaluate_probability_calibration(
            np.zeros((10, 4), dtype=np.float32),
            np.asarray([], dtype=np.str_),
            np.asarray([], dtype=np.int64),
        )


def test_calibration_rejects_unbalanced_folds():
    embeddings, labels, folds = make_calibration_data()
    folds[0] = 2

    with pytest.raises(ValueError, match="400 recordings"):
        calibration.evaluate_probability_calibration(
            embeddings,
            labels,
            folds,
        )


def test_calibration_uses_nested_holdouts_without_registration(monkeypatch):
    embeddings, labels, folds = make_calibration_data()
    rng = np.random.default_rng(42)
    embeddings[:] = rng.normal(
        loc=0.0,
        scale=1.0,
        size=embeddings.shape,
    ).astype(np.float32)

    class_indices = np.asarray(
        [ESC50_CLASSES.index(label) for label in labels],
        dtype=np.int64,
    )
    embeddings[np.arange(len(labels)), class_indices] += 0.25

    fit_sizes = []

    class RecordingCentroidClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, x, y):
            fit_sizes.append(len(x))
            self.classes_ = np.unique(y)
            self.centroids_ = np.vstack(
                [np.mean(x[y == label], axis=0) for label in self.classes_]
            )
            self.n_features_in_ = x.shape[1]
            return self

        def predict_proba(self, x):
            distances = np.sum(
                (x[:, None, :] - self.centroids_[None, :, :]) ** 2,
                axis=2,
            )
            scores = -distances / (2.0 * x.shape[1])
            scores -= np.max(scores, axis=1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return probabilities

        def predict(self, x):
            probabilities = self.predict_proba(x)
            return self.classes_[np.argmax(probabilities, axis=1)]

    monkeypatch.setattr(
        calibration.training,
        "build_classifier",
        lambda: RecordingCentroidClassifier(),
    )

    result = calibration.evaluate_probability_calibration(
        embeddings,
        labels,
        folds,
    )

    assert set(result["raw"]["folds"]) == {"1", "2", "3", "4", "5"}
    assert set(result["calibrated"]["folds"]) == {"1", "2", "3", "4", "5"}
    assert result["calibration_method"] == "sigmoid"
    assert result["inner_calibration_splits"] == 3
    assert result["deployment_changed"] is False
    assert result["model_registered"] is False

    # Five raw fits and five calibrated full-data refits.
    assert fit_sizes.count(1600) == 10

    # Three inner calibration fits per outer fold.
    assert sum(size in {1066, 1067} for size in fit_sizes) == 15
    assert len(fit_sizes) == 25

    for model_result in (result["raw"], result["calibrated"]):
        for fold_result in model_result["folds"].values():
            assert set(fold_result) == {
                "accuracy",
                "macro_f1",
                "log_loss",
                "brier_score",
                "top_label_ece",
            }
            assert all(
                np.isfinite(value) for value in fold_result.values()
            )
