from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from . import training
from .taxonomy import ESC50_CLASSES
from .yamnet import YAMNET_EMBEDDING_SIZE


RANDOM_STATE = 42
INNER_CALIBRATION_SPLITS = 3
CALIBRATION_METHOD = "sigmoid"


def _validate_dataset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> None:
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.shape != (2000, YAMNET_EMBEDDING_SIZE)
        or embeddings.dtype != np.float32
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Calibration embeddings have an invalid shape or value.")

    if (
        not isinstance(labels, np.ndarray)
        or labels.shape != (2000,)
        or not isinstance(folds, np.ndarray)
        or folds.shape != (2000,)
    ):
        raise ValueError("Calibration labels or folds have an invalid shape.")

    if set(labels.tolist()) != set(ESC50_CLASSES):
        raise ValueError("Calibration labels do not match the ESC-50 taxonomy.")

    if set(folds.tolist()) != {1, 2, 3, 4, 5}:
        raise ValueError("Calibration data must contain all five official folds.")

    for fold in range(1, 6):
        if int(np.count_nonzero(folds == fold)) != 400:
            raise ValueError("Each ESC-50 fold must contain 400 recordings.")

    for label in ESC50_CLASSES:
        for fold in range(1, 6):
            count = int(np.count_nonzero((labels == label) & (folds == fold)))
            if count != 8:
                raise ValueError(
                    "Each ESC-50 class must contain 8 recordings per fold."
                )


def _ordered_probabilities(
    estimator: Any,
    embeddings: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(
        estimator.predict_proba(embeddings),
        dtype=np.float64,
    )
    estimator_classes = np.asarray(estimator.classes_, dtype=np.str_)

    if (
        probabilities.shape != (len(embeddings), len(estimator_classes))
        or set(estimator_classes.tolist()) != set(ESC50_CLASSES)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Classifier returned invalid class probabilities.")

    class_indices = {
        label: index for index, label in enumerate(estimator_classes.tolist())
    }
    ordered = probabilities[
        :, [class_indices[label] for label in ESC50_CLASSES]
    ]

    # Remove harmless floating-point sum drift before log-loss evaluation.
    row_sums = ordered.sum(axis=1, keepdims=True)
    return ordered / row_sums


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    predicted_indices = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    correct = (
        np.asarray(ESC50_CLASSES, dtype=np.str_)[predicted_indices] == labels
    )

    bin_indices = np.minimum((confidences * 10).astype(int), 9)
    error = 0.0

    for bin_index in range(10):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue

        bin_accuracy = float(np.mean(correct[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        error += float(np.mean(mask)) * abs(bin_accuracy - bin_confidence)

    return float(error)


def _probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = np.asarray(ESC50_CLASSES, dtype=np.str_)[
        np.argmax(probabilities, axis=1)
    ]
    one_hot = np.column_stack(
        [labels == label for label in ESC50_CLASSES]
    ).astype(np.float64)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(ESC50_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                labels=list(ESC50_CLASSES),
            )
        ),
        # Multiclass Brier score: mean per-example sum of squared class errors.
        "brier_score": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "top_label_ece": _expected_calibration_error(labels, probabilities),
    }


def _summarize(
    fold_results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    metric_names = tuple(next(iter(fold_results.values())).keys())
    summary: dict[str, dict[str, float]] = {}

    for metric in metric_names:
        values = [result[metric] for result in fold_results.values()]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return summary


def evaluate_probability_calibration(
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> dict[str, Any]:
    """Compare raw and sigmoid-calibrated probabilities on untouched ESC-50 folds.

    Each official fold is used once as the outer holdout. Calibration is learned
    only from out-of-fold predictions inside that outer training partition.
    No model is registered and no production alias is modified.
    """

    _validate_dataset(embeddings, labels, folds)

    raw_results: dict[str, dict[str, float]] = {}
    calibrated_results: dict[str, dict[str, float]] = {}

    for fold in range(1, 6):
        train_mask = folds != fold
        test_mask = folds == fold

        if np.any(train_mask & test_mask):
            raise ValueError("Calibration cross-validation leakage detected.")

        x_train = embeddings[train_mask]
        y_train = labels[train_mask]
        x_test = embeddings[test_mask]
        y_test = labels[test_mask]

        raw_model = training.build_classifier()
        raw_model.fit(x_train, y_train)
        raw_probabilities = _ordered_probabilities(raw_model, x_test)

        inner_cv = StratifiedKFold(
            n_splits=INNER_CALIBRATION_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        calibrated_model = CalibratedClassifierCV(
            estimator=training.build_classifier(),
            method=CALIBRATION_METHOD,
            cv=inner_cv,
            ensemble=False,
        )
        calibrated_model.fit(x_train, y_train)
        calibrated_probabilities = _ordered_probabilities(
            calibrated_model,
            x_test,
        )

        raw_results[str(fold)] = _probability_metrics(
            y_test,
            raw_probabilities,
        )
        calibrated_results[str(fold)] = _probability_metrics(
            y_test,
            calibrated_probabilities,
        )

    return {
        "evaluation": (
            "Nested calibration evaluation using official five-fold "
            "ESC-50 outer holdouts"
        ),
        "calibration_method": CALIBRATION_METHOD,
        "inner_calibration_splits": INNER_CALIBRATION_SPLITS,
        "random_state": RANDOM_STATE,
        "brier_score_definition": (
            "Mean per-example sum of squared errors across all 50 classes"
        ),
        "ece_definition": (
            "Top-label expected calibration error using 10 equal-width "
            "confidence bins"
        ),
        "raw": {
            "folds": raw_results,
            "summary": _summarize(raw_results),
        },
        "calibrated": {
            "folds": calibrated_results,
            "summary": _summarize(calibrated_results),
        },
        "deployment_changed": False,
        "model_registered": False,
    }
