"""Classify network flows using port rules followed by a Random Forest."""

from pathlib import Path

import joblib
import pandas as pd

from rules import classify_by_rule
from train_compare import MODEL_FEATURES


ML_OVERRIDES_UNMATCHED = False

_MODEL = joblib.load(Path(__file__).resolve().with_name("rf.joblib"))
_MODEL_FEATURES = list(getattr(_MODEL, "feature_names_in_", MODEL_FEATURES))
if (
    len(_MODEL_FEATURES) != len(set(_MODEL_FEATURES))
    or any(feature not in MODEL_FEATURES for feature in _MODEL_FEATURES)
    or _MODEL_FEATURES
    != [feature for feature in MODEL_FEATURES if feature in _MODEL_FEATURES]
):
    raise ValueError(
        "rf.joblib feature names must be a unique subset in train_compare.py order"
    )


def _label_for_class(class_value):
    if class_value in (0, "0", "Known"):
        return "Known"
    if class_value in (1, "1", "Unknown"):
        return "Unknown"
    raise ValueError("Random Forest contains unsupported class: {!r}".format(class_value))


def classify_network(flow: dict) -> dict:
    """Classify a flow with rules first, then the saved Random Forest."""
    rule_label = classify_by_rule(flow)
    if rule_label is not None:
        return {"label": rule_label, "source": "rule"}

    row = pd.DataFrame(
        [[flow[feature] for feature in _MODEL_FEATURES]],
        columns=_MODEL_FEATURES,
    )
    predicted_class = _MODEL.predict(row)[0]
    class_labels = [_label_for_class(class_value) for class_value in _MODEL.classes_]
    predicted_label = _label_for_class(predicted_class)
    if predicted_label not in class_labels:
        raise ValueError(
            "Random Forest predicted a class not present in its class list: {!r}".format(
                predicted_class
            )
        )

    probabilities = _MODEL.predict_proba(row)[0]
    confidence = float(probabilities[class_labels.index(predicted_label)])
    if not ML_OVERRIDES_UNMATCHED:
        return {
            "label": "Unknown",
            "source": "default",
            "ml_label": predicted_label,
            "ml_confidence": confidence,
        }
    return {
        "label": predicted_label,
        "source": "ml",
        "confidence": confidence,
    }
