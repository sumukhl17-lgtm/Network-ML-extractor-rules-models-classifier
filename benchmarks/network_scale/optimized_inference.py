"""Isolated batch implementation matching classify_network.classify_network."""

from typing import Any, Dict, List, Sequence

import pandas as pd

from rules import classify_by_rule


def _label_for_class(class_value: Any) -> str:
    if class_value in (0, "0", "Known"):
        return "Known"
    if class_value in (1, "1", "Unknown"):
        return "Unknown"
    raise ValueError("Random Forest contains unsupported class: {!r}".format(class_value))


def classify_batch(
    flows: Sequence[Dict[str, Any]],
    model: Any,
    model_features: Sequence[str],
    ml_overrides_unmatched: bool = False,
) -> List[Dict[str, Any]]:
    """Classify a bounded batch, preserving rule precedence and model semantics."""
    results: List[Any] = [None] * len(flows)
    unmatched_indexes = []
    for index, flow in enumerate(flows):
        rule_label = classify_by_rule(flow)
        if rule_label is not None:
            results[index] = {"label": rule_label, "source": "rule"}
        else:
            unmatched_indexes.append(index)
    if not unmatched_indexes:
        return results

    rows = [
        [flows[index][feature] for feature in model_features]
        for index in unmatched_indexes
    ]
    features = pd.DataFrame(rows, columns=list(model_features))
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    class_labels = [_label_for_class(value) for value in model.classes_]

    for offset, index in enumerate(unmatched_indexes):
        predicted_label = _label_for_class(predictions[offset])
        if predicted_label not in class_labels:
            raise ValueError(
                "Random Forest predicted a class not present in its class list: {!r}".format(
                    predictions[offset]
                )
            )
        confidence = float(probabilities[offset][class_labels.index(predicted_label)])
        if ml_overrides_unmatched:
            results[index] = {
                "label": predicted_label,
                "source": "ml",
                "confidence": confidence,
            }
        else:
            results[index] = {
                "label": "Unknown",
                "source": "default",
                "ml_label": predicted_label,
                "ml_confidence": confidence,
            }
    return results
