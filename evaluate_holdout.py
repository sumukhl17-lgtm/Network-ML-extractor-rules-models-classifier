"""Evaluate saved classifiers on a labeled holdout CSV."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


LABELS = ["Known", "Unknown"]


def _get_model_features(model, model_name):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError(
            "{} does not store feature names; cannot safely select CSV columns".format(
                model_name
            )
        )
    return list(feature_names)


def _load_models(model_dir):
    random_forest = joblib.load(model_dir / "rf.joblib")
    xgboost_model = XGBClassifier()
    xgboost_model.load_model(str(model_dir / "xgb.json"))
    return [
        ("Random Forest", random_forest),
        ("XGBoost", xgboost_model),
    ]


def _evaluate_model(model_name, model, data, label_encoder):
    feature_names = _get_model_features(model, model_name)
    if not feature_names:
        raise ValueError("{} has no stored model features".format(model_name))
    missing_features = [name for name in feature_names if name not in data.columns]
    if missing_features:
        raise ValueError(
            "Input CSV is missing {} feature columns for {}: {}".format(
                model_name,
                ", ".join(missing_features),
            )
        )

    features = data.loc[:, feature_names]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise TypeError("{} input features must all be numeric".format(model_name))

    true_labels = label_encoder.inverse_transform(
        label_encoder.transform(data["label"])
    )
    model_classes = np.asarray(model.classes_)
    if not np.array_equal(model_classes, np.arange(len(LABELS))):
        raise ValueError(
            "{} classes do not match Known=0, Unknown=1: {}".format(
                model_name,
                model_classes.tolist(),
            )
        )
    encoded_predictions = model.predict(features)
    predictions = label_encoder.inverse_transform(
        np.asarray(encoded_predictions, dtype=int)
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        labels=LABELS,
        zero_division=0,
    )

    print("\n{}:".format(model_name))
    print("Confusion matrix (labels={}):".format(LABELS))
    print(confusion_matrix(true_labels, predictions, labels=LABELS))
    print("Per-class metrics:")
    for index, label in enumerate(LABELS):
        print(
            "  {}: precision={:.4f}, recall={:.4f}, F1={:.4f}".format(
                label,
                precision[index],
                recall[index],
                f1[index],
            )
        )
    unknown_recall = recall[LABELS.index("Unknown")]
    print("Unknown recall: {:.4f}".format(unknown_recall))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="labeled holdout CSV to evaluate")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    data = pd.read_csv(csv_path)
    if data.empty:
        parser.error("{} contains no rows to evaluate".format(csv_path))
    required_columns = {"label"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        parser.error(
            "{} is missing required column(s): {}".format(
                csv_path,
                ", ".join(sorted(missing_columns)),
            )
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(LABELS)
    unexpected_labels = set(data["label"].dropna()).difference(LABELS)
    if data["label"].isnull().any() or unexpected_labels:
        parser.error(
            "{} labels must be Known or Unknown; unexpected labels: {}".format(
                csv_path,
                sorted(unexpected_labels),
            )
        )

    model_dir = Path(__file__).resolve().parent
    for model_name, model in _load_models(model_dir):
        _evaluate_model(model_name, model, data, label_encoder)


if __name__ == "__main__":
    main()
