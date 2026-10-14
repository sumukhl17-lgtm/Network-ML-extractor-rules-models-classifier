"""Compare Random Forest and XGBoost classifiers for flow labels."""

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from rules import classify_by_rule


MODEL_FEATURES = [
    "protocol",
    "packet_count",
    "total_bytes",
    "avg_packet_size",
    "duration",
    "packet_rate",
    "byte_rate",
    "syn",
    "fin",
    "rst",
    "psh",
    "ack",
]
MLFLOW_EXPERIMENT_NAME = "network-known-unknown"
FORBIDDEN_MODEL_FEATURES = {
    "src_port",
    "dst_port",
    "application_name",
    "application_is_guessed",
    "source_file",
}

assert len(MODEL_FEATURES) == 12
assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))
assert set(MODEL_FEATURES).isdisjoint(FORBIDDEN_MODEL_FEATURES)


def _get_features(data, name, feature_names):
    missing = [column for column in feature_names if column not in data.columns]
    if missing:
        raise ValueError("{} is missing model features: {}".format(name, missing))

    features = data.loc[:, feature_names]
    assert features.columns.tolist() == feature_names
    assert set(features.columns).isdisjoint(FORBIDDEN_MODEL_FEATURES)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise TypeError("{} contains non-numeric model features".format(name))
    return features


def _make_models(scale_pos_weight):
    return {
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        ),
    }


def _scale_pos_weight(labels):
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("Both target classes must be present in model training data")
    return float(class_counts[0]) / float(class_counts[1])


def _print_metrics(name, y_true, predictions, probabilities, class_names):
    class_indexes = np.arange(len(class_names))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=class_indexes,
        zero_division=0,
    )

    print("\n{} metrics:".format(name))
    print("Per-class metrics:")
    for index, class_name in enumerate(class_names):
        print(
            "  {}: precision={:.4f}, recall={:.4f}, F1={:.4f}".format(
                class_name,
                precision[index],
                recall[index],
                f1[index],
            )
        )

    print("Confusion matrix (labels={}):".format(list(class_names)))
    print(confusion_matrix(y_true, predictions, labels=class_indexes))

    metrics = {}
    for index, class_name in enumerate(class_names):
        metrics["precision_{}".format(class_name)] = float(precision[index])
        metrics["recall_{}".format(class_name)] = float(recall[index])
        metrics["f1_{}".format(class_name)] = float(f1[index])

    if len(class_names) == 2 and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, probabilities[:, 1])
        print("ROC-AUC: {:.4f}".format(auc))
        metrics["roc_auc"] = float(auc)
    else:
        print("ROC-AUC: unavailable (evaluation data does not contain both classes)")
    return metrics


def _print_top_importances(model, feature_names):
    top_indexes = np.argsort(model.feature_importances_)[::-1][:10]
    print("Top 10 feature importances:")
    for index in top_indexes:
        print(
            "  {}: {:.6f}".format(
                feature_names[index],
                model.feature_importances_[index],
            )
        )


def _cross_validate(name, data, feature_names, label_encoder):
    captures = ("flows2.csv", "flows3.csv", "flows4.csv")
    available_captures = set(data["source_file"])
    missing_captures = set(captures).difference(available_captures)
    if missing_captures:
        raise ValueError(
            "Leave-one-capture-out data is missing captures: {}".format(
                sorted(missing_captures)
            )
        )

    fold_f1 = []
    for capture in captures:
        validation_mask = data["source_file"] == capture
        training_mask = ~validation_mask
        train_data = data.loc[training_mask]
        validation_data = data.loc[validation_mask]
        if "flows1.csv" not in set(train_data["source_file"]):
            raise ValueError("flows1.csv must remain in training data for every fold")

        train_features = _get_features(train_data, "training fold", feature_names)
        validation_features = _get_features(
            validation_data,
            "validation fold {}".format(capture),
            feature_names,
        )
        fold_labels = label_encoder.transform(train_data["label"])
        model = _make_models(_scale_pos_weight(fold_labels))[name]
        model.fit(train_features, fold_labels)
        predictions = label_encoder.inverse_transform(
            model.predict(validation_features).astype(int)
        )
        score = f1_score(
            validation_data["label"],
            predictions,
            labels=["Known", "Unknown"],
            average="macro",
            zero_division=0,
        )
        fold_f1.append(score)
        print("{} leave-one-capture-out {} F1: {:.4f}".format(name, capture, score))

    print(
        "{} leave-one-capture-out macro F1 ({} folds): mean={:.4f}, std={:.4f}".format(
            name,
            len(captures),
            float(np.mean(fold_f1)),
            float(np.std(fold_f1)),
        )
    )
    return float(np.mean(fold_f1))


def _model_parameters(model_name, model):
    if model_name == "Random Forest":
        names = (
            "class_weight",
            "random_state",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
        )
    else:
        names = (
            "scale_pos_weight",
            "random_state",
            "eval_metric",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
        )
    parameters = model.get_params()
    return {
        name: json.dumps(parameters[name])
        if isinstance(parameters[name], (list, tuple, dict))
        else str(parameters[name])
        for name in names
    }


def _log_model(model_name, model):
    if model_name == "Random Forest":
        mlflow.sklearn.log_model(model, artifact_path="model")
    else:
        mlflow.xgboost.log_model(model, artifact_path="model")


def _rule_unclassified_mask(test_data):
    def is_unclassified(row):
        flow = {
            "src_port": row["src_port"],
            "dst_port": row["dst_port"],
        }
        return classify_by_rule(flow) is None

    return test_data.apply(is_unclassified, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-features",
        default="",
        help="comma-separated model features to exclude",
    )
    args = parser.parse_args()
    excluded_features = {
        feature.strip()
        for feature in args.exclude_features.split(",")
        if feature.strip()
    }
    unknown_features = excluded_features.difference(MODEL_FEATURES)
    if unknown_features:
        parser.error(
            "unknown feature(s) to exclude: {}".format(
                ", ".join(sorted(unknown_features))
            )
        )

    feature_names = [
        feature for feature in MODEL_FEATURES if feature not in excluded_features
    ]
    if not feature_names:
        parser.error("--exclude-features cannot remove every model feature")

    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    required_metadata = {"label", "source_file"}
    for name, data in (("train.csv", train_data), ("test.csv", test_data)):
        missing = required_metadata.difference(data.columns)
        if missing:
            raise ValueError("{} is missing required columns: {}".format(name, missing))

    print("Model features used: {}".format(", ".join(feature_names)))
    train_features = _get_features(train_data, "train.csv", feature_names)
    test_features = _get_features(test_data, "test.csv", feature_names)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_data["label"])
    if len(label_encoder.classes_) != 2:
        raise ValueError("XGBoost class weighting requires exactly two target classes")
    test_labels = label_encoder.transform(test_data["label"])
    class_names = label_encoder.classes_

    cross_validation_data = pd.concat(
        [train_data, test_data],
        ignore_index=True,
    )
    if set(label_encoder.classes_) != {"Known", "Unknown"}:
        raise ValueError("Leave-one-capture-out F1 requires Known and Unknown labels")

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI") or str(Path("./mlruns").resolve())
    )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    cross_validation_scores = {}
    for model_name in ("Random Forest", "XGBoost"):
        cross_validation_scores[model_name] = _cross_validate(
            model_name,
            cross_validation_data,
            feature_names,
            label_encoder,
        )

    models = _make_models(_scale_pos_weight(train_labels))
    for model_name, model in models.items():
        run_name = "{}-{}".format(model_name, ",".join(feature_names))
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("data_label_source", "nDPI")
            mlflow.log_param("features", json.dumps(feature_names))
            mlflow.log_param("train_file", "train.csv")
            mlflow.log_param("test_file", "test.csv")
            mlflow.log_param(
                "train_source_files",
                json.dumps(sorted(train_data["source_file"].unique().tolist())),
            )
            mlflow.log_param(
                "test_source_files",
                json.dumps(sorted(test_data["source_file"].unique().tolist())),
            )
            mlflow.log_params(_model_parameters(model_name, model))
            mlflow.log_metric(
                "leave_one_capture_out_mean_f1",
                cross_validation_scores[model_name],
            )

            model.fit(train_features, train_labels)
            test_predictions = model.predict(test_features)
            test_probabilities = model.predict_proba(test_features)
            test_metrics = _print_metrics(
                "{} full test set".format(model_name),
                test_labels,
                test_predictions,
                test_probabilities,
                class_names,
            )
            mlflow.log_metrics(test_metrics)
            _print_top_importances(model, feature_names)

            unclassified_mask = _rule_unclassified_mask(test_data)
            filtered_labels = test_labels[unclassified_mask.to_numpy()]
            filtered_predictions = model.predict(test_features.loc[unclassified_mask])
            filtered_probabilities = model.predict_proba(
                test_features.loc[unclassified_mask]
            )
            _print_metrics(
                "{} rule-unclassified test set ({} rows)".format(
                    model_name,
                    int(unclassified_mask.sum()),
                ),
                filtered_labels,
                filtered_predictions,
                filtered_probabilities,
                class_names,
            )
            _log_model(model_name, model)

    joblib.dump(models["Random Forest"], "rf.joblib")
    models["XGBoost"].save_model("xgb.json")


if __name__ == "__main__":
    main()
