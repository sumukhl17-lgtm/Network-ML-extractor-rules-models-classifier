from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classifier import validate_classifier
from .config import Settings
from .decoder import decode_audio
from .preprocessing import preprocess_audio
from .registry import MODEL_NAME
from .taxonomy import ESC50_CLASSES, TAXONOMY_VERSION
from .yamnet import YAMNetEmbedder, YAMNET_EMBEDDING_SIZE


ESC50_REPOSITORY = "https://github.com/karolpiczak/ESC-50"
ESC50_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
ESC50_LICENSE = "CC BY-NC 3.0"
ESC50_CITATION = (
    "K. J. Piczak. ESC: Dataset for Environmental Sound Classification. "
    "Proceedings of the 23rd Annual ACM Conference on Multimedia, "
    "Brisbane, Australia, 2015."
)

DEFAULT_EXPERIMENT = "audio-classification-esc50"
CLASSIFIER_C = 1.0
CLASSIFIER_MAX_ITER = 2000
RANDOM_STATE = 42
CHECKPOINT_INTERVAL = 100


@dataclass(frozen=True)
class DatasetRecord:
    filename: str
    label: str
    fold: int
    path: Path


@dataclass(frozen=True)
class EmbeddingDataset:
    embeddings: np.ndarray
    labels: np.ndarray
    folds: np.ndarray
    filenames: tuple[str, ...]
    metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_manifest(
    dataset_dir: str | Path,
) -> tuple[tuple[DatasetRecord, ...], str]:
    """Validate the complete official ESC-50 five-fold dataset."""

    root = Path(dataset_dir)
    csv_path = root / "esc50.csv"
    audio_dir = root / "audio"

    if not csv_path.is_file() or not audio_dir.is_dir():
        raise ValueError("ESC-50 dataset directory is incomplete.")

    records: list[DatasetRecord] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "fold", "category"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("ESC-50 metadata is missing required columns.")

        for row in reader:
            filename = row["filename"]
            label = row["category"]

            if (
                not filename
                or Path(filename).name != filename
                or not filename.lower().endswith(".wav")
                or label not in ESC50_CLASSES
            ):
                raise ValueError("ESC-50 metadata contains an invalid record.")

            try:
                fold = int(row["fold"])
            except (TypeError, ValueError):
                raise ValueError(
                    "ESC-50 metadata contains an invalid fold."
                ) from None

            if fold not in range(1, 6):
                raise ValueError("ESC-50 fold must be between 1 and 5.")

            path = audio_dir / filename
            if not path.is_file():
                raise ValueError(f"ESC-50 audio file is missing: {filename}")

            records.append(
                DatasetRecord(
                    filename=filename,
                    label=label,
                    fold=fold,
                    path=path,
                )
            )

    if len(records) != 2000:
        raise ValueError("ESC-50 must contain exactly 2,000 recordings.")

    if len({record.filename for record in records}) != 2000:
        raise ValueError("ESC-50 metadata contains duplicate filenames.")

    labels = [record.label for record in records]
    folds = [record.fold for record in records]

    if set(labels) != set(ESC50_CLASSES):
        raise ValueError(
            "ESC-50 class labels do not match the application taxonomy."
        )

    if any(labels.count(label) != 40 for label in ESC50_CLASSES):
        raise ValueError("ESC-50 must contain 40 recordings per class.")

    if any(folds.count(fold) != 400 for fold in range(1, 6)):
        raise ValueError("ESC-50 must contain 400 recordings per fold.")

    for label in ESC50_CLASSES:
        for fold in range(1, 6):
            count = sum(
                record.label == label and record.fold == fold
                for record in records
            )
            if count != 8:
                raise ValueError(
                    "ESC-50 must contain 8 recordings per class in each fold."
                )

    return tuple(records), _sha256_file(csv_path)


def _save_cache(
    path: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    filenames: tuple[str, ...],
    metadata: dict[str, Any],
) -> None:
    """Atomically save a resumable embedding checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            np.savez_compressed(
                handle,
                embeddings=embeddings,
                labels=labels,
                folds=folds,
                filenames=np.asarray(filenames, dtype=np.str_),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _load_cache(
    path: Path,
    *,
    csv_sha256: str,
    yamnet_version: str,
    yamnet_sha256: str,
    records: tuple[DatasetRecord, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]] | None:
    """Load only a structurally valid cache for the exact dataset and model."""

    if not path.is_file():
        return None

    try:
        with np.load(path, allow_pickle=False) as archive:
            embeddings = archive["embeddings"].copy()
            labels = archive["labels"].copy()
            folds = archive["folds"].copy()
            filenames = tuple(str(value) for value in archive["filenames"])
            metadata = json.loads(str(archive["metadata"].item()))

        if not isinstance(metadata, dict):
            return None

        count = len(filenames)
        if not 1 <= count <= len(records):
            return None

        if metadata.get("csv_sha256") != csv_sha256:
            return None
        if metadata.get("yamnet_version") != yamnet_version:
            return None
        if metadata.get("yamnet_sha256") != yamnet_sha256:
            return None
        if metadata.get("samples") != count:
            return None
        if metadata.get("embedding_size") != YAMNET_EMBEDDING_SIZE:
            return None

        expected_filenames = tuple(
            record.filename for record in records[:count]
        )
        expected_labels = np.asarray(
            [record.label for record in records[:count]],
            dtype=np.str_,
        )
        expected_folds = np.asarray(
            [record.fold for record in records[:count]],
            dtype=np.int64,
        )

        if filenames != expected_filenames:
            return None
        if not np.array_equal(labels, expected_labels):
            return None
        if not np.array_equal(folds, expected_folds):
            return None
        if embeddings.shape != (count, YAMNET_EMBEDDING_SIZE):
            return None
        if embeddings.dtype != np.float32:
            return None
        if not np.isfinite(embeddings).all():
            return None

        return embeddings, labels, folds, count, metadata

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        return None


def _model_fingerprint(embedder: YAMNetEmbedder) -> tuple[str, str]:
    """Read model identity without running inference on an audio clip."""

    info = embedder.model_info()

    if not isinstance(info, dict) or info.get("loaded") is not True:
        raise ValueError("YAMNet did not report a loaded model.")

    version = info.get("version")
    sha256 = info.get("sha256")

    if (
        not isinstance(version, str)
        or not version
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        raise ValueError("YAMNet returned invalid model provenance.")

    return version, sha256


def _validate_embedding_result(
    result: Any,
    *,
    filename: str,
    yamnet_version: str,
    yamnet_sha256: str,
) -> np.ndarray:
    if (
        getattr(result, "model_version", None) != yamnet_version
        or getattr(result, "model_sha256", None) != yamnet_sha256
    ):
        raise ValueError("YAMNet model fingerprint changed during extraction.")

    embedding = getattr(result, "embedding", None)

    if (
        not isinstance(embedding, np.ndarray)
        or embedding.shape != (YAMNET_EMBEDDING_SIZE,)
        or embedding.dtype != np.float32
        or not np.isfinite(embedding).all()
    ):
        raise ValueError(f"Invalid YAMNet embedding for {filename}.")

    return embedding


def _embed_record(
    record: DatasetRecord,
    *,
    embedder: YAMNetEmbedder,
    settings: Settings,
) -> Any:
    decoded = decode_audio(record.path.read_bytes(), settings)
    waveform = preprocess_audio(decoded, settings)
    return embedder.embed(waveform)


def extract_dataset_embeddings(
    dataset_dir: str | Path,
    *,
    cache_path: str | Path,
    embedder: YAMNetEmbedder | None = None,
    settings: Settings | None = None,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
) -> EmbeddingDataset:
    """Extract production-path YAMNet features with resumable checkpoints."""

    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive.")

    records, csv_sha256 = load_dataset_manifest(dataset_dir)
    if not records:
        raise ValueError("ESC-50 contains no recordings.")

    settings = settings or Settings()
    embedder = embedder or YAMNetEmbedder()
    cache = Path(cache_path)

    embedder.load()
    yamnet_version, yamnet_sha256 = _model_fingerprint(embedder)

    cached = _load_cache(
        cache,
        csv_sha256=csv_sha256,
        yamnet_version=yamnet_version,
        yamnet_sha256=yamnet_sha256,
        records=records,
    )

    if cached is not None:
        embeddings, labels, folds, completed, metadata = cached

        if completed == len(records):
            return EmbeddingDataset(
                embeddings=embeddings,
                labels=labels,
                folds=folds,
                filenames=tuple(record.filename for record in records),
                metadata=metadata,
            )

        embedding_rows = [row.copy() for row in embeddings]

    else:
        first_result = _embed_record(
            records[0],
            embedder=embedder,
            settings=settings,
        )
        first_embedding = _validate_embedding_result(
            first_result,
            filename=records[0].filename,
            yamnet_version=yamnet_version,
            yamnet_sha256=yamnet_sha256,
        )

        embedding_rows = [first_embedding.copy()]
        completed = 1
        metadata: dict[str, Any] = {}

    def checkpoint() -> None:
        nonlocal metadata

        current_embeddings = np.stack(embedding_rows).astype(
            np.float32,
            copy=False,
        )
        current_records = records[: len(embedding_rows)]

        current_metadata = {
            "dataset": "ESC-50",
            "dataset_repository": ESC50_REPOSITORY,
            "dataset_commit": ESC50_COMMIT,
            "dataset_license": ESC50_LICENSE,
            "dataset_citation": ESC50_CITATION,
            "csv_sha256": csv_sha256,
            "yamnet_version": yamnet_version,
            "yamnet_sha256": yamnet_sha256,
            "embedding_size": YAMNET_EMBEDDING_SIZE,
            "samples": len(current_records),
        }

        current_labels = np.asarray(
            [record.label for record in current_records],
            dtype=np.str_,
        )
        current_folds = np.asarray(
            [record.fold for record in current_records],
            dtype=np.int64,
        )
        current_filenames = tuple(
            record.filename for record in current_records
        )

        _save_cache(
            cache,
            current_embeddings,
            current_labels,
            current_folds,
            current_filenames,
            current_metadata,
        )

        # Keep the returned dataset metadata in sync with the saved checkpoint.
        metadata = current_metadata

    for index in range(completed, len(records)):
        record = records[index]
        result = _embed_record(
            record,
            embedder=embedder,
            settings=settings,
        )
        embedding = _validate_embedding_result(
            result,
            filename=record.filename,
            yamnet_version=yamnet_version,
            yamnet_sha256=yamnet_sha256,
        )
        embedding_rows.append(embedding.copy())

        if len(embedding_rows) % checkpoint_interval == 0:
            checkpoint()
            print(
                f"Extracted {len(embedding_rows)}/{len(records)} "
                "ESC-50 embeddings",
                flush=True,
            )

    checkpoint()

    embeddings = np.stack(embedding_rows).astype(np.float32, copy=False)
    labels = np.asarray([record.label for record in records], dtype=np.str_)
    folds = np.asarray([record.fold for record in records], dtype=np.int64)
    filenames = tuple(record.filename for record in records)

    return EmbeddingDataset(
        embeddings=embeddings,
        labels=labels,
        folds=folds,
        filenames=filenames,
        metadata=metadata,
    )


def build_classifier() -> Pipeline:
    """Create the fixed, reproducible probabilistic classifier."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=CLASSIFIER_C,
                    max_iter=CLASSIFIER_MAX_ITER,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_five_fold(
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> dict[str, Any]:
    """Evaluate each official ESC-50 fold exactly once as held-out data."""

    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != YAMNET_EMBEDDING_SIZE
        or embeddings.shape[0] != 2000
        or embeddings.dtype != np.float32
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Training embeddings have an invalid shape or value.")

    if labels.shape != (2000,) or folds.shape != (2000,):
        raise ValueError("Training labels or folds have an invalid shape.")

    if set(labels.tolist()) != set(ESC50_CLASSES):
        raise ValueError("Training labels do not match the ESC-50 taxonomy.")

    if set(folds.tolist()) != {1, 2, 3, 4, 5}:
        raise ValueError("Training data must contain all five official folds.")

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

    fold_results: dict[str, dict[str, float]] = {}

    for fold in range(1, 6):
        train_mask = folds != fold
        test_mask = folds == fold

        if np.any(train_mask & test_mask):
            raise ValueError("Cross-validation data leakage detected.")

        model = build_classifier()
        model.fit(embeddings[train_mask], labels[train_mask])
        predictions = model.predict(embeddings[test_mask])

        fold_results[str(fold)] = {
            "accuracy": float(
                accuracy_score(labels[test_mask], predictions)
            ),
            "macro_f1": float(
                f1_score(
                    labels[test_mask],
                    predictions,
                    labels=list(ESC50_CLASSES),
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    accuracies = [result["accuracy"] for result in fold_results.values()]
    macro_f1_scores = [
        result["macro_f1"] for result in fold_results.values()
    ]

    return {
        "folds": fold_results,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1_scores)),
        "macro_f1_std": float(np.std(macro_f1_scores)),
        "evaluation": "official five-fold ESC-50 cross-validation",
    }


def train_and_register(
    *,
    dataset_dir: str | Path,
    cache_path: str | Path,
    tracking_uri: str,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> dict[str, Any]:
    """Extract features, evaluate, train, and register without alias promotion."""

    if not tracking_uri.strip():
        raise ValueError("An MLflow tracking URI is required.")

    dataset = extract_dataset_embeddings(
        dataset_dir,
        cache_path=cache_path,
    )

    metrics = evaluate_five_fold(
        dataset.embeddings,
        dataset.labels,
        dataset.folds,
    )

    final_model = build_classifier()
    final_model.fit(dataset.embeddings, dataset.labels)
    validate_classifier(final_model)

    metadata = {
        **dataset.metadata,
        "model_name": MODEL_NAME,
        "classifier": "StandardScaler + LogisticRegression",
        "classifier_parameters": {
            "C": CLASSIFIER_C,
            "max_iter": CLASSIFIER_MAX_ITER,
            "solver": "lbfgs",
            "random_state": RANDOM_STATE,
        },
        "taxonomy_version": TAXONOMY_VERSION,
        "class_labels": list(ESC50_CLASSES),
        "feature_extractor": "YAMNet",
        "feature_dimension": YAMNET_EMBEDDING_SIZE,
        "training_samples": int(dataset.embeddings.shape[0]),
        "cross_validation": metrics,
        "training_data_usage": (
            "Non-commercial research and prototype evaluation only; "
            "check dataset and model licensing before commercial use."
        ),
    }

    import mlflow
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    input_example = dataset.embeddings[:2]
    output_example = final_model.predict_proba(input_example)
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run(
        run_name="esc50-yamnet-logistic-regression"
    ) as run:
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "classifier": "StandardScaler+LogisticRegression",
                "classifier_C": CLASSIFIER_C,
                "classifier_max_iter": CLASSIFIER_MAX_ITER,
                "classifier_solver": "lbfgs",
                "random_state": RANDOM_STATE,
                "dataset": "ESC-50",
                "dataset_commit": ESC50_COMMIT,
                "dataset_license": ESC50_LICENSE,
                "yamnet_version": dataset.metadata["yamnet_version"],
                "yamnet_sha256": dataset.metadata["yamnet_sha256"],
                "embedding_dimension": YAMNET_EMBEDDING_SIZE,
                "training_samples": int(dataset.embeddings.shape[0]),
                "taxonomy_version": TAXONOMY_VERSION,
            }
        )

        mlflow.log_metrics(
            {
                "cv_accuracy_mean": metrics["accuracy_mean"],
                "cv_accuracy_std": metrics["accuracy_std"],
                "cv_macro_f1_mean": metrics["macro_f1_mean"],
                "cv_macro_f1_std": metrics["macro_f1_std"],
            }
        )

        for fold, fold_metrics in metrics["folds"].items():
            mlflow.log_metrics(
                {
                    f"fold_{fold}_accuracy": fold_metrics["accuracy"],
                    f"fold_{fold}_macro_f1": fold_metrics["macro_f1"],
                }
            )

        mlflow.log_dict(metadata, "audio_model_meta.json")

        mlflow.sklearn.log_model(
            final_model,
            name="model",
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "numpy==2.0.2",
                "scikit-learn==1.6.1",
            ],
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(
            model_uri,
            MODEL_NAME,
            await_registration_for=300,
        )

    return {
        "run_id": run.info.run_id,
        "model_name": MODEL_NAME,
        "model_version": str(registered.version),
        "model_uri": model_uri,
        "cross_validation": metrics,
        "champion_alias_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and register the ESC-50 YAMNet audio classifier."
    )
    parser.add_argument(
        "--dataset-dir",
        default="/workspace/datasets/ESC-50",
    )
    parser.add_argument(
        "--cache",
        default="/cache/esc50-yamnet-embeddings.npz",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
    )
    args = parser.parse_args()

    result = train_and_register(
        dataset_dir=args.dataset_dir,
        cache_path=args.cache,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
