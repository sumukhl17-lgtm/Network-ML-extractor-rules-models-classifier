"""Register the configured local text model in MLflow without retraining it."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

from . import config, registry

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def _read_metadata(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model metadata file not found: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read model metadata from {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Model metadata in {path} must be a JSON object")
    return metadata


def _evaluation_metrics(value: Any, prefix: str = "metrics") -> Dict[str, float]:
    """Flatten numeric metric leaves into MLflow metric names."""
    flattened: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            flattened.update(_evaluation_metrics(child, f"{prefix}__{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            flattened.update(_evaluation_metrics(child, f"{prefix}__{index}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            flattened[prefix] = numeric
    return flattened


def register_existing(promote: bool = False) -> Dict[str, Any]:
    """Log the configured joblib model and metadata, then register a new version.

    The configured alias is left untouched unless ``promote`` is explicitly true.
    """
    model_path = config.model_file()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    metadata = _read_metadata(config.meta_file())

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the existing model (pip install joblib)") from exc
    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load the existing model from {model_path}: {exc}") from exc

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip() or DEFAULT_TRACKING_URI
    try:
        mlflow, MlflowClient = registry._mlflow()
        if not os.getenv("MLFLOW_TRACKING_URI", "").strip():
            mlflow.set_tracking_uri(tracking_uri)

        model_name = config.registered_model_name()
        alias = config.model_alias()
        mlflow.set_experiment(config.experiment_name())
        with mlflow.start_run() as run:
            mlflow.log_metrics(_evaluation_metrics(metadata.get("metrics", {})))
            mlflow.log_dict(metadata, "model_meta.json")
            model_info = registry._log_sklearn_model(mlflow, pipeline)
            run_id = run.info.run_id
            model_uri = getattr(model_info, "model_uri", None) or f"runs:/{run_id}/model"

        model_version = mlflow.register_model(model_uri, model_name)
        version = str(model_version.version)
        if promote:
            client = MlflowClient()
            client.set_registered_model_alias(model_name, alias, version)
    except registry.RegistryUnavailable:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not register the existing model with MLflow at {tracking_uri}: {exc}"
        ) from exc

    return {
        "version": version,
        "run_id": run_id,
        "alias": alias,
        "promoted": promote,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register the existing text model in MLflow without retraining it."
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="assign the configured model alias to the newly registered version",
    )
    args = parser.parse_args(argv)

    try:
        result = register_existing(promote=args.promote)
    except Exception as exc:
        print(f"Registration failed: {exc}", file=sys.stderr)
        return 1

    print(f"Registered version {result['version']} (run ID: {result['run_id']}).")
    if result["promoted"]:
        print(f"Assigned alias '{result['alias']}' to version {result['version']}.")
    else:
        print(f"Alias '{result['alias']}' was not changed; use --promote to assign it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
