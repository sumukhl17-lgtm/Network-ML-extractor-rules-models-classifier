from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

from .service import ModelUnavailable

logger = logging.getLogger(__name__)
MODEL_NAME = "audio-classifier"
MODEL_ALIAS = "champion"


class AudioRegistryUnavailable(ModelUnavailable):
    pass


def _mlflow() -> tuple[Any, Any]:
    try:
        import mlflow
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException
    except ImportError:
        raise AudioRegistryUnavailable() from None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise AudioRegistryUnavailable()
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow, MlflowClient


def load_champion() -> tuple[object, dict[str, Any], str]:
    mlflow, MlflowClient = _mlflow()
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = client.download_artifacts(
                version.run_id,
                "audio_model_meta.json",
                temp_dir,
            )
            metadata = json.loads(open(path, encoding="utf-8").read())
    except (MlflowException, OSError, ValueError, TypeError):
        logger.exception("Could not load the Audio champion model.")
        raise AudioRegistryUnavailable() from None

    return model, metadata, str(version.version)
