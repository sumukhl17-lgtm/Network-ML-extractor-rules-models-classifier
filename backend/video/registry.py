"""Resolve the Video CLIP configuration from the shared MLflow champion alias.

The registry stores a small JSON configuration (model id + prompt-set version),
not the large pretrained CLIP weights. Hugging Face caches the weights locally.
"""
from __future__ import annotations

import os

from . import config


class RegistryUnavailable(RuntimeError):
    """MLflow is unavailable or no usable Video champion exists."""


def load_champion_config() -> dict:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RegistryUnavailable("MLFLOW_TRACKING_URI is not set")
    try:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))
        version = client.get_model_version_by_alias(config.registered_model_name(), config.model_alias())
    except ImportError as exc:
        raise RegistryUnavailable("MLflow is not installed") from exc
    except Exception as exc:
        raise RegistryUnavailable(
            f"No usable champion for {config.registered_model_name()}@{config.model_alias()}: {exc}"
        ) from exc

    tags = getattr(version, "tags", {}) or {}
    model_id = tags.get("video_model_id")
    prompt_version = tags.get("video_prompt_set", "v1")
    if not model_id:
        raise RegistryUnavailable(
            f"Champion version {version.version} has no video_model_id tag; register it with video.register_model"
        )
    return {
        "model_id": model_id,
        "version": str(version.version),
        "source": "registry",
        "prompt_set": prompt_version,
    }
