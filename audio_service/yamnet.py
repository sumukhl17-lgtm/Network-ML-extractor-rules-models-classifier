from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .preprocessing import MIN_MODEL_INPUT_SAMPLES
from .service import AudioServiceError, ModelUnavailable

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_VERSION = "google/yamnet/1"
YAMNET_CLASS_COUNT = 521
YAMNET_EMBEDDING_SIZE = 1024
logger = logging.getLogger(__name__)

ModelLoader = Callable[[], tuple[object, Sequence[str], str]]


@dataclass(frozen=True)
class YAMNetOutput:
    class_scores: np.ndarray
    embedding: np.ndarray
    class_names: tuple[str, ...]
    model_version: str
    model_sha256: str


def _invalid_audio() -> AudioServiceError:
    return AudioServiceError(
        400,
        "invalid_audio",
        "The preprocessed waveform is invalid.",
    )


def _invalid_model_output() -> AudioServiceError:
    return AudioServiceError(
        502,
        "invalid_model_output",
        "The audio model returned an invalid result.",
    )


def _asset_tree_sha256(model_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in model_root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("The YAMNet artifact directory is empty.")
    for path in files:
        relative = path.relative_to(model_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as artifact:
            while chunk := artifact.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _official_model_loader() -> tuple[object, Sequence[str], str]:
    if "tensorflow" in sys.modules and os.environ.get("TF_USE_LEGACY_KERAS") != "1":
        raise ModelUnavailable()
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        raise ModelUnavailable() from None

    try:
        model = hub.load(YAMNET_HANDLE)
        class_map_value = model.class_map_path().numpy()
        class_map_path = Path(class_map_value.decode("utf-8"))
        with class_map_path.open(encoding="utf-8", newline="") as class_map:
            class_names = tuple(
                row["display_name"] for row in csv.DictReader(class_map)
            )
        model_sha256 = _asset_tree_sha256(class_map_path.parent.parent)
    except (OSError, RuntimeError, ValueError, tf.errors.OpError) as exc:
        logger.error("YAMNet initialization failed: %s", type(exc).__name__)
        raise ModelUnavailable() from None

    return model, class_names, model_sha256


def _to_float32_array(value: object) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


class YAMNetEmbedder:
    """Lazy, process-cached YAMNet inference with temporal mean pooling."""

    def __init__(self, model_loader: ModelLoader | None = None) -> None:
        self._model_loader = model_loader or _official_model_loader
        self._model: object | None = None
        self._class_names: tuple[str, ...] | None = None
        self._model_sha256: str | None = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            model, class_names, model_sha256 = self._model_loader()
            names = tuple(str(name) for name in class_names)
            if (
                len(names) != YAMNET_CLASS_COUNT
                or any(not name for name in names)
                or len(model_sha256) != 64
                or any(char not in "0123456789abcdef" for char in model_sha256)
            ):
                raise ModelUnavailable()
            self._model = model
            self._class_names = names
            self._model_sha256 = model_sha256

    def load(self) -> None:
        """Load and validate the YAMNet model during worker startup."""
        self._ensure_loaded()

    def model_info(self) -> dict[str, str | bool | None]:
        if not self.loaded:
            return {
                "loaded": False,
                "version": None,
                "sha256": None,
            }
        return {
            "loaded": True,
            "version": YAMNET_VERSION,
            "sha256": self._model_sha256,
        }

    def embed(self, waveform: np.ndarray) -> YAMNetOutput:
        if (
            not isinstance(waveform, np.ndarray)
            or waveform.dtype != np.float32
            or waveform.ndim != 1
            or waveform.size < MIN_MODEL_INPUT_SAMPLES
            or not np.isfinite(waveform).all()
            or np.any(waveform < -1.0)
            or np.any(waveform > 1.0)
        ):
            raise _invalid_audio()

        self._ensure_loaded()
        assert self._model is not None
        assert self._class_names is not None
        assert self._model_sha256 is not None

        try:
            outputs = self._model(waveform)
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.error("YAMNet inference failed: %s", type(exc).__name__)
            raise ModelUnavailable() from None
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
            raise _invalid_model_output()

        try:
            frame_scores = _to_float32_array(outputs[0])
            frame_embeddings = _to_float32_array(outputs[1])
        except (TypeError, ValueError, AttributeError):
            raise _invalid_model_output() from None
        if (
            frame_scores.ndim != 2
            or frame_scores.shape[0] == 0
            or frame_scores.shape[1] != YAMNET_CLASS_COUNT
            or frame_embeddings.ndim != 2
            or frame_embeddings.shape[0] != frame_scores.shape[0]
            or frame_embeddings.shape[1] != YAMNET_EMBEDDING_SIZE
            or not np.isfinite(frame_scores).all()
            or np.any(frame_scores < 0.0)
            or np.any(frame_scores > 1.0)
            or not np.isfinite(frame_embeddings).all()
        ):
            raise _invalid_model_output()

        class_scores = np.mean(frame_scores, axis=0, dtype=np.float32)
        embedding = np.mean(frame_embeddings, axis=0, dtype=np.float32)
        if (
            class_scores.shape != (YAMNET_CLASS_COUNT,)
            or embedding.shape != (YAMNET_EMBEDDING_SIZE,)
            or not np.isfinite(class_scores).all()
            or not np.isfinite(embedding).all()
        ):
            raise _invalid_model_output()

        return YAMNetOutput(
            class_scores=class_scores,
            embedding=embedding,
            class_names=self._class_names,
            model_version=YAMNET_VERSION,
            model_sha256=self._model_sha256,
        )
