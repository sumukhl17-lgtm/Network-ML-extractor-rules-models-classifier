from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .service import ModelUnavailable

logger = logging.getLogger(__name__)

BUNDLE_VERSION = "bundle-v1"
REQUIRED_FILES = frozenset(
    {
        "model.skops",
        "MLmodel",
        "requirements.txt",
        "model_meta.json",
    }
)


def _bundle_path(configured_path: str | None = None) -> Path:
    if configured_path:
        return Path(configured_path).expanduser()
    return Path(__file__).resolve().parent / "model_bundle"


def _read_manifest(bundle_dir: Path) -> dict[str, str]:
    manifest_path = bundle_dir / "SHA256SUMS"

    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise ModelUnavailable() from None

    checksums: dict[str, str] = {}

    for line in lines:
        if not line:
            continue

        if len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise ModelUnavailable()

        digest = line[:64].lower()
        filename = line[66:]

        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or filename not in REQUIRED_FILES
            or filename in checksums
        ):
            raise ModelUnavailable()

        checksums[filename] = digest

    if set(checksums) != REQUIRED_FILES:
        raise ModelUnavailable()

    return checksums


def verify_bundle(bundle_dir: Path) -> None:
    """Verify every required model file against the packaged SHA-256 manifest."""

    checksums = _read_manifest(bundle_dir)

    for filename, expected_digest in checksums.items():
        path = bundle_dir / filename
        digest = hashlib.sha256()

        try:
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            raise ModelUnavailable() from None

        if digest.hexdigest() != expected_digest:
            raise ModelUnavailable()


def load_bundle(
    configured_path: str | None = None,
) -> tuple[object, dict[str, Any], str]:
    """Load the packaged Audio classifier after verifying its checksums."""

    bundle_dir = _bundle_path(configured_path)
    verify_bundle(bundle_dir)

    try:
        metadata = json.loads(
            (bundle_dir / "model_meta.json").read_text(encoding="utf-8")
        )
        if not isinstance(metadata, dict):
            raise ValueError("Model metadata must be a JSON object")

        import mlflow.sklearn

        model = mlflow.sklearn.load_model(str(bundle_dir))
    except Exception:
        logger.exception("Could not load the packaged Audio model bundle.")
        raise ModelUnavailable() from None

    return model, metadata, BUNDLE_VERSION
