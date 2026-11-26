"""All app settings, read from environment variables at call time.

Nothing is hard-coded: Docker / Kubernetes (Person 4) only has to set variables.
Reading at call time (not import time) lets tests change them without re-importing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

BACKEND_DIR = Path(__file__).resolve().parents[1]
NETWORK_DIR = BACKEND_DIR.parent


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


# --- uploads ------------------------------------------------------------------
def max_upload_bytes() -> int:
    """Limit for network uploads (PCAP / CSV). The text branch has its own TEXT_MAX_BYTES."""
    return _int("APP_MAX_UPLOAD_BYTES", 100 * 1024 * 1024)


def max_flows() -> int:
    """Refuse to classify more flows than this in one request (protects the server)."""
    return _int("APP_MAX_FLOWS", 20000)


# --- web ----------------------------------------------------------------------
def cors_origins() -> List[str]:
    raw = os.getenv("APP_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


def data_dir() -> Path:
    """Where the app keeps its own small files (the promote / rollback activity log)."""
    path = Path(os.getenv("APP_DATA_DIR", str(BACKEND_DIR / "app_data")))
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- registry -----------------------------------------------------------------
def mlflow_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "").strip()


def network_model_name() -> str:
    # Placeholder name agreed with Sumukh; nothing is registered under it yet.
    return os.getenv("NETWORK_MODEL_NAME", "network-classifier")


def text_model_name() -> str:
    return os.getenv("TEXT_MODEL_NAME", "text-classifier")


def model_alias() -> str:
    return os.getenv("MODEL_ALIAS", "champion")


def network_experiment() -> str:
    return os.getenv("NETWORK_MLFLOW_EXPERIMENT", "network-known-unknown")
