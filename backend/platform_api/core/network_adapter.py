"""The only place that touches Sumukh's network code at the repository root.

His files use flat imports (`from rules import ...`), so we put the repository root on
sys.path rather than editing his code. If he later switches to relative imports,
only this file changes.

NOTE (for Person 1): classify_network.py imports MODEL_FEATURES from train_compare.py,
and train_compare.py imports mlflow / xgboost at the top. So serving needs both installed.
Moving MODEL_FEATURES to a tiny shared module would remove that coupling.
"""
from __future__ import annotations

import importlib
import sys
import threading
from typing import Any, Dict, List

from .. import config
from .errors import AppError

_lock = threading.Lock()
_module = None


def _load():
    global _module
    if _module is None:
        with _lock:
            if _module is None:
                path = str(config.NETWORK_DIR)
                if path not in sys.path:
                    sys.path.insert(0, path)
                try:
                    _module = importlib.import_module("classify_network")
                except Exception as e:  # missing file, wrong scikit-learn version, missing mlflow...
                    raise AppError(
                        503, "NETWORK_MODEL_UNAVAILABLE",
                        "The network classifier could not be loaded: {}".format(e),
                        {"hint": "Install dependencies from requirements.txt (scikit-learn must be 1.6.1 for rf.joblib)."},
                    ) from e
    return _module


def reset_for_tests() -> None:
    global _module
    _module = None
    for name in ("classify_network", "rules", "train_compare"):
        sys.modules.pop(name, None)


def classify(flow: Dict[str, Any]) -> Dict[str, Any]:
    """Call Sumukh's classify_network(flow) unchanged."""
    return _load().classify_network(flow)


def model_features() -> List[str]:
    """The flow fields the loaded model needs (read from the model itself)."""
    return list(_load()._MODEL_FEATURES)


def ml_overrides_unmatched() -> bool:
    return bool(_load().ML_OVERRIDES_UNMATCHED)


def extract_pcap(path: str):
    """PCAP file -> pandas DataFrame of flows (his extract_flows.extract)."""
    if str(config.NETWORK_DIR) not in sys.path:
        sys.path.insert(0, str(config.NETWORK_DIR))
    try:
        extract_flows = importlib.import_module("extract_flows")
        return extract_flows.extract(path)
    except ImportError as e:
        raise AppError(503, "EXTRACTOR_UNAVAILABLE",
                       "PCAP extraction needs NFStream: {}".format(e),
                       {"hint": "Install nfstream from requirements.txt."}) from e
