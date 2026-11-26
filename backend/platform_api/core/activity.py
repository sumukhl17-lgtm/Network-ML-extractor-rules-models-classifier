"""Append-only activity log: who promoted / rolled back which model version, and when.

One JSON object per line in <APP_DATA_DIR>/activity.jsonl. A plain file is enough for a
single backend instance; Person 4 should mount APP_DATA_DIR as a volume so it survives
restarts (a multi-replica deployment would need a shared database instead).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .. import config

_lock = threading.Lock()


def _path():
    return config.data_dir() / "activity.jsonl"


def record(action: str, model: str, actor: str, from_version: Optional[str], to_version: str,
           note: str = "") -> Dict[str, Any]:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action,                 # promote | rollback
        "model": model,
        "actor": actor or "anonymous",
        "from_version": from_version,
        "to_version": to_version,
        "note": note,
    }
    with _lock:
        with open(_path(), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    return entry


def read(model: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Newest first."""
    path = _path()
    if not path.exists():
        return []
    entries = []
    with _lock:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue          # skip a corrupt line instead of failing the whole page
    if model:
        entries = [e for e in entries if e.get("model") == model]
    return list(reversed(entries))[:limit]
