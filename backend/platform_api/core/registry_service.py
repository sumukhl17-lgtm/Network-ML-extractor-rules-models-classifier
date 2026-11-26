"""Model registry logic: list versions, compare, promote, roll back, reload.

Works with any object that has the MlflowClient methods used below, so tests pass a
fake client and the real app passes mlflow.MlflowClient (MLflow 2.x and 3.x).

Vocabulary (also on slide 16 of the deck):
  version   every retrain registers a new, numbered version of a named model
  champion  an alias (a movable label) pointing at the version that should be live
  promote   move the champion alias to another version
  rollback  move it back to the previous champion
The backend loads "name@champion", so moving the alias changes what is served
without redeploying any code.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .. import config
from . import activity
from .errors import AppError

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

NETWORK_NOTE = (
    "The network classifier loads rf.joblib from the repository, not from the registry, "
    "so promoting a version here would not change what is served. This model is shown "
    "read-only until the network code loads its champion version from MLflow."
)


def _default_client():
    uri = config.mlflow_uri()
    if not uri:
        raise AppError(503, "REGISTRY_UNAVAILABLE", "MLFLOW_TRACKING_URI is not set, so there is no registry to talk to",
                       {"hint": "Start the MLflow server (mlflow/start_mlflow.sh) and export MLFLOW_TRACKING_URI."})
    try:
        from mlflow import MlflowClient
    except ImportError as e:
        raise AppError(503, "REGISTRY_UNAVAILABLE", "mlflow is not installed: {}".format(e)) from e
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def _text_reload() -> Dict[str, Any]:
    from backend import text
    text.reload_model()
    return text.model_info()


def _text_serving() -> Dict[str, Any]:
    from backend import text
    return text.model_info()


def _iso(ms: Optional[int]) -> Optional[str]:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")


def _check_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise AppError(400, "INVALID_MODEL_NAME", "Model names may only contain letters, digits, '.', '_' and '-'")
    return name


class RegistryService:
    def __init__(self, client_factory: Optional[Callable[[], Any]] = None,
                 reloaders: Optional[Dict[str, Callable[[], Dict[str, Any]]]] = None,
                 serving: Optional[Dict[str, Callable[[], Dict[str, Any]]]] = None):
        self._client_factory = client_factory or _default_client
        self._reloaders = reloaders if reloaders is not None else {"text": _text_reload}
        self._serving = serving if serving is not None else {"text": _text_serving}

    # ---- basics ------------------------------------------------------------------
    def _client(self):
        client = self._client_factory()
        return client

    @staticmethod
    def known_models() -> List[Dict[str, str]]:
        return [
            {"name": config.network_model_name(), "branch": "network"},
            {"name": config.text_model_name(), "branch": "text"},
        ]

    def branch_of(self, name: str) -> str:
        for entry in self.known_models():
            if entry["name"] == name:
                return entry["branch"]
        return "other"

    def serving_status(self, branch: str) -> Dict[str, Any]:
        if branch == "network":
            return {"source": "repo-file", "file": "rf.joblib", "wired_to_registry": False, "note": NETWORK_NOTE}
        fn = self._serving.get(branch)
        if fn is None:
            return {"source": "unknown", "wired_to_registry": False}
        try:
            info = fn()
        except Exception as e:  # the model may be missing; report it instead of failing the page
            return {"source": "error", "error": str(e), "wired_to_registry": False}
        return {"source": info.get("source"), "version": info.get("version"),
                "wired_to_registry": info.get("source") == "registry",
                "data_source": info.get("data_source"),
                "real_data_validated": bool(info.get("real_data_validated", False)),
                "evaluation_note": info.get("evaluation_note"),
                "unknown_threshold": info.get("unknown_threshold"),
                "metrics": info.get("metrics", {})}

    def _versions(self, client, name: str) -> List[Any]:
        found = list(client.search_model_versions("name='{}'".format(_check_name(name))))
        return sorted(found, key=lambda v: int(v.version), reverse=True)

    def _champion(self, client, name: str) -> Optional[str]:
        try:
            return str(client.get_model_version_by_alias(name, config.model_alias()).version)
        except Exception:
            return None

    def _run_metrics(self, client, run_id: Optional[str]) -> Dict[str, float]:
        if not run_id:
            return {}
        try:
            return {k: float(v) for k, v in client.get_run(run_id).data.metrics.items()}
        except Exception:
            return {}

    def _describe(self, client, name: str) -> Dict[str, Any]:
        versions = self._versions(client, name)
        champion = self._champion(client, name)
        described = []
        for v in versions:
            described.append({
                "version": str(v.version),
                "created_at": _iso(getattr(v, "creation_timestamp", None)),
                "run_id": getattr(v, "run_id", None),
                "status": str(getattr(v, "status", "")),
                "is_champion": str(v.version) == champion,
                "metrics": self._run_metrics(client, getattr(v, "run_id", None)),
            })
        branch = self.branch_of(name)
        return {"name": name, "branch": branch, "registered": bool(versions), "champion": champion,
                "can_promote": branch != "network", "serving": self.serving_status(branch), "versions": described}

    # ---- endpoints -----------------------------------------------------------------
    def list_models(self) -> Dict[str, Any]:
        """Never raises for a missing registry: the UI shows a banner instead of an error page."""
        try:
            client = self._client()
            names = [m["name"] for m in self.known_models()]
            for registered in client.search_registered_models():
                if registered.name not in names:
                    names.append(registered.name)
            models = [self._describe(client, n) for n in names]
        except AppError as e:
            return {"available": False, "reason": e.message, "alias": config.model_alias(), "models": []}
        except Exception as e:
            return {"available": False, "reason": "Could not reach the MLflow server: {}".format(e),
                    "alias": config.model_alias(), "models": []}
        return {"available": True, "reason": None, "alias": config.model_alias(), "models": models}

    def compare(self, name: str, versions: List[str]) -> Dict[str, Any]:
        client = self._client()
        if len(versions) < 2:
            raise AppError(400, "NEED_TWO_VERSIONS", "Pick at least two versions to compare")
        existing = {str(v.version): v for v in self._versions(client, name)}
        if not existing:
            raise AppError(404, "MODEL_NOT_FOUND", "No registered model named '{}'".format(name))
        missing = [v for v in versions if v not in existing]
        if missing:
            raise AppError(404, "VERSION_NOT_FOUND", "Unknown version(s): {}".format(", ".join(missing)))
        per_version = {v: self._run_metrics(client, getattr(existing[v], "run_id", None)) for v in versions}
        metric_names = sorted({m for values in per_version.values() for m in values})
        return {
            "name": name, "versions": versions, "champion": self._champion(client, name),
            "metrics": {m: {v: per_version[v].get(m) for v in versions} for m in metric_names},
        }

    def promote(self, name: str, version: str, actor: str, confirm: bool, reload: bool = True) -> Dict[str, Any]:
        self._require_confirm(confirm, "promote")
        branch = self._require_promotable(name)
        client = self._client()
        existing = {str(v.version) for v in self._versions(client, name)}
        if not existing:
            raise AppError(404, "MODEL_NOT_FOUND", "No registered model named '{}'".format(name))
        version = str(version)
        if version not in existing:
            raise AppError(404, "VERSION_NOT_FOUND", "Version {} of '{}' does not exist".format(version, name))
        current = self._champion(client, name)
        if current == version:
            raise AppError(409, "ALREADY_CHAMPION", "Version {} is already the champion".format(version))
        client.set_registered_model_alias(name, config.model_alias(), version)
        entry = activity.record("promote", name, actor, current, version)
        return self._after_change(branch, name, entry, reload)

    def rollback(self, name: str, actor: str, confirm: bool, to_version: Optional[str] = None,
                 reload: bool = True) -> Dict[str, Any]:
        self._require_confirm(confirm, "roll back")
        branch = self._require_promotable(name)
        client = self._client()
        existing = {str(v.version) for v in self._versions(client, name)}
        current = self._champion(client, name)
        if current is None:
            raise AppError(409, "NO_CHAMPION", "'{}' has no champion version to roll back from".format(name))
        target = str(to_version) if to_version else None
        if target is None:
            for entry in activity.read(model=name, limit=1000):       # newest first
                if entry.get("to_version") == current and entry.get("from_version"):
                    target = str(entry["from_version"])
                    break
        if target is None:
            raise AppError(409, "NO_PREVIOUS_VERSION",
                           "No earlier champion is recorded for '{}'. Choose a version and promote it instead.".format(name))
        if target not in existing:
            raise AppError(404, "VERSION_NOT_FOUND", "Version {} of '{}' does not exist".format(target, name))
        if target == current:
            raise AppError(409, "ALREADY_CHAMPION", "Version {} is already the champion".format(target))
        client.set_registered_model_alias(name, config.model_alias(), target)
        entry = activity.record("rollback", name, actor, current, target)
        return self._after_change(branch, name, entry, reload)

    def reload(self, name: str) -> Dict[str, Any]:
        branch = self.branch_of(_check_name(name))
        if branch == "network":
            return {"reloaded": False, "reason": NETWORK_NOTE}
        fn = self._reloaders.get(branch)
        if fn is None:
            raise AppError(404, "NOT_SERVED", "'{}' is not a model this backend serves".format(name))
        try:
            info = fn()
        except Exception as e:
            raise AppError(503, "RELOAD_FAILED", "Reloading '{}' failed: {}".format(name, e)) from e
        return {"reloaded": True, "serving": {"source": info.get("source"), "version": info.get("version")}}

    def network_runs(self, limit: int = 50) -> Dict[str, Any]:
        """Read-only list of Sumukh's MLflow runs (his model is not registered yet)."""
        client = self._client()
        experiment = client.get_experiment_by_name(config.network_experiment())
        if experiment is None:
            return {"experiment": config.network_experiment(), "found": False, "runs": []}
        runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"],
                                  max_results=limit)
        out = []
        for run in runs:
            params = run.data.params
            out.append({
                "run_id": run.info.run_id,
                "run_name": getattr(run.info, "run_name", None) or run.data.tags.get("mlflow.runName"),
                "started_at": _iso(getattr(run.info, "start_time", None)),
                "status": str(run.info.status),
                "metrics": {k: float(v) for k, v in run.data.metrics.items()},
                "params": {k: params[k] for k in ("features", "train_source_files", "test_source_files") if k in params},
            })
        return {"experiment": config.network_experiment(), "found": True, "runs": out}

    # ---- helpers ---------------------------------------------------------------------
    @staticmethod
    def _require_confirm(confirm: bool, verb: str) -> None:
        if confirm is not True:
            raise AppError(400, "CONFIRMATION_REQUIRED", "Set confirm=true to {} a model version".format(verb))

    def _require_promotable(self, name: str) -> str:
        branch = self.branch_of(_check_name(name))
        if branch == "network":
            raise AppError(409, "NOT_WIRED_TO_REGISTRY", NETWORK_NOTE)
        return branch

    def _after_change(self, branch: str, name: str, entry: Dict[str, Any], reload: bool) -> Dict[str, Any]:
        result: Dict[str, Any] = {"activity": entry, "champion": entry["to_version"], "reload": None}
        fn = self._reloaders.get(branch)
        if reload and fn is not None:
            try:
                info = fn()
                result["reload"] = {"reloaded": True, "serving": {"source": info.get("source"), "version": info.get("version")}}
            except Exception as e:   # the alias already moved; tell the caller rather than undo it
                result["reload"] = {"reloaded": False, "error": str(e)}
        return result
