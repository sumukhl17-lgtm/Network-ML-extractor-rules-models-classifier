import json
from pathlib import Path

import pytest

from backend.text import config, register_existing, registry
import fake_mlflow


def setup_existing_model(monkeypatch, tmp_path, metadata):
    model_path = tmp_path / "existing.joblib"
    model_path.write_bytes(b"existing-model-bytes")
    metadata_path = tmp_path / "model_meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(config, "model_file", lambda: model_path)
    monkeypatch.setattr(config, "meta_file", lambda: metadata_path)
    monkeypatch.setattr(config, "experiment_name", lambda: "existing-model-test")

    model = object()
    loaded_paths = []
    import joblib

    monkeypatch.setattr(joblib, "load", lambda path: loaded_paths.append(Path(path)) or model)
    return model_path, metadata_path, model, loaded_paths


def existing_metadata():
    return {
        "name": "text-classifier",
        "version": "local-b1095cad",
        "model_hash": "b1095cad",
        "data_source": "synthetic-demo",
        "real_data_validated": False,
        "evaluation_note": "DEMO MODEL; do not treat as real-world accuracy.",
        "metrics": {
            "accuracy": 0.91,
            "f1_macro": 0.89,
            "per_class": {"invoice": {"precision": 0.9, "support": 12}},
            "threshold_table": [{"threshold": 0.75, "unknown_recall": 0.8}],
            "unavailable_metric": None,
        },
    }


def fake_with_existing_champion(monkeypatch):
    store = fake_mlflow.install(monkeypatch)
    store["runs"]["old-run"] = {"params": {}, "metrics": {}, "artifacts": {}, "model": None}
    store["versions"]["text-classifier"] = ["old-run"]
    store["aliases"][("text-classifier", "champion")] = 1
    return store


def test_registers_existing_model_and_preserves_metadata_without_promotion(monkeypatch, tmp_path):
    metadata = existing_metadata()
    model_path, _, model, loaded_paths = setup_existing_model(monkeypatch, tmp_path, metadata)
    store = fake_with_existing_champion(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")

    result = register_existing.register_existing()

    assert loaded_paths == [model_path]
    assert result == {
        "version": "2",
        "run_id": "run1",
        "alias": "champion",
        "promoted": False,
    }
    assert store["aliases"][("text-classifier", "champion")] == 1
    run = store["runs"]["run1"]
    assert run["model"] is model
    assert run["artifacts"]["model_meta.json"] == metadata
    assert run["metrics"] == {
        "metrics__accuracy": 0.91,
        "metrics__f1_macro": 0.89,
        "metrics__per_class__invoice__precision": 0.9,
        "metrics__per_class__invoice__support": 12.0,
        "metrics__threshold_table__0__threshold": 0.75,
        "metrics__threshold_table__0__unknown_recall": 0.8,
    }
    assert store["log_model_calls"][0]["serialization_format"] == "skops"
    assert store["log_model_calls"][0]["skops_trusted_types"] == registry.SKOPS_TRUSTED_TYPES


def test_promote_assigns_alias_only_when_requested(monkeypatch, tmp_path):
    model_path, _, _, _ = setup_existing_model(monkeypatch, tmp_path, existing_metadata())
    store = fake_with_existing_champion(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")

    result = register_existing.register_existing(promote=True)

    assert model_path.is_file()
    assert result["promoted"] is True
    assert store["aliases"][("text-classifier", "champion")] == 2


def test_uses_requested_tracking_server_when_environment_is_unset(monkeypatch, tmp_path):
    setup_existing_model(monkeypatch, tmp_path, existing_metadata())
    store = fake_mlflow.install(monkeypatch)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    register_existing.register_existing()

    assert store["tracking_uri"] == "http://127.0.0.1:5000"


def test_mlflow_connection_failure_is_reported_with_tracking_uri(monkeypatch, tmp_path):
    setup_existing_model(monkeypatch, tmp_path, existing_metadata())
    fake_mlflow.install(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")
    import mlflow

    def unavailable(_name):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(mlflow, "set_experiment", unavailable)

    with pytest.raises(RuntimeError, match=r"http://fake:5000: connection refused"):
        register_existing.register_existing()


def test_missing_model_fails_before_mlflow(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing.joblib"
    metadata_path = tmp_path / "model_meta.json"
    metadata_path.write_text(json.dumps(existing_metadata()), encoding="utf-8")
    monkeypatch.setattr(config, "model_file", lambda: missing_model)
    monkeypatch.setattr(config, "meta_file", lambda: metadata_path)
    monkeypatch.setattr(registry, "_mlflow", lambda: pytest.fail("MLflow should not be contacted"))

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        register_existing.register_existing()


def test_missing_or_invalid_metadata_fails_clearly(monkeypatch, tmp_path):
    model_path = tmp_path / "existing.joblib"
    model_path.write_bytes(b"model")
    metadata_path = tmp_path / "model_meta.json"
    monkeypatch.setattr(config, "model_file", lambda: model_path)
    monkeypatch.setattr(config, "meta_file", lambda: metadata_path)

    with pytest.raises(FileNotFoundError, match="Model metadata file not found"):
        register_existing.register_existing()

    metadata_path.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not read model metadata"):
        register_existing.register_existing()
