from types import SimpleNamespace
import sys

import pytest

from backend.video import registry


def test_registry_requires_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(registry.RegistryUnavailable, match="MLFLOW_TRACKING_URI"):
        registry.load_champion_config()


def test_registry_reads_champion_model_id_and_version(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test")

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["tracking_uri"] == "http://mlflow.test"

        def get_model_version_by_alias(self, name, alias):
            assert name == "video-classifier"
            assert alias == "champion"
            return SimpleNamespace(version="7", tags={"video_model_id": "org/clip-demo", "video_prompt_set": "v1"})

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(MlflowClient=FakeClient))
    result = registry.load_champion_config()
    assert result == {"model_id": "org/clip-demo", "version": "7", "source": "registry", "prompt_set": "v1"}


def test_registry_rejects_champion_without_model_id(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test")

    class FakeClient:
        def __init__(self, **kwargs): pass
        def get_model_version_by_alias(self, name, alias):
            return SimpleNamespace(version="2", tags={})

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(MlflowClient=FakeClient))
    with pytest.raises(registry.RegistryUnavailable, match="video_model_id"):
        registry.load_champion_config()
