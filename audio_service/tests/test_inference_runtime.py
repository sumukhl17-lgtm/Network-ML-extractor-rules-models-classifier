
from types import SimpleNamespace

import pytest

from audio_service import inference_runtime
from audio_service.config import Settings
from audio_service.persistent_runtime import PersistentProcessSupervisor
from audio_service.service import AudioServiceError


def test_initialize_inference_loads_both_models_and_removes_service_token(
    monkeypatch,
):
    settings = Settings(service_token="test-secret", model_source="registry")
    classifier = object()
    metadata = {"test": "metadata"}
    loaded = {}

    class FakeEmbedder:
        def load(self):
            loaded["yamnet"] = True

    def fake_load_champion():
        loaded["classifier"] = True
        return classifier, metadata, "42"

    monkeypatch.setattr(inference_runtime, "load_champion", fake_load_champion)
    monkeypatch.setattr(inference_runtime, "validate_classifier", lambda model: ())
    monkeypatch.setattr(inference_runtime, "YAMNetEmbedder", FakeEmbedder)

    state = inference_runtime.initialize_inference(settings)

    assert loaded == {"classifier": True, "yamnet": True}
    assert state.classifier is classifier
    assert state.metadata is metadata
    assert state.model_version == "42"
    assert state.settings.service_token is None
    assert state.settings.max_upload_bytes == settings.max_upload_bytes


def test_initialize_inference_rejects_invalid_registry_metadata(monkeypatch):
    monkeypatch.setattr(
        inference_runtime,
        "load_champion",
        lambda: (object(), [], "42"),
    )

    with pytest.raises(AudioServiceError) as raised:
        inference_runtime.initialize_inference(Settings(model_source="registry"))

    assert raised.value.code == "model_unavailable"


def test_handle_inference_runs_complete_pipeline(monkeypatch):
    settings = Settings()
    waveform = object()
    embedding = object()
    decoded = SimpleNamespace(duration_seconds=2.5)
    classifier = object()
    calls = {}

    class FakeEmbedder:
        def embed(self, received_waveform):
            calls["waveform"] = received_waveform
            return SimpleNamespace(embedding=embedding)

    def fake_decode(audio_bytes, received_settings):
        calls["decode"] = (audio_bytes, received_settings)
        return decoded

    def fake_preprocess(received_decoded, received_settings):
        calls["preprocess"] = (received_decoded, received_settings)
        return waveform

    def fake_classify(**kwargs):
        calls["classify"] = kwargs
        return SimpleNamespace(
            model_dump=lambda mode: {
                "filename": kwargs["filename"],
                "category": "dog",
                "model_version": kwargs["model_version"],
            }
        )

    monkeypatch.setattr(inference_runtime, "decode_audio", fake_decode)
    monkeypatch.setattr(inference_runtime, "preprocess_audio", fake_preprocess)
    monkeypatch.setattr(inference_runtime, "classify_embedding", fake_classify)

    state = inference_runtime.InferenceState(
        settings=settings,
        embedder=FakeEmbedder(),
        classifier=classifier,
        metadata={},
        model_version="42",
    )

    result = inference_runtime.handle_inference(
        {
            "audio_bytes": b"fake-audio",
            "filename": "sample.wav",
            "content_type": "audio/wav",
        },
        state,
    )

    assert result == {
        "filename": "sample.wav",
        "category": "dog",
        "model_version": "42",
    }
    assert calls["decode"] == (b"fake-audio", settings)
    assert calls["preprocess"] == (decoded, settings)
    assert calls["waveform"] is waveform
    assert calls["classify"]["model"] is classifier
    assert calls["classify"]["embedding"] is embedding
    assert calls["classify"]["yamnet_output"].embedding is embedding
    assert calls["classify"]["filename"] == "sample.wav"
    assert calls["classify"]["duration_seconds"] == 2.5
    assert calls["classify"]["model_version"] == "42"


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (None, "invalid_request"),
        ({}, "invalid_request"),
        (
            {"audio_bytes": b"audio", "filename": "sample.wav", "extra": True},
            "invalid_request",
        ),
        ({"audio_bytes": b"", "filename": "sample.wav"}, "invalid_audio"),
        ({"audio_bytes": "not-bytes", "filename": "sample.wav"}, "invalid_audio"),
        ({"audio_bytes": b"audio", "filename": " "}, "invalid_filename"),
        (
            {
                "audio_bytes": b"audio",
                "filename": "sample.wav",
                "content_type": 123,
            },
            "invalid_content_type",
        ),
    ],
)
def test_handle_inference_rejects_invalid_payloads(payload, expected_code):
    state = inference_runtime.InferenceState(
        settings=Settings(),
        embedder=object(),
        classifier=object(),
        metadata={},
        model_version="42",
    )

    with pytest.raises(AudioServiceError) as raised:
        inference_runtime.handle_inference(payload, state)

    assert raised.value.code == expected_code


def test_handle_inference_rejects_upload_over_size_limit():
    state = inference_runtime.InferenceState(
        settings=Settings(max_upload_bytes=3),
        embedder=object(),
        classifier=object(),
        metadata={},
        model_version="42",
    )

    with pytest.raises(AudioServiceError) as raised:
        inference_runtime.handle_inference(
            {"audio_bytes": b"four", "filename": "sample.wav"},
            state,
        )

    assert raised.value.status_code == 413
    assert raised.value.code == "file_too_large"


def test_create_inference_supervisor_removes_service_token(monkeypatch):
    settings = Settings(service_token="test-secret")
    captured = {}

    class FakeSupervisor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        inference_runtime,
        "PersistentProcessSupervisor",
        FakeSupervisor,
    )

    supervisor = inference_runtime.create_inference_supervisor(settings)

    assert isinstance(supervisor, FakeSupervisor)
    assert captured["initializer"] is inference_runtime.initialize_inference
    assert captured["handler"] is inference_runtime.handle_inference
    assert captured["initializer_args"][0].service_token is None
    assert settings.service_token == "test-secret"
