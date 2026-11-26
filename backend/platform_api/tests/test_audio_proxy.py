from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from backend.platform_api.main import create_app


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeAsyncClient:
    response = None
    error = None
    last_request = None
    last_timeout = None

    def __init__(self, *args, **kwargs):
        type(self).last_timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, *, headers, files):
        type(self).last_request = {
            "url": url,
            "headers": headers,
            "files": files,
        }
        if type(self).error:
            raise type(self).error
        return type(self).response


def reset_fake():
    FakeAsyncClient.response = None
    FakeAsyncClient.error = None
    FakeAsyncClient.last_request = None
    FakeAsyncClient.last_timeout = None


def test_audio_proxy_forwards_authenticated_upload(monkeypatch):
    from backend.platform_api.routers import audio

    reset_fake()
    expected = {
        "filename": "sample.ogg",
        "category": "Animal",
        "sound_group": "Animals",
        "confidence": 0.91,
        "best_guess": "Dog",
        "top_k": [{"category": "Dog", "confidence": 0.91}],
        "reason": "Test prediction",
        "source": "yamnet",
        "model_version": "test-model",
        "duration_seconds": 5.0,
        "sample_rate": 16000,
        "channels": 1,
    }

    monkeypatch.setattr(audio.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("AUDIO_SERVICE_URL", "http://audio-test:8001")
    monkeypatch.setenv("AUDIO_SERVICE_TOKEN", "test-secret-token")
    FakeAsyncClient.response = FakeResponse(200, expected)

    response = TestClient(create_app()).post(
        "/classify/audio",
        files={"file": ("sample.ogg", b"audio-bytes", "audio/ogg")},
    )

    assert response.status_code == 200
    assert response.json() == expected
    assert FakeAsyncClient.last_request["url"] == (
        "http://audio-test:8001/internal/v1/classify"
    )
    assert FakeAsyncClient.last_request["headers"] == {
        "X-Audio-Service-Token": "test-secret-token"
    }
    assert FakeAsyncClient.last_request["files"]["file"] == (
        "sample.ogg",
        b"audio-bytes",
        "audio/ogg",
    )
    assert FakeAsyncClient.last_timeout.connect == 10.0
    assert FakeAsyncClient.last_timeout.read == 150.0


def test_audio_proxy_rejects_missing_service_token(monkeypatch):
    from backend.platform_api.routers import audio

    reset_fake()
    monkeypatch.delenv("AUDIO_SERVICE_TOKEN", raising=False)
    monkeypatch.setattr(audio.httpx, "AsyncClient", FakeAsyncClient)

    response = TestClient(create_app()).post(
        "/classify/audio",
        files={"file": ("sample.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AUDIO_SERVICE_NOT_CONFIGURED"
    assert FakeAsyncClient.last_request is None


def test_audio_proxy_normalizes_upstream_error(monkeypatch):
    from backend.platform_api.routers import audio

    reset_fake()
    monkeypatch.setenv("AUDIO_SERVICE_TOKEN", "test-secret-token")
    monkeypatch.setattr(audio.httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.response = FakeResponse(
        415,
        {"error": {"code": "unsupported_media", "message": "Unsupported audio format"}},
    )

    response = TestClient(create_app()).post(
        "/classify/audio",
        files={"file": ("sample.bin", b"data", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
    assert response.json()["error"]["message"] == "Unsupported audio format"
    assert "details" in response.json()["error"]


def test_audio_proxy_maps_internal_auth_failure_without_leaking_token(monkeypatch):
    from backend.platform_api.routers import audio

    reset_fake()
    secret = "do-not-expose-this-token"
    monkeypatch.setenv("AUDIO_SERVICE_TOKEN", secret)
    monkeypatch.setattr(audio.httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.response = FakeResponse(
        401,
        {"error": {"code": "unauthorized", "message": "Authentication is required."}},
    )

    response = TestClient(create_app()).post(
        "/classify/audio",
        files={"file": ("sample.wav", b"data", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AUDIO_SERVICE_AUTH_FAILED"
    assert secret not in response.text


def test_audio_proxy_rejects_oversized_upload(monkeypatch):
    from backend.platform_api.routers import audio

    reset_fake()
    monkeypatch.setenv("AUDIO_SERVICE_TOKEN", "test-secret-token")
    monkeypatch.setattr(audio, "_max_upload_bytes", lambda: 4)
    monkeypatch.setattr(audio.httpx, "AsyncClient", FakeAsyncClient)

    response = TestClient(create_app()).post(
        "/classify/audio",
        files={"file": ("sample.wav", b"12345", "audio/wav")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert FakeAsyncClient.last_request is None


def test_audio_proxy_reports_timeout(monkeypatch):
    from backend.platform_api.routers import audio

    reset_fake()
    monkeypatch.setenv("AUDIO_SERVICE_TOKEN", "test-secret-token")
    monkeypatch.setattr(audio.httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.error = httpx.ReadTimeout("upstream timed out")

    response = TestClient(create_app()).post(
        "/classify/audio",
        files={"file": ("sample.wav", b"data", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AUDIO_SERVICE_TIMEOUT"


def test_audio_proxy_reports_unavailable_service(monkeypatch):
    from backend.platform_api.routers import audio

    reset_fake()
    monkeypatch.setenv("AUDIO_SERVICE_TOKEN", "test-secret-token")
    monkeypatch.setattr(audio.httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.error = httpx.ConnectError("connection refused")

    response = TestClient(create_app()).post(
        "/classify/audio",
        files={"file": ("sample.wav", b"data", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AUDIO_SERVICE_UNAVAILABLE"


def test_audio_proxy_rejects_invalid_upstream_response(monkeypatch):
    from backend.platform_api.routers import audio

    reset_fake()
    monkeypatch.setenv("AUDIO_SERVICE_TOKEN", "test-secret-token")
    monkeypatch.setattr(audio.httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.response = FakeResponse(200, ValueError("invalid JSON"))

    response = TestClient(create_app()).post(
        "/classify/audio",
        files={"file": ("sample.wav", b"data", "audio/wav")},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "INVALID_AUDIO_RESPONSE"
