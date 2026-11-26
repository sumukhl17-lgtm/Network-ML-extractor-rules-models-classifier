from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from backend.platform_api.main import create_app


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeAsyncClient:
    response = None
    error = None
    last_request = None

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, files):
        type(self).last_request = {"url": url, "files": files}
        if type(self).error:
            raise type(self).error
        return type(self).response


def test_video_proxy_forwards_upload_and_returns_result(monkeypatch):
    from backend.platform_api.routers import video

    expected = {"category": "Animals", "confidence": 0.8}
    monkeypatch.setattr(video.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("VIDEO_SERVICE_URL", "http://video-test:8003")
    FakeAsyncClient.response = FakeResponse(200, expected)
    FakeAsyncClient.error = None

    client = TestClient(create_app())
    response = client.post(
        "/classify/video",
        files={"file": ("sample.mp4", b"video-bytes", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json() == expected
    assert FakeAsyncClient.last_request["url"] == (
        "http://video-test:8003/classify/video"
    )
    assert FakeAsyncClient.last_request["files"]["file"][0] == "sample.mp4"


def test_video_proxy_normalizes_upstream_error(monkeypatch):
    from backend.platform_api.routers import video

    monkeypatch.setattr(video.httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.response = FakeResponse(
        415, {"detail": "Unsupported video extension"}
    )
    FakeAsyncClient.error = None

    response = TestClient(create_app()).post(
        "/classify/video",
        files={"file": ("sample.txt", b"data", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
    assert response.json()["error"]["message"] == "Unsupported video extension"


def test_video_proxy_reports_unavailable_service(monkeypatch):
    from backend.platform_api.routers import video

    monkeypatch.setattr(video.httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.error = httpx.ConnectError("connection refused")

    response = TestClient(create_app()).post(
        "/classify/video",
        files={"file": ("sample.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "VIDEO_SERVICE_UNAVAILABLE"
    FakeAsyncClient.error = None


def test_video_proxy_rejects_oversized_upload(monkeypatch):
    from backend.platform_api.routers import video

    monkeypatch.setattr(video, "_max_upload_bytes", lambda: 4)

    response = TestClient(create_app()).post(
        "/classify/video",
        files={"file": ("sample.mp4", b"12345", "video/mp4")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"
