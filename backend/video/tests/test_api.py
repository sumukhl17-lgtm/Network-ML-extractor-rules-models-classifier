from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.platform_api.main import create_app
from backend.platform_api.routers import video


def test_video_endpoint_returns_classification_contract():
    expected = {
        "filename": "sample.mp4", "category": "Animals", "confidence": 0.8,
        "best_guess": "Animals", "top_k": [], "reason": "ok", "source": "pretrained",
        "model_version": "clip-test", "extractor": "ffmpeg", "file_type": "video/mp4",
        "duration_seconds": 2.0, "frame_count": 1, "sampling_interval_seconds": 2.0,
        "audio_detected": False, "audio_assisted": False, "real_data_validated": False,
    }

    class FakeResponse:
        status_code = 200

        def json(self):
            return expected

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, files):
            assert url.endswith("/classify/video")
            assert files["file"][0] == "sample.mp4"
            return FakeResponse()

    with patch.object(video.httpx, "AsyncClient", FakeAsyncClient):
        response = TestClient(create_app()).post(
            "/classify/video",
            files={"file": ("sample.mp4", b"bytes", "video/mp4")},
        )

    assert response.status_code == 200
    assert response.json() == expected
