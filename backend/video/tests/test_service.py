from unittest.mock import patch
import pytest
from backend.video.extract import VideoFrames
from backend.video.service import VideoServiceError, classify_video


def test_empty_upload_rejected():
    with pytest.raises(VideoServiceError) as exc:
        classify_video(b"", "clip.mp4")
    assert exc.value.status == 400


def test_unsupported_extension_rejected():
    with pytest.raises(VideoServiceError) as exc:
        classify_video(b"not empty", "clip.txt")
    assert exc.value.status == 415


def test_service_returns_video_metadata_and_visual_prediction():
    frames = VideoFrames([b"frame1", b"frame2"], 4.5, 2, 2.0, has_audio=True)
    prediction = {
        "category": "Animals", "confidence": 0.73, "best_guess": "Animals",
        "top_k": [{"category": "Animals", "confidence": 0.73}],
        "reason": "ok", "source": "pretrained", "model_version": "clip-test",
    }
    with patch("backend.video.service.extract_frames", return_value=frames), patch(
        "backend.video.service.classify_frames", return_value=prediction
    ):
        result = classify_video(b"video-bytes", "wildlife.mp4")
    assert result["category"] == "Animals"
    assert result["frame_count"] == 2
    assert result["audio_detected"] is True
    assert result["audio_assisted"] is False
    assert result["real_data_validated"] is False


def test_model_failure_is_service_unavailable():
    frames = VideoFrames([b"frame"], 1.0, 1, 2.0)
    with patch("backend.video.service.extract_frames", return_value=frames), patch(
        "backend.video.service.classify_frames", side_effect=RuntimeError("optional deps missing")
    ):
        with pytest.raises(VideoServiceError) as exc:
            classify_video(b"video-bytes", "clip.mp4")
    assert exc.value.status == 503


def test_oversized_upload_rejected(monkeypatch):
    from backend.video import config

    monkeypatch.setattr(config, "max_upload_bytes", lambda: 4)
    with pytest.raises(VideoServiceError) as exc:
        classify_video(b"12345", "clip.mp4")
    assert exc.value.status == 413


def test_corrupt_or_undecodable_video_maps_to_unprocessable(monkeypatch):
    from backend.video.extract import VideoExtractionError

    with patch("backend.video.service.extract_frames", side_effect=VideoExtractionError("Video decoding failed")):
        with pytest.raises(VideoServiceError) as exc:
            classify_video(b"not-a-real-video", "clip.mp4")
    assert exc.value.status == 422


def test_os_level_model_download_failure_maps_to_service_unavailable():
    frames = VideoFrames([b"frame"], 1.0, 1, 2.0)
    with patch("backend.video.service.extract_frames", return_value=frames), patch(
        "backend.video.service.classify_frames", side_effect=OSError("model download failed")
    ):
        with pytest.raises(VideoServiceError) as exc:
            classify_video(b"video-bytes", "clip.mp4")
    assert exc.value.status == 503
