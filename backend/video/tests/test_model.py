import pytest

from backend.video.model import classify_frames


def test_empty_frame_list_is_rejected_before_model_loading():
    with pytest.raises(ValueError, match="At least one extracted video frame"):
        classify_frames([])
