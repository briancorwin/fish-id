import sys
import os
import io
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# Mock onnxruntime.InferenceSession before main.py is imported — the session
# is created at module level so the patch must be in place at import time.
_mock_session = MagicMock()
_mock_input = MagicMock()
_mock_input.name = "images"
_mock_input.shape = [1, 3, 640, 640]
_mock_session.get_inputs.return_value = [_mock_input]

with patch("onnxruntime.InferenceSession", return_value=_mock_session):
    import main


NUM_CLASSES = 4  # matches the real model's 4-class output


def make_onnx_output(detections: list[dict]) -> np.ndarray:
    """Build a fake YOLOv8 ONNX output tensor [1, 4+nc, num_anchors].

    Each detection dict: {cx, cy, w, h, conf, class_id (optional, default 0)}
    in INPUT_SIZE pixel space.
    """
    num_anchors = 5376  # 64²+32²+16² for imgsz=512 (mocked as 640 in tests)
    output = np.zeros((1, 4 + NUM_CLASSES, num_anchors), dtype=np.float32)
    for i, d in enumerate(detections):
        output[0, 0, i] = d["cx"]
        output[0, 1, i] = d["cy"]
        output[0, 2, i] = d["w"]
        output[0, 3, i] = d["h"]
        class_id = d.get("class_id", 0)
        output[0, 4 + class_id, i] = d["conf"]
    return output


def make_jpeg(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid JPEG image as bytes."""
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def client():
    main.app.config["TESTING"] = True
    with main.app.test_client() as c:
        yield c


@pytest.fixture
def mock_session():
    return _mock_session


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset per-IP token buckets between tests."""
    from rate_limiter import _limiter
    _limiter._buckets.clear()
    yield
    _limiter._buckets.clear()
