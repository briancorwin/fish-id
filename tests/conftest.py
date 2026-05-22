import sys
import os
import io
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

NUM_CLASSES = 4

# Mock onnxruntime.InferenceSession before main.py is imported — the session is
# created at module level so the patch must be in place at import time.
_mock_input = MagicMock()
_mock_input.name = "images"
_mock_input.shape = [1, 3, 640, 640]  # (batch, C, H, W) — tests use 640x640 jpegs

_mock_meta = MagicMock()
_mock_meta.custom_metadata_map = {
    "names": '{"0": "Largemouth Bass", "1": "Bluegill", "2": "Crappie", "3": "Catfish"}'
}

_mock_session = MagicMock()
_mock_session.get_inputs.return_value = [_mock_input]
_mock_session.get_modelmeta.return_value = _mock_meta

with patch("onnxruntime.InferenceSession", return_value=_mock_session):
    import main


def make_onnx_output(
    detections: list[dict],
    scale: float = 1.0,
    pad_left: int = 0,
    pad_top: int = 0,
) -> np.ndarray:
    """Build a fake ONNX raw output array.

    Each detection dict: {x1, y1, x2, y2, conf, class_id (optional, default 0)}
    in original image pixel space. Returns shape (1, 4 + NUM_CLASSES, num_anchors).
    """
    num_anchors = max(len(detections), 1)
    output = np.zeros((1, 4 + NUM_CLASSES, num_anchors), dtype=np.float32)

    for i, d in enumerate(detections):
        cx = (d["x1"] + d["x2"]) / 2 * scale + pad_left
        cy = (d["y1"] + d["y2"]) / 2 * scale + pad_top
        bw = (d["x2"] - d["x1"]) * scale
        bh = (d["y2"] - d["y1"]) * scale
        output[0, 0, i] = cx
        output[0, 1, i] = cy
        output[0, 2, i] = bw
        output[0, 3, i] = bh
        class_id = int(d.get("class_id", 0))
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
