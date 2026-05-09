import sys
import os
import io
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# Mock ultralytics.YOLO before main.py is imported — the model is loaded at
# module level so the patch must be in place at import time.
_mock_model = MagicMock()

with patch("ultralytics.YOLO", return_value=_mock_model):
    import main


def make_yolo_result(detections: list[dict]):
    """Build a fake Ultralytics Results object.

    Each detection dict: {x1, y1, x2, y2, conf, class_id (optional, default 0)}
    in original image pixel space.
    """
    result = MagicMock()
    boxes = MagicMock()
    boxes.__len__ = MagicMock(return_value=len(detections))
    boxes.xyxy.tolist.return_value = [
        [d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections
    ]
    boxes.conf.tolist.return_value = [d["conf"] for d in detections]
    boxes.cls.tolist.return_value = [float(d.get("class_id", 0)) for d in detections]
    result.boxes = boxes
    return result


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
def mock_model():
    return _mock_model


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset per-IP token buckets between tests."""
    from rate_limiter import _limiter
    _limiter._buckets.clear()
    yield
    _limiter._buckets.clear()
