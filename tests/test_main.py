import io
import numpy as np
import pytest
import main
from conftest import make_jpeg, make_onnx_output


# ---------------------------------------------------------------------------
# Unit tests — _is_valid_image
# ---------------------------------------------------------------------------

class TestIsValidImage:
    def test_valid_jpeg(self):
        assert main._is_valid_image(b"\xff\xd8\xff\xe0" + b"\x00" * 20) is True

    def test_valid_png(self):
        assert main._is_valid_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20) is True

    def test_valid_gif87a(self):
        assert main._is_valid_image(b"GIF87a" + b"\x00" * 20) is True

    def test_valid_gif89a(self):
        assert main._is_valid_image(b"GIF89a" + b"\x00" * 20) is True

    def test_valid_webp(self):
        data = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 20
        assert main._is_valid_image(data) is True

    def test_riff_non_webp_rejected(self):
        data = b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 20
        assert main._is_valid_image(data) is False

    def test_random_bytes_rejected(self):
        assert main._is_valid_image(b"\x00\x01\x02\x03" * 10) is False

    def test_empty_bytes_rejected(self):
        assert main._is_valid_image(b"") is False


# ---------------------------------------------------------------------------
# Unit tests — _postprocess
# ---------------------------------------------------------------------------

class TestPostprocess:
    def test_no_detections_above_threshold(self):
        from conftest import NUM_CLASSES
        output = np.zeros((1, 4 + NUM_CLASSES, 5376), dtype=np.float32)
        output[0, 4, 0] = 0.2  # below CONF_THRESHOLD of 0.25
        result = main._postprocess(output, orig_w=640, orig_h=640)
        assert result == []

    def test_single_detection(self):
        output = make_onnx_output([{"cx": 320, "cy": 320, "w": 100, "h": 100, "conf": 0.9}])
        result = main._postprocess(output, orig_w=640, orig_h=640)
        assert len(result) == 1
        assert result[0]["class_id"] == 0
        assert result[0]["confidence"] == pytest.approx(0.9, abs=1e-3)
        box = result[0]["box"]
        assert box["x1"] == 270
        assert box["y1"] == 270
        assert box["x2"] == 370
        assert box["y2"] == 370

    def test_coordinates_scale_to_original_image_size(self):
        # Detection centred at (320, 320) in 640x640 space
        # Original image is 1280x1280 — coordinates should double
        output = make_onnx_output([{"cx": 320, "cy": 320, "w": 100, "h": 100, "conf": 0.9}])
        result = main._postprocess(output, orig_w=1280, orig_h=1280)
        assert len(result) == 1
        box = result[0]["box"]
        assert box["x1"] == 540
        assert box["y1"] == 540
        assert box["x2"] == 740
        assert box["y2"] == 740

    def test_multiple_non_overlapping_detections_all_kept(self):
        output = make_onnx_output([
            {"cx": 100, "cy": 100, "w": 50, "h": 50, "conf": 0.9},
            {"cx": 500, "cy": 500, "w": 50, "h": 50, "conf": 0.8},
        ])
        result = main._postprocess(output, orig_w=640, orig_h=640)
        assert len(result) == 2

    def test_overlapping_detections_nms_keeps_highest_confidence(self):
        # Two nearly identical boxes — NMS should suppress the lower-confidence one
        output = make_onnx_output([
            {"cx": 320, "cy": 320, "w": 100, "h": 100, "conf": 0.9},
            {"cx": 322, "cy": 322, "w": 100, "h": 100, "conf": 0.6},
        ])
        result = main._postprocess(output, orig_w=640, orig_h=640)
        assert len(result) == 1
        assert result[0]["confidence"] == pytest.approx(0.9, abs=1e-3)

    def test_multi_class_correct_class_id_returned(self):
        output = make_onnx_output([
            {"cx": 100, "cy": 100, "w": 50, "h": 50, "conf": 0.9, "class_id": 2},
        ])
        result = main._postprocess(output, orig_w=640, orig_h=640)
        assert len(result) == 1
        assert result[0]["class_id"] == 2

    def test_detection_below_threshold_excluded(self):
        output = make_onnx_output([
            {"cx": 320, "cy": 320, "w": 100, "h": 100, "conf": 0.9},
            {"cx": 100, "cy": 100, "w": 50,  "h": 50,  "conf": 0.2},  # below threshold
        ])
        result = main._postprocess(output, orig_w=640, orig_h=640)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Integration tests — Flask endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "ok"}


class TestDetectEndpoint:
    def test_missing_image_returns_400(self, client):
        resp = client.post("/detect")
        assert resp.status_code == 400
        assert "No image provided" in resp.get_json()["error"]

    def test_oversized_image_returns_400(self, client):
        large = b"\xff\xd8\xff" + b"\x00" * (5 * 1024 * 1024 + 1)
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(large), "big.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "5MB" in resp.get_json()["error"]

    def test_invalid_image_format_returns_400(self, client):
        bad = io.BytesIO(b"\x00\x01\x02\x03" * 100)
        resp = client.post("/detect",
                           data={"image": (bad, "file.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Invalid image format" in resp.get_json()["error"]

    def test_no_detections_returns_empty_list(self, client, mock_session):
        mock_session.run.return_value = [make_onnx_output([])]
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg()), "fish.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["fish_count"] == 0
        assert data["detections"] == []

    def test_detections_returned_correctly(self, client, mock_session):
        mock_session.run.return_value = [make_onnx_output([
            {"cx": 100, "cy": 100, "w": 50, "h": 50, "conf": 0.9},
            {"cx": 400, "cy": 400, "w": 60, "h": 60, "conf": 0.75},
        ])]
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg(640, 640)), "fish.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["fish_count"] == 2
        assert len(data["detections"]) == 2
        for det in data["detections"]:
            assert "class_id" in det
            assert "confidence" in det
            assert {"x1", "y1", "x2", "y2"} == set(det["box"].keys())

    def test_response_shape(self, client, mock_session):
        mock_session.run.return_value = [make_onnx_output([
            {"cx": 320, "cy": 320, "w": 100, "h": 100, "conf": 0.85},
        ])]
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg()), "fish.jpg")},
                           content_type="multipart/form-data")
        data = resp.get_json()
        assert set(data.keys()) == {"fish_count", "detections"}
        assert data["fish_count"] == len(data["detections"])
