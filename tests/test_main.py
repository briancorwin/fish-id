import io
import cv2
import pytest
from unittest.mock import patch
import main
from helpers import make_jpeg, make_onnx_output


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
# Integration tests — Flask endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "ok"}


class TestDetectEndpoint:
    def test_no_detections_returns_empty_list(self, client, mock_session):
        mock_session.run.return_value = [make_onnx_output([])]
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg(640, 640)), "fish.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["fish_count"] == 0
        assert data["detections"] == []

    def test_detections_returned_correctly(self, client, mock_session):
        mock_session.run.return_value = [make_onnx_output([
            {"x1": 75,  "y1": 75,  "x2": 125, "y2": 125, "conf": 0.9,  "class_id": 0},
            {"x1": 370, "y1": 370, "x2": 430, "y2": 430, "conf": 0.75, "class_id": 1},
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
            assert "class_name" in det
            assert "confidence" in det
            assert {"x1", "y1", "x2", "y2"} == set(det["box"].keys())

    def test_wide_image_coordinates_roundtrip(self, client, mock_session):
        # 1280×640 image: _preprocess produces scale=0.5, pad_top=160, pad_left=0
        scale, pad_left, pad_top = 0.5, 0, 160
        mock_session.run.return_value = [make_onnx_output([
            {"x1": 200, "y1": 100, "x2": 400, "y2": 200, "conf": 0.9, "class_id": 0},
        ], scale=scale, pad_left=pad_left, pad_top=pad_top)]
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg(1280, 640)), "fish.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.get_json()["detections"][0]["box"] == {"x1": 200, "y1": 100, "x2": 400, "y2": 200}

    def test_tall_image_coordinates_roundtrip(self, client, mock_session):
        # 640×1280 image: _preprocess produces scale=0.5, pad_left=160, pad_top=0
        scale, pad_left, pad_top = 0.5, 160, 0
        mock_session.run.return_value = [make_onnx_output([
            {"x1": 100, "y1": 200, "x2": 200, "y2": 400, "conf": 0.9, "class_id": 0},
        ], scale=scale, pad_left=pad_left, pad_top=pad_top)]
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg(640, 1280)), "fish.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.get_json()["detections"][0]["box"] == {"x1": 100, "y1": 200, "x2": 200, "y2": 400}

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

    def test_corrupt_image_returns_400(self, client):
        # Valid JPEG magic bytes but truncated body — passes _is_valid_image, fails cv2.imdecode
        corrupt = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        resp = client.post("/detect",
                           data={"image": (corrupt, "corrupt.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Could not decode image" in resp.get_json()["error"]

    def test_unknown_class_id_returns_500(self, client, mock_session):
        mock_session.run.return_value = [make_onnx_output([
            {"x1": 75, "y1": 75, "x2": 125, "y2": 125, "conf": 0.9, "class_id": 99},
        ])]
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg(640, 640)), "fish.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 500
        assert "Internal model error" in resp.get_json()["error"]

    def test_cv2_error_in_detect_returns_500(self, client):
        with patch.object(main._identifier, 'detect', side_effect=cv2.error()):
            resp = client.post("/detect",
                               data={"image": (io.BytesIO(make_jpeg(640, 640)), "fish.jpg")},
                               content_type="multipart/form-data")
        assert resp.status_code == 500
        assert "Internal model error" in resp.get_json()["error"]

    def test_runtime_error_in_detect_returns_500(self, client, mock_session):
        mock_session.run.side_effect = RuntimeError("ONNX inference error")
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg(640, 640)), "fish.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 500
        assert "Internal model error" in resp.get_json()["error"]

    def test_class_names_unavailable_returns_500(self, client):
        original = main._identifier._class_names
        try:
            main._identifier._class_names = None
            resp = client.post("/detect",
                               data={"image": (io.BytesIO(make_jpeg()), "fish.jpg")},
                               content_type="multipart/form-data")
            assert resp.status_code == 500
            assert "Model not ready" in resp.get_json()["error"]
        finally:
            main._identifier._class_names = original


class TestClassNamesEndpoint:
    def test_returns_all_class_names(self, client):
        resp = client.get("/class-names")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["class_names"] == {
            "0": "Largemouth Bass",
            "1": "Bluegill",
            "2": "Crappie",
            "3": "Catfish",
        }

    def test_keys_are_strings(self, client):
        resp = client.get("/class-names")
        for key in resp.get_json()["class_names"]:
            assert isinstance(key, str)

    def test_model_not_ready_returns_500(self, client):
        original = main._identifier._class_names
        try:
            main._identifier._class_names = None
            resp = client.get("/class-names")
            assert resp.status_code == 500
            assert "Model not ready" in resp.get_json()["error"]
        finally:
            main._identifier._class_names = original

    def test_model_load_failure_returns_500_on_detect(self, client):
        original = main._identifier
        try:
            main._identifier = None
            with patch('main.FishIdentifier', side_effect=RuntimeError("model file missing")):
                resp = client.post("/detect",
                                   data={"image": (io.BytesIO(make_jpeg()), "fish.jpg")},
                                   content_type="multipart/form-data")
            assert resp.status_code == 500
            assert "Model not ready" in resp.get_json()["error"]
        finally:
            main._identifier = original

    def test_model_load_failure_returns_500_on_class_names(self, client):
        original = main._identifier
        try:
            main._identifier = None
            with patch('main.FishIdentifier', side_effect=RuntimeError("model file missing")):
                resp = client.get("/class-names")
            assert resp.status_code == 500
            assert "Model not ready" in resp.get_json()["error"]
        finally:
            main._identifier = original
