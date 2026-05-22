import io
import pytest
import main
from conftest import make_jpeg, make_onnx_output

_ORIG_SHAPE = (640, 640, 3)


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
    def test_empty_result_returns_empty_list(self):
        output = make_onnx_output([])
        assert main._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0) == []

    def test_single_detection(self):
        output = make_onnx_output([{"x1": 270, "y1": 270, "x2": 370, "y2": 370, "conf": 0.9}])
        detections = main._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)
        assert len(detections) == 1
        assert detections[0]["class_id"] == 0
        assert detections[0]["class_name"] == "Largemouth Bass"
        assert detections[0]["confidence"] == pytest.approx(0.9, abs=1e-3)
        assert detections[0]["box"] == {"x1": 270, "y1": 270, "x2": 370, "y2": 370}

    def test_multiple_detections(self):
        output = make_onnx_output([
            {"x1": 75, "y1": 75, "x2": 125, "y2": 125, "conf": 0.9},
            {"x1": 475, "y1": 475, "x2": 530, "y2": 530, "conf": 0.8},
        ])
        assert len(main._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)) == 2

    def test_multi_class_correct_class_id_returned(self):
        output = make_onnx_output([
            {"x1": 75, "y1": 75, "x2": 125, "y2": 125, "conf": 0.9, "class_id": 2},
        ])
        detections = main._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)
        assert detections[0]["class_id"] == 2
        assert detections[0]["class_name"] == "Crappie"

    def test_confidence_rounded_to_4_decimal_places(self):
        output = make_onnx_output([
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "conf": 0.987654321},
        ])
        assert main._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)[0]["confidence"] == 0.9877

    def test_output_keys(self):
        output = make_onnx_output([
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "conf": 0.9},
        ])
        det = main._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)[0]
        assert set(det.keys()) == {"class_id", "class_name", "confidence", "box"}
        assert set(det["box"].keys()) == {"x1", "y1", "x2", "y2"}


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
                           data={"image": (io.BytesIO(make_jpeg(640, 640)), "fish.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["fish_count"] == 0
        assert data["detections"] == []

    def test_detections_returned_correctly(self, client, mock_session):
        mock_session.run.return_value = [make_onnx_output([
            {"x1": 75,  "y1": 75,  "x2": 125, "y2": 125, "conf": 0.9},
            {"x1": 370, "y1": 370, "x2": 430, "y2": 430, "conf": 0.75},
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

    def test_response_shape(self, client, mock_session):
        mock_session.run.return_value = [make_onnx_output([
            {"x1": 270, "y1": 270, "x2": 370, "y2": 370, "conf": 0.85},
        ])]
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg(640, 640)), "fish.jpg")},
                           content_type="multipart/form-data")
        data = resp.get_json()
        assert set(data.keys()) == {"fish_count", "detections"}
        assert data["fish_count"] == len(data["detections"])
