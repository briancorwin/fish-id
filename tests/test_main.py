import io
import logging
import numpy as np
import pytest
from unittest.mock import MagicMock
import main
from helpers import make_jpeg, make_onnx_output

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
# Unit tests — _load_class_names
# ---------------------------------------------------------------------------

class TestLoadClassNames:
    def test_returns_dict_from_valid_json(self):
        session = MagicMock()
        session.get_modelmeta.return_value.custom_metadata_map = {
            "names": '{"0": "Bass", "1": "Bluegill"}'
        }
        assert main._load_class_names(session) == {0: "Bass", 1: "Bluegill"}

    def test_falls_back_to_ast_on_invalid_json(self):
        session = MagicMock()
        session.get_modelmeta.return_value.custom_metadata_map = {
            "names": "{0: 'Bass', 1: 'Bluegill'}"  # Python literal, not valid JSON
        }
        assert main._load_class_names(session) == {0: "Bass", 1: "Bluegill"}

    def test_returns_none_and_logs_when_get_modelmeta_raises(self, caplog):
        session = MagicMock()
        session.get_modelmeta.side_effect = RuntimeError("no metadata")
        with caplog.at_level(logging.ERROR, logger="main"):
            result = main._load_class_names(session)
        assert result is None
        assert "Failed to load class names" in caplog.text

    def test_returns_none_and_logs_when_names_missing(self, caplog):
        session = MagicMock()
        session.get_modelmeta.return_value.custom_metadata_map = {}
        with caplog.at_level(logging.ERROR, logger="main"):
            result = main._load_class_names(session)
        assert result is None
        assert "no class names" in caplog.text


# ---------------------------------------------------------------------------
# Unit tests — _preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_blob_shape(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        blob, _, _, _ = main._preprocess(img)
        assert blob.shape == (1, 3, 640, 640)

    def test_blob_values_normalized(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        blob, _, _, _ = main._preprocess(img)
        assert blob.dtype == np.float32
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0

    def test_square_image_no_padding(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        _, scale, pad_left, pad_top = main._preprocess(img)
        assert scale == 1.0
        assert pad_left == 0
        assert pad_top == 0

    def test_wide_image_pads_top_and_bottom(self):
        # 1280×640 → scale=0.5, resized to 640×320, pad_top=160
        img = np.zeros((640, 1280, 3), dtype=np.uint8)
        _, scale, pad_left, pad_top = main._preprocess(img)
        assert scale == pytest.approx(0.5)
        assert pad_left == 0
        assert pad_top == 160

    def test_tall_image_pads_left_and_right(self):
        # 640×1280 → scale=0.5, resized to 320×640, pad_left=160
        img = np.zeros((1280, 640, 3), dtype=np.uint8)
        _, scale, pad_left, pad_top = main._preprocess(img)
        assert scale == pytest.approx(0.5)
        assert pad_left == 160
        assert pad_top == 0


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

    def test_below_threshold_confidence_filtered_out(self):
        output = make_onnx_output([
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "conf": 0.1},
        ])
        assert main._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0) == []

    def test_unknown_class_id_uses_fallback_name(self):
        output = make_onnx_output([
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "conf": 0.9, "class_id": 99},
        ])
        det = main._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)[0]
        assert det["class_id"] == 99
        assert det["class_name"] == "Class 99"


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

    def test_class_names_unavailable_returns_500(self, client):
        original = main._class_names
        try:
            main._class_names = None
            resp = client.post("/detect",
                               data={"image": (io.BytesIO(make_jpeg()), "fish.jpg")},
                               content_type="multipart/form-data")
            assert resp.status_code == 500
            assert "Model not ready" in resp.get_json()["error"]
        finally:
            main._class_names = original

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
