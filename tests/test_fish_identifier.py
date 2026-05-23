import logging
import numpy as np
import pytest
from unittest.mock import MagicMock
from fish_identifier import FishIdentifier
from helpers import make_onnx_output

_ORIG_SHAPE = (640, 640, 3)


@pytest.fixture
def fi():
    identifier = object.__new__(FishIdentifier)
    identifier._input_h = 640
    identifier._input_w = 640
    identifier._input_name = "images"
    identifier._class_names = {0: "Largemouth Bass", 1: "Bluegill", 2: "Crappie", 3: "Catfish"}
    identifier._session = MagicMock()
    return identifier


# ---------------------------------------------------------------------------
# Unit tests — FishIdentifier._load_class_names
# ---------------------------------------------------------------------------

class TestLoadClassNames:
    def test_returns_dict_from_valid_json(self, fi):
        session = MagicMock()
        session.get_modelmeta.return_value.custom_metadata_map = {
            "names": '{"0": "Bass", "1": "Bluegill"}'
        }
        assert fi._load_class_names(session) == {0: "Bass", 1: "Bluegill"}

    def test_falls_back_to_ast_on_invalid_json(self, fi):
        session = MagicMock()
        session.get_modelmeta.return_value.custom_metadata_map = {
            "names": "{0: 'Bass', 1: 'Bluegill'}"  # Python literal, not valid JSON
        }
        assert fi._load_class_names(session) == {0: "Bass", 1: "Bluegill"}

    def test_returns_none_and_logs_when_get_modelmeta_raises(self, fi, caplog):
        session = MagicMock()
        session.get_modelmeta.side_effect = RuntimeError("no metadata")
        with caplog.at_level(logging.ERROR, logger="fish_identifier"):
            result = fi._load_class_names(session)
        assert result is None
        assert "Failed to load class names" in caplog.text

    def test_returns_none_and_logs_when_names_missing(self, fi, caplog):
        session = MagicMock()
        session.get_modelmeta.return_value.custom_metadata_map = {}
        with caplog.at_level(logging.ERROR, logger="fish_identifier"):
            result = fi._load_class_names(session)
        assert result is None
        assert "no class names" in caplog.text


# ---------------------------------------------------------------------------
# Unit tests — FishIdentifier._preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_blob_shape(self, fi):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        blob, _, _, _ = fi._preprocess(img)
        assert blob.shape == (1, 3, 640, 640)

    def test_blob_values_normalized(self, fi):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        blob, _, _, _ = fi._preprocess(img)
        assert blob.dtype == np.float32
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0

    def test_square_image_no_padding(self, fi):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        _, scale, pad_left, pad_top = fi._preprocess(img)
        assert scale == 1.0
        assert pad_left == 0
        assert pad_top == 0

    def test_wide_image_pads_top_and_bottom(self, fi):
        # 1280×640 → scale=0.5, resized to 640×320, pad_top=160
        img = np.zeros((640, 1280, 3), dtype=np.uint8)
        _, scale, pad_left, pad_top = fi._preprocess(img)
        assert scale == pytest.approx(0.5)
        assert pad_left == 0
        assert pad_top == 160

    def test_tall_image_pads_left_and_right(self, fi):
        # 640×1280 → scale=0.5, resized to 320×640, pad_left=160
        img = np.zeros((1280, 640, 3), dtype=np.uint8)
        _, scale, pad_left, pad_top = fi._preprocess(img)
        assert scale == pytest.approx(0.5)
        assert pad_left == 160
        assert pad_top == 0


# ---------------------------------------------------------------------------
# Unit tests — FishIdentifier._postprocess
# ---------------------------------------------------------------------------

class TestPostprocess:
    def test_empty_result_returns_empty_list(self, fi):
        output = make_onnx_output([])
        assert fi._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0) == []

    def test_single_detection(self, fi):
        output = make_onnx_output([{"x1": 270, "y1": 270, "x2": 370, "y2": 370, "conf": 0.9, "class_id": 0}])
        detections = fi._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)
        assert len(detections) == 1
        assert detections[0]["class_id"] == 0
        assert detections[0]["class_name"] == "Largemouth Bass"
        assert detections[0]["confidence"] == pytest.approx(0.9, abs=1e-3)
        assert detections[0]["box"] == {"x1": 270, "y1": 270, "x2": 370, "y2": 370}

    def test_multiple_detections(self, fi):
        output = make_onnx_output([
            {"x1": 75, "y1": 75, "x2": 125, "y2": 125, "conf": 0.9, "class_id": 0},
            {"x1": 475, "y1": 475, "x2": 530, "y2": 530, "conf": 0.8, "class_id": 1},
        ])
        assert len(fi._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)) == 2

    def test_overlapping_boxes_nms_keeps_highest_confidence(self, fi):
        # IoU ≈ 0.68 between these two boxes, well above NMS_THRESHOLD — lower-confidence box suppressed
        output = make_onnx_output([
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "conf": 0.9, "class_id": 0},
            {"x1": 110, "y1": 110, "x2": 210, "y2": 210, "conf": 0.7, "class_id": 0},
        ])
        detections = fi._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)
        assert len(detections) == 1
        assert detections[0]["confidence"] == pytest.approx(0.9, abs=1e-3)

    def test_multi_class_correct_class_id_returned(self, fi):
        output = make_onnx_output([
            {"x1": 75, "y1": 75, "x2": 125, "y2": 125, "conf": 0.9, "class_id": 2},
        ])
        detections = fi._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)
        assert detections[0]["class_id"] == 2
        assert detections[0]["class_name"] == "Crappie"

    def test_confidence_rounded_to_4_decimal_places(self, fi):
        output = make_onnx_output([
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "conf": 0.987654321, "class_id": 0},
        ])
        assert fi._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)[0]["confidence"] == 0.9877

    def test_output_keys(self, fi):
        output = make_onnx_output([
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "conf": 0.9, "class_id": 0},
        ])
        det = fi._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)[0]
        assert set(det.keys()) == {"class_id", "class_name", "confidence", "box"}
        assert set(det["box"].keys()) == {"x1", "y1", "x2", "y2"}

    def test_below_threshold_confidence_filtered_out(self, fi):
        output = make_onnx_output([
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "conf": 0.1, "class_id": 0},
        ])
        assert fi._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0) == []

    def test_coordinate_roundtrip_wide_image(self, fi):
        # Simulates a 1280×640 image going through _preprocess: scale=0.5, pad_top=160, pad_left=0
        scale, pad_left, pad_top = 0.5, 0, 160
        orig_shape = (640, 1280, 3)
        output = make_onnx_output(
            [{"x1": 200, "y1": 100, "x2": 400, "y2": 200, "conf": 0.9, "class_id": 0}],
            scale=scale, pad_left=pad_left, pad_top=pad_top,
        )
        det = fi._postprocess(output, orig_shape, scale, pad_left, pad_top)[0]
        assert det["box"] == {"x1": 200, "y1": 100, "x2": 400, "y2": 200}

    def test_coordinate_roundtrip_tall_image(self, fi):
        # Simulates a 640×1280 image going through _preprocess: scale=0.5, pad_left=160, pad_top=0
        scale, pad_left, pad_top = 0.5, 160, 0
        orig_shape = (1280, 640, 3)
        output = make_onnx_output(
            [{"x1": 100, "y1": 200, "x2": 200, "y2": 400, "conf": 0.9, "class_id": 0}],
            scale=scale, pad_left=pad_left, pad_top=pad_top,
        )
        det = fi._postprocess(output, orig_shape, scale, pad_left, pad_top)[0]
        assert det["box"] == {"x1": 100, "y1": 200, "x2": 200, "y2": 400}

    def test_unknown_class_id_raises(self, fi):
        output = make_onnx_output([
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "conf": 0.9, "class_id": 99},
        ])
        with pytest.raises(ValueError, match="class_id 99"):
            fi._postprocess(output, _ORIG_SHAPE, 1.0, 0, 0)
