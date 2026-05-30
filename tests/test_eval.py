"""Unit tests for training/eval.py.

All external dependencies (GCS, YOLO/ultralytics, Vertex AI) are mocked.
No real infrastructure is required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

TRAINING_DIR = Path(__file__).parent.parent / "training"
sys.path.insert(0, str(TRAINING_DIR))

# Stub heavy deps that are not installed in the test environment
import google as _google_pkg
if not hasattr(_google_pkg, "cloud"):
    _gc_mock = MagicMock()
    _google_pkg.cloud = _gc_mock
    sys.modules["google.cloud"] = _gc_mock
    sys.modules["google.cloud.storage"] = MagicMock()
    sys.modules["google.cloud.aiplatform"] = MagicMock()
if "ultralytics" not in sys.modules:
    sys.modules["ultralytics"] = MagicMock()

import eval as eval_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EVAL_MANIFEST_FIXTURE = {
    "eval_files": ["fish001.jpg", "fish002.jpg"],
    "label_files": ["fish001.txt", "fish002.txt"],
    "class_names": ["Bass", "Bluegill"],
}

SAMPLE_METRICS = {
    "mAP50": 0.82,
    "mAP50_95": 0.61,
    "precision": 0.78,
    "recall": 0.75,
    "per_class_map50": [0.82, 0.81],
}


def _make_gcs_client(response_bytes):
    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = response_bytes
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return mock_client, mock_bucket, mock_blob


def _make_gcs_upload_client():
    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return mock_client, mock_bucket, mock_blob


# ---------------------------------------------------------------------------
# Test 1: Eval set download
# ---------------------------------------------------------------------------

class TestEvalSetDownload:
    """eval.py fetches eval/current.json and manifest from correct GCS paths."""

    def test_current_json_fetched_from_correct_path(self):
        payload = json.dumps({"eval_version": "ev1"}).encode()
        mock_client, mock_bucket, _ = _make_gcs_client(payload)

        result = eval_module._download_eval_current(mock_client, "my-training-bucket")

        mock_bucket.blob.assert_called_once_with("eval/current.json")
        assert result == "ev1"

    def test_current_json_uses_correct_bucket(self):
        payload = json.dumps({"eval_version": "ev1"}).encode()
        mock_client, _, _ = _make_gcs_client(payload)

        eval_module._download_eval_current(mock_client, "my-training-bucket")

        mock_client.bucket.assert_called_once_with("my-training-bucket")

    def test_manifest_fetched_from_correct_path(self):
        payload = json.dumps(EVAL_MANIFEST_FIXTURE).encode()
        mock_client, mock_bucket, _ = _make_gcs_client(payload)

        eval_module._download_eval_manifest(mock_client, "my-training-bucket", "ev1")

        mock_bucket.blob.assert_called_once_with("eval/versions/ev1/manifest.json")

    def test_manifest_returns_parsed_json(self):
        payload = json.dumps(EVAL_MANIFEST_FIXTURE).encode()
        mock_client, _, _ = _make_gcs_client(payload)

        result = eval_module._download_eval_manifest(mock_client, "my-training-bucket", "ev1")

        assert result["class_names"] == ["Bass", "Bluegill"]
        assert result["eval_files"] == ["fish001.jpg", "fish002.jpg"]

    def test_eval_image_files_downloaded_to_correct_paths(self):
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("pathlib.Path.mkdir"):
            eval_module._download_eval_files(mock_client, "my-training-bucket", EVAL_MANIFEST_FIXTURE)

        blob_calls = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert "eval/images/fish001.jpg" in blob_calls
        assert "eval/images/fish002.jpg" in blob_calls

    def test_label_files_downloaded_to_correct_paths(self):
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("pathlib.Path.mkdir"):
            eval_module._download_eval_files(mock_client, "my-training-bucket", EVAL_MANIFEST_FIXTURE)

        blob_calls = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert "eval/labels/fish001.txt" in blob_calls
        assert "eval/labels/fish002.txt" in blob_calls

    def test_falls_back_to_train_files_when_eval_files_absent(self):
        manifest_no_eval = {"train_files": ["img001.jpg"], "class_names": ["Bass"]}
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("pathlib.Path.mkdir"):
            image_files = eval_module._download_eval_files(
                mock_client, "my-training-bucket", manifest_no_eval
            )

        assert image_files == ["img001.jpg"]
        blob_calls = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert "eval/images/img001.jpg" in blob_calls

    def test_model_downloaded_from_correct_gcs_path(self):
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        eval_module._download_model(mock_client, "my-model-bucket", "run-abc")

        mock_client.bucket.assert_called_once_with("my-model-bucket")
        mock_bucket.blob.assert_called_once_with("runs/run-abc/fish-id.onnx")
        mock_blob.download_to_filename.assert_called_once_with("/tmp/fish-id.onnx")


# ---------------------------------------------------------------------------
# Test 2: Metrics extraction
# ---------------------------------------------------------------------------

class TestMetricsExtraction:
    """_extract_metrics returns correctly keyed float metrics from YOLO results."""

    def _make_mock_results(self, map50=0.82, map=0.61, mp=0.78, mr=0.75):
        mock_box = MagicMock()
        mock_box.map50 = map50
        mock_box.map = map
        mock_box.mp = mp
        mock_box.mr = mr
        mock_box.ap50 = MagicMock()
        mock_box.ap50.tolist.return_value = [map50, map50 - 0.01]
        mock_results = MagicMock()
        mock_results.box = mock_box
        return mock_results

    def test_required_keys_present(self):
        metrics = eval_module._extract_metrics(self._make_mock_results())
        assert set(metrics.keys()) == {"mAP50", "mAP50_95", "precision", "recall", "per_class_map50"}

    def test_map50_value_and_type(self):
        metrics = eval_module._extract_metrics(self._make_mock_results(map50=0.82))
        assert isinstance(metrics["mAP50"], float)
        assert abs(metrics["mAP50"] - 0.82) < 1e-6

    def test_map50_95_value_and_type(self):
        metrics = eval_module._extract_metrics(self._make_mock_results(map=0.61))
        assert isinstance(metrics["mAP50_95"], float)
        assert abs(metrics["mAP50_95"] - 0.61) < 1e-6

    def test_precision_and_recall(self):
        metrics = eval_module._extract_metrics(self._make_mock_results(mp=0.78, mr=0.75))
        assert abs(metrics["precision"] - 0.78) < 1e-6
        assert abs(metrics["recall"] - 0.75) < 1e-6

    def test_per_class_map50_is_list(self):
        metrics = eval_module._extract_metrics(self._make_mock_results())
        assert isinstance(metrics["per_class_map50"], list)


# ---------------------------------------------------------------------------
# Test 3: Eval result upload
# ---------------------------------------------------------------------------

class TestEvalResultUpload:
    """_upload_eval_results writes the correct payload to the correct GCS path."""

    def test_uploaded_to_correct_gcs_path(self):
        mock_client, mock_bucket, _ = _make_gcs_upload_client()

        with patch("builtins.open", create=True) as mock_open:
            import io
            mock_open.return_value.__enter__ = lambda s: io.StringIO()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            eval_module._upload_eval_results(
                mock_client, "my-model-bucket", "run-xyz", SAMPLE_METRICS, "ev1"
            )

        blob_paths = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert any("runs/run-xyz/eval_results.json" in p for p in blob_paths)

    def test_payload_contains_required_fields(self):
        mock_client, _, _ = _make_gcs_upload_client()

        with patch("builtins.open", create=True) as mock_open:
            import io
            mock_open.return_value.__enter__ = lambda s: io.StringIO()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            payload = eval_module._upload_eval_results(
                mock_client, "my-model-bucket", "run-xyz", SAMPLE_METRICS, "ev1"
            )

        assert payload["run_id"] == "run-xyz"
        assert payload["eval_version"] == "ev1"
        assert payload["mAP50"] == 0.82
        assert "scored_at" in payload

    def test_upload_from_filename_called_with_correct_path(self):
        mock_client, _, mock_blob = _make_gcs_upload_client()

        with patch("builtins.open", create=True) as mock_open:
            import io
            mock_open.return_value.__enter__ = lambda s: io.StringIO()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            eval_module._upload_eval_results(
                mock_client, "my-model-bucket", "run-xyz", SAMPLE_METRICS, "ev1"
            )

        mock_blob.upload_from_filename.assert_called_once_with("/tmp/eval_results.json")

    def test_payload_includes_all_metrics(self):
        mock_client, _, _ = _make_gcs_upload_client()

        with patch("builtins.open", create=True) as mock_open:
            import io
            mock_open.return_value.__enter__ = lambda s: io.StringIO()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            payload = eval_module._upload_eval_results(
                mock_client, "my-model-bucket", "run-xyz", SAMPLE_METRICS, "ev1"
            )

        for key in SAMPLE_METRICS:
            assert key in payload
