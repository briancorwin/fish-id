"""Unit tests for training/eval.py.

All external dependencies (GCS, YOLO/ultralytics, Vertex AI) are mocked.
No real infrastructure is required.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

TRAINING_DIR = Path(__file__).parent.parent / "training"
sys.path.insert(0, str(TRAINING_DIR))

# Stub heavy deps not installed in the test environment
import google as _google_pkg  # noqa: E402

if not hasattr(_google_pkg, "cloud"):
    _gc_mock = MagicMock()
    _google_pkg.cloud = _gc_mock
    sys.modules["google.cloud"] = _gc_mock
    sys.modules["google.cloud.storage"] = MagicMock()
    sys.modules["google.cloud.aiplatform"] = MagicMock()
if "ultralytics" not in sys.modules:
    sys.modules["ultralytics"] = MagicMock()

import eval as eval_module  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_METRICS = {
    "mAP50": 0.82,
    "mAP50_95": 0.61,
    "precision": 0.78,
    "recall": 0.75,
    "per_class_map50": [0.82, 0.81],
}


def _make_blob(data: bytes = b"") -> MagicMock:
    blob = MagicMock()
    blob.download_as_bytes.return_value = data
    blob.name = ""
    return blob


def _make_bucket(blob: MagicMock) -> MagicMock:
    bucket = MagicMock()
    bucket.blob.return_value = blob
    bucket.list_blobs.return_value = []
    return bucket


def _make_client(bucket: MagicMock) -> MagicMock:
    client = MagicMock()
    client.bucket.return_value = bucket
    return client


# ---------------------------------------------------------------------------
# _download_eval_data
# ---------------------------------------------------------------------------

class TestDownloadEvalData:
    def test_downloads_from_images_eval_prefix(self, tmp_path):
        blob = _make_blob()
        bucket = _make_bucket(blob)
        bucket.list_blobs.return_value = []
        client = _make_client(bucket)

        with patch("pathlib.Path.iterdir", return_value=iter([tmp_path / "fish.jpg"])), \
             patch("pathlib.Path.is_file", return_value=True):
            eval_module._download_eval_data(client, "my-bucket", tmp_path)

        prefixes = [c.kwargs.get("prefix") or c.args[0]
                    for c in bucket.list_blobs.call_args_list]
        assert any("images/eval/" in p for p in prefixes)

    def test_downloads_from_labels_eval_prefix(self, tmp_path):
        blob = _make_blob()
        bucket = _make_bucket(blob)
        bucket.list_blobs.return_value = []
        client = _make_client(bucket)

        with patch("pathlib.Path.iterdir", return_value=iter([tmp_path / "fish.jpg"])), \
             patch("pathlib.Path.is_file", return_value=True):
            eval_module._download_eval_data(client, "my-bucket", tmp_path)

        prefixes = [c.kwargs.get("prefix") or c.args[0]
                    for c in bucket.list_blobs.call_args_list]
        assert any("labels/eval/" in p for p in prefixes)

    def test_raises_when_no_eval_images_found(self, tmp_path):
        blob = _make_blob()
        bucket = _make_bucket(blob)
        bucket.list_blobs.return_value = []
        client = _make_client(bucket)

        with patch("pathlib.Path.iterdir", return_value=iter([])):
            with pytest.raises(RuntimeError, match="No eval images found"):
                eval_module._download_eval_data(client, "my-bucket", tmp_path)


# ---------------------------------------------------------------------------
# _get_class_names
# ---------------------------------------------------------------------------

class TestGetClassNames:
    def test_returns_names_from_data_yaml(self):
        payload = b"names:\n  - Bass\n  - Bluegill\n"
        blob = _make_blob(payload)
        client = _make_client(_make_bucket(blob))

        result = eval_module._get_class_names(client, "my-bucket")

        assert result == ["Bass", "Bluegill"]

    def test_fetches_from_correct_gcs_path(self):
        payload = b"names: []\n"
        blob = _make_blob(payload)
        bucket = _make_bucket(blob)
        client = _make_client(bucket)

        eval_module._get_class_names(client, "my-bucket")

        bucket.blob.assert_called_with("data.yaml")


# ---------------------------------------------------------------------------
# _run_validation / _extract_metrics (via _run_validation)
# ---------------------------------------------------------------------------

class TestRunValidation:
    def _make_mock_results(self, map50=0.82, map_=0.61, mp=0.78, mr=0.75):
        mock_box = MagicMock()
        mock_box.map50 = map50
        mock_box.map = map_
        mock_box.mp = mp
        mock_box.mr = mr
        mock_box.ap50 = MagicMock()
        mock_box.ap50.tolist.return_value = [map50, map50 - 0.01]
        mock_results = MagicMock()
        mock_results.box = mock_box
        return mock_results

    def test_returns_required_metric_keys(self):
        mock_yolo_cls = MagicMock()
        mock_yolo_cls.return_value.val.return_value = self._make_mock_results()
        with patch.dict(sys.modules, {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            import importlib
            import ultralytics
            ultralytics.YOLO = mock_yolo_cls
            metrics = eval_module._run_validation("/tmp/data.yaml")

        assert set(metrics.keys()) == {"mAP50", "mAP50_95", "precision", "recall", "per_class_map50"}

    def test_map50_is_float(self):
        mock_yolo_cls = MagicMock()
        mock_yolo_cls.return_value.val.return_value = self._make_mock_results(map50=0.82)
        with patch.object(eval_module, "YOLO", mock_yolo_cls):
            metrics = eval_module._run_validation("/tmp/data.yaml")
        assert isinstance(metrics["mAP50"], float)
        assert abs(metrics["mAP50"] - 0.82) < 1e-6


# ---------------------------------------------------------------------------
# _log_to_vertex
# ---------------------------------------------------------------------------

class TestLogToVertex:
    def test_inits_with_experiment(self):
        mock_aip = MagicMock()
        mock_aip.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_aip.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(eval_module, "aiplatform", mock_aip):
            eval_module._log_to_vertex("my-project", "us-central1", "fish-id-eval", "run-1", SAMPLE_METRICS)

        mock_aip.init.assert_called_once()
        call_kwargs = mock_aip.init.call_args.kwargs
        assert call_kwargs.get("experiment") == "fish-id-eval"
        assert call_kwargs.get("project") == "my-project"

    def test_logs_map50_and_map50_95(self):
        mock_aip = MagicMock()
        mock_aip.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_aip.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(eval_module, "aiplatform", mock_aip):
            eval_module._log_to_vertex("p", "r", "exp", "run-1", SAMPLE_METRICS)

        logged = mock_aip.log_metrics.call_args.args[0]
        assert "mAP50" in logged
        assert "mAP50_95" in logged

    def test_uses_resume_true(self):
        mock_aip = MagicMock()
        mock_aip.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_aip.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(eval_module, "aiplatform", mock_aip):
            eval_module._log_to_vertex("p", "r", "exp", "run-1", SAMPLE_METRICS)

        _, kwargs = mock_aip.start_run.call_args
        assert kwargs.get("resume") is True
