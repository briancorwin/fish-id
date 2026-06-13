"""Unit tests for training/train.py.

All external dependencies (GCS, YOLO/ultralytics) are mocked.
No real infrastructure is required.
"""

import io
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

# Add training/ to sys.path so we can import train as a module
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

import train as train_module


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Bass", "Bluegill"]

CONFIG_FIXTURE = {
    "model": "yolov8n.pt",
    "epochs": 5,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "AdamW",
    "lr0": 0.001,
}

CONFIG_YAML = "model: yolov8n.pt\nepochs: 5\nimgsz: 640\nbatch: 16\noptimizer: AdamW\nlr0: 0.001\n"


def _make_mock_results():
    r = MagicMock()
    r.epoch = 49
    r.save_dir = Path("/tmp/runs/train/exp")
    r.results_dict = {"train/box_loss": 0.123}
    return r


def _make_blob(name: str) -> MagicMock:
    b = MagicMock()
    b.name = name
    return b


# ---------------------------------------------------------------------------
# Test 1: _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_parsed_yaml(self):
        with patch("builtins.open", mock_open(read_data=CONFIG_YAML)):
            config = train_module._load_config()
        assert config["model"] == "yolov8n.pt"
        assert config["epochs"] == 5
        assert config["lr0"] == 0.001

    def test_reads_from_correct_path(self):
        with patch("builtins.open", mock_open(read_data=CONFIG_YAML)) as mo:
            train_module._load_config()
        mo.assert_called_once_with("/app/config.yaml")


# ---------------------------------------------------------------------------
# Test 2: _load_class_names
# ---------------------------------------------------------------------------

class TestLoadClassNames:
    def test_returns_class_list(self):
        mock_blob = MagicMock()
        mock_blob.download_as_text.return_value = "Bass\nBluegill\n"
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        result = train_module._load_class_names(mock_client, "my-training-bucket")

        assert result == ["Bass", "Bluegill"]

    def test_strips_whitespace_and_blank_lines(self):
        mock_blob = MagicMock()
        mock_blob.download_as_text.return_value = "Bass\n  Bluegill  \n\nPerch\n"
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        result = train_module._load_class_names(mock_client, "my-training-bucket")

        assert result == ["Bass", "Bluegill", "Perch"]

    def test_reads_from_correct_gcs_path(self):
        mock_blob = MagicMock()
        mock_blob.download_as_text.return_value = "Bass\n"
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        train_module._load_class_names(mock_client, "my-training-bucket")

        mock_client.bucket.assert_called_once_with("my-training-bucket")
        mock_bucket.blob.assert_called_once_with("class_names.txt")


# ---------------------------------------------------------------------------
# Test 3: Config YAML on disk
# ---------------------------------------------------------------------------

class TestConfig:
    """training/config.yaml has all required keys with valid values."""

    CONFIG_PATH = TRAINING_DIR / "config.yaml"

    def test_all_required_keys_present(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert {"model", "epochs", "imgsz", "batch", "optimizer", "lr0"} <= set(cfg.keys())

    def test_model_is_yolov8n(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"] == "yolov8n.pt"


# ---------------------------------------------------------------------------
# Test 4: _write_data_yaml
# ---------------------------------------------------------------------------

class TestWriteDataYaml:
    def test_writes_correct_class_names_and_count(self):
        buf = io.StringIO()
        with patch("pathlib.Path.mkdir"), \
             patch("builtins.open", create=True) as mock_open_:
            mock_open_.return_value.__enter__ = lambda s: buf
            mock_open_.return_value.__exit__ = MagicMock(return_value=False)
            train_module._write_data_yaml(["Bass", "Bluegill"])
        loaded = yaml.safe_load(buf.getvalue())
        assert loaded["names"] == ["Bass", "Bluegill"]
        assert loaded["nc"] == 2


# ---------------------------------------------------------------------------
# Test 5: _download_dataset
# ---------------------------------------------------------------------------

class TestDownloadDataset:
    """_download_dataset lists all blobs in each split and downloads them."""

    def _make_bucket_with_blobs(self):
        blobs_by_prefix = {
            "images/train/": [_make_blob("images/train/img001.jpg")],
            "images/val/":   [_make_blob("images/val/img002.jpg")],
            "labels/train/": [_make_blob("labels/train/img001.txt")],
            "labels/val/":   [_make_blob("labels/val/img002.txt")],
        }
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.side_effect = lambda prefix: blobs_by_prefix.get(prefix, [])
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        return mock_client, mock_bucket, blobs_by_prefix

    def test_all_four_splits_are_listed(self):
        mock_client, mock_bucket, _ = self._make_bucket_with_blobs()
        with patch("pathlib.Path.mkdir"):
            train_module._download_dataset(mock_client, "my-training-bucket")
        prefixes = {c.kwargs.get("prefix", c.args[0] if c.args else None)
                    for c in mock_bucket.list_blobs.call_args_list}
        assert prefixes == {"images/train/", "images/val/", "labels/train/", "labels/val/"}

    def test_each_blob_downloaded(self):
        mock_client, mock_bucket, blobs_by_prefix = self._make_bucket_with_blobs()
        all_blobs = [b for blobs in blobs_by_prefix.values() for b in blobs]
        with patch("pathlib.Path.mkdir"):
            train_module._download_dataset(mock_client, "my-training-bucket")
        for blob in all_blobs:
            blob.download_to_filename.assert_called_once()

    def test_correct_bucket_used(self):
        mock_client, mock_bucket, _ = self._make_bucket_with_blobs()
        with patch("pathlib.Path.mkdir"):
            train_module._download_dataset(mock_client, "my-training-bucket")
        mock_client.bucket.assert_called_with("my-training-bucket")

    def test_empty_directory_blobs_skipped(self):
        """Blobs whose name equals the prefix (directory placeholders) are skipped."""
        mock_bucket = MagicMock()
        dir_blob = _make_blob("images/train/")  # name == prefix → no filename
        file_blob = _make_blob("images/train/img001.jpg")
        mock_bucket.list_blobs.side_effect = lambda prefix: (
            [dir_blob, file_blob] if prefix == "images/train/" else []
        )
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("pathlib.Path.mkdir"):
            train_module._download_dataset(mock_client, "my-training-bucket")

        dir_blob.download_to_filename.assert_not_called()
        file_blob.download_to_filename.assert_called_once()

    def test_download_destination_path(self):
        mock_bucket = MagicMock()
        blob = _make_blob("images/train/img001.jpg")
        mock_bucket.list_blobs.side_effect = lambda prefix: [blob] if prefix == "images/train/" else []
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("pathlib.Path.mkdir"):
            train_module._download_dataset(mock_client, "my-training-bucket")

        dest = blob.download_to_filename.call_args.args[0]
        assert dest == "/tmp/dataset/images/train/img001.jpg"


# ---------------------------------------------------------------------------
# Test 6: _train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_yolo_instantiated_with_model_from_config(self):
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._train_model(CONFIG_FIXTURE)
        mock_yolo.assert_called_once_with("yolov8n.pt")

    def test_train_called_with_correct_hyperparams(self):
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._train_model(CONFIG_FIXTURE)
        mock_yolo.return_value.train.assert_called_once_with(
            data="/tmp/dataset/data.yaml",
            epochs=5,
            imgsz=640,
            batch=16,
            optimizer="AdamW",
            lr0=0.001,
        )

    def test_returns_train_results(self):
        with patch.object(train_module, "YOLO") as mock_yolo:
            result = train_module._train_model(CONFIG_FIXTURE)
        assert result is mock_yolo.return_value.train.return_value


# ---------------------------------------------------------------------------
# Test 7: _export_onnx
# ---------------------------------------------------------------------------

class TestExportOnnx:
    def test_yolo_loaded_with_best_pt_path(self):
        results = _make_mock_results()
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._export_onnx(results)
        mock_yolo.assert_called_once_with("/tmp/runs/train/exp/weights/best.pt")

    def test_export_called_with_onnx_format(self):
        results = _make_mock_results()
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._export_onnx(results)
        mock_yolo.return_value.export.assert_called_once_with(format="onnx")

    def test_returns_onnx_path(self):
        results = _make_mock_results()
        with patch.object(train_module, "YOLO"):
            path = train_module._export_onnx(results)
        assert path == "/tmp/runs/train/exp/weights/best.onnx"


# ---------------------------------------------------------------------------
# Test 8: _upload_artifacts
# ---------------------------------------------------------------------------

class TestArtifactUpload:
    def _make_metadata(self, run_id="run-001"):
        return train_module._build_metadata(run_id, CONFIG_FIXTURE, _make_mock_results(), 120.0)

    def _run_upload(self, tmp_path, run_id="run-001"):
        mock_bucket = MagicMock()
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        fake_onnx = tmp_path / "best.onnx"
        fake_onnx.write_bytes(b"fake")

        with patch("builtins.open", create=True) as mock_open_:
            mock_open_.return_value.__enter__ = lambda s: io.StringIO()
            mock_open_.return_value.__exit__ = MagicMock(return_value=False)
            train_module._upload_artifacts(
                mock_client, "my-model-bucket", run_id, str(fake_onnx), self._make_metadata(run_id)
            )

        return mock_bucket

    def test_run_onnx_uploaded_to_run_path(self, tmp_path):
        mock_bucket = self._run_upload(tmp_path)
        blob_paths = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert "runs/run-001/fish-id.onnx" in blob_paths

    def test_onnx_copied_to_production_path(self, tmp_path):
        mock_bucket = self._run_upload(tmp_path)
        blob_paths = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert "fish-id.onnx" in blob_paths

    def test_metadata_uploaded_to_run_path(self, tmp_path):
        mock_bucket = self._run_upload(tmp_path)
        blob_paths = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert "runs/run-001/metadata.json" in blob_paths

    def test_upload_from_filename_called_three_times(self, tmp_path):
        mock_bucket = self._run_upload(tmp_path)
        assert mock_bucket.blob.return_value.upload_from_filename.call_count == 3

    def test_metadata_content_has_correct_fields(self):
        metadata = self._make_metadata("run-abc")
        assert metadata["run_id"] == "run-abc"
        assert metadata["epochs_completed"] == 50  # results.epoch + 1
        assert metadata["model_architecture"] == "yolov8n"
        assert metadata["base_weights"] == "yolov8n.pt"
        assert "trained_at" in metadata
        assert metadata["duration_seconds"] == 120.0
        assert metadata["training_args"]["optimizer"] == "AdamW"


# ---------------------------------------------------------------------------
# Test 9: main() — env var wiring and call sequence
# ---------------------------------------------------------------------------

class TestMain:
    _ENV = {"RUN_ID": "run-test-001", "TRAINING_BUCKET": "my-training-bucket", "MODEL_BUCKET": "my-model-bucket"}

    def _enter_base_patches(self, stack, **overrides):
        specs = {
            "_load_config": dict(return_value=CONFIG_FIXTURE),
            "_load_class_names": dict(return_value=CLASS_NAMES),
            "_download_dataset": {},
            "_write_data_yaml": {},
            "_train_model": dict(return_value=_make_mock_results()),
            "_export_onnx": dict(return_value="/tmp/best.onnx"),
            "_upload_artifacts": {},
        }
        specs.update(overrides)
        mocks = {name: stack.enter_context(patch.object(train_module, name, **kw)) for name, kw in specs.items()}
        stack.enter_context(patch.object(train_module.gcs, "Client"))
        return mocks

    def test_all_steps_called_in_order(self, monkeypatch):
        for k, v in self._ENV.items():
            monkeypatch.setenv(k, v)

        call_order = []

        with patch.object(train_module, "_load_config", side_effect=lambda: call_order.append("load_config") or CONFIG_FIXTURE), \
             patch.object(train_module, "_load_class_names", side_effect=lambda *a: call_order.append("load_class_names") or CLASS_NAMES), \
             patch.object(train_module, "_download_dataset", side_effect=lambda *a: call_order.append("download_dataset")), \
             patch.object(train_module, "_write_data_yaml", side_effect=lambda *a: call_order.append("write_data_yaml")), \
             patch.object(train_module, "_train_model", side_effect=lambda *a: call_order.append("train_model") or _make_mock_results()), \
             patch.object(train_module, "_export_onnx", side_effect=lambda *a: call_order.append("export_onnx") or "/tmp/best.onnx"), \
             patch.object(train_module, "_upload_artifacts", side_effect=lambda *a: call_order.append("upload_artifacts")), \
             patch.object(train_module.gcs, "Client"):
            train_module.main()

        assert call_order == [
            "load_config", "load_class_names", "download_dataset",
            "write_data_yaml", "train_model", "export_onnx", "upload_artifacts",
        ]

    def test_run_id_passed_to_upload(self, monkeypatch):
        for k, v in self._ENV.items():
            monkeypatch.setenv(k, v)

        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.main()

        _, _, run_id, _, _ = mocks["_upload_artifacts"].call_args.args
        assert run_id == "run-test-001"

    def test_class_names_fetched_from_training_bucket(self, monkeypatch):
        for k, v in self._ENV.items():
            monkeypatch.setenv(k, v)

        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.main()

        assert mocks["_load_class_names"].call_args.args[1] == "my-training-bucket"

    def test_class_names_passed_to_write_data_yaml(self, monkeypatch):
        for k, v in self._ENV.items():
            monkeypatch.setenv(k, v)

        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.main()

        mocks["_write_data_yaml"].assert_called_once_with(CLASS_NAMES)

    def test_missing_env_var_raises(self, monkeypatch):
        for k in ("RUN_ID", "TRAINING_BUCKET", "MODEL_BUCKET"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(KeyError):
            train_module.main()
