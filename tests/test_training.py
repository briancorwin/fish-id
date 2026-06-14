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
if "torch" not in sys.modules:
    _torch_mock = MagicMock()
    _torch_mock.cuda.device_count.return_value = 0
    sys.modules["torch"] = _torch_mock

import train as train_module


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CONFIG_FIXTURE = {
    "model": "yolov8n.pt",
    "epochs": 5,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "AdamW",
    "lr0": 0.001,
}

CONFIG_YAML = "model: yolov8n.pt\nepochs: 5\nimgsz: 640\nbatch: 16\noptimizer: AdamW\nlr0: 0.001\n"


def _make_mock_model():
    """Return a mock YOLO model whose .trainer mirrors the real post-train shape."""
    model = MagicMock()
    model.trainer.epoch = 49
    model.trainer.save_dir = Path("/tmp/runs/train/exp")
    model.trainer.metrics = {"train/box_loss": 0.123}
    return model


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
# Test 2: Config YAML on disk
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
# Test 3: _train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    _DATA_YAML = "/gcs/my-training-bucket/data.yaml"
    _CHECKPOINT_DIR = "/gcs/my-model-bucket/runs/run-001"

    def test_yolo_instantiated_with_model_from_config(self, tmp_path):
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._train_model(CONFIG_FIXTURE, workers=4, data_yaml_path=self._DATA_YAML, checkpoint_dir=str(tmp_path))
        mock_yolo.assert_called_once_with("yolov8n.pt")

    def test_train_called_with_correct_hyperparams(self, tmp_path):
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._train_model(CONFIG_FIXTURE, workers=4, data_yaml_path=self._DATA_YAML, checkpoint_dir=str(tmp_path))
        mock_yolo.return_value.train.assert_called_once_with(
            data=self._DATA_YAML,
            epochs=5,
            imgsz=640,
            batch=16,
            optimizer="AdamW",
            lr0=0.001,
            workers=4,
            cache=False,
            project=str(tmp_path),
            name=".",
            save=True,
            save_period=1,
        )

    def test_returns_yolo_model(self, tmp_path):
        with patch.object(train_module, "YOLO") as mock_yolo:
            result = train_module._train_model(CONFIG_FIXTURE, workers=4, data_yaml_path=self._DATA_YAML, checkpoint_dir=str(tmp_path))
        assert result is mock_yolo.return_value


class TestTrainModelResume:
    _DATA_YAML = "/gcs/my-training-bucket/data.yaml"

    def test_resumes_from_checkpoint_when_last_pt_exists(self, tmp_path):
        last_pt = tmp_path / "weights" / "last.pt"
        last_pt.parent.mkdir()
        last_pt.write_bytes(b"fake")
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._train_model(CONFIG_FIXTURE, workers=4, data_yaml_path=self._DATA_YAML, checkpoint_dir=str(tmp_path))
        mock_yolo.assert_called_once_with(str(last_pt))
        mock_yolo.return_value.train.assert_called_once_with(resume=True)

    def test_fresh_run_when_no_checkpoint(self, tmp_path):
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._train_model(CONFIG_FIXTURE, workers=4, data_yaml_path=self._DATA_YAML, checkpoint_dir=str(tmp_path))
        mock_yolo.assert_called_once_with("yolov8n.pt")


# ---------------------------------------------------------------------------
# Test 7: _export_onnx
# ---------------------------------------------------------------------------

class TestExportOnnx:
    def test_yolo_loaded_with_best_pt_path(self):
        model = _make_mock_model()
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._export_onnx(model)
        mock_yolo.assert_called_once_with("/tmp/runs/train/exp/weights/best.pt")

    def test_export_called_with_onnx_format(self):
        model = _make_mock_model()
        with patch.object(train_module, "YOLO") as mock_yolo:
            train_module._export_onnx(model)
        mock_yolo.return_value.export.assert_called_once_with(format="onnx")

    def test_returns_onnx_path(self):
        model = _make_mock_model()
        with patch.object(train_module, "YOLO"):
            path = train_module._export_onnx(model)
        assert path == "/tmp/runs/train/exp/weights/best.onnx"


# ---------------------------------------------------------------------------
# Test 8: _upload_artifacts
# ---------------------------------------------------------------------------

class TestArtifactUpload:
    def _make_metadata(self, run_id="run-001"):
        return train_module._build_metadata(run_id, CONFIG_FIXTURE, _make_mock_model(), 120.0, cpu_count=4)

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
        assert metadata["epochs_completed"] == 50  # trainer.epoch + 1
        assert metadata["model_architecture"] == "yolov8n"
        assert metadata["base_weights"] == "yolov8n.pt"
        assert "trained_at" in metadata
        assert metadata["duration_seconds"] == 120.0
        assert metadata["training_args"]["optimizer"] == "AdamW"


# ---------------------------------------------------------------------------
# Test 9: main() — env var wiring and call sequence
# ---------------------------------------------------------------------------

class TestMain:
    _ARGV = ["train.py", "--run-id", "run-test-001", "--training-bucket", "my-training-bucket", "--model-bucket", "my-model-bucket"]

    def _enter_base_patches(self, stack, **overrides):
        specs = {
            "_load_config": dict(return_value=CONFIG_FIXTURE),
            "_train_model": dict(return_value=_make_mock_model()),
            "_export_onnx": dict(return_value="/tmp/best.onnx"),
            "_upload_artifacts": {},
        }
        specs.update(overrides)
        mocks = {name: stack.enter_context(patch.object(train_module, name, **kw)) for name, kw in specs.items()}
        stack.enter_context(patch.object(train_module.gcs, "Client"))
        return mocks

    def test_all_steps_called_in_order(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._ARGV)
        call_order = []

        with patch.object(train_module, "_load_config", side_effect=lambda: call_order.append("load_config") or CONFIG_FIXTURE), \
             patch.object(train_module, "_train_model", side_effect=lambda *a, **kw: call_order.append("train_model") or _make_mock_model()), \
             patch.object(train_module, "_export_onnx", side_effect=lambda *a: call_order.append("export_onnx") or "/tmp/best.onnx"), \
             patch.object(train_module, "_upload_artifacts", side_effect=lambda *a: call_order.append("upload_artifacts")), \
             patch.object(train_module.gcs, "Client"):
            train_module.main()

        assert call_order == ["load_config", "train_model", "export_onnx", "upload_artifacts"]

    def test_run_id_passed_to_upload(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._ARGV)
        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.main()

        _, _, run_id, _, _ = mocks["_upload_artifacts"].call_args.args
        assert run_id == "run-test-001"

    def test_data_yaml_path_uses_gcs_fuse(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._ARGV)
        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.main()

        assert mocks["_train_model"].call_args.kwargs["data_yaml_path"] == "/gcs/my-training-bucket/data.yaml"

    def test_checkpoint_dir_uses_gcs_fuse(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._ARGV)
        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.main()

        assert mocks["_train_model"].call_args.kwargs["checkpoint_dir"] == "/gcs/my-model-bucket/runs/run-test-001"

    def test_missing_args_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train.py"])
        with pytest.raises(SystemExit):
            train_module.main()
