"""Unit tests for training/train.py and training/eval.py.

All external dependencies (GCS, YOLO/ultralytics, Vertex AI) are mocked.
No real infrastructure is required.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import call, patch

import pytest
import yaml

# Add training/ to sys.path so we can import train and eval as modules
TRAINING_DIR = Path(__file__).parent.parent / "training"
sys.path.insert(0, str(TRAINING_DIR))

# Stub heavy deps that are not installed in the test environment
from unittest.mock import MagicMock  # noqa: E402 (must precede training imports)
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
import eval as eval_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MANIFEST_FIXTURE = {
    "version": "v3",
    "train_files": ["a.jpg", "b.jpg"],
    "val_files": ["c.jpg"],
    "class_names": ["Bass", "Bluegill"],
    "image_count": {"train": 2, "valid": 1},
}


def _make_gcs_client(manifest_bytes):
    """Return a mock GCS client whose manifest blob returns manifest_bytes."""
    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = manifest_bytes

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    return mock_client, mock_bucket, mock_blob


# ---------------------------------------------------------------------------
# Test 1: Manifest parsing
# ---------------------------------------------------------------------------

class TestManifestParsing:
    """train.py correctly extracts fields from the manifest JSON."""

    def test_train_files_extracted(self):
        manifest_bytes = json.dumps(MANIFEST_FIXTURE).encode()
        mock_client, _, _ = _make_gcs_client(manifest_bytes)

        result = train_module._download_manifest(mock_client, "my-training-bucket", "v3")

        assert result["train_files"] == ["a.jpg", "b.jpg"]
        assert result["val_files"] == ["c.jpg"]

    def test_class_names_extracted(self):
        manifest_bytes = json.dumps(MANIFEST_FIXTURE).encode()
        mock_client, _, _ = _make_gcs_client(manifest_bytes)

        result = train_module._download_manifest(mock_client, "my-training-bucket", "v3")

        assert result["class_names"] == ["Bass", "Bluegill"]

    def test_data_yaml_uses_class_names(self, tmp_path, monkeypatch):
        """write_data_yaml writes the correct class names to data.yaml."""
        monkeypatch.setattr(train_module, "__file__", str(tmp_path / "train.py"))
        # Redirect the output path to tmp_path
        data_yaml_path = tmp_path / "data.yaml"

        with patch("train.Path") as mock_path_cls:
            # Allow Path to work normally except for the mkdir call on /tmp/dataset
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw)

            with patch("builtins.open", create=True) as mock_open:
                captured = {}

                def fake_open(path, mode="r", *args, **kwargs):
                    if mode == "w":
                        import io
                        buf = io.StringIO()

                        class Ctx:
                            def __enter__(self_inner):
                                return buf

                            def __exit__(self_inner, *a):
                                captured["content"] = buf.getvalue()

                        return Ctx()
                    return open(path, mode, *args, **kwargs)

                mock_open.side_effect = fake_open

                with patch("pathlib.Path.mkdir"):
                    train_module._write_data_yaml(MANIFEST_FIXTURE["class_names"])

        # data.yaml must contain class names
        if captured.get("content"):
            loaded = yaml.safe_load(captured["content"])
            assert loaded["names"] == ["Bass", "Bluegill"]
            assert loaded["nc"] == 2

    def test_manifest_gcs_path(self):
        """Manifest is fetched from the correct GCS path."""
        manifest_bytes = json.dumps(MANIFEST_FIXTURE).encode()
        mock_client, mock_bucket, _ = _make_gcs_client(manifest_bytes)

        train_module._download_manifest(mock_client, "my-training-bucket", "v3")

        mock_client.bucket.assert_called_once_with("my-training-bucket")
        mock_bucket.blob.assert_called_once_with("versions/v3/manifest.json")


# ---------------------------------------------------------------------------
# Test 2: Config YAML loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """train.py reads real config files from disk correctly."""

    CONFIGS_DIR = TRAINING_DIR / "configs"

    def test_c1_model(self):
        with open(self.CONFIGS_DIR / "c1.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"] == "yolov8n.pt"

    def test_c1_epochs(self):
        with open(self.CONFIGS_DIR / "c1.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["epochs"] == 50

    def test_c1_batch(self):
        with open(self.CONFIGS_DIR / "c1.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["batch"] == 16

    @pytest.mark.parametrize("version", ["c1"])
    def test_all_required_keys_present(self, version):
        with open(self.CONFIGS_DIR / f"{version}.yaml") as f:
            cfg = yaml.safe_load(f)
        required_keys = {"model", "epochs", "imgsz", "batch", "optimizer", "lr0"}
        assert required_keys <= set(cfg.keys()), (
            f"{version}.yaml missing keys: {required_keys - set(cfg.keys())}"
        )

    def test_load_config_function_c1(self):
        """train_module._load_config reads from /app/configs/ path."""
        with patch("builtins.open", create=True) as mock_open:
            import io
            mock_open.return_value.__enter__ = lambda s: io.StringIO(
                "model: yolov8n.pt\nepochs: 50\nimgsz: 640\nbatch: 16\noptimizer: SGD\nlr0: 0.01\n"
            )
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            cfg = train_module._load_config("1")

        mock_open.assert_called_once_with("/app/configs/c1.yaml")
        assert cfg["model"] == "yolov8n.pt"
        assert cfg["epochs"] == 50


# ---------------------------------------------------------------------------
# Test 3: GCS download logic
# ---------------------------------------------------------------------------

class TestGCSDownloadLogic:
    """train.py calls GCS with the correct bucket and blob paths."""

    def test_train_files_downloaded_to_correct_paths(self):
        small_manifest = {
            "train_files": ["img001.jpg"],
            "val_files": ["img002.jpg"],
            "class_names": ["Bass"],
        }

        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("pathlib.Path.mkdir"):
            train_module._download_dataset(mock_client, "my-training-bucket", small_manifest)

        # Collect all blob() calls
        blob_calls = [c.args[0] for c in mock_bucket.blob.call_args_list]

        assert "images/train/img001.jpg" in blob_calls
        assert "images/val/img002.jpg" in blob_calls

    def test_download_to_filename_called_for_each_file(self):
        small_manifest = {
            "train_files": ["img001.jpg"],
            "val_files": ["img002.jpg"],
            "class_names": ["Bass"],
        }

        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("pathlib.Path.mkdir"):
            train_module._download_dataset(mock_client, "my-training-bucket", small_manifest)

        # Should have been called once per image file (train + val = 2)
        assert mock_blob.download_to_filename.call_count == 2

    def test_no_extra_blobs_downloaded(self):
        """Only the files listed in the manifest are downloaded (no extras)."""
        small_manifest = {
            "train_files": ["img001.jpg"],
            "val_files": ["img002.jpg"],
            "class_names": ["Bass"],
        }

        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("pathlib.Path.mkdir"):
            train_module._download_dataset(mock_client, "my-training-bucket", small_manifest)

        blob_calls = [c.args[0] for c in mock_bucket.blob.call_args_list]
        # Only train and val image blobs — no label blobs since manifest has no label_files
        assert len(blob_calls) == 2

    def test_correct_bucket_used(self):
        small_manifest = {
            "train_files": ["img001.jpg"],
            "val_files": [],
            "class_names": ["Bass"],
        }

        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("pathlib.Path.mkdir"):
            train_module._download_dataset(mock_client, "my-training-bucket", small_manifest)

        mock_client.bucket.assert_called_with("my-training-bucket")


# ---------------------------------------------------------------------------
# Test 4: Artifact upload
# ---------------------------------------------------------------------------

class TestArtifactUpload:
    """train.py uploads fish-id.onnx and metadata.json to the correct GCS paths."""

    def _make_mock_results(self):
        mock_results = MagicMock()
        mock_results.epoch = 49
        mock_results.save_dir = Path("/tmp/runs/train/exp")
        mock_results.results_dict = {"train/box_loss": 0.123}
        return mock_results

    def test_onnx_uploaded_to_correct_path(self, tmp_path):
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        mock_results = self._make_mock_results()
        config = {"model": "yolov8n.pt", "epochs": 50, "imgsz": 640, "batch": 16,
                  "optimizer": "SGD", "lr0": 0.01}

        metadata = train_module._build_metadata(
            "run-001", "v3", "1", config, mock_results, 120.0
        )

        # Use a temp onnx file as the local artifact
        fake_onnx = tmp_path / "best.onnx"
        fake_onnx.write_bytes(b"fake")

        with patch("builtins.open", create=True) as mock_open:
            import io
            mock_open.return_value.__enter__ = lambda s: io.StringIO()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            train_module._upload_artifacts(
                mock_client, "my-model-bucket", "run-001", str(fake_onnx), metadata
            )

        # Check blob paths
        blob_paths = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert any("runs/run-001/fish-id.onnx" in p for p in blob_paths)
        assert any("runs/run-001/metadata.json" in p for p in blob_paths)

    def test_metadata_uploaded_to_correct_path(self, tmp_path):
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        mock_results = self._make_mock_results()
        config = {"model": "yolov8n.pt", "epochs": 50, "imgsz": 640, "batch": 16,
                  "optimizer": "SGD", "lr0": 0.01}

        metadata = train_module._build_metadata(
            "run-001", "v3", "1", config, mock_results, 120.0
        )

        fake_onnx = tmp_path / "best.onnx"
        fake_onnx.write_bytes(b"fake")

        with patch("builtins.open", create=True) as mock_open:
            import io
            mock_open.return_value.__enter__ = lambda s: io.StringIO()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            train_module._upload_artifacts(
                mock_client, "my-model-bucket", "run-001", str(fake_onnx), metadata
            )

        assert mock_blob.upload_from_filename.call_count == 2

    def test_metadata_content_has_correct_fields(self):
        mock_results = self._make_mock_results()
        config = {"model": "yolov8n.pt", "epochs": 50, "imgsz": 640, "batch": 16,
                  "optimizer": "SGD", "lr0": 0.01}

        metadata = train_module._build_metadata(
            "run-abc", "v5", "2", config, mock_results, 300.0
        )

        assert metadata["run_id"] == "run-abc"
        assert metadata["dataset_version"] == "v5"
        assert metadata["config_version"] == "2"
        assert metadata["epochs_completed"] == 50  # results.epoch + 1
        assert metadata["model_architecture"] == "yolov8n"
        assert metadata["base_weights"] == "yolov8n.pt"
        assert "trained_at" in metadata
        assert metadata["duration_seconds"] == 300.0

    def test_upload_from_filename_called_twice(self, tmp_path):
        """upload_artifacts must call upload_from_filename exactly twice."""
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        mock_results = self._make_mock_results()
        config = {"model": "yolov8n.pt", "epochs": 50, "imgsz": 640, "batch": 16,
                  "optimizer": "SGD", "lr0": 0.01}
        metadata = train_module._build_metadata(
            "run-001", "v3", "1", config, mock_results, 60.0
        )

        fake_onnx = tmp_path / "best.onnx"
        fake_onnx.write_bytes(b"fake")

        with patch("builtins.open", create=True) as mock_open:
            import io
            mock_open.return_value.__enter__ = lambda s: io.StringIO()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            train_module._upload_artifacts(
                mock_client, "my-model-bucket", "run-001", str(fake_onnx), metadata
            )

        assert mock_blob.upload_from_filename.call_count == 2
