"""Unit tests for training/train.py.

All external dependencies (GCS, YOLO/ultralytics) are mocked.
No real infrastructure is required.
"""
# pylint: disable=protected-access,wrong-import-position,import-outside-toplevel,wrong-import-order,use-dict-literal

import io
import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock, mock_open, patch

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

HYPERPARAMS_FIXTURE = {
    "epochs": 5,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "patience": 20,
}


def _make_mock_model(epoch=49, box_loss=0.123, save_dir="/tmp/runs/train/exp"):
    """Return a mock YOLO model whose .trainer mirrors the real post-train shape."""
    model = MagicMock()
    model.trainer.epoch = epoch
    model.trainer.save_dir = Path(save_dir)
    model.trainer.metrics = {"train/box_loss": box_loss}
    return model


# ---------------------------------------------------------------------------
# Test 1: Config YAML on disk
# ---------------------------------------------------------------------------

class TestConfig:
    """training/config.yaml has all required keys with valid values."""

    CONFIG_PATH = TRAINING_DIR / "config.yaml"

    def test_all_required_keys_present(self):
        with open(self.CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert {
            "epochs", "imgsz", "batch", "optimizer", "lr0", "patience",
            "dis",
        } <= set(cfg.keys())

    def test_teacher_and_student_are_constants_in_train_module(self):
        assert train_module._TEACHER_MODEL == "yolov8m.pt"  # pylint: disable=protected-access
        assert train_module._STUDENT_MODEL == "yolov8n.pt"  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Test 3: _download_checkpoint
# ---------------------------------------------------------------------------

class TestDownloadCheckpoint:
    _PREFIX = "runs/run-001/checkpoint"

    def test_returns_none_when_no_checkpoint_in_gcs(self):
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value.exists.return_value = False
        result = train_module._download_checkpoint(mock_bucket, self._PREFIX)
        assert result is None

    def test_checks_correct_gcs_path(self):
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value.exists.return_value = False
        train_module._download_checkpoint(mock_bucket, self._PREFIX)
        mock_bucket.blob.assert_called_with(f"{self._PREFIX}/weights/last.pt")

    def test_returns_local_path_when_checkpoint_exists(self):
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value.exists.return_value = True
        result = train_module._download_checkpoint(mock_bucket, self._PREFIX)
        assert result == train_module._CHECKPOINT_DIR / "last.pt"

    def test_downloads_to_correct_local_path(self):
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value.exists.return_value = True
        train_module._download_checkpoint(mock_bucket, self._PREFIX)
        mock_bucket.blob.return_value.download_to_filename.assert_called_once_with(
            str(train_module._CHECKPOINT_DIR / "last.pt")
        )


# ---------------------------------------------------------------------------
# Test 4: GCSCheckpointCallback
# ---------------------------------------------------------------------------

class TestGCSCheckpointCallback:
    _PREFIX = "runs/run-001/checkpoint"

    def test_uploads_last_pt_after_epoch(self, tmp_path):
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "last.pt").write_bytes(b"checkpoint")

        mock_bucket = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.save_dir = tmp_path

        callback = train_module.GCSCheckpointCallback(mock_bucket, self._PREFIX)
        callback.on_train_epoch_end(mock_trainer)

        mock_bucket.blob.assert_called_with(f"{self._PREFIX}/weights/last.pt")
        mock_bucket.blob.return_value.upload_from_filename.assert_called_once_with(
            str(tmp_path / "weights" / "last.pt")
        )

    def test_skips_upload_when_last_pt_missing(self, tmp_path):
        mock_bucket = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.save_dir = tmp_path

        callback = train_module.GCSCheckpointCallback(mock_bucket, self._PREFIX)
        callback.on_train_epoch_end(mock_trainer)

        mock_bucket.blob.return_value.upload_from_filename.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: _train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    _DATA_YAML = "/app/data/data.yaml"
    _CHECKPOINT_PREFIX = "runs/run-001/checkpoint/student"
    _CHECKPOINT_PATH = Path("/tmp/yolo-checkpoint/last.pt")
    _RUN_DIR = Path("/tmp/yolo-runs/student")

    def _make_bucket(self):
        return MagicMock()

    def _call(self, **overrides):
        kwargs = {
            "model_name": "yolov8n.pt",
            "hyperparams": HYPERPARAMS_FIXTURE,
            "workers": 4,
            "data_yaml_path": self._DATA_YAML,
            "checkpoint_bucket": self._make_bucket(),
            "checkpoint_prefix": self._CHECKPOINT_PREFIX,
            "run_dir": self._RUN_DIR,
        }
        kwargs.update(overrides)
        return train_module._train_model(**kwargs)

    def test_fresh_start_yolo_instantiated_with_base_model(self):
        with patch.object(train_module, "_download_checkpoint", return_value=None), \
             patch.object(train_module, "YOLO") as mock_yolo:
            self._call()
        mock_yolo.assert_called_once_with("yolov8n.pt")

    def test_fresh_start_train_called_with_correct_hyperparams(self):
        with patch.object(train_module, "_download_checkpoint", return_value=None), \
             patch.object(train_module, "YOLO") as mock_yolo:
            self._call()
        mock_yolo.return_value.train.assert_called_once_with(
            data=self._DATA_YAML,
            epochs=5,
            imgsz=640,
            batch=16,
            optimizer="AdamW",
            lr0=0.001,
            patience=20,
            workers=4,
            cache=False,
            project=str(self._RUN_DIR),
            name=".",
            save=True,
        )

    def test_fresh_start_returns_yolo_model(self):
        with patch.object(train_module, "_download_checkpoint", return_value=None), \
             patch.object(train_module, "YOLO") as mock_yolo:
            result = self._call()
        assert result is mock_yolo.return_value

    def test_fresh_start_callback_registered(self):
        with patch.object(train_module, "_download_checkpoint", return_value=None), \
             patch.object(train_module, "YOLO") as mock_yolo:
            self._call()
        mock_yolo.return_value.add_callback.assert_called_once_with("on_train_epoch_end", ANY)

    def test_resume_loads_checkpoint_path(self):
        with patch.object(train_module, "_download_checkpoint", return_value=self._CHECKPOINT_PATH), \
             patch.object(train_module, "YOLO") as mock_yolo:
            self._call()
        mock_yolo.assert_called_once_with(str(self._CHECKPOINT_PATH))

    def test_resume_calls_train_with_resume_true(self):
        with patch.object(train_module, "_download_checkpoint", return_value=self._CHECKPOINT_PATH), \
             patch.object(train_module, "YOLO") as mock_yolo:
            self._call()
        mock_yolo.return_value.train.assert_called_once_with(resume=True)

    def test_resume_returns_yolo_model(self):
        with patch.object(train_module, "_download_checkpoint", return_value=self._CHECKPOINT_PATH), \
             patch.object(train_module, "YOLO") as mock_yolo:
            result = self._call()
        assert result is mock_yolo.return_value

    def test_resume_callback_registered(self):
        with patch.object(train_module, "_download_checkpoint", return_value=self._CHECKPOINT_PATH), \
             patch.object(train_module, "YOLO") as mock_yolo:
            self._call()
        mock_yolo.return_value.add_callback.assert_called_once_with("on_train_epoch_end", ANY)


# ---------------------------------------------------------------------------
# Test 5b: _download_teacher_best / _finalize_teacher_best
# ---------------------------------------------------------------------------

class TestTeacherBestCheckpoint:
    _PREFIX = "runs/run-001/checkpoint/teacher"

    def test_download_returns_none_when_missing(self):
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value.exists.return_value = False
        assert train_module._download_teacher_best(mock_bucket, self._PREFIX) is None

    def test_download_checks_best_pt_path(self):
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value.exists.return_value = False
        train_module._download_teacher_best(mock_bucket, self._PREFIX)
        mock_bucket.blob.assert_called_with(f"{self._PREFIX}/weights/best.pt")

    def test_download_returns_local_path_when_exists(self):
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value.exists.return_value = True
        result = train_module._download_teacher_best(mock_bucket, self._PREFIX)
        assert result == train_module._CHECKPOINT_DIR / "teacher-best.pt"

    def test_finalize_uploads_best_pt_from_trainer_save_dir(self, tmp_path):
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_bytes(b"weights")

        teacher_model = _make_mock_model(save_dir=str(tmp_path))
        mock_bucket = MagicMock()

        train_module._finalize_teacher_best(mock_bucket, self._PREFIX, teacher_model)

        mock_bucket.blob.assert_called_once_with(f"{self._PREFIX}/weights/best.pt")
        mock_bucket.blob.return_value.upload_from_filename.assert_called_once_with(
            str(weights_dir / "best.pt")
        )

    def test_finalize_copies_best_pt_to_fixed_local_path_and_returns_it(self, tmp_path):
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_bytes(b"weights")

        teacher_model = _make_mock_model(save_dir=str(tmp_path))
        mock_bucket = MagicMock()

        result = train_module._finalize_teacher_best(mock_bucket, self._PREFIX, teacher_model)

        expected_local_path = train_module._CHECKPOINT_DIR / "teacher-best.pt"
        assert result == expected_local_path
        assert expected_local_path.read_bytes() == b"weights"


# ---------------------------------------------------------------------------
# Test 5c: _train_model with distill_model/dis
# ---------------------------------------------------------------------------

class TestTrainModelDistillation:
    _DATA_YAML = "/app/data/data.yaml"
    _CHECKPOINT_PREFIX = "runs/run-001/checkpoint/student"
    _RUN_DIR = Path("/tmp/yolo-runs/student")

    def test_distill_model_and_dis_merged_into_fresh_start_train_call(self):
        with patch.object(train_module, "_download_checkpoint", return_value=None), \
             patch.object(train_module, "YOLO") as mock_yolo:
            train_module._train_model(
                "yolov8n.pt", HYPERPARAMS_FIXTURE, workers=4, data_yaml_path=self._DATA_YAML,
                checkpoint_bucket=MagicMock(), checkpoint_prefix=self._CHECKPOINT_PREFIX,
                run_dir=self._RUN_DIR,
                distill_model="/tmp/yolo-checkpoint/teacher-best.pt", dis=1.0,
            )
        _, train_kwargs = mock_yolo.return_value.train.call_args
        assert train_kwargs["distill_model"] == "/tmp/yolo-checkpoint/teacher-best.pt"
        assert train_kwargs["dis"] == 1.0

    def test_distill_model_and_dis_not_passed_on_resume(self):
        # Ultralytics re-reads distill_model from the resumed run's own saved args,
        # so it must not be passed again alongside resume=True.
        with patch.object(train_module, "_download_checkpoint", return_value=Path("/tmp/yolo-checkpoint/last.pt")), \
             patch.object(train_module, "YOLO") as mock_yolo:
            train_module._train_model(
                "yolov8n.pt", HYPERPARAMS_FIXTURE, workers=4, data_yaml_path=self._DATA_YAML,
                checkpoint_bucket=MagicMock(), checkpoint_prefix=self._CHECKPOINT_PREFIX,
                run_dir=self._RUN_DIR,
                distill_model="/tmp/yolo-checkpoint/teacher-best.pt", dis=1.0,
            )
        mock_yolo.return_value.train.assert_called_once_with(resume=True)

    def test_neither_distill_model_nor_dis_passed_when_omitted(self):
        # The teacher stage trains with no distill_model/dis at all.
        with patch.object(train_module, "_download_checkpoint", return_value=None), \
             patch.object(train_module, "YOLO") as mock_yolo:
            train_module._train_model(
                "yolov8m.pt", HYPERPARAMS_FIXTURE, workers=4, data_yaml_path=self._DATA_YAML,
                checkpoint_bucket=MagicMock(), checkpoint_prefix="runs/run-001/checkpoint/teacher",
                run_dir=Path("/tmp/yolo-runs/teacher"),
            )
        _, train_kwargs = mock_yolo.return_value.train.call_args
        assert "distill_model" not in train_kwargs
        assert "dis" not in train_kwargs


# ---------------------------------------------------------------------------
# Test 6: _download_training_data
# ---------------------------------------------------------------------------

class TestDownloadTrainingData:
    def _make_client(self, blobs_by_prefix=None):
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.side_effect = lambda prefix=None: (blobs_by_prefix or {}).get(prefix, [])
        return mock_client, mock_bucket

    def test_downloads_data_yaml(self, tmp_path):
        mock_client, mock_bucket = self._make_client()
        train_module._download_training_data(mock_client, "my-bucket", tmp_path)
        mock_bucket.blob.assert_called_with("data.yaml")
        mock_bucket.blob.return_value.download_to_filename.assert_called_with(str(tmp_path / "data.yaml"))

    def test_lists_all_four_prefixes(self, tmp_path):
        mock_client, mock_bucket = self._make_client()
        train_module._download_training_data(mock_client, "my-bucket", tmp_path)
        called_prefixes = {call.kwargs["prefix"] for call in mock_bucket.list_blobs.call_args_list}
        assert called_prefixes == {"images/train/", "images/val/", "labels/train/", "labels/val/"}

    def test_blob_downloaded_to_correct_local_path(self, tmp_path):
        blob = MagicMock()
        blob.name = "images/train/fish1.jpg"
        mock_client, _ = self._make_client({"images/train/": [blob]})
        train_module._download_training_data(mock_client, "my-bucket", tmp_path)
        blob.download_to_filename.assert_called_once_with(str(tmp_path / "images" / "train" / "fish1.jpg"))

    def test_creates_local_directories(self, tmp_path):
        blob = MagicMock()
        blob.name = "images/train/fish1.jpg"
        mock_client, _ = self._make_client({"images/train/": [blob]})
        train_module._download_training_data(mock_client, "my-bucket", tmp_path)
        assert (tmp_path / "images" / "train").is_dir()


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

    def test_returns_export_result_path(self):
        model = _make_mock_model()
        with patch.object(train_module, "YOLO") as mock_yolo:
            mock_yolo.return_value.export.return_value = "/tmp/runs/train/exp/weights/best.onnx"
            path = train_module._export_onnx(model)
        assert path == "/tmp/runs/train/exp/weights/best.onnx"


# ---------------------------------------------------------------------------
# Test 8: _upload_artifacts
# ---------------------------------------------------------------------------

class TestArtifactUpload:
    def _make_metadata(self, run_id="run-001", skip_teacher=False):
        teacher_model = (
            None if skip_teacher
            else _make_mock_model(epoch=99, box_loss=0.456, save_dir="/tmp/runs/train/teacher")
        )
        student_model = _make_mock_model(epoch=49, box_loss=0.123, save_dir="/tmp/runs/train/student")
        return train_module._build_metadata(
            run_id, HYPERPARAMS_FIXTURE, 1.0, teacher_model, student_model,
            teacher_duration_seconds=200.0, student_duration_seconds=120.0,
            cpu_count=4, dataset_generation=12345,
        )

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

    def test_production_onnx_not_written_directly(self, tmp_path):
        # promote_model in the pipeline handles the production ONNX copy after the quality gate.
        mock_bucket = self._run_upload(tmp_path)
        blob_paths = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert "fish-id.onnx" not in blob_paths

    def test_metadata_uploaded_to_run_path(self, tmp_path):
        mock_bucket = self._run_upload(tmp_path)
        blob_paths = [c.args[0] for c in mock_bucket.blob.call_args_list]
        assert "runs/run-001/metadata.json" in blob_paths

    def test_upload_from_filename_called_twice(self, tmp_path):
        mock_bucket = self._run_upload(tmp_path)
        assert mock_bucket.blob.return_value.upload_from_filename.call_count == 2

    def test_metadata_content_has_correct_fields(self):
        metadata = self._make_metadata("run-abc")
        assert metadata["run_id"] == "run-abc"
        assert metadata["teacher_model"] == "yolov8m"
        assert metadata["student_model"] == "yolov8n"
        assert metadata["distillation_loss_weight"] == 1.0
        assert "trained_at" in metadata
        assert metadata["teacher_duration_seconds"] == 200.0
        assert metadata["student_duration_seconds"] == 120.0
        assert metadata["teacher_epochs_completed"] == 100  # teacher trainer.epoch + 1
        assert metadata["student_epochs_completed"] == 50  # student trainer.epoch + 1
        assert metadata["teacher_final_train_loss"] == 0.456
        assert metadata["student_final_train_loss"] == 0.123
        assert metadata["training_args"]["optimizer"] == "AdamW"
        assert metadata["training_args"]["patience"] == 20

    def test_metadata_teacher_stats_are_none_when_teacher_stage_skipped(self):
        metadata = self._make_metadata("run-abc", skip_teacher=True)
        assert metadata["teacher_epochs_completed"] is None
        assert metadata["teacher_final_train_loss"] is None
        assert metadata["student_epochs_completed"] == 50
        assert metadata["student_final_train_loss"] == 0.123
        assert metadata["dataset_generation"] == 12345


# ---------------------------------------------------------------------------
# Test 9: run() — call sequence and arg wiring
# ---------------------------------------------------------------------------

class TestRun:
    _KWARGS = {
        "run_id": "run-test-001",
        "training_bucket": "my-training-bucket",
        "model_bucket": "my-model-bucket",
        "epochs": 5,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "patience": 20,
        "dis": 1.0,
    }

    def _enter_base_patches(self, stack, **overrides):
        specs = {
            "_read_dataset_generation": {"return_value": 99999},
            "_download_training_data": {},
            "_download_teacher_best": {"return_value": None},
            "_finalize_teacher_best": {"return_value": Path("/tmp/yolo-checkpoint/teacher-best.pt")},
            "_train_model": {"return_value": _make_mock_model()},
            "_export_onnx": {"return_value": "/tmp/best.onnx"},
            "_upload_artifacts": {},
        }
        specs.update(overrides)
        mocks = {name: stack.enter_context(patch.object(train_module, name, **kw)) for name, kw in specs.items()}
        stack.enter_context(patch.object(train_module.gcs, "Client"))
        return mocks

    def test_all_steps_called_in_order(self):
        call_order = []

        with patch.object(train_module, "_read_dataset_generation", side_effect=lambda *a: call_order.append("read_dataset_generation") or 99999), \
             patch.object(train_module, "_download_training_data", side_effect=lambda *a: call_order.append("download_training_data")), \
             patch.object(train_module, "_download_teacher_best", side_effect=lambda *a: call_order.append("download_teacher_best") or None), \
             patch.object(train_module, "_finalize_teacher_best", side_effect=lambda *a: call_order.append("finalize_teacher_best") or Path("/tmp/yolo-checkpoint/teacher-best.pt")), \
             patch.object(train_module, "_train_model", side_effect=lambda *a, **kw: call_order.append("train_model") or _make_mock_model()), \
             patch.object(train_module, "_export_onnx", side_effect=lambda *a, **kw: call_order.append("export_onnx") or "/tmp/best.onnx"), \
             patch.object(train_module, "_upload_artifacts", side_effect=lambda *a: call_order.append("upload_artifacts")), \
             patch.object(train_module.gcs, "Client"):
            train_module.run(**self._KWARGS)

        assert call_order == [
            "read_dataset_generation", "download_training_data",
            "download_teacher_best", "train_model", "finalize_teacher_best",
            "train_model", "export_onnx", "upload_artifacts",
        ]

    def test_run_id_passed_to_upload(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.run(**self._KWARGS)

        _, _, run_id, _, _ = mocks["_upload_artifacts"].call_args.args
        assert run_id == "run-test-001"

    def test_data_yaml_path_uses_local_dir(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.run(**self._KWARGS)

        assert mocks["_train_model"].call_args.kwargs["data_yaml_path"] == str(train_module._DATA_DIR / "data.yaml")

    def test_checkpoint_prefix_contains_run_id(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.run(**self._KWARGS)

        checkpoint_prefix = mocks["_train_model"].call_args.kwargs["checkpoint_prefix"]
        assert "run-test-001" in checkpoint_prefix

    def test_teacher_and_student_use_distinct_checkpoint_prefixes(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.run(**self._KWARGS)

        prefixes = [c.kwargs["checkpoint_prefix"] for c in mocks["_train_model"].call_args_list]
        assert prefixes == [
            "runs/run-test-001/checkpoint/teacher",
            "runs/run-test-001/checkpoint/student",
        ]

    def test_student_stage_passes_distill_model_and_dis(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            mocks = self._enter_base_patches(stack)
            train_module.run(**self._KWARGS)

        student_call = mocks["_train_model"].call_args_list[-1]
        assert student_call.kwargs["distill_model"] == str(train_module._CHECKPOINT_DIR / "teacher-best.pt")
        assert student_call.kwargs["dis"] == 1.0

    def test_teacher_stage_skipped_when_already_trained(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            mocks = self._enter_base_patches(
                stack, _download_teacher_best={"return_value": Path("/tmp/yolo-checkpoint/teacher-best.pt")}
            )
            train_module.run(**self._KWARGS)

        mocks["_finalize_teacher_best"].assert_not_called()
        assert mocks["_train_model"].call_count == 1
        assert mocks["_train_model"].call_args.kwargs["checkpoint_prefix"] == "runs/run-test-001/checkpoint/student"
