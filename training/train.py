import concurrent.futures
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import google.cloud.storage as gcs
import torch
from ultralytics import YOLO

_logger = logging.getLogger(__name__)

_DATA_DIR = Path("/app/data")
# Local staging area for files pulled down from GCS (resume checkpoints, teacher best.pt for
# distillation) — populated by _download_checkpoint/_download_teacher_best, not by YOLO itself.
_CHECKPOINT_DIR = Path("/tmp/yolo-checkpoint")
# Passed as `project=` to model.train(): the directory YOLO itself writes outputs to
# (weights/last.pt, weights/best.pt, etc.) under per-stage "teacher"/"student" subfolders.
_YOLO_RUNS_DIR = Path("/tmp/yolo-runs")
_METADATA_PATH = Path("/tmp/metadata.json")

# Fixed rather than config/pipeline parameters — expected to change rarely, and
# training/Dockerfile pre-downloads these exact weights into the image at build time
# to avoid runtime egress to ultralytics.com. Keep the two files in sync if changed.
_TEACHER_MODEL = "yolov8m.pt"
_STUDENT_MODEL = "yolov8n.pt"


class GCSCheckpointCallback:
    def __init__(self, bucket: gcs.Bucket, gcs_prefix: str) -> None:
        self._bucket = bucket
        self._gcs_prefix = gcs_prefix

    def on_train_epoch_end(self, trainer) -> None:
        local = Path(trainer.save_dir) / "weights" / "last.pt"
        if not local.exists():
            _logger.warning("[train] checkpoint not found at %s — skipping upload", local)
            return
        dest = f"{self._gcs_prefix}/weights/last.pt"
        self._bucket.blob(dest).upload_from_filename(str(local))
        _logger.info("[train] checkpoint uploaded to gs://%s/%s", self._bucket.name, dest)



def _download_checkpoint(bucket: gcs.Bucket, gcs_prefix: str) -> Path | None:
    blob = bucket.blob(f"{gcs_prefix}/weights/last.pt")
    if not blob.exists():
        return None
    local_path = _CHECKPOINT_DIR / "last.pt"
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    _logger.info("[train] checkpoint downloaded from gs://%s/%s", bucket.name, blob.name)
    return local_path


def _download_teacher_best(bucket: gcs.Bucket, gcs_prefix: str) -> Path | None:
    blob = bucket.blob(f"{gcs_prefix}/weights/best.pt")
    if not blob.exists():
        return None
    local_path = _CHECKPOINT_DIR / "teacher-best.pt"
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    _logger.info("[train] teacher best.pt downloaded from gs://%s/%s", bucket.name, blob.name)
    return local_path


def _finalize_teacher_best(bucket: gcs.Bucket, gcs_prefix: str, teacher_model: YOLO) -> Path:
    # Uploads the just-trained teacher's best.pt to GCS and also stages it at the fixed local
    # path the student stage's distill_model needs — see the comment at that call site in run().
    assert teacher_model.trainer is not None
    trained_best = teacher_model.trainer.save_dir / "weights/best.pt"

    dest = f"{gcs_prefix}/weights/best.pt"
    bucket.blob(dest).upload_from_filename(str(trained_best))
    _logger.info("[train] teacher best.pt uploaded to gs://%s/%s", bucket.name, dest)

    local_path = _CHECKPOINT_DIR / "teacher-best.pt"
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(trained_best, local_path)
    return local_path


def _train_model(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model_name: str,
    hyperparams: dict,
    workers: int,
    data_yaml_path: str,
    checkpoint_bucket: gcs.Bucket,
    checkpoint_prefix: str,
    run_dir: Path,
    distill_model: str | None = None,
    dis: float | None = None,
) -> YOLO:
    checkpoint_path = _download_checkpoint(checkpoint_bucket, checkpoint_prefix)
    callback = GCSCheckpointCallback(checkpoint_bucket, checkpoint_prefix)

    if checkpoint_path is not None:
        _logger.info("[train] resuming from checkpoint %s", checkpoint_path)
        model = YOLO(str(checkpoint_path))
        model.add_callback("on_train_epoch_end", callback.on_train_epoch_end)
        model.train(resume=True)
    else:
        _logger.info(
            "[train] no checkpoint found — starting fresh: model=%s epochs=%s imgsz=%s batch=%s optimizer=%s lr0=%s patience=%s workers=%d",
            model_name, hyperparams["epochs"], hyperparams["imgsz"], hyperparams["batch"],
            hyperparams["optimizer"], hyperparams["lr0"], hyperparams["patience"], workers,
        )
        distill_kwargs = {}
        if distill_model is not None:
            distill_kwargs["distill_model"] = distill_model
        if dis is not None:
            distill_kwargs["dis"] = dis

        model = YOLO(model_name)
        model.add_callback("on_train_epoch_end", callback.on_train_epoch_end)
        model.train(
            data=data_yaml_path,
            epochs=hyperparams["epochs"],
            imgsz=hyperparams["imgsz"],
            batch=hyperparams["batch"],
            optimizer=hyperparams["optimizer"],
            lr0=hyperparams["lr0"],
            patience=hyperparams["patience"],
            workers=workers,
            cache=False,
            # Keep output in /tmp so YOLO has a writable directory; all other
            # artifacts are discarded when the container exits.
            project=str(run_dir),
            # Prevent YOLO from auto-incrementing to train/, train2/, etc.
            name=".",
            save=True,
            **distill_kwargs,
        )

    assert model.trainer is not None
    _logger.info("[train] YOLO training finished. save_dir=%s", model.trainer.save_dir)
    return model


def _export_onnx(model: YOLO) -> str:
    assert model.trainer is not None
    best_pt = str(model.trainer.save_dir / "weights/best.pt")
    _logger.info("[train] exporting ONNX from %s", best_pt)
    best_model = YOLO(best_pt)
    onnx_path = best_model.export(format="onnx")
    _logger.info("[train] ONNX exported to %s", onnx_path)
    return onnx_path


def _read_image_tag() -> str:
    tag_file = Path("/app/image_tag.txt")
    if tag_file.exists():
        return tag_file.read_text(encoding="utf-8").strip()
    return "unknown"


def _gpu_info() -> list[dict]:
    return [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory_total_mb": torch.cuda.get_device_properties(i).total_memory // (1024 ** 2),
        }
        for i in range(torch.cuda.device_count())
    ]


def _download_prefix(bucket: gcs.Bucket, gcs_prefix: str, local_dir: Path) -> None:
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    _logger.info("[train] downloading %d files from %s", len(blobs), gcs_prefix)

    def _fetch(blob: gcs.Blob) -> None:
        filename = blob.name[len(gcs_prefix):]
        if not filename:
            return
        dest = local_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(_fetch, blobs))


def _download_training_data(storage_client: gcs.Client, training_bucket: str, local_dir: Path) -> None:
    bucket = storage_client.bucket(training_bucket)

    local_dir.mkdir(parents=True, exist_ok=True)
    bucket.blob("data.yaml").download_to_filename(str(local_dir / "data.yaml"))

    _download_prefix(bucket, "images/train/", local_dir / "images/train")
    _download_prefix(bucket, "images/val/",   local_dir / "images/val")
    _download_prefix(bucket, "labels/train/", local_dir / "labels/train")
    _download_prefix(bucket, "labels/val/",   local_dir / "labels/val")

    _logger.info("[train] data download complete")


def _read_dataset_generation(storage_client: gcs.Client, training_bucket: str) -> int:
    blob = storage_client.bucket(training_bucket).blob("data.yaml")
    blob.reload()
    return blob.generation


def _model_stats(model: YOLO) -> tuple[int, float | None]:
    assert model.trainer is not None
    trainer = model.trainer
    final_train_loss = (
        float(trainer.metrics.get("train/box_loss", 0.0)) if hasattr(trainer, "metrics") else None
    )
    return trainer.epoch + 1, final_train_loss


def _build_metadata(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_id: str,
    hyperparams: dict,
    dis: float,
    teacher_model: YOLO | None,
    student_model: YOLO,
    teacher_duration_seconds: float,
    student_duration_seconds: float,
    cpu_count: int,
    dataset_generation: int,
) -> dict:
    # teacher/student_duration_seconds are wall-clock time for *this process only* (time.time()
    # deltas in run()) — if a stage was interrupted and resumed in a later container, this only
    # reflects the final resumed attempt, undercounting the stage's true total training time.
    # When the teacher stage was skipped entirely (an already-trained checkpoint was reused
    # across runs), teacher_duration_seconds is the 0.0 sentinel set in run(), not real duration.
    # epochs_completed and final_train_loss are read from the checkpoint's own trainer state
    # (trainer.epoch / trainer.metrics), so they stay accurate across interruptions — resuming
    # restores that state rather than resetting it.
    teacher_epochs_completed, teacher_final_train_loss = (
        _model_stats(teacher_model) if teacher_model is not None else (None, None)
    )
    student_epochs_completed, student_final_train_loss = _model_stats(student_model)

    _logger.info(
        "[train] building metadata for run_id=%s teacher_duration=%.1fs student_duration=%.1fs",
        run_id, teacher_duration_seconds, student_duration_seconds,
    )
    metadata = {
        "run_id": run_id,
        "dataset_generation": dataset_generation,
        "container_image": _read_image_tag(),
        "teacher_model": _TEACHER_MODEL.replace(".pt", ""),
        "student_model": _STUDENT_MODEL.replace(".pt", ""),
        "distillation_loss_weight": dis,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "teacher_duration_seconds": teacher_duration_seconds,
        "student_duration_seconds": student_duration_seconds,
        "teacher_epochs_completed": teacher_epochs_completed,
        "student_epochs_completed": student_epochs_completed,
        "training_args": {
            "epochs": hyperparams["epochs"],
            "imgsz": hyperparams["imgsz"],
            "batch": hyperparams["batch"],
            "optimizer": hyperparams["optimizer"],
            "lr0": hyperparams["lr0"],
            "patience": hyperparams["patience"],
        },
        "teacher_final_train_loss": teacher_final_train_loss,
        "student_final_train_loss": student_final_train_loss,
        "cpu_count": cpu_count,
        "gpus": _gpu_info(),
    }
    _logger.info("[train] metadata: %s", metadata)
    return metadata


def _upload_artifacts(storage_client: gcs.Client, model_bucket: str, run_id: str, onnx_path: str, metadata: dict) -> None:
    bucket = storage_client.bucket(model_bucket)

    run_onnx_dest = f"runs/{run_id}/fish-id.onnx"
    _logger.info("[train] uploading %s -> gs://%s/%s", onnx_path, model_bucket, run_onnx_dest)
    bucket.blob(run_onnx_dest).upload_from_filename(onnx_path)
    _logger.info("[train] uploaded run-scoped ONNX")

    with open(_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metadata_dest = f"runs/{run_id}/metadata.json"
    _logger.info("[train] uploading metadata -> gs://%s/%s", model_bucket, metadata_dest)
    bucket.blob(metadata_dest).upload_from_filename(str(_METADATA_PATH))
    _logger.info("[train] uploaded metadata")


def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    run_id: str,
    training_bucket: str,
    model_bucket: str,
    epochs: int,
    imgsz: int,
    batch: int,
    optimizer: str,
    lr0: float,
    patience: int,
    dis: float,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    hyperparams = {
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "optimizer": optimizer,
        "lr0": lr0,
        "patience": patience,
    }

    cpu_count = os.cpu_count() or 1
    _logger.info("[train] cpu_count=%d", cpu_count)

    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    _logger.info("[train] set OMP_NUM_THREADS=%d MKL_NUM_THREADS=%d", cpu_count, cpu_count)

    _logger.info("[train] run_id=%s training_bucket=%s model_bucket=%s", run_id, training_bucket, model_bucket)

    _logger.info("[train] hyperparams: %s dis=%s", hyperparams, dis)

    _logger.info("[train] initializing GCS client")
    storage_client = gcs.Client()
    _logger.info("[train] GCS client ready")

    _logger.info("[train] reading dataset generation from gs://%s/data.yaml", training_bucket)
    dataset_generation = _read_dataset_generation(storage_client, training_bucket)
    _logger.info("[train] dataset_generation=%d", dataset_generation)

    _logger.info("[train] downloading training data to %s", _DATA_DIR)
    _download_training_data(storage_client, training_bucket, _DATA_DIR)

    data_yaml_path = str(_DATA_DIR / "data.yaml")
    _logger.info("[train] data_yaml_path=%s", data_yaml_path)

    model_bucket_obj = storage_client.bucket(model_bucket)
    teacher_checkpoint_prefix = f"runs/{run_id}/checkpoint/teacher"
    student_checkpoint_prefix = f"runs/{run_id}/checkpoint/student"

    # distill_model needs a stable local path: fresh-start passes it explicitly, but a
    # resumed student run re-reads it from its own saved args, so it must exist at the
    # same path in this container too — even if the teacher stage itself was already done.
    teacher_best_local = _download_teacher_best(model_bucket_obj, teacher_checkpoint_prefix)
    teacher_model = None
    if teacher_best_local is not None:
        _logger.info("[train] teacher already trained — reusing checkpoint for distillation")
        teacher_duration = 0.0
    else:
        _logger.info("[train] training teacher (%s, workers=%d)", _TEACHER_MODEL, cpu_count)
        start = time.time()
        teacher_model = _train_model(
            _TEACHER_MODEL,
            hyperparams,
            workers=cpu_count,
            data_yaml_path=data_yaml_path,
            checkpoint_bucket=model_bucket_obj,
            checkpoint_prefix=teacher_checkpoint_prefix,
            run_dir=_YOLO_RUNS_DIR / "teacher",
        )
        teacher_duration = time.time() - start
        _logger.info("[train] teacher training complete in %.1f seconds", teacher_duration)
        teacher_best_local = _finalize_teacher_best(model_bucket_obj, teacher_checkpoint_prefix, teacher_model)

    _logger.info("[train] training student (%s, workers=%d, distill_model=%s, dis=%s)",
                 _STUDENT_MODEL, cpu_count, teacher_best_local, dis)
    start = time.time()
    student_model = _train_model(
        _STUDENT_MODEL,
        hyperparams,
        workers=cpu_count,
        data_yaml_path=data_yaml_path,
        checkpoint_bucket=model_bucket_obj,
        checkpoint_prefix=student_checkpoint_prefix,
        run_dir=_YOLO_RUNS_DIR / "student",
        distill_model=str(teacher_best_local),
        dis=dis,
    )
    student_duration = time.time() - start
    _logger.info("[train] student training complete in %.1f seconds", student_duration)

    _logger.info("[train] exporting ONNX")
    onnx_path = _export_onnx(student_model)

    _logger.info("[train] building metadata")
    metadata = _build_metadata(
        run_id, hyperparams, dis, teacher_model, student_model,
        teacher_duration, student_duration, cpu_count, dataset_generation,
    )

    _logger.info("[train] uploading artifacts to gs://%s", model_bucket)
    _upload_artifacts(storage_client, model_bucket, run_id, onnx_path, metadata)

    _logger.info("[train] done. artifacts uploaded to gs://%s/runs/%s/", model_bucket, run_id)


