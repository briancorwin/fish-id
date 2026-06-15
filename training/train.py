import argparse
import concurrent.futures
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import google.cloud.storage as gcs
import torch
import yaml
from ultralytics import YOLO

_logger = logging.getLogger(__name__)


def _load_config() -> dict:
    with open("/app/config.yaml") as f:
        return yaml.safe_load(f)


def _train_model(config: dict, workers: int, data_yaml_path: str, gcs_checkpoint_dir: str) -> YOLO:
    last_checkpoint_path = os.path.join(gcs_checkpoint_dir, "weights", "last.pt")

    if os.path.exists(last_checkpoint_path):
        _logger.info("[train] checkpoint found — resuming from %s", last_checkpoint_path)
        model = YOLO(last_checkpoint_path)
        model.train(resume=True)
    else:
        _logger.info(
            "[train] no checkpoint found — starting fresh: model=%s epochs=%s imgsz=%s batch=%s optimizer=%s lr0=%s workers=%d",
            config["model"], config["epochs"], config["imgsz"], config["batch"],
            config["optimizer"], config["lr0"], workers,
        )
        model = YOLO(config["model"])
        model.train(
            data=data_yaml_path,
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            optimizer=config["optimizer"],
            lr0=config["lr0"],
            workers=workers,
            cache=False,
            project=gcs_checkpoint_dir,
            name=".",
            save=True,
            save_period=1,
        )

    _logger.info("[train] YOLO training finished. save_dir=%s", model.trainer.save_dir)
    return model


def _export_onnx(model: YOLO) -> str:
    best_pt = str(model.trainer.save_dir / "weights/best.pt")
    _logger.info("[train] exporting ONNX from %s", best_pt)
    best_model = YOLO(best_pt)
    best_model.export(format="onnx")
    onnx_path = best_pt.replace(".pt", ".onnx")
    _logger.info("[train] ONNX exported to %s", onnx_path)
    return onnx_path


def _read_image_tag() -> str:
    tag_file = Path("/app/image_tag.txt")
    if tag_file.exists():
        return tag_file.read_text().strip()
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


def _build_metadata(run_id: str, config: dict, model: YOLO, duration_seconds: float, cpu_count: int, dataset_generation: int) -> dict:
    trainer = model.trainer
    _logger.info("[train] building metadata for run_id=%s duration=%.1fs", run_id, duration_seconds)
    metadata = {
        "run_id": run_id,
        "dataset_generation": dataset_generation,
        "container_image": _read_image_tag(),
        "model_architecture": config["model"].replace(".pt", ""),
        "base_weights": config["model"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "epochs_completed": trainer.epoch + 1,
        "training_args": {
            "epochs": config["epochs"],
            "imgsz": config["imgsz"],
            "batch": config["batch"],
            "optimizer": config["optimizer"],
            "lr0": config["lr0"],
        },
        "final_train_loss": float(trainer.metrics.get("train/box_loss", 0.0))
        if hasattr(trainer, "metrics")
        else None,
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

    prod_onnx_dest = "fish-id.onnx"
    _logger.info("[train] uploading %s -> gs://%s/%s (production path)", onnx_path, model_bucket, prod_onnx_dest)
    # Overwrite the production serving path directly until quality gates are in place
    bucket.blob(prod_onnx_dest).upload_from_filename(onnx_path)
    _logger.info("[train] uploaded production ONNX")

    metadata_local = "/tmp/metadata.json"
    with open(metadata_local, "w") as f:
        json.dump(metadata, f, indent=2)

    metadata_dest = f"runs/{run_id}/metadata.json"
    _logger.info("[train] uploading metadata -> gs://%s/%s", model_bucket, metadata_dest)
    bucket.blob(metadata_dest).upload_from_filename(metadata_local)
    _logger.info("[train] uploaded metadata")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-bucket", required=True)
    parser.add_argument("--model-bucket", required=True)
    parsed = parser.parse_args()

    run_id = parsed.run_id
    training_bucket = parsed.training_bucket
    model_bucket = parsed.model_bucket

    cpu_count = os.cpu_count() or 1
    _logger.info("[train] cpu_count=%d", cpu_count)

    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    _logger.info("[train] set OMP_NUM_THREADS=%d MKL_NUM_THREADS=%d", cpu_count, cpu_count)

    _logger.info("[train] run_id=%s training_bucket=%s model_bucket=%s", run_id, training_bucket, model_bucket)

    _logger.info("[train] loading config")
    config = _load_config()
    _logger.info("[train] config loaded: %s", config)

    _logger.info("[train] initializing GCS client")
    storage_client = gcs.Client()
    _logger.info("[train] GCS client ready")

    _logger.info("[train] reading dataset generation from gs://%s/data.yaml", training_bucket)
    dataset_generation = _read_dataset_generation(storage_client, training_bucket)
    _logger.info("[train] dataset_generation=%d", dataset_generation)

    local_data_dir = Path("/app/data")
    _logger.info("[train] downloading training data to %s", local_data_dir)
    _download_training_data(storage_client, training_bucket, local_data_dir)

    data_yaml_path = str(local_data_dir / "data.yaml")
    _logger.info("[train] data_yaml_path=%s", data_yaml_path)

    # FUSE path so YOLO can write checkpoints directly to GCS each epoch without
    # hooking into its internal save logic; all other GCS access uses the client.
    gcs_checkpoint_dir = f"/gcs/{model_bucket}/runs/{run_id}"
    _logger.info("[train] checkpoint dir=%s", gcs_checkpoint_dir)

    _logger.info("[train] starting training (workers=%d)", cpu_count)
    start = time.time()
    model = _train_model(config, workers=cpu_count, data_yaml_path=data_yaml_path, gcs_checkpoint_dir=gcs_checkpoint_dir)
    duration = time.time() - start
    _logger.info("[train] training complete in %.1f seconds", duration)

    _logger.info("[train] exporting ONNX")
    onnx_path = _export_onnx(model)

    _logger.info("[train] building metadata")
    metadata = _build_metadata(run_id, config, model, duration, cpu_count, dataset_generation)

    _logger.info("[train] uploading artifacts to gs://%s", model_bucket)
    _upload_artifacts(storage_client, model_bucket, run_id, onnx_path, metadata)

    _logger.info("[train] done. artifacts uploaded to gs://%s/runs/%s/", model_bucket, run_id)


if __name__ == "__main__":
    main()
