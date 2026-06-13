import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import google.cloud.storage as gcs
import yaml
from ultralytics import YOLO

_logger = logging.getLogger(__name__)


def _load_config() -> dict:
    with open("/app/config.yaml") as f:
        return yaml.safe_load(f)


def _load_class_names(storage_client, training_bucket: str) -> list[str]:
    bucket = storage_client.bucket(training_bucket)
    blob = bucket.blob("class_names.txt")
    data = blob.download_as_text()
    return [line.strip() for line in data.splitlines() if line.strip()]


def _download_dataset(storage_client, training_bucket: str) -> None:
    bucket = storage_client.bucket(training_bucket)
    splits = [
        ("images/train/", "/tmp/dataset/images/train/"),
        ("images/val/",   "/tmp/dataset/images/val/"),
        ("labels/train/", "/tmp/dataset/labels/train/"),
        ("labels/val/",   "/tmp/dataset/labels/val/"),
    ]
    for gcs_prefix, local_dir in splits:
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        for blob in bucket.list_blobs(prefix=gcs_prefix):
            filename = blob.name[len(gcs_prefix):]
            if not filename:
                continue
            blob.download_to_filename(str(local_path / filename))


def _write_data_yaml(class_names: list) -> None:
    data_yaml = {
        "path": "/tmp/dataset",
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    Path("/tmp/dataset").mkdir(parents=True, exist_ok=True)
    with open("/tmp/dataset/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)


def _train_model(config: dict):
    results = YOLO(config["model"]).train(
        data="/tmp/dataset/data.yaml",
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
    )
    return results


def _export_onnx(results) -> str:
    best_pt = str(results.save_dir / "weights/best.pt")
    best_model = YOLO(best_pt)
    best_model.export(format="onnx")
    # YOLO exports to the same directory as best.pt with .onnx extension
    onnx_path = best_pt.replace(".pt", ".onnx")
    return onnx_path


def _build_metadata(run_id: str, config: dict, results, duration_seconds: float) -> dict:
    return {
        "run_id": run_id,
        "container_image": os.environ.get("CONTAINER_IMAGE", "unknown"),
        "model_architecture": config["model"].replace(".pt", ""),
        "base_weights": config["model"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "epochs_completed": results.epoch + 1,
        "training_args": {
            "epochs": config["epochs"],
            "imgsz": config["imgsz"],
            "batch": config["batch"],
            "optimizer": config["optimizer"],
            "lr0": config["lr0"],
        },
        "final_train_loss": float(results.results_dict.get("train/box_loss", 0.0))
        if hasattr(results, "results_dict")
        else None,
        "machine_type": os.environ.get("MACHINE_TYPE", "unknown"),
    }


def _upload_artifacts(storage_client, model_bucket: str, run_id: str, onnx_path: str, metadata: dict) -> None:
    bucket = storage_client.bucket(model_bucket)

    bucket.blob(f"runs/{run_id}/fish-id.onnx").upload_from_filename(onnx_path)

    # Overwrite the production serving path directly until quality gates are in place
    bucket.blob("fish-id.onnx").upload_from_filename(onnx_path)

    metadata_local = "/tmp/metadata.json"
    with open(metadata_local, "w") as f:
        json.dump(metadata, f, indent=2)

    bucket.blob(f"runs/{run_id}/metadata.json").upload_from_filename(metadata_local)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_id = os.environ["RUN_ID"]
    training_bucket = os.environ["TRAINING_BUCKET"]
    model_bucket = os.environ["MODEL_BUCKET"]

    _logger.info("[train] run_id=%s", run_id)

    config = _load_config()
    _logger.info("[train] config loaded: %s", config)

    storage_client = gcs.Client()

    class_names = _load_class_names(storage_client, training_bucket)
    _logger.info("[train] class names: %s", class_names)

    _download_dataset(storage_client, training_bucket)
    _write_data_yaml(class_names)

    start = time.time()
    results = _train_model(config)
    duration = time.time() - start

    onnx_path = _export_onnx(results)
    metadata = _build_metadata(run_id, config, results, duration)
    _upload_artifacts(storage_client, model_bucket, run_id, onnx_path, metadata)

    _logger.info("[train] done. artifacts uploaded to gs://%s/runs/%s/", model_bucket, run_id)


if __name__ == "__main__":
    main()
