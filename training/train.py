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


def _load_config(config_version):
    config_path = f"/app/configs/c{config_version.lstrip('c')}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _download_manifest(storage_client, training_bucket, dataset_version):
    bucket = storage_client.bucket(training_bucket)
    blob = bucket.blob(f"versions/{dataset_version}/manifest.json")
    data = blob.download_as_bytes()
    return json.loads(data)


def _download_dataset(storage_client, training_bucket, manifest):
    bucket = storage_client.bucket(training_bucket)

    for filename in manifest["train_files"]:
        dest = Path(f"/tmp/dataset/images/train/{filename}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(f"images/train/{filename}")
        blob.download_to_filename(str(dest))

    for filename in manifest["val_files"]:
        dest = Path(f"/tmp/dataset/images/val/{filename}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(f"images/val/{filename}")
        blob.download_to_filename(str(dest))

    for filename in manifest["train_files"]:
        label_file = Path(filename).stem + ".txt"
        dest = Path(f"/tmp/dataset/labels/train/{label_file}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(f"labels/train/{label_file}")
        blob.download_to_filename(str(dest))

    for filename in manifest["val_files"]:
        label_file = Path(filename).stem + ".txt"
        dest = Path(f"/tmp/dataset/labels/val/{label_file}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(f"labels/val/{label_file}")
        blob.download_to_filename(str(dest))


def _write_data_yaml(class_names):
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


def _train_model(config):
    results = YOLO(config["model"]).train(
        data="/tmp/dataset/data.yaml",
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
    )
    return results


def _export_onnx(results):
    best_pt = str(results.save_dir / "weights/best.pt")
    best_model = YOLO(best_pt)
    best_model.export(format="onnx")
    # YOLO exports to same dir as best.pt with .onnx extension
    onnx_path = best_pt.replace(".pt", ".onnx")
    return onnx_path


def _build_metadata(run_id, dataset_version, config_version, config, results, duration_seconds):
    return {
        "run_id": run_id,
        "dataset_version": dataset_version,
        "config_version": config_version,
        "config_file": f"c{config_version}.yaml",
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


def _upload_artifacts(storage_client, model_bucket, run_id, onnx_path, metadata):
    bucket = storage_client.bucket(model_bucket)

    onnx_dest = f"runs/{run_id}/fish-id.onnx"
    blob = bucket.blob(onnx_dest)
    blob.upload_from_filename(onnx_path)

    metadata_local = "/tmp/metadata.json"
    with open(metadata_local, "w") as f:
        json.dump(metadata, f, indent=2)

    meta_dest = f"runs/{run_id}/metadata.json"
    blob = bucket.blob(meta_dest)
    blob.upload_from_filename(metadata_local)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_id = os.environ["RUN_ID"]
    dataset_version = os.environ["DATASET_VERSION"]
    config_version = os.environ["CONFIG_VERSION"]
    training_bucket = os.environ["TRAINING_BUCKET"]
    model_bucket = os.environ["MODEL_BUCKET"]

    _logger.info(f"[train] run_id={run_id} dataset_version={dataset_version} config_version={config_version}")

    config = _load_config(config_version)
    _logger.info(f"[train] config loaded: {config}")

    storage_client = gcs.Client()

    manifest = _download_manifest(storage_client, training_bucket, dataset_version)
    _logger.info(f"[train] manifest: {len(manifest['train_files'])} train, {len(manifest['val_files'])} val files")

    _download_dataset(storage_client, training_bucket, manifest)
    _write_data_yaml(manifest["class_names"])

    start = time.time()
    results = _train_model(config)
    duration = time.time() - start

    onnx_path = _export_onnx(results)
    metadata = _build_metadata(run_id, dataset_version, config_version, config, results, duration)
    _upload_artifacts(storage_client, model_bucket, run_id, onnx_path, metadata)

    _logger.info(f"[train] done. artifacts uploaded to gs://{model_bucket}/runs/{run_id}/")


if __name__ == "__main__":
    main()
