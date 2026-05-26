"""Training script for fish-detection YOLOv8 model.

Reads env vars: JOB_MODE, RUN_ID, DATASET_VERSION, CONFIG_VERSION,
                TRAINING_BUCKET, MODEL_BUCKET, GCP_PROJECT_ID, GCP_REGION,
                CONTAINER_IMAGE, MACHINE_TYPE
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(config_version):
    config_path = f"/app/configs/c{config_version}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_manifest(storage_client, training_bucket, dataset_version):
    bucket = storage_client.bucket(training_bucket)
    blob = bucket.blob(f"versions/{dataset_version}/manifest.json")
    data = blob.download_as_bytes()
    return json.loads(data)


def download_dataset(storage_client, training_bucket, manifest):
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

    # Labels mirror the same filenames with .txt extension under labels/
    for filename in manifest.get("train_label_files", []):
        dest = Path(f"/tmp/dataset/labels/train/{filename}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(f"labels/train/{filename}")
        blob.download_to_filename(str(dest))

    for filename in manifest.get("val_label_files", []):
        dest = Path(f"/tmp/dataset/labels/val/{filename}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(f"labels/val/{filename}")
        blob.download_to_filename(str(dest))


def write_data_yaml(class_names):
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


def train_model(config):
    from ultralytics import YOLO
    results = YOLO(config["model"]).train(
        data="/tmp/dataset/data.yaml",
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
    )
    return results


def export_onnx(results):
    from ultralytics import YOLO
    best_pt = str(results.save_dir / "weights/best.pt")
    best_model = YOLO(best_pt)
    best_model.export(format="onnx")
    # YOLO exports to same dir as best.pt with .onnx extension
    onnx_path = best_pt.replace(".pt", ".onnx")
    return onnx_path


def build_metadata(run_id, dataset_version, config_version, config, results, duration_seconds):
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


def upload_artifacts(storage_client, model_bucket, run_id, onnx_path, metadata):
    bucket = storage_client.bucket(model_bucket)

    # Upload ONNX model
    onnx_dest = f"runs/{run_id}/fish-id.onnx"
    blob = bucket.blob(onnx_dest)
    blob.upload_from_filename(onnx_path)

    # Write and upload metadata
    metadata_local = "/tmp/metadata.json"
    with open(metadata_local, "w") as f:
        json.dump(metadata, f, indent=2)

    meta_dest = f"runs/{run_id}/metadata.json"
    blob = bucket.blob(meta_dest)
    blob.upload_from_filename(metadata_local)


def main():
    import google.cloud.storage as gcs

    run_id = os.environ["RUN_ID"]
    dataset_version = os.environ["DATASET_VERSION"]
    config_version = os.environ["CONFIG_VERSION"]
    training_bucket = os.environ["TRAINING_BUCKET"]
    model_bucket = os.environ["MODEL_BUCKET"]

    print(f"[train] run_id={run_id} dataset_version={dataset_version} config_version={config_version}")

    config = load_config(config_version)
    print(f"[train] config loaded: {config}")

    storage_client = gcs.Client()

    manifest = download_manifest(storage_client, training_bucket, dataset_version)
    print(f"[train] manifest: {len(manifest['train_files'])} train, {len(manifest['val_files'])} val files")

    download_dataset(storage_client, training_bucket, manifest)
    write_data_yaml(manifest["class_names"])

    start = time.time()
    results = train_model(config)
    duration = time.time() - start

    onnx_path = export_onnx(results)
    metadata = build_metadata(run_id, dataset_version, config_version, config, results, duration)
    upload_artifacts(storage_client, model_bucket, run_id, onnx_path, metadata)

    print(f"[train] done. artifacts uploaded to gs://{model_bucket}/runs/{run_id}/")


if __name__ == "__main__":
    main()
