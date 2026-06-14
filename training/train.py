import argparse
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
    _logger.info("[train] loading class_names.txt from gs://%s/class_names.txt", training_bucket)
    bucket = storage_client.bucket(training_bucket)
    blob = bucket.blob("class_names.txt")
    data = blob.download_as_text()
    names = [line.strip() for line in data.splitlines() if line.strip()]
    _logger.info("[train] loaded %d class names: %s", len(names), names)
    return names


def _download_dataset(storage_client, training_bucket: str) -> None:
    bucket = storage_client.bucket(training_bucket)
    splits = [
        ("images/train/", "/tmp/dataset/images/train/"),
        ("images/val/",   "/tmp/dataset/images/val/"),
        ("labels/train/", "/tmp/dataset/labels/train/"),
        ("labels/val/",   "/tmp/dataset/labels/val/"),
    ]
    for gcs_prefix, local_dir in splits:
        _logger.info("[train] downloading split %s -> %s", gcs_prefix, local_dir)
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        _logger.info("[train] found %d blobs under gs://%s/%s", len(blobs), training_bucket, gcs_prefix)
        for i, blob in enumerate(blobs):
            filename = blob.name[len(gcs_prefix):]
            if not filename:
                continue
            dest = str(local_path / filename)
            _logger.info("[train] [%d/%d] downloading %s -> %s", i + 1, len(blobs), blob.name, dest)
            try:
                blob.download_to_filename(dest)
            except Exception as exc:
                _logger.error("[train] failed to download %s -> %s: %s", blob.name, dest, exc)
                raise
        _logger.info("[train] split %s download complete", gcs_prefix)


def _write_data_yaml(class_names: list[str]) -> None:
    _logger.info("[train] writing data.yaml with %d classes", len(class_names))
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
    _logger.info("[train] data.yaml written to /tmp/dataset/data.yaml")


def _train_model(config: dict, workers: int):
    _logger.info(
        "[train] starting YOLO training: model=%s epochs=%s imgsz=%s batch=%s optimizer=%s lr0=%s workers=%d",
        config["model"], config["epochs"], config["imgsz"], config["batch"],
        config["optimizer"], config["lr0"], workers,
    )
    results = YOLO(config["model"]).train(
        data="/tmp/dataset/data.yaml",
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        workers=workers,
    )
    _logger.info("[train] YOLO training finished. save_dir=%s", results.save_dir)
    return results


def _export_onnx(results) -> str:
    best_pt = str(results.save_dir / "weights/best.pt")
    _logger.info("[train] exporting ONNX from %s", best_pt)
    best_model = YOLO(best_pt)
    best_model.export(format="onnx")
    onnx_path = best_pt.replace(".pt", ".onnx")
    _logger.info("[train] ONNX exported to %s", onnx_path)
    return onnx_path


def _build_metadata(run_id: str, config: dict, results, duration_seconds: float) -> dict:
    _logger.info("[train] building metadata for run_id=%s duration=%.1fs", run_id, duration_seconds)
    metadata = {
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
    _logger.info("[train] metadata: %s", metadata)
    return metadata


def _upload_artifacts(storage_client, model_bucket: str, run_id: str, onnx_path: str, metadata: dict) -> None:
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

    cpu_count = int(float(os.environ.get("AIP_REPLICA_CPU_CORES", 0))) or os.cpu_count() or 1
    _logger.info("[train] cpu_count=%d (AIP_REPLICA_CPU_CORES=%s)", cpu_count, os.environ.get("AIP_REPLICA_CPU_CORES", "unset"))

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

    class_names = _load_class_names(storage_client, training_bucket)

    _logger.info("[train] downloading dataset from gs://%s", training_bucket)
    _download_dataset(storage_client, training_bucket)
    _logger.info("[train] dataset download complete")

    _logger.info("[train] writing data.yaml")
    _write_data_yaml(class_names)

    _logger.info("[train] starting training (workers=%d)", cpu_count)
    start = time.time()
    results = _train_model(config, workers=cpu_count)
    duration = time.time() - start
    _logger.info("[train] training complete in %.1f seconds", duration)

    _logger.info("[train] exporting ONNX")
    onnx_path = _export_onnx(results)

    _logger.info("[train] building metadata")
    metadata = _build_metadata(run_id, config, results, duration)

    _logger.info("[train] uploading artifacts to gs://%s", model_bucket)
    _upload_artifacts(storage_client, model_bucket, run_id, onnx_path, metadata)

    _logger.info("[train] done. artifacts uploaded to gs://%s/runs/%s/", model_bucket, run_id)


if __name__ == "__main__":
    main()
