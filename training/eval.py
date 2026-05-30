import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import google.cloud.storage as gcs
import yaml
from google.cloud import aiplatform
from ultralytics import YOLO

_logger = logging.getLogger(__name__)


def _download_eval_current(storage_client, training_bucket):
    bucket = storage_client.bucket(training_bucket)
    blob = bucket.blob("eval/current.json")
    data = blob.download_as_bytes()
    current = json.loads(data)
    return current["eval_version"]


def _download_eval_manifest(storage_client, training_bucket, eval_version):
    bucket = storage_client.bucket(training_bucket)
    blob = bucket.blob(f"eval/versions/{eval_version}/manifest.json")
    data = blob.download_as_bytes()
    return json.loads(data)


def _download_eval_files(storage_client, training_bucket, manifest):
    bucket = storage_client.bucket(training_bucket)

    # Support either "eval_files" or fall back to "train_files"
    image_files = manifest.get("eval_files", manifest.get("train_files", []))

    for filename in image_files:
        dest = Path(f"/tmp/eval/images/{filename}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(f"eval/images/{filename}")
        blob.download_to_filename(str(dest))

    for filename in image_files:
        label_file = Path(filename).stem + ".txt"
        dest = Path(f"/tmp/eval/labels/{label_file}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(f"eval/labels/{label_file}")
        blob.download_to_filename(str(dest))

    return image_files


def _write_eval_data_yaml(class_names):
    data_yaml = {
        "path": "/tmp/eval",
        "train": "images",
        "val": "images",
        "nc": len(class_names),
        "names": class_names,
    }
    Path("/tmp/eval").mkdir(parents=True, exist_ok=True)
    with open("/tmp/eval/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)


def _download_model(storage_client, model_bucket, run_id):
    bucket = storage_client.bucket(model_bucket)
    blob = bucket.blob(f"runs/{run_id}/fish-id.onnx")
    blob.download_to_filename("/tmp/fish-id.onnx")


def _run_validation():
    model = YOLO("/tmp/fish-id.onnx")
    results = model.val(data="/tmp/eval/data.yaml")
    return results


def _extract_metrics(results):
    return {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class_map50": results.box.ap50.tolist(),
    }


def _upload_eval_results(storage_client, model_bucket, run_id, metrics, eval_version):
    payload = {
        **metrics,
        "eval_version": eval_version,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
    }

    local_path = "/tmp/eval_results.json"
    with open(local_path, "w") as f:
        json.dump(payload, f, indent=2)

    bucket = storage_client.bucket(model_bucket)
    blob = bucket.blob(f"runs/{run_id}/eval_results.json")
    blob.upload_from_filename(local_path)

    return payload


def _log_to_vertex(project_id, region, experiment, run_id, metrics):
    aiplatform.init(project=project_id, location=region, experiment=experiment)
    with aiplatform.start_run(run_id):
        aiplatform.log_metrics({
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_id = os.environ["RUN_ID"]
    training_bucket = os.environ["TRAINING_BUCKET"]
    model_bucket = os.environ["MODEL_BUCKET"]
    project_id = os.environ["GCP_PROJECT_ID"]
    region = os.environ["GCP_REGION"]
    experiment = os.environ.get("VERTEX_EXPERIMENT", "fish-id-eval")

    _logger.info(f"[eval] run_id={run_id}")

    storage_client = gcs.Client()

    eval_version = _download_eval_current(storage_client, training_bucket)
    _logger.info(f"[eval] eval_version={eval_version}")

    manifest = _download_eval_manifest(storage_client, training_bucket, eval_version)
    _download_eval_files(storage_client, training_bucket, manifest)
    _write_eval_data_yaml(manifest["class_names"])

    _download_model(storage_client, model_bucket, run_id)

    results = _run_validation()
    metrics = _extract_metrics(results)
    _logger.info(f"[eval] metrics: {metrics}")

    _upload_eval_results(storage_client, model_bucket, run_id, metrics, eval_version)
    _log_to_vertex(project_id, region, experiment, run_id, metrics)

    _logger.info("[eval] done.")


if __name__ == "__main__":
    main()
