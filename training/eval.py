import concurrent.futures
import logging
from pathlib import Path

import google.cloud.storage as gcs
import yaml
from google.cloud import aiplatform
from ultralytics import YOLO

_logger = logging.getLogger(__name__)

_DATA_DIR = Path("/tmp/eval-data")
_MODEL_PATH = Path("/tmp/fish-id.onnx")


def _download_prefix(bucket: gcs.Bucket, gcs_prefix: str, local_dir: Path) -> None:
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    _logger.info("[eval] downloading %d files from %s", len(blobs), gcs_prefix)

    def _fetch(blob: gcs.Blob) -> None:
        filename = blob.name[len(gcs_prefix):]
        if not filename:
            return
        dest = local_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(_fetch, blobs))


def _download_eval_data(storage_client: gcs.Client, training_bucket: str, local_dir: Path) -> None:
    bucket = storage_client.bucket(training_bucket)
    local_dir.mkdir(parents=True, exist_ok=True)
    _download_prefix(bucket, "images/eval/", local_dir / "images")
    _download_prefix(bucket, "labels/eval/", local_dir / "labels")
    image_count = sum(1 for p in (local_dir / "images").iterdir() if p.is_file())
    if image_count == 0:
        raise RuntimeError(
            f"No eval images found at gs://{training_bucket}/images/eval/ — "
            "run update-dataset.py to sync the eval split first"
        )
    _logger.info("[eval] downloaded %d eval images", image_count)


def _get_class_names(storage_client: gcs.Client, training_bucket: str) -> list[str]:
    data = yaml.safe_load(
        storage_client.bucket(training_bucket).blob("data.yaml").download_as_bytes()
    )
    return data.get("names", [])


def _write_data_yaml(class_names: list[str], local_dir: Path) -> str:
    path = local_dir / "data.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            {"path": str(local_dir), "train": "images", "val": "images",
             "nc": len(class_names), "names": class_names},
            f,
        )
    return str(path)


def _download_model(storage_client: gcs.Client, model_bucket: str, run_id: str) -> None:
    storage_client.bucket(model_bucket).blob(
        f"runs/{run_id}/fish-id.onnx"
    ).download_to_filename(str(_MODEL_PATH))
    _logger.info("[eval] model downloaded from gs://%s/runs/%s/fish-id.onnx", model_bucket, run_id)


def _run_validation(data_yaml_path: str) -> dict:
    model = YOLO(str(_MODEL_PATH))
    results = model.val(data=data_yaml_path)
    return {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class_map50": results.box.ap50.tolist(),
    }


def _log_to_vertex(
    project_id: str, region: str, experiment: str, run_id: str, metrics: dict
) -> None:
    aiplatform.init(project=project_id, location=region, experiment=experiment)
    with aiplatform.start_run(run_id, resume=True):
        aiplatform.log_metrics({
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })
    _logger.info("[eval] metrics logged to Vertex AI experiment=%s run=%s", experiment, run_id)


def run(
    run_id: str,
    training_bucket: str,
    model_bucket: str,
    project_id: str,
    region: str,
    vertex_experiment: str,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _logger.info("[eval] run_id=%s", run_id)

    storage_client = gcs.Client()

    _download_eval_data(storage_client, training_bucket, _DATA_DIR)
    class_names = _get_class_names(storage_client, training_bucket)
    data_yaml_path = _write_data_yaml(class_names, _DATA_DIR)
    _download_model(storage_client, model_bucket, run_id)

    _logger.info("[eval] running YOLO validation")
    metrics = _run_validation(data_yaml_path)
    _logger.info("[eval] metrics: %s", metrics)

    _log_to_vertex(project_id, region, vertex_experiment, run_id, metrics)

    _logger.info("[eval] done")
