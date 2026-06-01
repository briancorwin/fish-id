"""Cloud Functions v2 trigger: GCS object finalized → Vertex AI Pipeline run."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

import functions_framework
from google.cloud import aiplatform, storage

_logger = logging.getLogger(__name__)

_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
_REGION = os.environ["GCP_REGION"]
_TRAINING_BUCKET = os.environ["TRAINING_BUCKET"]
_MODEL_BUCKET = os.environ["MODEL_BUCKET"]
_PIPELINE_SA = os.environ["PIPELINE_SA"]
_PIPELINE_TEMPLATE_URI = os.environ["PIPELINE_TEMPLATE_URI"]
_VERTEX_EXPERIMENT = os.environ["VERTEX_EXPERIMENT"]

_MANIFEST_PATTERN = re.compile(r"^versions/[^/]+/manifest\.json$")


def _read_training_image() -> str:
    client = storage.Client(project=_PROJECT_ID)
    blob = client.bucket(_MODEL_BUCKET).blob("training-image-latest.json")
    return json.loads(blob.download_as_text())["image"]


def _make_run_id() -> str:
    return "run-" + datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")


@functions_framework.cloud_event
def trigger_pipeline(cloud_event: functions_framework.CloudEvent) -> None:
    logging.basicConfig(level=logging.INFO)

    object_name: str = cloud_event.data.get("name", "")
    if not _MANIFEST_PATTERN.match(object_name):
        _logger.info("Skipping %s — not a versioned manifest", object_name)
        return

    dataset_version = object_name.split("/")[1]
    training_image = _read_training_image()
    run_id = _make_run_id()

    _logger.info("Starting pipeline run %s for dataset %s", run_id, dataset_version)

    aiplatform.init(project=_PROJECT_ID, location=_REGION)
    pipeline_job = aiplatform.PipelineJob(
        display_name=f"fish-id-training-{run_id}",
        template_path=_PIPELINE_TEMPLATE_URI,
        pipeline_root=f"gs://{_MODEL_BUCKET}/pipeline-root",
        parameter_values={
            "project": _PROJECT_ID,
            "region": _REGION,
            "training_bucket": _TRAINING_BUCKET,
            "model_bucket": _MODEL_BUCKET,
            "dataset_version": dataset_version,
            "config_version": "1",
            "training_image": training_image,
            "run_id": run_id,
            "vertex_experiment": _VERTEX_EXPERIMENT,
        },
        enable_caching=False,
    )
    pipeline_job.submit(service_account=_PIPELINE_SA)
    _logger.info("Pipeline submitted: %s", pipeline_job.resource_name)
