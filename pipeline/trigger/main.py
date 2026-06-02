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

_MANIFEST_PATTERN = re.compile(r"^versions/[^/]+/manifest\.json$")


def _read_training_image(project_id: str, model_bucket: str) -> str:
    client = storage.Client(project=project_id)
    blob = client.bucket(model_bucket).blob("training-image-latest.json")
    return json.loads(blob.download_as_text())["image"]


def _make_run_id() -> str:
    return "run-" + datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")


@functions_framework.cloud_event
def trigger_pipeline(cloud_event: functions_framework.CloudEvent) -> None:
    logging.basicConfig(level=logging.INFO)

    project_id = os.environ["GCP_PROJECT_ID"]
    region = os.environ["GCP_REGION"]
    training_bucket = os.environ["TRAINING_BUCKET"]
    model_bucket = os.environ["MODEL_BUCKET"]
    pipeline_sa = os.environ["PIPELINE_SA"]
    pipeline_template_uri = os.environ["PIPELINE_TEMPLATE_URI"]
    vertex_experiment = os.environ["VERTEX_EXPERIMENT"]

    object_name: str = cloud_event.data.get("name", "")
    if not _MANIFEST_PATTERN.match(object_name):
        _logger.info("Skipping %s — not a versioned manifest", object_name)
        return

    dataset_version = object_name.split("/")[1]
    training_image = _read_training_image(project_id, model_bucket)
    run_id = _make_run_id()

    _logger.info("Starting pipeline run %s for dataset %s", run_id, dataset_version)

    aiplatform.init(project=project_id, location=region)
    pipeline_job = aiplatform.PipelineJob(
        display_name=f"fish-id-training-{run_id}",
        template_path=pipeline_template_uri,
        pipeline_root=f"gs://{model_bucket}/pipeline-root",
        parameter_values={
            "project": project_id,
            "region": region,
            "training_bucket": training_bucket,
            "model_bucket": model_bucket,
            "dataset_version": dataset_version,
            "config_version": "1",
            "training_image": training_image,
            "run_id": run_id,
            "vertex_experiment": vertex_experiment,
        },
        enable_caching=False,
    )
    pipeline_job.submit(service_account=pipeline_sa)
    _logger.info("Pipeline submitted: %s", pipeline_job.resource_name)
