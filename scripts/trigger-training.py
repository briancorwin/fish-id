#!/usr/bin/env python3
"""
Manually submit a Vertex AI training pipeline run.

Usage:
    python scripts/trigger-training.py [--image <image-uri>]

Environment variables:
    GCP_PROJECT_ID   GCP project ID (required)
    GCP_REGION       GCP region (required)
    TRAINING_BUCKET  GCS training bucket name (required)
    MODEL_BUCKET     GCS models bucket name (required)
"""

import argparse
import os
from datetime import datetime, timezone

from google.cloud import aiplatform


def _make_run_id() -> str:
    return "run-" + datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a Vertex AI training pipeline run.")
    parser.add_argument(
        "--image",
        help="Training container image URI. Defaults to the :latest tag in Artifact Registry.",
    )
    args = parser.parse_args()

    project = os.environ["GCP_PROJECT_ID"]
    region = os.environ["GCP_REGION"]
    training_bucket = os.environ["TRAINING_BUCKET"]
    model_bucket = os.environ["MODEL_BUCKET"]

    if args.image:
        training_image = args.image
    else:
        training_image = f"{region}-docker.pkg.dev/{project}/fish-id/fish-id-train:latest"

    run_id = _make_run_id()
    pipeline_template_uri = f"gs://{model_bucket}/pipeline/fish-id-training-pipeline.json"

    print(f"\nSubmitting pipeline run: {run_id}")
    print(f"  Template:        {pipeline_template_uri}")
    print(f"  Training image:  {training_image}")
    print(f"  Training bucket: {training_bucket}")
    print(f"  Model bucket:    {model_bucket}")

    aiplatform.init(project=project, location=region)
    pipeline_job = aiplatform.PipelineJob(
        display_name=f"fish-id-training-{run_id}",
        template_path=pipeline_template_uri,
        pipeline_root=f"gs://{model_bucket}/pipeline-root",
        parameter_values={
            "project": project,
            "region": region,
            "training_bucket": training_bucket,
            "model_bucket": model_bucket,
            "training_image": training_image,
            "run_id": run_id,
        },
        enable_caching=False,
    )

    workflows_sa = f"fish-id-workflows-sa@{project}.iam.gserviceaccount.com"
    pipeline_job.submit(service_account=workflows_sa)
    print(f"\nPipeline submitted: {pipeline_job.resource_name}")
    print(f"View at: https://console.cloud.google.com/vertex-ai/pipelines/runs?project={project}")


if __name__ == "__main__":
    main()
