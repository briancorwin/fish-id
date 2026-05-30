#!/usr/bin/env python3
"""
Re-run eval on the current production model to establish a new performance baseline.

Usage:
    python scripts/rebaseline-production.py
"""

import json
import os
import sys
import time

from google.cloud import aiplatform, storage


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"ERROR: {name} environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return value


def read_gcs_json(client: storage.Client, bucket_name: str, blob_path: str) -> dict:
    """Read and parse a JSON blob from GCS. Exits on failure."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        print(f"ERROR: gs://{bucket_name}/{blob_path} does not exist.", file=sys.stderr)
        sys.exit(1)
    return json.loads(blob.download_as_text())


def main() -> None:
    project_id = require_env("GCP_PROJECT_ID")
    region = require_env("GCP_REGION")

    training_bucket_name = f"{project_id}-fish-id-training"
    models_bucket_name = f"{project_id}-fish-id-models"

    gcs_client = storage.Client(project=project_id)

    # Step 1: Read production-run.json
    print(f"Reading gs://{models_bucket_name}/production-run.json...")
    prod_run = read_gcs_json(gcs_client, models_bucket_name, "production-run.json")
    production_run_id = prod_run.get("run_id")
    manual_override = prod_run.get("manual_override", False)

    if manual_override or production_run_id is None:
        print(
            "WARNING: production-run.json has manual_override=true or run_id is null. "
            "Rebaseline is not applicable for manually overridden models. Exiting."
        )
        sys.exit(0)

    print(f"  Production run ID: {production_run_id}")

    # Step 2: Read eval/current.json to get current eval version
    print(f"Reading gs://{training_bucket_name}/eval/current.json...")
    eval_current = read_gcs_json(gcs_client, training_bucket_name, "eval/current.json")
    eval_version = eval_current.get("version") or eval_current.get("eval_version")
    print(f"  Current eval version: {eval_version}")

    # Step 3: Read training-image-latest.json for container image
    print(f"Reading gs://{models_bucket_name}/training-image-latest.json...")
    image_data = read_gcs_json(gcs_client, models_bucket_name, "training-image-latest.json")
    image_uri = image_data.get("image_uri") or image_data.get("image")
    if not image_uri:
        print(
            "ERROR: training-image-latest.json does not contain 'image_uri' or 'image' field.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Container image: {image_uri}")

    # Step 4: Submit Vertex AI CustomJob
    service_account = f"fish-id-training-sa@{project_id}.iam.gserviceaccount.com"

    aiplatform.init(project=project_id, location=region)

    job_display_name = f"fish-id-rebaseline-{production_run_id}-{int(time.time())}"
    print(f"\nSubmitting Vertex AI CustomJob: {job_display_name}")

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_display_name,
        container_uri=image_uri,
    )

    model = job.run(
        args=[],
        environment_variables={
            "JOB_MODE": "eval",
            "RUN_ID": production_run_id,
            "TRAINING_BUCKET": training_bucket_name,
            "MODEL_BUCKET": models_bucket_name,
            "GCP_PROJECT_ID": project_id,
            "GCP_REGION": region,
            "VERTEX_EXPERIMENT": "fish-id-eval",
        },
        machine_type="n1-highmem-4",
        service_account=service_account,
        sync=True,  # Wait for job completion (polls via SDK)
    )

    # Step 5: Print success and path to eval results
    eval_results_path = (
        f"gs://{models_bucket_name}/runs/{production_run_id}/eval_results.json"
    )
    print(f"\nRebaseline complete.")
    print(f"  Job: {job_display_name}")
    print(f"  Eval results: {eval_results_path}")


if __name__ == "__main__":
    main()
