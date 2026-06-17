#!/usr/bin/env python3
# pylint: disable=invalid-name
"""
Manually submit a Vertex AI training pipeline run.

Usage:
    python scripts/trigger-training.py [--cpu-only] [--run-id <id>]

Flags:
    --cpu-only   Run on CPU only (no GPU). Useful when GPU quota is unavailable.
    --run-id     Reuse an existing run ID to restart an interrupted run. Defaults to a
                 new timestamped ID.

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
        "--cpu-only",
        action="store_true",
        help="Run on CPU only — omits GPU accelerator spec. Useful when GPU quota is unavailable.",
    )
    parser.add_argument(
        "--run-id",
        help="Reuse an existing run ID to restart an interrupted run. Defaults to a new timestamped ID.",
    )
    args = parser.parse_args()

    project = os.environ["GCP_PROJECT_ID"]
    region = os.environ["GCP_REGION"]
    training_bucket = os.environ["TRAINING_BUCKET"]
    model_bucket = os.environ["MODEL_BUCKET"]
    github_repo = os.environ["GITHUB_REPO"]

    run_id = args.run_id if args.run_id else _make_run_id()
    pipeline_template_uri = f"gs://{model_bucket}/pipeline/fish-id-training-pipeline.json"

    print(f"\nSubmitting pipeline run: {run_id}")
    print(f"  Template:        {pipeline_template_uri}")
    print(f"  Training bucket: {training_bucket}")
    print(f"  Model bucket:    {model_bucket}")
    print(f"  GPU:             {'no (CPU only)' if args.cpu_only else 'yes (T4 Spot)'}")

    aiplatform.init(project=project, location=region)
    pipeline_job = aiplatform.PipelineJob(
        display_name=f"fish-id-training-{run_id}",
        template_path=pipeline_template_uri,
        pipeline_root=f"gs://{model_bucket}/pipeline-root",
        parameter_values={
            "training_bucket": training_bucket,
            "model_bucket": model_bucket,
            "run_id": run_id,
            "project": project,
            "region": region,
            "github_repo": github_repo,
            "cpu_only": args.cpu_only,
        },
        enable_caching=False,
    )

    workflows_sa = f"fish-id-workflows-sa@{project}.iam.gserviceaccount.com"
    pipeline_job.submit(service_account=workflows_sa)
    print(f"\nPipeline submitted: {pipeline_job.resource_name}")
    print(f"View at: https://console.cloud.google.com/vertex-ai/pipelines/runs?project={project}")


if __name__ == "__main__":
    main()
