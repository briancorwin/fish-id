#!/usr/bin/env python3
"""
Run the fish-id training pipeline locally using the KFP SubprocessRunner.

Executes the pipeline graph without Vertex AI. The run_training_job component
will still attempt to submit a Vertex AI CustomJob unless SHORT_CIRCUIT=true
is set, in which case it skips submission and logs a no-op instead.

Usage:
    python scripts/run-pipeline-local.py [--image <image-uri>]

Environment variables:
    GCP_PROJECT_ID   GCP project ID (required)
    GCP_REGION       GCP region (required)
    TRAINING_BUCKET  GCS training bucket name (required)
    MODEL_BUCKET     GCS models bucket name (required)
    SHORT_CIRCUIT    Set to 'true' to skip CustomJob submission (optional)
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_logger = logging.getLogger(__name__)


def _make_run_id() -> str:
    return "run-" + datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Run the fish-id training pipeline locally.")
    parser.add_argument(
        "--image",
        help="Training container image URI. Defaults to the :latest tag in Artifact Registry.",
    )
    args = parser.parse_args()

    project = os.environ["GCP_PROJECT_ID"]
    region = os.environ["GCP_REGION"]
    training_bucket = os.environ["TRAINING_BUCKET"]
    model_bucket = os.environ["MODEL_BUCKET"]
    short_circuit = os.environ.get("SHORT_CIRCUIT", "").lower() == "true"

    training_image = (
        args.image
        if args.image
        else f"{region}-docker.pkg.dev/{project}/fish-id/fish-id-train:latest"
    )

    run_id = _make_run_id()

    _logger.info("Local pipeline run: %s", run_id)
    _logger.info("  Training image:  %s", training_image)
    _logger.info("  Training bucket: %s", training_bucket)
    _logger.info("  Model bucket:    %s", model_bucket)
    if short_circuit:
        _logger.info("  SHORT_CIRCUIT=true — CustomJob submission will be skipped")

    # Add repo root to sys.path so `pipeline.pipeline` resolves without install
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from kfp.local import SubprocessRunner, init  # noqa: PLC0415

    init(runner=SubprocessRunner(use_venv=True))

    from pipeline.pipeline import fish_id_training_pipeline  # noqa: PLC0415

    fish_id_training_pipeline(
        project=project,
        region=region,
        training_bucket=training_bucket,
        model_bucket=model_bucket,
        training_image=training_image,
        run_id=run_id,
    )


if __name__ == "__main__":
    main()
