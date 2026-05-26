#!/usr/bin/env python3
"""
Trigger the fish-id training pipeline via a GCP Workflow run.

Usage:
    python scripts/trigger-training.py \
        --dataset-version v5 \
        --config-version c2 \
        [--image us-central1-docker.pkg.dev/PROJECT/fish-id/fish-id-train:abc1234]
"""

import argparse
import json
import os
import subprocess
import sys

from google.cloud import storage


def get_models_bucket() -> str:
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        print("ERROR: GCP_PROJECT_ID environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return f"{project_id}-fish-id-models"


def read_latest_image(bucket_name: str) -> str:
    """Read the latest training container image URI from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob("training-image-latest.json")
    if not blob.exists():
        print(
            f"ERROR: gs://{bucket_name}/training-image-latest.json not found.",
            file=sys.stderr,
        )
        sys.exit(1)
    data = json.loads(blob.download_as_text())
    image_uri = data.get("image_uri") or data.get("image")
    if not image_uri:
        print(
            "ERROR: training-image-latest.json does not contain 'image_uri' or 'image' field.",
            file=sys.stderr,
        )
        sys.exit(1)
    return image_uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger the fish-id training workflow.")
    parser.add_argument("--dataset-version", required=True, help="Dataset version label (e.g. v5)")
    parser.add_argument("--config-version", required=True, help="Training config version label (e.g. c2)")
    parser.add_argument(
        "--image",
        required=False,
        help="Training container image URI. If omitted, reads from GCS training-image-latest.json.",
    )
    args = parser.parse_args()

    dataset_version: str = args.dataset_version
    config_version: str = args.config_version

    models_bucket = get_models_bucket()

    # Step 1: Resolve container image
    if args.image:
        image_uri = args.image
        print(f"Using provided image: {image_uri}")
    else:
        print(f"Reading image from gs://{models_bucket}/training-image-latest.json...")
        image_uri = read_latest_image(models_bucket)
        print(f"Resolved image: {image_uri}")

    # Step 2: Run the workflow
    workflow_data = json.dumps({
        "dataset_version": dataset_version,
        "config_version": config_version,
        "image": image_uri,
    })

    cmd = [
        "gcloud", "workflows", "run", "fish-id-training-pipeline",
        "--data", workflow_data,
    ]
    print(f"\nRunning workflow: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR: gcloud workflows run failed.", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Step 3: Print execution name/ID
    print(result.stdout)
    # Parse execution name from output if present
    for line in result.stdout.splitlines():
        if "name:" in line.lower() or "execution" in line.lower():
            print(f"  {line.strip()}")


if __name__ == "__main__":
    main()
