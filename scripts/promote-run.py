#!/usr/bin/env python3
"""
Promote a trained model run to production.

Usage:
    python scripts/promote-run.py --run-id run-2026-05-24-093015
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

from google.cloud import storage


def get_models_bucket() -> str:
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        print("ERROR: GCP_PROJECT_ID environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return f"{project_id}-fish-id-models"


def blob_exists(bucket: storage.Bucket, path: str) -> bool:
    return bucket.blob(path).exists()


def trigger_deploy_workflow(run_id: str) -> None:
    """POST a workflow_dispatch event to GitHub Actions deploy.yml."""
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    url = "https://api.github.com/repos/briancorwin/fish-id/actions/workflows/deploy.yml/dispatches"
    payload = json.dumps({
        "ref": "main",
        "inputs": {"run_id": run_id},
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: GitHub API returned {e.code}: {body}", file=sys.stderr)
        sys.exit(1)

    if status not in (200, 204):
        print(f"ERROR: GitHub API returned unexpected status {status}.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model run to production.")
    parser.add_argument("--run-id", required=True, help="Run ID to promote (e.g. run-2026-05-24-093015)")
    args = parser.parse_args()
    run_id: str = args.run_id

    models_bucket_name = get_models_bucket()
    client = storage.Client()
    bucket = client.bucket(models_bucket_name)

    # Step 1: Verify the ONNX model exists for this run
    model_path = f"runs/{run_id}/fish-id.onnx"
    print(f"Checking gs://{models_bucket_name}/{model_path}...")
    if not blob_exists(bucket, model_path):
        print(
            f"ERROR: gs://{models_bucket_name}/{model_path} does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("  Model found.")

    # Step 2: Warn that quality gates are bypassed and prompt for confirmation
    print("\nWARNING: Promoting this run bypasses automated quality gates.")
    print(f"  Run ID: {run_id}")
    print(f"  Bucket: {models_bucket_name}")
    answer = input("Are you sure you want to promote this run to production? [y/N] ").strip().lower()
    if answer != "y":
        print("Promotion cancelled.")
        sys.exit(0)

    # Step 3: Copy runs/{run_id}/fish-id.onnx to bucket root fish-id.onnx
    print("\nCopying model to production location...")
    src_blob = bucket.blob(model_path)
    bucket.copy_blob(src_blob, bucket, "fish-id.onnx")
    print(f"  Copied to gs://{models_bucket_name}/fish-id.onnx")

    # Step 4: Overwrite production-run.json
    production_run = {
        "run_id": run_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "manual_override": False,
    }
    prod_blob = bucket.blob("production-run.json")
    prod_blob.upload_from_string(
        json.dumps(production_run, indent=2),
        content_type="application/json",
    )
    print(f"  Updated gs://{models_bucket_name}/production-run.json")

    # Step 5: Trigger deploy workflow via GitHub API
    print("\nTriggering deploy workflow on GitHub...")
    trigger_deploy_workflow(run_id)
    print("  Deploy workflow dispatched (ref=main).")

    # Step 6: Print confirmation
    print(f"\nPromotion complete. Run '{run_id}' is now the production model.")


if __name__ == "__main__":
    main()
