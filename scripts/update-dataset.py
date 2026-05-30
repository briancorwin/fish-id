#!/usr/bin/env python3
"""
Update the GCS training dataset pool with a new Roboflow export and write a manifest.

Usage:
    python scripts/update-dataset.py \
        --roboflow-version 5 \
        --dataset-version v5 \
        --bucket {PROJECT_ID}-fish-id-training \
        --description "Added 200 new Bluegill images"
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
from google.cloud import storage


def get_gcloud_account() -> str:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def rsync_to_gcs(local_dir: Path, gcs_prefix: str) -> None:
    """Sync a local directory to a GCS prefix using gsutil rsync (no -d flag)."""
    cmd = ["gsutil", "-m", "rsync", "-r", str(local_dir), gcs_prefix]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"gsutil rsync failed with exit code {result.returncode}")


def list_gcs_filenames(client: storage.Client, bucket_name: str, prefix: str) -> list:
    """Return a sorted list of blob names (filenames only) under a GCS prefix."""
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=prefix)
    filenames = []
    for blob in blobs:
        name = blob.name[len(prefix):]
        if name:  # skip the prefix itself
            filenames.append(name)
    return sorted(filenames)


def get_previous_version(client: storage.Client, bucket_name: str, dataset_version: str) -> str | None:
    """Find the most recent manifest version before the current one, or None."""
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix="versions/"))
    versions = set()
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) >= 2:
            versions.add(parts[1])
    versions.discard(dataset_version)
    if not versions:
        return None
    # Return the lexicographically largest version that isn't the current one
    return sorted(versions)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload dataset version to GCS training bucket.")
    parser.add_argument("--roboflow-version", required=True, type=int, help="Roboflow dataset version number")
    parser.add_argument("--dataset-version", required=True, help="Dataset version label (e.g. v5)")
    parser.add_argument("--bucket", required=True, help="GCS training bucket name")
    parser.add_argument("--description", required=True, help="Human-readable description of changes")
    args = parser.parse_args()

    roboflow_version: int = args.roboflow_version
    dataset_version: str = args.dataset_version
    bucket_name: str = args.bucket
    description: str = args.description

    # Env vars
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE")
    project_name = os.environ.get("ROBOFLOW_PROJECT")
    for var, val in [("ROBOFLOW_API_KEY", api_key), ("ROBOFLOW_WORKSPACE", workspace), ("ROBOFLOW_PROJECT", project_name)]:
        if not val:
            print("ERROR: Required Roboflow configuration is missing.", file=sys.stderr)
            sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package is not installed. Run: pip install roboflow", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Step 1: Export from Roboflow
        print(f"Step 1: Downloading Roboflow {workspace}/{project_name} version {roboflow_version}...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(roboflow_version).download("yolov8", location=str(tmp_path))

        export_dir = tmp_path

        # Locate train/valid subdirs (Roboflow may nest inside a named folder)
        candidates = list(tmp_path.rglob("train/images"))
        if not candidates:
            print("ERROR: Could not find train/images in Roboflow export.", file=sys.stderr)
            sys.exit(1)
        export_dir = candidates[0].parent.parent  # parent of train/

        train_images = export_dir / "train" / "images"
        train_labels = export_dir / "train" / "labels"
        valid_images = export_dir / "valid" / "images"
        valid_labels = export_dir / "valid" / "labels"
        data_yaml = export_dir / "data.yaml"

        for p in [train_images, train_labels, valid_images, valid_labels, data_yaml]:
            if not p.exists():
                print(f"ERROR: Expected path does not exist: {p}", file=sys.stderr)
                sys.exit(1)

        # Step 2: Sync to GCS pool
        print("Step 2: Syncing data to GCS...")
        gcs_base = f"gs://{bucket_name}"
        rsync_to_gcs(train_images, f"{gcs_base}/images/train/")
        rsync_to_gcs(train_labels, f"{gcs_base}/labels/train/")
        rsync_to_gcs(valid_images, f"{gcs_base}/images/val/")
        rsync_to_gcs(valid_labels, f"{gcs_base}/labels/val/")

        # Step 3: Generate manifest
        print("Step 3: Generating manifest...")
        with open(data_yaml) as f:
            data_config = yaml.safe_load(f)
        class_names = data_config.get("names", [])
        if isinstance(class_names, dict):
            class_names = [class_names[k] for k in sorted(class_names)]

        gcs_client = storage.Client()

        train_files = list_gcs_filenames(gcs_client, bucket_name, "images/train/")
        val_files = list_gcs_filenames(gcs_client, bucket_name, "images/val/")

        try:
            created_by = get_gcloud_account()
        except Exception:
            created_by = "unknown"

        parent_version = get_previous_version(gcs_client, bucket_name, dataset_version)

        manifest = {
            "version": dataset_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": created_by,
            "description": description,
            "image_count": {
                "train": len(train_files),
                "valid": len(val_files),
            },
            "class_names": class_names,
            "roboflow_version": roboflow_version,
            "parent_version": parent_version,
            "train_files": train_files,
            "val_files": val_files,
        }

        # Step 4: Upload manifest (last step — Eventarc fires on this write)
        print("Step 4: Uploading manifest...")
        manifest_blob_path = f"versions/{dataset_version}/manifest.json"
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(manifest_blob_path)
        blob.upload_from_string(json.dumps(manifest, indent=2), content_type="application/json")
        print(f"Manifest uploaded to gs://{bucket_name}/{manifest_blob_path}")

    print(f"\nDataset version {dataset_version} successfully uploaded.")
    print(f"  Train images: {manifest['image_count']['train']}")
    print(f"  Val images:   {manifest['image_count']['valid']}")
    print(f"  Classes:      {class_names}")


if __name__ == "__main__":
    main()
