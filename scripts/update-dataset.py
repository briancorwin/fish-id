#!/usr/bin/env python3
"""
Export from Roboflow and sync to the GCS training bucket.

Images and labels are written to a flat pool — no versioning or manifests.
Re-running adds new files and overwrites changed ones; it does not delete files
that were removed from a newer Roboflow version (omit -d flag intentionally).

Usage:
    python scripts/update-dataset.py \
        --roboflow-version 5 \
        --bucket ${GCP_PROJECT_ID}-fish-id-training \
        --workspace my-workspace \
        --project fish-id

Environment variables:
    ROBOFLOW_API_KEY   Roboflow API key (required)
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _rsync_to_gcs(local_dir: Path, gcs_prefix: str) -> None:
    cmd = ["gsutil", "-m", "rsync", "-r", str(local_dir), gcs_prefix]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync a Roboflow dataset version to the GCS training bucket."
    )
    parser.add_argument("--roboflow-version", required=True, type=int)
    parser.add_argument("--bucket", required=True, help="GCS training bucket name")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    args = parser.parse_args()

    api_key = os.environ["ROBOFLOW_API_KEY"]

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: pip install roboflow", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Step 1: Export from Roboflow in YOLO format
        print(
            f"Step 1: Downloading Roboflow {args.workspace}/{args.project} "
            f"version {args.roboflow_version}..."
        )
        rf = Roboflow(api_key=api_key)
        (
            rf.workspace(args.workspace)
            .project(args.project)
            .version(args.roboflow_version)
            .download("yolov8", location=str(tmp_path), overwrite=True)
        )

        candidates = list(tmp_path.rglob("train/images"))
        if not candidates:
            print("ERROR: Could not find train/images in Roboflow export.", file=sys.stderr)
            sys.exit(1)
        export_dir = candidates[0].parent.parent

        dirs = {
            "train images": export_dir / "train" / "images",
            "train labels": export_dir / "train" / "labels",
            "val images":   export_dir / "valid" / "images",
            "val labels":   export_dir / "valid" / "labels",
        }
        for name, path in dirs.items():
            if not path.exists():
                print(f"ERROR: Expected {name} directory not found: {path}", file=sys.stderr)
                sys.exit(1)

        # Step 2: Sync to GCS flat pool
        print("Step 2: Syncing to GCS...")
        gcs_base = f"gs://{args.bucket}"
        _rsync_to_gcs(dirs["train images"], f"{gcs_base}/images/train/")
        _rsync_to_gcs(dirs["train labels"], f"{gcs_base}/labels/train/")
        _rsync_to_gcs(dirs["val images"],   f"{gcs_base}/images/val/")
        _rsync_to_gcs(dirs["val labels"],   f"{gcs_base}/labels/val/")

    print(f"\nDone. Dataset synced to gs://{args.bucket}/")


if __name__ == "__main__":
    main()
