#!/bin/bash
set -euo pipefail

TRAINING_BUCKET=""
MODEL_BUCKET=""

i=1
while [[ $i -le $# ]]; do
    case "${!i}" in
        --training-bucket)
            i=$((i+1))
            TRAINING_BUCKET="${!i}"
            ;;
        --model-bucket)
            i=$((i+1))
            MODEL_BUCKET="${!i}"
            ;;
    esac
    i=$((i+1))
done

[[ -n "$TRAINING_BUCKET" ]] || { echo "ERROR: --training-bucket required" >&2; exit 1; }
[[ -n "$MODEL_BUCKET" ]] || { echo "ERROR: --model-bucket required" >&2; exit 1; }

mount_bucket() {
    local bucket="$1"
    local mountpoint="/gcs/${bucket}"
    mkdir -p "${mountpoint}"
    gcsfuse --implicit-dirs "${bucket}" "${mountpoint}"
}

mount_bucket "${TRAINING_BUCKET}"
[[ "${MODEL_BUCKET}" != "${TRAINING_BUCKET}" ]] && mount_bucket "${MODEL_BUCKET}"

exec python /app/train.py "$@"
