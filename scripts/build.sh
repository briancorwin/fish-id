#!/bin/bash
set -e

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: scripts/build.sh <path-to-best.onnx> <gcp-project> [region]"
  exit 1
fi

MODEL_PATH="$1"
PROJECT="$2"
REGION="${3:-us-central1}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: model file not found: $MODEL_PATH"
  exit 1
fi

IMAGE="$REGION-docker.pkg.dev/$PROJECT/fish-id/fish-id"

cp "$MODEL_PATH" app/best.onnx
trap 'rm -f app/best.onnx' EXIT

gcloud builds submit app/ --tag "$IMAGE" --project "$PROJECT"
