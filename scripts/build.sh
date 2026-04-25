#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: scripts/build.sh <path-to-best.onnx> <gcp-project>"
  exit 1
fi

MODEL_PATH="$1"
PROJECT="$2"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: model file not found: $MODEL_PATH"
  exit 1
fi

cp "$MODEL_PATH" app/best.onnx
trap 'rm -f app/best.onnx' EXIT

gcloud builds submit app/ --tag gcr.io/"$PROJECT"/fish-id
