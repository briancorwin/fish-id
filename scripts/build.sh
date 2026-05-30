#!/bin/bash
set -e

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: scripts/build.sh <path-to-fish-id.onnx> <gcp-project> [region]"
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

cp "$MODEL_PATH" app/fish-id.onnx
trap 'rm -f app/fish-id.onnx' EXIT

gcloud builds submit app/ --tag "$IMAGE" --project "$PROJECT"

gcloud run deploy fish-id \
  --image "$IMAGE" \
  --region "$REGION" \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 5 \
  --max-instances 1 \
  --service-account "fish-id-cloud-run-sa@${PROJECT}.iam.gserviceaccount.com" \
  --set-env-vars "CORS_ORIGIN=https://${PROJECT}.web.app" \
  --allow-unauthenticated \
  --project "$PROJECT"

echo "WARNING: manual deploy via build.sh — recording manual_override in production-run.json"

MODEL_BUCKET="${PROJECT}-fish-id-models"
gsutil cp "$MODEL_PATH" "gs://${MODEL_BUCKET}/fish-id.onnx"

printf '{"run_id":null,"promoted_at":"%s","manual_override":true}' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | gsutil cp - "gs://${MODEL_BUCKET}/production-run.json"
