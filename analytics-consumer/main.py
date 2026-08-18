import base64
import json
import logging
import os
from typing import Any

from flask import Flask, request, jsonify
from google.api_core.exceptions import GoogleAPICallError
import google.cloud.bigquery as bigquery
import google.cloud.storage as storage

logger = logging.getLogger(__name__)

app = Flask(__name__)

_BQ_DATASET = os.environ.get("BQ_DATASET", "fish_id_analytics")
_BQ_TABLE = os.environ.get("BQ_TABLE", "detection_events")
_IMAGES_BUCKET = os.environ.get("REVIEW_IMAGES_BUCKET", "")

_bq_client: bigquery.Client | None = None
_gcs_client: storage.Client | None = None


def _get_bq_client() -> bigquery.Client:
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client()
    return _bq_client


def _get_gcs_client() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def _store_image(image_hash: str, image_bytes: bytes, content_type: str) -> bool:
    if not _IMAGES_BUCKET:
        return False
    try:
        bucket = _get_gcs_client().bucket(_IMAGES_BUCKET)
        blob = bucket.blob(image_hash)
        if not blob.exists():
            blob.upload_from_string(image_bytes, content_type=content_type)
        return True
    except GoogleAPICallError as e:
        logger.error("GCS upload failed for hash %s: %s", image_hash, e)
        return False


def _insert_row(payload: dict[str, Any], image_stored: bool) -> bool:
    row = {
        "timestamp": payload["timestamp"],
        "image_hash": payload["image_hash"],
        "fish_count": payload["fish_count"],
        "detections": payload["detections"],
        "low_confidence": payload["low_confidence"],
        "image_stored": image_stored,
    }
    table_id = f"{_get_bq_client().project}.{_BQ_DATASET}.{_BQ_TABLE}"
    try:
        errors = _get_bq_client().insert_rows_json(table_id, [row])
    except GoogleAPICallError as e:
        logger.error("BigQuery insert failed: %s", e)
        return False
    if errors:
        logger.error("BigQuery insert reported row errors: %s", errors)
        return False
    return True


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/push", methods=["POST"])
def push():
    envelope = request.get_json(silent=True)
    if not envelope or "message" not in envelope:
        logger.error("Invalid Pub/Sub push envelope: %s", envelope)
        return jsonify({"error": "invalid envelope"}), 400

    data_b64 = envelope["message"].get("data", "")
    try:
        payload = json.loads(base64.b64decode(data_b64))
    except (ValueError, UnicodeDecodeError) as e:
        logger.error("Could not decode Pub/Sub message data: %s", e)
        return jsonify({"error": "invalid message data"}), 400

    image_b64 = payload.pop("image_b64", None)
    image_content_type = payload.pop("image_content_type", "application/octet-stream")
    image_stored = (
        _store_image(payload["image_hash"], base64.b64decode(image_b64), image_content_type)
        if image_b64
        else False
    )

    if not _insert_row(payload, image_stored):
        return jsonify({"error": "insert failed"}), 500

    return "", 204


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
