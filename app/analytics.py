import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import TypedDict

from google.api_core.exceptions import GoogleAPICallError
from google.auth.exceptions import DefaultCredentialsError
import google.cloud.pubsub_v1 as pubsub_v1

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.5
PUBLISH_TIMEOUT_SECONDS = 5.0

_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
_TOPIC_ID = os.environ.get("ANALYTICS_TOPIC_ID", "fish-id-analytics-events")

_publisher: pubsub_v1.PublisherClient | None = None


class AnalyticsEvent(TypedDict):
    timestamp: str
    image_hash: str
    fish_count: int
    detections: list[dict]
    low_confidence: bool
    image_stored: bool


def _get_publisher() -> pubsub_v1.PublisherClient | None:
    global _publisher
    if _publisher is None:
        try:
            _publisher = pubsub_v1.PublisherClient()
        except DefaultCredentialsError as e:
            logger.error("Failed to initialize Pub/Sub publisher: %s", e)
    return _publisher


def _is_low_confidence(detections: list[dict]) -> bool:
    return not detections or any(d["confidence"] < LOW_CONFIDENCE_THRESHOLD for d in detections)


def publish_detection_event(image_bytes: bytes, detections: list[dict], content_type: str) -> None:
    if not _PROJECT_ID:
        return
    publisher = _get_publisher()
    if publisher is None:
        return

    low_confidence = _is_low_confidence(detections)
    event: AnalyticsEvent = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_hash": hashlib.sha256(image_bytes).hexdigest(),
        "fish_count": len(detections),
        "detections": detections,
        "low_confidence": low_confidence,
        "image_stored": low_confidence,
    }
    payload: dict = dict(event)
    if low_confidence:
        payload["image_b64"] = base64.b64encode(image_bytes).decode("ascii")
        payload["image_content_type"] = content_type

    topic_path = f"projects/{_PROJECT_ID}/topics/{_TOPIC_ID}"
    try:
        future = publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))
        future.result(timeout=PUBLISH_TIMEOUT_SECONDS)
    except GoogleAPICallError as e:
        logger.error("Failed to publish analytics event: %s", e)
    except TimeoutError as e:
        logger.error("Timed out publishing analytics event: %s", e)
