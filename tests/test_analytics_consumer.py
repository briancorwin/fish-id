# pylint: disable=protected-access,redefined-outer-name
import base64
import importlib.util
import json
import pathlib
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import GoogleAPICallError

_spec = importlib.util.spec_from_file_location(
    "analytics_consumer_main",
    pathlib.Path(__file__).parent.parent / "analytics-consumer" / "main.py",
)
assert _spec is not None and _spec.loader is not None
consumer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(consumer)


def _push_envelope(payload: dict) -> dict:
    data = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return {"message": {"data": data, "messageId": "1"}, "subscription": "test-sub"}


@pytest.fixture
def client():
    consumer.app.config["TESTING"] = True
    with consumer.app.test_client() as c:
        yield c


@pytest.fixture
def mock_bq(monkeypatch):
    mock_client = MagicMock()
    mock_client.project = "test-project"
    mock_client.insert_rows_json.return_value = []
    monkeypatch.setattr(consumer, "_get_bq_client", lambda: mock_client)
    return mock_client


@pytest.fixture
def mock_gcs(monkeypatch):
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    monkeypatch.setattr(consumer, "_get_gcs_client", lambda: mock_client)
    monkeypatch.setattr(consumer, "_IMAGES_BUCKET", "test-bucket")
    return mock_client, mock_bucket, mock_blob


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


class TestPushEndpoint:
    def test_valid_envelope_without_image(self, client, mock_bq):
        payload = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "image_hash": "abc123",
            "fish_count": 1,
            "detections": [{"confidence": 0.9}],
            "low_confidence": False,
        }
        resp = client.post("/push", json=_push_envelope(payload))
        assert resp.status_code == 204
        mock_bq.insert_rows_json.assert_called_once()
        table_id, rows = mock_bq.insert_rows_json.call_args[0]
        assert table_id == "test-project.fish_id_analytics.detection_events"
        assert rows[0]["image_stored"] is False

    def test_valid_envelope_with_new_image_uploads_and_dedupes(self, client, mock_bq, mock_gcs):
        _, mock_bucket, mock_blob = mock_gcs
        image_bytes = b"raw-image-bytes"
        payload = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "image_hash": "abc123",
            "fish_count": 0,
            "detections": [],
            "low_confidence": True,
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "image_content_type": "image/png",
        }
        resp = client.post("/push", json=_push_envelope(payload))
        assert resp.status_code == 204
        mock_bucket.blob.assert_called_once_with("abc123")
        mock_blob.upload_from_string.assert_called_once_with(image_bytes, content_type="image/png")
        row = mock_bq.insert_rows_json.call_args[0][1][0]
        assert row["image_stored"] is True

    def test_existing_image_skips_upload(self, client, mock_bq, mock_gcs):
        _, _, mock_blob = mock_gcs
        mock_blob.exists.return_value = True
        payload = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "image_hash": "abc123",
            "fish_count": 0,
            "detections": [],
            "low_confidence": True,
            "image_b64": base64.b64encode(b"data").decode("ascii"),
            "image_content_type": "image/jpeg",
        }
        resp = client.post("/push", json=_push_envelope(payload))
        assert resp.status_code == 204
        mock_blob.upload_from_string.assert_not_called()
        row = mock_bq.insert_rows_json.call_args[0][1][0]
        assert row["image_stored"] is True

    def test_malformed_envelope_returns_400(self, client, mock_bq):
        resp = client.post("/push", json={"not_a_message": True})
        assert resp.status_code == 400
        mock_bq.insert_rows_json.assert_not_called()

    def test_bigquery_insert_errors_return_500(self, client, mock_bq):
        mock_bq.insert_rows_json.return_value = [{"index": 0, "errors": "boom"}]
        payload = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "image_hash": "abc123",
            "fish_count": 0,
            "detections": [],
            "low_confidence": False,
        }
        resp = client.post("/push", json=_push_envelope(payload))
        assert resp.status_code == 500

    def test_gcs_failure_degrades_to_image_not_stored(self, client, mock_bq, mock_gcs):
        _, _, mock_blob = mock_gcs
        mock_blob.exists.side_effect = GoogleAPICallError("boom")
        payload = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "image_hash": "abc123",
            "fish_count": 0,
            "detections": [],
            "low_confidence": True,
            "image_b64": base64.b64encode(b"data").decode("ascii"),
            "image_content_type": "image/jpeg",
        }
        resp = client.post("/push", json=_push_envelope(payload))
        assert resp.status_code == 204
        row = mock_bq.insert_rows_json.call_args[0][1][0]
        assert row["image_stored"] is False
