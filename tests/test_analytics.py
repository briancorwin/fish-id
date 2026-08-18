# pylint: disable=protected-access,wrong-import-order
import base64
import hashlib
import json
from unittest.mock import MagicMock

from google.api_core.exceptions import GoogleAPICallError

import analytics


class TestIsLowConfidence:
    def test_empty_detections_is_low_confidence(self):
        assert analytics._is_low_confidence([]) is True

    def test_all_above_threshold_is_not_low_confidence(self):
        detections = [{"confidence": 0.9}, {"confidence": 0.75}]
        assert analytics._is_low_confidence(detections) is False

    def test_any_below_threshold_is_low_confidence(self):
        detections = [{"confidence": 0.9}, {"confidence": 0.4}]
        assert analytics._is_low_confidence(detections) is True


class TestPublishDetectionEvent:
    def _mock_publisher(self, monkeypatch, result_side_effect=None):
        mock_future = MagicMock()
        mock_future.result.side_effect = result_side_effect
        mock_publisher = MagicMock()
        mock_publisher.publish.return_value = mock_future
        monkeypatch.setattr(analytics, "_get_publisher", lambda: mock_publisher)
        monkeypatch.setattr(analytics, "_PROJECT_ID", "test-project")
        monkeypatch.setattr(analytics, "_TOPIC_ID", "test-topic")
        return mock_publisher

    def test_noop_when_project_id_unset(self, monkeypatch):
        get_publisher = MagicMock()
        monkeypatch.setattr(analytics, "_get_publisher", get_publisher)
        monkeypatch.setattr(analytics, "_PROJECT_ID", "")
        analytics.publish_detection_event(b"data", [], "image/jpeg")
        get_publisher.assert_not_called()

    def test_noop_when_publisher_construction_fails(self, monkeypatch):
        monkeypatch.setattr(analytics, "_get_publisher", lambda: None)
        monkeypatch.setattr(analytics, "_PROJECT_ID", "test-project")
        # Should not raise even though no publisher is available.
        analytics.publish_detection_event(b"data", [], "image/jpeg")

    def test_publishes_to_expected_topic_path(self, monkeypatch):
        mock_publisher = self._mock_publisher(monkeypatch)
        analytics.publish_detection_event(b"data", [{"confidence": 0.9}], "image/jpeg")
        args, _ = mock_publisher.publish.call_args
        assert args[0] == "projects/test-project/topics/test-topic"

    def test_image_hash_matches_sha256(self, monkeypatch):
        mock_publisher = self._mock_publisher(monkeypatch)
        image_bytes = b"some-image-bytes"
        analytics.publish_detection_event(image_bytes, [{"confidence": 0.9}], "image/jpeg")
        args, _ = mock_publisher.publish.call_args
        payload = json.loads(args[1])
        assert payload["image_hash"] == hashlib.sha256(image_bytes).hexdigest()

    def test_image_b64_present_when_low_confidence(self, monkeypatch):
        mock_publisher = self._mock_publisher(monkeypatch)
        image_bytes = b"some-image-bytes"
        analytics.publish_detection_event(image_bytes, [], "image/png")
        args, _ = mock_publisher.publish.call_args
        payload = json.loads(args[1])
        assert payload["low_confidence"] is True
        assert payload["image_stored"] is True
        assert base64.b64decode(payload["image_b64"]) == image_bytes
        assert payload["image_content_type"] == "image/png"

    def test_image_b64_absent_when_confidence_high(self, monkeypatch):
        mock_publisher = self._mock_publisher(monkeypatch)
        analytics.publish_detection_event(b"data", [{"confidence": 0.9}], "image/jpeg")
        args, _ = mock_publisher.publish.call_args
        payload = json.loads(args[1])
        assert payload["low_confidence"] is False
        assert payload["image_stored"] is False
        assert "image_b64" not in payload
        assert "image_content_type" not in payload

    def test_swallows_google_api_call_error(self, monkeypatch):
        self._mock_publisher(monkeypatch, result_side_effect=GoogleAPICallError("boom"))
        analytics.publish_detection_event(b"data", [{"confidence": 0.9}], "image/jpeg")

    def test_swallows_timeout_error(self, monkeypatch):
        self._mock_publisher(monkeypatch, result_side_effect=TimeoutError("boom"))
        analytics.publish_detection_event(b"data", [{"confidence": 0.9}], "image/jpeg")
