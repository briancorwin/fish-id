import time
import pytest
from unittest.mock import patch
from rate_limiter import RateLimiter


class TestRateLimiter:
    def test_allows_requests_within_burst(self):
        limiter = RateLimiter(requests_per_minute=5, burst=3)
        assert limiter.is_allowed("1.2.3.4") is True
        assert limiter.is_allowed("1.2.3.4") is True
        assert limiter.is_allowed("1.2.3.4") is True

    def test_denies_request_exceeding_burst(self):
        limiter = RateLimiter(requests_per_minute=5, burst=3)
        for _ in range(3):
            limiter.is_allowed("1.2.3.4")
        assert limiter.is_allowed("1.2.3.4") is False

    def test_independent_buckets_per_ip(self):
        limiter = RateLimiter(requests_per_minute=5, burst=1)
        assert limiter.is_allowed("1.1.1.1") is True
        assert limiter.is_allowed("1.1.1.1") is False
        assert limiter.is_allowed("2.2.2.2") is True  # different IP, fresh bucket

    def test_tokens_refill_over_time(self):
        limiter = RateLimiter(requests_per_minute=60, burst=1)
        limiter.is_allowed("1.2.3.4")  # deplete
        assert limiter.is_allowed("1.2.3.4") is False

        # Advance time by 1 second — at 60 rpm, that refills 1 token
        future = time.time() + 1.0
        with patch("rate_limiter.time.time", return_value=future):
            assert limiter.is_allowed("1.2.3.4") is True

    def test_burst_cap_respected_on_refill(self):
        limiter = RateLimiter(requests_per_minute=60, burst=2)
        # Advance time by 100 seconds — refill should cap at burst=2, not accumulate
        future = time.time() + 100.0
        with patch("rate_limiter.time.time", return_value=future):
            assert limiter.is_allowed("1.2.3.4") is True
            assert limiter.is_allowed("1.2.3.4") is True
            assert limiter.is_allowed("1.2.3.4") is False


class TestRateLimitDecorator:
    def test_returns_429_when_limit_exceeded(self, client):
        from conftest import make_jpeg
        img = make_jpeg()

        for _ in range(3):
            client.post("/detect", data={"image": (img, "test.jpg")},
                        content_type="multipart/form-data")

        resp = client.post("/detect", data={"image": (img, "test.jpg")},
                           content_type="multipart/form-data")
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.get_json()["error"]

    def test_uses_x_forwarded_for_header(self, client, mock_session):
        from conftest import make_jpeg, make_onnx_output
        mock_session.run.return_value = [make_onnx_output([])]
        img = make_jpeg()

        for _ in range(3):
            client.post("/detect",
                        data={"image": (img, "test.jpg")},
                        content_type="multipart/form-data",
                        headers={"X-Forwarded-For": "9.9.9.9"})

        resp = client.post("/detect",
                           data={"image": (img, "test.jpg")},
                           content_type="multipart/form-data",
                           headers={"X-Forwarded-For": "9.9.9.9"})
        assert resp.status_code == 429

    def test_different_ips_have_independent_limits(self, client, mock_session):
        from conftest import make_jpeg, make_onnx_output
        import io
        mock_session.run.return_value = [make_onnx_output([])]

        for _ in range(3):
            client.post("/detect",
                        data={"image": (io.BytesIO(make_jpeg()), "test.jpg")},
                        content_type="multipart/form-data",
                        headers={"X-Forwarded-For": "1.1.1.1"})

        # Different IP should still be allowed
        resp = client.post("/detect",
                           data={"image": (io.BytesIO(make_jpeg()), "test.jpg")},
                           content_type="multipart/form-data",
                           headers={"X-Forwarded-For": "2.2.2.2"})
        assert resp.status_code == 200
