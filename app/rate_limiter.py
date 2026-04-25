import time
import threading
from collections import defaultdict
from functools import wraps
from flask import request, jsonify


class RateLimiter:
    def __init__(self, requests_per_minute: int = 5, burst: int = 3):
        self.rpm = requests_per_minute
        self.burst = burst
        self._buckets = defaultdict(lambda: {"tokens": burst, "last_refill": time.time()})
        self._lock = threading.Lock()

    def _refill(self, bucket: dict):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(self.burst, bucket["tokens"] + elapsed * (self.rpm / 60.0))
        bucket["last_refill"] = now

    def is_allowed(self, ip: str) -> bool:
        with self._lock:
            bucket = self._buckets[ip]
            self._refill(bucket)
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False


_limiter = RateLimiter(requests_per_minute=5, burst=3)


def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
        if not _limiter.is_allowed(ip):
            return jsonify({"error": "Rate limit exceeded. Please wait before sending another image."}), 429
        return f(*args, **kwargs)
    return decorated
