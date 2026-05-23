import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# main.py no longer creates the ONNX session at import time — _load_model() must
# be called explicitly. Tests inject mock state directly into module globals instead.
import main

_mock_input = MagicMock()
_mock_input.name = "images"
_mock_input.shape = [1, 3, 640, 640]

_mock_meta = MagicMock()
_mock_meta.custom_metadata_map = {
    "names": '{"0": "Largemouth Bass", "1": "Bluegill", "2": "Crappie", "3": "Catfish"}'
}

_mock_session = MagicMock()
_mock_session.get_inputs.return_value = [_mock_input]
_mock_session.get_modelmeta.return_value = _mock_meta

# Inject mock model state — bypasses _load_model() and the real ONNX file entirely.
main._session = _mock_session
main._input_name = "images"
main._input_h = 640
main._input_w = 640
main._class_names = {0: "Largemouth Bass", 1: "Bluegill", 2: "Crappie", 3: "Catfish"}
main._model_ready = True


@pytest.fixture
def client():
    main.app.config["TESTING"] = True
    with main.app.test_client() as c:
        yield c


@pytest.fixture
def mock_session():
    yield _mock_session
    _mock_session.run.reset_mock()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset per-IP token buckets between tests."""
    from rate_limiter import _limiter
    _limiter._buckets.clear()
    yield
    _limiter._buckets.clear()
