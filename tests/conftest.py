# pylint: disable=protected-access,wrong-import-position,import-outside-toplevel,wrong-import-order
import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))  # makes helpers importable as a top-level module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import main
from fish_identifier import FishIdentifier

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

# Build a FishIdentifier bypassing __init__ and inject mock state.
_mock_identifier = object.__new__(FishIdentifier)
_mock_identifier._session = _mock_session
_mock_identifier._input_name = "images"
_mock_identifier._input_h = 640
_mock_identifier._input_w = 640
_mock_identifier._class_names = {0: "Largemouth Bass", 1: "Bluegill", 2: "Crappie", 3: "Catfish"}

main._identifier = _mock_identifier


@pytest.fixture
def client():
    main.app.config["TESTING"] = True
    with main.app.test_client() as c:
        yield c


@pytest.fixture
def mock_session():
    yield _mock_session
    _mock_session.run.side_effect = None
    _mock_session.run.reset_mock()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset per-IP token buckets between tests."""
    from rate_limiter import _limiter
    _limiter._buckets.clear()
    yield
    _limiter._buckets.clear()
