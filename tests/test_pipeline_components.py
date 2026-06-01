"""Unit tests for pipeline/pipeline.py components and pipeline compilation.

All GCP client calls are mocked. No real infrastructure or credentials required.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub packages not installed in the test environment
sys.modules.setdefault("google.cloud.aiplatform", MagicMock())
sys.modules.setdefault("google.cloud.secretmanager", MagicMock())

# Build one storage stub and wire it in two places so that
# 'from google.cloud import storage' inside component bodies always resolves to it.
# Two places are needed because test_eval.py may run first and replace
# sys.modules["google.cloud"] with a MagicMock; in that case Python satisfies
# 'from google.cloud import storage' via attribute lookup on the MagicMock
# (ignoring sys.modules["google.cloud.storage"]), so we must set both.
_storage_stub = MagicMock()
sys.modules["google.cloud.storage"] = _storage_stub  # for real namespace-package case
import google.cloud as _google_cloud  # noqa: E402
_google_cloud.storage = _storage_stub  # for MagicMock-google.cloud case

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline import (  # noqa: E402
    fish_id_pipeline,
    promote_model,
    quality_gate,
    write_production_run,
)

_STORAGE = _storage_stub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bucket(eval_map50, eval_map50_95, *, prod_run=None, prod_eval_map50=None, run_id="run-test"):
    """Return a mock GCS bucket wired to return the given metric values."""
    from google.api_core.exceptions import NotFound

    eval_blob = MagicMock()
    eval_blob.download_as_text.return_value = json.dumps(
        {"mAP50": eval_map50, "mAP50_95": eval_map50_95}
    )

    prod_blob = MagicMock()
    if prod_run is None:
        prod_blob.download_as_text.side_effect = NotFound("no production-run.json")
    else:
        prod_blob.download_as_text.return_value = json.dumps(prod_run)

    prod_eval_blob = MagicMock()
    if prod_eval_map50 is not None:
        prod_eval_blob.download_as_text.return_value = json.dumps({"mAP50": prod_eval_map50})

    def _blob(path):
        if path == "production-run.json":
            return prod_blob
        if prod_run and prod_run.get("run_id", "") in path:
            return prod_eval_blob
        return eval_blob

    bucket = MagicMock()
    bucket.blob.side_effect = _blob
    return bucket


def _run_quality_gate(eval_map50, eval_map50_95, *, prod_run=None, prod_eval_map50=None, run_id="run-test"):
    bucket = _make_bucket(
        eval_map50, eval_map50_95,
        prod_run=prod_run, prod_eval_map50=prod_eval_map50, run_id=run_id,
    )
    mock_client = MagicMock()
    mock_client.return_value.bucket.return_value = bucket
    with patch.object(_STORAGE, "Client", mock_client):
        return quality_gate.python_func(
            project="test-project", model_bucket="test-bucket", run_id=run_id
        )


# ---------------------------------------------------------------------------
# quality_gate
# ---------------------------------------------------------------------------

class TestQualityGate:
    def test_gate1_fails_when_map50_below_threshold(self):
        result = _run_quality_gate(eval_map50=0.49, eval_map50_95=0.40)
        assert result.passed == "false"

    def test_gate1_fails_when_map50_95_below_threshold(self):
        result = _run_quality_gate(eval_map50=0.60, eval_map50_95=0.34)
        assert result.passed == "false"

    def test_gate1_passes_exactly_at_thresholds(self):
        result = _run_quality_gate(eval_map50=0.50, eval_map50_95=0.35, prod_run=None)
        assert result.passed == "true"

    def test_first_run_skips_gate2(self):
        # No production-run.json → NotFound → skip regression check
        result = _run_quality_gate(eval_map50=0.55, eval_map50_95=0.40, prod_run=None)
        assert result.passed == "true"

    def test_manual_override_skips_gate2(self):
        prod_run = {"run_id": "run-prod", "manual_override": True}
        result = _run_quality_gate(
            eval_map50=0.55, eval_map50_95=0.40, prod_run=prod_run
        )
        assert result.passed == "true"

    def test_gate2_fails_on_regression_beyond_tolerance(self):
        prod_run = {"run_id": "run-prod", "manual_override": False}
        # new=0.55, prod=0.60 → drop of 0.05 exceeds 0.02 threshold
        result = _run_quality_gate(
            eval_map50=0.55, eval_map50_95=0.40,
            prod_run=prod_run, prod_eval_map50=0.60,
        )
        assert result.passed == "false"

    def test_gate2_passes_when_within_tolerance(self):
        prod_run = {"run_id": "run-prod", "manual_override": False}
        # new=0.59, prod=0.60 → drop of 0.01, within 0.02 tolerance
        result = _run_quality_gate(
            eval_map50=0.59, eval_map50_95=0.40,
            prod_run=prod_run, prod_eval_map50=0.60,
        )
        assert result.passed == "true"

    def test_gate2_passes_when_metrics_improve(self):
        prod_run = {"run_id": "run-prod", "manual_override": False}
        result = _run_quality_gate(
            eval_map50=0.65, eval_map50_95=0.45,
            prod_run=prod_run, prod_eval_map50=0.60,
        )
        assert result.passed == "true"

    def test_gate1_failure_writes_gate_failure_json(self):
        from google.api_core.exceptions import NotFound

        failure_blob = MagicMock()
        eval_blob = MagicMock()
        eval_blob.download_as_text.return_value = json.dumps({"mAP50": 0.3, "mAP50_95": 0.4})

        bucket = MagicMock()
        bucket.blob.side_effect = lambda path: (
            failure_blob if "gate_failure" in path else eval_blob
        )
        mock_client = MagicMock()
        mock_client.return_value.bucket.return_value = bucket

        with patch.object(_STORAGE, "Client", mock_client):
            quality_gate.python_func(project="p", model_bucket="b", run_id="run-x")

        written = [c.args[0] for c in bucket.blob.call_args_list]
        assert any("gate_failure.json" in p for p in written)
        failure_blob.upload_from_string.assert_called_once()

    def test_gate2_failure_writes_gate_failure_json(self):
        from google.api_core.exceptions import NotFound

        prod_run = {"run_id": "run-prod", "manual_override": False}
        failure_blob = MagicMock()

        eval_blob = MagicMock()
        eval_blob.download_as_text.return_value = json.dumps({"mAP50": 0.55, "mAP50_95": 0.40})

        prod_blob = MagicMock()
        prod_blob.download_as_text.return_value = json.dumps(prod_run)

        prod_eval_blob = MagicMock()
        prod_eval_blob.download_as_text.return_value = json.dumps({"mAP50": 0.60})

        def _blob(path):
            if "gate_failure" in path:
                return failure_blob
            if path == "production-run.json":
                return prod_blob
            if "run-prod" in path:
                return prod_eval_blob
            return eval_blob

        bucket = MagicMock()
        bucket.blob.side_effect = _blob
        mock_client = MagicMock()
        mock_client.return_value.bucket.return_value = bucket

        with patch.object(_STORAGE, "Client", mock_client):
            quality_gate.python_func(project="p", model_bucket="b", run_id="run-x")

        failure_blob.upload_from_string.assert_called_once()
        payload = json.loads(failure_blob.upload_from_string.call_args.args[0])
        assert payload["gate_failed"] == "gate2_regression"


# ---------------------------------------------------------------------------
# promote_model
# ---------------------------------------------------------------------------

class TestPromoteModel:
    def test_copies_from_run_path_to_production_path(self):
        mock_bucket = MagicMock()
        mock_client = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket

        with patch.object(_STORAGE, "Client", mock_client):
            promote_model.python_func(project="p", model_bucket="b", run_id="run-42")

        mock_bucket.copy_blob.assert_called_once()
        _src, dest_bucket, dest_name = mock_bucket.copy_blob.call_args.args
        assert dest_name == "fish-id.onnx"
        assert dest_bucket is mock_bucket

    def test_source_blob_uses_correct_run_id_path(self):
        mock_bucket = MagicMock()
        mock_client = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket

        with patch.object(_STORAGE, "Client", mock_client):
            promote_model.python_func(project="p", model_bucket="b", run_id="run-42")

        mock_bucket.blob.assert_called_with("runs/run-42/fish-id.onnx")


# ---------------------------------------------------------------------------
# write_production_run
# ---------------------------------------------------------------------------

class TestWriteProductionRun:
    def _call_and_capture(self, run_id="run-42"):
        captured = {}

        def fake_upload(data, **kwargs):
            captured["data"] = json.loads(data)

        mock_blob = MagicMock()
        mock_blob.upload_from_string.side_effect = fake_upload
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket

        with patch.object(_STORAGE, "Client", mock_client):
            write_production_run.python_func(project="p", model_bucket="b", run_id=run_id)

        return captured["data"], mock_bucket

    def test_writes_to_production_run_json(self):
        _data, mock_bucket = self._call_and_capture()
        mock_bucket.blob.assert_called_with("production-run.json")

    def test_payload_contains_run_id(self):
        data, _ = self._call_and_capture("run-42")
        assert data["run_id"] == "run-42"

    def test_payload_manual_override_is_false(self):
        data, _ = self._call_and_capture()
        assert data["manual_override"] is False

    def test_payload_contains_promoted_at(self):
        data, _ = self._call_and_capture()
        assert "promoted_at" in data


# ---------------------------------------------------------------------------
# Pipeline compilation
# ---------------------------------------------------------------------------

class TestPipelineCompilation:
    def test_compiles_without_error(self, tmp_path):
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_pipeline, str(output))
        assert output.exists()

    def test_compiled_output_is_non_trivial(self, tmp_path):
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_pipeline, str(output))
        assert output.stat().st_size > 5000

    def test_compiled_pipeline_name(self, tmp_path):
        import json as _json
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_pipeline, str(output))
        spec = _json.loads(output.read_text())
        assert spec["pipelineInfo"]["name"] == "fish-id-training-pipeline"
