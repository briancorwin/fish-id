"""Unit tests for pipeline/pipeline.py components and pipeline compilation.

All GCP client calls are mocked. No real infrastructure or credentials required.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("google.cloud.aiplatform", MagicMock())

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline import fish_id_training_pipeline, register_model  # noqa: E402


class TestPipelineCompilation:
    def test_compiles_without_error(self, tmp_path):
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_training_pipeline, str(output))
        assert output.exists()

    def test_compiled_output_is_non_trivial(self, tmp_path):
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_training_pipeline, str(output))
        assert output.stat().st_size > 1000

    def test_compiled_pipeline_name(self, tmp_path):
        import json as _json
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_training_pipeline, str(output))
        spec = _json.loads(output.read_text())
        assert spec["pipelineInfo"]["name"] == "fish-id-training-pipeline"


class TestRegisterModel:
    def _run(self, mock_aip: MagicMock, **kwargs) -> str:
        google_cloud_mock = MagicMock(aiplatform=mock_aip)
        with patch.dict(sys.modules, {
            "google.cloud": google_cloud_mock,
            "google.cloud.aiplatform": mock_aip,
        }):
            return register_model.python_func(
                project=kwargs.get("project", "test-project"),
                region=kwargs.get("region", "us-central1"),
                model_bucket=kwargs.get("model_bucket", "test-bucket"),
                run_id=kwargs.get("run_id", "run-2026-06-14-120000"),
            )

    def test_first_run_uploads_without_parent_model(self):
        mock_aip = MagicMock()
        mock_aip.Model.list.return_value = []
        mock_model = MagicMock()
        mock_model.resource_name = "projects/p/locations/us-central1/models/fish-id"
        mock_aip.Model.upload.return_value = mock_model

        result = self._run(mock_aip, run_id="run-2026-06-14-120000")

        mock_aip.Model.upload.assert_called_once()
        kwargs = mock_aip.Model.upload.call_args.kwargs
        assert "parent_model" not in kwargs
        assert kwargs["display_name"] == "fish-id"
        assert kwargs["artifact_uri"] == "gs://test-bucket/runs/run-2026-06-14-120000/"
        assert kwargs["version_description"] == "run-2026-06-14-120000"
        assert kwargs["is_default_version"] is True
        assert result == mock_model.resource_name

    def test_subsequent_run_uses_parent_model(self):
        mock_aip = MagicMock()
        existing = MagicMock()
        existing.resource_name = "projects/p/locations/us-central1/models/fish-id"
        mock_aip.Model.list.return_value = [existing]
        new_version = MagicMock()
        new_version.resource_name = "projects/p/locations/us-central1/models/fish-id/versions/2"
        mock_aip.Model.upload.return_value = new_version

        result = self._run(mock_aip, run_id="run-2026-06-14-130000")

        kwargs = mock_aip.Model.upload.call_args.kwargs
        assert kwargs["parent_model"] == existing.resource_name
        assert kwargs["version_description"] == "run-2026-06-14-130000"
        assert result == new_version.resource_name
