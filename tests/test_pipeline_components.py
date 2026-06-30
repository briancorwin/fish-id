"""Unit tests for pipeline/pipeline.py components and pipeline compilation.

All GCP client calls are mocked. No real infrastructure or credentials required.
"""
# pylint: disable=wrong-import-position,import-outside-toplevel
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline import (  # noqa: E402
    eval_model,
    fish_id_training_pipeline,
    promote_model,
    register_model,
    train_model,
    trigger_deploy,
)


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


class TestTrainModel:
    def test_component_has_expected_inputs(self):
        inputs = train_model.component_spec.inputs
        expected = {
            "run_id", "training_bucket", "model_bucket", "model_name",
            "epochs", "imgsz", "batch", "optimizer", "lr0",
        }
        assert set(inputs.keys()) == expected

    def test_component_uses_artifact_registry_image(self):
        image = train_model.component_spec.implementation.container.image
        assert "fish-id-train" in image


class TestEvalModel:
    def test_component_has_expected_inputs(self):
        inputs = eval_model.component_spec.inputs
        expected = {
            "run_id", "training_bucket", "model_bucket",
            "project_id", "region", "vertex_experiment",
        }
        assert set(inputs.keys()) == expected

    def test_component_uses_training_image(self):
        image = eval_model.component_spec.implementation.container.image
        assert "fish-id-train" in image


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

    def test_does_not_tag_production_alias(self):
        mock_aip = MagicMock()
        mock_aip.Model.list.return_value = []
        mock_aip.Model.upload.return_value = MagicMock(resource_name="projects/p/models/m")

        self._run(mock_aip)

        kwargs = mock_aip.Model.upload.call_args.kwargs
        assert "production" not in kwargs.get("version_aliases", [])

    def test_tags_latest_alias(self):
        mock_aip = MagicMock()
        mock_aip.Model.list.return_value = []
        mock_aip.Model.upload.return_value = MagicMock(resource_name="projects/p/models/m")

        self._run(mock_aip)

        kwargs = mock_aip.Model.upload.call_args.kwargs
        assert "latest" in kwargs.get("version_aliases", [])


class TestPromoteModel:
    def _make_mocks(self):
        mock_aip = MagicMock()
        mock_aip_v1 = MagicMock()
        mock_aip_v1_types = MagicMock()
        return mock_aip, mock_aip_v1, mock_aip_v1_types

    def _run(self, mock_aip, mock_aip_v1, mock_aip_v1_types, **kwargs):
        google_cloud_mock = MagicMock(aiplatform=mock_aip, aiplatform_v1=mock_aip_v1)
        with patch.dict(sys.modules, {
            "google.cloud": google_cloud_mock,
            "google.cloud.aiplatform": mock_aip,
            "google.cloud.aiplatform_v1": mock_aip_v1,
            "google.cloud.aiplatform_v1.types": mock_aip_v1_types,
        }):
            return promote_model.python_func(
                project=kwargs.get("project", "test-project"),
                region=kwargs.get("region", "us-central1"),
                model_resource_name=kwargs.get(
                    "model_resource_name",
                    "projects/test-project/locations/us-central1/models/123/versions/2",
                ),
                vertex_experiment=kwargs.get("vertex_experiment", "fish-id-eval"),
            )

    def test_auto_promotes_when_no_production_model_exists(self):
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-abc")
        mock_aip.Model.side_effect = [latest_model, Exception("no production alias")]

        result = self._run(mock_aip, mock_aip_v1, mock_aip_v1_types)

        assert result is True

    def test_promotes_when_current_better_than_production(self):
        import pandas as pd
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-new")
        prod_model = MagicMock(version_description="run-prod")
        mock_aip.Model.side_effect = [latest_model, prod_model]
        mock_aip.get_experiment_df.return_value = pd.DataFrame([
            {"run_name": "run-new", "metric.mAP50": 0.85},
            {"run_name": "run-prod", "metric.mAP50": 0.80},
        ])

        result = self._run(mock_aip, mock_aip_v1, mock_aip_v1_types)

        assert result is True

    def test_skips_when_current_worse_than_production(self):
        import pandas as pd
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-new")
        prod_model = MagicMock(version_description="run-prod")
        mock_aip.Model.side_effect = [latest_model, prod_model]
        mock_aip.get_experiment_df.return_value = pd.DataFrame([
            {"run_name": "run-new", "metric.mAP50": 0.60},
            {"run_name": "run-prod", "metric.mAP50": 0.80},
        ])

        result = self._run(mock_aip, mock_aip_v1, mock_aip_v1_types)

        assert result is False

    def test_calls_merge_version_aliases_on_promote(self):
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-abc")
        mock_aip.Model.side_effect = [latest_model, Exception("no production alias")]

        self._run(mock_aip, mock_aip_v1, mock_aip_v1_types)

        mock_service_client = mock_aip_v1.ModelServiceClient.return_value
        mock_service_client.merge_version_aliases.assert_called_once()


class TestTriggerDeploy:
    def _run(self, mock_sm: MagicMock, mock_requests: MagicMock, **kwargs) -> None:
        google_cloud_mock = MagicMock(secretmanager=mock_sm)
        with patch.dict(sys.modules, {
            "google.cloud": google_cloud_mock,
            "google.cloud.secretmanager": mock_sm,
            "requests": mock_requests,
        }):
            trigger_deploy.python_func(
                project=kwargs.get("project", "test-project"),
                github_repo=kwargs.get("github_repo", "owner/fish-id"),
            )

    def test_fetches_token_from_correct_secret(self):
        mock_sm = MagicMock()
        mock_sm.SecretManagerServiceClient.return_value.access_secret_version.return_value.payload.data = b"ghp_token"
        mock_req = MagicMock()

        self._run(mock_sm, mock_req, project="my-project")

        name_arg = mock_sm.SecretManagerServiceClient.return_value.access_secret_version.call_args.kwargs["name"]
        assert name_arg == "projects/my-project/secrets/fish-id-github-deploy-token/versions/latest"

    def test_dispatches_to_correct_github_url(self):
        mock_sm = MagicMock()
        mock_sm.SecretManagerServiceClient.return_value.access_secret_version.return_value.payload.data = b"ghp_token"
        mock_req = MagicMock()

        self._run(mock_sm, mock_req, github_repo="owner/fish-id")

        url = mock_req.post.call_args.args[0]
        assert url == "https://api.github.com/repos/owner/fish-id/actions/workflows/deploy-api.yml/dispatches"
        assert mock_req.post.call_args.kwargs["json"] == {"ref": "main"}
        assert mock_req.post.call_args.kwargs["headers"]["Authorization"] == "Bearer ghp_token"

    def test_raises_on_github_api_error(self):
        mock_sm = MagicMock()
        mock_sm.SecretManagerServiceClient.return_value.access_secret_version.return_value.payload.data = b"ghp_token"
        mock_req = MagicMock()
        mock_req.post.return_value.raise_for_status.side_effect = RuntimeError("403 Forbidden")

        with pytest.raises(RuntimeError, match="403"):
            self._run(mock_sm, mock_req)
