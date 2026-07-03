"""Unit tests for pipeline/pipeline.py components and pipeline compilation.

All GCP client calls are mocked. No real infrastructure or credentials required.
"""
# pylint: disable=wrong-import-position,import-outside-toplevel
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import NotFound

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline import (  # noqa: E402
    eval_latest_and_production_models,
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

    def test_eval_latest_and_production_models_task_retries(self, tmp_path):
        import json as _json
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_training_pipeline, str(output))
        spec = _json.loads(output.read_text())

        # tasks live inside the cpu/gpu conditional branches, not at the pipeline root
        eval_retry_counts = [
            task["retryPolicy"]["maxRetryCount"]
            for comp in spec["components"].values()
            for task_name, task in comp.get("dag", {}).get("tasks", {}).items()
            if task_name.startswith("eval-latest-and-production-models")
        ]
        assert eval_retry_counts, "expected eval_latest_and_production_models tasks in the compiled spec"
        assert all(count == 3 for count in eval_retry_counts)


class TestTrainModel:
    def test_component_has_expected_inputs(self):
        inputs = train_model.component_spec.inputs
        expected = {
            "run_id", "training_bucket", "model_bucket",
            "epochs", "imgsz", "batch", "optimizer", "lr0", "patience",
            "dis",
        }
        assert set(inputs.keys()) == expected

    def test_component_uses_artifact_registry_image(self):
        image = train_model.component_spec.implementation.container.image
        assert "fish-id-train" in image


class TestEvalLatestAndProductionModels:
    def _make_mocks(self):
        mock_aip = MagicMock()
        mock_eval = MagicMock()
        return mock_aip, mock_eval

    def _run(self, mock_aip, mock_eval, **kwargs):
        google_cloud_mock = MagicMock(aiplatform=mock_aip)
        with patch.dict(sys.modules, {
            "google.cloud": google_cloud_mock,
            "google.cloud.aiplatform": mock_aip,
            "eval": mock_eval,
        }):
            return eval_latest_and_production_models.python_func(
                run_id=kwargs.get("run_id", "run-new"),
                training_bucket=kwargs.get("training_bucket", "test-training-bucket"),
                model_bucket=kwargs.get("model_bucket", "test-model-bucket"),
                project_id=kwargs.get("project_id", "test-project"),
                region=kwargs.get("region", "us-central1"),
                vertex_experiment=kwargs.get("vertex_experiment", "fish-id-eval"),
                model_resource_name=kwargs.get(
                    "model_resource_name",
                    "projects/test-project/locations/us-central1/models/123/versions/2",
                ),
            )

    def test_component_has_expected_inputs(self):
        inputs = eval_latest_and_production_models.component_spec.inputs
        expected = {
            "run_id", "training_bucket", "model_bucket",
            "project_id", "region", "vertex_experiment", "model_resource_name",
        }
        assert set(inputs.keys()) == expected

    def test_component_uses_training_image(self):
        image = eval_latest_and_production_models.component_spec.implementation.container.image
        assert "fish-id-train" in image

    def test_evaluates_current_run_id(self):
        mock_aip, mock_eval = self._make_mocks()
        mock_aip.Model.side_effect = NotFound("no production alias")
        mock_eval.run.return_value = {"mAP50": 0.85, "dataset_generation": 42}

        self._run(mock_aip, mock_eval, run_id="run-new")

        first_call_kwargs = mock_eval.run.call_args_list[0].kwargs
        assert first_call_kwargs["run_id"] == "run-new"

    def test_returns_dataset_generation(self):
        mock_aip, mock_eval = self._make_mocks()
        mock_aip.Model.side_effect = NotFound("no production alias")
        mock_eval.run.return_value = {"mAP50": 0.85, "dataset_generation": 42}

        result = self._run(mock_aip, mock_eval)

        assert result == 42

    def test_resolves_production_alias_from_the_registered_models_base_resource_name(self):
        mock_aip, mock_eval = self._make_mocks()
        mock_aip.Model.side_effect = NotFound("no production alias")
        mock_eval.run.return_value = {"mAP50": 0.85, "dataset_generation": 42}

        self._run(
            mock_aip, mock_eval,
            model_resource_name="projects/p/locations/us-central1/models/123/versions/7",
        )

        model_name_arg = mock_aip.Model.call_args.kwargs["model_name"]
        assert model_name_arg == "projects/p/locations/us-central1/models/123@production"

    def test_skips_production_check_when_no_production_alias(self):
        mock_aip, mock_eval = self._make_mocks()
        mock_aip.Model.side_effect = NotFound("no production alias")
        mock_eval.run.return_value = {"mAP50": 0.85, "dataset_generation": 42}

        self._run(mock_aip, mock_eval)

        assert mock_eval.run.call_count == 1

    def test_propagates_unexpected_error_resolving_production_alias(self):
        mock_aip, mock_eval = self._make_mocks()
        mock_aip.Model.side_effect = RuntimeError("transient API error")
        mock_eval.run.return_value = {"mAP50": 0.85, "dataset_generation": 42}

        with pytest.raises(RuntimeError, match="transient API error"):
            self._run(mock_aip, mock_eval)

    def test_reevaluates_production_when_not_already_evaluated_at_current_generation(self):
        import pandas as pd
        mock_aip, mock_eval = self._make_mocks()
        mock_aip.Model.side_effect = [MagicMock(version_description="run-prod")]
        mock_aip.get_experiment_df.return_value = pd.DataFrame([
            {"run_name": "run-new-gen42", "metric.mAP50": 0.85},
        ])
        mock_eval.run.return_value = {"mAP50": 0.85, "dataset_generation": 42}
        mock_eval._vertex_run_name.side_effect = lambda run_id, gen: f"{run_id}-gen{gen}"

        self._run(mock_aip, mock_eval, training_bucket="tb", model_bucket="mb")

        assert mock_eval.run.call_count == 2
        second_call_kwargs = mock_eval.run.call_args_list[1].kwargs
        assert second_call_kwargs == {
            "run_id": "run-prod",
            "training_bucket": "tb",
            "model_bucket": "mb",
            "project_id": "test-project",
            "region": "us-central1",
            "vertex_experiment": "fish-id-eval",
        }

    def test_skips_reeval_when_already_evaluated_at_current_generation(self):
        import pandas as pd
        mock_aip, mock_eval = self._make_mocks()
        mock_aip.Model.side_effect = [MagicMock(version_description="run-prod")]
        mock_aip.get_experiment_df.return_value = pd.DataFrame([
            {"run_name": "run-new-gen42", "metric.mAP50": 0.85},
            {"run_name": "run-prod-gen42", "metric.mAP50": 0.90},
        ])
        mock_eval.run.return_value = {"mAP50": 0.85, "dataset_generation": 42}
        mock_eval._vertex_run_name.side_effect = lambda run_id, gen: f"{run_id}-gen{gen}"

        self._run(mock_aip, mock_eval)

        assert mock_eval.run.call_count == 1


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
                current_dataset_generation=kwargs.get("current_dataset_generation", 42),
            )

    def test_component_has_expected_inputs(self):
        inputs = promote_model.component_spec.inputs
        expected = {
            "project", "region", "model_resource_name", "vertex_experiment",
            "current_dataset_generation",
        }
        assert set(inputs.keys()) == expected

    def test_auto_promotes_when_no_production_model_exists(self):
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-abc")
        mock_aip.Model.side_effect = [latest_model, NotFound("no production alias")]

        result = self._run(mock_aip, mock_aip_v1, mock_aip_v1_types)

        assert result is True

    def test_fails_gate_when_no_model_marked_latest(self):
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        mock_aip.Model.side_effect = NotFound("no model with alias 'latest'")

        result = self._run(mock_aip, mock_aip_v1, mock_aip_v1_types)

        assert result is False

    def test_does_not_tag_production_when_no_model_marked_latest(self):
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        mock_aip.Model.side_effect = NotFound("no model with alias 'latest'")

        self._run(mock_aip, mock_aip_v1, mock_aip_v1_types)

        mock_service_client = mock_aip_v1.ModelServiceClient.return_value
        mock_service_client.merge_version_aliases.assert_not_called()

    def test_propagates_unexpected_error_resolving_latest(self):
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        mock_aip.Model.side_effect = RuntimeError("transient API error")

        with pytest.raises(RuntimeError, match="transient API error"):
            self._run(mock_aip, mock_aip_v1, mock_aip_v1_types)

    def test_promotes_when_current_better_than_production(self):
        import pandas as pd
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-new")
        prod_model = MagicMock(version_description="run-prod")
        mock_aip.Model.side_effect = [latest_model, prod_model]
        mock_aip.get_experiment_df.return_value = pd.DataFrame([
            {"run_name": "run-new-gen42", "metric.mAP50": 0.85},
            {"run_name": "run-prod-gen42", "metric.mAP50": 0.80},
        ])

        result = self._run(mock_aip, mock_aip_v1, mock_aip_v1_types, current_dataset_generation=42)

        assert result is True

    def test_skips_when_current_worse_than_production(self):
        import pandas as pd
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-new")
        prod_model = MagicMock(version_description="run-prod")
        mock_aip.Model.side_effect = [latest_model, prod_model]
        mock_aip.get_experiment_df.return_value = pd.DataFrame([
            {"run_name": "run-new-gen42", "metric.mAP50": 0.60},
            {"run_name": "run-prod-gen42", "metric.mAP50": 0.80},
        ])

        result = self._run(mock_aip, mock_aip_v1, mock_aip_v1_types, current_dataset_generation=42)

        assert result is False

    def test_skips_gate_when_experiment_lookup_raises_api_error(self):
        from google.api_core.exceptions import ServiceUnavailable
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-new")
        prod_model = MagicMock(version_description="run-prod")
        mock_aip.Model.side_effect = [latest_model, prod_model]
        mock_aip.get_experiment_df.side_effect = ServiceUnavailable("backend unavailable")

        result = self._run(mock_aip, mock_aip_v1, mock_aip_v1_types, current_dataset_generation=42)

        assert result is False

    def test_propagates_unexpected_error_from_experiment_lookup(self):
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-new")
        prod_model = MagicMock(version_description="run-prod")
        mock_aip.Model.side_effect = [latest_model, prod_model]
        mock_aip.get_experiment_df.side_effect = NameError("bug, not an expected failure")

        with pytest.raises(NameError):
            self._run(mock_aip, mock_aip_v1, mock_aip_v1_types, current_dataset_generation=42)

    def test_skips_when_production_has_no_result_at_current_generation(self):
        import pandas as pd
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-new")
        prod_model = MagicMock(version_description="run-prod")
        mock_aip.Model.side_effect = [latest_model, prod_model]
        # Production was only ever evaluated at generation 41 — eval_latest_and_production_models
        # should have re-evaluated it at 42 before promote_model ever runs; if that didn't happen,
        # fail safe rather than compare against a stale number.
        mock_aip.get_experiment_df.return_value = pd.DataFrame([
            {"run_name": "run-new-gen42", "metric.mAP50": 0.85},
            {"run_name": "run-prod-gen41", "metric.mAP50": 0.80},
        ])

        result = self._run(mock_aip, mock_aip_v1, mock_aip_v1_types, current_dataset_generation=42)

        assert result is False

    def test_calls_merge_version_aliases_on_promote(self):
        mock_aip, mock_aip_v1, mock_aip_v1_types = self._make_mocks()
        latest_model = MagicMock(version_description="run-abc")
        mock_aip.Model.side_effect = [latest_model, NotFound("no production alias")]

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
