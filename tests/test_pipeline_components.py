"""Unit tests for pipeline/pipeline.py components and pipeline compilation.

All GCP client calls are mocked. No real infrastructure or credentials required.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("google.cloud.aiplatform", MagicMock())
sys.modules.setdefault("google.cloud.secretmanager", MagicMock())

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline import fish_id_training_pipeline, register_model, run_gpu_training_job, trigger_deploy  # noqa: E402


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


class TestRunGpuTrainingJob:
    def _run(self, mock_aip: MagicMock, mock_scheduling_module: MagicMock, **kwargs) -> None:
        with patch.dict(sys.modules, {
            "google.cloud.aiplatform": mock_aip,
            "google.cloud.aiplatform_v1": MagicMock(),
            "google.cloud.aiplatform_v1.types": MagicMock(),
            "google.cloud.aiplatform_v1.types.custom_job": mock_scheduling_module,
        }):
            run_gpu_training_job.python_func(
                project=kwargs.get("project", "test-project"),
                region=kwargs.get("region", "us-central1"),
                training_image=kwargs.get("training_image", "gcr.io/test/image:v1"),
                run_id=kwargs.get("run_id", "run-2026-06-14-120000"),
                training_bucket=kwargs.get("training_bucket", "test-training-bucket"),
                model_bucket=kwargs.get("model_bucket", "test-model-bucket"),
            )

    def test_creates_custom_job_with_t4_gpu(self):
        mock_aip = MagicMock()
        mock_scheduling_module = MagicMock()
        self._run(mock_aip, mock_scheduling_module)

        kwargs = mock_aip.CustomJob.call_args.kwargs
        worker_specs = kwargs["worker_pool_specs"]
        assert worker_specs[0]["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_T4"
        assert worker_specs[0]["machine_spec"]["accelerator_count"] == 1

    def test_runs_with_spot_scheduling(self):
        mock_aip = MagicMock()
        mock_scheduling_module = MagicMock()
        self._run(mock_aip, mock_scheduling_module)

        run_kwargs = mock_aip.CustomJob.return_value.run.call_args.kwargs
        assert run_kwargs["scheduling_strategy"] == mock_scheduling_module.Scheduling.Strategy.SPOT
        assert run_kwargs["restart_job_on_worker_restart"] is True
        assert run_kwargs["sync"] is True

    def test_passes_run_id_in_display_name(self):
        mock_aip = MagicMock()
        mock_scheduling_module = MagicMock()
        self._run(mock_aip, mock_scheduling_module, run_id="run-abc-123")

        display_name = mock_aip.CustomJob.call_args.kwargs["display_name"]
        assert "run-abc-123" in display_name

    def test_passes_run_id_in_container_args(self):
        mock_aip = MagicMock()
        mock_scheduling_module = MagicMock()
        self._run(mock_aip, mock_scheduling_module, run_id="run-xyz")

        worker_specs = mock_aip.CustomJob.call_args.kwargs["worker_pool_specs"]
        args = worker_specs[0]["container_spec"]["args"]
        assert any("run-xyz" in str(a) for a in args)


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
        assert url == "https://api.github.com/repos/owner/fish-id/actions/workflows/deploy.yml/dispatches"
        assert mock_req.post.call_args.kwargs["json"] == {"ref": "main"}
        assert mock_req.post.call_args.kwargs["headers"]["Authorization"] == "Bearer ghp_token"

    def test_raises_on_github_api_error(self):
        mock_sm = MagicMock()
        mock_sm.SecretManagerServiceClient.return_value.access_secret_version.return_value.payload.data = b"ghp_token"
        mock_req = MagicMock()
        mock_req.post.return_value.raise_for_status.side_effect = RuntimeError("403 Forbidden")

        with pytest.raises(RuntimeError, match="403"):
            self._run(mock_sm, mock_req)
