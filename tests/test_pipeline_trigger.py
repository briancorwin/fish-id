"""Unit tests for pipeline/trigger/main.py.

Tests the GCS manifest pattern, run_id format, and trigger routing without
making real GCS or Vertex AI calls.
"""
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Set required env vars before the trigger module is imported
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_REGION", "us-central1")
os.environ.setdefault("TRAINING_BUCKET", "test-training")
os.environ.setdefault("MODEL_BUCKET", "test-models")
os.environ.setdefault("PIPELINE_SA", "sa@test.iam.gserviceaccount.com")
os.environ.setdefault("PIPELINE_TEMPLATE_URI", "gs://test-models/pipeline/test.json")
os.environ.setdefault("VERTEX_EXPERIMENT", "fish-id-eval")

# Stub packages not in the test environment before importing trigger
_mock_ff = MagicMock()
_mock_ff.cloud_event = lambda f: f  # pass-through — lets us call trigger_pipeline directly
sys.modules["functions_framework"] = _mock_ff
sys.modules.setdefault("google.cloud.aiplatform", MagicMock())

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "pipeline_trigger",
    str(Path(__file__).parent.parent / "pipeline" / "trigger" / "main.py"),
)
trigger_module = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(trigger_module)


# ---------------------------------------------------------------------------
# GCS manifest pattern
# ---------------------------------------------------------------------------

class TestManifestPattern:
    def test_matches_versioned_manifest(self):
        assert trigger_module._MANIFEST_PATTERN.match("versions/v1/manifest.json")

    def test_matches_any_version_string(self):
        assert trigger_module._MANIFEST_PATTERN.match("versions/2024-01-15/manifest.json")

    def test_does_not_match_non_manifest_file(self):
        assert not trigger_module._MANIFEST_PATTERN.match("versions/v1/images.tar.gz")

    def test_does_not_match_unversioned_path(self):
        assert not trigger_module._MANIFEST_PATTERN.match("manifest.json")

    def test_does_not_match_nested_manifest(self):
        # versions/v1/sub/manifest.json has an extra path component
        assert not trigger_module._MANIFEST_PATTERN.match("versions/v1/sub/manifest.json")

    def test_does_not_match_images_path(self):
        assert not trigger_module._MANIFEST_PATTERN.match("images/fish/foo.jpg")


# ---------------------------------------------------------------------------
# Run ID format
# ---------------------------------------------------------------------------

class TestRunId:
    def test_run_id_starts_with_run_prefix(self):
        run_id = trigger_module._make_run_id()
        assert run_id.startswith("run-")

    def test_run_id_matches_expected_format(self):
        run_id = trigger_module._make_run_id()
        # Expected: run-YYYY-MM-DD-HH-MM-SS
        assert re.match(r"^run-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", run_id), run_id

    def test_successive_run_ids_are_unique(self):
        import time
        id1 = trigger_module._make_run_id()
        time.sleep(1.1)
        id2 = trigger_module._make_run_id()
        assert id1 != id2


# ---------------------------------------------------------------------------
# trigger_pipeline routing
# ---------------------------------------------------------------------------

class TestTriggerPipeline:
    def _make_event(self, object_name):
        event = MagicMock()
        event.data = {"name": object_name}
        return event

    def test_skips_non_manifest_objects(self):
        event = self._make_event("images/foo.jpg")
        with patch.object(trigger_module, "_read_training_image") as mock_read, \
             patch.object(trigger_module, "aiplatform") as mock_aip:
            trigger_module.trigger_pipeline(event)
            mock_read.assert_not_called()
            mock_aip.PipelineJob.assert_not_called()

    def test_submits_pipeline_for_versioned_manifest(self):
        event = self._make_event("versions/v3/manifest.json")
        with patch.object(trigger_module, "_read_training_image",
                          return_value="us-central1-docker.pkg.dev/p/fish-id/fish-id-train:abc"), \
             patch.object(trigger_module, "aiplatform") as mock_aip:
            trigger_module.trigger_pipeline(event)
            mock_aip.PipelineJob.assert_called_once()
            mock_aip.PipelineJob.return_value.submit.assert_called_once()

    def test_pipeline_parameters_include_dataset_version(self):
        event = self._make_event("versions/v3/manifest.json")
        with patch.object(trigger_module, "_read_training_image", return_value="img:tag"), \
             patch.object(trigger_module, "aiplatform") as mock_aip:
            trigger_module.trigger_pipeline(event)
            kwargs = mock_aip.PipelineJob.call_args.kwargs
            assert kwargs["parameter_values"]["dataset_version"] == "v3"

    def test_pipeline_parameters_include_training_image(self):
        event = self._make_event("versions/v3/manifest.json")
        with patch.object(trigger_module, "_read_training_image",
                          return_value="us-central1-docker.pkg.dev/p/fish-id/fish-id-train:abc"), \
             patch.object(trigger_module, "aiplatform") as mock_aip:
            trigger_module.trigger_pipeline(event)
            kwargs = mock_aip.PipelineJob.call_args.kwargs
            assert kwargs["parameter_values"]["training_image"] == \
                "us-central1-docker.pkg.dev/p/fish-id/fish-id-train:abc"

    def test_pipeline_submitted_with_service_account(self):
        event = self._make_event("versions/v1/manifest.json")
        with patch.object(trigger_module, "_read_training_image", return_value="img:tag"), \
             patch.object(trigger_module, "aiplatform") as mock_aip:
            trigger_module.trigger_pipeline(event)
            submit_call = mock_aip.PipelineJob.return_value.submit
            submit_call.assert_called_once_with(
                service_account=os.environ["PIPELINE_SA"]
            )
