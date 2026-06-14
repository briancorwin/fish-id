"""Unit tests for pipeline/pipeline.py components and pipeline compilation.

All GCP client calls are mocked. No real infrastructure or credentials required.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("google.cloud.aiplatform", MagicMock())

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline import fish_id_training_pipeline  # noqa: E402


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
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_training_pipeline, str(output))
        spec = json.loads(output.read_text())
        assert spec["pipelineInfo"]["name"] == "fish-id-training-pipeline"

    def test_gpu_branch_uses_custom_training_job(self, tmp_path):
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_training_pipeline, str(output))
        spec = json.loads(output.read_text())
        component_names = list(spec["deploymentSpec"]["executors"].keys())
        assert any("custom-training-job" in name for name in component_names)

    def test_gpu_branch_uses_spot_strategy(self, tmp_path):
        from kfp import compiler
        output = tmp_path / "pipeline.json"
        compiler.Compiler().compile(fish_id_training_pipeline, str(output))
        spec_text = output.read_text()
        assert "SPOT" in spec_text
