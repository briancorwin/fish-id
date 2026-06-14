"""Vertex AI Pipeline: run training container directly as a container component."""
import logging
from pathlib import Path

from kfp import compiler, dsl

_logger = logging.getLogger(__name__)


@dsl.container_component
def run_training_job(
    run_id: str,
    training_bucket: str,
    model_bucket: str,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image="placeholder",
        command=["python", "/app/train.py"],
        args=[
            "--run-id", run_id,
            "--training-bucket", training_bucket,
            "--model-bucket", model_bucket,
        ],
    )


@dsl.pipeline(name="fish-id-training-pipeline")
def fish_id_training_pipeline(
    training_bucket: str,
    model_bucket: str,
    training_image: str,
    run_id: str,
    cpu_only: bool = False,
) -> None:
    with dsl.If(cpu_only == True):
        (
            run_training_job(
                run_id=run_id,
                training_bucket=training_bucket,
                model_bucket=model_bucket,
            )
            .set_container_image(training_image)
            .set_cpu_request("16").set_cpu_limit("16")
            .set_memory_request("64G").set_memory_limit("64G")
        )

    with dsl.Else():
        (
            run_training_job(
                run_id=run_id,
                training_bucket=training_bucket,
                model_bucket=model_bucket,
            )
            .set_container_image(training_image)
            .set_cpu_request("4").set_cpu_limit("4")
            .set_memory_request("16G").set_memory_limit("16G")
            .set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit("1")
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).parent / "fish-id-training-pipeline.json"
    compiler.Compiler().compile(pipeline_func=fish_id_training_pipeline, package_path=str(output))
    _logger.info("Pipeline compiled to %s", output)


if __name__ == "__main__":
    main()
