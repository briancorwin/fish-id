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


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform>=1.60.0"],
)
def register_model(
    project: str,
    region: str,
    model_bucket: str,
    run_id: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    artifact_uri = f"gs://{model_bucket}/runs/{run_id}/"

    existing = aiplatform.Model.list(
        filter='display_name="fish-id"',
        order_by="create_time desc",
        project=project,
        location=region,
    )

    upload_kwargs: dict = dict(
        display_name="fish-id",
        artifact_uri=artifact_uri,
        is_default_version=True,
        version_aliases=["latest", "production"],
        version_description=run_id,
    )
    if existing:
        upload_kwargs["parent_model"] = existing[0].resource_name

    model = aiplatform.Model.upload(**upload_kwargs)
    return model.resource_name


@dsl.pipeline(name="fish-id-training-pipeline")
def fish_id_training_pipeline(
    training_bucket: str,
    model_bucket: str,
    training_image: str,
    run_id: str,
    project: str,
    region: str,
    cpu_only: bool = False,
) -> None:
    with dsl.If(cpu_only == True):
        cpu_train = (
            run_training_job(
                run_id=run_id,
                training_bucket=training_bucket,
                model_bucket=model_bucket,
            )
            .set_container_image(training_image)
            .set_cpu_request("16").set_cpu_limit("16")
            .set_memory_request("64G").set_memory_limit("64G")
        )
        register_model(
            project=project,
            region=region,
            model_bucket=model_bucket,
            run_id=run_id,
        ).after(cpu_train)

    with dsl.Else():
        gpu_train = (
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
        register_model(
            project=project,
            region=region,
            model_bucket=model_bucket,
            run_id=run_id,
        ).after(gpu_train)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).parent / "fish-id-training-pipeline.json"
    compiler.Compiler().compile(pipeline_func=fish_id_training_pipeline, package_path=str(output))
    _logger.info("Pipeline compiled to %s", output)


if __name__ == "__main__":
    main()
