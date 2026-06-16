"""Vertex AI Pipeline: CPU path uses a container component; GPU path submits a Spot Custom Job via the aiplatform SDK."""
# pylint: disable=import-outside-toplevel
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
def run_gpu_training_job(
    project: str,
    region: str,
    training_image: str,
    run_id: str,
    training_bucket: str,
    model_bucket: str,
) -> None:
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1.types.custom_job import Scheduling

    aiplatform.init(project=project, location=region, staging_bucket=f"gs://{model_bucket}")

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "n1-standard-4",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": training_image,
            "command": ["python", "/app/train.py"],
            "args": [
                f"--run-id={run_id}",
                f"--training-bucket={training_bucket}",
                f"--model-bucket={model_bucket}",
            ],
        },
    }]

    custom_job = aiplatform.CustomJob(
        display_name=f"fish-id-{run_id}",
        worker_pool_specs=worker_pool_specs,
        project=project,
        location=region,
    )
    custom_job.run(
        scheduling_strategy=Scheduling.Strategy.SPOT,
        restart_job_on_worker_restart=True,
        sync=True,
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

    upload_kwargs: dict = {
        "display_name": "fish-id",
        "artifact_uri": artifact_uri,
        # Required by the API but never used: we serve from Cloud Run, not Vertex AI.
        # The Cloud Run image URI would be more accurate but isn't knowable here —
        # it's built by the deploy workflow that runs *after* this step completes.
        "serving_container_image_uri": "us-docker.pkg.dev/vertex-ai/prediction/onnx-cpu.1-14:latest",
        "is_default_version": True,
        "version_aliases": ["latest", "production"],
        "version_description": run_id,
    }
    if existing:
        upload_kwargs["parent_model"] = existing[0].resource_name

    model = aiplatform.Model.upload(**upload_kwargs)
    return model.resource_name


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-secret-manager>=2.0.0", "requests>=2.31.0"],
)
def trigger_deploy(
    project: str,
    github_repo: str,
) -> None:
    import requests
    from google.cloud import secretmanager  # type: ignore[attr-defined]  # pylint: disable=no-name-in-module

    client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{project}/secrets/fish-id-github-deploy-token/versions/latest"
    token = client.access_secret_version(name=secret_name).payload.data.decode()

    resp = requests.post(
        f"https://api.github.com/repos/{github_repo}/actions/workflows/deploy-api.yml/dispatches",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": "main"},
        timeout=30,
    )
    resp.raise_for_status()


@dsl.pipeline(name="fish-id-training-pipeline")
def fish_id_training_pipeline(
    training_bucket: str,
    model_bucket: str,
    training_image: str,
    run_id: str,
    project: str,
    region: str,
    github_repo: str,
    cpu_only: bool = False,
) -> None:
    with dsl.If(cpu_only == True):  # pylint: disable=singleton-comparison
        cpu_train = (
            run_training_job(  # pylint: disable=no-member
                run_id=run_id,
                training_bucket=training_bucket,
                model_bucket=model_bucket,
            )
            .set_container_image(training_image)
            .set_cpu_request("16").set_cpu_limit("16")
            .set_memory_request("64G").set_memory_limit("64G")
            .set_retry(num_retries=3)
        )
        reg_cpu = register_model(
            project=project,
            region=region,
            model_bucket=model_bucket,
            run_id=run_id,
        ).after(cpu_train)
        trigger_deploy(project=project, github_repo=github_repo).after(reg_cpu)

    with dsl.Else():
        gpu_train = run_gpu_training_job(
            project=project,
            region=region,
            training_image=training_image,
            run_id=run_id,
            training_bucket=training_bucket,
            model_bucket=model_bucket,
        ).set_retry(num_retries=2)
        reg_gpu = register_model(
            project=project,
            region=region,
            model_bucket=model_bucket,
            run_id=run_id,
        ).after(gpu_train)
        trigger_deploy(project=project, github_repo=github_repo).after(reg_gpu)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).parent / "fish-id-training-pipeline.json"
    compiler.Compiler().compile(pipeline_func=fish_id_training_pipeline, package_path=str(output))
    _logger.info("Pipeline compiled to %s", output)


if __name__ == "__main__":
    main()
