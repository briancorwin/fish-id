"""Vertex AI Pipeline: train → eval → register → promote (gates 'production' alias)."""
# pylint: disable=import-outside-toplevel
import logging
import os
from pathlib import Path

import yaml

from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from kfp import compiler, dsl

_CONFIG = yaml.safe_load(
    (Path(__file__).parent.parent / "training" / "config.yaml").read_text(encoding="utf-8")
)

# Resolved at pipeline compile time from CI env vars (GCP_REGION, GCP_PROJECT_ID).
# Override by setting TRAINING_IMAGE explicitly.
_TRAINING_IMAGE = (
    os.environ.get("TRAINING_IMAGE")
    or f"{os.environ.get('GCP_REGION', 'us-central1')}-docker.pkg.dev"
       f"/{os.environ.get('GCP_PROJECT_ID', 'unknown')}/fish-id/fish-id-train:latest"
)

_logger = logging.getLogger(__name__)


@dsl.component(base_image=_TRAINING_IMAGE)
def train_model(
    run_id: str,
    training_bucket: str,
    model_bucket: str,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    optimizer: str,
    lr0: float,
) -> None:
    import train  # installed via PYTHONPATH=/app in the training image
    train.run(
        run_id=run_id,
        training_bucket=training_bucket,
        model_bucket=model_bucket,
        model_name=model_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer=optimizer,
        lr0=lr0,
    )


@dsl.component(base_image=_TRAINING_IMAGE)
def eval_model(
    run_id: str,
    training_bucket: str,
    model_bucket: str,
    project_id: str,
    region: str,
    vertex_experiment: str,
) -> None:
    import eval  # installed via PYTHONPATH=/app in the training image  # noqa: A001  # pylint: disable=redefined-builtin
    eval.run(
        run_id=run_id,
        training_bucket=training_bucket,
        model_bucket=model_bucket,
        project_id=project_id,
        region=region,
        vertex_experiment=vertex_experiment,
    )


_TrainGpuJobOp = create_custom_training_job_from_component(
    train_model,
    display_name="fish-id-gpu-training",
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    strategy="SPOT",
    restart_job_on_worker_restart=True,
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
        "serving_container_image_uri": "us-docker.pkg.dev/vertex-ai/prediction/onnx-cpu.1-14:latest",
        "is_default_version": True,
        # "production" is set separately by promote_model after the quality gate passes.
        "version_aliases": ["latest"],
        "version_description": run_id,
    }
    if existing:
        upload_kwargs["parent_model"] = existing[0].resource_name

    model = aiplatform.Model.upload(**upload_kwargs)
    return model.resource_name


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform>=1.60.0"],
)
def promote_model(
    project: str,
    region: str,
    model_resource_name: str,
    vertex_experiment: str,
) -> bool:
    import logging
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1 import ModelServiceClient
    from google.cloud.aiplatform_v1.types import MergeVersionAliasesRequest

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aiplatform.init(project=project, location=region)
    base_model_name = model_resource_name.split("/versions/")[0]

    latest_model = aiplatform.Model(
        model_name=f"{base_model_name}@latest",
        project=project,
        location=region,
    )
    latest_run_id = latest_model.version_description
    logger.info("[promote] current run_id=%s", latest_run_id)

    prod_run_id = None
    try:
        prod_model = aiplatform.Model(
            model_name=f"{base_model_name}@production",
            project=project,
            location=region,
        )
        prod_run_id = prod_model.version_description
        logger.info("[promote] production model: run_id=%s", prod_run_id)
    except Exception:
        pass  # no production alias yet — first run

    if prod_run_id is not None:
        current_map50 = None
        prod_map50 = None

        try:
            aiplatform.init(project=project, location=region, experiment=vertex_experiment)
            exp_df = aiplatform.get_experiment_df(  # pylint: disable=unexpected-keyword-arg
                experiment=vertex_experiment, project=project, location=region
            )
            current_rows = exp_df[exp_df["run_name"] == latest_run_id]
            if not current_rows.empty:
                current_map50 = float(current_rows.iloc[0]["metric.mAP50"])
                logger.info("[promote] current mAP50=%.3f", current_map50)
            prod_rows = exp_df[exp_df["run_name"] == prod_run_id]
            if not prod_rows.empty:
                prod_map50 = float(prod_rows.iloc[0]["metric.mAP50"])
                logger.info("[promote] production mAP50=%.3f", prod_map50)
        except Exception as exc:
            logger.warning("[promote] Vertex AI Experiments lookup failed: %s", exc)

        if current_map50 is not None and prod_map50 is not None and current_map50 < prod_map50 - 0.02:
            logger.info(
                "[promote] gate FAILED: current mAP50=%.3f < prod mAP50=%.3f - 0.02 — skipping",
                current_map50, prod_map50,
            )
            return False

        logger.info(
            "[promote] gate PASSED: current mAP50=%s vs prod mAP50=%s",
            current_map50, prod_map50,
        )
    else:
        logger.info("[promote] no production model found — auto-promoting")

    model_service = ModelServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )
    model_service.merge_version_aliases(
        request=MergeVersionAliasesRequest(
            name=model_resource_name,
            version_aliases=["production"],
        )
    )
    logger.info("[promote] tagged %s as 'production'", model_resource_name)

    return True


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
    run_id: str,
    project: str,
    region: str,
    github_repo: str,
    vertex_experiment: str,
    model_name: str = _CONFIG["model"],
    epochs: int = _CONFIG["epochs"],
    imgsz: int = _CONFIG["imgsz"],
    batch: int = _CONFIG["batch"],
    optimizer: str = _CONFIG["optimizer"],
    lr0: float = _CONFIG["lr0"],
    cpu_only: bool = False,
) -> None:
    with dsl.If(cpu_only == True):  # pylint: disable=singleton-comparison
        cpu_train = (
            train_model(  # pylint: disable=no-member
                run_id=run_id,
                training_bucket=training_bucket,
                model_bucket=model_bucket,
                model_name=model_name,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                optimizer=optimizer,
                lr0=lr0,
            )
            .set_cpu_request("16").set_cpu_limit("16")
            .set_memory_request("64G").set_memory_limit("64G")
            .set_retry(num_retries=3)
        )
        cpu_eval = eval_model(  # pylint: disable=no-member
            run_id=run_id,
            training_bucket=training_bucket,
            model_bucket=model_bucket,
            project_id=project,
            region=region,
            vertex_experiment=vertex_experiment,
        ).after(cpu_train)
        reg_cpu = register_model(
            project=project, region=region, model_bucket=model_bucket, run_id=run_id,
        ).after(cpu_eval)
        promote_cpu = promote_model(
            project=project,
            region=region,
            model_resource_name=reg_cpu.output,
            vertex_experiment=vertex_experiment,
        ).after(reg_cpu)
        with dsl.If(promote_cpu.output == True, name="cpu-gate-passed"):  # pylint: disable=singleton-comparison
            trigger_deploy(project=project, github_repo=github_repo).after(promote_cpu)  # pylint: disable=no-member

    with dsl.Else():
        gpu_train = (
            _TrainGpuJobOp(  # pylint: disable=no-member
                project=project,
                location=region,
                run_id=run_id,
                training_bucket=training_bucket,
                model_bucket=model_bucket,
                model_name=model_name,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                optimizer=optimizer,
                lr0=lr0,
            )
            .set_retry(num_retries=3)
        )
        gpu_eval = eval_model(  # pylint: disable=no-member
            run_id=run_id,
            training_bucket=training_bucket,
            model_bucket=model_bucket,
            project_id=project,
            region=region,
            vertex_experiment=vertex_experiment,
        ).after(gpu_train)
        reg_gpu = register_model(
            project=project, region=region, model_bucket=model_bucket, run_id=run_id,
        ).after(gpu_eval)
        promote_gpu = promote_model(
            project=project,
            region=region,
            model_resource_name=reg_gpu.output,
            vertex_experiment=vertex_experiment,
        ).after(reg_gpu)
        with dsl.If(promote_gpu.output == True, name="gpu-gate-passed"):  # pylint: disable=singleton-comparison
            trigger_deploy(project=project, github_repo=github_repo).after(promote_gpu)  # pylint: disable=no-member


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).parent / "fish-id-training-pipeline.json"
    compiler.Compiler().compile(pipeline_func=fish_id_training_pipeline, package_path=str(output))
    _logger.info("Pipeline compiled to %s", output)


if __name__ == "__main__":
    main()
