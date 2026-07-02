"""Vertex AI Pipeline: train → register → eval → promote (gates 'production' alias)."""
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
    patience: int,
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
        patience=patience,
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


@dsl.component(base_image=_TRAINING_IMAGE)
def eval_latest_and_production_models(
    run_id: str,
    training_bucket: str,
    model_bucket: str,
    project_id: str,
    region: str,
    vertex_experiment: str,
    model_resource_name: str,
) -> int:
    import eval  # installed via PYTHONPATH=/app in the training image  # noqa: A001  # pylint: disable=redefined-builtin
    import logging
    from google.api_core.exceptions import NotFound
    from google.cloud import aiplatform

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    metrics = eval.run(
        run_id=run_id,
        training_bucket=training_bucket,
        model_bucket=model_bucket,
        project_id=project_id,
        region=region,
        vertex_experiment=vertex_experiment,
    )
    dataset_generation = metrics["dataset_generation"]

    aiplatform.init(project=project_id, location=region)
    base_model_name = model_resource_name.split("/versions/")[0]

    prod_run_id = None
    try:
        prod_model = aiplatform.Model(
            model_name=f"{base_model_name}@production",
            project=project_id,
            location=region,
        )
        prod_run_id = prod_model.version_description
    except NotFound as exc:
        logger.info("[eval] no production alias found: %s", exc)

    if prod_run_id is None:
        return dataset_generation

    aiplatform.init(project=project_id, location=region, experiment=vertex_experiment)
    exp_df = aiplatform.get_experiment_df(experiment=vertex_experiment)
    expected_run_name = eval._vertex_run_name(prod_run_id, dataset_generation)  # pylint: disable=protected-access
    if not exp_df[exp_df["run_name"] == expected_run_name].empty:
        logger.info(
            "[eval] production run=%s already evaluated at dataset_generation=%d — skipping re-eval",
            prod_run_id, dataset_generation,
        )
        return dataset_generation

    logger.info(
        "[eval] production run=%s has no eval at dataset_generation=%d — re-evaluating on the current eval set",
        prod_run_id, dataset_generation,
    )
    eval.run(
        run_id=prod_run_id,
        training_bucket=training_bucket,
        model_bucket=model_bucket,
        project_id=project_id,
        region=region,
        vertex_experiment=vertex_experiment,
    )

    return dataset_generation


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
    packages_to_install=["google-cloud-aiplatform[metadata]>=1.60.0"],
)
def promote_model(
    project: str,
    region: str,
    model_resource_name: str,
    vertex_experiment: str,
    current_dataset_generation: int,
) -> bool:
    import logging
    from google.api_core.exceptions import GoogleAPIError, NotFound
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1 import ModelServiceClient
    from google.cloud.aiplatform_v1.types import MergeVersionAliasesRequest

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aiplatform.init(project=project, location=region)
    base_model_name = model_resource_name.split("/versions/")[0]

    latest_run_id = None
    try:
        latest_model = aiplatform.Model(
            model_name=f"{base_model_name}@latest",
            project=project,
            location=region,
        )
        latest_run_id = latest_model.version_description
        logger.info("[promote] latest run_id=%s", latest_run_id)
    except NotFound as exc:
        logger.error(
            "[promote] gate FAILED: could not resolve '@latest' alias for %s: %s",
            base_model_name, exc,
        )
        return False

    prod_run_id = None
    try:
        prod_model = aiplatform.Model(
            model_name=f"{base_model_name}@production",
            project=project,
            location=region,
        )
        prod_run_id = prod_model.version_description
        logger.info("[promote] production model: run_id=%s", prod_run_id)
    except NotFound:
        pass  # no production alias yet — first run

    if prod_run_id is not None:
        latest_map50 = None
        prod_map50 = None

        # eval_latest_and_production_models has already re-evaluated the production model
        # at current_dataset_generation if it wasn't evaluated there already, so both runs'
        # results are guaranteed to be looked up on the same dataset.
        try:
            aiplatform.init(project=project, location=region, experiment=vertex_experiment)
            exp_df = aiplatform.get_experiment_df(experiment=vertex_experiment)
            latest_run_name = f"{latest_run_id}-gen{current_dataset_generation}"
            prod_run_name = f"{prod_run_id}-gen{current_dataset_generation}"
            latest_rows = exp_df[exp_df["run_name"] == latest_run_name]
            if not latest_rows.empty:
                latest_map50 = float(latest_rows.iloc[0]["metric.mAP50"])
                logger.info("[promote] latest mAP50=%.3f", latest_map50)
            prod_rows = exp_df[exp_df["run_name"] == prod_run_name]
            if not prod_rows.empty:
                prod_map50 = float(prod_rows.iloc[0]["metric.mAP50"])
                logger.info("[promote] production mAP50=%.3f", prod_map50)
        except (GoogleAPIError, KeyError, ValueError, TypeError) as exc:
            logger.warning("[promote] Vertex AI Experiments lookup failed: %s", exc)

        if latest_map50 is None or prod_map50 is None:
            logger.warning(
                "[promote] gate FAILED: could not retrieve mAP50 for latest or "
                "production run at dataset_generation=%d — skipping",
                current_dataset_generation,
            )
            return False

        if latest_map50 < prod_map50 - 0.02:
            logger.info(
                "[promote] gate FAILED: latest mAP50=%.3f < prod mAP50=%.3f - 0.02 — skipping",
                latest_map50, prod_map50,
            )
            return False

        logger.info(
            "[promote] gate PASSED: latest mAP50=%.3f vs prod mAP50=%.3f",
            latest_map50, prod_map50,
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
    patience: int = _CONFIG["patience"],
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
                patience=patience,
            )
            .set_cpu_request("16").set_cpu_limit("16")
            .set_memory_request("64G").set_memory_limit("64G")
            .set_retry(num_retries=3)
        )
        reg_cpu = register_model(
            project=project, region=region, model_bucket=model_bucket, run_id=run_id,
        ).after(cpu_train)
        cpu_eval = eval_latest_and_production_models(  # pylint: disable=no-member
            run_id=run_id,
            training_bucket=training_bucket,
            model_bucket=model_bucket,
            project_id=project,
            region=region,
            vertex_experiment=vertex_experiment,
            model_resource_name=reg_cpu.output,
        ).set_retry(num_retries=3).after(reg_cpu)
        promote_cpu = promote_model(
            project=project,
            region=region,
            model_resource_name=reg_cpu.output,
            vertex_experiment=vertex_experiment,
            current_dataset_generation=cpu_eval.output,
        ).after(cpu_eval)
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
                patience=patience,
            )
            .set_retry(num_retries=3)
        )
        reg_gpu = register_model(
            project=project, region=region, model_bucket=model_bucket, run_id=run_id,
        ).after(gpu_train)
        gpu_eval = eval_latest_and_production_models(  # pylint: disable=no-member
            run_id=run_id,
            training_bucket=training_bucket,
            model_bucket=model_bucket,
            project_id=project,
            region=region,
            vertex_experiment=vertex_experiment,
            model_resource_name=reg_gpu.output,
        ).set_retry(num_retries=3).after(reg_gpu)
        promote_gpu = promote_model(
            project=project,
            region=region,
            model_resource_name=reg_gpu.output,
            vertex_experiment=vertex_experiment,
            current_dataset_generation=gpu_eval.output,
        ).after(gpu_eval)
        with dsl.If(promote_gpu.output == True, name="gpu-gate-passed"):  # pylint: disable=singleton-comparison
            trigger_deploy(project=project, github_repo=github_repo).after(promote_gpu)  # pylint: disable=no-member


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).parent / "fish-id-training-pipeline.json"
    compiler.Compiler().compile(pipeline_func=fish_id_training_pipeline, package_path=str(output))
    _logger.info("Pipeline compiled to %s", output)


if __name__ == "__main__":
    main()
