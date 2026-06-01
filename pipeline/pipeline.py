"""Vertex AI Pipeline: training → eval → quality gate → model promotion."""
import logging
from pathlib import Path
from typing import NamedTuple

from kfp import compiler, dsl
from kfp.dsl import component

_logger = logging.getLogger(__name__)
_BASE_IMAGE = "python:3.11-slim"


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@component(
    base_image=_BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform==1.70.0"],
)
def run_training_job(
    project: str,
    region: str,
    training_image: str,
    run_id: str,
    dataset_version: str,
    config_version: str,
    training_bucket: str,
    model_bucket: str,
    vertex_experiment: str,
    machine_type: str,
) -> None:
    import logging  # noqa: PLC0415 — required inside KFP component body
    from google.cloud import aiplatform  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aiplatform.init(project=project, location=region, experiment=vertex_experiment)
    job = aiplatform.CustomJob(
        display_name=f"fish-id-train-{run_id}",
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": training_image,
                    "env": [
                        {"name": "JOB_MODE", "value": "train"},
                        {"name": "RUN_ID", "value": run_id},
                        {"name": "DATASET_VERSION", "value": dataset_version},
                        {"name": "CONFIG_VERSION", "value": config_version},
                        {"name": "TRAINING_BUCKET", "value": training_bucket},
                        {"name": "MODEL_BUCKET", "value": model_bucket},
                        {"name": "GCP_PROJECT_ID", "value": project},
                        {"name": "GCP_REGION", "value": region},
                        {"name": "VERTEX_EXPERIMENT", "value": vertex_experiment},
                        {"name": "CONTAINER_IMAGE", "value": training_image},
                        {"name": "MACHINE_TYPE", "value": machine_type},
                    ],
                },
            }
        ],
    )
    training_sa = f"fish-id-training-sa@{project}.iam.gserviceaccount.com"
    logger.info("Submitting training job fish-id-train-%s", run_id)
    job.run(sync=True, service_account=training_sa)
    logger.info("Training job completed: %s", job.resource_name)


@component(
    base_image=_BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform==1.70.0"],
)
def run_eval_job(
    project: str,
    region: str,
    training_image: str,
    run_id: str,
    dataset_version: str,
    config_version: str,
    training_bucket: str,
    model_bucket: str,
    vertex_experiment: str,
) -> None:
    import logging  # noqa: PLC0415
    from google.cloud import aiplatform  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aiplatform.init(project=project, location=region, experiment=vertex_experiment)
    job = aiplatform.CustomJob(
        display_name=f"fish-id-eval-{run_id}",
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-highmem-4"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": training_image,
                    "env": [
                        {"name": "JOB_MODE", "value": "eval"},
                        {"name": "RUN_ID", "value": run_id},
                        {"name": "DATASET_VERSION", "value": dataset_version},
                        {"name": "CONFIG_VERSION", "value": config_version},
                        {"name": "TRAINING_BUCKET", "value": training_bucket},
                        {"name": "MODEL_BUCKET", "value": model_bucket},
                        {"name": "GCP_PROJECT_ID", "value": project},
                        {"name": "GCP_REGION", "value": region},
                        {"name": "VERTEX_EXPERIMENT", "value": vertex_experiment},
                        {"name": "CONTAINER_IMAGE", "value": training_image},
                    ],
                },
            }
        ],
    )
    training_sa = f"fish-id-training-sa@{project}.iam.gserviceaccount.com"
    logger.info("Submitting eval job fish-id-eval-%s", run_id)
    job.run(sync=True, service_account=training_sa)
    logger.info("Eval job completed: %s", job.resource_name)


@component(
    base_image=_BASE_IMAGE,
    packages_to_install=["google-cloud-storage==2.18.2"],
)
def quality_gate(
    project: str,
    model_bucket: str,
    run_id: str,
) -> NamedTuple("GateResult", [("passed", str)]):
    import json  # noqa: PLC0415
    import logging  # noqa: PLC0415
    from collections import namedtuple  # noqa: PLC0415
    from google.api_core.exceptions import NotFound  # noqa: PLC0415
    from google.cloud import storage  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    GateResult = namedtuple("GateResult", ["passed"])

    client = storage.Client(project=project)
    bucket = client.bucket(model_bucket)

    eval_results = json.loads(bucket.blob(f"runs/{run_id}/eval_results.json").download_as_text())
    map50 = eval_results["mAP50"]
    map50_95 = eval_results["mAP50_95"]

    if map50 < 0.50 or map50_95 < 0.35:
        bucket.blob(f"runs/{run_id}/gate_failure.json").upload_from_string(
            json.dumps({"gate_failed": "gate1_absolute_floor", "run_id": run_id,
                        "metric_values": {"mAP50": map50, "mAP50_95": map50_95}}),
            content_type="application/json",
        )
        logger.info("Gate 1 failed: mAP50=%.3f mAP50_95=%.3f", map50, map50_95)
        return GateResult(passed="false")

    try:
        prod_run = json.loads(bucket.blob("production-run.json").download_as_text())
    except NotFound:
        logger.info("No production-run.json — first run, skipping gate 2")
        return GateResult(passed="true")

    if prod_run.get("manual_override"):
        return GateResult(passed="true")

    prod_map50 = json.loads(
        bucket.blob(f"runs/{prod_run['run_id']}/eval_results.json").download_as_text()
    )["mAP50"]

    if map50 < prod_map50 - 0.02:
        bucket.blob(f"runs/{run_id}/gate_failure.json").upload_from_string(
            json.dumps({"gate_failed": "gate2_regression", "run_id": run_id,
                        "metric_values": {"mAP50": map50, "prod_mAP50": prod_map50}}),
            content_type="application/json",
        )
        logger.info("Gate 2 failed: mAP50=%.3f < prod_mAP50=%.3f - 0.02", map50, prod_map50)
        return GateResult(passed="false")

    logger.info("All gates passed: mAP50=%.3f mAP50_95=%.3f", map50, map50_95)
    return GateResult(passed="true")


@component(
    base_image=_BASE_IMAGE,
    packages_to_install=["google-cloud-storage==2.18.2"],
)
def promote_model(project: str, model_bucket: str, run_id: str) -> None:
    import logging  # noqa: PLC0415
    from google.cloud import storage  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    client = storage.Client(project=project)
    bucket = client.bucket(model_bucket)
    bucket.copy_blob(bucket.blob(f"runs/{run_id}/fish-id.onnx"), bucket, "fish-id.onnx")
    logging.getLogger(__name__).info("Promoted runs/%s/fish-id.onnx → fish-id.onnx", run_id)


@component(
    base_image=_BASE_IMAGE,
    packages_to_install=["google-cloud-storage==2.18.2"],
)
def write_production_run(project: str, model_bucket: str, run_id: str) -> None:
    import json  # noqa: PLC0415
    import logging  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415
    from google.cloud import storage  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    client = storage.Client(project=project)
    bucket = client.bucket(model_bucket)
    bucket.blob("production-run.json").upload_from_string(
        json.dumps({"run_id": run_id, "promoted_at": datetime.now(timezone.utc).isoformat(),
                    "manual_override": False}),
        content_type="application/json",
    )
    logging.getLogger(__name__).info("Wrote production-run.json for run %s", run_id)


@component(
    base_image=_BASE_IMAGE,
    packages_to_install=["google-cloud-secret-manager==2.20.2", "requests==2.32.3"],
)
def trigger_github_redeploy(project: str, run_id: str) -> None:
    import logging  # noqa: PLC0415
    import requests  # noqa: PLC0415
    from google.cloud import secretmanager  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    sm = secretmanager.SecretManagerServiceClient()
    pat = sm.access_secret_version(
        request={"name": f"projects/{project}/secrets/github-deploy-pat/versions/latest"}
    ).payload.data.decode("utf-8").strip()

    resp = requests.post(
        "https://api.github.com/repos/briancorwin/fish-id/actions/workflows/deploy.yml/dispatches",
        headers={
            "Authorization": f"Bearer {pat}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": "main", "inputs": {"run_id": run_id}},
        timeout=30,
    )
    resp.raise_for_status()
    logger.info("Triggered GitHub deploy for run %s (HTTP %d)", run_id, resp.status_code)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dsl.pipeline(name="fish-id-training-pipeline")
def fish_id_pipeline(
    project: str,
    region: str,
    training_bucket: str,
    model_bucket: str,
    dataset_version: str,
    config_version: str,
    training_image: str,
    run_id: str,
    vertex_experiment: str,
    machine_type: str = "n1-standard-4",
) -> None:
    train_op = run_training_job(
        project=project,
        region=region,
        training_image=training_image,
        run_id=run_id,
        dataset_version=dataset_version,
        config_version=config_version,
        training_bucket=training_bucket,
        model_bucket=model_bucket,
        vertex_experiment=vertex_experiment,
        machine_type=machine_type,
    )

    eval_op = run_eval_job(
        project=project,
        region=region,
        training_image=training_image,
        run_id=run_id,
        dataset_version=dataset_version,
        config_version=config_version,
        training_bucket=training_bucket,
        model_bucket=model_bucket,
        vertex_experiment=vertex_experiment,
    ).after(train_op)

    gate_op = quality_gate(
        project=project,
        model_bucket=model_bucket,
        run_id=run_id,
    ).after(eval_op)

    with dsl.If(gate_op.outputs["passed"] == "true", name="gate-passed"):
        promote_op = promote_model(project=project, model_bucket=model_bucket, run_id=run_id)

        write_op = write_production_run(
            project=project, model_bucket=model_bucket, run_id=run_id
        ).after(promote_op)

        trigger_github_redeploy(project=project, run_id=run_id).after(write_op)


# ---------------------------------------------------------------------------
# Compiler entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).parent / "fish-id-training-pipeline.json"
    compiler.Compiler().compile(pipeline_func=fish_id_pipeline, package_path=str(output))
    _logger.info("Pipeline compiled to %s", output)


if __name__ == "__main__":
    main()
