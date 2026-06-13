"""Vertex AI Pipeline: submit training CustomJob, write artifacts to GCS."""
import logging
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import component

_logger = logging.getLogger(__name__)
_BASE_IMAGE = "python:3.11-slim"


@component(
    base_image=_BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform==1.70.0"],
)
def run_training_job(
    project: str,
    region: str,
    training_image: str,
    run_id: str,
    training_bucket: str,
    model_bucket: str,
    machine_type: str,
) -> None:
    import logging  # noqa: PLC0415 — required inside KFP component body
    import os  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if os.environ.get("SHORT_CIRCUIT", "").lower() == "true":
        logger.info("SHORT_CIRCUIT=true — skipping CustomJob submission for run %s", run_id)
        return

    from google.cloud import aiplatform  # noqa: PLC0415
    from google.cloud.aiplatform_v1.types.custom_job import Scheduling  # noqa: PLC0415

    aiplatform.init(project=project, location=region)
    job = aiplatform.CustomJob(
        display_name=f"fish-id-train-{run_id}",
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": machine_type},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": training_image,
                    "env": [
                        {"name": "RUN_ID", "value": run_id},
                        {"name": "TRAINING_BUCKET", "value": training_bucket},
                        {"name": "MODEL_BUCKET", "value": model_bucket},
                        {"name": "CONTAINER_IMAGE", "value": training_image},
                        {"name": "MACHINE_TYPE", "value": machine_type},
                    ],
                },
            }
        ],
    )
    training_sa = f"fish-id-training-sa@{project}.iam.gserviceaccount.com"
    logger.info("Submitting training job fish-id-train-%s", run_id)
    job.run(sync=True, service_account=training_sa, scheduling_strategy=Scheduling.Strategy.SPOT)
    logger.info("Training job completed: %s", job.resource_name)


@dsl.pipeline(name="fish-id-training-pipeline")
def fish_id_training_pipeline(
    project: str,
    region: str,
    training_bucket: str,
    model_bucket: str,
    training_image: str,
    run_id: str,
    machine_type: str = "n1-highmem-4",
) -> None:
    run_training_job(
        project=project,
        region=region,
        training_image=training_image,
        run_id=run_id,
        training_bucket=training_bucket,
        model_bucket=model_bucket,
        machine_type=machine_type,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).parent / "fish-id-training-pipeline.json"
    compiler.Compiler().compile(pipeline_func=fish_id_training_pipeline, package_path=str(output))
    _logger.info("Pipeline compiled to %s", output)


if __name__ == "__main__":
    main()
