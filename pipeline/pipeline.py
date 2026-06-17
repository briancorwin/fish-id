"""Vertex AI Pipeline: training runs as a dsl.component wrapped as a Vertex AI Custom Job for the GPU Spot path."""
# pylint: disable=import-outside-toplevel
import logging
from pathlib import Path

from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from kfp import compiler, dsl

_logger = logging.getLogger(__name__)


@dsl.component(
    base_image="pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
    packages_to_install=[
        "ultralytics==8.4.66",
        "google-cloud-storage==3.12.0",
    ],
)
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
    import concurrent.futures
    import json
    import logging as _logging
    import os
    import time
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    import google.cloud.storage as gcs
    import torch
    from ultralytics import YOLO

    _log = _logging.getLogger(__name__)
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")

    class GCSCheckpointCallback:
        def __init__(self, bucket: gcs.Bucket, gcs_prefix: str) -> None:
            self._bucket = bucket
            self._gcs_prefix = gcs_prefix

        def on_train_epoch_end(self, trainer) -> None:
            local = _Path(trainer.save_dir) / "weights" / "last.pt"
            if not local.exists():
                _log.warning("[train] checkpoint not found at %s — skipping upload", local)
                return
            dest = f"{self._gcs_prefix}/weights/last.pt"
            self._bucket.blob(dest).upload_from_filename(str(local))
            _log.info("[train] checkpoint uploaded to gs://%s/%s", self._bucket.name, dest)

    def _download_checkpoint(bucket: gcs.Bucket, gcs_prefix: str, local_dir: _Path):
        blob = bucket.blob(f"{gcs_prefix}/weights/last.pt")
        if not blob.exists():
            return None
        local_path = local_dir / "last.pt"
        local_dir.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        _log.info("[train] checkpoint downloaded from gs://%s/%s", bucket.name, blob.name)
        return local_path

    def _download_prefix(bucket: gcs.Bucket, gcs_prefix: str, local_dir: _Path) -> None:
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        _log.info("[train] downloading %d files from %s", len(blobs), gcs_prefix)

        def _fetch(blob: gcs.Blob) -> None:
            filename = blob.name[len(gcs_prefix):]
            if not filename:
                return
            dest = local_dir / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(_fetch, blobs))

    def _download_training_data(storage_client: gcs.Client, bucket_name: str, local_dir: _Path) -> None:
        bucket = storage_client.bucket(bucket_name)
        local_dir.mkdir(parents=True, exist_ok=True)
        bucket.blob("data.yaml").download_to_filename(str(local_dir / "data.yaml"))
        _download_prefix(bucket, "images/train/", local_dir / "images/train")
        _download_prefix(bucket, "images/val/",   local_dir / "images/val")
        _download_prefix(bucket, "labels/train/", local_dir / "labels/train")
        _download_prefix(bucket, "labels/val/",   local_dir / "labels/val")
        _log.info("[train] data download complete")

    def _gpu_info() -> list:
        return [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_mb": torch.cuda.get_device_properties(i).total_memory // (1024 ** 2),
            }
            for i in range(torch.cuda.device_count())
        ]

    cpu_count = os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    _log.info(
        "[train] run_id=%s training_bucket=%s model_bucket=%s cpu_count=%d",
        run_id, training_bucket, model_bucket, cpu_count,
    )

    storage_client = gcs.Client()

    dataset_blob = storage_client.bucket(training_bucket).blob("data.yaml")
    dataset_blob.reload()
    dataset_generation = dataset_blob.generation
    _log.info("[train] dataset_generation=%d", dataset_generation)

    local_data_dir = _Path("/tmp/data")
    _download_training_data(storage_client, training_bucket, local_data_dir)
    data_yaml_path = str(local_data_dir / "data.yaml")

    checkpoint_prefix = f"runs/{run_id}/checkpoint"
    model_bucket_obj = storage_client.bucket(model_bucket)
    checkpoint_path = _download_checkpoint(
        model_bucket_obj, checkpoint_prefix, _Path("/tmp/yolo-checkpoint")
    )
    callback = GCSCheckpointCallback(model_bucket_obj, checkpoint_prefix)

    start = time.time()
    if checkpoint_path is not None:
        _log.info("[train] resuming from checkpoint %s", checkpoint_path)
        yolo = YOLO(str(checkpoint_path))
        yolo.add_callback("on_train_epoch_end", callback.on_train_epoch_end)
        yolo.train(resume=True)
    else:
        _log.info(
            "[train] starting fresh: model=%s epochs=%d imgsz=%d batch=%d optimizer=%s lr0=%s workers=%d",
            model_name, epochs, imgsz, batch, optimizer, lr0, cpu_count,
        )
        yolo = YOLO(model_name)
        yolo.add_callback("on_train_epoch_end", callback.on_train_epoch_end)
        yolo.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            optimizer=optimizer,
            lr0=lr0,
            workers=cpu_count,
            cache=False,
            project="/tmp/yolo-runs",
            name=".",
            save=True,
        )
    duration = time.time() - start
    _log.info("[train] training complete in %.1f seconds", duration)

    assert yolo.trainer is not None
    trainer = yolo.trainer
    _log.info("[train] YOLO training finished. save_dir=%s", trainer.save_dir)

    best_pt = str(trainer.save_dir / "weights/best.pt")
    _log.info("[train] exporting ONNX from %s", best_pt)
    best_model = YOLO(best_pt)
    best_model.export(format="onnx")
    onnx_path = best_pt.replace(".pt", ".onnx")
    _log.info("[train] ONNX exported to %s", onnx_path)

    metadata = {
        "run_id": run_id,
        "dataset_generation": dataset_generation,
        "model_architecture": model_name.replace(".pt", ""),
        "base_weights": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "epochs_completed": trainer.epoch + 1,
        "training_args": {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "optimizer": optimizer,
            "lr0": lr0,
        },
        "final_train_loss": (
            float(trainer.metrics.get("train/box_loss", 0.0)) if hasattr(trainer, "metrics") else None
        ),
        "cpu_count": cpu_count,
        "gpus": _gpu_info(),
    }
    _log.info("[train] metadata: %s", metadata)

    bucket = storage_client.bucket(model_bucket)
    run_onnx_dest = f"runs/{run_id}/fish-id.onnx"
    _log.info("[train] uploading %s -> gs://%s/%s", onnx_path, model_bucket, run_onnx_dest)
    bucket.blob(run_onnx_dest).upload_from_filename(onnx_path)
    _log.info("[train] uploading %s -> gs://%s/fish-id.onnx (production path)", onnx_path, model_bucket)
    bucket.blob("fish-id.onnx").upload_from_filename(onnx_path)

    metadata_local = "/tmp/metadata.json"
    with open(metadata_local, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    metadata_dest = f"runs/{run_id}/metadata.json"
    _log.info("[train] uploading metadata -> gs://%s/%s", model_bucket, metadata_dest)
    bucket.blob(metadata_dest).upload_from_filename(metadata_local)

    _log.info("[train] done. artifacts uploaded to gs://%s/runs/%s/", model_bucket, run_id)


_TrainGpuJobOp = create_custom_training_job_from_component(
    train_model,
    display_name="fish-id-gpu-training",
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    strategy="FLEX_START",
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
    run_id: str,
    project: str,
    region: str,
    github_repo: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 5,
    imgsz: int = 640,
    batch: int = 16,
    optimizer: str = "AdamW",
    lr0: float = 0.001,
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
        reg_cpu = register_model(
            project=project, region=region, model_bucket=model_bucket, run_id=run_id,
        ).after(cpu_train)
        trigger_deploy(project=project, github_repo=github_repo).after(reg_cpu)

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
        reg_gpu = register_model(
            project=project, region=region, model_bucket=model_bucket, run_id=run_id,
        ).after(gpu_train)
        trigger_deploy(project=project, github_repo=github_repo).after(reg_gpu)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).parent / "fish-id-training-pipeline.json"
    compiler.Compiler().compile(pipeline_func=fish_id_training_pipeline, package_path=str(output))
    _logger.info("Pipeline compiled to %s", output)


if __name__ == "__main__":
    main()
