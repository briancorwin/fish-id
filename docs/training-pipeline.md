# Fish Detector — Training Pipeline

---

## Overview

```
scripts/trigger-training.py
  → submits Vertex AI PipelineJob
    → Vertex AI Pipeline (KFP v2): fish-id-training-pipeline
        └─ run_training_job: Submit Vertex AI CustomJob
               Reads: gs://{PROJECT_ID}-fish-id-training/images/ + labels/
               Writes: gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/fish-id.onnx
```

The trained ONNX is manually retrieved from GCS and deployed via `scripts/deploy-app.sh`.

---

## Training Container Build

The training container is built by `.github/workflows/build-training-image.yml` on every push to `main` that touches `training/**` or `pipeline/**`, plus `workflow_dispatch` for ad-hoc rebuilds.

The workflow:
1. Builds `training/Dockerfile` (build context: repo root), tagged `{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:{SHA}`
2. Pushes both `:{SHA}` and `:latest` tags to Artifact Registry
3. Compiles `pipeline/pipeline.py` and uploads the compiled JSON to `gs://{PROJECT_ID}-fish-id-models/pipeline/fish-id-training-pipeline.json`

The compiled pipeline template must be current before triggering a run. Merging any change to `training/` or `pipeline/` keeps both in sync automatically.

---

## Dataset Management

**`scripts/update-dataset.py`** exports from Roboflow and syncs to the GCS training bucket:

```bash
export ROBOFLOW_API_KEY=your_key_here

python scripts/update-dataset.py \
    --roboflow-version 5 \
    --bucket ${GCP_PROJECT_ID}-fish-id-training \
    --workspace my-workspace \
    --project fish-id
```

Uses `gsutil -m rsync` without `-d` — files are never deleted from the pool. Re-running with the same version is safe and idempotent.

---

## GCS Bucket Structure

**`{PROJECT_ID}-fish-id-training`** — training data (flat pool, no versioning):

```
{PROJECT_ID}-fish-id-training/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/          # YOLO .txt label files
    └── val/
```

**`{PROJECT_ID}-fish-id-models`** — model artifacts and pipeline assets:

```
{PROJECT_ID}-fish-id-models/
├── fish-id.onnx                    # production serving path
├── pipeline/
│   └── fish-id-training-pipeline.json   # compiled KFP pipeline template
└── runs/
    ├── run-2026-06-05-143000/
    │   └── fish-id.onnx
    └── ...
```

---

## Triggering a Training Run

**`scripts/trigger-training.py`** submits a Vertex AI PipelineJob using the pipeline template from GCS and the `:latest` training image from Artifact Registry:

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
export TRAINING_BUCKET=${GCP_PROJECT_ID}-fish-id-training
export MODEL_BUCKET=${GCP_PROJECT_ID}-fish-id-models

python scripts/trigger-training.py
```

Use `--image <uri>` to override the training container image (e.g. to pin a specific `:{SHA}` tag).

The run appears under **Vertex AI → Pipelines → Runs** in the Cloud Console.

---

## Training Job (Vertex AI CustomJob)

- **Container image**: `:latest` tag in Artifact Registry (or `--image` override)
- **Machine type**: `n1-highmem-4` (CPU-only)
- **Timeout**: `7200s` (2 hours)
- **Service account**: `fish-id-training-sa`

**`training/config.yaml`** — single training config baked into the container image:

```yaml
model: yolov8n.pt
epochs: 50
imgsz: 640
batch: 16
optimizer: AdamW
lr0: 0.001
```

**Pipeline parameters:**

| Parameter | Example | Derived from |
|---|---|---|
| `run_id` | `run-2026-06-05-143000` | Generated at submission: UTC timestamp `run-YYYY-MM-DD-HHMMSS` |
| `training_bucket` | `my-project-fish-id-training` | Env var `TRAINING_BUCKET` |
| `model_bucket` | `my-project-fish-id-models` | Env var `MODEL_BUCKET` |
| `project` | `my-gcp-project` | Env var `GCP_PROJECT_ID` |
| `region` | `us-central1` | Env var `GCP_REGION` |
| `training_image` | `...fish-id-train:latest` | Artifact Registry `:latest` tag |

**Training script behavior (`train.py`):**
1. Read `config.yaml` from the container filesystem
2. Download all images + labels from GCS (`{TRAINING_BUCKET}/images/` and `labels/`) to `/tmp/dataset/`
3. `YOLO(config["model"]).train(data=..., **params)`
4. Export best checkpoint to ONNX
5. Upload `fish-id.onnx` to `gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/`

---

## Local Pipeline Testing

### KFP Local Runner

**`scripts/run-pipeline-local.py`** runs the pipeline locally via the KFP SubprocessRunner (no Vertex AI Pipelines). Accepts the same env vars and `--image` flag as `trigger-training.py`.

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
export TRAINING_BUCKET=${GCP_PROJECT_ID}-fish-id-training
export MODEL_BUCKET=${GCP_PROJECT_ID}-fish-id-models

python scripts/run-pipeline-local.py
```

Set `SHORT_CIRCUIT=true` to skip CustomJob submission and test graph wiring only — no training cost incurred:

```bash
SHORT_CIRCUIT=true python scripts/run-pipeline-local.py
```

---

## IAM

**`fish-id-training-sa`** (Vertex AI CustomJob containers):
- `roles/storage.objectViewer` on `{PROJECT_ID}-fish-id-training` bucket (read training data)
- `roles/storage.objectCreator` on `{PROJECT_ID}-fish-id-models` bucket (write run artifacts)
- `roles/aiplatform.user` on project

**`fish-id-workflows-sa`** (pipeline components):
- `roles/aiplatform.user` on project (submit PipelineJobs and CustomJobs)
- `roles/storage.objectAdmin` on `{PROJECT_ID}-fish-id-models` bucket (pipeline root artifacts)
- `roles/logging.logWriter` on project

**`fish-id-cicd-sa`** (GitHub Actions):
- `roles/storage.objectAdmin` on `{PROJECT_ID}-fish-id-models` bucket (write compiled pipeline JSON)

---

## Terraform Resources

| Resource | Type |
|---|---|
| `{PROJECT_ID}-fish-id-terraform-state` bucket | `google_storage_bucket` (bootstrap: create manually first, then import) |
| `{PROJECT_ID}-fish-id-training` bucket | `google_storage_bucket` |
| `fish-id-training-sa` | `google_service_account` |
| `fish-id-workflows-sa` | `google_service_account` |
| IAM bindings for both SAs (see above) | `google_storage_bucket_iam_member`, `google_project_iam_member` |

**API required in `apis.tf`:** `aiplatform.googleapis.com`

---

## Cost Controls

| Control | What it prevents |
|---|---|
| Vertex AI CustomJob `timeout: 7200s` | Runaway training cost |
| `n1-highmem-4` CPU-only machine type | GPU billing on a small dataset |
| Manual trigger only | Accidental pipeline runs |

**Recommended out-of-band**: a project-level GCP budget alert at `$50/month` — configure via Billing → Budgets & alerts.

---

## Deferred (add back once the pipeline is working e2e)

- **Automated promotion + redeploy** — pipeline ONNX output is manually retrieved from GCS and deployed via `deploy-app.sh` for now
- **Quality gates (Gate 1 + Gate 2)** — no eval or gating; every successful run produces an artifact
- **Eval dataset + eval job** — removed for now
- **Eventarc / Cloud Functions trigger** — pipeline is triggered manually only
- **Multi-config support** — single `config.yaml` baked into the container
