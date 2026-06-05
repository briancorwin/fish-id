# Fish Detector — GCP Architecture Reference

---

## System Overview

```
Browser (Firebase Hosting) → Cloud Run (Flask API + ONNX model)
```

YOLO model runs directly on Cloud Run CPU via ONNX export (~300–600ms per inference).

---

## Repo Structure

```
fish-id/
├── app/                        # Cloud Run web API
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── fish_identifier.py
│   └── rate_limiter.py
├── frontend/                   # Firebase Hosting (static)
│   ├── public/
│   │   ├── index.html
│   │   ├── css/styles.css
│   │   └── js/app.js
│   ├── firebase.json
│   └── .firebaserc
├── training/                   # Vertex AI CustomJob training container
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train.py                # entrypoint — downloads dataset from GCS, trains, exports ONNX
│   └── config.yaml             # training config (architecture + hyperparameters)
├── pipeline/                   # Vertex AI Pipeline (KFP v2)
│   ├── pipeline.py             # KFP pipeline definition; compile with: python pipeline/pipeline.py
│   └── requirements.txt
├── terraform/                  # GCP infrastructure-as-code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── apis.tf
│   ├── iam.tf
│   ├── storage.tf
│   ├── artifact_registry.tf
│   └── .terraform.lock.hcl    # committed to pin provider versions
├── .github/
│   └── workflows/
│       ├── ci.yml                       # tests + lint on every PR
│       ├── deploy.yml                   # backend + frontend deploy on merge to main
│       └── build-training-image.yml     # builds + pushes training container on training/** changes
├── scripts/
│   ├── requirements.txt
│   ├── deploy-app.sh           # manual CLI build: copies fish-id.onnx into app/, builds container, cleans up
│   ├── update-dataset.py       # exports from Roboflow, syncs images + labels to GCS training bucket
│   └── trigger-training.py     # manually submits a Vertex AI PipelineJob
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── requirements.txt
│   ├── test_main.py
│   ├── test_fish_identifier.py
│   ├── test_rate_limiter.py
│   ├── test_pipeline_components.py
│   └── test_training.py
└── README.md
```

`fish-id.onnx` is not committed to the repo. In production it is stored in GCS and downloaded during CI/CD. For local builds, place it in `app/` before running `scripts/build.sh`.

---

## GCP Services

| Need | GCP Service |
|---|---|
| Host web API | Cloud Run |
| Container builds (CLI) | Cloud Build |
| Container images | Artifact Registry |
| Container builds (GitHub Actions) | Docker on runner |
| Frontend hosting | Firebase Hosting |
| Model storage | Cloud Storage |
| Keyless CI/CD auth | Workload Identity Federation |
| Managed training jobs | Vertex AI CustomJob |
| Pipeline orchestration | Vertex AI Pipelines (KFP v2) |

---

## Key Components

### Flask API (`app/main.py`)

Accepts image uploads, runs ONNX inference, returns bounding box coordinates and fish count.
Served by gunicorn (not the Flask dev server).

Endpoints:
- `POST /detect` — accepts multipart image, returns `{ fish_count, detections: [{ class_id, confidence, box: { x1, y1, x2, y2 } }] }`
- `GET /health` — health check

The frontend is responsible for drawing bounding boxes on the image using canvas.

Image validation on upload:
- 5MB size limit
- Magic bytes checked server-side (not just Content-Type header) to confirm valid image

`CORS_ORIGIN` is set to the Firebase Hosting URL (`https://PROJECT_ID.web.app`) via Cloud Run environment variable, injected at deploy time.

### YOLO Model

- YOLOv8n fine-tuned on a specialized fish dataset created using Roboflow
- Exported to ONNX for CPU inference (~300–600ms on Cloud Run 2 vCPU)
- When deployed via GitHub Actions, `fish-id.onnx` is downloaded from GCS (`PROJECT_ID-fish-id-models` bucket) during the workflow. When deploying manually via CLI, `scripts/build.sh` expects a local copy of `fish-id.onnx`.

### Rate Limiting

- In-process token bucket (`app/rate_limiter.py`): 5 req/min per IP, burst of 3
- Cloud Run: `--max-instances 1`, `--concurrency 5`

### Frontend

Static site on Firebase Hosting; talks to Cloud Run API.
Draws bounding boxes on the image via canvas using the box coordinates returned by `/detect`.
Shows fish count and inference time.

When deployed via GitHub Actions, the Cloud Run URL is injected into `frontend/public/js/app.js` automatically, and `GCP_PROJECT_ID` in `frontend/.firebaserc` is replaced with the real project ID. When deploying manually via CLI, you must replace `https://YOUR_CLOUD_RUN_URL` in `API_BASE` in `app.js` and set `projects.default` in `.firebaserc` to your project ID before running `firebase deploy`. CORS on Cloud Run is restricted to the Firebase Hosting origin.

---

## CI/CD

GitHub Actions handles all deploys on merge to `main`. Manual CLI deployment via `scripts/build.sh` remains available.

### Workflow: `.github/workflows/deploy.yml`

Two jobs run sequentially on every merged PR:

1. **`deploy-api`** — downloads `fish-id.onnx` from GCS into `app/`, builds the Docker image, pushes to Artifact Registry, deploys to Cloud Run
2. **`deploy-frontend`** — fetches the Cloud Run URL, injects it into `app.js`, deploys `frontend/` to Firebase Hosting

Both jobs authenticate using **Workload Identity Federation** — no long-lived service account keys are stored anywhere. GitHub Actions receives a short-lived OIDC token that is exchanged for GCP credentials scoped to the `fish-id-cicd-sa` service account.

### GitHub Secrets Required

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | Cloud Run / Artifact Registry region |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output from `terraform output workload_identity_provider` |
| `GCP_SERVICE_ACCOUNT` | Output from `terraform output cicd_service_account_email` |
| `ONNX_MODEL_GCS_URI` | `gs://PROJECT_ID-fish-id-models/fish-id.onnx` |

---

## Continuous Training Pipeline

### Overview

```
scripts/trigger-training.py
  → submits Vertex AI PipelineJob
    → Vertex AI Pipeline (KFP v2): fish-id-training-pipeline
        └─ run_training_job: Submit Vertex AI CustomJob
               Reads: gs://{PROJECT_ID}-fish-id-training/images/ + labels/
               Writes: gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/fish-id.onnx
                       gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/metadata.json
```

Pipeline output is manually retrieved from GCS and deployed via `scripts/deploy-app.sh`.

---

### Dataset Management

**`scripts/update-dataset.py`** exports from Roboflow and syncs to the GCS training bucket:

```bash
python scripts/update-dataset.py \
    --roboflow-version 5 \
    --bucket {PROJECT_ID}-fish-id-training \
    --workspace my-workspace \
    --project fish-id
```

Requires `ROBOFLOW_API_KEY` env var. Uses `gsutil -m rsync` (no `-d` flag — files are never deleted from the pool). Re-running with the same version is safe and idempotent.

---

### GCS Bucket Structure

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

**`{PROJECT_ID}-fish-id-models`** — model artifacts:

```
{PROJECT_ID}-fish-id-models/
├── fish-id.onnx                    # production serving path — unchanged
├── training-image-latest.json      # SHA of the latest training container image
├── production-run.json             # { "run_id": "run-2026-06-05-143000", "promoted_at": "..." }
└── runs/
    ├── run-2026-06-05-143000/
    │   ├── fish-id.onnx
    │   └── metadata.json
    └── ...
```

The root `fish-id.onnx` path is unchanged, so `deploy.yml`'s existing `gsutil cp` step continues to work without modification.

---

### Training Container Build

The training container is built and pushed by `.github/workflows/build-training-image.yml`, separate from `deploy.yml`.

**Trigger**: `push` to `main` with path filter on `training/**`, plus `workflow_dispatch` for ad-hoc rebuilds.

**Auth**: same WIF / `fish-id-cicd-sa` pattern as `deploy.yml`.

**Steps:**
1. Build `training/Dockerfile`, tag as `{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:${{ github.sha }}`
2. Push to Artifact Registry
3. Write `gs://{PROJECT_ID}-fish-id-models/training-image-latest.json`:
   ```json
   { "image": "{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:abc1234", "built_at": "<UTC>", "git_sha": "abc1234" }
   ```

`scripts/trigger-training.py` reads `training-image-latest.json` to determine the container image when submitting the PipelineJob. Use `--image <ref>` to override for ad-hoc experimentation.

**Dockerfile responsibilities:**
- Base: `python:3.11-slim`
- Install dependencies: `ultralytics>=8.3`, `google-cloud-storage`, `google-cloud-aiplatform`
- Pre-download base weights: `RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` — bakes pretrained weights into the image so training has no runtime dependency on `ultralytics.com`
- `COPY training/config.yaml` so the training config is part of the image
- `ENTRYPOINT ["python", "train.py"]`

---

### Training Job (Vertex AI CustomJob)

- **Container image**: resolved at pipeline submission from `training-image-latest.json`
- **Machine type**: `n1-highmem-4` (CPU-only)
- **Timeout**: `7200s` (2 hours)
- **Service account**: `fish-id-training-sa`

**`training/config.yaml`** — single training config baked into the image:

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
| `training_image` | `...fish-id-train:{SHA}` | Read from `training-image-latest.json` |

**Training script behavior (`train.py`):**
1. Read `config.yaml` from the container filesystem
2. Download all images + labels from GCS (`{TRAINING_BUCKET}/images/` and `labels/`) to `/tmp/dataset/`
3. `YOLO(config["model"]).train(data=..., **params)`
4. Export best checkpoint to ONNX
5. Upload `fish-id.onnx` and `metadata.json` to `gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/`

**`metadata.json` schema:**
```json
{
  "run_id": "run-2026-06-05-143000",
  "container_image": "{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:abc1234",
  "model_architecture": "yolov8n",
  "base_weights": "yolov8n.pt",
  "trained_at": "2026-06-05T14:30:00Z",
  "duration_seconds": 1842,
  "epochs_completed": 50,
  "training_args": { "imgsz": 640, "batch": 16, "optimizer": "AdamW", "lr0": 0.001 },
  "machine_type": "n1-highmem-4"
}
```

---

### Local Pipeline Testing

#### KFP Local Runner

The pipeline can be run locally without Vertex AI using the KFP local runner (KFP >= 2.7):

```python
from kfp.local import init, SubprocessRunner
init(runner=SubprocessRunner(use_venv=True))

from pipeline.pipeline import fish_id_training_pipeline
fish_id_training_pipeline(
    run_id="run-test-local",
    training_bucket="my-project-fish-id-training",
    model_bucket="my-project-fish-id-models",
    project="my-gcp-project",
    region="us-central1",
    training_image="...",
)
```

`SubprocessRunner` executes each KFP component in a subprocess on the local machine. Use `DockerRunner` to test with the full container image.

#### Colab Short-Circuits

Pipeline components that submit Vertex AI CustomJobs support a `SHORT_CIRCUIT` env var. When set, the training component runs a minimal inline training pass instead of submitting a CustomJob. This lets you verify the full pipeline graph executes and that data flows correctly between components without incurring training cost.

```python
# In Colab, set before running the pipeline
import os
os.environ["SHORT_CIRCUIT"] = "true"
```

Each component checks:
```python
if os.environ.get("SHORT_CIRCUIT"):
    # run a minimal inline pass (e.g. 1 epoch on 10 images)
else:
    # submit the Vertex AI CustomJob
```

Short-circuit mode uses the same code paths as production — it just skips the CustomJob submission and runs inline, so component logic and data flow are verified end-to-end.

---

### IAM

**`fish-id-training-sa`** (Vertex AI CustomJob containers):
- `roles/storage.objectViewer` on `{PROJECT_ID}-fish-id-training` bucket (read training data)
- `roles/storage.objectCreator` on `{PROJECT_ID}-fish-id-models` bucket (write run artifacts)
- `roles/aiplatform.user` on project

**`fish-id-workflows-sa`** (pipeline components):
- `roles/aiplatform.user` on project (submit PipelineJobs and CustomJobs)
- `roles/storage.objectAdmin` on `{PROJECT_ID}-fish-id-models` bucket (pipeline root artifacts)
- `roles/logging.logWriter` on project

**`fish-id-cicd-sa`** (GitHub Actions — existing SA, one new binding):
- Add `roles/storage.objectAdmin` on `{PROJECT_ID}-fish-id-models` bucket (to write `training-image-latest.json` — objectCreator is insufficient for overwriting)

---

### Terraform Resources

| Resource | Type |
|---|---|
| `{PROJECT_ID}-fish-id-terraform-state` bucket | `google_storage_bucket` (bootstrap: create manually first, then import) |
| `{PROJECT_ID}-fish-id-training` bucket | `google_storage_bucket` |
| `fish-id-training-sa` | `google_service_account` |
| `fish-id-workflows-sa` | `google_service_account` |
| IAM bindings for both SAs (see above) | `google_storage_bucket_iam_member`, `google_project_iam_member` |

**APIs to enable in `apis.tf`:**
- `aiplatform.googleapis.com`

---

### Pipeline Cost Controls

| Control | What it prevents |
|---|---|
| Vertex AI CustomJob `timeout: 7200s` | Runaway training cost |
| `n1-highmem-4` CPU-only machine type | GPU billing on a small dataset |
| Manual trigger only (no Eventarc) | Accidental pipeline runs during bulk dataset uploads |

**Recommended out-of-band**: a project-level GCP budget alert at `$50/month`. Not enforced by code — configure via Billing → Budgets & alerts.

---

## Deferred (add back once the pipeline is working e2e)

These were intentionally removed to establish a working e2e baseline first:

- **Eventarc / Cloud Functions trigger** — removed entirely; pipeline is triggered manually via `scripts/trigger-training.py`
- **Quality gates (Gate 1 + Gate 2)** — no eval or gating; every successful training run auto-promotes
- **Eval dataset + eval job** — `eval.py`, `run_eval_job` component, and eval GCS structure removed for now
- **Re-baselining** — not applicable without quality gates
- **Multi-config support** — single `config.yaml` instead of per-run config selection
- **Automated promotion + redeploy** (`promote_model`, `trigger_github_redeploy` pipeline components) — pipeline output (ONNX) is manually retrieved from GCS and deployed via `deploy-app.sh` for now

---

## Security

See **[docs/security.md](security.md)** for the full security reference, including GitHub repository security settings (secret scanning, Dependabot, CodeQL), CI security jobs, CI/CD authentication, and runtime controls.

---

## Cost Controls (Serving)

| Control | What it prevents |
|---|---|
| `--max-instances 1` on Cloud Run | Horizontal scaling charges |
| Per-IP token bucket (5 rpm) | Single user spamming |
| 5MB image size limit | Large payload abuse |

---

## YOLO Performance Reference (Cloud Run 2 vCPU, ONNX)

| Variant | Params | CPU (ONNX) |
|---|---|---|
| YOLOv8n | 3.2M | ~300–600ms |
| YOLOv8s | 11.2M | ~800ms–1.5s |
| YOLOv8m | 25.9M | ~2–4s |

**Stick with nano or small** to keep inference under 1s on Cloud Run CPU.
