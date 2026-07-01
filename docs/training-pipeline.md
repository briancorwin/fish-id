# Fish Detector — Training Pipeline

---

## Overview

```
scripts/trigger-training.py
  → submits Vertex AI PipelineJob
    → Vertex AI Pipeline (KFP v2): fish-id-training-pipeline
        ├─ train_model: Vertex AI CustomJob (GPU Spot T4 by default, or CPU-only branch via --cpu-only)
        │      Reads: gs://{PROJECT_ID}-fish-id-training/images/ + labels/ + data.yaml
        │      Writes: gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/fish-id.onnx + metadata.json
        ├─ eval_model: KFP component (runs on the training image)
        │      Reads: gs://{PROJECT_ID}-fish-id-training/images/eval/ + labels/eval/ + data.yaml
        │             gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/fish-id.onnx
        │      Writes: mAP50 / mAP50-95 / precision / recall to a Vertex AI Experiments run
        ├─ register_model: Vertex AI Model Registry
        │      Registers artifact as "fish-id" with alias "latest"
        ├─ promote_model: Vertex AI Model Registry
        │      Gates on eval mAP50 vs the current "production" alias; tags "production" on pass
        └─ trigger_deploy: GitHub API workflow_dispatch (only runs if the gate passes)
               Fires deploy.yml on main → Cloud Run redeploy with the new model
```

Each successful training run is evaluated against a held-out eval set; if it clears the quality gate it is registered, promoted, and triggers a production deploy automatically. Manual retrieval and `deploy-app.sh` are only needed for out-of-band deploys.

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

Syncs Roboflow's `train` and `valid` splits to `images/train/`, `labels/train/`, `images/val/`, `labels/val/`. If the Roboflow export has a `test` split, it's also synced to `images/eval/` and `labels/eval/` for the eval step below; if absent, the eval split is left untouched and a warning is printed.

Also uploads a single `data.yaml` to the bucket root (class names + counts, no separate `class_names.txt`). Its GCS object generation number is used by the training job as the `dataset_generation` value in `metadata.json`, so re-syncing a new Roboflow version is traceable back to a specific `data.yaml` generation.

---

## GCS Bucket Structure

**`{PROJECT_ID}-fish-id-training`** — training data (flat pool, no versioning):

```
{PROJECT_ID}-fish-id-training/
├── data.yaml            # class names + split paths, written by update-dataset.py
├── images/
│   ├── train/
│   ├── val/
│   └── eval/            # held out for eval_model; only present if the Roboflow export has a test split
└── labels/
    ├── train/           # YOLO .txt label files
    ├── val/
    └── eval/
```

**`{PROJECT_ID}-fish-id-models`** — model artifacts and pipeline assets:

```
{PROJECT_ID}-fish-id-models/
├── pipeline/
│   └── fish-id-training-pipeline.json   # compiled KFP pipeline template
└── runs/
    ├── run-2026-06-05-143000/
    │   ├── fish-id.onnx
    │   ├── metadata.json
    │   └── checkpoint/weights/last.pt   # per-epoch checkpoint; lets an interrupted run resume via --run-id
    └── ...
```

The `production` serving artifact is tracked via the Vertex AI Model Registry (`production` alias), not a fixed GCS path. The deploy workflow resolves the artifact URI from the registry at deploy time.

---

## Triggering a Training Run

**`scripts/trigger-training.py`** submits a Vertex AI PipelineJob using the pipeline template from GCS and the `:latest` training image from Artifact Registry:

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
export TRAINING_BUCKET=${GCP_PROJECT_ID}-fish-id-training
export MODEL_BUCKET=${GCP_PROJECT_ID}-fish-id-models
export GITHUB_REPO=owner/repo-name   # e.g. briancorwin/fish-id
export VERTEX_EXPERIMENT=fish-id-eval

python scripts/trigger-training.py
```

Use `--image <uri>` to override the training container image (e.g. to pin a specific `:{SHA}` tag).
Use `--cpu-only` to skip the GPU accelerator and run on a larger CPU machine instead.

The run appears under **Vertex AI → Pipelines → Runs** in the Cloud Console.

---

## Training Job (Vertex AI CustomJob)

The pipeline branches on the `cpu_only` parameter:

- **Default (GPU)**: `n1-standard-4` + 1x `NVIDIA_TESLA_T4`, `strategy=SPOT`, `restart_job_on_worker_restart=True` (via `create_custom_training_job_from_component`)
- **`--cpu-only`**: same component run as a plain KFP step with custom resource requests — 16 vCPU / 64G memory, no accelerator
- **Container image**: `:latest` tag in Artifact Registry (or `--image` override)
- **Timeout**: `7200s` (2 hours)
- **Retries**: `set_retry(num_retries=3)` on both branches
- **Service account**: `fish-id-workflows-sa` (there is no separate training service account — training, eval, register, and promote all run under the pipeline's own SA)

Both branches are resumable: if `gs://{MODEL_BUCKET}/runs/{run_id}/checkpoint/weights/last.pt` exists for the given `run_id`, training resumes from it instead of starting fresh. A checkpoint is uploaded after every epoch. `--run-id <existing-id>` on `trigger-training.py` restarts an interrupted run from its last checkpoint.

**`training/config.yaml`** — single training config baked into the container image, used as the pipeline's default parameter values:

```yaml
model: yolov8n.pt
epochs: 100
imgsz: 640
batch: 16
optimizer: AdamW
lr0: 0.001
patience: 20
```

`patience` stops training early if `mAP50` hasn't improved for that many epochs, so 100 epochs is a ceiling, not a guaranteed runtime.

**Pipeline parameters:**

| Parameter | Example | Derived from |
|---|---|---|
| `run_id` | `run-2026-06-05-143000` | Generated at submission: UTC timestamp `run-YYYY-MM-DD-HHMMSS` |
| `training_bucket` | `my-project-fish-id-training` | Env var `TRAINING_BUCKET` |
| `model_bucket` | `my-project-fish-id-models` | Env var `MODEL_BUCKET` |
| `project` | `my-gcp-project` | Env var `GCP_PROJECT_ID` |
| `region` | `us-central1` | Env var `GCP_REGION` |
| `training_image` | `...fish-id-train:latest` | Artifact Registry `:latest` tag |
| `github_repo` | `briancorwin/fish-id` | Env var `GITHUB_REPO` |
| `vertex_experiment` | `fish-id-eval` | Env var `VERTEX_EXPERIMENT` — used by `eval_model` and `promote_model` |
| `cpu_only` | `false` | `--cpu-only` flag (default: false) |
| `model_name`, `epochs`, `imgsz`, `batch`, `optimizer`, `lr0`, `patience` | see above | Default from `training/config.yaml` at compile time |

**Training script behavior (`train.py`):**
1. Read the dataset generation from `{TRAINING_BUCKET}/data.yaml` (its GCS object generation becomes `dataset_generation` in the run's metadata)
2. Download `data.yaml` + all `train`/`val` images and labels from `{TRAINING_BUCKET}` to `/app/data/`
3. If `gs://{MODEL_BUCKET}/runs/{run_id}/checkpoint/weights/last.pt` exists, resume from it; otherwise `YOLO(config["model"]).train(data=..., **params)` from scratch, uploading `weights/last.pt` to that checkpoint path after every epoch
4. Export the best checkpoint to ONNX
5. Build `metadata.json` (run_id, dataset_generation, container image tag, model architecture, training args, duration, GPU info, etc.)
6. Upload `fish-id.onnx` and `metadata.json` to `gs://{MODEL_BUCKET}/runs/run-{R}/`

**Eval Job (`eval_model` component, `training/eval.py`)** — runs after training, on the same training container image, but as a plain KFP step (not a CustomJob):
1. Download `images/eval/` + `labels/eval/` and `data.yaml` from `{TRAINING_BUCKET}`
2. Download `fish-id.onnx` for `run_id` from `{MODEL_BUCKET}`
3. Run `YOLO(...).val(...)` against the eval split and compute `mAP50`, `mAP50_95`, `precision`, `recall`, and per-class `mAP50`
4. Log the metrics to Vertex AI Experiments under `vertex_experiment`, as a run named `run_id` — this is what `promote_model` reads to decide whether to promote
5. Raises if no eval images are found at `images/eval/` — the eval split must be synced via `update-dataset.py` before training (see [Dataset Management](#dataset-management))

**`register_model` component:**
- Registers `gs://{MODEL_BUCKET}/runs/{run_id}/` as the artifact URI in Vertex AI Model Registry
- Display name: `"fish-id"`; version aliases: `"latest"`; `is_default_version=True`; `version_description=run_id`
- If a `"fish-id"` model already exists, adds a new version (passes `parent_model`); otherwise creates the model

**`promote_model` component:**
- Reads `run_id` from the `@latest` alias and `prod_run_id` from the `@production` alias
- Gates promotion: queries Vertex AI Experiments for both run IDs; if current `mAP50 < prod mAP50 - 0.02`, skips promotion
- On gate pass (or no production model yet): tags the registered version as `"production"` via `MergeVersionAliasesRequest`

**`trigger_deploy` component:**
- Reads the GitHub PAT from Secret Manager secret `fish-id-github-deploy-token`
- POSTs a `workflow_dispatch` event to `deploy.yml` on `main` via the GitHub API
- The deploy workflow then pulls the `production`-aliased model from Vertex AI Registry and redeploys Cloud Run

---

## IAM

There is no separate training service account — `train_model`, `eval_model`, `register_model`, and `promote_model` all run under the pipeline's own runtime SA (`fish-id-workflows-sa`), passed explicitly via `pipeline_job.submit(service_account=...)` in `trigger-training.py`.

**`fish-id-workflows-sa`** (pipeline components):
- `roles/aiplatform.user` on project (submit PipelineJobs and CustomJobs, register models)
- `roles/storage.objectAdmin` on `{PROJECT_ID}-fish-id-models` bucket (pipeline root artifacts)
- `roles/storage.objectViewer` on `{PROJECT_ID}-fish-id-training` bucket (read training data)
- `roles/logging.logWriter` on project
- `roles/secretmanager.secretAccessor` on `fish-id-github-deploy-token` secret (read GitHub PAT)

**`fish-id-cicd-sa`** (GitHub Actions):
- `roles/storage.objectAdmin` on `{PROJECT_ID}-fish-id-models` bucket (write compiled pipeline JSON)
- `roles/aiplatform.viewer` on project (list Model Registry versions to resolve `production` alias at deploy time)

---

## Terraform Resources

| Resource | Type |
|---|---|
| `{PROJECT_ID}-fish-id-terraform-state` bucket | `google_storage_bucket` (bootstrap: create manually first, then import) |
| `{PROJECT_ID}-fish-id-training` bucket | `google_storage_bucket` |
| `fish-id-workflows-sa` | `google_service_account` |
| IAM bindings for `fish-id-workflows-sa` (see above) | `google_storage_bucket_iam_member`, `google_project_iam_member` |
| `fish-id-github-deploy-token` | `google_secret_manager_secret` (shell only — populate manually after apply) |
| `workflows_deploy_token` IAM binding | `google_secret_manager_secret_iam_member` |

**APIs required in `apis.tf`:** `aiplatform.googleapis.com`, `secretmanager.googleapis.com`

**Post-`terraform apply` manual step** — populate the GitHub PAT secret (requires a PAT with `repo` scope or `actions:write`):

```bash
echo -n "YOUR_PAT" | gcloud secrets versions add fish-id-github-deploy-token --data-file=-
```

---

## Cost Controls

| Control | What it prevents |
|---|---|
| Vertex AI CustomJob `timeout: 7200s` | Runaway training cost |
| `strategy=SPOT` on the default GPU (T4) job | Full on-demand GPU billing |
| `--cpu-only` fallback (16 vCPU / 64G, no accelerator) | Training when GPU quota/Spot capacity is unavailable |
| Manual trigger only | Accidental pipeline runs |

**Recommended out-of-band**: a project-level GCP budget alert at `$50/month` — configure via Billing → Budgets & alerts.

---

## Deferred

- **Eventarc / Cloud Functions trigger** — pipeline is triggered manually only
- **Multi-config support** — single `config.yaml` baked into the container
