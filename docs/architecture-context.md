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
│   ├── train.py                # entrypoint — reads manifest, trains, exports ONNX
│   ├── eval.py                 # entrypoint — runs YOLO.val() against eval set
│   └── configs/
│       ├── c1.yaml
│       └── c2.yaml
├── pipeline/                   # Vertex AI Pipeline (KFP v2)
│   ├── pipeline.py             # KFP pipeline definition; compile with: python pipeline/pipeline.py
│   ├── requirements.txt
│   └── trigger/
│       ├── main.py             # Cloud Functions v2 entry point (Eventarc → PipelineJob)
│       └── requirements.txt
├── terraform/                  # GCP infrastructure-as-code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── apis.tf
│   ├── iam.tf
│   ├── storage.tf
│   ├── artifact_registry.tf
│   └── .terraform.lock.hcl    # committed to pin provider versions
├── notebooks/                  # Jupyter notebooks (e.g. Colab training)
├── .github/
│   └── workflows/
│       ├── ci.yml                       # tests + lint on every PR
│       ├── deploy.yml                   # backend + frontend deploy on merge to main
│       └── build-training-image.yml     # builds + pushes training container on training/** changes
├── scripts/
│   ├── build.sh                  # manual CLI build: copies fish-id.onnx into app/, builds container, cleans up
│   ├── update-dataset.py         # exports from Roboflow, syncs to GCS pool, writes manifest
│   ├── trigger-training.py       # manually triggers a training run with chosen dataset + config
│   ├── promote-run.py            # promotes any previous run to production (rollback)
│   └── rebaseline-production.py  # re-scores the production model against a new eval set
├── tests/
│   ├── conftest.py
│   ├── requirements.txt
│   ├── test_main.py
│   └── test_rate_limiter.py
└── README.md
```

`fish-id.onnx` is not committed to the repo. In production it is stored in GCS and downloaded during CI/CD. For local builds, place it in `app/` before running `scripts/deploy-app.sh`.

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
| Eval metric tracking & trend analysis | Vertex AI Experiments |
| Dataset-change trigger | Cloud Functions v2 (Eventarc) |
| Pipeline orchestration | Vertex AI Pipelines (KFP v2) |
| Sensitive config values | Secret Manager |

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
- When deployed via GitHub Actions, `fish-id.onnx` is downloaded from GCS (`PROJECT_ID-fish-id-models` bucket) during the workflow. When deploying manually via CLI, `scripts/deploy-app.sh` expects a local copy of `fish-id.onnx` passed as an argument.

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

GitHub Actions handles all deploys on merge to `main`. Manual CLI deployment via `scripts/deploy-app.sh` remains available.

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
Dataset upload (GCS manifest.json finalized)
  → Eventarc trigger
    → Cloud Functions v2: fish-id-pipeline-trigger
        reads training-image-latest.json, submits Vertex AI PipelineJob
      → Vertex AI Pipeline run (KFP v2): fish-id-training-pipeline
          ├─ run_training_job: Submit Vertex AI CustomJob (train)
          │    Reads:  gs://{PROJECT_ID}-fish-id-training/versions/v{N}/manifest.json
          │            gs://{PROJECT_ID}-fish-id-training/images/ + labels/ (pool, manifest-filtered)
          │            configs/c{M}.yaml (baked into container image)
          │    Writes: gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/fish-id.onnx
          │            gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/metadata.json
          ├─ run_eval_job: Submit Vertex AI CustomJob (eval)
          │    Reads:  gs://{PROJECT_ID}-fish-id-training/eval/current.json (version pointer)
          │            gs://{PROJECT_ID}-fish-id-training/eval/images/ + labels/ (manifest-filtered)
          │            gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/fish-id.onnx
          │    Writes: gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/eval_results.json
          │    Logs:   metrics to Vertex AI Experiments (experiment=fish-id-eval, run=run-{R})
          ├─ quality_gate: Read eval_results.json, apply Gate 1 + Gate 2
          │    PASS → continue
          │    FAIL → write runs/run-{R}/gate_failure.json, halt
          ├─ promote_model: Copy runs/run-{R}/fish-id.onnx → gs://{PROJECT_ID}-fish-id-models/fish-id.onnx
          ├─ write_production_run: Overwrite production-run.json
          │    → { "run_id": "run-{R}", "promoted_at": "...", "manual_override": false }
          └─ trigger_github_redeploy: HTTP call to GitHub Actions API (PAT from Secret Manager)
               → workflow_dispatch on .github/workflows/deploy.yml
                 → existing deploy-api job: build container → Artifact Registry → Cloud Run
```

---

### Eventarc Trigger

- **Source**: `google.cloud.storage.object.v1.finalized`
- **Bucket**: `{PROJECT_ID}-fish-id-training`
- **Path filter**: none at the Eventarc level — the Cloud Functions v2 trigger fires on every object finalization in the training bucket and filters internally for `versions/*/manifest.json` via regex
- **Target**: Cloud Functions v2 function `fish-id-pipeline-trigger`

Filtering inside the function (not at the Eventarc level) is simpler than Eventarc's path-pattern syntax and avoids firing on every individual image upload during a bulk sync. A new dataset version is considered "ready" only when its manifest is written.

---

### GCS Bucket Structure

**New bucket: `{PROJECT_ID}-fish-id-training`** (versioning enabled, uniform bucket-level access)

Images and labels are stored once in a flat pool. Each dataset version is a manifest that lists which files belong to it — no full copies per version.

```
{PROJECT_ID}-fish-id-training/
├── images/
│   ├── train/                     # flat pool of all training images (never deleted)
│   └── val/
├── labels/
│   ├── train/                     # YOLO .txt label files (one per image)
│   └── val/
├── versions/
│   ├── v1/
│   │   ├── manifest.json          # lists which filenames are in this version's train/val splits
│   │   └── data.yaml              # ultralytics dataset config referencing local paths
│   └── v2/
│       ├── manifest.json          # superset of v1 files + newly added images
│       └── data.yaml
└── eval/
    ├── images/                    # flat pool of eval images (never deleted)
    ├── labels/
    ├── versions/
    │   ├── v1/
    │   │   └── manifest.json      # lists which eval images belong to this version
    │   └── v2/
    │       └── manifest.json
    └── current.json               # { "eval_version": "v1" } — pointer to active eval version
```

The training container reads `versions/v{N}/manifest.json` to get the file list, downloads only those files from the pool to `/tmp/dataset/`, then runs training. New images added in v2 sit alongside v1 images in the pool; v2's manifest simply lists all of them.

The eval job follows the same pattern: reads `eval/current.json` to find the active version, reads `eval/versions/v{K}/manifest.json` for the file list, then downloads only those files from the eval pool. Updating the eval set means uploading new images to `eval/images/` and `eval/labels/`, writing a new `eval/versions/v{K+1}/manifest.json`, and updating `eval/current.json` to point to it — no files are ever duplicated.

**Additions to existing `{PROJECT_ID}-fish-id-models`**:

```
{PROJECT_ID}-fish-id-models/
├── fish-id.onnx                                   # production serving path — unchanged
├── training-image-latest.json                     # SHA of the latest training container build (written by build-training-image.yml)
├── production-run.json                            # { "run_id": "run-2026-05-25-104530", "promoted_at": "...", "manual_override": false }
└── runs/
    ├── run-2026-05-24-093015/
    │   ├── fish-id.onnx                           # ONNX export for this run
    │   ├── metadata.json                          # training provenance (run ID, dataset version, config version, container image, args)
    │   └── eval_results.json                      # quality metrics from eval job
    ├── run-2026-05-25-104530/
    │   └── ...
    └── ...
```

The root `fish-id.onnx` path is unchanged, so `deploy.yml`'s existing `gsutil cp` step (which reads `gs://{PROJECT_ID}-fish-id-models/fish-id.onnx` via the `ONNX_MODEL_GCS_URI` secret) continues to work without modification. The promotion pipeline just controls what ends up there.

`scripts/deploy-app.sh` (manual CLI build) is orthogonal to the automated pipeline. It currently takes a local ONNX path as a positional argument and bakes that file into the container at build time, with no GCS interaction. The pipeline doesn't depend on `deploy-app.sh`, but `deploy-app.sh` itself gets one new responsibility — see "Manual model override" below.

---

### Dataset Versioning

Versions are sequential integers prefixed with `v` (`v1`, `v2`, …). A version is "ready" — and the training pipeline is triggered — when `versions/v{N}/manifest.json` is finalized in GCS.

**Updating the dataset (Roboflow → GCS workflow):**

`scripts/update-dataset.py` handles the full update flow. Run it locally whenever a new labeled dataset version is ready in Roboflow:

```bash
python scripts/update-dataset.py --roboflow-version 5 --dataset-version v5 \
  --bucket {PROJECT_ID}-fish-id-training \
  --description "Added 200 new Bluegill images from Lake Michigan survey"
```

Steps the script performs (in order):
1. **Export from Roboflow**: calls `project.version(N).download("yolov8")` via the Roboflow Python SDK; downloads and extracts the zip to a local temp directory. Roboflow API key is read from the `ROBOFLOW_API_KEY` env var (stored in `.env`, never committed).
2. **Sync to GCS pool**: runs `gsutil -m rsync -r` for each of the four directories (`train/images`, `train/labels`, `valid/images`, `valid/labels`) from the local export into the corresponding GCS pool paths. Only new or changed files are uploaded; unchanged files are skipped. Does **not** use `-d` (delete) so files removed from a Roboflow version are retained in the pool.
3. **Generate manifest**: lists all objects currently under `gs://{BUCKET}/images/train/` and `gs://{BUCKET}/images/val/` using the GCS client library (`google-cloud-storage`). Builds the `manifest.json` with version metadata, file lists, and image counts (see schema below).
4. **Upload manifest**: writes `manifest.json` to `gs://{BUCKET}/versions/v{N}/manifest.json`. This is the final step — Eventarc fires on this write and triggers the training pipeline. If any earlier step fails, the script exits before this write so no pipeline is triggered on a partial sync.

**`manifest.json` schema:**
```json
{
  "version": "v3",
  "created_at": "2026-05-22T10:00:00Z",
  "created_by": "user@example.com",
  "description": "Added 200 new Bluegill images from Lake Michigan survey",
  "image_count": { "train": 1240, "valid": 310 },
  "class_names": ["Largemouth Bass", "Bluegill", "Walleye"],
  "roboflow_version": 5,
  "parent_version": "v2",
  "train_files": ["img001.jpg", "img002.jpg", "..."],
  "val_files": ["img_val001.jpg", "..."]
}
```

The `train_files` and `val_files` arrays are what the training container uses to resolve which files to download from the pool. Files removed from a newer Roboflow version are simply absent from the new manifest; they remain in the pool but are not used for training.

**Note on `class_names`**: this field is informational and is used by the training container to construct the `data.yaml` it passes to Ultralytics. The Cloud Run serving app does **not** read class names from the manifest — it reads them from the ONNX model's embedded metadata, which YOLOv8 populates at export time from the dataset definition. The two are always in sync because the training run is what produces the ONNX file. The manifest's copy exists so that the human-readable dataset record is self-contained without having to crack open the ONNX file.

The eval dataset uses the same pool + manifest pattern and is versioned separately. `eval/current.json` is only updated by deliberate human action — not automatically when training data changes. This ensures quality gate comparisons between consecutive model versions always measure the same thing.

---

### Training Container Build

The training container is built and pushed by a dedicated workflow, `.github/workflows/build-training-image.yml`, separate from `deploy.yml` so that pushes that don't touch training code don't trigger unnecessary CI cost.

**Trigger**: `push` to `main` with a path filter on `training/**` (and `manual_dispatch` for ad-hoc rebuilds).

**Auth**: same WIF / `fish-id-cicd-sa` pattern as `deploy.yml`. No new GitHub secrets required.

**Steps:**
1. Build `training/Dockerfile` and tag as `{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:${{ github.sha }}`
2. Push to Artifact Registry (the existing `fish-id` repo holds this second image alongside `fish-id`)
3. Write `gs://{PROJECT_ID}-fish-id-models/training-image-latest.json` with:
   ```json
   { "image": "{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:abc1234", "built_at": "<UTC>", "git_sha": "abc1234" }
   ```

The Cloud Functions v2 trigger reads `training-image-latest.json` when submitting the PipelineJob to determine which training container image to pass as a pipeline parameter. Auto-triggered runs always use the latest image; `scripts/trigger-training.py --image <ref>` can override for ad-hoc experimentation.

**Dockerfile responsibilities** (relevant because they remove runtime egress dependencies):
- Base: `python:3.11-slim`
- Install dependencies: `ultralytics>=8.3`, `google-cloud-storage`, `google-cloud-aiplatform`
- **Pre-download base weights**: `RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8m.pt')"` bakes the Ultralytics pretrained weights into the image so training has no runtime dependency on `ultralytics.com`. If new architectures are added to a config, the Dockerfile's pre-download list must be updated.
- `COPY training/configs/` so configs are part of the image (no GCS read needed at training time)
- `ENTRYPOINT` selects `train.py` or `eval.py` based on a `JOB_MODE` env var

**IAM requirement**: the existing `fish-id-cicd-sa` already has `roles/artifactregistry.writer` on the `fish-id` repo, so no IAM changes are needed for the push. It also needs `roles/storage.objectCreator` on the `{PROJECT_ID}-fish-id-models` bucket to write `training-image-latest.json` — this is a new binding (existing bindings only grant `roles/storage.objectViewer` for downloading `fish-id.onnx`).

---

### Training Job (Vertex AI CustomJob)

- **Container image**: `{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:{GIT_SHA}` — resolved at pipeline submission time by the Cloud Functions trigger reading `training-image-latest.json` and passing it as a pipeline parameter (see "Training Container Build" above)
- **Machine type**: `n1-highmem-4` (CPU-only) — switch to `n1-standard-4 + T4` only if dataset exceeds ~5000 images
- **CustomJob `timeout`**: `7200s` (2 hours). Hard cap on runaway training cost; any run exceeding this is killed by Vertex AI and `gate_failure.json` is written.
- **Service account**: `fish-id-training-sa`

**Training configs** live in the repo at `training/configs/` and are baked into the training container image at build time — they're part of the Docker build context. The training script selects the right file using the `CONFIG_VERSION` env var. Each file is a YAML of ultralytics training parameters:

```
training/configs/
├── c1.yaml
└── c2.yaml
```

```yaml
# c2.yaml — example switching to a larger architecture
model: yolov8m.pt
epochs: 75
imgsz: 640
batch: 8
optimizer: AdamW
lr0: 0.001
```

A new config version (`c2.yaml`) is created by adding a file to the repo and pushing — no other changes required. The training script reads `configs/c{M}.yaml` from the local container filesystem and passes the parameters to `YOLO(config["model"]).train(**params)`.

**Pipeline parameters** (passed by the Cloud Functions trigger when submitting the PipelineJob):

| Parameter | Example value | Purpose | Derived from |
|---|---|---|---|
| `run_id` | `run-2026-05-25-104530` | Output path prefix — constructs `runs/run-2026-05-25-104530/` in the models bucket | Cloud Functions trigger generates at submission time: UTC timestamp formatted `run-YYYY-MM-DD-HH-MM-SS`. Naturally sortable and stateless |
| `dataset_version` | `v5` | Which dataset manifest to fetch — constructs `versions/v5/manifest.json` | Parsed from the Eventarc event payload GCS object path |
| `config_version` | `c2` | Which training config to read from the container — selects `configs/c2.yaml` from the baked-in configs directory | Auto-trigger: latest config version; manual trigger: caller-specified |
| `training_bucket` | `my-project-fish-id-training` | GCS bucket for image pool, manifests, and configs | Cloud Functions env var `TRAINING_BUCKET` |
| `model_bucket` | `my-project-fish-id-models` | GCS bucket for model artifact output | Cloud Functions env var `MODEL_BUCKET` |
| `project` | `my-gcp-project` | Required by GCS and Vertex AI Python clients | Cloud Functions env var `GCP_PROJECT_ID` |
| `region` | `us-central1` | Required by Vertex AI Python client for regional API endpoints | Cloud Functions env var `GCP_REGION` |
| `vertex_experiment` | `fish-id-eval` | Vertex AI Experiment name for metric logging | Cloud Functions env var `VERTEX_EXPERIMENT` |
| `training_image` | `{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:{SHA}` | Container image used for train + eval CustomJobs | Read from `training-image-latest.json` by Cloud Functions trigger |

**Triggering:**
- **Auto-trigger**: Eventarc fires on new dataset manifest upload → Cloud Functions v2 trigger filters for `versions/*/manifest.json`, reads `training-image-latest.json`, and submits a Vertex AI PipelineJob with the triggering dataset version and the latest config version
- **Manual trigger**: `scripts/trigger-training.py --dataset-version v5 --config-version c2 [--image <ref>]` submits a Vertex AI PipelineJob directly, allowing experimentation with any dataset + config combination without uploading new data. The optional `--image` flag overrides the default of reading `training-image-latest.json`, useful for testing an unmerged training-image build

**Training script behavior:**
1. Read `configs/c{M}.yaml` from the local container filesystem to get architecture and hyperparameters
2. Read `versions/v{N}/manifest.json` from GCS to get the training file lists
3. Download only the listed files from the pool to `/tmp/dataset/`
4. `YOLO(config["model"]).train(data=..., **params)` — base pretrained weights specified by config
5. Export best checkpoint to ONNX
6. Upload `fish-id.onnx` and `metadata.json` to `gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/`

**`metadata.json` schema:**
```json
{
  "run_id": "run-2026-05-25-104530",
  "dataset_version": "v5",
  "config_version": "c2",
  "config_file": "c2.yaml",
  "container_image": "{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:abc1234",
  "model_architecture": "yolov8m",
  "base_weights": "yolov8m.pt",
  "trained_at": "2026-05-25T10:45:30Z",
  "duration_seconds": 1842,
  "epochs_completed": 75,
  "training_args": { "imgsz": 640, "batch": 8, "optimizer": "AdamW", "lr0": 0.001 },
  "final_train_loss": 0.032,
  "machine_type": "n1-highmem-4"
}
```

The `container_image` field with the git SHA captures which version of the training script ran — no separate script version counter needed.

**The evaluation job** runs as a second CustomJob (same container, `eval` mode):
1. Read `eval/current.json` to find the active eval version
2. Read `eval/versions/v{K}/manifest.json` to get the eval file list
3. Download only those files from the eval pool to `/tmp/eval/`
4. Load `fish-id.onnx` from `runs/run-{R}/`
5. Run `YOLO.val()` against `/tmp/eval/`
6. Write `eval_results.json`; log metrics to Vertex AI Experiments run `run-{R}`

---

### Model Tracking

Model versioning uses the same GCS pool + metadata pattern as the training and eval datasets — no separate registry service.

Each run produces three files in `gs://{PROJECT_ID}-fish-id-models/runs/run-{R}/`:

| File | Contents |
|---|---|
| `fish-id.onnx` | The trained model artifact |
| `metadata.json` | Full provenance: run ID, dataset version, config version, container image SHA, training args, duration, machine type |
| `eval_results.json` | Quality metrics from the eval job; also logged to Vertex AI Experiments |

`production-run.json` at the bucket root is the promotion pointer — it records which run is currently serving production traffic. Cloud Workflows reads it during the quality gate (to get the baseline metrics for regression comparison) and overwrites it on promotion.

Run directories are never deleted. Any run can be promoted at any time via `scripts/promote-run.py`.

---

### Quality Gates (MLOps)

**Metrics tracked per run** (written to `eval_results.json` and logged to Vertex AI Experiments):

| Metric | What it measures | Gating? |
|---|---|---|
| `mAP50` | Mean Average Precision at IoU≥0.5 — a detection counts as correct if the predicted box overlaps the ground truth by at least 50%. Primary signal for "did we find the fish and locate it approximately?" | Yes — Gate 1 floor + Gate 2 regression |
| `mAP50-95` | Average of mAP at ten IoU thresholds (0.50, 0.55, … 0.95). Rewards tight, accurately-fitted boxes; penalises sloppy localisation. | Yes — Gate 1 floor only |
| `precision` | Of all predicted detections, what fraction were correct. Low precision = model hallucinates fish that aren't there. | No — tracked only |
| `recall` | Of all ground-truth fish in the eval set, what fraction were detected. Low recall = model misses fish. | No — tracked only |
| per-class mAP50 | mAP50 broken down per species (Largemouth Bass, Bluegill, etc.) | No — tracked only |

Both zero-detection and multi-detection edge cases are handled by the mAP calculation: missed fish reduce recall and AP; duplicate or hallucinated boxes are collapsed by NMS first, then any remainder score as false positives reducing precision and mAP. A model that consistently outputs nothing fails Gate 1 immediately; a model that hallucinates freely fails through low mAP50 even if recall appears high.

**Gate 1 — Absolute floor** (both must pass):
- `mAP50 >= 0.50`
- `mAP50-95 >= 0.35`

**Gate 2 — Regression protection**:
- `new_mAP50 >= prod_mAP50 - 0.02` (2% slack for eval variance)

On first deployment (no production model exists), only Gate 1 applies.

**On gate success**: the `quality_gate` pipeline component passes, and downstream components (`promote_model`, `write_production_run`, `trigger_github_redeploy`) execute in sequence.

**On gate failure**: the `quality_gate` component writes `gate_failure.json` to `runs/run-{R}/` (capturing which gate failed and the metric values) and raises an exception, causing the pipeline run to fail. No redeploy occurs. The run's artifacts remain in GCS for inspection. The Vertex AI Pipeline run graph is queryable in the Cloud Console under **Vertex AI → Pipelines → Runs**.

Threshold note: A well-tuned YOLOv8n on a specialized dataset of ~1000 images typically achieves mAP50 of 0.60–0.85. The 0.50 floor should be tightened toward observed baselines after the first successful run.

Vertex AI Experiments provides a free, queryable time-series view of all metrics across runs — enabling trend analysis ("is quality improving over time?") directly in the Cloud Console without custom query logic.

---

### Re-baselining the production model

Gate 2's regression check (`new_mAP50 >= prod_mAP50 - 0.02`) compares the new run's `eval_results.json` to the production run's `eval_results.json`. This comparison is only valid when both were scored against the **same** eval set. Because `eval/current.json` only changes by deliberate human action, this invariant holds automatically across consecutive training runs.

However, when the eval set itself is updated (new eval images added, `eval/current.json` advanced to a new version), the production run's `eval_results.json` reflects scores against the *old* eval set and is no longer comparable. Without intervention, the next training run's Gate 2 silently becomes apples-to-oranges.

**`scripts/rebaseline-production.py`** handles this. Run it once, immediately after updating `eval/current.json` to a new version, before any new training:

```bash
python scripts/rebaseline-production.py
```

Steps:
1. Read `production-run.json` to get `run_id`. If `run_id` is null (manual override active), print a warning that re-baselining is not applicable and exit.
2. Submit a Vertex AI CustomJob in `JOB_MODE=eval` with `RUN_ID=<production_run_id>` and the current eval version. The same training container image is used; the job downloads `runs/{production_run_id}/fish-id.onnx` and runs `YOLO.val()` against the new eval set.
3. Overwrite `runs/{production_run_id}/eval_results.json` with the new scores. The new values are also logged to Vertex AI Experiments as a separate run (`run-{production_run_id}-rebaseline-{timestamp}`) so the trend record is preserved.

Operational rule: **always re-baseline immediately after updating `eval/current.json`**. The plan deliberately puts this in a human-triggered script rather than wiring it into the Vertex AI Pipeline because the eval-set update itself is human-triggered — coupling them keeps the operator in the loop and avoids invisible scoring changes.

---

### Redeployment Trigger

After model promotion, the `trigger_github_redeploy` pipeline component makes an HTTP call to the GitHub Actions API to trigger `workflow_dispatch` on the existing `.github/workflows/deploy.yml`. A `run_id` input (e.g. `run-2026-05-25-104530`) is passed to the workflow so the Cloud Run revision can be labelled for traceability.

The GitHub PAT (with `workflow` scope) is stored in **Secret Manager** and read by the pipeline component at execution time (no long-lived credentials in source).

The existing `deploy.yml` gains the following changes (more involved than the framing implies — the existing `gcloud run deploy` call is a single multi-line bash command, so the conditional flag has to be built up as a shell variable):

1. Add a `workflow_dispatch` trigger alongside the existing `push` trigger, with an optional `run_id` string input. The `push` trigger's behavior is unchanged.
2. Plumb `inputs.run_id` from the workflow level into the `deploy-api` job — workflow-level inputs are not automatically visible inside job steps and have to be referenced via `${{ inputs.run_id }}` (or via a step-level `env:` mapping).
3. Inside the `deploy-api` "Deploy to Cloud Run" step, build a conditional flag string before invoking `gcloud run deploy`:
   ```bash
   REVISION_SUFFIX_FLAG=""
   if [[ -n "${{ inputs.run_id }}" ]]; then
     REVISION_SUFFIX_FLAG="--revision-suffix=${{ inputs.run_id }}"
   fi
   ```
4. Append `$REVISION_SUFFIX_FLAG` to the existing `gcloud run deploy` command (no other flags change). When triggered by `push`, the variable is empty and Cloud Run picks the default revision suffix; when triggered by `workflow_dispatch` with a `run_id`, the revision is labelled `fish-id-<run_id>` for traceability.

Note: Cloud Run revision suffixes must match `[a-z][a-z0-9-]*` and be ≤ 63 chars. The chosen `run-YYYY-MM-DD-HHMMSS` format (24 chars, all lowercase + digits + hyphens) satisfies both.

No new build file or new build mechanism is introduced — the Docker build + Artifact Registry push path is identical to the existing automated deploy.

GitHub doesn't accept GCP-issued OIDC tokens for API auth, so a PAT (or GitHub App token) is the standard option here — there's no Workload Identity Federation alternative for inbound calls to the GitHub Actions API.

---

### Rollback / Manual Promotion

All run artifacts are retained permanently in GCS (`runs/run-{R}/` is never deleted), so any previous run can be promoted to production at any time.

`scripts/promote-run.py` performs the full promotion + redeploy sequence without training or evaluation, using the developer's local `gcloud` credentials:

```bash
python scripts/promote-run.py --run-id run-2026-05-24-093015
```

Steps:
1. Verify `gs://{PROJECT_ID}-fish-id-models/runs/run-2026-05-24-093015/fish-id.onnx` exists
2. Copy it to `gs://{PROJECT_ID}-fish-id-models/fish-id.onnx`
3. Overwrite `production-run.json` with `{ "run_id": "run-2026-05-24-093015", "promoted_at": "...", "manual_override": false }`
4. Trigger `workflow_dispatch` on `.github/workflows/deploy.yml` with `run_id=run-2026-05-24-093015` to rebuild and redeploy Cloud Run

The script prints a warning that quality gates are being bypassed before proceeding. No new IAM permissions are required — the script runs with developer credentials, the same as `trigger-training.py`.

---

### Manual model override (`scripts/deploy-app.sh`)

`scripts/deploy-app.sh` is the manual CLI deploy path. It bakes a local ONNX file into the container image and deploys to Cloud Run via `gcloud builds submit`. Because it bypasses the pipeline, the served model has no corresponding `runs/run-{R}/` directory — there is no training metadata, no eval results, and no provenance for it in GCS.

To keep `production-run.json` truthful (i.e. always reflecting what Cloud Run is actually serving), `deploy-app.sh` gains two new steps that run after the Cloud Run deploy succeeds:

1. **Upload the local ONNX to the canonical serving path:** `gsutil cp <local-fish-id.onnx> gs://{PROJECT_ID}-fish-id-models/fish-id.onnx`. This keeps the GCS root in sync with the deployed Cloud Run revision so that the next automated promotion (or rollback) starts from a consistent state.
2. **Overwrite `production-run.json`** with:
   ```json
   {
     "run_id": null,
     "manual_override": true,
     "promoted_at": "<UTC timestamp>",
     "source": "scripts/deploy-app.sh",
     "operator": "<gcloud config get-value account>",
     "local_onnx_path": "<absolute path passed to build.sh>"
   }
   ```

`deploy-app.sh` prints a warning before performing these uploads explaining that the production model is being set outside the run-tracking system.

**Effect on the quality gate**: when the next training run reaches Gate 2, the `quality_gate` pipeline component reads `production-run.json` and sees `run_id: null` / `manual_override: true`. Because there is no `runs/{run_id}/eval_results.json` to read for a baseline, Gate 2 is skipped for that single cycle (same behavior as the very first deploy when no production model exists). Gate 1 (absolute floor) still applies. After a successful auto-promotion, `production-run.json` is rewritten with `manual_override: false` and a real `run_id`, restoring normal regression-gate behavior on subsequent runs.

---

### New & Modified IAM

**New: `fish-id-training-sa`** (Vertex AI CustomJob containers):
- `roles/storage.objectAdmin` on `{PROJECT_ID}-fish-id-training` bucket (read training pool + manifests; write pipeline-state)
- `roles/storage.objectCreator` on `{PROJECT_ID}-fish-id-models` bucket (write run artifacts)
- `roles/aiplatform.user` on project (log metrics to Vertex AI Experiments)

**Modified: `fish-id-cicd-sa`** (existing CI/CD SA used by GitHub Actions):
- Add `roles/storage.objectCreator` on `{PROJECT_ID}-fish-id-models` bucket so `build-training-image.yml` can write `training-image-latest.json`. The existing `roles/storage.objectViewer` binding remains.

**New: `fish-id-workflows-sa`** (Cloud Functions v2 trigger + Vertex AI Pipeline components):
- `roles/aiplatform.user` on project (submit Vertex AI PipelineJobs and CustomJobs)
- `roles/storage.objectAdmin` on `{PROJECT_ID}-fish-id-models` bucket (read eval_results.json, production-run.json, and training-image-latest.json; write production-run.json and gate_failure.json; copy fish-id.onnx; pipeline root writes)
- `roles/secretmanager.secretAccessor` on `github-deploy-pat` secret (call GitHub API for `workflow_dispatch`)
- `roles/logging.logWriter` on project

---

### New Terraform Resources

| Resource | Type | Note |
|---|---|---|
| `{PROJECT_ID}-fish-id-training` bucket | `google_storage_bucket` | New |
| `fish-id-training-sa` | `google_service_account` | New |
| `fish-id-workflows-sa` | `google_service_account` | New |
| `fish-id-training-sa` bindings on training bucket (objectAdmin) | `google_storage_bucket_iam_member` | New |
| `fish-id-training-sa` bindings on models bucket (objectCreator) | `google_storage_bucket_iam_member` | New binding on existing bucket |
| `fish-id-training-sa` project binding (aiplatform.user) | `google_project_iam_member` | New |
| `fish-id-workflows-sa` binding on models bucket (objectAdmin) | `google_storage_bucket_iam_member` | New binding on existing bucket |
| `fish-id-workflows-sa` project bindings (aiplatform.user, logging.logWriter) | `google_project_iam_member` | New |
| `fish-id-workflows-sa` secret binding (secretmanager.secretAccessor) | `google_secret_manager_secret_iam_member` | New |
| `fish-id-cicd-sa` objectCreator binding on models bucket | `google_storage_bucket_iam_member` | New binding on existing SA |
| Pipeline trigger source zip (versioned by content hash) | `google_storage_bucket_object` | New |
| Cloud Functions v2 trigger (`fish-id-pipeline-trigger`) | `google_cloudfunctions2_function` | New (replaces `google_workflows_workflow` + `google_eventarc_trigger`) |
| Cloud Run invoker IAM for Eventarc → Cloud Functions | `google_cloud_run_v2_service_iam_member` | New |
| `github-deploy-pat` secret | `google_secret_manager_secret` | New |

**New APIs to enable in `apis.tf`**:
- `aiplatform.googleapis.com`
- `cloudfunctions.googleapis.com`
- `eventarc.googleapis.com`
- `secretmanager.googleapis.com`

---

### Pipeline Cost Controls

The existing "Cost Controls" table is for serving. The pipeline introduces a new cost surface (Vertex AI training jobs) that needs its own controls:

| Control | What it prevents |
|---|---|
| Vertex AI CustomJob `timeout: 7200s` (2h) | Runaway training cost from a misconfigured run or stuck job |
| `n1-highmem-4` CPU-only machine type | GPU billing on a dataset size that doesn't justify GPU |
| Cloud Functions trigger filters for `versions/*/manifest.json` internally | Pipeline firing on every individual image upload during a bulk sync |
| Run artifacts retained, not deleted | (cost-neutral guardrail) avoids re-running training to recover a model |

**Recommended out-of-band**: a project-level GCP budget alert at `$50/month` (or chosen threshold). Not enforced by code, but called out here so the operator sets it up — the auto-trigger means a malformed dataset upload could in principle cause repeated full-cost training runs, and the budget alert is the catch-all backstop. Configure via Billing → Budgets & alerts, scoped to the GCP project.

---

## Security

See **[docs/security.md](security.md)** for the full security reference, including GitHub repository security settings (secret scanning, Dependabot, CodeQL), CI security jobs, CI/CD authentication, runtime controls, and training pipeline IAM.

---

## Cost Controls

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
