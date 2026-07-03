# Fish Detector — CI/CD Reference

GitHub Actions handles all deploys on merge to `main`. Manual CLI deployment via `scripts/deploy-app.sh` remains available.

---

## Workflow: `.github/workflows/deploy-api.yml`

Runs on push to `main` when files under `app/**` change, **or** on `workflow_dispatch` (which the training pipeline fires automatically after each successful model registration): resolves the `production`-aliased version in Vertex AI Model Registry, downloads `fish-id.onnx` from its GCS artifact URI, builds the Docker image, pushes to Artifact Registry, and deploys to Cloud Run.

Authenticates using **Workload Identity Federation** — no long-lived service account keys are stored anywhere. GitHub Actions receives a short-lived OIDC token that is exchanged for GCP credentials scoped to the `fish-id-cicd-sa` service account.

The `workflow_dispatch` trigger accepts an optional `run_id` input; when provided by the training pipeline, it is used as a Cloud Run revision suffix for traceability.

---

## Workflow: `.github/workflows/deploy-frontend.yml`

Runs on push to `main` when files under `frontend/**` change, **or** on `workflow_dispatch`. Fetches the Cloud Run API URL, injects it into `frontend/public/js/app.js` and the project ID into `frontend/.firebaserc`, then deploys `frontend/` to Firebase Hosting. Authenticates via the same WIF / `fish-id-cicd-sa` pattern as `deploy-api`.

---

## Workflow: `.github/workflows/ci.yml`

Runs on every PR (opened, synchronised, reopened). Three parallel jobs:

1. **`test`** — installs deps for `app/`, `analytics-consumer/`, `training/`, `pipeline/`, `scripts/`, and `tests/`, runs `pytest tests/ -v`, runs `pylint app/ analytics-consumer/ training/ pipeline/ scripts/ tests/` (fail threshold 7.0), and runs `mypy` in two invocations (`app/main.py` and `analytics-consumer/main.py` share a filename, which mypy treats as a duplicate module across a single invocation)
2. **`dependency-scan`** — runs `pip-audit` against `app/requirements.txt` and `analytics-consumer/requirements.txt` to check for known CVEs
3. **`secret-scan`** — runs Gitleaks over the full git history to detect committed secrets

PRs cannot merge if any job fails.

---

## Workflow: `.github/workflows/deploy-analytics-consumer.yml`

Runs on push to `main` when files under `analytics-consumer/**` change, plus `workflow_dispatch` for ad-hoc redeploys. Authenticates via the same WIF / `fish-id-cicd-sa` pattern as `deploy-api`.

Builds `analytics-consumer/Dockerfile`, pushes to the same `fish-id` Artifact Registry repo under the `analytics-consumer` image name, and deploys to Cloud Run as `fish-id-analytics-consumer` with `--max-instances 1` and `--no-allow-unauthenticated` (the only caller is the Pub/Sub push subscription's OIDC identity — see `terraform/pubsub.tf`).

The very first deploy of this service is also the manual step needed to unblock the Pub/Sub push subscription's Terraform bootstrap (see [docs/architecture-context.md](architecture-context.md)).

---

## Workflow: `.github/workflows/train-pipeline.yml`

Runs on push to `main` when files under `training/**` or `pipeline/**` change, plus `workflow_dispatch` for ad-hoc reruns. Authenticates via the same WIF / `fish-id-cicd-sa` pattern. Runs use a `concurrency` group so back-to-back pushes queue rather than firing concurrent (expensive, GPU) training runs. Three sequential jobs:

1. **`build-and-push`** — builds `training/Dockerfile` (build context: repo root) tagged `{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:{SHA}`, pushes both `:{SHA}` and `:latest` tags to Artifact Registry
2. **`compile-and-upload`** — compiles `pipeline/pipeline.py` and uploads the compiled JSON to `gs://{PROJECT_ID}-fish-id-models/pipeline/fish-id-training-pipeline.json` — this is the template URI that `scripts/trigger-training.py` passes when submitting a PipelineJob
3. **`trigger-training`** — submits a Vertex AI PipelineJob via `scripts/trigger-training.py`

Merging any change to `training/` or `pipeline/` automatically keeps the image, the compiled pipeline template, and a triggered training run all in sync — no manual step needed.

---

## GitHub Secrets Required

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | Cloud Run / Artifact Registry region |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output from `terraform output workload_identity_provider` |
| `GCP_SERVICE_ACCOUNT` | Output from `terraform output cicd_service_account_email` |

`ONNX_MODEL_GCS_URI` is no longer used and can be removed from GitHub repo settings — the model is now resolved dynamically from Vertex AI Model Registry at deploy time.
