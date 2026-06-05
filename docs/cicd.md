# Fish Detector — CI/CD Reference

GitHub Actions handles all deploys on merge to `main`. Manual CLI deployment via `scripts/deploy-app.sh` remains available.

---

## Workflow: `.github/workflows/deploy.yml`

Two jobs run sequentially on every merged PR:

1. **`deploy-api`** — downloads `fish-id.onnx` from GCS into `app/`, builds the Docker image, pushes to Artifact Registry, deploys to Cloud Run
2. **`deploy-frontend`** — fetches the Cloud Run URL, injects it into `app.js`, deploys `frontend/` to Firebase Hosting

Both jobs authenticate using **Workload Identity Federation** — no long-lived service account keys are stored anywhere. GitHub Actions receives a short-lived OIDC token that is exchanged for GCP credentials scoped to the `fish-id-cicd-sa` service account.

---

## Workflow: `.github/workflows/ci.yml`

Runs on every PR (opened, synchronised, reopened). Three parallel jobs:

1. **`test`** — installs `app/` and `tests/` deps, runs `pytest tests/ -v`, runs `pylint app/main.py app/rate_limiter.py` (fail threshold 7.0)
2. **`dependency-scan`** — runs `pip-audit -r app/requirements.txt` to check for known CVEs
3. **`secret-scan`** — runs Gitleaks over the full git history to detect committed secrets

PRs cannot merge if any job fails.

---

## Workflow: `.github/workflows/build-training-image.yml`

Runs on push to `main` when files under `training/**` or `pipeline/**` change, plus `workflow_dispatch` for ad-hoc rebuilds. Authenticates via the same WIF / `fish-id-cicd-sa` pattern.

Steps:
1. Build `training/Dockerfile` (build context: repo root) tagged `{REGION}-docker.pkg.dev/{PROJECT_ID}/fish-id/fish-id-train:{SHA}`
2. Push both `:{SHA}` and `:latest` tags to Artifact Registry
3. Compile `pipeline/pipeline.py` and upload the compiled JSON to `gs://{PROJECT_ID}-fish-id-models/pipeline/fish-id-training-pipeline.json` — this is the template URI that `scripts/trigger-training.py` passes when submitting a PipelineJob

The compiled pipeline template must be current before a training run is triggered. Merging any change to `training/` or `pipeline/` automatically keeps both in sync.

---

## GitHub Secrets Required

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | Cloud Run / Artifact Registry region |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output from `terraform output workload_identity_provider` |
| `GCP_SERVICE_ACCOUNT` | Output from `terraform output cicd_service_account_email` |
| `ONNX_MODEL_GCS_URI` | `gs://${GCP_PROJECT_ID}-fish-id-models/fish-id.onnx` |
