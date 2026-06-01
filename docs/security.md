# Fish Detector — Security Reference

---

## GitHub Repository Security

### Secret Scanning

GitHub secret scanning should be enabled. It scans every push for known credential patterns (API keys, tokens, service account keys) and alerts immediately on a match. For public repos it is on by default. Verify at **Settings → Code security → Secret scanning**.

### Dependabot

Two separate Dependabot features should both be active:

- **Dependabot alerts** — monitors installed dependencies for newly published CVEs and alerts when one is found
- **Dependabot security updates** — automatically opens PRs to bump a vulnerable dependency to a patched version

Enable both at **Settings → Code security → Dependabot**.

A third feature, **Dependabot version updates**, is configured in `.github/dependabot.yml` to scan `app/requirements.txt` weekly and open PRs for outdated packages regardless of CVE status. This is separate from security updates.

### CodeQL

CodeQL should be enabled and runs on every PR. It performs full dataflow analysis — tracing tainted user input through the call graph to detect SQL injection, command injection, path traversal, and similar classes of vulnerability. Pattern-matching tools (e.g. Bandit) cannot match this coverage. Configure at **Settings → Code security → Code scanning**.

---

## CI Security (`.github/workflows/ci.yml`)

Three jobs run on every PR in parallel with tests and lint:

### `dependency-scan`

Runs `pip-audit` against `app/requirements.txt`. Blocks merges that introduce or include dependencies with known CVEs. Complements Dependabot (which monitors the repo on a schedule) by providing a synchronous gate at PR time.

### `secret-scan`

Runs Gitleaks against the full commit history of the PR (`fetch-depth: 0`). Catches secrets accidentally committed in any commit in the branch, not just the tip.

Both jobs are scoped to `permissions: contents: read` — no broader access is granted.

---

## CI/CD Security (`.github/workflows/deploy.yml`)

### Workload Identity Federation

Both `deploy-api` and `deploy-frontend` jobs authenticate to GCP using **Workload Identity Federation (WIF)** — no long-lived service account keys are stored anywhere. GitHub Actions receives a short-lived OIDC token that is exchanged for GCP credentials scoped to the `fish-id-cicd-sa` service account.

WIF is configured to accept tokens only from this specific repository and only from the `main` branch. Feature branches cannot impersonate the CI/CD service account.

### GitHub Secrets

The following secrets are stored in GitHub and injected as environment variables at deploy time. They contain no credentials — only resource identifiers.

| Secret | Purpose |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | Cloud Run / Artifact Registry region |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | WIF provider resource name (output of `terraform output`) |
| `GCP_SERVICE_ACCOUNT` | CI/CD service account email (output of `terraform output`) |
| `ONNX_MODEL_GCS_URI` | GCS path to the production model artifact |

---

## Runtime Security

### Cloud Run Service Account

The Cloud Run service account is assigned **no IAM roles**. The Flask API makes no GCP API calls, so no permissions are needed. This limits blast radius if the application is ever exploited.

### Image Validation

Magic bytes are checked server-side to reject non-image payloads regardless of the `Content-Type` header. This prevents polyglot file attacks and ensures the ONNX inference pipeline only receives valid image data.

### CORS

Restricted to the Firebase Hosting origin (`https://PROJECT_ID.web.app`). The origin is injected via the `CORS_ORIGIN` Cloud Run environment variable at deploy time — not hardcoded.

### Rate Limiting

Per-IP token bucket (`app/rate_limiter.py`): 5 requests/minute per IP, burst of 3. Prevents a single caller from running up inference costs or abusing the API.

---

## Training Pipeline Security

### IAM Least Privilege

Each service account in the training pipeline is scoped to only the permissions it requires:

| Service Account | Role | Scope |
|---|---|---|
| `fish-id-training-sa` (Vertex AI job containers) | `roles/storage.objectAdmin` | Training bucket |
| | `roles/storage.objectCreator` | Models bucket |
| | `roles/aiplatform.user` | Project |
| `fish-id-workflows-sa` (Cloud Functions v2 trigger / Vertex AI Pipeline) | `roles/storage.objectAdmin` | Models bucket |
| | `roles/secretmanager.secretAccessor` | `github-deploy-pat` secret only |
| | `roles/aiplatform.user` + `roles/logging.logWriter` | Project |
| `fish-id-cicd-sa` (GitHub Actions) | `roles/storage.objectCreator` | Models bucket (for `training-image-latest.json`) |

### Secret Manager

The GitHub PAT (with `workflow` scope) used to trigger `workflow_dispatch` after model promotion is stored in **Secret Manager** (`github-deploy-pat`) and read by Cloud Workflows at execution time via the Secret Manager API. It is never written to source code, environment variables, or GitHub secrets.

### Roboflow API Key

The Roboflow API key used by `scripts/update-dataset.py` is read from the `ROBOFLOW_API_KEY` environment variable, stored in a local `.env` file. `.env` is listed in `.gitignore` and is never committed.
