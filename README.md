# fish-id

Simple web app to identify fresh water fish species (e.g., Largemouth Bass) in images.

Hosted on GCP: Cloud Run API + Firebase Hosting frontend.

---

## CLI Tools

### Required

| Tool | Purpose |
|------|---------|
| [`gcloud`](https://cloud.google.com/sdk/docs/install) | Authenticate to GCP, build/push container via Cloud Build, deploy to Cloud Run |
| [`firebase`](https://firebase.google.com/docs/cli) | Deploy the static frontend to Firebase Hosting (`npm install -g firebase-tools`) |
| [`terraform`](https://developer.hashicorp.com/terraform/install) | Provision all GCP infrastructure (APIs, service accounts, Workload Identity, Artifact Registry, GCS bucket) |

### Optional

| Tool | Purpose |
|------|---------|
| `gsutil` | Upload `fish-id.onnx` to GCS for the GitHub Actions CI/CD path (bundled with `gcloud`) |
| [`gh`](https://cli.github.com) | Manage GitHub Actions secrets and open PRs from the CLI |

---

## Prerequisites

- `gcloud`, `firebase`, and `terraform` installed and authenticated (see [CLI Tools](#cli-tools) above)
- A trained `fish-id.onnx` model file

---

## App Setup via CLI (Manual Alternative to Terraform)

### 1. Enable required APIs

```bash
gcloud services enable \
  firebasehosting.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
```

### 2. Create the Artifact Registry repository

```bash
gcloud artifacts repositories create fish-id \
  --repository-format=docker \
  --location=${GCP_REGION} \
  --project=${GCP_PROJECT_ID}
```

### 3. Create a service account for Cloud Run

The service account is granted no roles — the app makes no GCP API calls.

```bash
gcloud iam service-accounts create fish-id-cloud-run-sa \
  --display-name="fish-id Cloud Run SA"
```

---

## App Deployment via CLI

### API (Cloud Run)

`scripts/deploy-app.sh` handles the full build and deploy in one step. It copies `fish-id.onnx` into `app/` for the build, submits to Cloud Build, pushes the image to Artifact Registry, deploys to Cloud Run, then cleans up the local model copy.

```bash
scripts/deploy-app.sh /path/to/fish-id.onnx ${GCP_PROJECT_ID} [${GCP_REGION}]
```

### Frontend (Firebase Hosting)

Create the Firebase Hosting site (one-time):

```bash
firebase hosting:sites:create ${YOUR_SITE_NAME}
```

Then deploy:

```bash
cd frontend/

# Set projects.default in .firebaserc to ${GCP_PROJECT_ID}
# Add "site": "${YOUR_SITE_NAME}" under "hosting" in firebase.json
# In public/js/app.js, replace YOUR_CLOUD_RUN_URL in API_BASE with your Cloud Run URL

firebase deploy --only hosting
```

---

## Infrastructure Setup

Terraform provisions all GCP infrastructure required by both the app and the continuous training pipeline: APIs, service accounts, IAM bindings, Workload Identity Federation, Artifact Registry, GCS buckets, Cloud Workflows, Eventarc trigger, and Secret Manager.

### 0. Authenticate

Terraform uses Application Default Credentials. Before running any Terraform commands, authenticate with your GCP account:

```bash
gcloud auth application-default login
```

If you're in a headless environment: `gcloud auth application-default login --no-launch-browser`

### 1. Apply

```bash
cd terraform/
terraform init
terraform plan -var="project_id=${GCP_PROJECT_ID}" -var="github_repo=${YOUR_GITHUB_ORG}/fish-id"
terraform apply -var="project_id=${GCP_PROJECT_ID}" -var="github_repo=${YOUR_GITHUB_ORG}/fish-id"
```

Region defaults to `us-central1`. To override:

```bash
terraform apply \
  -var="project_id=${GCP_PROJECT_ID}" \
  -var="region=${GCP_REGION}" \
  -var="github_repo=${YOUR_GITHUB_ORG}/fish-id"
```

### 2. Capture outputs

After apply, retrieve the values needed for GitHub secrets:

```bash
terraform output workload_identity_provider
terraform output cicd_service_account_email
terraform output model_bucket_name
```

### 3. Store the GitHub PAT in Secret Manager

The continuous training pipeline triggers a Cloud Run redeploy via the GitHub Actions API after promoting a new model. It authenticates using a GitHub Personal Access Token stored in Secret Manager.

**Create the PAT on GitHub:**
1. Go to **Settings → Developer settings → Personal access tokens → Fine-grained tokens**
2. Click **Generate new token**
3. Set a name (e.g. `fish-id-workflow-dispatch`) and an expiration
4. Set **Resource owner** to your account and **Repository access** to **Only select repositories** → `briancorwin/fish-id`
5. Under **Permissions → Repository permissions**, find **Actions** and set it to **Read and write** — this grants the ability to trigger workflow runs. Leave all other resources at No access
6. Click **Generate token** and copy it immediately — it is only shown once

**Store it in Secret Manager:**

```bash
echo -n "ghp_..." | gcloud secrets versions add github-deploy-pat \
  --data-file=- \
  --project=${GCP_PROJECT_ID}
```

The `-n` flag is required — omitting it stores a trailing newline in the secret, which causes GitHub API auth to fail.

**Verify:**

```bash
gcloud secrets versions list github-deploy-pat --project=${GCP_PROJECT_ID}
```

One version in state `ENABLED` confirms it is ready.

---

## App Deployment via GitHub Actions

Requires [Infrastructure Setup](#infrastructure-setup) to be completed first.

On every PR merged to `main`, two jobs run in sequence: `deploy-api` (Cloud Run) then `deploy-frontend` (Firebase Hosting). Both authenticate to GCP using Workload Identity Federation — no long-lived credentials required.

### 1. Upload the initial model to GCS

The deploy workflow downloads `fish-id.onnx` from GCS at deploy time. Upload a starting model before the first deploy:

```bash
gsutil cp /path/to/fish-id.onnx gs://$(terraform -chdir=terraform output -raw model_bucket_name)/fish-id.onnx
```

Once the continuous training pipeline is running, promotions overwrite this path automatically.

### 2. Set GitHub Actions secrets

Navigate to **Settings → Secrets and variables → Actions** in the GitHub repo and add:

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_REGION` | Region used during Terraform apply (e.g. `us-central1`) |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output of `terraform output workload_identity_provider` |
| `GCP_SERVICE_ACCOUNT` | Output of `terraform output cicd_service_account_email` |
| `ONNX_MODEL_GCS_URI` | `gs://${BUCKET_NAME}/fish-id.onnx` (bucket from `terraform output model_bucket_name`) |

### 3. Deploy

Merge a PR to `main`. The workflow builds and pushes the container image to Artifact Registry, deploys to Cloud Run, injects the Cloud Run URL into the frontend, and deploys to Firebase Hosting.

---

## Continuous Training Pipeline

Requires [Infrastructure Setup](#infrastructure-setup) to be completed first.

The pipeline runs on GCP — no manual steps are needed for individual training runs once it is bootstrapped. The flow is:

```
New dataset manifest uploaded to GCS
  → Eventarc trigger → Cloud Workflows execution
    → Vertex AI training job → Vertex AI eval job
      → Quality gates → Model promotion → Cloud Run redeploy
```

### 1. Build the training container

The training container is built automatically by `.github/workflows/build-training-image.yml` whenever changes are merged to `main` under `training/**`. It pushes the image to Artifact Registry and writes `training-image-latest.json` to the models bucket, which the pipeline reads to know which container to use.

Verify it ran and wrote the file after your first merge:

```bash
gsutil cat gs://$(terraform -chdir=terraform output -raw model_bucket_name)/training-image-latest.json
```

### 2. Bootstrap the eval dataset

The eval dataset is versioned separately from training data and only changes by deliberate human action. Set it up once before the first training run:

```bash
# Upload eval images and labels to the pool
gsutil -m cp -r /path/to/eval/images/ gs://${GCP_PROJECT_ID}-fish-id-training/eval/images/
gsutil -m cp -r /path/to/eval/labels/ gs://${GCP_PROJECT_ID}-fish-id-training/eval/labels/

# Write the eval manifest (listing filenames in the eval pool)
gsutil cp eval-manifest-v1.json gs://${GCP_PROJECT_ID}-fish-id-training/eval/versions/v1/manifest.json

# Point current.json at v1
echo '{"eval_version":"v1"}' | gsutil cp - gs://${GCP_PROJECT_ID}-fish-id-training/eval/current.json
```

### 3. Upload the initial training dataset

Run `scripts/update-dataset.py` to export from Roboflow, sync to the GCS training pool, and write the dataset manifest. Writing the manifest finalizes the dataset version and automatically triggers the pipeline via Eventarc:

```bash
export ROBOFLOW_API_KEY=your_key_here

python scripts/update-dataset.py \
  --roboflow-version 1 \
  --dataset-version v1 \
  --bucket ${GCP_PROJECT_ID}-fish-id-training \
  --workspace your-roboflow-workspace \
  --project fish-id \
  --description "Initial dataset"
```

Watch the pipeline execute in the Cloud Console under **Workflows → fish-id-training-pipeline**.

### 4. Ongoing: adding new training data

Run `scripts/update-dataset.py` with an incremented `--dataset-version` whenever a new labeled dataset version is ready in Roboflow. The pipeline triggers automatically each time. `ROBOFLOW_API_KEY` must be set in your environment.

### Manual pipeline trigger (no new data)

To re-run training on existing data — e.g. to test a new config — without uploading a new dataset:

```bash
python scripts/trigger-training.py --dataset-version v1 --config-version c1
```

---

## Local Development

All commands below are run from the repo root.

### First-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt -r tests/requirements.txt
```

### Running the API

Place `fish-id.onnx` in `app/` before starting — the app loads it at startup and will crash without it.

```bash
source .venv/bin/activate
python3 app/main.py
```

The API is available at `http://localhost:8080`.

```bash
# Health check
curl http://localhost:8080/health

# Run detection on an image
curl -X POST http://localhost:8080/detect \
  -F "image=@/path/to/fish.jpg;type=image/jpeg"
```

### Running the frontend

```bash
python3 -m http.server 3000 --directory frontend/public/
```

Open `http://localhost:3000`. The API defaults `CORS_ORIGIN` to `*` when the env var is unset, so requests to the local API at `http://localhost:8080` work without changes.

### Running tests

```bash
source .venv/bin/activate
python3 -m pytest tests/ -v
```

---

## Scripts

All scripts in `scripts/` are run locally with your `gcloud` credentials. None require service account keys.

### Setup

The scripts have their own dependencies separate from the app. Create a virtualenv once:

```bash
python3 -m venv scripts/.venv
source scripts/.venv/bin/activate
pip install -r scripts/requirements.txt
```

Activate it before running any script:

```bash
source scripts/.venv/bin/activate
```

| Script | Purpose |
|---|---|
| `deploy-app.sh` | Manual CLI deploy. Bakes a local `fish-id.onnx` into the app container, builds and deploys to Cloud Run, and updates `production-run.json` with `manual_override: true`. Use for quick one-off deploys or testing a model outside the training pipeline. |
| `update-dataset.py` | Exports a dataset version from Roboflow, syncs images and labels to the GCS training pool, and writes `versions/vN/manifest.json`. Writing the manifest triggers the Eventarc → Cloud Workflows training pipeline automatically. Requires `ROBOFLOW_API_KEY` env var. |
| `trigger-training.py` | Manually fires the training pipeline for a specific dataset version and config version without uploading new data. Calls `gcloud workflows run` directly. Use to re-run training on existing data or test a new config. |
| `promote-run.py` | Promotes any previous run to production — copies its `fish-id.onnx` to the production path, updates `production-run.json`, and triggers a Cloud Run redeploy. Bypasses quality gates. Use for rollback or manual promotion. |
| `rebaseline-production.py` | Re-scores the current production model against the current eval set. Run this once immediately after updating `eval/current.json` to a new eval version, before any new training run, to keep Gate 2 regression comparisons valid. |
