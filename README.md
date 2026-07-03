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

## Web App Setup via CLI (Manual Alternative to Terraform)

### 0. Prerequisites

- `gcloud`, `firebase`, and `terraform` installed and authenticated (see [CLI Tools](#cli-tools) above)
- A trained `fish-id.onnx` model file

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

The service account is granted `roles/pubsub.publisher` on the analytics topic (see `terraform/pubsub.tf`) — the app's only GCP API call is publishing detection events.

```bash
gcloud iam service-accounts create fish-id-cloud-run-sa \
  --display-name="fish-id Cloud Run SA"
```

---

## Web App Deployment via CLI

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

Terraform provisions all GCP infrastructure: APIs, service accounts, IAM bindings, Workload Identity Federation, Artifact Registry, GCS buckets, and Vertex AI resources.

### 0. Authenticate

Terraform uses Application Default Credentials. Before running any Terraform commands:

```bash
gcloud auth application-default login
```

If you're in a headless environment: `gcloud auth application-default login --no-launch-browser`

### 1. Create the Terraform state bucket

Terraform stores its state in GCS, but the state bucket must exist before `terraform init` can use it. Create it once manually:

```bash
gcloud storage buckets create gs://${GCP_PROJECT_ID}-fish-id-terraform-state \
  --location=${GCP_REGION} \
  --uniform-bucket-level-access \
  --project=${GCP_PROJECT_ID}
```

### 2. Apply

`${GITHUB_REPO}` is the full `owner/name` form, e.g. `briancorwin/fish-id`. Region defaults to `us-central1` if `-var="region=..."` is omitted.

```bash
cd terraform/
terraform init \
  -backend-config="bucket=${GCP_PROJECT_ID}-fish-id-terraform-state"
terraform apply \
  -var="project_id=${GCP_PROJECT_ID}" \
  -var="region=${GCP_REGION}" \
  -var="github_repo=${GITHUB_REPO}"
```

After apply, import the state bucket so Terraform manages it going forward:

```bash
terraform import \
  -var="project_id=${GCP_PROJECT_ID}" \
  -var="region=${GCP_REGION}" \
  -var="github_repo=${GITHUB_REPO}" \
  google_storage_bucket.terraform_state \
  ${GCP_PROJECT_ID}-fish-id-terraform-state
```

### 3. Capture outputs

Retrieve the values needed for GitHub secrets:

```bash
terraform output workload_identity_provider
terraform output cicd_service_account_email
terraform output model_bucket_name
```

### 4. Populate the GitHub deploy token secret

Terraform creates the Secret Manager secret shell but does not populate it. After `terraform apply`, seed the GitHub PAT that the training pipeline uses to trigger deploys (requires a PAT with `repo` scope or `actions:write`):

```bash
echo -n "YOUR_GITHUB_PAT" | gcloud secrets versions add fish-id-github-deploy-token \
  --data-file=- \
  --project=${GCP_PROJECT_ID}
```

### 5. Finish the analytics bootstrap (two-phase apply)

The Pub/Sub push subscription in `terraform/pubsub.tf` needs the `analytics-consumer` Cloud Run service's URL, which doesn't exist until that service is deployed once — so the first `terraform apply` above (step 2) intentionally skips creating the subscription (`analytics_consumer_url` defaults to `""`).

1. Deploy `analytics-consumer` once — merge a PR touching `analytics-consumer/**` (triggers `.github/workflows/deploy-analytics-consumer.yml`), or run it manually the same way as `scripts/deploy-app.sh`.
2. Get its URL:
   ```bash
   gcloud run services describe fish-id-analytics-consumer \
     --region=${GCP_REGION} --project=${GCP_PROJECT_ID} \
     --format='value(status.url)'
   ```
3. Re-apply with that URL to create the push subscription and its Cloud Run invoker binding:
   ```bash
   terraform apply \
     -var="project_id=${GCP_PROJECT_ID}" \
     -var="region=${GCP_REGION}" \
     -var="github_repo=${GITHUB_REPO}" \
     -var="analytics_consumer_url=<url from step 2>"
   ```

---

## Web App Deployment via GitHub Actions

Requires [Infrastructure Setup](#infrastructure-setup) to be completed first.

On every PR merged to `main`, two jobs run in sequence: `deploy-api` (Cloud Run) then `deploy-frontend` (Firebase Hosting). Both authenticate to GCP using Workload Identity Federation — no long-lived credentials required.

### 1. Seed the initial model in Vertex AI Model Registry

The deploy workflow resolves `fish-id.onnx` from the `production` alias in Vertex AI Model Registry. The training pipeline populates this automatically after the first successful run. For an initial deploy before any training run, register the model manually:

```bash
gcloud ai models upload \
  --display-name=fish-id \
  --artifact-uri=gs://${GCP_PROJECT_ID}-fish-id-models/runs/<run-id>/ \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/onnx-cpu.1-14:latest \
  --version-aliases=latest,production \
  --region=${GCP_REGION} \
  --project=${GCP_PROJECT_ID}
```

### 2. Set GitHub Actions secrets

Navigate to **Settings → Secrets and variables → Actions** in the GitHub repo and add:

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_REGION` | Region used during Terraform apply (e.g. `us-central1`) |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output of `terraform output workload_identity_provider` |
| `GCP_SERVICE_ACCOUNT` | Output of `terraform output cicd_service_account_email` |

### 3. Deploy

Merge a PR to `main`. The workflow builds and pushes the container image to Artifact Registry, deploys to Cloud Run, injects the Cloud Run URL into the frontend, and deploys to Firebase Hosting.

---

## Training Pipeline

Requires [Infrastructure Setup](#infrastructure-setup) to be completed first.

### 1. Build the training container

The training container is built automatically by `.github/workflows/train-pipeline.yml` whenever changes to `training/**` or `pipeline/**` are merged to `main`. It pushes both `:{SHA}` and `:latest` tags to Artifact Registry, compiles the pipeline definition and uploads it to GCS, then submits a Vertex AI training run — no manual step needed.

Verify the build ran after your first merge:

```bash
gcloud artifacts docker images list \
  ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/fish-id-train \
  --filter="tags:latest"
```

### 2. Upload training data

Export from Roboflow and sync to the GCS training bucket (always `${GCP_PROJECT_ID}-fish-id-training`). A successful sync automatically submits a training run against the new data, so also set the variables `scripts/trigger-training.py` needs (`GCP_PROJECT_ID`, `GCP_REGION`, `GITHUB_REPO`):

```bash
export ROBOFLOW_API_KEY=your-roboflow-api-key-here
export ROBOFLOW_WORKSPACE=your-roboflow-workspace-here
export ROBOFLOW_PROJECT=your-roboflow-project-here
export GCP_PROJECT_ID=your-project-id-here
export GCP_REGION=us-central1
export GITHUB_REPO=owner/repo-name   # e.g. briancorwin/fish-id

python scripts/update-dataset.py --roboflow-version ${ROBOFLOW_VERSION_NUMBER}
```

Pass `--no-trigger-training` to sync without submitting a training run (e.g. to sync several dataset versions before training on the final one).

### 3. Trigger a training run manually (optional)

Training is triggered automatically after step 1 (container/config/pipeline changes) and step 2 (dataset syncs). For an ad-hoc run — e.g. `--cpu-only`, or resuming a specific `--run-id` — call the same script directly:

```bash
export GCP_PROJECT_ID=your-project-id-here
export GCP_REGION=us-central1
export GITHUB_REPO=owner/repo-name   # e.g. briancorwin/fish-id

python scripts/trigger-training.py
```

Use `--cpu-only` to skip the GPU accelerator.

The run appears in the Cloud Console under **Vertex AI → Pipelines → Runs**. After training, the pipeline evaluates the model against a held-out eval set; if it clears the quality gate against the current production model, it's registered in Vertex AI Model Registry with a `production` alias and `deploy.yml` is triggered automatically to redeploy Cloud Run with the new model. See [docs/training-pipeline.md](docs/training-pipeline.md) for the full flow.

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

The API is available at `http://localhost:8080`. `GCP_PROJECT_ID` is unset by default in local dev, so the analytics event publish (see [docs/web-app.md](docs/web-app.md#analytics-appanalyticspy-analytics-consumer)) silently no-ops — no GCP credentials are needed to run locally.

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
| `deploy-app.sh` | Manual CLI deploy. Bakes a local `fish-id.onnx` into the app container, builds via Cloud Build, and deploys to Cloud Run. Use to deploy a model retrieved from the training pipeline. |
| `update-dataset.py` | Exports a Roboflow dataset version in YOLO format and syncs images and labels to the GCS training bucket, then calls `trigger-training.py` to submit a training run (`--no-trigger-training` to skip). Requires `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT` plus everything `trigger-training.py` requires. |
| `trigger-training.py` | Submits a Vertex AI PipelineJob using the compiled pipeline template and training image from GCS. Requires `GCP_PROJECT_ID`, `GCP_REGION`, and `GITHUB_REPO` env vars. |
