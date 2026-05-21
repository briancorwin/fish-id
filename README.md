# fish-id

Simple web app to identify fresh water fish species (e.g., Largemouth Bass) in images.

Hosted on GCP: Cloud Run API + Firebase Hosting frontend.

---

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) — authenticated and pointed at your project
- [Firebase CLI](https://firebase.google.com/docs/cli) — `npm install -g firebase-tools && firebase login`
- A trained `best.onnx` model file

---

## Initial Setup via CLI

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

## Deployment via CLI

### API (Cloud Run)

`scripts/build.sh` takes the path to your `best.onnx`, your GCP project ID, and an optional region (default: `us-central1`). It copies the model into `app/` for the build, then removes it.

```bash
scripts/build.sh /path/to/best.onnx ${GCP_PROJECT_ID} [${GCP_REGION}]
```

This submits the build to Cloud Build, which builds the image and pushes it to Artifact Registry:

```bash
gcloud builds submit app/ \
  --tag ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/fish-id/fish-id \
  --project ${GCP_PROJECT_ID}
```

Then deploy to Cloud Run:

```bash
gcloud run deploy fish-id \
  --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/fish-id/fish-id \
  --region ${GCP_REGION} \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 5 \
  --max-instances 1 \
  --service-account fish-id-cloud-run-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --set-env-vars CORS_ORIGIN=https://${GCP_PROJECT_ID}.web.app \
  --allow-unauthenticated
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

## Initial Setup via Terraform

Terraform provisions all required GCP infrastructure: APIs, service accounts, Workload Identity Federation, Artifact Registry, and the GCS bucket for model storage.

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

---

## Deployment via GitHub Actions

Requires [Initial Setup via Terraform](#initial-setup-via-terraform) to be completed first — it creates the GCS bucket for model storage, the CI/CD service account, and the Workload Identity Federation configuration that GitHub Actions authenticates with.

On every PR merged to `main`, two jobs run in sequence: `deploy-api` (Cloud Run) then `deploy-frontend` (Firebase Hosting). Both authenticate to GCP using Workload Identity Federation — no long-lived credentials required.

### 1. Upload the model to GCS

```bash
gsutil cp /path/to/best.onnx gs://$(terraform -chdir=terraform output -raw model_bucket_name)/best.onnx
```

### 2. Set GitHub Actions secrets

Navigate to **Settings → Secrets and variables → Actions** in the GitHub repo and add:

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_REGION` | Region used during Terraform apply (e.g. `us-central1`) |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output of `terraform output workload_identity_provider` |
| `GCP_SERVICE_ACCOUNT` | Output of `terraform output cicd_service_account_email` |
| `ONNX_MODEL_GCS_URI` | `gs://${BUCKET_NAME}/best.onnx` (bucket from `terraform output model_bucket_name`) |

### 3. Deploy

Merge a PR to `main`. The Actions workflow will build and push the container image to Artifact Registry, deploy to Cloud Run, inject the Cloud Run URL into the frontend, and deploy to Firebase Hosting automatically.

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

Place `best.onnx` in `app/` before starting — the app loads it at startup and will crash without it.

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
