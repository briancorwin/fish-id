# fish-id

Simple web app to identify fresh water fish species (e.g., Largemouth Bass) in images.

Hosted on GCP: Cloud Run API + Firebase Hosting frontend.

---

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) — authenticated and pointed at your project
- [Firebase CLI](https://firebase.google.com/docs/cli) — `npm install -g firebase-tools && firebase login`
- Docker (for local container builds)
- A trained `best.onnx` model file

---

## Initial Setup

### 1. Enable required APIs

```bash
gcloud services enable \
  firebasehosting.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com
```

### 2. Create a service account for Cloud Run

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
scripts/build.sh /path/to/best.onnx GCP_PROJECT_ID [GCP_REGION]
```

This builds and pushes to Artifact Registry:

```bash
gcloud auth configure-docker GCP_REGION-docker.pkg.dev
docker build -t GCP_REGION-docker.pkg.dev/GCP_PROJECT_ID/fish-id/fish-id app/
docker push GCP_REGION-docker.pkg.dev/GCP_PROJECT_ID/fish-id/fish-id
```

Then deploy to Cloud Run:

```bash
gcloud run deploy fish-id \
  --image GCP_REGION-docker.pkg.dev/GCP_PROJECT_ID/fish-id/fish-id \
  --region GCP_REGION \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 5 \
  --max-instances 1 \
  --service-account fish-id-cloud-run-sa@GCP_PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars CORS_ORIGIN=https://GCP_PROJECT_ID.web.app \
  --allow-unauthenticated
```

### Frontend (Firebase Hosting)

Create the Firebase Hosting site (one-time):

```bash
firebase hosting:sites:create YOUR_SITE_NAME
```

Then deploy:

```bash
cd frontend/

# Set projects.default in .firebaserc to GCP_PROJECT_ID
# Add "site": "YOUR_SITE_NAME" under "hosting" in firebase.json
# In public/js/app.js, replace YOUR_CLOUD_RUN_URL in API_BASE with your Cloud Run URL

firebase deploy --only hosting
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
