# fish-id

Simple web app to identify fresh water fish species (e.g., Largemouth Bass) in pictures.

Hosted on GCP: Cloud Run API + Firebase Hosting frontend.

---

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) — authenticated and pointed at your project
- [Firebase CLI](https://firebase.google.com/docs/cli) — `npm install -g firebase-tools && firebase login`
- Docker (for local container builds)
- A trained `best.onnx` model file

---

## Setup

### 1. Enable required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com
```

### 2. Create a service account for Cloud Run

The service account is granted no roles — the app makes no GCP API calls.

```bash
gcloud iam service-accounts create fish-id-sa \
  --display-name="fish-id Cloud Run SA"
```

### 3. Place your model file

Copy `best.onnx` into `app/` before building. It is not committed to the repo.

```bash
cp /path/to/best.onnx app/best.onnx
```

---

## Deploy

### API (Cloud Run)

```bash
gcloud builds submit app/ --tag gcr.io/YOUR_PROJECT/fish-id

gcloud run deploy fish-id \
  --image gcr.io/YOUR_PROJECT/fish-id \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 5 \
  --max-instances 1 \
  --service-account fish-id-sa@YOUR_PROJECT.iam.gserviceaccount.com \
  --allow-unauthenticated
```

### Frontend (Firebase Hosting)

```bash
cd frontend/
# Update API_BASE in public/js/app.js with your Cloud Run URL
firebase deploy --only hosting
```

---

## Local Development

```bash
cd app/
pip install -r requirements.txt
flask run --port 8080
```
