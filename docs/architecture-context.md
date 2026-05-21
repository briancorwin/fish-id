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
├── terraform/                  # GCP infrastructure-as-code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── apis.tf
│   ├── iam.tf
│   ├── storage.tf
│   └── artifact_registry.tf
├── .github/
│   └── workflows/
│       ├── ci.yml              # tests + lint on every PR
│       └── deploy.yml          # backend + frontend deploy on merge to main
├── scripts/
│   └── build.sh                # manual CLI build: copies best.onnx into app/, builds container, cleans up
├── tests/
└── README.md
```

`best.onnx` is not committed to the repo. In production it is stored in GCS and downloaded during CI/CD. For local builds, place it in `app/` before running `scripts/build.sh`.

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
- When deployed via GitHub Actions, `best.onnx` is downloaded from GCS (`PROJECT_ID-fish-id-models` bucket) during the workflow. When deploying manually via CLI, `scripts/build.sh` expects a local copy of `best.onnx` passed as an argument.

### Rate Limiting (`app/rate_limiter.py`)

- In-process token bucket: 5 req/min per IP, burst of 3
- Cloud Run: `--max-instances 1`, `--concurrency 5`

### Frontend

Static site on Firebase Hosting; talks to Cloud Run API.
Draws bounding boxes on the image via canvas using the box coordinates returned by `/detect`.
Shows fish count and inference time.

When deployed via GitHub Actions, the Cloud Run URL is injected into `frontend/public/js/app.js` automatically. When deploying manually via CLI, you must replace `YOUR_CLOUD_RUN_URL` in `API_BASE` in `app.js` before running `firebase deploy`. CORS on Cloud Run is restricted to the Firebase Hosting origin.

---

## CI/CD

GitHub Actions handles all deploys on merge to `main`. Manual CLI deployment via `scripts/build.sh` remains available.

### Workflow: `.github/workflows/deploy.yml`

Two jobs run sequentially on every merged PR:

1. **`deploy-api`** — builds the Docker image, pushes to Artifact Registry, downloads `best.onnx` from GCS, deploys to Cloud Run
2. **`deploy-frontend`** — fetches the Cloud Run URL, injects it into `app.js`, deploys `frontend/` to Firebase Hosting

Both jobs authenticate using **Workload Identity Federation** — no long-lived service account keys are stored anywhere. GitHub Actions receives a short-lived OIDC token that is exchanged for GCP credentials scoped to the `fish-id-cicd-sa` service account.

### GitHub Secrets Required

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | Cloud Run / Artifact Registry region |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output from `terraform output workload_identity_provider` |
| `GCP_SERVICE_ACCOUNT` | Output from `terraform output cicd_service_account_email` |
| `ONNX_MODEL_GCS_URI` | `gs://PROJECT_ID-fish-id-models/best.onnx` |

---

## Security

- **Workload Identity Federation** — no long-lived service account keys; short-lived tokens are scoped to the specific repo and restricted to the `main` branch, so feature branches cannot impersonate the CI/CD service account
- **Cloud Run service account** — assigned no IAM roles. The app makes no GCP API calls, so no permissions are needed. This limits blast radius if the app is ever exploited.
- **Image validation** — magic bytes checked server-side to reject non-image payloads regardless of Content-Type header
- **CORS** — restricted to the Firebase Hosting origin (`https://PROJECT_ID.web.app`)
- **Rate limiting** — per-IP token bucket prevents a single caller from running up inference costs
- **No secrets** — no API keys or credentials in this architecture; nothing to leak

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
