# Fish Detector — Web App Reference

---

## Flask API (`app/main.py`)

Accepts image uploads, runs ONNX inference, returns bounding box coordinates and fish count.
Served by gunicorn (not the Flask dev server).

Endpoints:
- `POST /detect` — accepts multipart image, returns `{ fish_count, detections: [{ class_id, confidence, box: { x1, y1, x2, y2 } }] }`
- `GET /class-names` — returns `{ class_names: { "0": "Largemouth Bass", "1": "Bluegill", ... } }` from the ONNX model's embedded metadata
- `GET /health` — health check

The frontend is responsible for drawing bounding boxes on the image using canvas.

Image validation on upload:
- 5MB size limit
- Magic bytes checked server-side (not just Content-Type header) to confirm valid image

`CORS_ORIGIN` is set to the Firebase Hosting URL (`https://PROJECT_ID.web.app`) via Cloud Run environment variable, injected at deploy time.

---

## YOLO Model (`app/fish_identifier.py`)

- YOLO fine-tuned on a specialized fish dataset created using Roboflow
- Exported to ONNX for CPU inference (~300–600ms on Cloud Run 2 vCPU)
- When deployed via GitHub Actions, `fish-id.onnx` is downloaded from GCS (`PROJECT_ID-fish-id-models` bucket) during the workflow. When deploying manually via CLI, `scripts/deploy-app.sh` expects a local copy of `fish-id.onnx`.

`fish-id.onnx` is not committed to the repo. Class names are embedded in the ONNX model's metadata at export time and read back at inference time — `GET /class-names` surfaces them.

---

## Rate Limiting (`app/rate_limiter.py`)

- In-process token bucket: 5 req/min per IP, burst of 3
- Cloud Run: `--max-instances 1`, `--concurrency 5`

---

## Frontend (`frontend/`)

Static site on Firebase Hosting; talks to the Cloud Run API.
Draws bounding boxes on the image via canvas using the box coordinates returned by `/detect`.
Shows fish count and inference time.

When deployed via GitHub Actions, the Cloud Run URL is injected into `frontend/public/js/app.js` automatically, and `GCP_PROJECT_ID` in `frontend/.firebaserc` is replaced with the real project ID. When deploying manually via CLI, you must replace `https://YOUR_CLOUD_RUN_URL` in `API_BASE` in `app.js` and set `projects.default` in `.firebaserc` to your project ID before running `firebase deploy`. CORS on Cloud Run is restricted to the Firebase Hosting origin.

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
