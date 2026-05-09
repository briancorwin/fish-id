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
├── scripts/
│   └── build.sh                # copies best.onnx into app/, builds container, cleans up
├── tests/
├── .env.example
└── README.md
```

`best.onnx` is not committed to the repo. Place it in `app/` before building the container.

---

## GCP Services

| Need | GCP Service |
|---|---|
| Host web API | Cloud Run |
| Container builds | Cloud Build |
| Frontend hosting | Firebase Hosting |

---

## Key Components

### Flask API (`app/main.py`)

Accepts image uploads, runs ONNX inference, returns annotated image + fish count.
Served by gunicorn (not the Flask dev server).

Endpoints:
- `POST /detect` — accepts multipart image, returns `{ fish_count, detections: [{ class_id, confidence, box: { x1, y1, x2, y2 } }] }`
- `GET /health` — health check

The frontend is responsible for drawing bounding boxes on the image using canvas.

Image validation on upload:
- 5MB size limit
- Magic bytes checked server-side (not just Content-Type header) to confirm valid image

### YOLO Model

- YOLOv8n fine-tuned on a specialized fish dataset created using Roboflow
- Exported to ONNX for CPU inference (~300–600ms on Cloud Run 2 vCPU)
- `best.onnx` is placed in `app/` locally and baked into the container at build time

### Rate Limiting (`app/rate_limiter.py`)

- In-process token bucket: 5 req/min per IP, burst of 3
- Cloud Run: `--max-instances 1`, `--concurrency 5`

### Frontend

Static site on Firebase Hosting; talks to Cloud Run API.
Draws bounding boxes on the image via canvas using the box coordinates returned by `/detect`.
Shows fish count and inference time.

CORS on Cloud Run restricted to the Firebase Hosting origin.

---

## Security

- **Cloud Run service account** — assigned no IAM roles. The app makes no GCP API calls, so no permissions are needed. This limits blast radius if the app is ever exploited.
- **Image validation** — magic bytes checked server-side to reject non-image payloads regardless of Content-Type header
- **CORS** — restricted to the Firebase Hosting origin
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
