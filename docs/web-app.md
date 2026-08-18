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

## Analytics (`app/analytics.py`, `analytics-consumer/`)

Every `/detect` request publishes one Pub/Sub event so results can be analyzed
later in BigQuery, and low-confidence images are saved to GCS for review.

- **Why synchronous**: Cloud Run only guarantees CPU allocation during active
  request processing by default — a background thread started right before
  returning a response risks being frozen before it finishes. So
  `publish_detection_event()` is called synchronously, right before
  `/detect`'s `return jsonify(...)`, using `future.result(timeout=5.0)`. It's
  wrapped in try/except so a Pub/Sub outage never fails the user's request.
- **Event fields**: `timestamp`, `image_hash` (SHA-256 of the raw uploaded
  bytes), `fish_count`, `detections`, `low_confidence`, `image_stored`.
- **Low-confidence trigger** (`LOW_CONFIDENCE_THRESHOLD = 0.5` in
  `app/analytics.py`): any single detection with `confidence < 0.5`, or zero
  detections at all. Only then does the image ride along in the same message,
  base64-encoded as `image_b64`, alongside its `image_content_type` (e.g.
  `image/jpeg`) — this keeps the write path to one publish call.
- **Consumer** (`analytics-consumer/`, a separate Cloud Run service, pushed to
  via a Pub/Sub push subscription): inserts one row per event into BigQuery
  (`fish_id_analytics.detection_events`), and if `image_b64` is present,
  uploads to GCS keyed by `image_hash` with the GCS object's `content_type` set
  from `image_content_type` — but only after checking `blob.exists()` first, so
  re-uploads of the same bytes are a no-op (dedup is exact-byte SHA-256 match,
  not perceptual). `image_content_type` comes from the magic-byte check
  `app/main.py` already performs to validate the upload, rather than trusting
  the client's `Content-Type` header or re-sniffing bytes in the consumer.
  A GCS failure degrades to `image_stored=False` without losing the BigQuery
  row; only a BigQuery failure triggers a Pub/Sub retry. Both Cloud Run
  services scale to zero and are billed per-invocation — no always-on worker.
- **Env vars**: `GCP_PROJECT_ID` (required for the main app to publish at all —
  unset means the feature silently no-ops), `ANALYTICS_TOPIC_ID` (optional,
  defaults to `fish-id-analytics-events`), and on the consumer: `BQ_DATASET`,
  `BQ_TABLE`, `REVIEW_IMAGES_BUCKET`.

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
| `--max-instances 1` on analytics-consumer | Horizontal scaling charges |
| 90-day GCS lifecycle rule on stored images | Unbounded storage growth |

---

## YOLO Performance Reference (Cloud Run 2 vCPU, ONNX)

| Variant | Params | CPU (ONNX) |
|---|---|---|
| YOLOv8n | 3.2M | ~300–600ms |
| YOLOv8s | 11.2M | ~800ms–1.5s |
| YOLOv8m | 25.9M | ~2–4s |

**Stick with nano or small** to keep inference under 1s on Cloud Run CPU.
