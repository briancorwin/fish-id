# Fish Detector — GCP Architecture Reference

---

## System Overview

```
Browser (Firebase Hosting) → Cloud Run (Flask API + ONNX model)
                                     │ (sync Pub/Sub publish)
                                     ▼
                          Pub/Sub topic: fish-id-analytics-events
                                     │ (push subscription, OIDC auth)
                                     ▼
                     Cloud Run (analytics-consumer) → BigQuery + GCS
```

YOLO model runs directly on Cloud Run CPU via ONNX export (~300–600ms per inference).
Every detection also publishes an analytics event (see [docs/web-app.md](web-app.md#analytics-appanalyticspy-analytics-consumer)) —
both Cloud Run services scale to zero and are billed per-invocation, so there's no always-on worker.

---

## Repo Structure

```
fish-id/
├── app/                        # Cloud Run web API
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── fish_identifier.py
│   ├── rate_limiter.py
│   └── analytics.py            # publishes detection events to Pub/Sub
├── analytics-consumer/         # Cloud Run service, pushed to via Pub/Sub
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                 # inserts into BigQuery, dedup-uploads images to GCS
├── frontend/                   # Firebase Hosting (static)
│   ├── public/
│   │   ├── index.html
│   │   ├── css/styles.css
│   │   └── js/app.js
│   ├── firebase.json
│   └── .firebaserc
├── training/                   # Vertex AI CustomJob training container
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train.py                # entrypoint — downloads dataset from GCS, trains, exports ONNX
│   ├── eval.py                 # entrypoint — evaluates a trained model against the eval split, logs to Vertex AI Experiments
│   └── config.yaml             # training config (architecture + hyperparameters)
├── pipeline/                   # Vertex AI Pipeline (KFP v2)
│   ├── pipeline.py             # KFP pipeline definition; compile with: python pipeline/pipeline.py
│   └── requirements.txt
├── docs/
│   ├── architecture-context.md
│   ├── web-app.md
│   ├── cicd.md
│   ├── training-pipeline.md
│   └── security.md
├── terraform/                  # GCP infrastructure-as-code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── apis.tf
│   ├── iam.tf
│   ├── storage.tf
│   ├── artifact_registry.tf
│   ├── pubsub.tf               # analytics topic + push subscription (bootstrap-gated, see below)
│   ├── bigquery.tf             # analytics dataset + table
│   └── .terraform.lock.hcl    # committed to pin provider versions
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       ├── ci.yml                       # tests + lint on every PR
│       ├── deploy-api.yml               # backend deploy on app/** changes
│       ├── deploy-frontend.yml          # frontend deploy on frontend/** changes
│       ├── deploy-analytics-consumer.yml # builds + deploys analytics-consumer/ on its own path changes
│       └── train-pipeline.yml           # builds training image + compiles pipeline + triggers a run on training/**, pipeline/** changes
├── scripts/
│   ├── requirements.txt
│   ├── deploy-app.sh           # manual CLI build: copies fish-id.onnx into app/, builds container, cleans up
│   ├── update-dataset.py       # exports from Roboflow, syncs images + labels to GCS training bucket
│   └── trigger-training.py     # manually submits a Vertex AI PipelineJob
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── requirements.txt
│   ├── test_main.py
│   ├── test_analytics.py
│   ├── test_analytics_consumer.py
│   ├── test_fish_identifier.py
│   ├── test_rate_limiter.py
│   ├── test_pipeline_components.py
│   └── test_training.py
├── AGENTS.md
├── CLAUDE.md
└── README.md
```

`fish-id.onnx` is not committed to the repo. In production it is stored in GCS and downloaded during CI/CD. For local builds, place it in `app/` before running `scripts/deploy-app.sh`.

The Pub/Sub push subscription in `terraform/pubsub.tf` needs a two-phase
`terraform apply`, the same manual-bootstrap pattern already used for the
`terraform_state` GCS bucket and the GitHub deploy-token secret above: apply
once with the default empty `analytics_consumer_url`, deploy
`analytics-consumer` to obtain its Cloud Run URL, then re-apply with
`-var="analytics_consumer_url=<url>"` to create the subscription. See
`terraform/variables.tf` for details.

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
| Managed training jobs | Vertex AI CustomJob |
| Pipeline orchestration | Vertex AI Pipelines (KFP v2) |
| Model versioning + promotion | Vertex AI Model Registry |
| GitHub PAT for deploy trigger | Secret Manager |
| Async event delivery | Pub/Sub |
| Analytics warehouse | BigQuery |

---

## Web App

See **[docs/web-app.md](web-app.md)** for the full reference: API endpoints, YOLO model, rate limiting, frontend, cost controls, and performance reference.

---

## CI/CD

See **[docs/cicd.md](cicd.md)** for the full reference: all workflows (`ci.yml`, `deploy-api.yml`, `deploy-frontend.yml`, `deploy-analytics-consumer.yml`, `train-pipeline.yml`) and required GitHub secrets.

---

## Continuous Training Pipeline

See **[docs/training-pipeline.md](training-pipeline.md)** for the full reference: dataset management, GCS bucket structure, training container build, job parameters, local testing, IAM, Terraform resources, cost controls, and deferred work.

---

## Security

See **[docs/security.md](security.md)** for the full security reference, including GitHub repository security settings (secret scanning, Dependabot, CodeQL), CI security jobs, CI/CD authentication, and runtime controls.

