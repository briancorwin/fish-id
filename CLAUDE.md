# Fish Detector — Claude Code Guide

## Project

A web app that accepts an uploaded image, runs a fine-tuned YOLOv8 fish-detection model on CPU, and returns an annotated result. Hosted on GCP: Cloud Run API + Firebase Hosting frontend.

## Architecture

Full architecture decisions and GCP service mapping are documented in:

**[docs/architecture-context.md](docs/architecture-context.md)**

Key points:
- `app/` — Flask API on Cloud Run; ONNX inference runs directly on CPU
- `frontend/` — Static site on Firebase Hosting

## Stack

- **Model:** YOLOv8n (Ultralytics), ONNX export for CPU inference
- **API:** Flask + Cloud Run (`--max-instances 1`)
- **Frontend:** Firebase Hosting (static HTML/CSS/JS)

## Development Conventions

- Python 3.11+
- `requirements.txt` lives in `app/`
- Environment variables (never hardcoded): `GCP_PROJECT`, `GCP_REGION`
- Secrets via Secret Manager, not `.env` files in production
- Image upload limit: 5MB enforced in both frontend JS and Flask
- `best.onnx` is not committed to the repo — place it in `app/` before building the container
- GCP setup steps (APIs, service account, deploy commands) live in `README.md`

## Cost Guard

Always preserve `--max-instances 1` on Cloud Run. Do not raise this without explicit confirmation — it is the primary cost control for this demo.
