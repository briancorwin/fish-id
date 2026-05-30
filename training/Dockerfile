FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download base weights to avoid runtime egress to ultralytics.com
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY training/configs/ /app/configs/
COPY training/train.py /app/train.py
COPY training/eval.py /app/eval.py

ENTRYPOINT ["sh", "-c", "case \"$JOB_MODE\" in eval) python /app/eval.py ;; train) python /app/train.py ;; *) echo 'JOB_MODE must be train or eval' >&2; exit 1 ;; esac"]
