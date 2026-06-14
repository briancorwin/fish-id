FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download base weights to avoid runtime egress to ultralytics.com
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY training/config.yaml /app/config.yaml
COPY training/train.py /app/train.py

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/train.py"]
