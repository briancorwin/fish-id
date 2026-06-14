FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 lsb-release gnupg curl \
    && echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/gcsfuse.list \
    && curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y --no-install-recommends gcsfuse \
    && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download base weights to avoid runtime egress to ultralytics.com
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

ARG IMAGE_TAG=unknown
RUN echo "${IMAGE_TAG}" > /app/image_tag.txt

COPY training/config.yaml /app/config.yaml
COPY training/train.py /app/train.py
COPY training/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
