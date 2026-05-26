FROM python:3.11-slim

WORKDIR /app

COPY training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download base weights to avoid runtime egress to ultralytics.com
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8m.pt')"

COPY training/configs/ /app/configs/
COPY training/train.py /app/train.py
COPY training/eval.py /app/eval.py

ENTRYPOINT ["sh", "-c", "if [ \"$JOB_MODE\" = 'eval' ]; then python /app/eval.py; else python /app/train.py; fi"]
