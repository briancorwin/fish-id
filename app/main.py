import os
import numpy as np
import cv2
from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS

from rate_limiter import rate_limit

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.environ.get("CORS_ORIGIN", "*")}})

MAX_IMAGE_BYTES = 5 * 1024 * 1024
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.4

# Valid image magic bytes: (header, optional extra check at offset 8)
_IMAGE_MAGIC = [
    (b"\xff\xd8\xff", None),       # JPEG
    (b"\x89PNG\r\n\x1a\n", None),  # PNG
    (b"GIF87a", None),             # GIF
    (b"GIF89a", None),             # GIF
    (b"RIFF", b"WEBP"),            # WEBP
]


def _is_valid_image(data: bytes) -> bool:
    for magic, extra in _IMAGE_MAGIC:
        if data[: len(magic)] == magic:
            if extra is None:
                return True
            return data[8 : 8 + len(extra)] == extra
    return False


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")
_model = YOLO(_MODEL_PATH)

# Used only if the model was exported without class name metadata
_FALLBACK_NAMES: dict[int, str] = {
    # 0: "Largemouth Bass",
    # 1: "Bluegill",
}

_class_names: dict = _model.names or _FALLBACK_NAMES


def _postprocess(result) -> list:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    return [
        {
            "class_id": int(cls),
            "class_name": _class_names.get(int(cls), f"Class {int(cls)}"),
            "confidence": round(float(conf), 4),
            "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
        }
        for (x1, y1, x2, y2), conf, cls in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
        )
    ]


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/detect", methods=["POST"])
@rate_limit
def detect():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    image_bytes = file.read()

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image must be under 5MB"}), 400

    if not _is_valid_image(image_bytes):
        return jsonify({"error": "Invalid image format. Supported: JPEG, PNG, GIF, WEBP"}), 400

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    results = _model(image, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD, verbose=False)
    detections = _postprocess(results[0])

    return jsonify({
        "fish_count": len(detections),
        "detections": detections,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
