import os
import numpy as np
import cv2
import onnxruntime as ort
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


# Load model once at startup
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")
_session = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name
INPUT_SIZE = _session.get_inputs()[0].shape[2]


def _preprocess(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    # OpenCV loads BGR; YOLOv8 expects RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)


def _postprocess(output: np.ndarray, orig_w: int, orig_h: int):
    # YOLOv8 ONNX output: [4+nc, num_anchors] — transpose to [num_anchors, 4+nc]
    preds = output[0].T
    nc = preds.shape[1] - 4

    class_scores = preds[:, 4:]                        # [num_anchors, nc]
    conf = np.max(class_scores, axis=1)                # max score across classes
    class_ids = np.argmax(class_scores, axis=1)        # winning class per anchor

    mask = conf >= CONF_THRESHOLD
    if not mask.any():
        return []

    preds, conf, class_ids = preds[mask], conf[mask], class_ids[mask]
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]

    scale_x, scale_y = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE
    x1 = ((cx - w / 2) * scale_x).astype(int)
    y1 = ((cy - h / 2) * scale_y).astype(int)
    x2 = ((cx + w / 2) * scale_x).astype(int)
    y2 = ((cy + h / 2) * scale_y).astype(int)

    boxes = list(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()))
    scores = conf.tolist()
    classes = class_ids.tolist()

    cv2_boxes = [[bx1, by1, bx2 - bx1, by2 - by1] for bx1, by1, bx2, by2 in boxes]
    indices = cv2.dnn.NMSBoxes(cv2_boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
    if len(indices) == 0:
        return []

    return [
        {
            "class_id": classes[i],
            "confidence": round(scores[i], 4),
            "box": {"x1": boxes[i][0], "y1": boxes[i][1], "x2": boxes[i][2], "y2": boxes[i][3]},
        }
        for i in indices.flatten()
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

    orig_h, orig_w = image.shape[:2]
    outputs = _session.run(None, {_input_name: _preprocess(image)})
    detections = _postprocess(outputs[0], orig_w, orig_h)

    return jsonify({
        "fish_count": len(detections),
        "detections": detections,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
