import os
import ast
import json
import logging
import numpy as np
import cv2
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

from rate_limiter import rate_limit

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.environ.get("CORS_ORIGIN", "*")}})

MAX_IMAGE_BYTES = 5 * 1024 * 1024
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.4
_LETTERBOX_FILL = 114

# Valid image magic bytes: (header, optional extra check at offset 8)
_IMAGE_MAGIC = [
    (b"\xff\xd8\xff", None),       # JPEG
    (b"\x89PNG\r\n\x1a\n", None),  # PNG
    (b"GIF87a", None),             # GIF
    (b"GIF89a", None),             # GIF
    (b"RIFF", b"WEBP"),            # WEBP
]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")

_session = None
_input_name = None
_input_h = None
_input_w = None
_class_names = None
_model_ready = False


def _is_valid_image(data: bytes) -> bool:
    for magic, extra in _IMAGE_MAGIC:
        if data[: len(magic)] == magic:
            if extra is None:
                return True
            return data[8 : 8 + len(extra)] == extra
    return False


def _load_class_names(session) -> dict[int, str] | None:
    try:
        meta = session.get_modelmeta()
        names_str = meta.custom_metadata_map.get("names", "")
        if names_str:
            try:
                raw = json.loads(names_str)
            except (json.JSONDecodeError, ValueError):
                raw = ast.literal_eval(names_str)
            return {int(k): v for k, v in raw.items()}
    except Exception as e:
        logger.error("Failed to load class names from model metadata: %s", e)
        return None
    logger.error("Model metadata contains no class names")
    return None


def _load_model():
    global _session, _input_name, _input_h, _input_w, _class_names, _model_ready
    if _model_ready:
        return
    session = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    _session = session
    _input_name = input_meta.name
    _input_h = input_meta.shape[2]
    _input_w = input_meta.shape[3]
    _class_names = _load_class_names(session)
    _model_ready = True


def _preprocess(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    target_h, target_w = _input_h, _input_w
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    padded = np.full((target_h, target_w, 3), _LETTERBOX_FILL, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    return blob, scale, pad_left, pad_top


def _postprocess(
    raw_output: np.ndarray, orig_shape: tuple, scale: float, pad_left: int, pad_top: int
) -> list:
    # raw_output shape: (1, 4 + num_classes, num_anchors)
    predictions = raw_output[0].T  # (num_anchors, 4 + num_classes)

    scores = predictions[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences >= CONF_THRESHOLD
    if not mask.any():
        return []

    boxes_raw = predictions[mask, :4]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    orig_h, orig_w = orig_shape[:2]
    x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_top) / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_top) / scale, 0, orig_h)

    boxes_for_nms = [
        [float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])]
        for i in range(len(x1))
    ]
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms, confidences.tolist(), CONF_THRESHOLD, NMS_THRESHOLD
    )

    if len(indices) == 0:
        return []

    result = []
    for i in np.array(indices).flatten():
        cid = int(class_ids[i])
        result.append({
            "class_id": cid,
            "class_name": _class_names.get(cid, f"Class {cid}"),
            "confidence": round(float(confidences[i]), 4),
            "box": {
                "x1": int(x1[i]),
                "y1": int(y1[i]),
                "x2": int(x2[i]),
                "y2": int(y2[i]),
            },
        })
    return result


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/detect", methods=["POST"])
@rate_limit
def detect():
    _load_model()
    if _class_names is None:
        return jsonify({"error": "Model not ready"}), 500

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

    blob, scale, pad_left, pad_top = _preprocess(image)
    raw_output = _session.run(None, {_input_name: blob})[0]
    detections = _postprocess(raw_output, image.shape, scale, pad_left, pad_top)

    return jsonify({
        "fish_count": len(detections),
        "detections": detections,
    })


if __name__ == "__main__":
    _load_model()
    app.run(host="0.0.0.0", port=8080)
