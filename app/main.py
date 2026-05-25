import os
import logging
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

from rate_limiter import rate_limit
from fish_identifier import FishIdentifier

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.environ.get("CORS_ORIGIN", "*")}})

MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Valid image magic bytes: (header, optional extra check at offset 8)
_IMAGE_MAGIC = [
    (b"\xff\xd8\xff", None),       # JPEG
    (b"\x89PNG\r\n\x1a\n", None),  # PNG
    (b"GIF87a", None),             # GIF
    (b"GIF89a", None),             # GIF
    (b"RIFF", b"WEBP"),            # WEBP
]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "fish-id.onnx")
_identifier: FishIdentifier | None = None


def _is_valid_image(data: bytes) -> bool:
    for magic, extra in _IMAGE_MAGIC:
        if data[: len(magic)] == magic:
            if extra is None:
                return True
            return data[8 : 8 + len(extra)] == extra
    return False


def _load_model():
    global _identifier
    if _identifier is None:
        try:
            _identifier = FishIdentifier(_MODEL_PATH)
        except Exception as e:
            logger.error("Failed to load model: %s", e)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/class-names", methods=["GET"])
def class_names():
    _load_model()
    names = _identifier.get_class_names() if _identifier is not None else None
    if not names:
        return jsonify({"error": "Model not ready"}), 500
    return jsonify({"class_names": {str(k): v for k, v in names.items()}})


@app.route("/detect", methods=["POST"])
@rate_limit
def detect():
    _load_model()
    if _identifier is None or not _identifier.get_class_names():
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

    try:
        detections = _identifier.detect(image)
    except ValueError as e:
        logger.error("Postprocessing failed: %s", e)
        return jsonify({"error": "Internal model error"}), 500
    except cv2.error as e:
        logger.error("Image processing failed: %s", e)
        return jsonify({"error": "Internal model error"}), 500
    except RuntimeError as e:
        logger.error("Model inference failed: %s", e)
        return jsonify({"error": "Internal model error"}), 500

    return jsonify({
        "fish_count": len(detections),
        "detections": detections,
    })


if __name__ == "__main__":
    _load_model()
    app.run(host="0.0.0.0", port=8080)
