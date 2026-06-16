# pylint: disable=import-outside-toplevel
import numpy as np


def make_onnx_output(
    detections: list[dict],
    scale: float = 1.0,
    pad_left: int = 0,
    pad_top: int = 0,
) -> np.ndarray:
    """Build a fake ONNX raw output array.

    Each detection dict: {x1, y1, x2, y2, conf, class_id (optional, default 0)}
    in original image pixel space. Returns shape (1, 4 + num_classes, num_anchors)
    where num_classes is inferred from the highest class_id present.
    """
    num_classes = max((d.get("class_id", 0) for d in detections), default=0) + 1
    num_anchors = max(len(detections), 1)
    output = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)

    for i, d in enumerate(detections):
        cx = (d["x1"] + d["x2"]) / 2 * scale + pad_left
        cy = (d["y1"] + d["y2"]) / 2 * scale + pad_top
        bw = (d["x2"] - d["x1"]) * scale
        bh = (d["y2"] - d["y1"]) * scale
        output[0, 0, i] = cx
        output[0, 1, i] = cy
        output[0, 2, i] = bw
        output[0, 3, i] = bh
        class_id = int(d.get("class_id", 0))
        output[0, 4 + class_id, i] = d["conf"]

    return output


def make_jpeg(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid JPEG image as bytes."""
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
