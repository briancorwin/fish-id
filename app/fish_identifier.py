import ast
import json
import logging
import numpy as np
import cv2
import onnxruntime as ort

logger = logging.getLogger(__name__)

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.4
_LETTERBOX_FILL = 114


class FishIdentifier:
    def __init__(self, model_path: str):
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        self._session = session
        self._input_name = input_meta.name
        self._input_h = input_meta.shape[2]
        self._input_w = input_meta.shape[3]
        self._class_names = self._load_class_names(session)

    def getClassNames(self) -> dict[int, str] | None:
        return self._class_names

    def detect(self, image: np.ndarray) -> list:
        blob, scale, pad_left, pad_top = self._preprocess(image)
        raw_output = self._session.run(None, {self._input_name: blob})[0]
        return self._postprocess(raw_output, image.shape, scale, pad_left, pad_top)

    def _load_class_names(self, session) -> dict[int, str] | None:
        try:
            meta = session.get_modelmeta()
            names_str = meta.custom_metadata_map.get("names", "")
            if names_str:
                try:
                    raw = json.loads(names_str)
                except (json.JSONDecodeError, ValueError):
                    # Some YOLOv8 export versions write names as a Python dict literal
                    # (e.g. {0: 'Bass'}) rather than valid JSON — ast.literal_eval handles both.
                    raw = ast.literal_eval(names_str)
                return {int(k): v for k, v in raw.items()}
        except Exception as e:
            logger.error("Failed to load class names from model metadata: %s", e)
            return dict()
        logger.error("Model metadata contains no class names")
        return dict()

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        # image:    BGR ndarray (H, W, 3) uint8
        # Returns: (blob, scale, pad_left, pad_top)
        #   blob:     float32 ndarray (1, 3, _input_h, _input_w), values in [0.0, 1.0]
        #   scale:    float — factor applied to original dimensions so the image fits the model input
        #   pad_left: int   — gray pixels added to the left edge during letterboxing
        #   pad_top:  int   — gray pixels added to the top edge during letterboxing
        h, w = image.shape[:2]
        target_h, target_w = self._input_h, self._input_w

        # Scale uniformly so the longer side fits the model input; preserve aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Center the resized image on a gray canvas (letterboxing)
        padded = np.full((target_h, target_w, 3), _LETTERBOX_FILL, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        # BGR → RGB, uint8 → float32 [0.0, 1.0], HWC → NCHW
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
        return blob, scale, pad_left, pad_top

    def _postprocess(
        self, raw_output: np.ndarray, orig_shape: tuple, scale: float, pad_left: int, pad_top: int
    ) -> list:
        # raw_output: float32 ndarray (1, 4 + num_classes, num_anchors) — direct ONNX output
        # orig_shape: (H, W, 3) of the image before _preprocess, used to clamp box coordinates
        # scale, pad_left, pad_top: values returned by _preprocess, reversed here to map
        #   model-space box coordinates back to original image pixel space

        # YOLO produces one candidate box per grid cell position — num_anchors total across the image.
        # Raw output is (1, 4 + num_classes, num_anchors); transposing makes each row one candidate:
        # (num_anchors, 4 + num_classes) = [cx, cy, w, h, class_0_score, ..., class_N_score]
        predictions = raw_output[0].T

        scores = predictions[:, 4:]
        # For each anchor, find the class with the highest score — that index is the predicted class_id
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= CONF_THRESHOLD
        if not mask.any():
            return []

        boxes_raw = predictions[mask, :4]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Model outputs boxes as (cx, cy, w, h) in model coordinate space;
        # convert to corner format (x1, y1, x2, y2) = (left, top, right, bottom)
        cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Reverse the letterbox transform — undo padding offset then undo scale —
        # to map corners from model coordinate space back to original image pixel space
        orig_h, orig_w = orig_shape[:2]
        x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_top) / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_top) / scale, 0, orig_h)

        boxes_for_nms = [
            [float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])]
            for i in range(len(x1))
        ]
        # Non-Maximum Suppression: when several overlapping boxes detect the same object,
        # keep only the highest-confidence one and suppress others whose IoU exceeds NMS_THRESHOLD
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, confidences.tolist(), CONF_THRESHOLD, NMS_THRESHOLD
        )

        if len(indices) == 0:
            return []

        result = []
        for i in np.array(indices).flatten():
            cid = int(class_ids[i])
            if cid not in self._class_names:
                raise ValueError(f"class_id {cid} not found in model class names")
            result.append({
                "class_id": cid,
                "class_name": self._class_names[cid],
                "confidence": round(float(confidences[i]), 4),
                "box": {
                    "x1": int(x1[i]),
                    "y1": int(y1[i]),
                    "x2": int(x2[i]),
                    "y2": int(y2[i]),
                },
            })
        return result
