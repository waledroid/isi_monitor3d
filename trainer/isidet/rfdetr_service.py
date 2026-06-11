"""RF-DETR inference microservice — runs in its own Docker container.

Exposes:
    POST /load    — load a .pth model
    POST /predict — run inference on a base64-encoded frame
    GET  /health  — healthcheck
"""

import base64
import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
engine = None


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": engine is not None,
    })


@app.route('/load', methods=['POST'])
def load_model():
    global engine
    data = request.json or {}
    weights = data.get('weights')
    conf = data.get('conf_threshold', 0.3)
    imgsz = data.get('imgsz', None)
    device = data.get('device', None)

    if not weights:
        return jsonify({"status": "error", "message": "weights path required"}), 400

    try:
        from src.inference.rfdetr_inferencer import RFDETRInferencer
        engine = RFDETRInferencer(model_path=weights, conf_threshold=conf, imgsz=imgsz, device=device)
        logger.info(f"RF-DETR model loaded: {weights} (imgsz={imgsz}, device={device})")
        return jsonify({"status": "ok", "weights": weights})
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if engine is None:
        return jsonify({"status": "error", "message": "No model loaded"}), 503

    data = request.json or {}
    frame_b64 = data.get('frame')
    if not frame_b64:
        return jsonify({"status": "error", "message": "frame required"}), 400

    # Decode base64 → numpy frame
    frame_bytes = base64.b64decode(frame_b64)
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "invalid frame data"}), 400

    detections = engine.predict_frame(frame)

    # Serialize detections
    result = {
        "xyxy": detections.xyxy.tolist() if detections.xyxy is not None and len(detections.xyxy) > 0 else [],
        "confidence": detections.confidence.tolist() if detections.confidence is not None else [],
        "class_id": detections.class_id.tolist() if detections.class_id is not None else [],
    }

    # Serialize masks if present
    if detections.mask is not None and len(detections.mask) > 0:
        # Encode each mask as a compressed base64 PNG
        masks_b64 = []
        for mask in detections.mask:
            _, buf = cv2.imencode('.png', (mask * 255).astype(np.uint8))
            masks_b64.append(base64.b64encode(buf).decode('ascii'))
        result["masks"] = masks_b64

    return jsonify(result)


if __name__ == '__main__':
    import os
    port = int(os.environ.get('RFDETR_PORT', 9510))
    logger.info(f"RF-DETR service starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
