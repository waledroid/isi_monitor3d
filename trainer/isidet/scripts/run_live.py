#!/usr/bin/env python3
import sys
import cv2
import queue
import threading
import argparse
import logging
import yaml  # <-- NEW: For reading the config
import supervision as sv
import numpy as np
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Inferencers imported lazily in main() to avoid rfdetr CUDA segfault in Docker
from src.utils.event_logger import EventLogger

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("IsiDetector-Live")

class LiveStreamReader:
    """Background thread to keep the RTSP/Camera buffer fresh (Zero-Lag)."""
    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(self.source)
        self.q = queue.Queue(maxsize=2)
        self.running = True
        
        if not self.cap.isOpened():
            logger.error(f"❌ Could not connect to: {source}")
            sys.exit(1)
            
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

    def get_frame(self):
        return self.q.get()

    def stop(self):
        self.running = False
        self.cap.release()

def draw_isi_ui(frame, counts, mode_text):
    """Custom UI with per-class counters and Mode indicator."""
    h, w = frame.shape[:2]
    max_text_width = 320
    for name, count in counts.items():
        text = f"{count:,} {name.upper()}S"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        max_text_width = max(max_text_width, tw + 40)

    cv2.rectangle(frame, (w - max_text_width, 0), (w, 130), (0, 0, 0), -1)
    y_off = 50
    for name, count in counts.items():
        color = (0, 255, 0) if "carton" in name.lower() else (255, 120, 0)
        cv2.putText(frame, f"{count:,} {name.upper()}S", (w - (max_text_width - 20), y_off), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        y_off += 50

    cv2.rectangle(frame, (w - 200, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, mode_text, (w - 180, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

def main():
    parser = argparse.ArgumentParser(description="IsiDetector Live RTSP Stream")
    parser.add_argument('--weights', type=str, required=True, help="Path to weights")
    parser.add_argument('--source', type=str, required=True, help="RTSP URL or 0 for Webcam")
    # --- NEW: Config Argument ---
    parser.add_argument('--config', type=str, default='configs/train.yaml', help="Path to YAML config")
    parser.add_argument('--device', type=str, default=None, help="Device: None=auto-GPU, 'cpu'=force CPU, 'cuda'=force GPU")
    args = parser.parse_args()

    # ==========================================
    # 0. LOAD CONFIGURATION
    # ==========================================
    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"⚠️ Config not found at {config_path}. Using hardcoded defaults.")

    # Safely extract inference variables (with fallbacks if missing)
    inf_cfg = config.get('inference', {})
    conf_thresh = inf_cfg.get('conf_threshold', 0.3)
    
    track_cfg = inf_cfg.get('tracker', {})
    t_buffer = track_cfg.get('track_buffer', 60)
    m_thresh = track_cfg.get('match_thresh', 0.9)
    
    log_cfg = inf_cfg.get('logging', {})
    save_int = log_cfg.get('save_interval', 3600)
    log_dir = log_cfg.get('log_dir', 'logs')

    # ==========================================
    # 1. ENGINE & MODE LOGIC
    # ==========================================
    weights_path = str(args.weights).lower()
    
    device = args.device  # None = auto-GPU, "cpu" = force CPU, "cuda" = force GPU
    device_label = "CPU" if device == "cpu" else "GPU"

    ext = Path(args.weights).suffix.lower()
    in_docker = Path('/.dockerenv').exists()
    if ext == '.engine':
        from src.inference.tensorrt_inferencer import TensorRTInferencer
        engine = TensorRTInferencer(model_path=args.weights, conf_threshold=conf_thresh, device=device)
        mode_text = f"TensorRT • {device_label}"
    elif ext == '.xml':
        from src.inference.openvino_inferencer import OpenVINOInferencer
        engine = OpenVINOInferencer(model_path=args.weights, conf_threshold=conf_thresh, device=device)
        mode_text = f"OpenVINO • {device_label}"
    elif ext == '.onnx':
        from src.inference.onnx_inferencer import ONNXInferencer
        engine = ONNXInferencer(model_path=args.weights, conf_threshold=conf_thresh, device=device)
        mode_text = f"ONNX • {device_label}"
    elif ext == '.pth':
        if in_docker:
            from src.inference.remote_rfdetr_inferencer import RemoteRFDETRInferencer
            engine = RemoteRFDETRInferencer(model_path=args.weights, conf_threshold=conf_thresh, device=device)
            mode_text = f"RF-DETR Remote • {device_label}"
        else:
            from src.inference.rfdetr_inferencer import RFDETRInferencer
            engine = RFDETRInferencer(model_path=args.weights, conf_threshold=conf_thresh, device=device)
            mode_text = f"RF-DETR • {device_label}"
    else:  # .pt
        from src.inference.yolo_inferencer import YOLOInferencer
        engine = YOLOInferencer(model_path=args.weights, conf_threshold=conf_thresh, device=device)
        mode_text = f"YOLO • {device_label}"

    # ==========================================
    # 2. SETUP ANALYTICS & TRACKING
    # ==========================================
    stream = LiveStreamReader(args.source)
    first_frame = stream.get_frame()
    h, w = first_frame.shape[:2]
    
    # Pass config variables directly to the tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=conf_thresh, # Was track_thresh
        lost_track_buffer=t_buffer,             # Was track_buffer
        minimum_matching_threshold=m_thresh     # Was match_thresh
    )
    

    line_x = int(w * 0.65)
    line_zone = sv.LineZone(
        start=sv.Point(line_x, 0), 
        end=sv.Point(line_x, h),
        triggering_anchors=[sv.Position.BOTTOM_CENTER] 
    )

    mask_annotator = sv.MaskAnnotator()
    trace_annotator = sv.TraceAnnotator()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=0, text_scale=0)

    counted_ids = set()
    class_names_list = list(engine.class_names.values())
    class_totals = {name: 0 for name in class_names_list}

    # Per-event logger — one CSV row per line crossing, 30-day retention.
    retention_days = int(log_cfg.get('retention_days', 30))
    events_dir = f"{str(log_dir).rstrip('/')}/events"
    event_logger = EventLogger(log_dir=events_dir, retention_days=retention_days)

    logger.info(f"📡 {mode_text} Live Stream Active: {args.source}")
    logger.info(f"⚙️ Config Loaded: Conf={conf_thresh}, TrackerBuffer={t_buffer}, SaveInterval={save_int}s")
    
    try:
        while True:
            frame = stream.get_frame()
            if frame is None: continue

            frame_input = np.ascontiguousarray(frame)
            detections = engine.predict_frame(frame_input)
            detections = tracker.update_with_detections(detections)
            
            in_cross, out_cross = line_zone.trigger(detections=detections)
            all_crossings = in_cross | out_cross
            
            for i, crossed in enumerate(all_crossings):
                if crossed and detections.tracker_id is not None:
                    t_id = detections.tracker_id[i]
                    if t_id not in counted_ids:
                        class_id = detections.class_id[i]
                        name = engine.class_names.get(class_id, "object")
                        class_totals[name] = class_totals.get(name, 0) + 1
                        counted_ids.add(t_id)
                        event_logger.log(name, int(t_id))

            annotated = frame.copy()
            if detections.mask is not None:
                annotated = mask_annotator.annotate(scene=annotated, detections=detections)
            
            annotated = trace_annotator.annotate(scene=annotated, detections=detections)
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            
            if detections.tracker_id is not None:
                labels = [f"#{t_id} {engine.class_names.get(c_id, 'obj')}" 
                          for c_id, t_id in zip(detections.class_id, detections.tracker_id)]
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            
            annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)
            annotated = draw_isi_ui(annotated, class_totals, mode_text)

            cv2.imshow("IsiDetector", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        logger.info(f"🛑 Shutting down stream. Final counts: {class_totals}")
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()