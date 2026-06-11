#!/usr/bin/env python3
import sys
import cv2
import argparse
import logging
import supervision as sv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.yolo_inferencer import YOLOInferencer
from src.inference.rfdetr_inferencer import RFDETRInferencer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("IsiDetector")

def draw_isi_ui(frame, counts, mode_text):
    """Industrial UI: Handles counts up to millions with clean formatting."""
    h, w = frame.shape[:2]

    # 1. Format the numbers with commas (1,000,000)
    # Or use shorthand: if count > 999999: f"{count/1e6:.2f}M"
    
    # Let's use the comma version for precision
    y_offset = 50
    
    # Calculate how wide the box needs to be based on the longest count
    max_text_width = 320 # Default minimum
    for class_name, count in counts.items():
        text = f"{count:,} {class_name.upper()}S"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        if tw + 40 > max_text_width:
            max_text_width = tw + 40

    # Draw the Background Box with dynamic width
    cv2.rectangle(frame, (w - max_text_width, 0), (w, 130), (0, 0, 0), -1)

    for class_name, count in counts.items():
        # The :, adds commas automatically!
        text = f"{count:,} {class_name.upper()}S"
        color = (0, 255, 0) if "carton" in class_name.lower() else (255, 120, 0)
        cv2.putText(frame, text, (w - (max_text_width - 20), y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        y_offset += 50

    # 2. Mode Indicator (stays the same)
    cv2.rectangle(frame, (w - 200, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, mode_text, (w - 180, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description="IsiDetector Local Analytics")
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    args = parser.parse_args()

    # 1. Engine Switchboard & Mode Logic
    if 'rfdetr' in str(args.weights).lower():
        engine = RFDETRInferencer(model_path=args.weights, conf_threshold=0.3)
        mode_text = "MODE 2" # RF-DETR
    else:
        engine = YOLOInferencer(model_path=args.weights, conf_threshold=0.3)
        mode_text = "MODE 1" # YOLO

    # 2. Setup Analytics
    video_info = sv.VideoInfo.from_video_path(args.source)
    tracker = sv.ByteTrack(frame_rate=video_info.fps)
    
    # Vertical Line
    line_start = sv.Point(video_info.width // 2, 0)
    line_end = sv.Point(video_info.width // 2, video_info.height)
    line_zone = sv.LineZone(start=line_start, end=line_end)

    # 3. Annotators
    mask_annotator = sv.MaskAnnotator()
    trace_annotator = sv.TraceAnnotator()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=0, text_scale=0)

    counted_ids = set()
    class_totals = {name: 0 for name in engine.class_names.values()}

    generator = sv.get_video_frames_generator(source_path=args.source)

    for frame in generator:
        detections = engine.predict_frame(frame)
        detections = tracker.update_with_detections(detections)
        
        # Unpack crossings
        crossings_in, crossings_out = line_zone.trigger(detections=detections)
        all_crossings = crossings_in | crossings_out
        
        for i, has_crossed in enumerate(all_crossings):
            if has_crossed and detections.tracker_id is not None:
                t_id = detections.tracker_id[i]
                if t_id not in counted_ids:
                    class_id = detections.class_id[i]
                    class_name = engine.class_names.get(class_id, "object")
                    class_totals[class_name] = class_totals.get(class_name, 0) + 1
                    counted_ids.add(t_id)

        annotated = frame.copy()
        if detections.mask is not None:
            annotated = mask_annotator.annotate(scene=annotated, detections=detections)
        
        annotated = trace_annotator.annotate(scene=annotated, detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        
        labels = [
            f"#{t_id} {engine.class_names.get(c_id, 'obj')} {conf:.2f}"
            for c_id, t_id, conf in zip(detections.class_id, detections.tracker_id, detections.confidence)
        ]
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)

        # Apply refined UI
        annotated = draw_isi_ui(annotated, class_totals, mode_text)

        # Window name set to IsiDetector
        cv2.imshow("IsiDetector", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
