#!/usr/bin/env python3
import sys
import argparse
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import our modular engines
from src.inference.yolo_inferencer import YOLOInferencer
from src.inference.rfdetr_inferencer import RFDETRInferencer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Inference")

def main():
    parser = argparse.ArgumentParser(description="Industrial Inference Engine")
    parser.add_argument('--weights', type=str, required=True, help='Path to best.pt or checkpoint.pth')
    parser.add_argument('--source', type=str, default='0', help='File, folder, "0" (USB), or RTSP')
    parser.add_argument('--conf', type=float, default=0.50, help='Confidence threshold')
    
    parser.add_argument('--show', action='store_true', help='Show live video window')
    parser.add_argument('--save', action='store_true', help='Save results to disk')
    args = parser.parse_args()

    # 1. THE SWITCHBOARD: Automatically select the right engine
    weights_path_str = str(args.weights).lower()
    
    if 'rfdetr' in weights_path_str:
        logger.info(f"🚀 Initializing RF-DETR Transformer Engine: {Path(args.weights).name}")
        engine = RFDETRInferencer(model_path=args.weights, conf_threshold=args.conf)
    else:
        logger.info(f"🚀 Initializing YOLO CNN Engine: {Path(args.weights).name}")
        engine = YOLOInferencer(model_path=args.weights, conf_threshold=args.conf)

    logger.info(f"📡 Connecting to Source: {args.source}")
    
    # 2. Run the Stream
    try:
        for frame_idx, result in enumerate(engine.predict(args.source, show=args.show, save=args.save)):
            
            summary = engine.get_summary(result)
            detected_str = ", ".join([f"{v} {k}" for k, v in summary['counts'].items()])
            detected_str = detected_str if detected_str else "Nothing detected"
            
            if frame_idx % 30 == 0 or not str(args.source).startswith(('rtsp', 'http', '0')):
                logger.info(f"🖼️ [{summary['file_name']}] -> {detected_str}")

    except KeyboardInterrupt:
        logger.info("\n🛑 User initiated emergency stop (Ctrl+C).")
    finally:
        logger.info("✅ Inference safely closed.")

if __name__ == "__main__":
    main()
