#!/usr/bin/env python3
import sys
from pathlib import Path

# 🛑 FIX: Add the project root to Python's path BEFORE importing from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import argparse
import yaml
import logging

from src.shared.registry import TRAINERS
from src.training.hooks import *

# 🛑 FIX: Import the YOLO trainer so the @register decorator fires!
import src.training.trainers.yolo

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("ValPipeline")

def main():
    # We already defined PROJECT_ROOT at the top, so we can remove it from here
    parser = argparse.ArgumentParser(description="Industrial Validation Pipeline")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to config file')
    parser.add_argument('--weights', type=str, default='runs/segment/models/yolov26/weights/best.pt', help='Path to trained weights')
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    weights_path = PROJECT_ROOT / args.weights

    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return
        
    if not weights_path.exists():
        logger.error(f"❌ Weights file not found: {weights_path}. Did you train the model yet?")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_type = config.get('model_type')
    logger.info(f"🚀 INITIALIZING VALIDATION: {config.get('project_name', 'Unknown')}")
    logger.info(f"🔍 Loading Model Architecture: {model_type}")
    logger.info(f"⚖️ Loading Weights: {weights_path.name}")

    TrainerClass = TRAINERS.get(model_type)
    
    # Initialize the trainer
    trainer = TrainerClass(config)
    
    # 🛑 Override the blank model with our highly trained weights
    from ultralytics import YOLO
    trainer.model = YOLO(str(weights_path))

    # Run the evaluation
    trainer.evaluate()

if __name__ == "__main__":
    main()
