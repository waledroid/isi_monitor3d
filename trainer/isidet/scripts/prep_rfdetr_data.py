#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import logging
from tqdm import tqdm  # 🛑 Added tqdm for progress bars

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("DataPrep")

def main():
    # Resolve paths off this script's location so the tool works from any
    # CWD. isidet/scripts/prep_rfdetr_data.py → parent.parent = isidet/.
    ISIDET = Path(__file__).resolve().parent.parent
    src_dir = ISIDET / "data" / "universal_dataset"
    dest_dir = ISIDET / "data" / "rfdetr_dataset"

    if not src_dir.exists():
        logger.error(f"❌ Source directory {src_dir} not found!")
        return

    # YOLO uses 'val', Roboflow COCO expects 'valid'
    splits = {'train': 'train', 'val': 'valid'} 

    for yolo_split, coco_split in splits.items():
        logger.info(f"Processing {yolo_split} -> {coco_split}...")
        
        split_dir = dest_dir / coco_split
        split_dir.mkdir(parents=True, exist_ok=True)

        # 1. Symlink Images with Progress Bar
        img_src = src_dir / "images" / yolo_split
        if img_src.exists():
            # Grab all files first so tqdm knows the total count
            image_files = [f for f in img_src.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            if image_files:
                # 🛑 Wrap the loop in tqdm!
                for img_file in tqdm(image_files, desc=f"Linking {coco_split.upper()} Images", unit="img", colour="green"):
                    dest_file = split_dir / img_file.name
                    if not dest_file.exists():
                        os.symlink(img_file.resolve(), dest_file)
            else:
                logger.warning(f"⚠️ No valid images found at {img_src}")
        else:
            logger.warning(f"⚠️ Image directory not found at {img_src}")

        # 2. Copy the JSON Annotations
        json_src = src_dir / "annotations" / f"{yolo_split}.json"
        json_dest = split_dir / "_annotations.coco.json"
        
        if json_src.exists():
            shutil.copy(json_src, json_dest)
            logger.info(f"✅ Copied COCO JSON to {json_dest}\n")
        else:
            logger.error(f"❌ Missing {json_src}. RF-DETR Seg requires this file!\n")

    logger.info(f"🎉 RF-DETR Dataset formatted successfully at: {dest_dir}")

if __name__ == "__main__":
    main()
