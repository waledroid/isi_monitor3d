#!/usr/bin/env python3
import os
import json
import yaml  # 🛑 Added for dynamic config reading
import random
import copy
from pathlib import Path
import logging
from tqdm import tqdm
import cv2
import albumentations as A

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("UniversalDataPrep")

def get_warehouse_augmentations():
    """Pixel-level warehouse augmentations with finalized modern API arguments."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.MotionBlur(blur_limit=(3, 7), p=0.4),
        
        # 🛑 ULTIMATE FIX: Pass the range as a list [min, max] 
        # Modern Albumentations prefers lists for limits to distinguish from single values.
        A.GaussNoise(var_limit=[10.0, 50.0], p=0.4), 
        
        # Standardized range for compression
        A.ImageCompression(quality_range=(80, 100), p=0.3),
        
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    ])

def main():
    # Dynamically find the root folder (logistic/) no matter where you run this
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # 1. Define Absolute Paths
    master_json_path = PROJECT_ROOT / "data/annotations/coco_instance_segmentation.json"
    original_images_dir = PROJECT_ROOT / "data/isi_3k_dataset"
    output_base = PROJECT_ROOT / "data/universal_dataset"
    config_path = PROJECT_ROOT / "configs/train.yaml"
    
    if not master_json_path.exists():
        logger.error(f"❌ Cannot find COCO JSON at {master_json_path}")
        return

    # Create directories
    for split in ['train', 'val']:
        (output_base / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_base / 'labels' / split).mkdir(parents=True, exist_ok=True) # For YOLO
    (output_base / 'annotations').mkdir(parents=True, exist_ok=True)       # For RF-DETR

    logger.info("📦 Loading Master COCO Annotation File...")
    with open(master_json_path, 'r') as f:
        coco_data = json.load(f)

    # Base COCO structures for RF-DETR
    coco_train = {"images": [], "annotations": [], "categories": coco_data['categories']}
    coco_val = {"images": [], "annotations": [], "categories": coco_data['categories']}
    coco_splits = {'train': coco_train, 'val': coco_val}

    # 🛑 DYNAMIC FILTERING: Pull target classes directly from your Master Config!
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    raw_classes = config.get('class_names', ['carton', 'polybag'])
    TARGET_CLASSES = [cls.lower() for cls in raw_classes]
    
    categories = {cat['id']: cat['name'].lower() for cat in coco_data['categories']}
    cat_id_to_yolo_id = {}
    
    for cat_id, cat_name in categories.items():
        if cat_name in TARGET_CLASSES:
            # Dynamically maps to 0, 1, 2, etc., based strictly on your YAML order
            cat_id_to_yolo_id[cat_id] = TARGET_CLASSES.index(cat_name)
            
    logger.info(f"🏷️ Mapped YOLO Classes dynamically from config: {cat_id_to_yolo_id}")
    
    images_info = {img['id']: img for img in coco_data['images']}
    annotations_map = {img_id: [] for img_id in images_info.keys()}
    for ann in coco_data['annotations']:
        if 'segmentation' in ann and len(ann['segmentation']) > 0:
            annotations_map[ann['image_id']].append(ann)

    # Shuffle and Split
    image_ids = list(images_info.keys())
    random.seed(42)
    random.shuffle(image_ids)
    split_index = int(len(image_ids) * 0.8)
    split_mapping = {'train': image_ids[:split_index], 'val': image_ids[split_index:]}

    augmenter = get_warehouse_augmentations()
    AUGMENTATION_MULTIPLIER = 2 
    
    # Global IDs to keep COCO JSON perfectly valid
    global_img_id = 1
    global_ann_id = 1

    for split, ids in split_mapping.items():
        logger.info(f"⚙️ Building {split.upper()} set (YOLO + COCO)...")
        
        for orig_img_id in tqdm(ids, desc=f"{split.capitalize()} Progress"):
            img_info = images_info[orig_img_id]
            file_name = img_info['file_name']
            width, height = img_info['width'], img_info['height']
            
            src_image_path = original_images_dir / file_name
            if not src_image_path.exists(): continue
            img_bgr = cv2.imread(str(src_image_path))

            # Helper to process and save a single iteration (Original or Augmented)
            def save_instance(image_array, suffix=""):
                nonlocal global_img_id, global_ann_id
                base_stem = Path(file_name).stem
                ext = Path(file_name).suffix
                new_file_name = f"{base_stem}{suffix}{ext}"
                
                # 1. Save Image
                cv2.imwrite(str(output_base / 'images' / split / new_file_name), image_array)
                
                # 2. Save YOLO Labels
                yolo_lines = []
                
                # 3. Save COCO Image Entry
                coco_splits[split]['images'].append({
                    "id": global_img_id, "file_name": new_file_name,
                    "width": width, "height": height
                })

                for ann in annotations_map[orig_img_id]:
                    # 🛑 CRITICAL FIX: Skip classes that are NOT in our train.yaml
                    if ann['category_id'] not in cat_id_to_yolo_id:
                        continue
                        
                    # YOLO Formatting
                    yolo_class_id = cat_id_to_yolo_id[ann['category_id']]
                    for polygon in ann['segmentation']:
                        norm_poly = []
                        for i in range(0, len(polygon), 2):
                            x_n = max(0.0, min(1.0, polygon[i] / width))
                            y_n = max(0.0, min(1.0, polygon[i+1] / height))
                            norm_poly.extend([x_n, y_n])
                        yolo_lines.append(f"{yolo_class_id} " + " ".join([f"{v:.6f}" for v in norm_poly]) + "\n")
                    
                    # COCO Formatting (Clone annotation, update IDs)
                    new_ann = copy.deepcopy(ann)
                    new_ann['id'] = global_ann_id
                    new_ann['image_id'] = global_img_id
                    coco_splits[split]['annotations'].append(new_ann)
                    global_ann_id += 1

                # Write YOLO file
                with open(output_base / 'labels' / split / f"{base_stem}{suffix}.txt", 'w') as f:
                    f.writelines(yolo_lines)
                
                global_img_id += 1

            # --- PROCESS ORIGINAL ---
            save_instance(img_bgr, suffix="")

            # --- PROCESS AUGMENTATIONS (TRAIN SET ONLY) ---
            if split == 'train':
                for aug_idx in range(AUGMENTATION_MULTIPLIER):
                    augmented = augmenter(image=img_bgr)
                    save_instance(augmented['image'], suffix=f"_aug{aug_idx}")

    # 🌟 FINAL STEP: Save the COCO JSONs for RF-DETR
    logger.info("📝 Saving COCO JSONs for RF-DETR...")
    with open(output_base / 'annotations' / 'train.json', 'w') as f:
        json.dump(coco_train, f)
    with open(output_base / 'annotations' / 'val.json', 'w') as f:
        json.dump(coco_val, f)

    logger.info(f"✅ Universal Dataset successfully built at: {output_base.absolute()}")

if __name__ == "__main__":
    main()
