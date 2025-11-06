"""
Convert BDD100K JSON labels to YOLO format


This script converts BDD100K JSON annotations to YOLO format text files.
Each image gets a corresponding .txt file with one line per object:
class_id x_center y_center width height (all normalized 0-1)

Author: Bosch Assignment - Model Task
Date: November 2025
"""

import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


# Import BDD100K class configuration
from utils import CLASSES

class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}


def convert_bdd_to_yolo(
    json_path: str,
    output_dir: str,
    split: str = 'train'
):
    """
    Convert BDD100K JSON labels to YOLO format.

    Args:
        json_path: Path to BDD100K JSON label file
        output_dir: Directory to save YOLO format labels
        split: Dataset split name (train/val)
    """
    print(f"\n{'='*70}")
    print(f"Converting BDD100K {split} labels to YOLO format")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        labels_data = json.load(f)

    print(f"Total images: {len(labels_data)}")
    print(f"Output directory: {output_dir}\n")

    converted_count = 0
    skipped_count = 0
    total_boxes = 0

    for item in tqdm(labels_data, desc=f"Converting {split}"):
        img_name = item['name']

        # Get image dimensions (BDD100K is 1280x720)
        # These are hardcoded since BDD100K has fixed dimensions
        img_width = 1280
        img_height = 720

        # Parse bounding boxes
        yolo_lines = []

        for label in item.get('labels', []):
            if 'box2d' not in label:
                continue

            category = label['category']

            # Skip if category not in our class list
            if category not in class_to_idx:
                continue

            box2d = label['box2d']
            x1, y1 = box2d['x1'], box2d['y1']
            x2, y2 = box2d['x2'], box2d['y2']

            # Validate box coordinates
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes

            # Convert to YOLO format: [x_center, y_center, width, height]
            # (normalized)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Clip to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            class_id = class_to_idx[category]

            # YOLO format: class_id x_center y_center width height
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            total_boxes += 1

        # Only save if there are annotations
        if yolo_lines:
            # Change extension from .jpg to .txt
            txt_name = Path(img_name).stem + '.txt'
            txt_path = output_path / txt_name

            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            converted_count += 1
        else:
            skipped_count += 1

    print(f"\n{'='*70}")
    print(f"Conversion completed!")
    print(f"  Converted: {converted_count} images")
    print(f"  Skipped (no annotations): {skipped_count} images")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Average boxes per image: {total_boxes/converted_count:.2f}")
    print(f"{'='*70}\n")


def main():
    """Main conversion function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert BDD100K labels to YOLO format')
    parser.add_argument('--data-root', type=str,
                        default='../../phase1_data_analysis/data',
                        help='Root directory of BDD100K JSON labels')
    parser.add_argument('--output-root', type=str,
                        default='../../data/bdd100k_labels_yolo',
                        help='Root directory for YOLO format labels')

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    # Convert train set
    train_json = data_root / 'labels/bdd100k_labels_images_train.json'
    print("-----------------------------------------", train_json)
    train_output = output_root / 'train'

    if train_json.exists():
        convert_bdd_to_yolo(
            json_path=str(train_json),
            output_dir=str(train_output),
            split='train'
        )
    else:
        print(f"WARNING: Train JSON not found at {train_json}")

    # Convert val set
    val_json = data_root / 'labels/bdd100k_labels_images_val.json'
    val_output = output_root / 'val'

    if val_json.exists():
        convert_bdd_to_yolo(
            json_path=str(val_json),
            output_dir=str(val_output),
            split='val'
        )
    else:
        print(f"WARNING: Val JSON not found at {val_json}")

    print("\n✓ All conversions completed!")
    print(f"\nYOLO labels saved to: {output_root}")
    print("\nNext steps:")
    print("  1. Update configs/bdd100k.yaml to point to the labels directories")
    print("  2. Run training: python src/training.py")


if __name__ == "__main__":
    main()
