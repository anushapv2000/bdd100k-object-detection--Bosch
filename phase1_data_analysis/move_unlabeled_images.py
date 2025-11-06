"""
Script to move images without corresponding label files to a separate folder.
This helps clean up the training dataset while preserving the unlabeled images.
"""

import os
import shutil
from pathlib import Path

def move_unlabeled_images(dataset_path, split='train'):
    """
    Move images that don't have corresponding label files to a separate folder.
    
    
    Args:
        dataset_path: Path to the YOLO dataset root
        split: 'train' or 'val'
    """
    images_dir = Path(dataset_path) / split / 'images'
    labels_dir = Path(dataset_path) / split / 'labels'
    
    # Create directory for unlabeled images
    unlabeled_dir = Path(dataset_path) / split / 'images_without_labels'
    unlabeled_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {split} split...")
    print(f"{'='*60}\n")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    print(f"Total images found: {len(image_files)}")
    
    # Check for corresponding labels
    images_without_labels = []
    
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            images_without_labels.append(img_path)
    
    print(f"Images without labels: {len(images_without_labels)}")
    
    if images_without_labels:
        print(f"\nMoving {len(images_without_labels)} images to: {unlabeled_dir}")
        
        # Move images
        for img_path in images_without_labels:
            dest_path = unlabeled_dir / img_path.name
            shutil.move(str(img_path), str(dest_path))
            if len(images_without_labels) <= 10:  # Show details for small numbers
                print(f"  Moved: {img_path.name}")
        
        print(f"\n✓ Successfully moved {len(images_without_labels)} images")
        
        # Verify counts after moving
        remaining_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
        total_labels = len(list(labels_dir.glob('*.txt')))
        
        print(f"\nAfter moving:")
        print(f"  Images in {split}/images: {remaining_images}")
        print(f"  Labels in {split}/labels: {total_labels}")
        print(f"  Images in {split}/images_without_labels: {len(list(unlabeled_dir.glob('*.*')))}")
        
        if remaining_images == total_labels:
            print(f"\n✓ {split} split is now balanced! ✓")
        else:
            print(f"\n⚠ Warning: Still {abs(remaining_images - total_labels)} mismatch")
    else:
        print(f"\n✓ All images have corresponding labels. No action needed.")
    
    return len(images_without_labels)

if __name__ == "__main__":
    # Dataset path
    dataset_path = "/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data/bdd100k_yolo_dataset"
    
    print("\n" + "="*60)
    print("Moving unlabeled images to separate folder")
    print("="*60)
    
    # Process train split
    train_moved = move_unlabeled_images(dataset_path, 'train')
    
    # Process val split
    val_moved = move_unlabeled_images(dataset_path, 'val')
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Train images moved: {train_moved}")
    print(f"Val images moved: {val_moved}")
    print(f"Total images moved: {train_moved + val_moved}")
    print(f"\n✓ Done! You can now retry training.")
    print(f"{'='*60}\n")
