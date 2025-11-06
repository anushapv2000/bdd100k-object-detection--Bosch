#!/usr/bin/env python3
"""
Setup Script for BDD100K Object Detection Project

This script helps set up the project structure and validates paths
for all three phases of the Bosch assignment.
"""

import os
import sys
from pathlib import Path


def check_project_structure():
    """Validate the project structure"""
    print("🔍 Checking Project Structure...")
    
    base_dir = Path(__file__).parent
    required_dirs = [
        "data",
        "phase1_data_analysis",
        "phase2_model", 
        "phase3_evaluation"
    ]
    
    required_files = [
        "phase1_data_analysis/README.md",
        "phase2_model/src/training.py",
        "phase2_model/configs/bdd100k.yaml",
        "phase3_evaluation/src/evaluator.py"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Project structure is valid!")
    return True


def check_data_availability():
    """Check if BDD100K data is available"""
    print("\n📁 Checking Data Availability...")
    
    base_dir = Path(__file__).parent
    
    # Check for original JSON labels (in phase1_data_analysis/data/)
    phase1_data_dir = base_dir / "phase1_data_analysis" / "data"
    json_files = [
        "labels/bdd100k_labels_images_train.json",
        "labels/bdd100k_labels_images_val.json"
    ]
    
    # Check for YOLO format dataset (in data/)
    yolo_data_dir = base_dir / "data"
    yolo_structure = [
        "bdd100k_yolo_dataset/train/images",
        "bdd100k_yolo_dataset/train/labels", 
        "bdd100k_yolo_dataset/val/images",
        "bdd100k_yolo_dataset/val/labels"
    ]
    
    missing_json = []
    missing_yolo = []
    
    # Check JSON files
    for json_file in json_files:
        full_path = phase1_data_dir / json_file
        if not full_path.exists():
            missing_json.append(json_file)
    
    # Check YOLO structure
    for yolo_path in yolo_structure:
        full_path = yolo_data_dir / yolo_path
        if not full_path.exists():
            missing_yolo.append(yolo_path)
    
    json_available = len(missing_json) == 0
    yolo_available = len(missing_yolo) == 0
    
    if json_available:
        print("✅ BDD100K JSON label files found!")
    else:
        print(f"⚠️  Missing JSON files: {missing_json}")
    
    if yolo_available:
        print("✅ YOLO format dataset found!")
        # Count some files to show structure
        train_images = list((yolo_data_dir / "bdd100k_yolo_dataset/train/images").glob("*.jpg"))
        train_labels = list((yolo_data_dir / "bdd100k_yolo_dataset/train/labels").glob("*.txt"))
        print(f"   📊 Train: {len(train_images)} images, {len(train_labels)} labels")
    else:
        print(f"⚠️  Missing YOLO structure: {missing_yolo}")
        print("📝 Run Phase 2 conversion script to generate YOLO format")
    
    if not json_available and not yolo_available:
        print("📝 Please ensure BDD100K dataset is downloaded and extracted")
        return False
    
    return json_available or yolo_available


def display_setup_instructions():
    """Display setup instructions for each phase"""
    print("\n🚀 Setup Instructions:")
    print("=" * 50)
    
    instructions = {
        "Phase 1: Data Analysis": [
            "cd phase1_data_analysis/",
            "pip install -r requirements.txt",
            "python dashboard.py  # Interactive dashboard",
            "docker build -t bdd-analysis . && docker run bdd-analysis  # Containerized"
        ],
        
        "Phase 2: Model Training": [
            "cd phase2_model/",
            "pip install -r requirements.txt", 
            "python src/convert_to_yolo.py  # Convert to YOLO format",
            "python src/model_selection.py  # Model analysis",
            "python src/training.py --epochs 1 --demo  # 1-epoch demo",
            "python src/training.py --epochs 100  # Full training"
        ],
        
        "Phase 3: Evaluation": [
            "cd phase3_evaluation/",
            "pip install -r requirements.txt",
            "python src/evaluator.py --model ../phase2_model/runs/train/exp/weights/best.pt",
            "python src/visualizer.py --results outputs/metrics/"
        ]
    }
    
    for phase, commands in instructions.items():
        print(f"\n{phase}:")
        print("-" * len(phase))
        for cmd in commands:
            print(f"  {cmd}")


def main():
    """Main setup function"""
    print("🎯 BDD100K Object Detection - Project Setup")
    print("=" * 60)
    
    # Check project structure
    if not check_project_structure():
        print("\n❌ Project structure validation failed!")
        sys.exit(1)
    
    # Check data availability
    data_available = check_data_availability()
    
    # Display instructions
    display_setup_instructions()
    
    print(f"\n{'='*60}")
    print("✅ Setup validation complete!")
    
    if not data_available:
        print("⚠️  Note: Download BDD100K dataset before running training/evaluation")
    else:
        print("🎉 Ready to run all phases!")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()