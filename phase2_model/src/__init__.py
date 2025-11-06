"""
Model package for BDD100k object detection using YOLOv8.

This package contains all the necessary modules for training and inference
of YOLOv8 on the BDD100k dataset.

Author: Bosch Assignment - Model Task
Date: November 2025
"""

__version__ = "1.0.0"
__author__ = "Bosch Assignment"

from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent

# BDD100k class names
CLASSES = [
    'bike', 'bus', 'car', 'motor', 'person',
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]

# Default configuration
DEFAULT_CONFIG = {
    'model': 'yolov8m.pt',
    'img_size': 640,
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'device': 'auto',
    'batch_size': 16,
}

__all__ = [
    'CLASSES',
    'DEFAULT_CONFIG',
    'PACKAGE_ROOT',
]
