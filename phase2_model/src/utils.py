"""
Optimized Utility Functions for BDD100k Object Detection

Consolidated utility functions with no redundancy.
Provides visualization, metrics computation, and data processing helpers.

Author: Bosch Assignment - Phase 2
Date: November 2025
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


# BDD100K class configuration
CLASSES = [
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motor', 'bike', 'traffic light', 'traffic sign'
]

# Colors for visualization (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),      # person - blue
    (0, 255, 0),      # rider - green
    (0, 0, 255),      # car - red
    (255, 255, 0),    # truck - cyan
    (255, 0, 255),    # bus - magenta
    (0, 255, 255),    # train - yellow
    (128, 0, 128),    # motor - purple
    (255, 128, 0),    # bike - orange
    (0, 128, 255),    # traffic light - light blue
    (128, 255, 0),    # traffic sign - lime
]


def draw_bboxes_on_image(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw bounding boxes on image with labels and confidence scores.

    Args:
        image: Input image (BGR format)
        boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        labels: Class labels [N]
        scores: Confidence scores [N] (optional)
        class_names: List of class names (optional, uses CLASSES if None)
        colors: List of colors for each class (optional, uses COLORS if None)

    Returns:
        Image with drawn bounding boxes
    """
    image = image.copy()

    # Use default class names and colors if not provided
    if class_names is None:
        class_names = CLASSES
    if colors is None:
        colors = COLORS

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        label_idx = int(labels[i])
        color = colors[label_idx % len(colors)]

        # Draw bounding box rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Create label text
        if label_idx < len(class_names):
            label_text = class_names[label_idx]
        else:
            label_text = f"Class {label_idx}"

        if scores is not None:
            label_text += f": {scores[i]:.2f}"

        # Get label size for background rectangle
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            image, label_text, (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return image


def save_image_with_predictions(
    image_path: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    save_path: str,
    conf_threshold: float = 0.25
):
    """
    Load image, draw predictions, and save result.

    Args:
        image_path: Path to input image
        boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        scores: Confidence scores [N]
        classes: Class IDs [N]
        save_path: Path to save visualization
        conf_threshold: Minimum confidence to display
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return

    # Filter by confidence threshold
    valid_indices = scores >= conf_threshold
    if not valid_indices.any():
        print(f"Warning: No detections above threshold {conf_threshold}")
        return

    filtered_boxes = boxes[valid_indices]
    filtered_scores = scores[valid_indices]
    filtered_classes = classes[valid_indices]

    # Draw predictions
    result_image = draw_bboxes_on_image(
        image, filtered_boxes, filtered_classes, filtered_scores
    )

    # Save result
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(save_path), result_image)
    if success:
        print(f"Saved visualization: {save_path}")
    else:
        print(f"Failed to save visualization: {save_path}")


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: Box 1 in format [x1, y1, x2, y2]
        box2: Box 2 in format [x1, y1, x2, y2]

    Returns:
        IoU score (0.0 to 1.0)
    """
    # Compute intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    # Compute IoU
    return intersection / union if union > 0 else 0.0


def validate_bbox_format(
        boxes: np.ndarray, image_shape: Tuple[int, int]) -> bool:
    """
    Validate bounding box format and coordinates.

    Args:
        boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        image_shape: Image shape (height, width)

    Returns:
        True if all boxes are valid, False otherwise
    """
    if len(boxes) == 0:
        return True

    height, width = image_shape

    # Check box format
    if boxes.shape[1] != 4:
        print(f"Error: Expected 4 coordinates per box, got {boxes.shape[1]}")
        return False

    # Check coordinate ranges
    valid_x = (boxes[:, [0, 2]] >= 0) & (boxes[:, [0, 2]] <= width)
    valid_y = (boxes[:, [1, 3]] >= 0) & (boxes[:, [1, 3]] <= height)

    if not (valid_x.all() and valid_y.all()):
        print("Error: Some box coordinates are outside image bounds")
        return False

    # Check that x2 > x1 and y2 > y1
    valid_order = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not valid_order.all():
        print("Error: Some boxes have invalid coordinate ordering")
        return False

    return True


def load_and_validate_config(config_path: str) -> Dict:
    """
    Load and validate YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    # Validate class count consistency
    if len(config['names']) != config['nc']:
        raise ValueError(
            f"Class count mismatch: nc={config['nc']}, names={len(config['names'])}")

    return config


def print_model_summary(model, input_size: Tuple[int, int] = (640, 640)):
    """
    Print summary of model architecture and parameters.

    Args:
        model: YOLOv8 model instance
        input_size: Input image size (height, width)
    """
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    # Model info
    if hasattr(model, 'model'):
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.model.parameters() if p.requires_grad)

        print(f"Model: {model.__class__.__name__}")
        print(f"Input Size: {input_size[0]}×{input_size[1]}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # Class information
    if hasattr(model, 'names'):
        print(f"\nClasses ({len(model.names)}):")
        for i, name in model.names.items():
            print(f"  {i}: {name}")

    print("=" * 60)
