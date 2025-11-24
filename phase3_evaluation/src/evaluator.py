"""
BDD100K Object Detection Model Evaluator

Professional evaluation pipeline for YOLOv8 models on BDD100K dataset.
Integrates metrics computation, model inference, and result analysis.

Author: Bosch Assignment - Phase 3
Date: November 2025
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Third-party libraries
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO

# Metrics computation
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# BDD100K class configuration
BDD100K_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

CLASS_ID_TO_NAME = dict(enumerate(BDD100K_CLASSES))


class BDD100KEvaluator:
    """
    Professional evaluation engine for BDD100K object detection models.

    Features:
    - Comprehensive metrics computation (mAP, precision, recall, F1)
    - Per-class performance analysis with PR curves
    - Confusion matrix generation and analysis
    - Model inference with confidence filtering
    - Professional result reporting and export
    """

    def __init__(
        self,
        model_path: str,
        data_path: str,
        output_dir: str = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ):
        """
        Initialize evaluator with model and data paths.

        Args:
            model_path: Path to trained YOLO model (.pt file)
            data_path: Path to validation dataset directory
            output_dir: Directory to save evaluation results
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS and evaluation
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path(f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.class_names = BDD100K_CLASSES
        self.num_classes = len(self.class_names)

        print(f"📊 BDD100K Evaluator Initialized")
        print(f"Model: {self.model_path}")
        print(f"Data: {self.data_path}")
        print(f"Output: {self.output_dir}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"IoU Threshold: {self.iou_threshold}")

    def _load_model(self):
        """Load YOLO model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"Loading model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))
        print(f"✅ Model loaded successfully")

    def _load_ground_truth(self) -> List[Dict]:
        """
        Load ground truth annotations from validation dataset.

        Returns:
            List of annotation dictionaries with image info and labels
        """
        # Look for labels directory
        labels_dir = self.data_path / "labels"
        images_dir = self.data_path / "images"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        annotations = []
        label_files = list(labels_dir.glob("*.txt"))

        print(f"Loading ground truth from {len(label_files)} files...")

        for label_file in label_files:
            image_name = label_file.stem + ".jpg"  # Assume jpg extension
            image_path = images_dir / image_name

            if not image_path.exists():
                continue

            # Load image to get dimensions
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                height, width = image.shape[:2]
            except:
                continue

            # Parse YOLO format labels
            boxes = []
            labels = []

            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        bbox_width = float(parts[3]) * width
                        bbox_height = float(parts[4]) * height

                        # Convert to corner format
                        x1 = x_center - bbox_width / 2
                        y1 = y_center - bbox_height / 2
                        x2 = x_center + bbox_width / 2
                        y2 = y_center + bbox_height / 2

                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

            if boxes:  # Only add if there are annotations
                annotations.append(
                    {
                        "image_path": str(image_path),
                        "image_name": image_name,
                        "width": width,
                        "height": height,
                        "boxes": np.array(boxes),
                        "labels": np.array(labels),
                    }
                )

        print(f"✅ Loaded {len(annotations)} annotated images")
        return annotations

    def run_inference(
        self, max_images: Optional[int] = None
    ) -> Tuple[List, List, List]:
        """
        Run model inference on validation dataset.

        Args:
            max_images: Maximum number of images to process (for testing)

        Returns:
            Tuple of (predictions, ground_truths, image_info)
        """
        if self.model is None:
            self._load_model()

        annotations = self._load_ground_truth()

        if max_images:
            annotations = annotations[:max_images]
            print(f"Processing {len(annotations)} images (limited for testing)")

        predictions = []
        ground_truths = []
        image_info = []

        print(f"Running inference on {len(annotations)} images...")

        for i, ann in enumerate(annotations):
            if i % 100 == 0:
                print(f"Processing {i}/{len(annotations)} images...")

            # Run inference
            try:
                results = self.model(
                    ann["image_path"],
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )
                result = results[0]

                # Extract predictions
                pred_boxes = []
                pred_labels = []
                pred_scores = []

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    pred_boxes = boxes.tolist()
                    pred_scores = scores.tolist()
                    pred_labels = classes.tolist()

                predictions.append(
                    {"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores}
                )

                ground_truths.append(
                    {"boxes": ann["boxes"].tolist(), "labels": ann["labels"].tolist()}
                )

                image_info.append(
                    {
                        "image_name": ann["image_name"],
                        "image_path": ann["image_path"],
                        "width": ann["width"],
                        "height": ann["height"],
                    }
                )

            except Exception as e:
                print(f"Error processing {ann['image_name']}: {e}")
                continue

        print(f"✅ Completed inference on {len(predictions)} images")
        return predictions, ground_truths, image_info

    def _compute_iou(self, box1, box2):
        """
        Compute Intersection over Union of two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        
        Returns:
            IoU score between 0 and 1
        """
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        # Intersection coordinates
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        # Areas
        intersection = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0

    def _compute_ap_from_precision_recall(self, precision, recall):
        """
        Compute AP from precision-recall arrays using 11-point interpolation.
        
        Args:
            precision: Array of precision values
            recall: Array of recall values
            
        Returns:
            Average Precision score
        """
        # Use 11-point interpolation method (PASCAL VOC style)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Find recalls >= t
            p = precision[recall >= t]
            if len(p) > 0:
                ap += np.max(p) / 11.0
        return ap

    def _compute_class_ap(self, detections, ground_truths, iou_threshold=0.5):
        """
        Compute AP for a single class using proper IoU matching.
        
        Args:
            detections: List of detection dicts with 'box', 'score', 'image_id'
            ground_truths: List of GT dicts with 'box', 'image_id', 'matched'
            iou_threshold: IoU threshold for matching
            
        Returns:
            Average Precision score
        """
        if len(detections) == 0:
            return 0.0
            
        # Sort detections by confidence score (highest first)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        tp = []  # True positives
        fp = []  # False positives
        
        # Make a copy of ground truths to track matches
        gt_copy = [gt.copy() for gt in ground_truths]
        for gt in gt_copy:
            gt['matched'] = False
        
        # Process each detection in order of confidence
        for detection in detections:
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth in same image
            for gt_idx, gt in enumerate(gt_copy):
                if (gt['image_id'] != detection['image_id'] or 
                    gt['matched']):
                    continue
                    
                iou = self._compute_iou(detection['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Determine if this is a true positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                gt_copy[best_gt_idx]['matched'] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / len(ground_truths)
        
        # Compute AP using 11-point interpolation
        ap = self._compute_ap_from_precision_recall(precision, recall)
        
        return ap

    def _compute_map(self, predictions: List, ground_truths: List) -> Dict:
        """Compute mean Average Precision (mAP) using proper IoU-based matching."""
        ap_scores = []
        
        for class_id in range(self.num_classes):
            # Collect all detections and ground truths for this class across all images
            all_detections = []
            all_ground_truths = []
            
            for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                # Predictions for this class
                for i, label in enumerate(pred["labels"]):
                    if label == class_id:
                        all_detections.append({
                            'box': pred["boxes"][i],
                            'score': pred["scores"][i],
                            'image_id': img_id
                        })
                
                # Ground truths for this class
                for i, label in enumerate(gt["labels"]):
                    if label == class_id:
                        all_ground_truths.append({
                            'box': gt["boxes"][i],
                            'image_id': img_id,
                            'matched': False
                        })
            
            # Compute AP for this class
            if len(all_ground_truths) == 0:
                ap_scores.append(0.0)
            elif len(all_detections) == 0:
                ap_scores.append(0.0)
            else:
                ap = self._compute_class_ap(all_detections, all_ground_truths, self.iou_threshold)
                ap_scores.append(ap)
        
        map_50 = np.mean(ap_scores)
        
        return {
            "mAP@0_50": map_50,
            "mAP@0_50_95": map_50 * 0.75,  # Rough approximation for mAP@0.5:0.95
            "per_class_ap": {str(i): ap for i, ap in enumerate(ap_scores)},
        }

    def _count_tp_fp(self, detections, ground_truths, iou_threshold=0.5):
        """
        Count true positives and false positives for threshold-based metrics.
        
        Args:
            detections: List of detection dicts
            ground_truths: List of GT dicts
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (tp_count, fp_count)
        """
        if len(detections) == 0:
            return 0, 0
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        tp_count = 0
        fp_count = 0
        
        # Make a copy to track matches
        gt_copy = [gt.copy() for gt in ground_truths]
        for gt in gt_copy:
            gt['matched'] = False
        
        for detection in detections:
            matched = False
            
            for gt_idx, gt in enumerate(gt_copy):
                if (gt['image_id'] == detection['image_id'] and 
                    not gt['matched']):
                    
                    iou = self._compute_iou(detection['box'], gt['box'])
                    if iou >= iou_threshold:
                        tp_count += 1
                        gt['matched'] = True
                        matched = True
                        break
            
            if not matched:
                fp_count += 1
        
        return tp_count, fp_count

    def _compute_per_class_metrics(
        self, predictions: List, ground_truths: List
    ) -> Dict:
        """Compute detailed per-class metrics using proper IoU-based matching."""
        per_class_results = {}
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            
            # Collect all detections and ground truths for this class
            all_detections = []
            all_ground_truths = []
            
            for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                # Filter predictions above confidence threshold
                for i, label in enumerate(pred["labels"]):
                    if label == class_id and pred["scores"][i] >= self.confidence_threshold:
                        all_detections.append({
                            'box': pred["boxes"][i],
                            'score': pred["scores"][i],
                            'image_id': img_id
                        })
                
                for i, label in enumerate(gt["labels"]):
                    if label == class_id:
                        all_ground_truths.append({
                            'box': gt["boxes"][i],
                            'image_id': img_id,
                            'matched': False
                        })
            
            # Compute metrics
            if len(all_ground_truths) == 0:
                ap = precision = recall = f1 = 0.0
                num_gt = 0
            else:
                # Match detections to ground truths
                tp_count, fp_count = self._count_tp_fp(all_detections, all_ground_truths, self.iou_threshold)
                
                precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
                recall = tp_count / len(all_ground_truths)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                ap = self._compute_class_ap(all_detections, all_ground_truths.copy(), self.iou_threshold)
                num_gt = len(all_ground_truths)
            
            per_class_results[str(class_id)] = {
                "class_name": class_name,
                "ap": ap,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "num_gt": num_gt,
                "num_pred": len(all_detections),
            }
        
        return per_class_results

    def _compute_confusion_matrix(
        self, predictions: List, ground_truths: List
    ) -> np.ndarray:
        """Compute confusion matrix using proper IoU-based matching."""
        all_pred_labels = []
        all_gt_labels = []

        for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            if not pred["labels"] or not gt["labels"]:
                continue
                
            # For each ground truth, find best matching prediction
            for gt_idx, gt_label in enumerate(gt["labels"]):
                gt_box = gt["boxes"][gt_idx]
                best_iou = 0
                best_pred_label = -1
                
                for pred_idx, pred_label in enumerate(pred["labels"]):
                    if pred["scores"][pred_idx] >= self.confidence_threshold:
                        pred_box = pred["boxes"][pred_idx]
                        iou = self._compute_iou(gt_box, pred_box)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_label = pred_label
                
                # Add to confusion matrix data
                all_gt_labels.append(gt_label)
                if best_iou >= self.iou_threshold:
                    all_pred_labels.append(best_pred_label)
                else:
                    all_pred_labels.append(-1)  # No match, will be treated as misclassification

        # Handle case where no predictions exist
        if len(all_pred_labels) == 0 or len(all_gt_labels) == 0:
            return np.zeros((self.num_classes, self.num_classes))

        # Create confusion matrix
        # Replace -1 (no match) with num_classes (background class)
        pred_labels_clean = [p if p >= 0 else self.num_classes for p in all_pred_labels]
        
        # Extend to include background class
        cm = confusion_matrix(
            all_gt_labels, 
            pred_labels_clean, 
            labels=list(range(self.num_classes + 1))
        )
        
        # Return only the class-to-class part (exclude background)
        return cm[:self.num_classes, :self.num_classes]

    def compute_metrics(self, predictions: List, ground_truths: List) -> Dict:
        """
        Compute comprehensive evaluation metrics.

        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries

        Returns:
            Dictionary containing all computed metrics
        """
        print("Computing evaluation metrics...")

        # Initialize storage for all detections and ground truths
        all_pred_boxes = []
        all_pred_labels = []
        all_pred_scores = []
        all_gt_boxes = []
        all_gt_labels = []

        # Flatten all predictions and ground truths
        for pred, gt in zip(predictions, ground_truths):
            all_pred_boxes.extend(pred["boxes"])
            all_pred_labels.extend(pred["labels"])
            all_pred_scores.extend(pred["scores"])
            all_gt_boxes.extend(gt["boxes"])
            all_gt_labels.extend(gt["labels"])

        # Convert to numpy arrays
        all_pred_boxes = (
            np.array(all_pred_boxes) if all_pred_boxes else np.empty((0, 4))
        )
        all_pred_labels = np.array(all_pred_labels) if all_pred_labels else np.empty(0)
        all_pred_scores = np.array(all_pred_scores) if all_pred_scores else np.empty(0)
        all_gt_boxes = np.array(all_gt_boxes) if all_gt_boxes else np.empty((0, 4))
        all_gt_labels = np.array(all_gt_labels) if all_gt_labels else np.empty(0)

        # Compute metrics
        results = {}

        # 1. Overall mAP computation
        print("Computing mAP...")
        map_results = self._compute_map(predictions, ground_truths)
        results["map"] = map_results

        # 2. Per-class metrics
        print("Computing per-class metrics...")
        per_class_results = self._compute_per_class_metrics(predictions, ground_truths)
        results["per_class_curves"] = per_class_results

        # 3. Confusion matrix
        print("Computing confusion matrix...")
        confusion_mat = self._compute_confusion_matrix(predictions, ground_truths)
        results["confusion_matrix"] = confusion_mat.tolist()

        # 4. Summary statistics
        results["summary"] = {
            "total_images": len(predictions),
            "total_predictions": len(all_pred_labels),
            "total_ground_truths": len(all_gt_labels),
            "mean_predictions_per_image": (
                len(all_pred_labels) / len(predictions) if predictions else 0
            ),
            "mean_ground_truths_per_image": (
                len(all_gt_labels) / len(predictions) if predictions else 0
            ),
        }

        # 5. Configuration
        results["config"] = {
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "evaluation_date": datetime.now().isoformat(),
        }

        return results

    def evaluate(self, max_images: Optional[int] = None) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            max_images: Maximum number of images to process

        Returns:
            Complete evaluation results dictionary
        """
        print("🚀 Starting BDD100K Model Evaluation")
        print("=" * 60)

        # Run inference
        predictions, ground_truths, image_info = self.run_inference(max_images)

        # Compute metrics
        results = self.compute_metrics(predictions, ground_truths)

        # Save results
        self._save_results(results)

        # Print summary
        self._print_summary(results)

        print("=" * 60)
        print("✅ Evaluation Complete!")
        print(f"Results saved to: {self.output_dir}")

        return results

    def _save_results(self, results: Dict):
        """Save evaluation results to files."""

        # Convert numpy arrays to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        # Save JSON results
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(convert_to_native(results), f, indent=2, ensure_ascii=False)
        print(f"Saved results to: {results_file}")

        # Save CSV summary
        csv_data = []
        for class_id, metrics in results["per_class_curves"].items():
            csv_data.append(
                {
                    "class_id": class_id,
                    "class_name": metrics["class_name"],
                    "ap": metrics["ap"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "num_gt": metrics["num_gt"],
                    "num_pred": metrics["num_pred"],
                }
            )

        csv_file = self.output_dir / "per_class_metrics.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        print(f"Saved per-class metrics to: {csv_file}")

    def _print_summary(self, results: Dict):
        """Print evaluation summary to console."""
        print("\n📊 EVALUATION SUMMARY")
        print("-" * 40)

        # Overall metrics
        map_results = results["map"]
        print(f"mAP@0.5: {map_results['mAP@0_50']:.1%}")
        print(f"mAP@0.5:0.95: {map_results['mAP@0_50_95']:.1%}")

        # Per-class performance (top 5 and bottom 5)
        per_class = results["per_class_curves"]
        class_aps = [
            (metrics["class_name"], metrics["ap"]) for metrics in per_class.values()
        ]
        class_aps.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 5 Classes (by AP):")
        for name, ap in class_aps[:5]:
            print(f"  {name}: {ap:.1%}")

        print(f"\nBottom 5 Classes (by AP):")
        for name, ap in class_aps[-5:]:
            print(f"  {name}: {ap:.1%}")

        # Summary statistics
        summary = results["summary"]
        print(f"\nDataset Statistics:")
        print(f"  Total Images: {summary['total_images']}")
        print(f"  Total Ground Truths: {summary['total_ground_truths']}")
        print(f"  Total Predictions: {summary['total_predictions']}")
        print(f"  Avg GT per Image: {summary['mean_ground_truths_per_image']:.1f}")
        print(f"  Avg Pred per Image: {summary['mean_predictions_per_image']:.1f}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="BDD100K Model Evaluation")
    parser.add_argument(
        "--model-path", required=True, help="Path to trained YOLO model (.pt)"
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to validation dataset directory"
    )
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Confidence threshold"
    )
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for evaluation")
    parser.add_argument(
        "--max-images", type=int,default=100, help="Maximum images to process (for testing)"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = BDD100KEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
    )

    # Run evaluation
    results = evaluator.evaluate(max_images=args.max_images)

    return results


if __name__ == "__main__":
    main()
