"""
Optimized Inference Module for YOLOv8 on BDD100k

Streamlined inference with consolidated utilities and no redundancy.
Focuses on core inference functionality for the assignment.

Author: Bosch Assignment - Phase 2
Date: November 2025
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from ultralytics import YOLO
from utils import (
    CLASSES, save_image_with_predictions,
    print_model_summary, validate_bbox_format
)


class BDD100kInference:
    """
    Optimized inference class for YOLOv8 on BDD100k dataset.
    """

    def __init__(
        self,
        model_path: str = 'yolov8s.pt',
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize inference class.

        Args:
            model_path: Path to model weights or model name
            device: Device to run on ('auto', 'cuda', 'cpu')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model
        print(f"🔧 Loading model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully on {self.device}")
            print_model_summary(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def predict_image(
        self,
        image_path: str,
        save_visualization: bool = True,
        output_dir: str = 'outputs/inference_samples'
    ) -> Dict:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image
            save_visualization: Whether to save visualization
            output_dir: Directory to save visualization

        Returns:
            Dictionary with predictions
        """
        # Run inference
        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]

        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # Validate boxes
        if len(boxes) > 0:
            image_shape = results.orig_shape  # (height, width)
            if not validate_bbox_format(boxes, image_shape):
                print("Warning: Invalid bounding box format detected")

        # Create prediction dictionary
        predictions = {
            'image': Path(image_path).name,
            'num_detections': len(boxes),
            'detections': []
        }

        for i in range(len(boxes)):
            predictions['detections'].append({
                'class_id': int(classes[i]),
                'class_name': CLASSES[classes[i]] if classes[i] < len(CLASSES) else f'unknown_{classes[i]}',
                'confidence': float(scores[i]),
                'bbox': boxes[i].tolist()  # [x1, y1, x2, y2]
            })

        # Save visualization if requested
        if save_visualization and len(boxes) > 0:
            output_path = Path(output_dir) / f"pred_{Path(image_path).name}"
            save_image_with_predictions(
                image_path, boxes, scores, classes,
                str(output_path), self.conf_threshold
            )

        return predictions

    def predict_batch(
        self,
        image_paths: List[str],
        output_dir: str = 'outputs/inference_samples',
        save_visualizations: bool = True,
        save_json: bool = True
    ) -> List[Dict]:
        """
        Run inference on multiple images.

        Args:
            image_paths: List of image paths
            output_dir: Directory to save results
            save_visualizations: Whether to save visualizations
            save_json: Whether to save JSON results

        Returns:
            List of prediction dictionaries
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"🚀 Running inference on {len(image_paths)} images...")

        all_predictions = []

        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                predictions = self.predict_image(
                    image_path,
                    save_visualization=save_visualizations,
                    output_dir=output_dir
                )
                all_predictions.append(predictions)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        # Save JSON results
        if save_json:
            json_path = output_path / 'predictions.json'
            with open(json_path, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            print(f"💾 Saved predictions to: {json_path}")

        # Print summary
        total_detections = sum(pred['num_detections']
                               for pred in all_predictions)
        print(f"\n📊 Inference Summary:")
        print(f"   Images processed: {len(all_predictions)}")
        print(f"   Total detections: {total_detections}")
        print(
            f"   Average per image: {total_detections/len(all_predictions):.1f}")

        return all_predictions

    def benchmark_speed(
        self,
        image_path: str,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference speed on a single image.

        Args:
            image_path: Path to test image
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with timing statistics
        """
        import time

        print(f"🏃 Benchmarking inference speed...")
        print(f"   Image: {Path(image_path).name}")
        print(f"   Runs: {num_runs} (+ {warmup_runs} warmup)")

        # Warmup runs
        for _ in range(warmup_runs):
            _ = self.model.predict(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

        # Benchmark runs
        times = []
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            start_time = time.time()
            _ = self.model.predict(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        stats = {
            'average_ms': avg_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'fps': 1000 / avg_time,
            'device': self.device,
            'num_runs': num_runs
        }

        print(f"\n⚡ Speed Benchmark Results:")
        print(f"   Average: {avg_time:.1f}ms ({stats['fps']:.1f} FPS)")
        print(f"   Min: {min_time:.1f}ms")
        print(f"   Max: {max_time:.1f}ms")
        print(f"   Device: {self.device}")

        return stats


def main():
    """
    Main function for running inference.

    Usage:
        # Single image inference
        python inference.py --image path/to/image.jpg

        # Batch inference (all images)
        python inference.py --batch path/to/images/*.jpg

        # Batch inference (limited count)
        python inference.py --batch path/to/images/*.jpg --count 5

        # Speed benchmark
        python inference.py --benchmark path/to/image.jpg
    """
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description='Run YOLOv8 inference on BDD100k')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='Model path or name')
    parser.add_argument('--image', type=str,
                        help='Single image path')
    parser.add_argument('--batch', type=str,
                        help='Batch images pattern (e.g., "images/*.jpg")')
    parser.add_argument('--benchmark', type=str,
                        help='Image path for speed benchmark')
    parser.add_argument('--output', type=str, default='outputs/inference_samples',
                        help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    parser.add_argument('--count', type=int, default=None,
                        help='Number of images to process (for batch inference)')

    args = parser.parse_args()

    # Initialize inference
    inference = BDD100kInference(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # Run inference based on mode
    if args.benchmark:
        stats = inference.benchmark_speed(args.benchmark)
        print(f"\n🎯 Benchmark completed: {stats['average_ms']:.1f}ms average")

    elif args.image:
        predictions = inference.predict_image(
            args.image, output_dir=args.output)
        print(
            f"\n🎯 Inference completed: {predictions['num_detections']} detections")

    elif args.batch:
        image_paths = glob.glob(args.batch)
        if not image_paths:
            print(f"❌ No images found matching pattern: {args.batch}")
            return

        # Limit number of images if count is specified
        if args.count is not None:
            image_paths = image_paths[:args.count]
            print(f"📊 Processing {len(image_paths)} images (limited by --count {args.count})")

        predictions = inference.predict_batch(
            image_paths, output_dir=args.output
        )
        print(f"\n🎯 Batch inference completed on {len(predictions)} images")

    else:
        print("❌ Please specify --image, --batch, or --benchmark")
        parser.print_help()


if __name__ == "__main__":
    main()
