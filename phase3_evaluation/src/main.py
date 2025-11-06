"""
BDD100K Complete Evaluation Pipeline

Professional orchestrator script that runs the complete evaluation workflow:
1. Model evaluation with metrics computation
2. Comprehensive visualization generation
3. Personal analysis and improvement recommendations

Author: Bosch Assignment - Phase 3
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
import json

# Import evaluation modules
from evaluator import BDD100KEvaluator
from visualizer import BDD100KVisualizer
from analyzer import BDD100KAnalyzer


def run_complete_evaluation(
    model_path: str,
    data_path: str,
    output_dir: str = None,
    phase1_path: str = None,
    max_images: int = None,
    confidence: float = 0.25,
    iou: float = 0.5,
):
    """
    Run complete BDD100K evaluation pipeline.

    Args:
        model_path: Path to trained YOLO model
        data_path: Path to validation dataset
        output_dir: Output directory for all results
        phase1_path: Optional Phase 1 analysis results
        max_images: Maximum images to process (for testing)
        confidence: Confidence threshold for detections
        iou: IoU threshold for NMS
    """
    print("=" * 80)
    print("BDD100K COMPLETE EVALUATION PIPELINE".center(80))
    print("=" * 80)

    # Initialize output directory
    if output_dir is None:
        from datetime import datetime

        output_dir = f"evaluation_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output Directory: {output_path}")
    print()

    # Step 1: Model Evaluation
    print("🚀 STEP 1: MODEL EVALUATION")
    print("-" * 40)

    evaluator = BDD100KEvaluator(
        model_path=model_path,
        data_path=data_path,
        output_dir=output_path,
        confidence_threshold=confidence,
        iou_threshold=iou,
    )

    results = evaluator.evaluate(max_images=max_images)
    results_path = output_path / "evaluation_results.json"

    print(f"✅ Evaluation completed: {results_path}")
    print()

    # Step 2: Visualization Generation
    print("🎨 STEP 2: VISUALIZATION GENERATION")
    print("-" * 40)

    viz_dir = output_path / "visualizations"
    visualizer = BDD100KVisualizer(
        results_path=str(results_path), output_dir=str(viz_dir)
    )

    visualizer.create_all_visualizations(include_qualitative=False)

    print(f"✅ Visualizations completed: {viz_dir}")
    print()

    # Step 3: Comprehensive Analysis
    print("🔍 STEP 3: COMPREHENSIVE ANALYSIS")
    print("-" * 40)

    analyzer = BDD100KAnalyzer(
        results_path=str(results_path), phase1_insights_path=phase1_path
    )

    analysis_path = output_path / "analysis_report"
    report = analyzer.generate_comprehensive_report(output_path=str(analysis_path))

    print(f"✅ Analysis completed: {analysis_path}.md")
    print()

    # Summary
    print("📊 EVALUATION SUMMARY")
    print("-" * 40)

    overall_map = results.get("map", {}).get("mAP@0_50", 0)
    print(f"Overall mAP@0.5: {overall_map:.1%}")

    # Top and bottom performing classes
    per_class = results.get("per_class_curves", {})
    if per_class:
        class_aps = [
            (metrics.get("class_name", ""), metrics.get("ap", 0))
            for metrics in per_class.values()
        ]
        class_aps.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 3 Classes:")
        for name, ap in class_aps[:3]:
            print(f"  {name}: {ap:.1%}")

        print(f"\nBottom 3 Classes:")
        for name, ap in class_aps[-3:]:
            print(f"  {name}: {ap:.1%}")

    # File summary
    print(f"\n📁 Generated Files:")
    print(f"  📄 Evaluation Results: evaluation_results.json")
    print(f"  📄 Per-class Metrics: per_class_metrics.csv")
    print(f"  📊 Visualizations: visualizations/ (4 plots)")
    print(f"  📋 Analysis Report: analysis_report.md")
    print(f"  📋 Analysis Data: analysis_report.json")

    # Recommendations preview
    recommendations = report.get("improvement_recommendations", {})
    high_priority = recommendations.get("high_priority", [])

    if high_priority:
        print(f"\n🎯 Top Improvement Recommendations:")
        for i, rec in enumerate(high_priority[:3], 1):
            print(f"  {i}. {rec['title']}: {rec['description']}")

    print("\n" + "=" * 80)
    print("✅ COMPLETE EVALUATION FINISHED SUCCESSFULLY!")
    print(f"📁 All results saved to: {output_path}")
    print("=" * 80)

    return {
        "output_dir": str(output_path),
        "results": results,
        "analysis": report,
        "summary": {
            "overall_map": overall_map,
            "top_classes": class_aps[:3] if per_class else [],
            "bottom_classes": class_aps[-3:] if per_class else [],
            "high_priority_improvements": len(high_priority),
        },
    }


def main():
    """Main entry point for complete evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="BDD100K Complete Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete evaluation
  python main.py --model-path best.pt --data-path validation/
  
  # Quick test with limited images
  python main.py --model-path model.pt --data-path val/ --max-images 50
  
  # With Phase 1 integration
  python main.py --model-path model.pt --data-path val/ --phase1-path phase1_results.json
  
  # Custom thresholds
  python main.py --model-path model.pt --data-path val/ --confidence 0.3 --iou 0.6
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model-path", required=True, help="Path to trained YOLO model (.pt file)"
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to validation dataset directory"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        help="Output directory for all results (default: timestamped folder)",
    )
    parser.add_argument(
        "--phase1-path", help="Path to Phase 1 analysis results (for integration)"
    )
    parser.add_argument(
        "--max-images", type=int, help="Maximum images to process (for testing)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU threshold for NMS (default: 0.5)"
    )

    args = parser.parse_args()

    # Validate inputs
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)

    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)

    if not data_path.exists():
        print(f"❌ Error: Data directory not found: {data_path}")
        sys.exit(1)

    # Check for required subdirectories
    if not (data_path / "images").exists():
        print(f"❌ Error: Images directory not found: {data_path / 'images'}")
        sys.exit(1)

    if not (data_path / "labels").exists():
        print(f"❌ Error: Labels directory not found: {data_path / 'labels'}")
        sys.exit(1)

    # Validate Phase 1 path if provided
    if args.phase1_path and not Path(args.phase1_path).exists():
        print(f"⚠️  Warning: Phase 1 file not found: {args.phase1_path}")
        print("Continuing without Phase 1 integration...")
        args.phase1_path = None

    try:
        # Run complete evaluation
        results = run_complete_evaluation(
            model_path=str(model_path),
            data_path=str(data_path),
            output_dir=args.output_dir,
            phase1_path=args.phase1_path,
            max_images=args.max_images,
            confidence=args.confidence,
            iou=args.iou,
        )

        print(f"\n🎉 Success! Evaluation completed successfully.")
        print(f"📂 Results available at: {results['output_dir']}")

        return results

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
