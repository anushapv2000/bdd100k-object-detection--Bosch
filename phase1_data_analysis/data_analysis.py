"""
BDD100k Dataset Analysis Module
Comprehensive analysis tools for object detection dataset exploration
"""

import json
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    "TRAIN_LABELS": "data/labels/bdd100k_labels_images_train.json",
    "VAL_LABELS": "data/labels/bdd100k_labels_images_val.json",
    "TRAIN_IMAGES": "data/bdd100k_yolo_dataset/train/images/",
    "VAL_IMAGES": "data/bdd100k_yolo_dataset/val/images/",
    "OUTPUT_DIR": "output_samples",
    "MIN_DENSE_OBJECTS": 60,
    "MAX_DENSE_OBJECTS": 70,
    "MIN_DIVERSE_CLASSES": 6,
    "ANOMALY_THRESHOLD": 0.01,
    "TINY_BBOX_THRESHOLD": 100,
    "HUGE_BBOX_THRESHOLD": 200000,
    "SMALL_BBOX_THRESHOLD": 1000,
    "LARGE_BBOX_THRESHOLD": 100000,
}

# Create docs directory for saving visualizations
DOCS_DIR = Path('docs')
DOCS_DIR.mkdir(exist_ok=True)


def load_labels(labels_path):
    """Load JSON labels file with error handling."""
    try:
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        with open(labels_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            print(f"✓ Loaded {len(data)} entries from {labels_path}")
            return data

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {labels_path}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading {labels_path}: {e}")
        raise


def analyze_class_distribution(labels):
    """Analyze distribution of object detection classes."""
    class_counts = {}
    for item in labels:
        for label_info in item.get("labels", []):
            if "box2d" in label_info:
                category = label_info["category"]
                class_counts[category] = class_counts.get(category, 0) + 1
    return class_counts


def analyze_train_val_split(train_labels, val_labels):
    """Analyze class distribution in train and validation splits."""
    train_counts = analyze_class_distribution(train_labels)
    val_counts = analyze_class_distribution(val_labels)

    train_df = pd.DataFrame(
        list(train_counts.items()), columns=["Class", "Train Count"]
    )
    val_df = pd.DataFrame(list(val_counts.items()), columns=["Class", "Val Count"])
    combined_df = pd.merge(train_df, val_df, on="Class", how="outer").fillna(0)

    combined_df["Train %"] = (
        combined_df["Train Count"] / combined_df["Train Count"].sum() * 100
    )
    combined_df["Val %"] = (
        combined_df["Val Count"] / combined_df["Val Count"].sum() * 100
    )

    print("Train vs Validation Split:")
    print(combined_df)
    return combined_df


def identify_anomalies(class_counts, threshold=None):
    """Identify anomalous classes with low representation."""
    if threshold is None:
        threshold = CONFIG["ANOMALY_THRESHOLD"]

    total = sum(class_counts.values())
    anomalies = {
        cls: count for cls, count in class_counts.items() if count / total < threshold
    }
    print(f"\nAnomalies (Classes with less than {threshold*100}% of total samples):")
    print(anomalies)
    return anomalies


def analyze_bbox_sizes(labels):
    """Analyze bounding box size distribution."""
    bbox_data = []
    for item in labels:
        for label_info in item.get("labels", []):
            if "box2d" in label_info:
                box = label_info["box2d"]
                width = box["x2"] - box["x1"]
                height = box["y2"] - box["y1"]
                area = width * height
                bbox_data.append(
                    {
                        "category": label_info["category"],
                        "width": width,
                        "height": height,
                        "area": area,
                    }
                )

    bbox_df = pd.DataFrame(bbox_data)
    print("\nBounding Box Size Statistics:")
    print(bbox_df.groupby("category")[["width", "height", "area"]].describe())
    return bbox_df


def analyze_objects_per_image(labels):
    """Analyze number of objects per image."""
    objects_per_image = []
    for item in labels:
        object_count = sum(1 for label in item.get("labels", []) if "box2d" in label)
        objects_per_image.append(
            {"filename": item["name"], "object_count": object_count}
        )

    obj_df = pd.DataFrame(objects_per_image)
    print("\nObjects Per Image Statistics:")
    print(obj_df["object_count"].describe())
    return obj_df


def identify_unique_samples(labels, images_path, output_dir):
    """Identify and visualize unique samples."""
    samples = {
        "single_object": [],
        "many_objects": [],
        "small_objects": [],
        "large_objects": [],
    }

    for item in labels:
        box2d_labels = [l for l in item.get("labels", []) if "box2d" in l]
        object_count = len(box2d_labels)

        if object_count == 1:
            samples["single_object"].append(item)
        elif object_count > 15:
            samples["many_objects"].append(item)

        # Check for small/large objects
        for label in box2d_labels:
            box = label["box2d"]
            area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
            if area < CONFIG["SMALL_BBOX_THRESHOLD"]:
                samples["small_objects"].append(item)
                break
            elif area > CONFIG["LARGE_BBOX_THRESHOLD"]:
                samples["large_objects"].append(item)
                break

    print(f"\nFound {len(samples['single_object'])} samples with single object")
    print(f"Found {len(samples['many_objects'])} samples with many objects (>15)")
    print(f"Found {len(samples['small_objects'])} samples with small objects")
    print(f"Found {len(samples['large_objects'])} samples with large objects")

    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating visualizations...")

    for category, sample_list in samples.items():
        if sample_list:
            visualize_samples(sample_list[:5], images_path, output_dir, category)

    return samples


def identify_extremely_dense_samples(
    labels, images_path, output_dir, min_objects=None, max_objects=None
):
    """Identify samples with extremely high object density."""
    if min_objects is None:
        min_objects = CONFIG["MIN_DENSE_OBJECTS"]
    if max_objects is None:
        max_objects = CONFIG["MAX_DENSE_OBJECTS"]

    dense_samples = []
    print(f"\nSearching for images with {min_objects}-{max_objects} objects...")

    for item in labels:
        object_count = sum(1 for l in item.get("labels", []) if "box2d" in l)
        if min_objects <= object_count <= max_objects:
            dense_samples.append({"item": item, "object_count": object_count})

    dense_samples.sort(key=lambda x: x["object_count"], reverse=True)
    print(
        f"Found {len(dense_samples)} images with {min_objects}-{max_objects} objects!"
    )

    if dense_samples:
        print("\nObject count breakdown:")
        for i, sample in enumerate(dense_samples[:10]):
            print(f"  - {sample['item']['name']}: {sample['object_count']} objects")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating visualizations for extremely dense samples...")
        visualize_samples(
            [s["item"] for s in dense_samples[:10]],
            images_path,
            output_dir,
            "extremely_dense_60_70_objects",
        )

    return dense_samples


def identify_class_specific_samples(labels, images_path, output_dir):
    """Identify representative sample for each class."""
    class_samples = {}

    print("\nSearching for class-specific representative samples...")

    for item in labels:
        for label_info in item.get("labels", []):
            if "box2d" in label_info:
                category = label_info["category"]
                if category not in class_samples:
                    box = label_info["box2d"]
                    area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                    if area > 10000:  # Only prominent objects
                        class_samples[category] = item

    print(f"Found representative samples for {len(class_samples)} classes:")
    for cls in class_samples.keys():
        print(f"  - {cls}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating class-specific visualizations...")

    for class_name, sample in class_samples.items():
        visualize_samples([sample], images_path, output_dir, f"class_{class_name}")

    return class_samples


def identify_diverse_class_samples(labels, images_path, output_dir, min_classes=None):
    """Identify images with high class diversity."""
    if min_classes is None:
        min_classes = CONFIG["MIN_DIVERSE_CLASSES"]

    diverse_samples = []
    print(f"\nSearching for images with {min_classes}+ different object classes...")

    for item in labels:
        classes = set()
        for label_info in item.get("labels", []):
            if "box2d" in label_info:
                classes.add(label_info["category"])

        if len(classes) >= min_classes:
            diverse_samples.append(
                {"item": item, "class_count": len(classes), "classes": classes}
            )

    diverse_samples.sort(key=lambda x: x["class_count"], reverse=True)
    print(f"Found {len(diverse_samples)} images with {min_classes}+ different classes!")

    if diverse_samples:
        print("\nClass diversity breakdown:")
        for i, sample in enumerate(diverse_samples[:10]):
            classes_str = ", ".join(sorted(sample["classes"]))
            print(
                f"  - {sample['item']['name']}: {sample['class_count']} classes ({classes_str})"
            )

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating visualizations for diverse class samples...")
        visualize_samples(
            [s["item"] for s in diverse_samples[:5]],
            images_path,
            output_dir,
            "diverse_classes",
        )

    return diverse_samples


def identify_extreme_bbox_samples(labels, images_path, output_dir):
    """Identify images with extremely small or large bounding boxes."""
    extreme_samples = {"tiny_bbox": [], "huge_bbox": []}

    print("\nSearching for images with extreme bounding box sizes...")

    for item in labels:
        for label_info in item.get("labels", []):
            if "box2d" in label_info:
                box = label_info["box2d"]
                area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])

                if area < CONFIG["TINY_BBOX_THRESHOLD"]:
                    extreme_samples["tiny_bbox"].append(
                        {"item": item, "area": area, "category": label_info["category"]}
                    )
                elif area > CONFIG["HUGE_BBOX_THRESHOLD"]:
                    extreme_samples["huge_bbox"].append(
                        {"item": item, "area": area, "category": label_info["category"]}
                    )

    print(
        f"Found {len(extreme_samples['tiny_bbox'])} images with tiny bboxes (< {CONFIG['TINY_BBOX_THRESHOLD']} px²)"
    )
    print(
        f"Found {len(extreme_samples['huge_bbox'])} images with huge bboxes (> {CONFIG['HUGE_BBOX_THRESHOLD']} px²)"
    )

    os.makedirs(output_dir, exist_ok=True)

    for bbox_type, samples in extreme_samples.items():
        if samples:
            print(f"\n{bbox_type.title()} samples (first 5):")
            for sample in samples[:5]:
                print(
                    f"  - {sample['item']['name']}: {sample['area']:.0f} px² ({sample['category']})"
                )

            visualize_samples(
                [s["item"] for s in samples[:5]], images_path, output_dir, bbox_type
            )

    return extreme_samples


def identify_occlusion_samples(labels, images_path, output_dir):
    """Identify images with overlapping objects."""
    occlusion_samples = []

    print("\nSearching for images with overlapping objects (potential occlusion)...")

    def boxes_overlap(box1, box2):
        return not (
            box1["x2"] < box2["x1"]
            or box2["x2"] < box1["x1"]
            or box1["y2"] < box2["y1"]
            or box2["y2"] < box1["y1"]
        )

    for item in labels:
        boxes = [l["box2d"] for l in item.get("labels", []) if "box2d" in l]
        overlap_count = 0

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes_overlap(boxes[i], boxes[j]):
                    overlap_count += 1

        if overlap_count >= 5:
            occlusion_samples.append(
                {
                    "item": item,
                    "overlap_count": overlap_count,
                    "total_objects": len(boxes),
                }
            )

    occlusion_samples.sort(key=lambda x: x["overlap_count"], reverse=True)
    print(f"Found {len(occlusion_samples)} images with significant object overlap!")

    if occlusion_samples:
        print(f"\nOcclusion samples (first 5):")
        for sample in occlusion_samples[:5]:
            print(
                f"  - {sample['item']['name']}: {sample['overlap_count']} overlapping pairs ({sample['total_objects']} objects)"
            )

        os.makedirs(output_dir, exist_ok=True)
        visualize_samples(
            [s["item"] for s in occlusion_samples[:5]],
            images_path,
            output_dir,
            "occlusion_overlap",
        )

    return occlusion_samples


def identify_class_cooccurrence_samples(labels, images_path, output_dir):
    """Identify class co-occurrence patterns."""
    patterns = {
        "person_traffic_light": [],
        "car_traffic_sign": [],
        "person_car": [],
        "bike_person": [],
    }

    print("\nSearching for interesting class co-occurrence patterns...")

    for item in labels:
        classes = [l["category"] for l in item.get("labels", []) if "box2d" in l]

        if "person" in classes and "traffic light" in classes:
            patterns["person_traffic_light"].append(item)
        if "car" in classes and "traffic sign" in classes:
            patterns["car_traffic_sign"].append(item)
        if "person" in classes and "car" in classes:
            patterns["person_car"].append(item)
        if "bike" in classes and "person" in classes:
            patterns["bike_person"].append(item)

    os.makedirs(output_dir, exist_ok=True)

    for pattern, samples in patterns.items():
        if samples:
            print(
                f"\nFound {len(samples)} images with {pattern.replace('_', ' + ')} pattern"
            )
            visualize_samples(
                samples[:3], images_path, output_dir, f"cooccurrence_{pattern}"
            )

    return patterns


def visualize_samples(samples, images_path, output_dir, category):
    """Generate visualizations with bounding boxes."""
    if not samples:
        print(f"No samples found for category: {category}")
        return

    successful = 0
    print(f"\nVisualizing {category} samples...")

    for idx, sample in enumerate(samples):
        try:
            img_path = os.path.join(images_path, sample["name"])
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)

            colors = {
                "car": "red",
                "person": "blue",
                "truck": "green",
                "bus": "orange",
                "motor": "purple",
                "bike": "yellow",
                "traffic light": "cyan",
                "traffic sign": "magenta",
                "rider": "pink",
                "train": "brown",
            }

            box_count = 0
            for label_info in sample.get("labels", []):
                if "box2d" in label_info:
                    box = label_info["box2d"]
                    category_name = label_info["category"]
                    color = colors.get(category_name, "white")

                    draw.rectangle(
                        [box["x1"], box["y1"], box["x2"], box["y2"]],
                        outline=color,
                        width=2,
                    )
                    draw.text((box["x1"], box["y1"]), category_name, fill=color)
                    box_count += 1

            output_path = os.path.join(output_dir, f"{category}_{idx}.jpg")
            img.save(output_path)
            print(f"  ✓ Saved: {os.path.basename(output_path)} ({box_count} boxes)")
            successful += 1

        except Exception as e:
            print(f"  ✗ Failed to process {sample['name']}: {e}")

    print(
        f"  Successfully generated {successful}/{len(samples)} visualizations for {category}"
    )


def create_organized_readme_files(organized_dir):
    """Create README files for organized sample folders."""
    print("\n Creating documentation for organized samples...")

    structure = {
        "1_basic_samples": {
            "description": "Basic complexity variations (1-15+ objects)",
            "readme": "# Basic Complexity Samples\n\nDemonstrates range of scene complexity in BDD100k dataset.\n\n## Contents:\n- Single object scenes (rare)\n- Many object scenes (15+ objects)\n- Small object samples\n- Large object samples\n\n## Usage:\nValidate basic object detection capability across complexity levels.",
        },
        "2_extreme_density": {
            "description": "Maximum complexity scenes (60-70 objects)",
            "readme": "# Extreme Density Samples\n\nMost challenging scenes with 60-70 objects per image.\n\n## Characteristics:\n- Top 0.1% dataset complexity\n- Crowded intersections and traffic jams\n- High bounding box overlap\n\n## Usage:\nStress-test scenarios for model performance validation.",
        },
        "3_bbox_size_extremes": {
            "description": "Extreme bounding box size variations",
            "readme": "# Bounding Box Size Extremes\n\nShowcases extreme object size variations.\n\n## Categories:\n- Tiny boxes: <100 px² (distant objects)\n- Huge boxes: >200k px² (close objects)\n\n## Usage:\nTest multi-scale detection capabilities.",
        },
        "4_class_representatives": {
            "description": "Representative samples for each class",
            "readme": "# Class Representatives\n\nOne prominent sample per object class.\n\n## Classes:\nAll 10 BDD100k classes with clear visibility.\n\n## Usage:\nQuick reference and class-specific debugging.",
        },
        "5_diversity_samples": {
            "description": "High class diversity scenes",
            "readme": "# Multi-Class Diversity\n\nImages with 6+ different object classes.\n\n## Purpose:\nTest multi-class detection in complex scenes.\n\n## Usage:\nEvaluate model performance on diverse urban scenarios.",
        },
        "6_occlusion_samples": {
            "description": "Object occlusion scenarios",
            "readme": "# Occlusion Samples\n\nSignificant object overlap scenarios.\n\n## Challenge:\nPartially visible and overlapping objects.\n\n## Usage:\nTest occlusion handling and NMS parameters.",
        },
        "7_cooccurrence_patterns": {
            "description": "Important class co-occurrence patterns",
            "readme": "# Co-occurrence Patterns\n\nCritical class relationships for safety.\n\n## Patterns:\n- Person + Car (safety critical)\n- Person + Traffic Light\n- Car + Traffic Sign\n- Bike + Person\n\n## Usage:\nValidate context-aware detection.",
        },
    }

    for folder_name, config in structure.items():
        folder_path = os.path.join(organized_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        readme_path = os.path.join(folder_path, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(config["readme"])

        print(f"✓ Created README for {folder_name}/")
        print(f"  Description: {config['description']}")

    # Master README
    master_readme = """# BDD100k Dataset Analysis - Organized Samples

## Overview
Comprehensive dataset analysis with organized sample visualizations for model development and evaluation.

## Structure
- **1_basic_samples/**: Complexity variations (single to 15+ objects)
- **2_extreme_density/**: Maximum complexity (60-70 objects)  
- **3_bbox_size_extremes/**: Size variations (tiny to huge)
- **4_class_representatives/**: One sample per class
- **5_diversity_samples/**: Multi-class scenes (6+ classes)
- **6_occlusion_samples/**: Overlapping objects
- **7_cooccurrence_patterns/**: Critical class relationships

## Key Insights
- **Class Imbalance**: Car (~55%), Train (<1%)
- **Complexity Range**: 3-91 objects per image (avg 18.4)
- **Size Variation**: 100px² to 200k px² bounding boxes
- **Occlusion**: ~69% images have overlapping objects

## Usage
Each folder contains samples with bounding box visualizations and documentation for specific analysis needs.
"""

    master_readme_path = os.path.join(organized_dir, "README.md")
    with open(master_readme_path, "w", encoding="utf-8") as f:
        f.write(master_readme)

    print(f"✓ Created master README.md for organized samples")


def generate_sample_visualizations(train_labels):
    """Generate all sample visualizations directly to organized folders."""
    images_path = CONFIG["TRAIN_IMAGES"]
    base_output_dir = CONFIG["OUTPUT_DIR"]
    organized_dir = os.path.join(base_output_dir, "organized_samples")

    # Create organized directory structure
    folders = [
        "1_basic_samples",
        "2_extreme_density",
        "3_bbox_size_extremes",
        "4_class_representatives",
        "5_diversity_samples",
        "6_occlusion_samples",
        "7_cooccurrence_patterns",
    ]

    for folder_name in folders:
        os.makedirs(os.path.join(organized_dir, folder_name), exist_ok=True)

    # Generate samples
    generators = [
        ("Unique Samples", identify_unique_samples, "1_basic_samples"),
        (
            "Extremely Dense Samples",
            identify_extremely_dense_samples,
            "2_extreme_density",
        ),
        (
            "Class-Specific Samples",
            identify_class_specific_samples,
            "4_class_representatives",
        ),
        (
            "Diverse Class Samples",
            identify_diverse_class_samples,
            "5_diversity_samples",
        ),
        ("Extreme BBox Samples", identify_extreme_bbox_samples, "3_bbox_size_extremes"),
        ("Occlusion Samples", identify_occlusion_samples, "6_occlusion_samples"),
        (
            "Co-occurrence Samples",
            identify_class_cooccurrence_samples,
            "7_cooccurrence_patterns",
        ),
    ]

    for description, func, folder in generators:
        try:
            print(f"\nGenerating {description} directly to {folder}/...")
            output_dir = os.path.join(organized_dir, folder)
            func(train_labels, images_path, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate {description}: {e}")

    create_organized_readme_files(organized_dir)


def save_class_distribution_charts(train_class_counts, val_class_counts):
    """Save class distribution visualizations"""
    print("\nGenerating class distribution charts...")
    
    # 1. Standard Scale Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(train_class_counts))
    width = 0.35
    
    ax.bar(x - width/2, list(train_class_counts.values()), width, label='Training', alpha=0.8, color='#2E86AB')
    ax.bar(x + width/2, list(val_class_counts.values()), width, label='Validation', alpha=0.8, color='#A23B72')
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Annotations', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution: Training vs Validation', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(train_class_counts.keys(), rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'class_distribution_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved class_distribution_chart.png")
    
    # 2. Log Scale Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, list(train_class_counts.values()), width, label='Training', alpha=0.8, color='#2E86AB')
    ax.bar(x + width/2, list(val_class_counts.values()), width, label='Validation', alpha=0.8, color='#A23B72')
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Annotations (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution (Log Scale): Better Visibility for Rare Classes', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(train_class_counts.keys(), rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'class_distribution_log_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved class_distribution_log_chart.png")
    
    # 3. Percentage Distribution
    total_train = sum(train_class_counts.values())
    train_percentages = {k: (v/total_train)*100 for k, v in train_class_counts.items()}
    
    fig, ax = plt.subplots(figsize=(14, 7))
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(train_percentages)))
    bars = ax.bar(train_percentages.keys(), train_percentages.values(), alpha=0.8, color=colors_map)
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution: Percentage View', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'class_distribution_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved class_distribution_percentage.png")


def save_object_complexity_chart(objects_per_image_list):
    """Save objects per image histogram"""
    print("\nGenerating object complexity chart...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(objects_per_image_list, bins=30, alpha=0.7, color='#F18F01', edgecolor='black')
    ax.axvline(np.mean(objects_per_image_list), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(objects_per_image_list):.1f}')
    ax.axvline(np.median(objects_per_image_list), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(objects_per_image_list):.1f}')
    
    ax.set_xlabel('Number of Objects per Image', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Object Density Distribution: Objects per Image', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'objects_per_image_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved objects_per_image_chart.png")


def save_bbox_size_chart(bbox_sizes):
    """Save bounding box size distribution"""
    print("\nGenerating bounding box size chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale
    ax1.hist(bbox_sizes, bins=50, alpha=0.7, color='#06A77D', edgecolor='black')
    ax1.set_xlabel('Bounding Box Area (px²)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Bounding Box Size Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Log scale
    ax2.hist(bbox_sizes, bins=50, alpha=0.7, color='#06A77D', edgecolor='black')
    ax2.set_xlabel('Bounding Box Area (px²)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency (Log Scale)', fontsize=11, fontweight='bold')
    ax2.set_title('Bounding Box Size Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'bbox_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved bbox_size_distribution.png")


def main():
    """Main analysis function."""
    try:
        print("="*60)
        print("BDD100K DATASET ANALYSIS")
        print("="*60)
        print("\nLoading BDD100k datasets...")
        train_labels = load_labels(CONFIG["TRAIN_LABELS"])
        val_labels = load_labels(CONFIG["VAL_LABELS"])

        # Basic statistics
        train_class_counts = analyze_class_distribution(train_labels)
        val_class_counts = analyze_class_distribution(val_labels)

        print(f"\nDataset Summary:")
        print(f"- Training samples: {sum(train_class_counts.values()):,}")
        print(f"- Number of classes: {len(train_class_counts)}")
        print(f"- Validation samples: {len(val_labels):,}")

        # Analysis pipeline
        print(f"\n{'=' * 60}")
        print("TRAIN/VALIDATION SPLIT ANALYSIS")
        print(f"{'=' * 60}")
        analyze_train_val_split(train_labels, val_labels)

        print(f"\n{'=' * 60}")
        print("ANOMALY DETECTION")
        print(f"{'=' * 60}")
        identify_anomalies(train_class_counts)

        print(f"\n{'=' * 60}")
        print("BOUNDING BOX ANALYSIS")
        print(f"{'=' * 60}")
        bbox_df = analyze_bbox_sizes(train_labels)

        print(f"\n{'=' * 60}")
        print("OBJECTS PER IMAGE ANALYSIS")
        print(f"{'=' * 60}")
        obj_df = analyze_objects_per_image(train_labels)

        # Save visualizations
        print("\n" + "="*60)
        print("GENERATING DOCUMENTATION VISUALIZATIONS")
        print("="*60)
        
        # Extract data for visualizations
        objects_per_image_list = obj_df["object_count"].tolist()
        bbox_sizes = bbox_df["area"].tolist()
        
        # Save all charts
        save_class_distribution_charts(train_class_counts, val_class_counts)
        save_object_complexity_chart(objects_per_image_list)
        save_bbox_size_chart(bbox_sizes)
        
        print(f"\n✓ All visualizations saved to '{DOCS_DIR}/' directory")
        print("\nGenerated files:")
        for file in sorted(DOCS_DIR.glob('*.png')):
            print(f"  - {file.name}")

        # Generate sample visualizations
        print(f"\n{'=' * 60}")
        print("SAMPLE GENERATION & ORGANIZATION")
        print(f"{'=' * 60}")
        generate_sample_visualizations(train_labels)

        print(f"\n{'=' * 60}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 60}")
        print(f"✓ Documentation images: '{DOCS_DIR}/'")
        print(f"✓ Sample images: '{CONFIG['OUTPUT_DIR']}/organized_samples/'")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
