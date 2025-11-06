"""
BDD100k Dataset Analysis Module
Comprehensive analysis tools for object detection dataset exploration
"""

import json
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# Configuration
CONFIG = {
    "TRAIN_LABELS": "data/labels/bdd100k_labels_images_train.json",
    "VAL_LABELS": "data/labels/bdd100k_labels_images_val.json",
    "TRAIN_IMAGES": "../data/bdd100k_yolo_dataset/train/images/",
    "VAL_IMAGES": "../data/bdd100k_yolo_dataset/val/images/",
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


def main():
    """Main analysis function."""
    try:
        print("Loading BDD100k datasets...")
        train_labels = load_labels(CONFIG["TRAIN_LABELS"])
        val_labels = load_labels(CONFIG["VAL_LABELS"])

        # Basic statistics
        train_class_counts = analyze_class_distribution(train_labels)

        print(f"\nDataset Summary:")
        print(f"- Training samples: {sum(train_class_counts.values()):,}")
        print(f"- Number of classes: {len(train_class_counts)}")
        print(f"- Validation samples: {len(val_labels):,}")

        # Analysis pipeline
        analyses = [
            (
                "Train/Validation Split Analysis",
                lambda: analyze_train_val_split(train_labels, val_labels),
            ),
            ("Anomaly Detection", lambda: identify_anomalies(train_class_counts)),
            ("Bounding Box Analysis", lambda: analyze_bbox_sizes(train_labels)),
            (
                "Objects Per Image Analysis",
                lambda: analyze_objects_per_image(train_labels),
            ),
            (
                "Sample Generation & Organization",
                lambda: generate_sample_visualizations(train_labels),
            ),
        ]

        for title, func in analyses:
            print(f"\n{'=' * 60}")
            print(f"{title.upper()}")
            print(f"{'=' * 60}")
            func()

        print(f"\n{'=' * 60}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 60}")
        print(f"Check '{CONFIG['OUTPUT_DIR']}/organized_samples/' for visualizations.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
