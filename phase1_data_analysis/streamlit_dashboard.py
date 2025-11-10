"""
Interactive Streamlit Dashboard for BDD Dataset Analysis

This module creates a comprehensive Streamlit-based dashboard for visualizing
BDD100k dataset statistics, including class distributions, anomalies,
and comparative analysis between train and validation splits.
"""

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from data_analysis import (
    analyze_class_distribution,
    analyze_objects_per_image,
    analyze_train_val_split,
    identify_anomalies,
    load_labels,
    analyze_bbox_sizes,
)

# Configuration constants
TRAIN_LABELS_PATH = "data/labels/bdd100k_labels_images_train.json"
VAL_LABELS_PATH = "data/labels/bdd100k_labels_images_val.json"
IMAGES_PATH = "data/bdd100k_yolo_dataset/train/images/"

# Style constants
COLORS = {
    "train": "#3498db",
    "val": "#2ecc71",
    "primary": "#2c3e50",
    "secondary": "#7f8c8d",
    "danger": "#e74c3c",
    "success": "#27ae60",
    "warning": "#f39c12",
    "info": "#3498db",
}

COMMON_LAYOUT = {
    "plot_bgcolor": "rgba(240,240,240,0.5)",
    "paper_bgcolor": "white",
    "font": {"family": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"},
}

LEGEND_HORIZONTAL = {
    "orientation": "h",
    "yanchor": "bottom",
    "y": 1.02,
    "xanchor": "right",
    "x": 1,
}

# Create docs directory for saving visualizations
DOCS_DIR = Path('docs')
DOCS_DIR.mkdir(exist_ok=True)


def create_bar_chart(df, x_col, y_cols, colors, title, y_title, log_scale=False):
    """Create a standardized bar chart with train/val comparison."""
    fig = go.Figure()

    for i, (col, name) in enumerate(y_cols):
        fig.add_trace(
            go.Bar(
                name=name,
                x=df[x_col],
                y=df[col],
                marker_color=colors[i],
                text=df[col],
                texttemplate="%{text:,}" if col.endswith("Count") else "%{text:.2f}%",
                textposition="outside",
                textfont={"size": 10},
            )
        )

    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 20, "color": COLORS["primary"]},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_tickangle=-45,
        barmode="group",
        height=550,
        xaxis_title=x_col,
        yaxis_title=y_title,
        yaxis_type="log" if log_scale else "linear",
        legend=LEGEND_HORIZONTAL,
        **COMMON_LAYOUT,
    )
    return fig


def create_anomalies_chart(anomalies_dict):
    """Create anomalies visualization chart."""
    if anomalies_dict:
        anomaly_df = pd.DataFrame(
            list(anomalies_dict.items()), columns=["Class", "Count"]
        ).sort_values("Count", ascending=True)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=anomaly_df["Count"],
                y=anomaly_df["Class"],
                orientation="h",
                marker={
                    "color": anomaly_df["Count"],
                    "colorscale": "Reds",
                    "showscale": False,
                },
                text=anomaly_df["Count"],
                texttemplate="%{text:,}",
                textposition="outside",
            )
        )

        fig.update_layout(
            title={
                "text": "Underrepresented Classes (<1% of total)",
                "font": {"size": 20, "color": COLORS["danger"]},
                "x": 0.5,
                "xanchor": "center",
            },
            height=400,
            xaxis_title="Number of Instances",
            yaxis_title="Class",
            plot_bgcolor="rgba(255,240,240,0.5)",
            paper_bgcolor="white",
        )
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="No anomalies detected - All classes well represented (>1%)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 18, "color": COLORS["success"]},
        )
        fig.update_layout(height=400, plot_bgcolor="rgba(240,255,240,0.5)")
    return fig


def create_histogram(train_data, val_data, title, x_title, y_title):
    """Create a standardized histogram for train/val comparison."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=train_data["object_count"],
            name="Training",
            opacity=0.75,
            marker_color=COLORS["train"],
            nbinsx=30,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=val_data["object_count"],
            name="Validation",
            opacity=0.75,
            marker_color=COLORS["val"],
            nbinsx=30,
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 20, "color": COLORS["primary"]},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        barmode="overlay",
        height=450,
        legend=LEGEND_HORIZONTAL,
        **COMMON_LAYOUT,
    )
    return fig


@st.cache_data
def load_and_analyze_data():
    """Load and analyze dataset with caching for better performance."""
    # Load datasets
    train_labels = load_labels(TRAIN_LABELS_PATH)
    val_labels = load_labels(VAL_LABELS_PATH)

    # Analyze data
    train_class_counts = analyze_class_distribution(train_labels)
    val_class_counts = analyze_class_distribution(val_labels)
    combined_df = analyze_train_val_split(train_labels, val_labels)
    train_anomalies = identify_anomalies(train_class_counts)
    train_obj_per_img = analyze_objects_per_image(train_labels)
    val_obj_per_img = analyze_objects_per_image(val_labels)

    return (
        train_labels,
        val_labels,
        train_class_counts,
        val_class_counts,
        combined_df,
        train_anomalies,
        train_obj_per_img,
        val_obj_per_img,
    )


def save_matplotlib_visualizations(train_labels, val_labels, train_class_counts, val_class_counts):
    """Save static matplotlib visualizations to docs folder"""
    
    try:
        # Get bbox and object data
        from data_analysis import analyze_bbox_sizes, analyze_objects_per_image
        bbox_df = analyze_bbox_sizes(train_labels)
        obj_df = analyze_objects_per_image(train_labels)
        
        objects_per_image_list = obj_df["object_count"].tolist()
        bbox_sizes = bbox_df["area"].tolist()
        
        # Import the save functions from data_analysis
        from data_analysis import (
            save_class_distribution_charts,
            save_object_complexity_chart,
            save_bbox_size_chart
        )
        
        print("\n" + "="*60)
        print("GENERATING DOCUMENTATION VISUALIZATIONS")
        print("="*60)
        
        save_class_distribution_charts(train_class_counts, val_class_counts)
        save_object_complexity_chart(objects_per_image_list)
        save_bbox_size_chart(bbox_sizes)
        
        print(f"\n✓ All visualizations saved to '{DOCS_DIR}/' directory")
        return True
        
    except Exception as e:
        print(f"Warning: Could not save visualizations: {e}")
        return False


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="BDD100k Dataset Analysis",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #856404;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>BDD100k Object Detection Dataset Analysis</h1>
        <p>Comprehensive Analysis of Bounding Box Annotations for 10 Object Classes</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load data
    with st.spinner("Loading and analyzing dataset..."):
        (
            train_labels,
            val_labels,
            train_class_counts,
            val_class_counts,
            combined_df,
            train_anomalies,
            train_obj_per_img,
            val_obj_per_img,
        ) = load_and_analyze_data()

    # Save matplotlib visualizations to docs folder (ADD THIS SECTION)
    if not (DOCS_DIR / 'class_distribution_chart.png').exists():
        with st.spinner("Generating documentation images..."):
            success = save_matplotlib_visualizations(
                train_labels, val_labels, train_class_counts, val_class_counts
            )
            if success:
                st.success(f"✓ Documentation images saved to '{DOCS_DIR}/' folder")
                # Show list of generated files
                st.info("Generated files:")
                for img_file in sorted(DOCS_DIR.glob('*.png')):
                    st.text(f"  ✓ {img_file.name}")

    # Generate sample visualizations if needed
    output_dir = "output_samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    organized_dir = os.path.join(output_dir, "organized_samples")

    if not os.path.exists(organized_dir):
        with st.spinner("Generating sample visualizations..."):
            try:
                from data_analysis import generate_sample_visualizations

                generate_sample_visualizations(train_labels)
                st.success(f"Generated organized visualizations in {organized_dir}/")
            except ImportError as e:
                st.warning(f"Could not generate sample visualizations: {e}")

    # Calculate statistics
    total_train_objects = sum(train_class_counts.values())
    total_val_objects = sum(val_class_counts.values())
    num_train_images = len(train_labels)
    num_val_images = len(val_labels)

    num_train_images_with_box2d = sum(
        1
        for item in train_labels
        if any("box2d" in label for label in item.get("labels", []))
    )
    num_val_images_with_box2d = sum(
        1
        for item in val_labels
        if any("box2d" in label for label in item.get("labels", []))
    )

    avg_train_objects = (
        total_train_objects / num_train_images_with_box2d
        if num_train_images_with_box2d > 0
        else 0
    )

    # Dataset Summary Section
    st.header("Dataset Summary")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Training Images",
            value=f"{num_train_images_with_box2d:,}",
            help="Images with bounding box annotations",
        )

    with col2:
        st.metric(
            label="Validation Images",
            value=f"{num_val_images_with_box2d:,}",
            help="Images with bounding box annotations",
        )

    with col3:
        st.metric(
            label="Training Annotations",
            value=f"{total_train_objects:,}",
            help="Total bounding boxes in training set",
        )

    with col4:
        st.metric(
            label="Avg Objects/Image",
            value=f"{avg_train_objects:.1f}",
            help="Average objects per training image",
        )

    # Information box
    st.markdown(
        f"""
    <div class="info-box">
        <strong>Important:</strong> Analysis focuses on {num_train_images_with_box2d:,} training images and 
        {num_val_images_with_box2d:,} validation images with bounding box annotations for the 10 object detection classes.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Prepare visualization data
    data_train_df = pd.DataFrame(
        list(train_class_counts.items()), columns=["Class", "Train Count"]
    )
    data_val_df = pd.DataFrame(
        list(val_class_counts.items()), columns=["Class", "Validation Count"]
    )

    # Merge and sort data
    all_classes = sorted(set(data_train_df["Class"]).union(set(data_val_df["Class"])))
    data_train_df = (
        data_train_df.set_index("Class")
        .reindex(all_classes, fill_value=0)
        .reset_index()
    )
    data_val_df = (
        data_val_df.set_index("Class").reindex(all_classes, fill_value=0).reset_index()
    )

    combined_viz_df = pd.merge(data_train_df, data_val_df, on="Class")
    combined_viz_df = combined_viz_df.sort_values("Train Count", ascending=False)

    # Class Distribution Analysis
    st.header("Class Distribution Analysis")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Standard View", "Log Scale", "Percentage"])

    with tab1:
        fig_combined = create_bar_chart(
            combined_viz_df,
            "Class",
            [("Train Count", "Training"), ("Validation Count", "Validation")],
            [COLORS["train"], COLORS["val"]],
            "Class Distribution: Training vs Validation",
            "Number of Instances",
        )
        st.plotly_chart(fig_combined, use_container_width=True)

    with tab2:
        fig_log_scale = create_bar_chart(
            combined_viz_df,
            "Class",
            [("Train Count", "Training"), ("Validation Count", "Validation")],
            [COLORS["train"], COLORS["val"]],
            "Class Distribution: Log Scale (Better visibility for rare classes)",
            "Number of Instances (Log Scale)",
            log_scale=True,
        )
        st.plotly_chart(fig_log_scale, use_container_width=True)

    with tab3:
        fig_percentage = create_bar_chart(
            combined_df,
            "Class",
            [("Train %", "Training %"), ("Val %", "Validation %")],
            [COLORS["train"], COLORS["val"]],
            "Percentage Distribution Across Classes",
            "Percentage (%)",
        )
        st.plotly_chart(fig_percentage, use_container_width=True)

    # Data Quality Analysis
    st.header("Data Quality Analysis")

    fig_anomalies = create_anomalies_chart(train_anomalies)
    st.plotly_chart(fig_anomalies, use_container_width=True)

    if train_anomalies:
        st.warning(
            "Some classes are underrepresented (<1% of total). Consider data augmentation or class balancing techniques."
        )
    else:
        st.success("All classes are well represented (>1% each). Good class balance!")

    # Object Density Analysis
    st.header("Object Density Analysis")

    fig_obj_per_img = create_histogram(
        train_obj_per_img,
        val_obj_per_img,
        "Distribution of Objects per Image",
        "Number of Objects per Image",
        "Frequency (Number of Images)",
    )
    st.plotly_chart(fig_obj_per_img, use_container_width=True)

    # Additional insights
    st.header("Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Statistics")
        # Create a summary table
        summary_df = combined_df.copy()
        summary_df = summary_df.round(2)
        st.dataframe(summary_df.style.highlight_max(axis=0), use_container_width=True)

    with col2:
        st.subheader("Dataset Balance")
        # Show class imbalance ratio
        max_count = max(train_class_counts.values())
        min_count = min(train_class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        st.metric(
            label="Class Imbalance Ratio",
            value=f"{imbalance_ratio:.1f}:1",
            help="Ratio between most and least frequent classes",
        )

        # Most and least common classes
        most_common = max(train_class_counts.items(), key=lambda x: x[1])
        least_common = min(train_class_counts.items(), key=lambda x: x[1])

        st.info(f"**Most common:** {most_common[0]} ({most_common[1]:,} instances)")
        st.info(f"**Least common:** {least_common[0]} ({least_common[1]:,} instances)")

    # Sample Visualizations Info
    if os.path.exists(organized_dir):
        st.header("Sample Visualizations")
        st.info(f"Generated sample visualizations are available in: `{organized_dir}/`")

        # Show folder structure
        sample_folders = [
            f
            for f in os.listdir(organized_dir)
            if os.path.isdir(os.path.join(organized_dir, f))
        ]
        if sample_folders:
            st.write("**Available sample categories:**")
            for folder in sorted(sample_folders):
                folder_path = os.path.join(organized_dir, folder)
                num_files = len(
                    [
                        f
                        for f in os.listdir(folder_path)
                        if f.endswith((".png", ".jpg", ".jpeg"))
                    ]
                )
                st.write(f"- `{folder}`: {num_files} samples")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d; padding: 20px;'>"
        "BDD100k Dataset Analysis Dashboard | Object Detection Focus | 10 Classes"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
