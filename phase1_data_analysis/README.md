# Phase 1: BDD100k Dataset Analysis

## Overview
Comprehensive analysis of BDD100k dataset focusing on 10 object detection classes with bounding box annotations.

## Quick Start
```bash
# Run analysis
python data_analysis.py

# Launch interactive Streamlit dashboard
streamlit run streamlit_dashboard.py

# Or specify custom port
streamlit run streamlit_dashboard.py --server.port 8502
```

## Features
- **Class Distribution Analysis**: Train/validation split comparison
- **Statistical Analysis**: Bounding box sizes, objects per image
- **Sample Visualization**: 67 organized samples across 7 categories
- **Interactive Dashboard**: Modern Streamlit web interface
- **Anomaly Detection**: Underrepresented classes identification
- **Responsive UI**: Native metrics, tabs, and professional design

## Comprehensive Analysis Results

### Dataset Overview
- **Training Set**: 69,863 images with 1,286,871 bounding box annotations
- **Validation Set**: 10,073 images with 178,732 annotations  
- **Object Classes**: 10 categories (car, person, bike, truck, bus, train, motor, traffic light, traffic sign, rider)

### Class Distribution Analysis
- **Dominant Classes**: Car (55.2%), Person (18.7%), Traffic Sign (8.9%)
- **Underrepresented**: Train (0.7%), Bus (1.8%), Motor (2.1%)
- **Imbalance Ratio**: 79:1 (Car vs Train)
- **Analysis**: Significant class imbalance requiring attention during model training

### Object Complexity Analysis  
- **Objects per Image**: Range 3-91, Average 18.4, Median 15
- **High Density Images**: 15% have 30+ objects
- **Simple Scenes**: 25% have <10 objects
- **Analysis**: Wide complexity variation suitable for robust model training

### Bounding Box Analysis
- **Size Range**: 100px² to 200,000px² (2000:1 ratio)
- **Small Objects**: 35% under 5,000px²
- **Large Objects**: 10% over 50,000px²
- **Analysis**: Multi-scale detection challenges present

### Data Quality Insights
- **Occlusion Rate**: 69% of images contain overlapping objects
- **Edge Cases**: Extreme weather, night scenes, dense traffic
- **Diversity**: Urban, highway, residential environments

## Analysis Outputs

### Generated Visualizations
- **67 Sample Images**: Organized across 7 analysis categories
- **Interactive Dashboard**: Real-time data exploration at `http://localhost:8501`
- **Statistical Reports**: Comprehensive analysis printed to console

### Sample Categories
1. **Basic Samples** (10 images): Complexity variations from simple to complex scenes
2. **Extreme Density** (10 images): Maximum complexity cases (60-70 objects)
3. **BBox Size Extremes** (10 images): Tiny to huge object size variations
4. **Class Representatives** (10 images): One sample per object class
5. **Diversity Samples** (10 images): Multi-class scenes (6+ different classes)
6. **Occlusion Samples** (10 images): Overlapping and partially hidden objects
7. **Co-occurrence Patterns** (7 images): Critical class relationship examples

All samples include bounding box visualizations with class labels for detailed inspection.

### Key Visualizations

#### Class Distribution Analysis
![Class Distribution](docs/class_distribution_chart.png)
*Training vs Validation class distribution showing significant imbalance (Car: 55%, Train: <1%)*

![Class Distribution Log Scale](docs/class_distribution_log_chart.png)
*Log scale view highlighting underrepresented classes for better visibility*

#### Object Complexity Analysis
![Objects per Image](docs/objects_per_image_chart.png)
*Distribution showing 3-91 objects per image with average of 18.4 objects*

#### Sample Dataset Examples

| Category | Example |
|----------|---------|
| **Basic Complexity** | ![Basic Sample](docs/sample_basic_complexity.jpg) |
| **Extreme Density** | ![Extreme Density](docs/sample_extreme_density.jpg) |
| **Tiny Objects** | ![Tiny Objects](docs/sample_tiny_objects.jpg) |
| **Dominant Class (Car)** | ![Car Sample](docs/sample_class_car.jpg) |
| **Rare Class (Train)** | ![Train Sample](docs/sample_class_train.jpg) |

*All samples include bounding box annotations with class labels*

## Key Analysis Findings Summary

The comprehensive analysis reveals several critical insights for model development:

1. **Severe Class Imbalance**: 79:1 ratio between most common (Car) and rarest (Train) classes
2. **Complex Scene Variations**: Objects per image range from 3 to 91 with high standard deviation
3. **Multi-scale Challenges**: Bounding box sizes vary by 2000:1 ratio
4. **High Occlusion Rate**: 69% of images contain overlapping objects
5. **Training Recommendations**: Class balancing, multi-scale detection, and occlusion handling required

## Project Structure
```
phase1_data_analysis/
├── data_analysis.py           # Main analysis script
├── streamlit_dashboard.py    # Interactive Streamlit dashboard
├── requirements.txt          # Dependencies
├── data/                     # Dataset (not included)
└── output_samples/           # Generated visualizations
    └── organized_samples/    # 7 categorized folders
```

## Interactive Dashboard

### Dashboard Overview
The Streamlit dashboard provides a comprehensive web-based interface for exploring the BDD100k dataset analysis results.

### Key Dashboard Features
- **Dataset Summary**: Key metrics with training/validation statistics
- **Class Distribution**: Three visualization modes:
  - Standard view for overall comparison
  - Log scale for better visibility of rare classes  
  - Percentage distribution for relative analysis
- **Data Quality Analysis**: Anomaly detection and class imbalance warnings
- **Object Density**: Histogram showing objects-per-image distribution
- **Key Insights**: Summary tables and class balance metrics

### Technical Features
- **Performance**: Cached data loading with `@st.cache_data`
- **Responsive**: Mobile-friendly design with adaptive layouts
- **Interactive**: Plotly charts with zoom, pan, and hover details
- **Professional**: Clean interface suitable for presentations

## Dependencies
```bash
pip install -r requirements.txt
```


## Docker Support
```bash
docker build -t bdd100k-analysis .
docker run -p 8501:8501 bdd100k-analysis
```
Dashboard will be available at `http://localhost:8501`

---
*Assignment submission for BDD100k object detection dataset analysis*