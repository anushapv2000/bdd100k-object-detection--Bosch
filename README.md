# BDD100K Object Detection - Complete Assignment

**End-to-end object detection pipeline on Berkeley Deep Drive dataset using YOLOv8**

## Overview

Three-phase implementation covering data analysis, model training, and evaluation:
- **Phase 1** (5 pts): Data analysis and preprocessing
- **Phase 2** (5 pts): YOLOv8 model training pipeline  
- **Phase 3** (10 pts): Evaluation, visualization, and performance analysis

## Prerequisites

**BDD100K Dataset Required**: Download JSON annotation files from [Berkeley Deep Drive](https://bdd-data.berkeley.edu/)
- `bdd100k_labels_images_train.json` → `phase1_data_analysis/data/labels/`
- `bdd100k_labels_images_val.json` → `phase1_data_analysis/data/labels/`

## Structure

```
assignment_data_bdd/
├── phase1_data_analysis/     # Docker-based data analysis
│   └── data/labels/         # Place BDD100K JSON files here
├── phase2_model/            # Virtual env training pipeline
├── phase3_evaluation/       # Virtual env evaluation framework
└── README.md
```

## Quick Start

### Phase 1: Data Analysis (Docker)📄 **[phase1_data_analysis/README.md](phase1_data_analysis/README.md)**
```bash
cd phase1_data_analysis
# Ensure JSON files are in data/labels/ directory
docker build -t bdd100k-analysis .
docker run -p 8501:8501 bdd100k-analysis
# Access: http://localhost:8501
```

### Phase 2: Model Training (Virtual Environment)📄 **[phase2_model/README.md](phase2_model/README.md)**
```bash
cd phase2_model
python -m venv training_env
training_env\Scripts\activate  # Windows
pip install -r requirements.txt

# 1-epoch demo training
python src/training.py --epochs 1 --batch 4 --create-subset --subset-yolo-size 10
```

### Phase 3: Evaluation (Virtual Environment)📄 **[phase3_evaluation/README.md](phase3_evaluation/README.md)**
```bash
cd phase3_evaluation
pip install -r src/requirements.txt
python src/main.py --model-path ../phase2_model/runs/detect/train/weights/best.pt --data-path ../phase1_data_analysis/data --output-dir results/
```

## Implementation Summary

| Phase | Environment | Key Features | Output |
|-------|-------------|--------------|--------|
| **Phase 1** | Docker + Streamlit | Dataset analysis, class distribution | Interactive dashboard, processed data |
| **Phase 2** | Virtual env + GPU | YOLOv8s training, custom loader | Trained model, metrics |
| **Phase 3** | Virtual env | Evaluation, visualization, analysis | Performance reports, recommendations |

## Key Results

- **Dataset**: 69K images, 10 classes, significant class imbalance identified
- **Model**: YOLOv8s trained with 1-epoch demo, full pipeline functional  
- **Evaluation**: Comprehensive metrics, visualizations, and improvement recommendations

## Assignment Requirements Coverage

✅ **Phase 1**: Complete data analysis with interactive dashboard  
✅ **Phase 2**: YOLOv8 training pipeline with 1-epoch demo  
✅ **Phase 3**: Quantitative/qualitative evaluation with personal analysis  
✅ **Integration**: Cross-phase data insights inform training and evaluation strategies  
✅ **Documentation**: Professional README files and comprehensive code documentation  

## 🎯 Future Improvements

Based on evaluation insights:
1. **Data Augmentation**: Targeted augmentation for rare classes
2. **Architecture**: Small object detection improvements
3. **Training Strategy**: Class-balanced sampling and loss functions
4. **Post-processing**: Optimized NMS and confidence thresholds

