# Week 6 – YOLOv8 Object Detection on the NEU Surface Defect Dataset

## Overview

This project trains and evaluates YOLOv8 object detection models on the NEU Surface Defect Dataset. Four YOLOv8 variants (Nano, Small, Medium, and Large) were trained and compared to determine the best-performing model based on standard object detection metrics.

## Dataset

Dataset: NEU Surface Defect Dataset

https://www.kaggle.com/datasets/zymzym/neu-yolo

The dataset contains images of steel surface defects with YOLO-format annotations for six defect categories.

Classes:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

## Models Evaluated

- YOLOv8 Nano (yolov8n)
- YOLOv8 Small (yolov8s)
- YOLOv8 Medium (yolov8m)
- YOLOv8 Large (yolov8l)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Optimizer | Default YOLOv8 Optimizer |
| GPU | NVIDIA Tesla T4 |
| Framework | Ultralytics YOLOv8 |

## Evaluation Metrics

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|----------:|--------:|--------:|----------:|
| YOLOv8 Nano | XX | XX | XX | XX |
| YOLOv8 Small | XX | XX | XX | XX |
| YOLOv8 Medium | XX | XX | XX | XX |
| YOLOv8 Large | XX | XX | XX | XX |

Replace the values above with your actual results.

## Best Model

The best-performing model was **YOLOv8 ______** based on the highest mAP@50-95 score.

## Repository Contents

```
Week6_YOLOv8.ipynb
best.pt
results/
README.md
requirements.txt
```

## Results

The repository includes:

- Training curves
- Confusion matrix
- Sample predictions
- Trained model weights

## How to Run

Install dependencies:

```bash
pip install ultralytics
```

Run the notebook:

```bash
jupyter notebook Week6_YOLOv8.ipynb
```

or execute it directly in Kaggle using a GPU runtime.

## Author

Arth Shivhare (24b2404)
