# Pneumonia Detection from Chest X-Rays using EfficientNetB0 🏥🩺

![Pneumonia Detection](https://img.shields.io/badge/Pneumonia-Detection-brightgreen)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.08%25-success)

A state-of-the-art deep learning model for detecting pneumonia from chest X-ray images with **98.08% accuracy**, leveraging transfer learning with EfficientNetB0.

## Table of Contents
- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Model Overview
This project implements a high-performance binary classification model to distinguish between:
- Normal chest X-rays (Healthy)
- Pneumonia-infected chest X-rays

**Key Achievements:**
- Achieves **98.08% test accuracy**
- Utilizes EfficientNetB0 with custom preprocessing
- Balanced dataset through strategic resampling
- Comprehensive image augmentation pipeline

## Dataset
The model uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

### Dataset Distribution

#### Original Distribution
| Set       | Normal | Pneumonia | Total | Normal % | Pneumonia % |
|-----------|--------|-----------|-------|----------|-------------|
| Train     | 1,341  | 3,875     | 5,216 | 25.7%    | 74.3%       |
| Val       | 8      | 8         | 16    | 50.0%    | 50.0%       |
| Test      | 234    | 390       | 624   | 37.5%    | 62.5%       |

#### Balanced Distribution (After Resplitting)
| Set       | Normal | Pneumonia | Total | Normal % | Pneumonia % |
|-----------|--------|-----------|-------|----------|-------------|
| Train     | 1,108  | 2,990     | 4,098 | 27.0%    | 73.0%       |
| Val       | 158    | 428       | 586   | 27.0%    | 73.0%       |
| Test      | 317    | 855       | 1,172 | 27.0%    | 73.0%       |

## Preprocessing Pipeline
Our advanced preprocessing pipeline ensures optimal image quality for the model:

1. **Grayscale Conversion** (when needed)
2. **Normalization** (0-1 range)
3. **Noise Reduction**:
   - Gaussian Blur (3×3 kernel)
   - Median Filtering (3×3 kernel)
4. **Contrast Enhancement**:
   - CLAHE (Clip Limit=0.03, 8×8 grid)
5. **Gamma Correction** (γ=0.7)
6. **RGB Conversion** (3 channels)
7. **EfficientNet-specific preprocessing**

![Preprocessing Visualization Example](preprocessing_stages.png)

## Model Architecture

```python
model = tf.keras.Sequential([
    EfficientNetB0(include_top=False, weights='imagenet'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])
```

## Model Specifications

### Key Specifications

| Component            | Specification                          |
|----------------------|----------------------------------------|
| **Base Model**       | EfficientNetB0 (ImageNet pretrained)   |
| **Input Shape**      | 224×224×3 (RGB)                         |
| **Trainable Params** | 4,049,249                               |
| **Optimizer**        | Adam (lr=1e-4)                          |
| **Loss Function**    | Binary Crossentropy                     |
| **Metrics**          | Accuracy                                |

## Training Configuration

The model was carefully trained with the following parameters:

- **Training Duration**: 40 epochs  
- **Batch Size**: 32 samples  
- **Data Augmentation**:
  - 🔄 **Rotation**: ±20°
  - ↔️ **Width/Height Shift**: ±20% of total dimension
  - ✂️ **Shear**: ±20% intensity
  - 🔍 **Zoom**: ±20% magnification
  - ↔️ **Horizontal Flip**: Randomly applied

## Performance Results

### Evaluation Metrics

| Metric             | Score    |
|--------------------|----------|
| **Test Accuracy**  | 98.08%   |
| **Test Loss**      | 0.0574   |

