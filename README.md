# pneumonia_detection
Pneumonia Detection Model 98% Accuracy

# Pneumonia Detection from Chest X-rays using EfficientNetB0

![Pneumonia Detection](https://img.shields.io/badge/Pneumonia-Detection-brightgreen) 
![Deep Learning](https://img.shields.io/badge/Deep-Learning-blue) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

A deep learning model for detecting pneumonia from chest X-ray images using transfer learning with EfficientNetB0.

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
This project implements a binary classification model to distinguish between:
- Normal chest X-rays
- Pneumonia-infected chest X-rays

The model achieves **98% test accuracy** using transfer learning with EfficientNetB0 as the base model.

## Dataset
The model uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

### Original Dataset Distribution
| Set | Normal | Pneumonia | Total | Normal % | Pneumonia % |
|-----|--------|-----------|-------|----------|-------------|
| Train | 1,341 | 3,875 | 5,216 | 25.7% | 74.3% |
| Val | 8 | 8 | 16 | 50.0% | 50.0% |
| Test | 234 | 390 | 624 | 37.5% | 62.5% |

### Resplit Dataset Distribution
After combining and resplitting for better balance:
| Set | Normal | Pneumonia | Total | Normal % | Pneumonia % |
|-----|--------|-----------|-------|----------|-------------|
| Train | 1,108 | 2,990 | 4,098 | 27.0% | 73.0% |
| Val | 158 | 428 | 586 | 27.0% | 73.0% |
| Test | 317 | 855 | 1,172 | 27.0% | 73.0% |

## Preprocessing Pipeline
The images undergo an advanced preprocessing pipeline:

1. **Grayscale Conversion** (if needed)
2. **Normalization** to [0,1] range
3. **Noise Reduction**:
   - Gaussian Blur (3x3 kernel)
   - Median Filtering (3x3 kernel)
4. **Contrast Enhancement**:
   - CLAHE (Clip Limit=0.03, Tile Grid=8x8)
5. **Gamma Correction** (γ=0.7)
6. **RGB Conversion** (3 channels)
7. **EfficientNet-specific preprocessing**

![Preprocessing Visualization](preprocessing_stages.png)

## Model Architecture
```python
model = Sequential([
    EfficientNetB0(include_top=False, weights='imagenet'),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
## Key Specifications:
- **Base Model**: EfficientNetB0 (pretrained on ImageNet)
- **Input Shape**: (224, 224, 3)
- **Trainable Parameters**: 4,049,249
- **Optimizer**: Adam (learning_rate=1e-4)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## Training
The model was trained for 40 epochs with:
- **Batch Size**: 32
- **Data Augmentation**:
  - Rotation (±20°)
  - Width/Height Shift (±20%)
  - Shear (±20%)
  - Zoom (±20%)
  - Horizontal Flip

## Results
### Final Evaluation Metrics:
- **Test Accuracy**: 98.08%
- **Test Loss**: 0.0574
