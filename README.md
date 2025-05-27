# Pneumonia Detection from Chest X-Rays using ResNet50 🏥🩺

![Pneumonia Detection](https://img.shields.io/badge/Pneumonia-Detection-brightgreen)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.08%25-success)

A high-performance deep learning model for detecting pneumonia from chest X-ray images using **ResNet50** and a robust custom preprocessing pipeline. Achieves **98.08% accuracy** on the test set.

---

## 📚 Table of Contents
- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

---

## 🧠 Model Overview

This project presents a binary classification model that differentiates between:
- ✅ Normal (healthy) chest X-rays
- ❌ Pneumonia-infected chest X-rays

### ✅ Highlights
- Achieves **97.87% test accuracy**
- Uses **ResNet50** pretrained on ImageNet for feature extraction
- Applies custom image preprocessing tailored for medical images
- Employs a stratified and balanced data split strategy
- Integrates comprehensive image augmentation to improve generalization

---

## 📁 Dataset

The model is trained on the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

### 🔎 Dataset Distribution

#### 📉 Original
| Set   | Normal | Pneumonia | Total | Normal % | Pneumonia % |
|--------|--------|-----------|--------|-----------|--------------|
| Train  | 1,341  | 3,875     | 5,216  | 25.7%     | 74.3%        |
| Val    | 8      | 8         | 16     | 50.0%     | 50.0%        |
| Test   | 234    | 390       | 624    | 37.5%     | 62.5%        |

#### ⚖️ Balanced (After Resplitting)
| Set   | Normal | Pneumonia | Total | Normal % | Pneumonia % |
|--------|--------|-----------|--------|-----------|--------------|
| Train  | 1,108  | 2,990     | 4,098  | 27.0%     | 73.0%        |
| Val    | 158    | 428       | 586    | 27.0%     | 73.0%        |
| Test   | 317    | 855       | 1,172  | 27.0%     | 73.0%        |

---

## 🧼 Preprocessing Pipeline

Custom preprocessing ensures enhanced image quality and consistency:

1. **Grayscale Conversion** (if required)
2. **Min-Max Normalization** to [0, 1]
3. **Noise Reduction**:
   - Gaussian Blur (3×3)
   - Median Filter (3×3)
4. **Contrast Enhancement**:
   - CLAHE (Clip Limit = 0.03, Tile Grid = 8×8)
5. **Gamma Correction** (γ = 0.7)
6. **RGB Conversion** for compatibility
7. **ResNet50 Preprocessing** via `tf.keras.applications.resnet50.preprocess_input`

![Preprocessing Visualization](preprocessing_stages.png)

---

## 🏗️ Model Architecture

```python
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])


## ⚙️ Model Specifications

| **Component**        | **Specification**                 |
|----------------------|-----------------------------------|
| Base Model           | ResNet50 (ImageNet pretrained)    |
| Input Shape          | 224×224×3                         |
| Trainable Params     | 23,558,785                        |
| Optimizer            | Adam (lr=1e-4)                    |
| Loss Function        | Binary Crossentropy               |
| Metrics              | Accuracy                          |

---

## 🏋️‍♂️ Training Configuration

| **Parameter**        | **Value**             |
|----------------------|-----------------------|
| Epochs               | 40                    |
| Batch Size           | 32                    |
| Learning Rate        | 0.0001                |
| Validation Split     | Custom stratified     |

---

## 🧪 Data Augmentation

- 🔄 **Rotation**: ±20°
- ↔️ **Width/Height Shift**: ±20%
- ✂️ **Shear Intensity**: ±20%
- 🔍 **Zoom**: ±20%
- ↔️ **Horizontal Flip**: Random

---

## 📊 Performance Results

**Final Evaluation on Test Set:**

| **Metric**           | **Value**  |
|----------------------|------------|
| Accuracy             | 97.87%     |
| Loss                 | 0.076     |

