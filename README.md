<div align="center">

<!-- IMAGE 1: Banner — upload a project banner/cover image to Cloudinary and replace the URL below -->
<img src="https://res.cloudinary.com/YOUR_CLOUD_NAME/image/upload/YOUR_BANNER_IMAGE.png"
     alt="Skin Cancer Classification Banner"
     width="100%"/>

# Skin Cancer Classification System
### Deep Learning · DenseNet-121 · Transfer Learning · Binary Classification

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-DenseNet121-D00000?style=flat&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![University](https://img.shields.io/badge/SLIIT-University%20Project-blue?style=flat)

An end-to-end deep learning pipeline to classify skin lesion images as **Benign** or **Malignant**,
achieving approximately **~92% test accuracy** using transfer learning on DenseNet-121.

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)

---

## Project Overview

Skin cancer is among the most prevalent and life-threatening cancers worldwide.
Early and accurate detection significantly improves patient outcomes.
This project applies **transfer learning** with a **DenseNet-121** backbone (pre-trained on ImageNet)
to perform binary classification of clinical skin lesion images.

**Key design decisions:**

| Component | Choice | Reason |
| :--- | :--- | :--- |
| Base model | DenseNet-121 | Strong feature reuse, compact architecture |
| Optimiser | Adam (`lr=1e-4`) | Stable fine-tuning on small medical datasets |
| Augmentation | Rotation, zoom, flip, shear | Reduces overfitting on limited data |
| Early stopping | `patience=3` on val_loss | Prevents over-training |

---

## Pipeline Architecture

```
Raw Images
    │
    ▼
ImageDataGenerator  ──── Augmentation (train only)
    │                    Normalisation (all splits)
    ▼
DenseNet-121 (ImageNet weights, trainable)
    │
GlobalAveragePooling2D
    │
Dense(128, ReLU)  →  Dropout(0.5)
    │
Dense(1, Sigmoid)
    │
    ▼
Prediction: Benign (0) / Malignant (1)
```

---

## Dataset

Dataset sourced from [Roboflow](https://roboflow.com/) — *Skin Cancer Detection v1*.

<!-- IMAGE 2: Benign sample — upload a sample benign lesion image to Cloudinary and replace the URL below -->
<!-- IMAGE 3: Malignant sample — upload a sample malignant lesion image to Cloudinary and replace the URL below -->

<div align="center">

| Benign Sample | Malignant Sample |
| :---: | :---: |
| <img src="https://res.cloudinary.com/YOUR_CLOUD_NAME/image/upload/YOUR_BENIGN_SAMPLE.jpg" alt="Benign lesion sample" width="300"/> | <img src="https://res.cloudinary.com/YOUR_CLOUD_NAME/image/upload/YOUR_MALIGNANT_SAMPLE.jpg" alt="Malignant lesion sample" width="300"/> |
| Benign (No Cancer) | Malignant (Cancer Detected) |

</div>

**Data splits:**

| Split | Images |
| :--- | :--- |
| Train | ~80% |
| Validation | ~10% |
| Test | ~10% |

Data augmentation applied during training: `rotation_range=20`, `zoom_range=0.2`,
`horizontal_flip=True`, `width/height_shift=0.2`, `shear_range=0.2`.

---

## Results

### Performance Summary

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | ~92% |
| **Model** | DenseNet-121 |
| **Input Size** | 224 × 224 RGB |
| **Loss Function** | Binary Cross-Entropy |
| **Optimiser** | Adam (lr = 0.0001) |

### Confusion Matrix

<!-- IMAGE 4: Confusion matrix — export and upload your confusion matrix plot to Cloudinary, then replace the URL below -->

<div align="center">
<img src="https://res.cloudinary.com/YOUR_CLOUD_NAME/image/upload/YOUR_CONFUSION_MATRIX.png"
     alt="Confusion Matrix"
     width="500"/>
</div>

### Classification Metrics

| Class | Precision | Recall (Sensitivity) | F1-Score |
| :--- | :---: | :---: | :---: |
| Benign | — | — | — |
| Malignant | — | — | — |

> Fill in your actual classification report values after training.

### Sample Prediction Output

<!-- IMAGE 5: Benign prediction — upload a prediction result screenshot (benign case) to Cloudinary -->
<!-- IMAGE 6: Malignant prediction — upload a prediction result screenshot (malignant case) to Cloudinary -->

<div align="center">

| Benign Prediction | Malignant Prediction |
| :---: | :---: |
| <img src="https://res.cloudinary.com/YOUR_CLOUD_NAME/image/upload/YOUR_PREDICTION_BENIGN.png" alt="Benign prediction output" width="380"/> | <img src="https://res.cloudinary.com/YOUR_CLOUD_NAME/image/upload/YOUR_PREDICTION_MALIGNANT.png" alt="Malignant prediction output" width="380"/> |

</div>

---

## Getting Started

### Prerequisites

- Python 3.10+
- A Roboflow account and API key
- (Recommended) Google Colab with GPU runtime

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/IT24101183/Skin-Cancer-Classification-DenseNet121.git
cd Skin-Cancer-Classification-DenseNet121

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Roboflow API key as an environment variable
export ROBOFLOW_API_KEY="your_api_key_here"

# 4. Run the script
python skin_cancer_detection_densenet121.py
```

### Running on Google Colab

1. Open the script in [Google Colab](https://colab.research.google.com/).
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**.
3. Add your Roboflow API key to Colab Secrets (key name: `ROBOFLOW_API_KEY`).
4. Run all cells sequentially.

---

## Project Structure

```
Skin-Cancer-Classification-DenseNet121/
│
├── skin_cancer_detection_densenet121.py   # Main training and evaluation script
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

### Dependencies (`requirements.txt`)

```
tensorflow>=2.12.0
roboflow
scikit-learn
matplotlib
Pillow
numpy
```

---

## Disclaimer

> This project is developed for **academic and research purposes only**.
> The model's predictions must **not** be used as a substitute for professional medical diagnosis.
> Always consult a qualified dermatologist for any clinical concerns.

---

<div align="center">

Developed as a University Project at **[SLIIT](https://www.sliit.lk/)** · Sri Lanka Institute of Information Technology

</div>
