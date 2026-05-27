# Skin Cancer Classification System (AI/ML)

An end-to-end Machine Learning pipeline to classify medical skin lesion images as Benign or Malignant using Deep Learning.

## Project Overview
This project leverages **DenseNet121** via Transfer Learning to assist in dermatological diagnostics. By processing clinical medical images, the model achieves ~92% classification accuracy.

## Pipeline Architecture


## Key Features
* **Transfer Learning:** Utilized pre-trained DenseNet121 weights (ImageNet) for high-feature extraction efficiency.
* **Data Augmentation:** Implemented `ImageDataGenerator` with rotation, zoom, and flipping to improve model generalization.
* **Performance Metrics:** Evaluated using Confusion Matrix and Classification Reports (Sensitivity/Specificity analysis).

## Results
| Metric | Value |
| :--- | :--- |
| **Accuracy** | ~92% |
| **Model** | DenseNet121 |
| **Input Size** | 224x224 RGB |

## Getting Started
1. Clone this repo: `git clone https://github.com/IT24101183/Skin-Cancer-Classification-DenseNet121.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python Skin_Cancer_Detection_DenseNet121.py`

*Developed as a University Project at SLIIT.*
