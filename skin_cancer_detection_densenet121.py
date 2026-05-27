"""
Skin Cancer Detection using DenseNet-121
=========================================
Binary classification of skin lesion images (Benign vs Malignant)
using transfer learning with a pre-trained DenseNet-121 backbone.

Architecture   : DenseNet-121 (ImageNet weights) + custom classifier head
Framework      : TensorFlow / Keras
Dataset        : Roboflow - Skin Cancer Detection dataset
Task           : Binary classification (Benign / Malignant)
"""

# ============================================================
# BLOCK 1: INSTALL LIBRARIES & IMPORTS
# ============================================================

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

# Install Roboflow if not already installed
# Run: pip install roboflow
from roboflow import Roboflow

print("Libraries imported successfully.")
print(f"TensorFlow version: {tf.__version__}")

# Check GPU availability
gpu_available = tf.config.list_physical_devices('GPU')
if gpu_available:
    print(f"GPU available: {gpu_available}")
else:
    print("No GPU detected. Running on CPU (training will be slower).")


# ============================================================
# BLOCK 2: DOWNLOAD DATASET FROM ROBOFLOW
# ============================================================

# Load API key from environment variable for security.
# Set this before running: export ROBOFLOW_API_KEY="your_api_key_here"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise EnvironmentError(
        "ROBOFLOW_API_KEY environment variable is not set. "
        "Please set it before running this script."
    )

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

project = rf.workspace("test-rdtzf").project("skin-cancer-detection-zvlv0")
version = project.version(1)
dataset = version.download("folder")

print(f"Dataset downloaded successfully. Location: {dataset.location}")


# ============================================================
# BLOCK 3: VERIFY DATASET STRUCTURE
# ============================================================

BASE_PATH = '/content/Skin-Cancer-Detection-1'

print("=" * 60)
print("DATASET STRUCTURE AND STATISTICS")
print("=" * 60)

for split in ['train', 'valid', 'test']:
    split_path = os.path.join(BASE_PATH, split)

    if os.path.exists(split_path):
        print(f"\n{split.upper()} SET:")
        print("-" * 40)

        classes = os.listdir(split_path)
        print(f"  Classes: {classes}")

        total_images = 0
        for cls in classes:
            class_path = os.path.join(split_path, cls)
            img_count = len(os.listdir(class_path))
            print(f"  {cls}: {img_count} images")
            total_images += img_count

        print(f"  Total: {total_images} images")
    else:
        print(f"\n[WARNING] '{split}' folder not found at: {split_path}")

print("\n" + "=" * 60)
print("Dataset verification complete.")


# ============================================================
# BLOCK 4: SETUP DATA GENERATORS
# ============================================================

IMG_SIZE = (224, 224)   # Required input size for DenseNet-121
BATCH_SIZE = 16         # Adjust based on available GPU memory

print("Configuring data generators...")

# Training generator with data augmentation to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

# Validation and test generators — normalization only, no augmentation
val_datagen  = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(BASE_PATH, 'valid'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_PATH, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Data generators configured.")
print(f"  Training samples  : {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples      : {test_generator.samples}")
print(f"  Class indices     : {train_generator.class_indices}")


# ============================================================
# BLOCK 5: BUILD DENSENET-121 MODEL
# ============================================================

print("Building DenseNet-121 model with transfer learning...")

model = Sequential([
    # DenseNet-121 base pre-trained on ImageNet.
    # include_top=False removes the original classification head.
    DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    ),

    # Reduce spatial dimensions to a single feature vector
    GlobalAveragePooling2D(),

    # Custom classification head for binary skin cancer detection
    Dense(128, activation='relu'),
    Dropout(0.5),                   # Regularisation to reduce overfitting
    Dense(1, activation='sigmoid')  # Output: probability of malignancy
])

# Allow all DenseNet-121 layers to be updated during fine-tuning
model.layers[0].trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"\nModel built successfully. Total parameters: {model.count_params():,}")


# ============================================================
# BLOCK 6: TRAIN THE MODEL
# ============================================================

print("Starting model training...")

# Early stopping monitors validation loss and restores the best weights
# if no improvement is observed for the specified number of epochs.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

train_steps = math.ceil(train_generator.samples / BATCH_SIZE)
val_steps   = math.ceil(val_generator.samples / BATCH_SIZE)

print(f"  Training steps per epoch  : {train_steps}")
print(f"  Validation steps per epoch: {val_steps}")
print(f"  Batch size                : {BATCH_SIZE}\n")

history = model.fit(
    train_generator,
    steps_per_epoch=None,
    epochs=20,
    validation_data=val_generator,
    validation_steps=None,
    callbacks=[early_stopping],
    verbose=1
)

print("\nTraining complete.")

# --- Plot training history ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],     label='Training Accuracy',   marker='o')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
axes[0].set_title('Model Accuracy Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Training Loss',   marker='o')
axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
axes[1].set_title('Model Loss Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Summary ---
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"Epochs trained           : {len(history.history['accuracy'])}")
print(f"Best training accuracy   : {max(history.history['accuracy']) * 100:.2f}%")
print(f"Best validation accuracy : {max(history.history['val_accuracy']) * 100:.2f}%")
print(f"Final training loss      : {history.history['loss'][-1]:.4f}")
print(f"Final validation loss    : {history.history['val_loss'][-1]:.4f}")
print("=" * 60)


# ============================================================
# BLOCK 7: EVALUATE MODEL PERFORMANCE
# ============================================================

print("Evaluating model on the test set...\n")

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# Generate predictions
print("\nGenerating predictions on the test set...")
test_generator.reset()
predictions      = model.predict(test_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes      = test_generator.classes
class_names       = list(test_generator.class_indices.keys())

# Classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(true_classes, predicted_classes, target_names=class_names, digits=4))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
tn, fp, fn, tp = cm.ravel()

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(f"\n{'':15} Predicted")
print(f"{'':10} Benign  Malignant")
print(f"Actual Benign    {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"       Malignant {cm[1][0]:4d}    {cm[1][1]:4d}")

print("\n" + "=" * 60)
print("DETAILED METRICS")
print("=" * 60)
print(f"True Negatives  (Benign correctly identified)    : {tn}")
print(f"True Positives  (Malignant correctly identified) : {tp}")
print(f"False Positives (Benign predicted as Malignant)  : {fp}")
print(f"False Negatives (Malignant predicted as Benign)  : {fn}")
print(f"\nSensitivity (Recall) : {tp / (tp + fn) * 100:.2f}%")
print(f"Specificity          : {tn / (tn + fp) * 100:.2f}%")

# Visualise confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix — DenseNet-121')
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("\nEvaluation complete.")


# ============================================================
# BLOCK 8: SAVE THE TRAINED MODEL
# ============================================================

print("Saving trained model...\n")

# Save in HDF5 format
model_path = '/content/skin_cancer_densenet121.h5'
model.save(model_path)
print(f"Model saved (HDF5)  : {model_path}")

# Save in native Keras format (recommended for TensorFlow >= 2.12)
keras_model_path = '/content/skin_cancer_densenet121.keras'
model.save(keras_model_path)
print(f"Model saved (Keras) : {keras_model_path}")

print("\nTo download: right-click the file in the Colab file browser and select 'Download'.")


# ============================================================
# BLOCK 9: PREDICT ON A SINGLE IMAGE (COLAB)
# ============================================================

from google.colab import files
from IPython.display import display


def predict_skin_cancer(image_path: str, display_image: bool = True) -> tuple[str, float]:
    """
    Predict whether a skin lesion image is benign or malignant.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    display_image : bool, optional
        Whether to render the image and prediction result (default: True).

    Returns
    -------
    prediction_label : str
        'Benign (No Cancer)' or 'Malignant (Cancer Detected)'.
    confidence : float
        Model confidence as a percentage.
    """
    img       = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    raw_prediction = model.predict(img_array, verbose=0)[0][0]

    if raw_prediction > 0.5:
        label      = "Malignant (Cancer Detected)"
        confidence = raw_prediction * 100
        color      = 'red'
    else:
        label      = "Benign (No Cancer)"
        confidence = (1 - raw_prediction) * 100
        color      = 'green'

    if display_image:
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.6, label,       fontsize=20, fontweight='bold', ha='center', color=color)
        plt.text(0.5, 0.4, f'Confidence: {confidence:.2f}%', fontsize=16, ha='center')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Prediction Result')

        plt.tight_layout()
        plt.show()

    return label, confidence


print("Upload a skin lesion image to run inference.\n")
print("=" * 60)

uploaded = files.upload()

if uploaded:
    image_path = list(uploaded.keys())[0]
    print(f"\nImage uploaded: {image_path}")
    print("\nRunning inference...")
    print("=" * 60)

    label, confidence = predict_skin_cancer(image_path, display_image=True)

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Diagnosis  : {label}")
    print(f"Confidence : {confidence:.2f}%")
    print("=" * 60)

    if "Malignant" in label:
        print("\nWARNING: The model predicts potential malignancy.")
        print("Please consult a qualified dermatologist promptly.")
    else:
        print("\nThe model predicts the lesion is benign.")
        print("Regular dermatological check-ups are still advisable.")

    print("\nDISCLAIMER: This output is generated by an AI model and must not")
    print("replace or delay a professional medical diagnosis.")
else:
    print("No image was uploaded.")


# ============================================================
# BLOCK 10: DOWNLOAD THE TRAINED MODEL (COLAB)
# ============================================================

model.save('skin_cancer_densenet121_model.h5')
files.download('skin_cancer_densenet121_model.h5')

print("Model file downloaded: skin_cancer_densenet121_model.h5")
