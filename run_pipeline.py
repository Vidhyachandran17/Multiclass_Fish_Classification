# run_pipeline.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Paths
# -----------------------------
train_dir = 'data/Fish/data/train'
val_dir = 'data/Fish/data/val'
REPORT_DIR = 'reports'
MODEL_DIR = 'saved_models'
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Image settings
# -----------------------------
img_size = (128, 128)
batch_size = 32
epochs = 10

# -----------------------------
# Data generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_ds = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# Save class mapping
# -----------------------------
class_mapping = {v: k for k, v in train_ds.class_indices.items()}
with open(os.path.join(MODEL_DIR, "class_mapping.json"), "w") as f:
    json.dump(class_mapping, f)

# Convert mapping to int keys for safe access
class_mapping_int = {int(k): v for k, v in class_mapping.items()}

# -----------------------------
# Build model
# -----------------------------
model = Sequential([
    tf.keras.layers.InputLayer(input_shape=(128, 128, 3), name="input_layer"),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_mapping_int), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Train model
# -----------------------------
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# -----------------------------
# Save model
# -----------------------------
MODEL_PATH = os.path.join(MODEL_DIR, "fish_model.keras")
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# -----------------------------
# Plot training metrics
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "training_metrics.png"))
plt.show()
print("Training metrics plot saved to reports/training_metrics.png")

# -----------------------------
# Evaluate model
# -----------------------------
print("\nEvaluating model...")

# True labels
y_true = val_ds.classes  # Numpy array of true labels

# Predictions
steps = val_ds.samples // val_ds.batch_size
if val_ds.samples % val_ds.batch_size != 0:
    steps += 1

y_pred_probs = model.predict(val_ds, steps=steps)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=list(class_mapping_int.values()),
    output_dict=True
)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(REPORT_DIR, "classification_report.csv"))
print("Classification report saved.")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(class_mapping_int.values()),
            yticklabels=list(class_mapping_int.values()))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.show()
print("Confusion matrix saved.")

# Overall accuracy
accuracy = np.mean(y_true == y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Top misclassified images
# -----------------------------
misclassified = []
for i, (img_batch, label_batch) in enumerate(val_ds):
    for j in range(len(img_batch)):
        true_label = np.argmax(label_batch[j])
        pred_label = np.argmax(model.predict(img_batch[j:j+1]))
        if true_label != pred_label:
            misclassified.append((img_batch[j], true_label, pred_label))
    if (i+1) >= steps:
        break

num_to_show = min(10, len(misclassified))
plt.figure(figsize=(15, 5))
for i in range(num_to_show):
    img_array, true, pred = misclassified[i]
    plt.subplot(2, 5, i+1)
    plt.imshow(img_array)
    plt.title(f"True: {class_mapping_int[true]}\nPred: {class_mapping_int[pred]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Save misclassified images
mis_dir = os.path.join(REPORT_DIR, "misclassified")
os.makedirs(mis_dir, exist_ok=True)
for i, (img_array, true, pred) in enumerate(misclassified[:10]):
    plt.imsave(os.path.join(
        mis_dir, f"mis_{i}_true_{class_mapping_int[true]}_pred_{class_mapping_int[pred]}.png"),
        img_array)
print(f"Top {num_to_show} misclassified images saved to {mis_dir}")
