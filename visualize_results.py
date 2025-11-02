import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# -----------------------------
# Load the trained model
# -----------------------------
model_path = "saved_models/fish_model.keras"
model = load_model(model_path)

# -----------------------------
# Load validation dataset
# -----------------------------
val_dir = "data/Fish/data/val"  # update if your validation path differs
img_size = (128, 128)
batch_size = 32

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# -----------------------------
# Show a few predictions
# -----------------------------
x_batch, y_batch = next(val_generator)
predictions = model.predict(x_batch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_batch, axis=1)
class_labels = list(val_generator.class_indices.keys())

plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_batch[i])
    plt.title(f"Pred: {class_labels[predicted_classes[i]]}\nTrue: {class_labels[true_classes[i]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
