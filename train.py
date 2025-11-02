# train.py
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Dataset paths
# -----------------------------
train_dir = 'data/Fish/data/train'
val_dir = 'data/Fish/data/val'

# Image settings
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
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/class_mapping.json", "w") as f:
    json.dump(class_mapping, f)

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
    Dense(len(class_mapping), activation='softmax')  # dynamic number of classes
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
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# -----------------------------
# Save model in Keras format
# -----------------------------
model.save("saved_models/fish_model.h5")  # <-- use .h5 for compatibility with app.py
print("Model saved to saved_models/fish_model.h5")
