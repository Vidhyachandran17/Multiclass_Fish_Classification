# evaluate.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json, os, pandas as pd

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "saved_models/fish_model.keras"
DATA_PATH = "data/Fish/data/val"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# Load model & class mapping
# -----------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

mapping_path = os.path.join("saved_models", "class_mapping.json")
with open(mapping_path, "r") as f:
    class_names = json.load(f)

print(f"Loaded {len(class_names)} classes:", class_names)

# -----------------------------
# Prepare validation dataset
# -----------------------------
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_PATH,
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)

# -----------------------------
# Predict on validation data
# -----------------------------
print("Generating predictions...")
y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# Classification report
# -----------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(REPORT_DIR, "classification_report.csv"))
print("\nClassification report saved to reports/classification_report.csv")

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.show()

# -----------------------------
# Overall Accuracy
# -----------------------------
accuracy = np.mean(y_true == y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Display top misclassified images
# -----------------------------
import tensorflow.keras.utils as ku

# Prepare list of file paths and true labels
val_gen = ku.image_dataset_from_directory(
    DATA_PATH,
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)

file_paths = []
for batch in val_gen:
    for img in batch[0]:
        file_paths.append(img)

# Collect misclassified images
misclassified = []
true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
for i, (true, pred) in enumerate(zip(true_labels, y_pred)):
    if true != pred:
        misclassified.append((file_paths[i], true, pred))

# Display top 10 misclassified images
num_to_show = min(10, len(misclassified))
plt.figure(figsize=(15, 5))
for i in range(num_to_show):
    img_array, true, pred = misclassified[i]
    plt.subplot(2, 5, i+1)
    plt.imshow(img_array.numpy())
    plt.title(f"True: {class_names[str(true)]}\nPred: {class_names[str(pred)]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# -----------------------------
# Save misclassified images to reports folder
# -----------------------------
mis_dir = os.path.join(REPORT_DIR, "misclassified")
os.makedirs(mis_dir, exist_ok=True)

for i, (img_array, true, pred) in enumerate(misclassified[:10]):
    plt.imsave(os.path.join(mis_dir, f"mis_{i}_true_{class_names[str(true)]}_pred_{class_names[str(pred)]}.png"),
               img_array.numpy())
print(f"Top {num_to_show} misclassified images saved to {mis_dir}")
