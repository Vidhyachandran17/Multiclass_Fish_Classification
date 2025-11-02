import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

# -----------------------------
# Load the trained model
# -----------------------------
model = load_model('saved_models/fish_model.h5')  # match your trained model

# -----------------------------
# Load class mapping dynamically
# -----------------------------
with open("saved_models/class_mapping.json", "r") as f:
    class_mapping = json.load(f)

# Convert mapping to list of classes ordered by index
class_labels = [class_mapping[str(i)] for i in range(len(class_mapping))]

# -----------------------------
# Map raw folder names to display-friendly labels
# -----------------------------
display_labels = {
    "Bream": "Bream",
    "Roach": "Roach",
    "Pike": "Pike",
    "Smelt": "Smelt",
    "Perch": "Perch",
    "Dab": "Dab",
    "animal fish": "Animal Fish"  # add any other folder names here
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fish Classifier", layout="centered")
st.title("🐟 Multi-class Fish Classification")
st.write("Upload an image of a fish to predict its class.")

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and preprocess image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        img = img.resize((128, 128))  # adjust to your model input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)

        # Safety check
        if prediction.shape[1] == len(class_labels):
            predicted_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_index]
            confidence = np.max(prediction)

            # Map to display-friendly name
            predicted_class_display = display_labels.get(predicted_class, predicted_class)

            st.success(f"Predicted Class: **{predicted_class_display}** ({confidence*100:.2f}% confidence)")
        else:
            st.error(f"Prediction shape mismatch! Expected {len(class_labels)} classes, got {prediction.shape[1]}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
