Multiclass Fish Classification
==============================

A Streamlit web app allows users to upload an image and get predictions with confidence scores.

Project Overview
----------------
- Objective: Build a deep learning model to classify multiple fish species from images.
- Input: Fish image uploaded via the Streamlit app.
- Output: Predicted fish species with confidence percentage.

Technologies Used
-----------------
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Streamlit
- Pillow

Project Structure
-----------------
Multiclass_Fish_Classification/
|
├─ saved_models/         # Trained Keras model files
├─ src/                  # Source code scripts
├─ data/                 # Dataset files
├─ app.py                # Streamlit web app
├─ train.py              # Model training script
├─ visualize_results.py  # Data visualization script
├─ README.txt            # Project documentation
└─ venv/                 # Virtual environment

Installation
------------
1. Clone the repository:
   git clone https://github.com/Vidhyachandran17/Multiclass_Fish_Classification.git

2. Navigate to the project folder:
   cd Multiclass_Fish_Classification

3. Create and activate a virtual environment:
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt

Usage
-----
1. Run the Streamlit app:
   streamlit run app.py

2. Upload an image of a fish.
3. View the predicted fish species and confidence score.

Model Details
-------------
- Built using Convolutional Neural Networks (CNNs)
- Trained on a dataset of multiple fish species
- Evaluated on accuracy and prediction confidence

Future Improvements
-------------------
- Add support for more fish species
- Improve model accuracy with data augmentation
- Deploy as a publicly accessible web app










