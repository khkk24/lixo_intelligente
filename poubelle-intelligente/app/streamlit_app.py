import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from src.prediction.predictor import Predictor

# Load the trained model
model_path = 'models/full_model.keras'
# model_path = 'models/model_cnn.h5'
predictor = Predictor(model_path)

st.title("Poubelle Intelligente - Classification des Déchets")
st.write("Téléchargez une image pour savoir si elle est recyclable ou non.")

# Upload image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée.', use_column_width=True)
    st.write("")
    
    # Preprocess the image for prediction
    image = image.resize((224, 224))  # Resize to the input size of the model
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = predictor.predict(image_array)
    
    # Display prediction result
    if prediction == 1:
        st.success("Cette image est recyclable.")
    else:
        st.error("Cette image n'est pas recyclable.")
