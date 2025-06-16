import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
import numpy as np
from src.prediction.predictor import Predictor

# Load the trained model
@st.cache_resource
def load_predictor():
    return Predictor('models/full_model.keras')

predictor = load_predictor()

st.title("Lixo Intelligente - Classificação de dejetos ")
st.write("Carregue uma imagem para classificar se é reciclável ou não.")

# Upload image
uploaded_file = st.file_uploader("Escolha uma Imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert('RGB')  # Ensure RGB format
    st.image(image, caption='Imagem carregada.', use_container_width=True)
    st.write("")

    # Preprocess the image for prediction
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = predictor.model.predict(image_array)[0][0]  # Probabilité (sigmoid)

    # Display prediction result
    if prediction >= 0.5:
        st.success(f"Essa imagem é reciclável. ({prediction:.2%} de certeza)")
    else:
        st.error(f"Essa imagem NÃO é reciclável. ({(1 - prediction):.2%} de certeza)")
    st.write("Classificação baseada no modelo treinado.")
else:
    st.warning("Por favor, carregue uma imagem para classificar.")
# Add a footer
st.markdown("""
    <style>
        footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #b0c4de;
            text-align: center;
        }
        footer p {
            margin: 0;
            padding: 10px;
            color: #000080;
        }
        footer p:hover {
            color: #4682b4;
        }
    </style>
    <footer>
        <p >Desenvolvido por [KOKOUVI & DALTON] - Lixo Intelligente</p>
    </footer>
""", unsafe_allow_html=True)