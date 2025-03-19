import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Define Google Drive file ID and output file path
file_id = "1jyNIdLXLQg5_EjK0gjLgt2fABnMe-lYh"  # Your file ID
model_path = "solar_agro_dryer_model.h5"

# Download the model if not already present
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    st.write("Downloading model... Please wait.")
    gdown.download(url, model_path, quiet=False)

# Load the model
st.write("Loading model...")
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ["Dried Apples", "Dried Bananas", "Dried Tomatoes", 
                "Undried Apples", "Undried Bananas", "Undried Tomatoes"]

st.title("Solar Agro Dryer - Image Classification")
st.write("Upload an image to classify whether it is dried or undried.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Adjust to your model's input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]
    
    st.write(f"### Prediction: {predicted_class}")
