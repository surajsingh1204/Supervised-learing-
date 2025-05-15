import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Crop Disease Detector", layout="centered")
st.title("Crop Leaf Disease Classifier")
st.markdown("Upload a potato leaf image to classify it as Early Blight, Late Blight, or Healthy.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("crop_disease_detection_model.h5")  # Model file name
    return model

model = load_model()

# Class labels (according to training)
classes = ['Early Blight disease', 'Late Blight disease', 'Healthy']

# File uploader
uploaded_file = st.file_uploader("Upload an Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    IMG_SIZE = 256
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions)
    predicted_class = classes[predicted_index]
    confidence = float(np.max(predictions))

    # Display results
    st.markdown("---")
    st.subheader("Prediction Result")
    st.success(f"**Prediction :** `{predicted_class}`")
else:
    st.info("Please upload a potato leaf image to classify.")