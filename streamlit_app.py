import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Constants
IMG_SIZE = 64
MODEL_PATH = "model_v2.keras"

# Load the trained model
@st.cache_resource
def load_model_once():
    return load_model(MODEL_PATH)

model = load_model_once()

# Prediction function
def predict_alphabet(image: Image.Image):
    # Convert to grayscale
    img = image.convert("L")
    # Resize to match model input
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # Normalize and reshape
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # (64, 64, 1)
    img = np.expand_dims(img, axis=0)   # (1, 64, 64, 1)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_char = chr(predicted_class + 65)  # 0 -> A, 1 -> B, ...
    return predicted_char

# Streamlit UI
st.title("🅰️ Alphabet Predictor (A-Z)")
st.markdown("Upload an image containing a handwritten or printed English alphabet, and get the predicted letter.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=200)

    if st.button("Predict"):
        result = predict_alphabet(image)
        st.success(f"**Predicted Alphabet: {result}**")
