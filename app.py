import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("deepfake_detector_model.keras")

# Page settings
st.set_page_config(page_title="Deepfake Detection", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
        }
        .main {
            padding-top: 2rem;
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .subtitle {
            font-size: 1.2em;
            color: #555;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<div class='title'>🧠 Deepfake Detection App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a face image to check if it's REAL or FAKE</div>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file is not None:
    with st.spinner("⏳ Analyzing image..."):
        img = Image.open(uploaded_file).convert("RGB").resize((128, 128))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

    st.markdown("---")
    if prediction[0][0] > 0.5:
        st.error("❌ This is a FAKE Face", icon="🚫")
    else:
        st.success("✅ This is a REAL Face", icon="✅")

    st.markdown("Prediction complete. You can try another image above.")
else:
    st.info("Please upload a JPG or PNG image to begin.", icon="ℹ️")