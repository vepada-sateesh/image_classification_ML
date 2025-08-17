# -------------------------------
# Cat vs Not-Cat Classifier (Streamlit)
# -------------------------------

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ----------- STEP 1: Load Model -----------
MODEL_PATH = "cat_classifier_cnn.h5"

@st.cache_resource  # cache the model so it doesn't reload every time
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ----------- STEP 2: Streamlit UI -----------
st.set_page_config(page_title="Cat Classifier", page_icon="🐱")
st.title("🐱 Cat vs Not-Cat Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load & preprocess image
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)

    # ----------- STEP 3: Prediction -----------
    prediction = model.predict(img_array)[0][0]
    label = "Cat 🐱" if prediction >= 0.5 else "Not a Cat 🚫"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.subheader("Prediction Result")
    st.write(f"**{label}**")
    st.write(f"Confidence: `{confidence * 100:.2f}%`")
