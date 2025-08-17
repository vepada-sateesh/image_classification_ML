import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("cat_not_cat.h5")

st.title("🐱 Cat vs Not-Cat Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((64, 64))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    prediction = model.predict(img_array)[0][0]
    label = "Cat 🐱" if prediction > 0.5 else "Not Cat 🐯"

    st.write(f"Prediction: **{label}** (confidence: {prediction:.2f})")
