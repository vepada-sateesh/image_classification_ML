# -------------------------------
# Cat vs Not-Cat Classifier (Deployed Version)
# -------------------------------

from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)

# ----------- STEP 1: Load Pretrained Model -----------
model_file = "cat_classifier_cnn.h5"

if os.path.exists(model_file):
    print("📦 Loading trained model...")
    model = load_model(model_file)
else:
    raise FileNotFoundError("❌ Trained model file not found. Make sure cat_classifier_cnn.h5 is in the project folder.")

# ----------- STEP 2: Define Flask Routes -----------

@app.route("/")
def index():
    return render_template("index.html")  # Optional: you can remove this if no HTML UI

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle image upload, preprocess, and predict using CNN.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Preprocess uploaded image
        try:
            img = Image.open(file).convert('RGB').resize((128, 128))  # 👈 match model input
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

        # Predict
        prediction = model.predict(img_array)[0][0]
        label = "Cat 🐱" if prediction >= 0.5 else "Not a Cat 🚫"
        confidence = f"{prediction * 100:.2f}%" if prediction >= 0.5 else f"{(1 - prediction) * 100:.2f}%"

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        print(f"🔥 Backend error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ----------- STEP 3: Run Flask App -----------

if __name__ == "__main__":
    app.run(debug=True, port=9999)
