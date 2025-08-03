# Import libraries
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# ----------- STEP 1: Load and Preprocess Raw Images -----------

def load_images(folder, label, size=(64, 64)):
    """
    Load images from a folder, resize to 'size', flatten them into 1D vectors,
    and assign the given label (1=cat, 0=not-cat).
    """
    data = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            # Open image, convert to RGB, resize, flatten to vector
            img = Image.open(filepath).convert('RGB').resize(size)
            img_array = np.array(img).flatten() / 255.0  # Normalize pixel values
            data.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
    return data, labels

# Paths to dataset folders
cats_folder = "dataset/cats"
not_cats_folder = "dataset/not_cats"

# Load images and labels
cats_data, cats_labels = load_images(cats_folder, 1)
not_cats_data, not_cats_labels = load_images(not_cats_folder, 0)

# Combine cats and not-cats data
X = np.array(cats_data + not_cats_data)
y = np.array(cats_labels + not_cats_labels)

print(f"✅ Loaded {len(X)} images: {sum(y)} cats, {len(y) - sum(y)} not-cats.")

# ----------- STEP 2: Train Logistic Regression Model -----------

# Scale features (standardize pixel values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset for training/testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"🎯 Model trained. Test accuracy: {accuracy * 100:.2f}%")

# ----------- STEP 3: Define Flask Routes -----------

@app.route("/")
def index():
    # Render frontend HTML
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle image upload from frontend, preprocess it, and predict if it's a cat or not.
    """
    print("yes iam in")
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Process uploaded image
        try:
            img = Image.open(file).convert('RGB').resize((64, 64))
            print(img)
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

        img_array = np.array(img).flatten() / 255.0  # Flatten and normalize

        # Check scaler and model are initialized
        if scaler is None or model is None:
            return jsonify({"error": "Model not initialized. Please check backend logs."}), 500

        img_scaled = scaler.transform([img_array])

        # Predict
        prediction = model.predict(img_scaled)[0]
        confidence = model.predict_proba(img_scaled)[0][prediction]

        return jsonify({
            "prediction": "Cat 🐱" if prediction == 1 else "Not a Cat 🚫",
            "confidence": f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        print(f"🔥 Backend error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ----------- STEP 4: Run Flask App -----------

if __name__ == "__main__":
    app.run(debug=True, port=9999)
