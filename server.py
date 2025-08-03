# -------------------------------
# Cat vs Not-Cat Classifier (CNN)
# -------------------------------

# Import libraries
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Initialize Flask app
app = Flask(__name__)

# ----------- STEP 1: Load and Preprocess Raw Images -----------

def load_images(folder, label, size=(64, 64)):
    """
    Load images from a folder, resize to 'size', normalize, and assign label.
    """
    data = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            img = Image.open(filepath).convert('RGB').resize(size)
            img_array = np.array(img) / 255.0  # Normalize pixel values
            data.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {filepath}: {e}")
    return data, labels

# Paths to dataset folders
cats_folder = "dataset/cats"
not_cats_folder = "dataset/not_cats"

# Load images
cats_data, cats_labels = load_images(cats_folder, 1)
not_cats_data, not_cats_labels = load_images(not_cats_folder, 0)

# Combine and prepare dataset
X = np.array(cats_data + not_cats_data)
y = np.array(cats_labels + not_cats_labels)

print(f"✅ Loaded {len(X)} images: {sum(y)} cats, {len(y) - sum(y)} not-cats.")

# ----------- STEP 2: Train or Load CNN Model -----------

model_file = "cat_classifier_cnn.h5"

if os.path.exists(model_file):
    print("📦 Loading existing model...")
    model = load_model(model_file)
else:
    print("🚀 Training new enhanced CNN model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build CNN (Deeper + Dropout + BatchNorm)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Helps prevent overfitting
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Early stopping to prevent overtraining
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train CNN with augmented data
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,  # Allow up to 50 epochs
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"🎯 Enhanced CNN Model trained. Test accuracy: {accuracy * 100:.2f}%")

    # Save model
    model.save(model_file)
    print(f"💾 Model saved as {model_file}")

# ----------- STEP 3: Define Flask Routes -----------

@app.route("/")
def index():
    return render_template("index.html")

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
            img = Image.open(file).convert('RGB').resize((64, 64))
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)

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

# ----------- STEP 4: Run Flask App -----------

if __name__ == "__main__":
    app.run(debug=True, port=9999)
