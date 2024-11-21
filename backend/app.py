from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
CORS(app)
# 確保不限制請求的最大大小
app.config['MAX_CONTENT_LENGTH'] = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_breed_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "cat_breed_class_indices.json")

try:
    model = load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_labels = json.load(f)
    class_labels = {v: k for k, v in class_labels.items()}
    print("模型與分類標籤加載成功")
except Exception as e:
    print(f"模型加載失敗: {e}")
    model, class_labels = None, {}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Cat Breed Classifier API"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = os.path.join(BASE_DIR, "uploads", file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    try:
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        breed_name = class_labels.get(predicted_class, "Unknown")
        confidence = predictions[0][predicted_class]

        os.remove(file_path)
        return jsonify({"breed_name": breed_name, "confidence": float(confidence)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
