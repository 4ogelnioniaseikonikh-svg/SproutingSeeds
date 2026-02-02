from flask import Flask, request, jsonify
from seed_check import check_planting
import os
import uuid

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/", methods=["GET"])
def health():
    return "SproutingSeeds API is running 🌱"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    result = check_planting(path)

    os.remove(path)

    message = (
        "🌱 Seed planting detected!"
        if result["planting"]
        else "❌ No seed planting detected"
    )

    return jsonify({
        "planting": result["planting"],
        "detected_objects": result["objects"],
        "message": message
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
