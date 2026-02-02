from flask import Flask, request, jsonify
import os
from seed_check import is_seed_planting

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    ok, message = is_seed_planting(path)

    os.remove(path)

    return jsonify({
        "success": ok,
        "message": message
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
