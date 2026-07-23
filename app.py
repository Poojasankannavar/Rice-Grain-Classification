from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import joblib
import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Model
model = joblib.load(os.path.join(BASE_DIR, "rice_ml_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "lable_encoder.pkl"))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template("index.html", error="Please upload an image file."), 400

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="Please select an image."), 400

    if not allowed_file(file.filename):
        return render_template("index.html", error="Please upload a PNG, JPG, JPEG, BMP, or WEBP image."), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Read Image
    image = cv2.imread(filepath)
    if image is None:
        return render_template("index.html", error="The uploaded file could not be read as an image."), 400

    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    feature = image.flatten().reshape(1, -1)

    prediction = model.predict(feature)[0]
    confidence = np.max(model.predict_proba(feature)) * 100

    label = encoder.inverse_transform([prediction])[0]

    return render_template(
        "index.html",
        prediction=label,
        confidence=round(confidence, 2),
        image=f"uploads/{filename}"
    )


if __name__ == "__main__":
    app.run(debug=True)
