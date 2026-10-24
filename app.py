from pathlib import Path
import base64
import os
from uuid import uuid4

import cv2
import joblib
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "rice_ml_model.pkl"
ENCODER_PATH = BASE_DIR / "lable_encoder.pkl"  # Keep the existing artifact filename.
IMAGE_SIZE = (64, 64)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
RICE_VARIETIES = {"Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"}
DEFAULT_PORT = 5001
MIN_RICE_FOREGROUND = 0.05
MAX_RICE_FOREGROUND = 0.285

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.is_file() or not ENCODER_PATH.is_file():
    raise FileNotFoundError("The model and label encoder must be present beside app.py.")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def render_home(*, error=None, status=200, **context):
    return render_template("index.html", error=error, **context), status


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return render_home(
        error="The image is too large. Please upload an image smaller than 10 MB.",
        status=413,
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_image(filename):
    """Serve upload URLs used by the original result template as well."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("image")
    if uploaded_file is None:
        return render_home(error="Please choose an image file.", status=400)
    if not uploaded_file.filename:
        return render_home(error="Please select an image.", status=400)
    if not allowed_file(uploaded_file.filename):
        return render_home(
            error="Please upload a PNG, JPG, JPEG, BMP, or WEBP image.", status=400
        )

    contents = uploaded_file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        return render_home(
            error="The image is too large. Please upload an image smaller than 10 MB.",
            status=413,
        )
    original_image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original_image is None or original_image.size == 0:
        return render_home(
            error="The uploaded file could not be read as an image.", status=400
        )

    # The training photos show a rice sample on a mostly dark background.
    # MRI scans have a much larger foreground area and otherwise receive
    # misleading rice labels from this closed-set classifier.
    domain_image = cv2.resize(original_image, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(domain_image, cv2.COLOR_BGR2GRAY)
    foreground_ratio = float(np.mean(gray > 20))
    if not MIN_RICE_FOREGROUND <= foreground_ratio < MAX_RICE_FOREGROUND:
        return render_home(
            error=(
                "This image does not match the rice-grain photos supported here. "
                "Upload a rice grain image on a dark background. For MRI scans, "
                "use the separate Brain Tumor MRI Classifier."
            ),
            status=400,
        )

    image = cv2.resize(original_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feature = image.reshape(1, -1)
    if feature.shape[1] != model.n_features_in_:
        app.logger.error(
            "Model expects %s features; preprocessing produced %s",
            model.n_features_in_,
            feature.shape[1],
        )
        return render_home(
            error="The classifier is configured incorrectly. Please contact the site owner.",
            status=500,
        )

    probabilities = model.predict_proba(feature)[0]
    model_labels = encoder.inverse_transform(model.classes_)
    # Older saved artifacts were trained with the notebook's demo-only "pred"
    # folder. Never expose that folder as a rice variety in the web app.
    valid_classes = np.array([str(label) in RICE_VARIETIES for label in model_labels])
    if not np.any(valid_classes):
        app.logger.error("The saved model does not contain any recognized rice varieties")
        return render_home(
            error="The classifier has no recognized rice varieties configured.", status=500
        )
    rice_probabilities = probabilities[valid_classes]
    rice_classes = model.classes_[valid_classes]
    prediction = rice_classes[int(np.argmax(rice_probabilities))]
    confidence = float(np.max(rice_probabilities) / np.sum(rice_probabilities)) * 100
    label = str(encoder.inverse_transform([prediction])[0])

    # Persist only validated, decodable images and use an unguessable unique name.
    filename = f"{uuid4().hex}.{uploaded_file.filename.rsplit('.', 1)[1].lower()}"
    if not cv2.imwrite(str(UPLOAD_FOLDER / filename), original_image):
        app.logger.error("Could not save uploaded image %s", filename)
        return render_home(error="The image could not be saved. Please try again.", status=500)

    # Embed the saved image in the result page so it remains visible even when
    # a hosting setup does not serve Flask's /static/ path correctly.
    image_bytes = (UPLOAD_FOLDER / filename).read_bytes()
    image_type = "jpeg" if filename.endswith((".jpg", ".jpeg")) else filename.rsplit(".", 1)[1]
    image_data_url = f"data:image/{image_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    return render_template(
        "index.html",
        prediction=label,
        confidence=round(confidence, 2),
        image_data_url=image_data_url,
        # Keep the old {{ image }} template variable working during upgrades.
        image=f"/uploads/{filename}",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("RICE_APP_PORT", DEFAULT_PORT)))
