# 🌾 Rice Grain Image Classifier

A Machine Learning web application that classifies rice grain images into five varieties: **Arborio, Basmati, Ipsala, Jasmine, and Karacadag**. Built using Python, Flask, OpenCV, NumPy, and scikit-learn, the application predicts the rice variety from an uploaded image and displays the predicted class with a confidence score.

<p align="center">
  <a href="https://rice-grain-classification-4ydg.onrender.com">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20Application-2E8B57?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web%20Application-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
</p>

## 🌐 Live Demo

🚀 **Try the application:** [Rice Grain Image Classifier](https://rice-grain-classification-4ydg.onrender.com)

Upload a rice grain image to predict its variety from five categories. The application validates the uploaded image and displays the predicted variety with a confidence score.

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Rice Varieties](#-rice-varieties)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Model and Prediction](#-model-and-prediction)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [Model Training](#-model-training)
- [Deployment](#-deployment)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

## 📖 Overview

The Rice Grain Image Classifier is a machine learning project designed to identify rice varieties from uploaded grain photographs.

The application accepts an image, validates its format and size, checks whether it resembles the dark-background rice images used during model training, preprocesses the image, and passes it to a trained image-classification model.

The predicted variety and associated confidence score are displayed through a Flask-based web interface.

This project demonstrates image preprocessing, machine learning inference, input validation, and web application development.

## ✨ Features

- 🌾 Classification of five rice varieties.
- 📤 Upload rice grain images through a web interface.
- 🖼️ Preview uploaded images.
- 🤖 Predict the rice variety using a trained machine learning model.
- 📊 Display the predicted class and confidence score.
- 🔍 Validate uploaded image formats.
- 📦 Enforce a maximum upload size of 10 MB.
- 🖤 Check whether images resemble the expected dark-background training images.
- ❌ Reject unsupported or unsuitable images.
- 🧩 Load the trained model and label encoder using Joblib.
- 🌐 Serve predictions through Flask.
- 🚀 Support production deployment using Gunicorn.

## 🌾 Rice Varieties

The classifier predicts one of five rice varieties.

| Variety | Description |
|---|---|
| **Arborio** | An Italian short-grain rice variety commonly used for risotto. |
| **Basmati** | A long-grain, aromatic rice variety. |
| **Ipsala** | A rice variety associated with the Ipsala region of Türkiye. |
| **Jasmine** | An aromatic rice variety commonly associated with Southeast Asian cuisine. |
| **Karacadag** | A traditional rice variety associated with the Karacadağ region of Türkiye. |

## ⚙️ How It Works

The application follows this image-classification workflow:

1. **Image Upload:** The user uploads a photograph of a rice grain.
2. **File Validation:** The application checks the supported image format and enforces the 10 MB upload limit.
3. **Image Suitability Check:** The application checks whether the image resembles the expected dark-background training images.
4. **Image Preprocessing:** OpenCV and NumPy decode, resize, and prepare the image in the format expected by the trained model.
5. **Model Inference:** The saved machine learning model predicts the rice variety.
6. **Label Decoding:** The label encoder converts the predicted class into its corresponding variety name.
7. **Result Display:** The Flask application displays the predicted variety and confidence score.

### Workflow Diagram

```text
       Upload Rice Grain Image
                 |
                 v
          Validate Image
       Format and File Size
                 |
                 v
        Check Image Suitability
       Dark Background Pattern
                 |
                 v
        Image Preprocessing
          OpenCV + NumPy
                 |
                 v
        Trained ML Classifier
                 |
                 v
           Predict Class
                 |
                 v
           Label Encoder
                 |
                 v
        Variety + Confidence
```

## 🛠️ Technology Stack

### Programming and Machine Learning

- **Python** — Core programming language.
- **scikit-learn** — Machine learning model and prediction workflow.
- **OpenCV** — Image decoding and preprocessing.
- **NumPy** — Numerical operations and image arrays.
- **Joblib** — Loading saved model artifacts.

### Web Development

- **Flask** — Web server, upload handling, and prediction routes.
- **HTML5** — Web page structure.
- **CSS3** — Interface styling.
- **JavaScript** — Client-side interactions, where implemented.
- **Jinja2** — Flask template rendering.

### Development and Deployment

- **Jupyter Notebook** — Model training and experimentation.
- **Git and GitHub** — Version control and source code management.
- **Gunicorn** — Production WSGI server.

## 🧪 Model and Prediction

The application uses a trained machine learning classifier to identify the rice variety from a preprocessed image.

### Model Workflow

| Component | Purpose |
|---|---|
| Input | Uploaded rice grain photograph |
| Image processing | OpenCV and NumPy |
| Image validation | Format, file size, and expected visual characteristics |
| Classifier | Saved scikit-learn model |
| Label encoder | Maps model labels to variety names |
| Output | Predicted rice variety |
| Confidence | Model-provided confidence score, where supported |

### Supported Classes

```text
Arborio
Basmati
Ipsala
Jasmine
Karacadag
```

The trained classifier and label encoder are stored as serialized `.pkl` files and loaded by the Flask application.

**Note:** The exact model algorithm, input dimensions, feature extraction method, and confidence calculation depend on the implementation in `rice.ipynb` and `app.py`. A confidence score should not automatically be interpreted as a calibrated probability.

## 📂 Project Structure

```text
Rice-Grain-Image-Classifier/
│
├── app.py
├── rice.ipynb
├── README.md
├── requirements.txt
│
├── model.pkl
├── label_encoder.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
└── .gitignore
```

**Note:** This is an illustrative project structure. Keep the actual filenames and folders aligned with your repository. The model and label encoder filenames above are examples; use the names referenced in `app.py`. Uploaded images should not be committed to Git unnecessarily.

## 💻 Installation and Setup

### Prerequisites

Install the following tools:

- Python 3.x compatible with the project's dependencies.
- Git.
- Visual Studio Code, PyCharm, or another Python editor.

### 1. Clone the Repository

Replace the placeholder URL with your actual GitHub repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd Rice-Grain-Image-Classifier
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

If your repository contains `requirements.txt`, run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Otherwise, install the main dependencies:

```bash
pip install Flask scikit-learn opencv-python numpy joblib jupyter gunicorn
```

Use dependency versions compatible with your saved model and training environment.

### 4. Verify Model Files

Ensure the model and label encoder files expected by `app.py` are present in the correct locations.

For example:

```text
model.pkl
label_encoder.pkl
```

These filenames are illustrative. Use the actual filenames referenced by your application.

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

Upload a supported rice grain photograph to view the predicted variety.

For production deployment, use a WSGI server such as Gunicorn rather than Flask's development server.

Example:

```bash
gunicorn app:app
```

This command assumes that `app.py` exposes the Flask application through a variable named `app`.

## 📓 Model Training

The Jupyter Notebook `rice.ipynb` contains the model development workflow.

Open the notebook using:

```bash
jupyter notebook rice.ipynb
```

Use the notebook to inspect the training pipeline, evaluate the classifier, and retrain the model if the required dataset and training code are available.

Before deploying a newly trained model, ensure that its preprocessing steps, expected input format, class order, and label encoder are compatible with the Flask application.

## 🚀 Deployment

The application is deployed on Render.

**Live application:** [Rice Grain Image Classifier](https://rice-grain-classification-4ydg.onrender.com)

### Deployment Checklist

- [ ] Push the project source code to GitHub.
- [ ] Include a compatible `requirements.txt`.
- [ ] Ensure all model artifacts are accessible.
- [ ] Configure the Render build and start commands.
- [ ] Use Gunicorn for production serving.
- [ ] Configure upload size and image validation on the server.
- [ ] Test predictions for all five rice varieties.
- [ ] Test unsupported files and unsuitable images.
- [ ] Avoid storing unnecessary uploaded images.
- [ ] Verify the deployed application after deployment.

Example production start command:

```bash
gunicorn app:app
```

The actual deployment commands depend on the repository structure and hosting configuration.

## 🔮 Future Improvements

- Evaluate model performance on an independent test dataset.
- Report accuracy, precision, recall, and F1-score.
- Add a confusion matrix for the five rice varieties.
- Improve detection of non-rice images.
- Add automated tests for uploads and predictions.
- Improve accessibility and responsive design.
- Add prediction history where appropriate.
- Document the model algorithm and image preprocessing parameters.
- Improve confidence calibration and error handling.
- Add continuous integration for testing.

## 👩‍💻 Author

**Pooja Sankannavar**

MCA Student | Full-Stack Developer | Software Development and AI Enthusiast

- **GitHub:** [@Poojasankannavar](https://github.com/Poojasankannavar)
- **LinkedIn:** [Pooja Sankannavar](https://www.linkedin.com/in/pooja-sankannavar-8751a0318/)
- **Live Project:** [Rice Grain Image Classifier](https://rice-grain-classification-4ydg.onrender.com)

---

⭐ If you find this project interesting, consider giving the repository a star!

*Developed as a learning project exploring image classification, machine learning, Python, and Flask web development.*
