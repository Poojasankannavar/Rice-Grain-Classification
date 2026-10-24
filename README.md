# Rice Grain Classification

Flask app that classifies rice-grain photos into Arborio, Basmati, Ipsala, Jasmine, or Karacadag. It is trained on isolated grain images against a dark background and rejects the large foreground area typical of MRI scans, so those images cannot be labeled as rice.

## Run locally

Use Python 3.11–3.13, then install the pinned dependencies and start Flask:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5001>. The model (`rice_ml_model.pkl`) and label encoder (`lable_encoder.pkl`) must remain beside `app.py`. Uploads are limited to 10 MB and saved under `static/uploads` with generated filenames. The MRI classifier runs separately on port 5000; the rice app only permits its five known rice-variety labels.

## Production server

On a platform that supports Gunicorn, start the WSGI app with:

```sh
gunicorn --bind 127.0.0.1:5001 app:app
```

## Retrain the model

Run the cells in `rice.ipynb` from the project directory. Training uses the five named variety folders and skips the `pred` demo folder. Keep the 64×64 RGB preprocessing in the notebook aligned with the preprocessing in `app.py`; the final notebook cell saves both model artifacts.
