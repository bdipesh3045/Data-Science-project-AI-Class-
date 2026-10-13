import os
import string

import h5py
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_sign_language.keras")
LETTERS = string.ascii_uppercase[:25]  # Sign Language MNIST labels 0-24 = A-Y; 9 (J) never appears in training data
SKIP = {"J"}

TOP_K = 3
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def build_model():
    # Same architecture as data.ipynb. The saved file is a Keras 2.2.4 HDF5 file whose config
    # Keras 3 can't parse, so rebuild the graph here and load only the weights.
    return tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation="softmax"),
    ])


def load_model(path):
    model = build_model()
    with h5py.File(path, "r") as f:
        group = f["model_weights"]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in group.attrs["layer_names"]]
        if len(names) != len(model.layers):
            raise ValueError(f"{path} has {len(names)} layers, expected {len(model.layers)}")
        for layer, name in zip(model.layers, names):
            weights = group[name]
            layer.set_weights([weights[w][()] for w in weights.attrs["weight_names"]])
    return model


def preprocess(file):
    image = ImageOps.exif_transpose(Image.open(file)).convert("L")
    image = ImageOps.fit(image, (28, 28))  # center-crop to square, then resize; avoids squashing the hand
    return (np.asarray(image, dtype=np.float32) / 255.0).reshape(1, 28, 28, 1)


def top_predictions(probs):
    ranked = [i for i in np.argsort(probs)[::-1] if LETTERS[i] not in SKIP][:TOP_K]
    return [{"letter": LETTERS[i], "score": round(float(probs[i]) * 100, 1)} for i in ranked]


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
model = load_model(MODEL_PATH)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify(error="Choose an image first."), 400
    try:
        batch = preprocess(file.stream)
    except Exception:  # PIL raises several types for bad, truncated, or oversized images
        return jsonify(error="That file isn't a readable image. Try a PNG, JPG or WEBP."), 400
    probs = model.predict(batch, verbose=0)[0]
    return jsonify(predictions=top_predictions(probs))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Keep your image under 8 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)))
