"""Backend for Image Prediction using the improved image model (v4).

Model: MobileNetV2 (alpha 1.4) fine-tuned in three stages on 200 bird species
(320 x 320 input, Mixup + augmentation). Test accuracy 83.32 %, Top-5 95.87 %.

The public functions have the same names as ``src/image_predictor.py`` so the
UI can switch between the original and the improved model without changes.
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf

# Re-use the upload validation and label formatting of the original module.
from src.image_predictor import format_image_species_label, load_uploaded_image  # noqa: F401

IMAGE_MODEL_PATH = os.path.join("Models", "image_v4", "bird_image_model_v4.keras")
IMAGE_LABELS_PATH = os.path.join("Models", "image_v4", "bird_class_names.json")

# The v4 model was trained on 320 x 320 images.
IMAGE_TARGET_SIZE = (320, 320)


@st.cache_resource
def load_image_model():
    """Load and cache the v4 image model and its 200 class labels."""
    if not os.path.exists(IMAGE_MODEL_PATH):
        raise FileNotFoundError(f"Image model not found at: {IMAGE_MODEL_PATH}")
    if not os.path.exists(IMAGE_LABELS_PATH):
        raise FileNotFoundError(f"Image labels not found at: {IMAGE_LABELS_PATH}")

    model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)
    with open(IMAGE_LABELS_PATH, "r", encoding="utf-8") as label_file:
        labels = json.load(label_file)

    if model.output_shape[-1] != len(labels):
        raise ValueError("Image model and label count do not match.")
    return model, labels


def preprocess_image(image):
    """RGB PIL image -> (1, 320, 320, 3) tensor in [-1, 1].

    Uses the same steps as training: bilinear resize with ``tf.image.resize``
    followed by ``mobilenet_v2.preprocess_input``.
    """
    img = tf.convert_to_tensor(np.asarray(image, dtype=np.float32))
    img = tf.image.resize(img, IMAGE_TARGET_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return tf.expand_dims(img, 0).numpy()


def predict_image_species(image, top_k=3):
    """Return the Top-K ``(species_name, confidence_percent)`` predictions."""
    image_model, class_names = load_image_model()
    probabilities = image_model.predict(preprocess_image(image), verbose=0)[0]
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    return [
        (format_image_species_label(class_names[i]), float(probabilities[i] * 100))
        for i in top_indices
    ]
