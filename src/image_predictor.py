"""Backend module for the Image Prediction component of Avian AI.

Handles cached loading of the pre-trained MobileNetV2 image classification
model and class label mapping, safe validation of uploaded image files,
model-consistent preprocessing, and Top-3 species prediction generation.
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Paths to the existing trained image classification model and class label list.
IMAGE_MODEL_PATH = os.path.join("Models", "image", "bird_mobilenetv2_final.keras")
IMAGE_LABELS_PATH = os.path.join("Models", "image", "bird_class_names.json")

# Input spatial dimensions required by the trained MobileNetV2 image model.
IMAGE_TARGET_SIZE = (224, 224)


@st.cache_resource
def load_image_model():
    """Load and cache the trained MobileNetV2 bird image model and its labels.

    Uses ``@st.cache_resource`` so TensorFlow loads the ``.keras`` model from
    disk once per application session rather than on every Streamlit rerun.
    ``compile=False`` is passed because the model is used strictly for inference.

    Returns:
        tuple: ``(model, labels)`` where ``model`` is the loaded Keras model and
            ``labels`` is the ordered list of 200 raw class names.

    Raises:
        FileNotFoundError: If the model or label file does not exist.
        ValueError: If the model output dimension does not match the label count.
    """
    if not os.path.exists(IMAGE_MODEL_PATH):
        raise FileNotFoundError(f"Image model not found at: {IMAGE_MODEL_PATH}")
    if not os.path.exists(IMAGE_LABELS_PATH):
        raise FileNotFoundError(f"Image labels not found at: {IMAGE_LABELS_PATH}")

    model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)

    with open(IMAGE_LABELS_PATH, "r", encoding="utf-8") as label_file:
        labels = json.load(label_file)

    # Guard against mismatched label files by verifying output layer size.
    if model.output_shape[-1] != len(labels):
        raise ValueError("Image model and label count do not match.")

    return model, labels


def load_uploaded_image(uploaded_file):
    """Safely open and validate an uploaded image file as an RGB PIL Image.

    Converts grayscale or RGBA images (such as PNGs with transparency) to
    3-channel RGB so the array shape always matches the model's ``(224, 224, 3)``
    input expectation.

    Args:
        uploaded_file: File-like object returned by ``st.file_uploader``.

    Returns:
        PIL.Image.Image: Validated RGB image ready for preview and preprocessing.

    Raises:
        ValueError: If the uploaded file is empty, corrupted, or not a valid image.
    """
    if uploaded_file is None:
        raise ValueError("No image file was provided.")

    try:
        # Reset pointer in case the file buffer was previously read on rerun.
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        image = Image.open(uploaded_file)
        # Force full decode now so truncated or corrupted files fail early.
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(
            "Invalid or corrupted image file. Please upload a valid JPG, JPEG, or PNG image."
        ) from exc


def preprocess_image(image):
    """Transform a PIL RGB image into a MobileNetV2-ready input tensor.

    Preprocessing steps match the original training pipeline:
    1. Resize to 224x224 pixels, the spatial input shape of MobileNetV2.
    2. Convert to a float32 NumPy array and add a batch dimension -> (1, 224, 224, 3).
    3. Apply ``mobilenet_v2.preprocess_input`` to scale pixel values from
       [0, 255] into [-1, 1], which the trained weights expect.

    Args:
        image (PIL.Image.Image): Validated RGB bird image.

    Returns:
        numpy.ndarray: Preprocessed tensor of shape ``(1, 224, 224, 3)``.
    """
    resized_image = image.resize(IMAGE_TARGET_SIZE)
    img_array = np.array(resized_image, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_batch)


def format_image_species_label(raw_name):
    """Map a raw dataset class label to a readable bird species name.

    The image model's label file stores CUB-style class names prefixed with a
    numeric index and underscores (for example, ``"001.Black_footed_Albatross"``).
    This function strips the numeric prefix up to the first period and replaces
    underscores with spaces so species names display cleanly and match Wikipedia
    queries.

    Args:
        raw_name (str): Raw label string from ``bird_class_names.json``.

    Returns:
        str: Human-readable bird species name.
    """
    species = raw_name.split(".", 1)[1] if "." in raw_name else raw_name
    return species.replace("_", " ")


def predict_image_species(image, top_k=3):
    """Predict the Top-K bird species and confidence percentages for an image.

    Workflow:
    - Load the cached MobileNetV2 model and 200-class label list.
    - Preprocess the RGB image to shape ``(1, 224, 224, 3)``.
    - Run inference to obtain class probability scores for all 200 species.
    - Select the ``top_k`` highest-scoring indices in descending order and map
      each index to ``(formatted_species_name, confidence_percentage)``.

    Args:
        image (PIL.Image.Image): Validated RGB bird image.
        top_k (int): Number of top predictions to return (default: 3).

    Returns:
        list[tuple[str, float]]: Ranked ``(species_name, confidence_percent)`` pairs.
    """
    image_model, class_names = load_image_model()
    processed_batch = preprocess_image(image)

    # Model output is a 1D probability distribution across the 200 bird classes.
    probabilities = image_model.predict(processed_batch, verbose=0)[0]

    # Sort indices ascending, slice the last top_k, and reverse for descending rank.
    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    top_predictions = []
    for index in top_indices:
        species_name = format_image_species_label(class_names[index])
        confidence_percent = float(probabilities[index] * 100)
        top_predictions.append((species_name, confidence_percent))

    return top_predictions
