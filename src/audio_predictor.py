"""Backend module for the Audio Prediction component of Avian AI.

Handles cached loading of the fine-tuned MobileNetV2 bird audio model and its
59 class labels, safe temporary-file handling for uploaded recordings,
5-second peak-energy sliding-window segment selection, Mel spectrogram
preprocessing, and Top-3 species prediction generation.
"""

import json
import os
import tempfile

import cv2
import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Paths to the existing fine-tuned audio classification model and label list.
AUDIO_MODEL_PATH = os.path.join("Models", "audio", "bird_audio_finetuned.keras")
AUDIO_LABELS_PATH = os.path.join("Models", "audio", "bird_audio_labels.json")

# Audio preprocessing constants matching the original training pipeline.
AUDIO_SAMPLE_RATE = 22050
WINDOW_DURATION_SECONDS = 5
HOP_DURATION_SECONDS = 2.5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SPECTROGRAM_SIZE = (224, 224)
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mpeg", ".ogg", ".flac", ".m4a"}


@st.cache_resource
def load_audio_model():
    """Load and cache the fine-tuned MobileNetV2 audio model and 59 species labels.

    Uses ``@st.cache_resource`` so TensorFlow loads the ``.keras`` model once
    per application session instead of reloading on every Streamlit rerun.
    ``compile=False`` is used because the model is loaded strictly for inference.

    Returns:
        tuple: ``(model, labels)`` where ``model`` is the loaded Keras model and
            ``labels`` is the ordered list of 59 bird species names.

    Raises:
        FileNotFoundError: If the model or label file does not exist.
        ValueError: If the model output dimension does not match the label count.
    """
    if not os.path.exists(AUDIO_MODEL_PATH):
        raise FileNotFoundError(f"Audio model not found at: {AUDIO_MODEL_PATH}")
    if not os.path.exists(AUDIO_LABELS_PATH):
        raise FileNotFoundError(f"Audio labels not found at: {AUDIO_LABELS_PATH}")

    model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)

    with open(AUDIO_LABELS_PATH, "r", encoding="utf-8") as label_file:
        labels = json.load(label_file)

    # Verify that the final classification layer aligns with the 59-label JSON.
    if model.output_shape[-1] != len(labels):
        raise ValueError("Audio model and label count do not match.")

    return model, labels


def validate_audio_upload(uploaded_file):
    """Validate an uploaded audio object and return its raw bytes and filename.

    Args:
        uploaded_file: File-like object returned by ``st.file_uploader``.

    Returns:
        tuple[bytes, str]: ``(audio_bytes, filename)`` ready for decoding.

    Raises:
        ValueError: If no file was provided or the uploaded file is empty.
    """
    if uploaded_file is None:
        raise ValueError("No audio file was provided.")

    audio_bytes = uploaded_file.getvalue()
    if not audio_bytes:
        raise ValueError("The uploaded audio file is empty.")

    filename = getattr(uploaded_file, "name", "recording.wav")
    return audio_bytes, filename


def _resolve_temp_suffix(filename):
    """Return a safe audio file suffix for temporary storage."""
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_AUDIO_EXTENSIONS:
            return ext
    return ".wav"


def load_audio_waveform(audio_bytes, filename=None):
    """Decode uploaded audio bytes into a mono 22,050 Hz float waveform.

    Librosa decodes audio from a file path, so uploaded bytes are written to a
    temporary file and always removed in the ``finally`` block. Audio is
    resampled to 22,050 Hz and converted to mono to match training inputs.

    Args:
        audio_bytes (bytes): Raw bytes of the uploaded audio recording.
        filename (str | None): Optional original filename used to preserve
            the container extension for audio decoders.

    Returns:
        numpy.ndarray: 1D mono audio time-series at 22,050 Hz.

    Raises:
        ValueError: If the audio buffer is empty or cannot be decoded.
    """
    if not audio_bytes:
        raise ValueError("The uploaded audio is empty.")

    temp_path = None
    suffix = _resolve_temp_suffix(filename)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            audio, _ = librosa.load(temp_path, sr=AUDIO_SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ValueError(
                "Could not decode the uploaded audio file. Please upload a valid WAV, MP3, MPEG, OGG, FLAC, or M4A recording."
            ) from exc

        if len(audio) == 0:
            raise ValueError("The uploaded audio is empty.")

        return audio
    finally:
        # Always remove temporary audio files created during request processing.
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def select_peak_energy_segment(audio, sample_rate=AUDIO_SAMPLE_RATE):
    """Select the 5-second audio window with the highest mean-squared energy.

    Sliding-window and peak-energy logic:
    - If the recording is 5 seconds or shorter (<= 110,250 samples), zero-pad
      the end so the segment matches the exact 5-second duration.
    - If the recording is longer than 5 seconds, slide a 5-second window across
      the waveform using a 2.5-second hop (50% overlap) and select the window
      with maximum mean squared amplitude ``np.mean(candidate ** 2)``, which
      captures the strongest bird vocalization activity.

    Args:
        audio (numpy.ndarray): 1D mono audio waveform.
        sample_rate (int): Waveform sampling rate in Hz (default: 22050).

    Returns:
        numpy.ndarray: 5-second audio segment of length ``5 * sample_rate``.
    """
    window_size = WINDOW_DURATION_SECONDS * sample_rate
    hop_size = int(HOP_DURATION_SECONDS * sample_rate)

    if len(audio) <= window_size:
        return np.pad(audio, (0, window_size - len(audio)))

    best_energy = -1
    segment = audio[:window_size]

    for start in range(0, len(audio) - window_size + 1, hop_size):
        candidate = audio[start:start + window_size]
        energy = np.mean(candidate ** 2)

        if energy > best_energy:
            best_energy = energy
            segment = candidate

    return segment


def extract_mel_spectrogram_tensor(segment, sample_rate=AUDIO_SAMPLE_RATE):
    """Convert a 5-second audio segment into a MobileNetV2 spectrogram tensor.

    Preserves the exact preprocessing pipeline used with the trained model:
    1. Compute a 128-band Mel spectrogram (``n_fft=2048``, ``hop_length=512``).
    2. Convert power values to decibels relative to the segment peak.
    3. Resize the spectrogram image to ``224x224`` using ``cv2.INTER_AREA``.
    4. Min-max normalize values to ``[0, 255]`` uint8.
    5. Stack across 3 channels to produce an RGB-shaped spectrogram image.
    6. Convert to ``float32``, apply ``mobilenet_v2.preprocess_input`` scaling
       to ``[-1, 1]``, and add a batch dimension -> ``(1, 224, 224, 3)``.

    Args:
        segment (numpy.ndarray): 5-second mono audio array.
        sample_rate (int): Sampling rate in Hz (default: 22050).

    Returns:
        numpy.ndarray: Preprocessed tensor of shape ``(1, 224, 224, 3)``.
    """
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_resized = cv2.resize(
        mel_db,
        SPECTROGRAM_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    mel_normalized = (
        255
        * (mel_resized - mel_resized.min())
        / (mel_resized.max() - mel_resized.min() + 1e-8)
    ).astype(np.uint8)

    mel_rgb = np.stack([mel_normalized] * 3, axis=-1)

    model_input = preprocess_input(mel_rgb.astype(np.float32))
    return np.expand_dims(model_input, axis=0)


def predict_audio_species(audio_bytes, filename=None, top_k=3):
    """Predict the Top-K bird species and confidence scores from audio bytes.

    Workflow:
    - Load the cached fine-tuned audio model and 59 species labels.
    - Decode audio to 22,050 Hz mono and select the 5-second peak-energy segment.
    - Convert the selected segment into a 224x224x3 MobileNetV2 spectrogram tensor.
    - Run inference and extract the ``top_k`` indices in descending probability order.
    - Map each predicted index directly to ``audio_class_names[index]`` and
      percentage confidence ``float(prediction[index] * 100)``.

    Args:
        audio_bytes (bytes): Raw bytes of the uploaded audio recording.
        filename (str | None): Optional filename for decoder extension matching.
        top_k (int): Number of top predictions to return (default: 3).

    Returns:
        list[tuple[str, float]]: Ranked ``(species_name, confidence_percent)`` pairs.
    """
    audio_model, audio_class_names = load_audio_model()

    audio = load_audio_waveform(audio_bytes, filename=filename)
    segment = select_peak_energy_segment(audio, sample_rate=AUDIO_SAMPLE_RATE)
    model_input = extract_mel_spectrogram_tensor(segment, sample_rate=AUDIO_SAMPLE_RATE)

    prediction = audio_model.predict(model_input, verbose=0)[0]
    top_indices = np.argsort(prediction)[-top_k:][::-1]

    top3 = []
    for index in top_indices:
        species = audio_class_names[index]
        confidence = float(prediction[index] * 100)
        top3.append((species, confidence))

    return top3
