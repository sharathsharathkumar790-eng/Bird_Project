"""Backend for Audio Prediction using the improved audio model (v2).

Model: MobileNetV2 (alpha 1.4) on Mel-spectrograms, 59 species, trained on
duplicate-free recordings with SpecAugment and Mixup.
Test accuracy 61.46 % on 96 unseen recordings (Top-3 71.88 %).

Difference from the original pipeline: instead of only the single loudest
5-second window, up to 12 of the loudest 5-second windows (2.5-second hop) are
classified and their probabilities are averaged. The spectrogram scaling
matches training exactly (fixed 80 dB range instead of min-max).

The public functions have the same names as ``src/audio_predictor.py``.
"""

import json
import os

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf

# Re-use upload validation and temporary-file decoding of the original module.
from src.audio_predictor import load_audio_waveform, validate_audio_upload  # noqa: F401

AUDIO_MODEL_PATH = os.path.join("Models", "audio_v2", "bird_audio_model_v2.keras")
AUDIO_LABELS_PATH = os.path.join("Models", "audio_v2", "bird_audio_labels.json")

# Pre-processing constants - identical to the training notebook (Bird_Audio_Training_v2).
AUDIO_SAMPLE_RATE = 22050
WINDOW_DURATION_SECONDS = 5.0
HOP_DURATION_SECONDS = 2.5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
MAX_SECONDS = 180          # only the first 3 minutes of very long recordings are used
ENERGY_KEEP = 0.10         # ignore windows quieter than 10 % of the loudest window
MAX_WINDOWS = 12           # windows averaged per recording
IMG_SIZE = 224

AUDIO_RESULT_NOTE = (
    "Averaged over up to 12 of the loudest 5-second windows (2.5-second step)."
)


@st.cache_resource
def load_audio_model():
    """Load and cache the v2 audio model and its 59 species labels."""
    if not os.path.exists(AUDIO_MODEL_PATH):
        raise FileNotFoundError(f"Audio model not found at: {AUDIO_MODEL_PATH}")
    if not os.path.exists(AUDIO_LABELS_PATH):
        raise FileNotFoundError(f"Audio labels not found at: {AUDIO_LABELS_PATH}")

    model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
    with open(AUDIO_LABELS_PATH, "r", encoding="utf-8") as label_file:
        labels = json.load(label_file)

    if model.output_shape[-1] != len(labels):
        raise ValueError("Audio model and label count do not match.")
    return model, labels


def select_windows(audio, max_windows=MAX_WINDOWS):
    """Return up to ``max_windows`` 5-second segments, loudest first."""
    win = int(WINDOW_DURATION_SECONDS * AUDIO_SAMPLE_RATE)
    hop = int(HOP_DURATION_SECONDS * AUDIO_SAMPLE_RATE)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    starts = list(range(0, len(audio) - win + 1, hop))
    energy = np.array([np.mean(audio[s:s + win] ** 2) for s in starts])
    order = np.argsort(-energy)
    keep = [starts[i] for i in order if energy[i] >= ENERGY_KEEP * energy[order[0]]]
    return [audio[s:s + win] for s in keep[:max_windows]]


def segment_to_spectrogram(segment):
    """5-second segment -> uint8 Mel-spectrogram image (128 x 216)."""
    mel = librosa.feature.melspectrogram(
        y=segment, sr=AUDIO_SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return (np.clip((mel_db + 80.0) / 80.0, 0, 1) * 255).astype(np.uint8)


def spectrograms_to_model_input(specs):
    """uint8 (N, 128, 216) -> float32 (N, 224, 224, 3) in [-1, 1]."""
    x = tf.cast(tf.convert_to_tensor(specs), tf.float32)[..., None]
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    x = tf.tile(x, [1, 1, 1, 3])
    return (x / 127.5 - 1.0).numpy()


def predict_audio_species(audio_bytes, filename=None, top_k=3):
    """Return the Top-K ``(species_name, confidence_percent)`` predictions."""
    audio_model, class_names = load_audio_model()

    audio = load_audio_waveform(audio_bytes, filename=filename)
    audio = audio[: MAX_SECONDS * AUDIO_SAMPLE_RATE]
    segments = select_windows(audio)
    specs = np.stack([segment_to_spectrogram(s) for s in segments])

    probabilities = audio_model.predict(spectrograms_to_model_input(specs), verbose=0).mean(axis=0)
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    return [(class_names[i], float(probabilities[i] * 100)) for i in top_indices]
