"""Backend for Audio Prediction using audio model v3.

Model: MobileNetV2 (alpha 1.4) on Mel-spectrograms, 70 species, trained on the
BirdCLEF 2022 dataset (Xeno-canto recordings, 32 kHz) with SpecAugment, Mixup,
label smoothing and class weights (notebook: Bird_Audio_Training_v3_BirdCLEF).
Test accuracy 84.89 % on 1,138 unseen recordings (Top-3 93.23 %, macro F1 0.818).

Pre-processing is identical to training: mono 32 kHz, first 60 seconds,
5-second windows with a 2.5-second hop, windows below 10 % of the loudest
window's energy are dropped, up to 12 of the loudest windows are classified and
their probabilities are averaged. Each window becomes a 128-band Mel-spectrogram
(50 Hz - 14 kHz, 80 dB range scaled to 0..255).

The public functions have the same names as ``src/audio_predictor.py``.
"""

import json
import os
import tempfile

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf

# Re-use the upload validation of the original module.
from src.audio_predictor import _resolve_temp_suffix, validate_audio_upload  # noqa: F401

AUDIO_MODEL_PATH = os.path.join("Models", "audio_v3", "bird_audio_model_v3.keras")
AUDIO_LABELS_PATH = os.path.join("Models", "audio_v3", "bird_audio_labels_v3.json")

# Pre-processing constants - identical to the training notebook (hyperparameters.json).
AUDIO_SAMPLE_RATE = 32000
WINDOW_DURATION_SECONDS = 5.0
HOP_DURATION_SECONDS = 2.5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 50
FMAX = 14000
MAX_SECONDS = 60           # only the first 60 seconds of a recording are used
ENERGY_KEEP = 0.10         # ignore windows quieter than 10 % of the loudest window
MAX_WINDOWS = 12           # windows averaged per recording
IMG_SIZE = 224

AUDIO_RESULT_NOTE = (
    "Averaged over up to 12 of the loudest 5-second windows (2.5-second step) from the first 60 seconds."
)


@st.cache_resource
def load_audio_model():
    """Load and cache the v3 audio model and its 70 species labels."""
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


def load_audio_waveform(audio_bytes, filename=None):
    """Decode uploaded audio bytes into mono 32 kHz audio (first 60 seconds)."""
    if not audio_bytes:
        raise ValueError("The uploaded audio is empty.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=_resolve_temp_suffix(filename)) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name
        try:
            audio, _ = librosa.load(temp_path, sr=AUDIO_SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
        except Exception as error:
            raise ValueError(
                "Could not decode this audio file. Please upload a valid WAV, MP3, OGG, FLAC or M4A recording."
            ) from error
        if audio is None or len(audio) == 0:
            raise ValueError("The uploaded audio contains no samples.")
        return audio.astype(np.float32)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


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
    """5-second segment -> uint8 Mel-spectrogram image (128 x 313)."""
    mel = librosa.feature.melspectrogram(
        y=segment, sr=AUDIO_SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return np.round((mel_db + 80.0) / 80.0 * 255.0).astype(np.uint8)


def spectrograms_to_model_input(specs):
    """uint8 (N, 128, 313) -> float32 (N, 224, 224, 3) in [-1, 1]."""
    x = tf.cast(tf.convert_to_tensor(specs), tf.float32)[..., None]
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    x = x / 127.5 - 1.0
    return tf.repeat(x, 3, axis=-1).numpy()


def predict_audio_species(audio_bytes, filename=None, top_k=3):
    """Return the Top-K ``(species_name, confidence_percent)`` predictions."""
    audio_model, class_names = load_audio_model()

    audio = load_audio_waveform(audio_bytes, filename=filename)
    segments = select_windows(audio)
    specs = np.stack([segment_to_spectrogram(s) for s in segments])

    probabilities = audio_model.predict(spectrograms_to_model_input(specs), verbose=0).mean(axis=0)
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    return [(class_names[i], float(probabilities[i] * 100)) for i in top_indices]
