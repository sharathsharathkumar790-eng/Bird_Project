"""Chooses which models the app uses.

Set ``USE_NEW_MODELS = False`` to go back to the original models
(``Models/image`` and ``Models/audio``) without changing any other file.

``AUDIO_MODEL_VERSION`` picks the improved audio model:
    "v3"  BirdCLEF 2022, 70 species, 84.89 % test accuracy (default)
    "v2"  Kaggle 114-species subset, 59 species, 61.46 % test accuracy
If the chosen model files are missing, the app falls back automatically
(v3 -> v2 -> original models).
"""

import os

USE_NEW_MODELS = True
AUDIO_MODEL_VERSION = "v3"

_IMAGE_V4 = os.path.join("Models", "image_v4", "bird_image_model_v4.keras")
_AUDIO_FILES = {
    "v3": os.path.join("Models", "audio_v3", "bird_audio_model_v3.keras"),
    "v2": os.path.join("Models", "audio_v2", "bird_audio_model_v2.keras"),
}

# Facts shown in the app (sidebar, guide card, analytics). Taken from each model's test_results.json.
AUDIO_MODEL_INFO = {
    "v3": dict(version="v3", species=70, test_accuracy="84.89%", top3="93.23%", macro_f1="0.818",
               test_recordings=1138, dataset="BirdCLEF 2022 (Xeno-canto)",
               reports_dir=os.path.join("reports_v3", "audio"),
               samples_dir=os.path.join("test_samples", "audio_v3")),
    "v2": dict(version="v2", species=59, test_accuracy="61.46%", top3="71.88%", macro_f1="0.509",
               test_recordings=96, dataset="Kaggle 114-species subset",
               reports_dir=os.path.join("reports_v2", "audio"),
               samples_dir=os.path.join("test_samples", "audio")),
    "original": dict(version="original", species=59, test_accuracy=None, top3=None, macro_f1=None,
                     test_recordings=None, dataset="Kaggle 114-species subset",
                     reports_dir=None, samples_dir=os.path.join("test_samples", "audio")),
}

_audio_version = None
if USE_NEW_MODELS and os.path.exists(_IMAGE_V4):
    for _v in [AUDIO_MODEL_VERSION, "v3", "v2"]:
        if _v in _AUDIO_FILES and os.path.exists(_AUDIO_FILES[_v]):
            _audio_version = _v
            break
NEW_MODELS_ACTIVE = _audio_version is not None

if NEW_MODELS_ACTIVE:
    from src.image_predictor_v2 import (  # noqa: F401
        IMAGE_LABELS_PATH, IMAGE_MODEL_PATH, load_image_model, load_uploaded_image, predict_image_species,
    )
    if _audio_version == "v3":
        from src import audio_predictor_v3 as AUDIO_BACKEND
    else:
        from src import audio_predictor_v2 as AUDIO_BACKEND
    AUDIO_INFO = AUDIO_MODEL_INFO[_audio_version]
    MODEL_VERSION_LABEL = f"Improved models (image v4, audio {_audio_version})"
else:
    from src.image_predictor import (  # noqa: F401
        IMAGE_LABELS_PATH, IMAGE_MODEL_PATH, load_image_model, load_uploaded_image, predict_image_species,
    )
    from src import audio_predictor as AUDIO_BACKEND
    AUDIO_INFO = AUDIO_MODEL_INFO["original"]
    MODEL_VERSION_LABEL = "Original models"

AUDIO_LABELS_PATH = AUDIO_BACKEND.AUDIO_LABELS_PATH
AUDIO_MODEL_PATH = AUDIO_BACKEND.AUDIO_MODEL_PATH
load_audio_model = AUDIO_BACKEND.load_audio_model
predict_audio_species = AUDIO_BACKEND.predict_audio_species
validate_audio_upload = AUDIO_BACKEND.validate_audio_upload
AUDIO_RESULT_NOTE = getattr(AUDIO_BACKEND, "AUDIO_RESULT_NOTE",
                            "Analyzed 5-second high-activity window (2.5-second step).")
