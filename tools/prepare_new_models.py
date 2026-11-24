"""One-time preparation of the new models for the Streamlit app.

Usage (from the project folder, with the virtual environment):
    .venv\\Scripts\\python tools\\prepare_new_models.py

It reads the two files downloaded from Google Drive into Models/_downloads/:
    bird_image_model_v4.keras   (from Bird_Project/image_training_v4/models)
    bird_audio_model_v2.keras   (from Bird_Project/audio_training_v2/models)
and writes compact float32 copies (about 19 MB each, same predictions,
faster on CPU, no optimizer state) to:
    Models/image_v4/bird_image_model_v4.keras
    Models/audio_v2/bird_audio_model_v2.keras
The downloaded originals are not changed.
"""

import os
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = [
    ("Image v4", "bird_image_model_v4.keras", os.path.join("Models", "image_v4")),
    ("Audio v2", "bird_audio_model_v2.keras", os.path.join("Models", "audio_v2")),
]


def to_float32(model):
    """Rebuild the model with float32 layers and copy the trained weights."""
    def fix(obj):
        if isinstance(obj, dict):
            cfg = obj.get("config")
            if obj.get("class_name") == "DTypePolicy" and isinstance(cfg, dict) and cfg.get("name") == "mixed_float16":
                cfg["name"] = "float32"
            for value in obj.values():
                fix(value)
        elif isinstance(obj, list):
            for value in obj:
                fix(value)
        return obj

    new_model = tf.keras.Model.from_config(fix(model.get_config()))
    new_model.set_weights(model.get_weights())
    return new_model


def main():
    ok = True
    for label, filename, target_dir in JOBS:
        src = os.path.join(ROOT, "Models", "_downloads", filename)
        dst = os.path.join(ROOT, target_dir, filename)
        print(f"\n[{label}] {src}")
        if not os.path.exists(src):
            print("  NOT FOUND - download it from Google Drive into Models/_downloads/")
            ok = False
            continue
        original = tf.keras.models.load_model(src, compile=False)
        compact = to_float32(original)

        x = np.random.default_rng(0).uniform(-1, 1, (2,) + tuple(original.input_shape[1:])).astype("float32")
        p1, p2 = original.predict(x, verbose=0), compact.predict(x, verbose=0)
        diff = float(np.abs(p1 - p2).max())
        same_top1 = bool((p1.argmax(1) == p2.argmax(1)).all())
        print(f"  check: max probability difference {diff:.5f}, same top-1: {same_top1}")
        if not same_top1 or diff > 0.01:
            print("  Check failed - the compact copy was NOT written.")
            ok = False
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        compact.save(dst)
        print(f"  saved {dst} ({os.path.getsize(dst) / 1e6:.1f} MB, original {os.path.getsize(src) / 1e6:.1f} MB)")

    print("\nDone." if ok else "\nFinished with problems - see messages above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
