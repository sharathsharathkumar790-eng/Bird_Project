# Avian AI — MULTIMODAL BIRD SPECIES IDENTIFICATION
USING DEEP LEARNING WITH IMAGE AND AUDIO
ANALYSIS

An interactive AI-powered wildlife identification dashboard built with **Python 3.11**, **Streamlit**, and **TensorFlow / Keras**. Avian AI provides two independent deep-learning identification workflows—one for bird photographs and one for field audio recordings—alongside ranked Top-3 confidence scores, Wikipedia species profiles, and a Model Analytics dashboard.

---

## Project Overview & Objectives

Identifying bird species in the field often relies on either visual observation or acoustic vocalizations. Avian AI brings both capabilities into a single, easy-to-use Streamlit web interface while keeping the image and audio pipelines strictly independent.

**Core Objectives:**
- Enable fast bird species identification from uploaded photographs across **200 bird species** (image model v4, test accuracy **83.32%**, Top-5 **95.87%**).
- Enable bird call and song identification from field audio recordings across **70 bird species** using 5-second sliding windows, Mel spectrogram analysis and averaging over the loudest windows (audio model v3, trained on BirdCLEF 2022; test accuracy **84.89%** on 1,138 unseen recordings, Top-3 **93.23%**).
- Present clear **Top-3 predictions** with confidence percentages and contextual **Wikipedia species information**.
- Provide an honest **Model Analytics** dashboard for inspecting verified model specifications, optional exported training/evaluation metrics, and live session prediction breakdowns.

---

## Key Features

- **Image-Based Bird Species Identification:** Upload `JPG`, `JPEG`, or `PNG` photographs, preview the image with resolution/file metadata, and run MobileNetV2 classification across 200 bird classes.
- **Audio-Based Bird Species Identification:** Upload `WAV`, `MP3`, `MPEG`, `OGG`, `FLAC`, or `M4A` recordings, play back the audio in-app, and classify vocalizations across 70 bird species.
- **Independent Image and Audio Prediction Branches:** Each modality operates independently—users do not need to upload both an image and an audio recording together (no image-audio fusion is performed).
- **Audio Multi-Window Analysis:** Resamples audio to `32,000` Hz mono and uses the first 60 seconds, splits it into 5-second windows with a 2.5-second hop (zero-padding shorter clips), drops windows below 10% of the loudest window's energy, keeps up to the 12 loudest windows, converts each into a 128-band Mel spectrogram (`n_fft=2048`, `hop_length=512`, 50 Hz–14 kHz, fixed 80 dB scale) resized to `224×224×3`, and averages the model's predictions over all windows.
- **Switchable Models:** The improved models are used by default; the original models can be restored by setting `USE_NEW_MODELS = False` in `src/active_models.py`.
- **Top-3 Predictions & Confidence Scores:** Displays the top predicted species prominently alongside ranked `#1`, `#2`, and `#3` confidence cards.
- **Wikipedia Bird Information:** Automatically retrieves species summaries, thumbnails, and article links from the Wikipedia API when available.
- **Model Analytics Dashboard:** Displays model dimensions and class counts, an original-vs-improved comparison, training curves, per-species results and confusion matrices from the real result files in `reports_v2/` (image) and `reports_v3/` (audio), and the latest session Top-3 confidence chart.

---

## Technologies & Libraries Used

| Category | Technology / Library | Version in `.venv` |
|---|---|---|
| Language & Runtime | Python | `3.11.3` |
| Web Dashboard | Streamlit | `1.67.0` |
| Deep Learning | TensorFlow / Keras (MobileNetV2) | `2.20.0` |
| Audio Processing | Librosa | `0.11.0` |
| Image & Matrix Processing | OpenCV (`opencv-python`), Pillow, NumPy | `5.0.0.93`, `13.0.0`, `2.4.6` |
| Data & Charts | Pandas, Altair | `3.1.1`, `6.3.0` |
| External Species Lookup | Requests (Wikipedia REST / OpenSearch API) | `2.34.2` |

---

## Project Directory Structure

```text
Avian_AI_Streamlit/
│
├── .streamlit/
│   └── config.toml                     # Streamlit theme configuration
├── analytics/
│   ├── __init__.py
│   ├── analytics_backend.py            # Original models: Excel report loading & model metadata
│   ├── analytics_ui.py                 # Original models: Model Analytics dashboard UI
│   └── analytics_v2.py                 # Improved models: Model Analytics dashboard (reads reports_v2/ and reports_v3/)
├── assets/
│   └── styles.css                      # Custom dashboard styling
├── data/                               # Optional local data directory
├── docs/                               # Optional documentation directory
├── Models/
│   ├── image_v4/                       # Improved image model (used by default)
│   │   ├── bird_image_model_v4.keras
│   │   └── bird_class_names.json
│   ├── audio_v3/                       # Audio model v3 (used by default)
│   │   ├── bird_audio_model_v3.keras
│   │   └── bird_audio_labels_v3.json
│   ├── audio_v2/                       # Audio model v2 (fallback, AUDIO_MODEL_VERSION = "v2")
│   │   ├── bird_audio_model_v2.keras
│   │   └── bird_audio_labels.json
│   ├── _downloads/                     # Original Colab exports used by tools/prepare_new_models.py
│   ├── image/                          # Original image model (kept as fallback)
│   │   ├── bird_mobilenetv2_final.keras
│   │   └── bird_class_names.json
│   └── audio/                          # Original audio model (kept as fallback)
│       ├── bird_audio_finetuned.keras
│       └── bird_audio_labels.json
├── reports/                            # Original models: Excel evaluation reports
├── reports_v2/                         # Image model v4 and audio model v2: results, histories, plots
│   ├── image/
│   ├── audio/
│   └── model_comparison.json
├── reports_v3/                         # Audio model v3: results, history, confusion matrix, splits
│   └── audio/
├── src/
│   ├── __init__.py
│   ├── active_models.py                # Selects improved (default) or original models; AUDIO_MODEL_VERSION picks v3 or v2
│   ├── image_predictor.py              # Original image model: loading, preprocessing & prediction
│   ├── image_predictor_v2.py           # Improved image model v4 (320×320 input)
│   ├── image_prediction_ui.py          # Image Bird Identification workspace
│   ├── audio_predictor.py              # Original audio model: peak-energy window & prediction
│   ├── audio_predictor_v2.py           # Audio model v2: multi-window averaging (22,050 Hz)
│   ├── audio_predictor_v3.py           # Audio model v3: 32 kHz, 50 Hz–14 kHz, multi-window averaging
│   └── audio_prediction_ui.py          # Audio Bird Identification workspace
├── test_samples/                       # Test-set images and recordings with expected results
├── tests/                              # Optional tests directory
├── tools/
│   └── prepare_new_models.py           # One-time conversion of the Colab models to compact float32 copies
├── app.py                              # Main Streamlit application entry point
├── requirements.txt                    # Pinned Python dependencies
└── README.md                           # Project documentation
```

---

## Model Details

The app uses the **improved models** below. The original models are kept unchanged in `Models/image` and `Models/audio` and can be re-enabled in `src/active_models.py`.

### 1. Image Classification Model (v4)
- **Model File:** `Models/image_v4/bird_image_model_v4.keras` (19.5 MB, float32)
- **Label File:** `Models/image_v4/bird_class_names.json`
- **Architecture:** MobileNetV2 (alpha 1.4, ImageNet weights) + GlobalAveragePooling + Dropout(0.4) + Dense(200, softmax)
- **Input / Output:** `(None, 320, 320, 3)` scaled with `mobilenet_v2.preprocess_input` → `(None, 200)`
- **Dataset:** 11,788 images of 200 species, stratified split 8,251 / 1,768 / 1,769 (seed 42)
- **Training:** augmentation (flip, rotation, zoom, translation, contrast), Mixup (α = 0.2), label smoothing 0.1; three stages: head only (lr 1e-3) → top 60 layers (cosine lr 1e-4) → all layers (cosine lr 1e-5)
- **Test results:** accuracy **83.32%**, Top-5 **95.87%**, macro F1 **0.833** (original model: 60.63%)

### 2. Audio Classification Model (v3)
- **Model File:** `Models/audio_v3/bird_audio_model_v3.keras` (18.5 MB, float32)
- **Label File:** `Models/audio_v3/bird_audio_labels_v3.json` (70 species, alphabetical by common name)
- **Architecture:** MobileNetV2 (alpha 1.4, ImageNet weights) on Mel spectrograms + GlobalAveragePooling + Dropout(0.4) + Dense(70, softmax, L2 1e-4)
- **Input / Output:** `(None, 224, 224, 3)` spectrogram scaled to [-1, 1] → `(None, 70)`
- **Dataset:** BirdCLEF 2022 (Kaggle; Xeno-canto recordings, 32 kHz, 152 species, 14,852 recordings, no duplicate files). The 70 species with at least 50 recordings were kept and capped at 300 recordings each, giving 11,332 recordings split by recording 9,056 / 1,138 / 1,138 (80/10/10 per species, seed 42).
- **Training:** up to 5 loudest 5-second windows per training recording (35,001 windows), SpecAugment (time shift, ±6 dB gain, 2 frequency and 2 time masks), Mixup (α = 0.2), label smoothing 0.1, square-root inverse-frequency class weights, three-stage fine-tuning (head → top 60 layers → all layers, 67.5 minutes on a T4 GPU); model selected by recording-level validation accuracy.
- **Test results (1,138 unseen recordings):** accuracy **84.89%**, Top-3 **93.23%**, Top-5 **95.08%**, macro F1 **0.818**, weighted F1 **0.845**; the same model using only the single loudest window reaches 80.05%.
- **Previous audio model (v2):** 59 species from the Kaggle *Sound of 114 Species of Birds Till 2022* subset (930 unique recordings after removing 749 duplicates), test accuracy 61.46% on 96 recordings. It is kept in `Models/audio_v2` and can be selected with `AUDIO_MODEL_VERSION = "v2"` in `src/active_models.py`. The two models use different datasets and species, so their accuracies are not a like-for-like comparison.

### 3. Preparing the models (one time)
The Colab exports (about 85 MB each) are placed in `Models/_downloads/` and converted to compact float32 copies with:

```powershell
.\.venv\Scripts\python.exe tools\prepare_new_models.py
```

The script checks that the converted models give the same predictions before saving them. If the improved model files are missing, the app automatically falls back to the original models.

---

## Installation & Setup (Python 3.11)

### 1. Prerequisites
- **Python 3.11** installed on your system.
- Local clone or copy of the `Avian_AI_Streamlit` directory containing the trained files under `Models/`.

### 2. Virtual Environment Setup

Navigate to the project root (`D:\Avian_AI_Streamlit`):

```powershell
cd D:\Avian_AI_Streamlit
```

If `.venv` does not already exist, create and activate a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If `.venv` already exists in the project folder, simply activate it with `.\.venv\Scripts\Activate.ps1`.)*

### 3. Dependency Installation

With the virtual environment activated, install the pinned dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

---

## Running the Streamlit Application

Run the application locally from the project root (`D:\Avian_AI_Streamlit`):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or, with `.venv` activated:

```powershell
streamlit run app.py
```

Streamlit will start a local development server and open the dashboard in your browser (typically at `http://localhost:8501`).

> **Note:** This project is configured for local execution and has not been publicly deployed.

---

## How to Use the Dashboard

### 1. Image Prediction (`Image Prediction` tab)
1. Upload a bird photograph (`JPG`, `JPEG`, or `PNG`).
2. Review the image preview and file resolution details.
3. Click **Predict Bird Species**.
4. View the top predicted species, confidence percentage, Top-3 ranked prediction cards, and Wikipedia species profile.

### 2. Audio Prediction (`Audio Prediction` tab)
1. Upload a bird audio recording (`WAV`, `MP3`, `MPEG`, `OGG`, `FLAC`, or `M4A`).
2. Use the built-in audio player to preview the recording.
3. Click **Predict Bird Species**.
4. The backend cuts the recording into 5-second windows (2.5-second step), classifies up to the 12 loudest windows, averages their predictions, and displays the top predicted species, confidence score, Top-3 ranked matches, and Wikipedia species profile.

Sample files for testing (taken from the test sets, never seen in training) are in `test_samples/`, with the expected results in `test_samples/README.txt`.

### 3. Model Analytics (`Model Analytics` tab)
- **Overall Model Overview:** Displays verified model filenames, class counts, input/output dimensions, and file sizes.
- **Original vs Improved Models:** Test accuracy, Top-5 and macro F1 of the original and improved models, with notes on how each audio result was measured.
- **Image Model (v4) / Audio Model (v3):** Metric cards and training curves across all three stages, read from the real result files in `reports_v2/image/` and `reports_v3/audio/`.
- **Image vs Audio:** Test accuracy compared with random chance for 200 and 70 classes.
- **Latest Session Prediction Breakdown:** Visualizes the Top-3 confidence distribution from your most recent Image or Audio prediction in the current session.

---

## Current Limitations & Future Enhancements

### Current Limitations
- **Separate Species Coverage:** The image model covers 200 species while the audio model covers a distinct set of 70 species; predictions are independent and do not fuse audio and image inputs.
- **Similar Species:** Most image errors are between near-identical species (for example Common Raven / American Crow, gulls, terns, flycatchers); most audio errors are between related waterbirds, for example Gadwall / Mallard, Green-winged / Blue-winged Teal, terns and sandpipers.
- **Audio Window Limit:** Only the first 60 seconds of a recording are analysed, so a bird that calls only later in a long recording can be missed.
- **External Wikipedia Dependency:** Species descriptions and thumbnails require an active internet connection and matching Wikipedia article titles.
- **Local Execution:** Configured for local use via Streamlit rather than cloud deployment.

### Future Enhancements
- Add the remaining BirdCLEF species with fewer recordings using few-shot techniques or extra xeno-canto recordings.
- Evaluate the audio model with k-fold cross-validation for a more stable accuracy estimate.
- Expand the audio model's species coverage to align more closely with visual species classes.
- Add automated unit and integration tests inside `tests/`.
- Package and deploy the application to a cloud hosting environment if remote access is required.
