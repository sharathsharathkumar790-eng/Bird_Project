"""Optional frontend extras for Avian AI.

Each feature can be switched off on its own by changing its flag to ``False``.
With all three flags set to ``False`` the app looks and works exactly as it did
before these extras were added.

    LAYOUT_TIDY     joined upload card, richer right-hand panel, no Deploy button
    SAMPLE_BUTTONS  one-click sample images and recordings from ``test_samples``
    AUDIO_WAVEFORM  waveform with the 5-second windows the audio model used
    SIDEBAR_SLIM    shorter sidebar: models in use + tips (replaces the old cards)
    SPECIES_SEARCH  "Is my bird supported?" search in the sidebar
    SESSION_HISTORY last 5 predictions of this session in the sidebar
    IMAGE_HEATMAP   "Where the model looked" Grad-CAM overlay for photo results

To remove the extras completely, run ``restore_frontend_before_extras.bat``
from the app folder (see ``backups/frontend_before_extras/README.txt``).
"""

import glob
import html
import io
import os

import numpy as np
import streamlit as st

LAYOUT_TIDY = True
SAMPLE_BUTTONS = True
AUDIO_WAVEFORM = True
SIDEBAR_SLIM = True
SPECIES_SEARCH = True
SESSION_HISTORY = True
IMAGE_HEATMAP = True

SAMPLE_DIR = "test_samples"
SAMPLE_BUTTON_COUNT = 3

_CSS = """
<style>
[data-testid="stAppDeployButton"], .stAppDeployButton { display: none !important; }

.st-key-image_panel_joined, .st-key-audio_panel_joined {
    background-color: #FFFFFF;
    border: 1px solid #DCE3DC;
    border-radius: 14px;
    padding: 0 0 1.1rem 0;
    gap: 0.9rem;
}
.st-key-image_panel_joined .avian-panel, .st-key-audio_panel_joined .avian-panel {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    margin-bottom: 0 !important;
    padding-bottom: 1.2rem !important;
}
.st-key-image_panel_joined [data-testid="stFileUploader"],
.st-key-audio_panel_joined [data-testid="stFileUploader"] {
    padding: 0 1.25rem;
}

.avian-ready-card {
    background-color: #FFFFFF;
    border: 1px solid #DCE3DC;
    border-radius: 14px;
    padding: 1.3rem 1.4rem 1.1rem 1.4rem;
}
.avian-ready-title { font-size: 1.05rem; font-weight: 700; color: #1B4332; margin: 0 0 0.3rem 0; }
.avian-ready-body { font-size: 0.9rem; color: #52606D; margin: 0 0 1rem 0; line-height: 1.5; }
.avian-ready-steps { display: flex; flex-direction: column; gap: 0.65rem; margin-bottom: 1rem; }
.avian-ready-step { display: flex; gap: 0.75rem; align-items: flex-start; }
.avian-ready-num {
    flex-shrink: 0; width: 26px; height: 26px; border-radius: 999px;
    background-color: #1B4332; color: #FFFFFF; font-size: 0.78rem; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center;
}
.avian-ready-step-text { font-size: 0.88rem; color: #1F2933; line-height: 1.45; padding-top: 0.15rem; }
.avian-ready-step-text strong { color: #1B4332; }
.avian-ready-facts { display: flex; flex-wrap: wrap; gap: 0.5rem; padding-top: 0.85rem; border-top: 1px solid #DCE3DC; }
.avian-ready-fact {
    background-color: #E8F0EB; color: #1B4332; border-radius: 999px;
    padding: 0.25rem 0.7rem; font-size: 0.78rem; font-weight: 600;
}

[class*="st-key-image_sample_"] button p, [class*="st-key-audio_sample_"] button p {
    font-size: 0.8rem; white-space: normal; line-height: 1.25;
}
.avian-sample-label { font-size: 0.8rem; font-weight: 700; color: #52606D; margin: 0.2rem 0 0 0; letter-spacing: 0.02em; }
.avian-sample-active {
    background-color: #E8F0EB; border: 1px solid #B7E4C7; border-radius: 10px;
    padding: 0.55rem 0.8rem; font-size: 0.85rem; color: #1B4332;
}

.st-key-image_heatmap_box {
    background-color: #FFFFFF; border: 1px solid #DCE3DC; border-radius: 14px;
    padding: 1rem 1.15rem 0.8rem 1.15rem; margin-top: 0.85rem;
}
.avian-scale { display: flex; align-items: center; gap: 0.5rem; font-size: 0.76rem; color: #52606D; margin-top: 0.1rem; }
.avian-scale-bar { flex: 1; height: 8px; border-radius: 999px;
    background: linear-gradient(90deg, #440154, #3B528B, #21918C, #5EC962, #FDE725); }
.st-key-audio_wave_box {
    background-color: #FFFFFF; border: 1px solid #DCE3DC; border-radius: 14px;
    padding: 1rem 1.15rem 0.6rem 1.15rem; margin-top: 0.85rem;
}
.avian-wave-title { font-size: 1rem; font-weight: 700; color: #1B4332; margin: 0 0 0.2rem 0; }
.avian-wave-note { font-size: 0.82rem; color: #52606D; margin: 0; line-height: 1.45; }
.avian-sb-brand { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.25rem; }
.avian-sb-brand img { width: 42px; height: 42px; }
.avian-sb-name { font-size: 1.3rem !important; font-weight: 750; color: #1B4332; margin: 0 !important; line-height: 1.1; }
.avian-sb-tagline { font-size: 0.86rem; color: #52606D; margin: 0.35rem 0 0 0; line-height: 1.45; }
.avian-sb-section { border-top: 1px solid #CFDCD2; padding-top: 0.85rem; margin-top: 0.9rem; }
.avian-sb-heading {
    font-size: 0.72rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase;
    color: #2D6A4F; margin: 0 0 0.55rem 0;
}
.avian-sb-model { margin-bottom: 0.55rem; }
.avian-sb-model-name { font-size: 0.88rem; font-weight: 700; color: #1F2933; }
.avian-sb-model-meta { font-size: 0.8rem; color: #52606D; }
.avian-sb-list { margin: 0; padding-left: 1.05rem; font-size: 0.84rem; color: #1F2933; line-height: 1.5; }
.avian-sb-list li + li { margin-top: 0.25rem; }
.avian-sb-empty { font-size: 0.82rem; color: #52606D; margin: 0; }
.avian-sb-match { display: flex; justify-content: space-between; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0; border-bottom: 1px dashed #CFDCD2; font-size: 0.84rem; color: #1F2933; }
.avian-sb-match:last-child { border-bottom: none; }
.avian-sb-tags { display: flex; gap: 0.25rem; flex-shrink: 0; }
.avian-sb-tag { font-size: 0.68rem; font-weight: 700; border-radius: 999px; padding: 0.1rem 0.45rem; }
.avian-sb-tag.yes { background-color: #1B4332; color: #FFFFFF; }
.avian-sb-tag.no { background-color: #E3EBE4; color: #7C8B83; text-decoration: line-through; }
.avian-sb-count { font-size: 0.76rem; color: #52606D; margin: 0.2rem 0 0.3rem 0; }
.avian-sb-hist { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;
    padding: 0.45rem 0; border-bottom: 1px dashed #CFDCD2; }
.avian-sb-hist:last-child { border-bottom: none; }
.avian-sb-hist-species { font-size: 0.86rem; font-weight: 650; color: #1F2933; line-height: 1.3; }
.avian-sb-hist-file { font-size: 0.74rem; color: #52606D; word-break: break-all; }
.avian-sb-hist-right { text-align: right; flex-shrink: 0; }
.avian-sb-hist-kind { font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #2D6A4F; }
.avian-sb-hist-conf { font-size: 0.86rem; font-weight: 700; color: #1B4332; }
[data-testid="stSidebar"] .st-key-species_search input { font-size: 0.86rem; }
[data-testid="stSidebar"] p.avian-sb-empty, [data-testid="stSidebar"] p.avian-sb-count { font-size: 0.8rem !important; }
[data-testid="stSidebar"] .avian-sb-list li { font-size: 0.84rem !important; line-height: 1.5 !important; }
[data-testid="stSidebar"] p.avian-sb-tagline { font-size: 0.86rem !important; }
</style>
"""


def inject_css():
    """Add the extras' styles once per page run (only if any extra is on)."""
    if LAYOUT_TIDY or SAMPLE_BUTTONS or AUDIO_WAVEFORM or SIDEBAR_SLIM or SPECIES_SEARCH or SESSION_HISTORY \
            or IMAGE_HEATMAP:
        st.markdown(_CSS, unsafe_allow_html=True)


def joined_panel(kind):
    """Container that joins the upload card and uploader into one panel.

    Returns a plain container when LAYOUT_TIDY is off, so callers can always
    use ``with joined_panel(...):``.
    """
    if LAYOUT_TIDY:
        return st.container(key=f"{kind}_panel_joined")
    return st.container()


# ---------------- Ready panel (right-hand side before a prediction) ----------------

def _model_facts(kind):
    try:
        from src.active_models import AUDIO_INFO, NEW_MODELS_ACTIVE
    except Exception:
        NEW_MODELS_ACTIVE, AUDIO_INFO = False, {"species": 59, "test_accuracy": None}
    if kind == "image":
        facts = ["200 species", "MobileNetV2"]
        if NEW_MODELS_ACTIVE:
            facts.append("83.32% test accuracy")
    else:
        facts = [f"{AUDIO_INFO['species']} species", "Mel-spectrograms", "MobileNetV2"]
        if NEW_MODELS_ACTIVE and AUDIO_INFO.get("test_accuracy"):
            facts += [f"{AUDIO_INFO['test_accuracy']} test accuracy", "Up to 12 windows averaged"]
    return facts


def render_ready(kind, title, body, render_ready_state):
    """Richer ready card when LAYOUT_TIDY is on; original card otherwise."""
    if not LAYOUT_TIDY:
        render_ready_state(title=title, body=body)
        return

    if kind == "image":
        steps = [
            ("Upload", "a JPG or PNG photo where the bird is clearly visible."),
            ("Preview", "the image and check the bird fills a good part of the frame."),
            ("Predict", "to see the top match, the Top-3 scores and a species profile."),
        ]
    else:
        steps = [
            ("Upload", "a WAV, MP3, OGG, FLAC or M4A recording of the bird call."),
            ("Listen", "with the player to make sure the call is audible."),
            ("Predict", "to see the top match, the Top-3 scores and which parts of the recording were used."),
        ]
    if SAMPLE_BUTTONS and _sample_files(kind):
        steps[0] = (steps[0][0], steps[0][1] + " Or click one of the samples on the left.")

    steps_html = "".join(
        f"<div class='avian-ready-step'><span class='avian-ready-num'>{i}</span>"
        f"<span class='avian-ready-step-text'><strong>{html.escape(a)}</strong> {html.escape(b)}</span></div>"
        for i, (a, b) in enumerate(steps, start=1)
    )
    facts_html = "".join(f"<span class='avian-ready-fact'>{html.escape(f)}</span>" for f in _model_facts(kind))
    st.markdown(
        f"<div class='avian-ready-card'>"
        f"<div class='avian-ready-title'>{html.escape(title)}</div>"
        f"<p class='avian-ready-body'>{html.escape(body)}</p>"
        f"<div class='avian-ready-steps'>{steps_html}</div>"
        f"<div class='avian-ready-facts'>{facts_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------- Sample buttons ----------------

class SampleFile(io.BytesIO):
    """In-memory file that looks like a Streamlit upload (name, size, type)."""

    def __init__(self, path):
        with open(path, "rb") as handle:
            data = handle.read()
        super().__init__(data)
        self.name = os.path.basename(path)
        self.size = len(data)
        ext = os.path.splitext(self.name)[1].lower()
        self.type = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
            ".flac": "audio/flac", ".m4a": "audio/mp4",
        }.get(ext, "application/octet-stream")


def _sample_files(kind):
    if kind == "image":
        folder = os.path.join(SAMPLE_DIR, "images")
    else:
        try:
            from src.active_models import AUDIO_INFO
            folder = AUDIO_INFO["samples_dir"]
        except Exception:
            folder = os.path.join(SAMPLE_DIR, "audio")
    patterns = ["*.jpg", "*.jpeg", "*.png"] if kind == "image" else ["*.mp3", "*.wav", "*.ogg", "*.flac", "*.m4a"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    files = sorted(f for f in files if "_HARD_" not in os.path.basename(f))
    return files[:SAMPLE_BUTTON_COUNT]


def _sample_label(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        stem = parts[1]
    return stem.replace("_", " ")


def _select_sample(kind, path):
    st.session_state[f"{kind}_sample_path"] = path


def clear_sample(kind):
    st.session_state.pop(f"{kind}_sample_path", None)


def sample_picker(kind):
    """Show sample buttons (or the active sample) and return a SampleFile or None."""
    if not SAMPLE_BUTTONS:
        return None
    files = _sample_files(kind)
    if not files:
        return None

    active = st.session_state.get(f"{kind}_sample_path")
    if active and os.path.exists(active):
        cols = st.columns([3, 1.2], vertical_alignment="center")
        with cols[0]:
            st.markdown(
                f"<div class='avian-sample-active'>Using sample: <strong>{html.escape(_sample_label(active))}</strong></div>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.button("Clear sample", key=f"{kind}_sample_clear", on_click=clear_sample, args=(kind,),
                      use_container_width=True)
        return SampleFile(active)

    st.markdown("<p class='avian-sample-label'>No file handy? Try a sample:</p>", unsafe_allow_html=True)
    cols = st.columns(len(files))
    for col, path in zip(cols, files):
        with col:
            st.button(_sample_label(path), key=f"{kind}_sample_{os.path.basename(path)}",
                      on_click=_select_sample, args=(kind, path), use_container_width=True)
    return None


# ---------------- Audio waveform with the windows the model used ----------------

@st.cache_data(show_spinner=False, max_entries=8)
def _audio_windows(audio_bytes, filename):
    import tempfile

    import librosa
    from src.active_models import AUDIO_BACKEND as backend

    AUDIO_SAMPLE_RATE = backend.AUDIO_SAMPLE_RATE
    ENERGY_KEEP, MAX_SECONDS, MAX_WINDOWS = backend.ENERGY_KEEP, backend.MAX_SECONDS, backend.MAX_WINDOWS
    WINDOW_DURATION_SECONDS, HOP_DURATION_SECONDS = backend.WINDOW_DURATION_SECONDS, backend.HOP_DURATION_SECONDS

    full = backend.load_audio_waveform(audio_bytes, filename=filename)
    full_seconds = len(full) / AUDIO_SAMPLE_RATE
    try:  # true length of the whole recording (the loader may stop early)
        suffix = os.path.splitext(filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
        full_seconds = max(full_seconds, float(librosa.get_duration(path=tmp.name)))
        os.remove(tmp.name)
    except Exception:
        pass
    audio = full[: MAX_SECONDS * AUDIO_SAMPLE_RATE]

    # Same window selection as the active audio predictor's select_windows.
    win = int(WINDOW_DURATION_SECONDS * AUDIO_SAMPLE_RATE)
    hop = int(HOP_DURATION_SECONDS * AUDIO_SAMPLE_RATE)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    starts = list(range(0, len(audio) - win + 1, hop))
    energy = np.array([np.mean(audio[s:s + win] ** 2) for s in starts])
    order = np.argsort(-energy)
    keep = [starts[i] for i in order if energy[i] >= ENERGY_KEEP * energy[order[0]]][:MAX_WINDOWS]

    buckets = 900
    step = max(1, len(audio) // buckets)
    trimmed = audio[: (len(audio) // step) * step].reshape(-1, step)
    peak = np.abs(trimmed).max(axis=1)
    times = np.arange(len(peak)) * step / AUDIO_SAMPLE_RATE

    windows = [
        {"start": s / AUDIO_SAMPLE_RATE, "end": (s + win) / AUDIO_SAMPLE_RATE, "rank": r}
        for r, s in enumerate(keep, start=1)
    ]
    return {
        "times": times, "peak": peak, "windows": windows,
        "total_windows": len(starts), "full_seconds": full_seconds,
        "used_seconds": len(audio) / AUDIO_SAMPLE_RATE, "max_seconds": MAX_SECONDS,
    }


def render_audio_windows(audio_bytes, filename):
    """Waveform card showing which 5-second windows were averaged."""
    if not AUDIO_WAVEFORM or not audio_bytes:
        return
    try:
        from src.active_models import NEW_MODELS_ACTIVE
        if not NEW_MODELS_ACTIVE:
            return
        import altair as alt
        import pandas as pd

        data = _audio_windows(audio_bytes, filename)
    except Exception:
        return

    peak = data["peak"]
    scale = float(peak.max()) or 1.0
    wave = pd.DataFrame({"t": data["times"], "hi": peak / scale, "lo": -peak / scale})
    wins = pd.DataFrame(data["windows"])

    shading = alt.Chart(wins).mark_rect(color="#52B788", opacity=0.22).encode(
        x="start:Q", x2="end:Q",
        tooltip=[alt.Tooltip("rank:Q", title="Loudness rank"),
                 alt.Tooltip("start:Q", title="Start (s)", format=".1f"),
                 alt.Tooltip("end:Q", title="End (s)", format=".1f")],
    )
    waveform = alt.Chart(wave).mark_area(color="#1B4332", opacity=0.85).encode(
        x=alt.X("t:Q", title="Time (seconds)", scale=alt.Scale(domain=[0, data["used_seconds"]], nice=False)),
        y=alt.Y("hi:Q", axis=None, scale=alt.Scale(domain=[-1.05, 1.05])),
        y2="lo:Q",
    )
    chart = (shading + waveform).properties(height=150, background="#FFFFFF").configure_view(stroke=None).configure_axis(
        labelColor="#52606D", titleColor="#52606D", gridColor="#EEF2ED", domainColor="#DCE3DC",
    )

    used = len(data["windows"])
    note = (
        f"The green bands are the {used} loudest 5-second windows (out of {data['total_windows']}) "
        f"that the model classified and averaged. Darker green means windows overlap."
    )
    if data["full_seconds"] > data["max_seconds"] + 0.5:
        limit = data["max_seconds"]
        limit_text = f"{limit} seconds" if limit < 120 else f"{limit // 60} minutes"
        note += f" Only the first {limit_text} of this recording are used."

    with st.container(key="audio_wave_box"):
        st.markdown(
            "<div class='avian-wave-title'>What the model listened to</div>"
            f"<p class='avian-wave-note'>{html.escape(note)}</p>",
            unsafe_allow_html=True,
        )
        st.altair_chart(chart, use_container_width=True, theme=None)


_VIRIDIS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
], dtype=np.float32)


def _viridis(values):
    """Map values in [0, 1] to viridis RGB (uint8) without matplotlib."""
    v = np.clip(values, 0, 1) * (len(_VIRIDIS) - 1)
    lo = np.floor(v).astype(int)
    hi = np.minimum(lo + 1, len(_VIRIDIS) - 1)
    t = (v - lo)[..., None]
    return (_VIRIDIS[lo] * (1 - t) + _VIRIDIS[hi] * t).astype(np.uint8)


# ---------------- Image heatmap (Grad-CAM) ----------------

@st.cache_data(show_spinner=False, max_entries=8)
def _grad_cam(pixels):
    """Grad-CAM map (0..1) for the top predicted class of an RGB uint8 array."""
    import tensorflow as tf
    from src.active_models import load_image_model
    from src.image_predictor_v2 import preprocess_image

    from PIL import Image

    model, _ = load_image_model()
    base = next(layer for layer in model.layers if hasattr(layer, "layers") and layer.name.startswith("mobilenetv2"))
    gap = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D))
    dense = model.layers[-1]

    x = tf.convert_to_tensor(preprocess_image(Image.fromarray(pixels)))
    with tf.GradientTape() as tape:
        features = base(x, training=False)
        tape.watch(features)
        logits = tf.matmul(gap(features), dense.kernel) + dense.bias
        top = int(tf.argmax(logits[0]))
        score = logits[:, top]
    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(features * weights[:, None, None, :], axis=-1))[0].numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def render_image_heatmap(image):
    """'Where the model looked' card: Grad-CAM overlay on the uploaded photo."""
    if not IMAGE_HEATMAP or image is None:
        return
    try:
        from src.active_models import NEW_MODELS_ACTIVE
        if not NEW_MODELS_ACTIVE:
            return
        from PIL import Image

        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        cam = _grad_cam(pixels)
    except Exception:
        return

    h, w = pixels.shape[:2]
    cam_big = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                         dtype=np.float32) / 255.0
    colour = _viridis(cam_big).astype(np.float32)
    alpha = (0.15 + 0.55 * cam_big)[..., None]
    overlay = (pixels.astype(np.float32) * (1 - alpha) + colour * alpha).astype(np.uint8)

    with st.container(key="image_heatmap_box"):
        st.markdown(
            "<div class='avian-wave-title'>Where the model looked</div>"
            "<p class='avian-wave-note'>Grad-CAM heatmap for the top prediction. Yellow areas influenced the "
            "decision most, purple areas least. A good result focuses on the bird, not the background.</p>",
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            st.image(pixels, caption="Your photo", use_container_width=True)
        with right:
            st.image(overlay, caption="Model attention", use_container_width=True)
        st.markdown("<div class='avian-scale'><span>Less</span><span class='avian-scale-bar'></span><span>More</span></div>",
                    unsafe_allow_html=True)


# ---------------- Sidebar: slim layout, species search, session history ----------------

def _section(heading, body_html=""):
    return (f"<div class='avian-sb-section'><div class='avian-sb-heading'>{html.escape(heading)}</div>"
            f"{body_html}</div>")


def _models_html():
    try:
        from src.active_models import NEW_MODELS_ACTIVE
    except Exception:
        NEW_MODELS_ACTIVE = False
    try:
        from src.active_models import AUDIO_INFO
    except Exception:
        AUDIO_INFO = {"version": "", "species": 59, "test_accuracy": None}
    if NEW_MODELS_ACTIVE:
        rows = [("Image model v4", "MobileNetV2 · 200 species · 83.32% test accuracy"),
                (f"Audio model {AUDIO_INFO['version']}",
                 f"MobileNetV2 · {AUDIO_INFO['species']} species · {AUDIO_INFO['test_accuracy']} test accuracy")]
    else:
        rows = [("Image model", "MobileNetV2 · 200 species"),
                ("Audio model", f"MobileNetV2 · {AUDIO_INFO['species']} species")]
    return "".join(
        f"<div class='avian-sb-model'><div class='avian-sb-model-name'>{html.escape(a)}</div>"
        f"<div class='avian-sb-model-meta'>{html.escape(b)}</div></div>"
        for a, b in rows
    )


_TIPS_HTML = (
    "<ul class='avian-sb-list'>"
    "<li>Photos: one bird, centred, well lit, little clutter.</li>"
    "<li>Recordings: the call should be clearly audible over background noise.</li>"
    "<li>Low confidence usually means an unclear input or a species the model was not trained on.</li>"
    "</ul>"
)


def render_slim_sidebar(logo_html):
    """Draw the slim sidebar. Returns False (draws nothing) when SIDEBAR_SLIM is off."""
    if not SIDEBAR_SLIM:
        return False
    with st.sidebar:
        st.markdown(
            f"<div class='avian-sb-brand'>{logo_html}<p class='avian-sb-name'>Avian AI</p></div>"
            "<p class='avian-sb-tagline'>Identify bird species from photographs or bird-call recordings.</p>"
            + _section("Models in use", _models_html()),
            unsafe_allow_html=True,
        )
    render_sidebar_tools()
    with st.sidebar:
        st.markdown(_section("Tips for best results", _TIPS_HTML), unsafe_allow_html=True)
    return True


def render_sidebar_tools():
    """Species search and session history (each behind its own flag)."""
    with st.sidebar:
        if SPECIES_SEARCH:
            _render_species_search()
        if SESSION_HISTORY:
            st.markdown(_section("This session"), unsafe_allow_html=True)
            st.session_state["_avian_history_slot"] = st.empty()
            _draw_history()


@st.cache_data(show_spinner=False)
def _species_lists():
    import json
    from src.active_models import AUDIO_LABELS_PATH, IMAGE_LABELS_PATH
    from src.image_predictor import format_image_species_label

    with open(IMAGE_LABELS_PATH, "r", encoding="utf-8") as handle:
        image_names = [format_image_species_label(n) for n in json.load(handle)]
    with open(AUDIO_LABELS_PATH, "r", encoding="utf-8") as handle:
        audio_names = list(json.load(handle))
    return image_names, audio_names


def _render_species_search():
    st.markdown(_section("Is my bird supported?"), unsafe_allow_html=True)
    query = st.text_input(
        "Search species", key="species_search", placeholder="Type a name, e.g. jay",
        label_visibility="collapsed",
    ).strip().lower()
    if len(query) < 2:
        try:
            from src.active_models import AUDIO_INFO
            n_audio = AUDIO_INFO["species"]
        except Exception:
            n_audio = 59
        st.markdown(f"<p class='avian-sb-empty'>Checks the 200 photo species and {n_audio} call species.</p>",
                    unsafe_allow_html=True)
        return
    try:
        image_names, audio_names = _species_lists()
    except Exception:
        st.markdown("<p class='avian-sb-empty'>Species lists could not be loaded.</p>", unsafe_allow_html=True)
        return

    image_set = {n.lower() for n in image_names}
    audio_set = {n.lower() for n in audio_names}
    names = {}
    for n in image_names + audio_names:
        names.setdefault(n.lower(), n)
    matches = sorted(v for k, v in names.items() if query in k)

    if not matches:
        st.markdown("<p class='avian-sb-empty'>Not found in either model.</p>", unsafe_allow_html=True)
        return

    def tag(label, ok):
        return f"<span class='avian-sb-tag {'yes' if ok else 'no'}'>{label}</span>"

    shown = matches[:8]
    rows = "".join(
        f"<div class='avian-sb-match'><span>{html.escape(n)}</span><span class='avian-sb-tags'>"
        f"{tag('Photo', n.lower() in image_set)}{tag('Call', n.lower() in audio_set)}</span></div>"
        for n in shown
    )
    count = f"{len(matches)} match{'es' if len(matches) != 1 else ''}"
    if len(matches) > len(shown):
        count += f", showing first {len(shown)}"
    st.markdown(f"<p class='avian-sb-count'>{count}</p>{rows}", unsafe_allow_html=True)


def record_prediction(kind, filename, top3):
    """Store a finished prediction and refresh the sidebar history."""
    if not SESSION_HISTORY or not top3:
        return
    history = st.session_state.setdefault("avian_prediction_history", [])
    species, confidence = top3[0]
    history.insert(0, {"kind": kind, "file": filename or "", "species": species, "confidence": float(confidence)})
    del history[5:]
    _draw_history()


def _draw_history():
    slot = st.session_state.get("_avian_history_slot")
    if slot is None:
        return
    history = st.session_state.get("avian_prediction_history", [])
    if not history:
        slot.markdown("<p class='avian-sb-empty'>Your predictions will appear here.</p>", unsafe_allow_html=True)
        return
    rows = "".join(
        "<div class='avian-sb-hist'><div>"
        f"<div class='avian-sb-hist-species'>{html.escape(h['species'])}</div>"
        f"<div class='avian-sb-hist-file'>{html.escape(h['file'])}</div></div>"
        "<div class='avian-sb-hist-right'>"
        f"<div class='avian-sb-hist-kind'>{'Photo' if h['kind'] == 'image' else 'Call'}</div>"
        f"<div class='avian-sb-hist-conf'>{h['confidence']:.1f}%</div></div></div>"
        for h in history
    )
    slot.markdown(rows, unsafe_allow_html=True)
