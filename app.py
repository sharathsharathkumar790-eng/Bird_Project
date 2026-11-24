
import base64
import html
import os

# Disable TensorFlow oneDNN optimizations before importing TensorFlow-backed
# modules to avoid MKL depthwise-convolution primitive failures on Windows.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import requests
import streamlit as st

from src.active_models import NEW_MODELS_ACTIVE
from src import ui_extras as extras

if NEW_MODELS_ACTIVE:
    from analytics.analytics_v2 import render_analytics_dashboard
else:
    from analytics.analytics_ui import render_analytics_dashboard

from src.audio_prediction_ui import render_audio_prediction_component
# Model selection (new or original models) is controlled in src/active_models.py
from src.active_models import (
    AUDIO_LABELS_PATH,
    AUDIO_MODEL_PATH,
    IMAGE_LABELS_PATH,
    IMAGE_MODEL_PATH,
    load_audio_model,
    load_image_model,
    predict_audio_species,
    predict_image_species,
)
from src.image_prediction_ui import render_image_prediction_component

# ---------------- PAGE SETTINGS ----------------

LOGO_SVG_PATH = os.path.join("assets", "logo.svg")
FAVICON_PATH = os.path.join("assets", "favicon.png")

st.set_page_config(
    page_title="Avian AI Classifier",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else "🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

STYLES_PATH = os.path.join("assets", "styles.css")


def load_styles():
    """Load custom dashboard CSS from assets/styles.css if present."""
    if os.path.exists(STYLES_PATH):
        with open(STYLES_PATH, "r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def logo_img(css_class):
    """Return the Avian AI logo as an inline <img> tag, or an empty string if missing."""
    if not os.path.exists(LOGO_SVG_PATH):
        return ""
    with open(LOGO_SVG_PATH, "rb") as logo_file:
        encoded = base64.b64encode(logo_file.read()).decode("ascii")
    return f'<img class="{css_class}" src="data:image/svg+xml;base64,{encoded}" alt="Avian AI logo">'


def render_header():
    """Render the Avian AI hero header."""
    st.markdown(
        f"""
        <div class="avian-hero">
            {logo_img("avian-hero-logo")}
            <div class="avian-hero-text">
                <div class="avian-hero-brand">Avian AI · Wildlife Identification</div>
                <h1 class="avian-hero-title">Avian AI Species Identifier</h1>
                <p class="avian-hero-subtitle">
                    Identify wild bird species from photographs or field audio recordings.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the clean navigation and guidance sidebar."""
    # Optional slim sidebar, species search and history (src/ui_extras.py).
    if extras.render_slim_sidebar(logo_img("avian-sidebar-logo")):
        return

    with st.sidebar:
        st.markdown(
            """
            <div class="avian-sidebar-brand">
                <div class="avian-sidebar-logo-row">
                    """ + logo_img("avian-sidebar-logo") + """
                    <div class="avian-sidebar-badge">Avian AI</div>
                </div>
                <h2 class="avian-sidebar-title">Field Assistant</h2>
                <p class="avian-sidebar-tagline">
                    Independent photo and bird-call species identification.
                </p>
            </div>
            <div class="avian-sidebar-card">
                <h4>How to Use</h4>
                <ol>
                    <li>Select <strong>Image Prediction</strong> or <strong>Audio Prediction</strong>.</li>
                    <li>Upload a clear bird photograph or field audio recording.</li>
                    <li>Review the top match, Top-3 confidence scores, and species profile.</li>
                </ol>
            </div>
            <div class="avian-sidebar-card">
                <h4>Supported Media</h4>
                <ul>
                    <li><strong>Images:</strong> JPG, JPEG, PNG</li>
                    <li><strong>Audio:</strong> WAV, MP3, MPEG, OGG, FLAC, M4A</li>
                </ul>
            </div>
            <div class="avian-sidebar-card">
                <h4>Tips for Best Results</h4>
                <ul>
                    <li>Use centered, well-lit bird photos with minimal clutter.</li>
                    <li>Upload recordings where the bird call is clearly audible.</li>
                </ul>
            </div>
            <div class="avian-sidebar-footer">
                Avian AI · Smart Bird Species Identification
            </div>
            """,
            unsafe_allow_html=True,
        )
    extras.render_sidebar_tools()


load_styles()
render_sidebar()
render_header()


# ---------------- WIKIPEDIA INFORMATION ----------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_bird_information(species):

    try:
        headers = {
            "User-Agent": "AvianAI/1.0 (Bird Species Identification)"
        }

        search_response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": species,
                "limit": 1,
                "format": "json"
            },
            headers=headers,
            timeout=8
        )

        search_response.raise_for_status()
        search_data = search_response.json()

        if not search_data[1]:
            return None

        title = search_data[1][0]

        response = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + requests.utils.quote(title.replace(" ", "_")),
            headers=headers,
            timeout=8
        )

        response.raise_for_status()
        return response.json()

    except Exception:
        return None


# ---------------- SHARED UI COMPONENTS ----------------

def render_upload_panel(title, description, formats):
    """Render a styled upload section header with supported format tags."""
    tags = "".join(
        f"<span class='avian-format-tag'>{html.escape(fmt)}</span>"
        for fmt in formats
    )
    st.markdown(
        f"""
        <div class="avian-panel">
            <h3 class="avian-panel-title">{html.escape(title)}</h3>
            <p class="avian-panel-desc">{html.escape(description)}</p>
            <div class="avian-format-row">{tags}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ready_state(title, body):
    """Render an inviting placeholder card before a file is uploaded."""
    st.markdown(
        f"""
        <div class="avian-empty-state">
            <div class="avian-empty-title">{html.escape(title)}</div>
            <p class="avian-empty-body">{html.escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Confidence bands used only for colouring the results (not part of the model).
CONFIDENCE_HIGH = 70.0
CONFIDENCE_MEDIUM = 40.0


def confidence_level(confidence):
    """Return (css class, label) for a confidence percentage."""
    value = float(confidence)
    if value >= CONFIDENCE_HIGH:
        return "high", "High"
    if value >= CONFIDENCE_MEDIUM:
        return "medium", "Medium"
    return "low", "Low"


def render_primary_prediction(species, confidence, modality_label, note=None):
    """Render the primary predicted species and confidence hero card."""
    level_class, level_text = confidence_level(confidence)
    note_html = (
        f"<div class='avian-result-note'>{html.escape(note)}</div>"
        if note
        else ""
    )
    st.markdown(
        f"""
        <div class="avian-result-hero">
            <div class="avian-result-kicker">{html.escape(modality_label)}</div>
            <div class="avian-result-row">
                <div>
                    <h2 class="avian-species-name">{html.escape(species)}</h2>
                </div>
                <div class="avian-confidence-box conf-{level_class}">
                    <div class="avian-confidence-value">{confidence:.2f}%</div>
                    <div class="avian-confidence-label">{level_text} confidence</div>
                </div>
            </div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top3_predictions(predictions):
    """Render Top-3 ranked prediction cards with proportional confidence bars."""
    st.markdown(
        "<div class='avian-section-heading'>Top 3 Predictions</div>",
        unsafe_allow_html=True,
    )

    cards_html = []
    for rank, (species, confidence) in enumerate(predictions, start=1):
        clamped = min(max(float(confidence), 0.0), 100.0)
        cards_html.append(
            f"""
            <div class="avian-rank-card rank-{rank} conf-{confidence_level(confidence)[0]}">
                <div class="avian-rank-top">
                    <div class="avian-rank-left">
                        <span class="avian-rank-badge">#{rank}</span>
                        <span class="avian-rank-species">{html.escape(species)}</span>
                    </div>
                    <span class="avian-rank-score">{confidence:.2f}%</span>
                </div>
                <div class="avian-bar-track">
                    <div class="avian-bar-fill" style="width: {clamped:.2f}%;"></div>
                </div>
            </div>
            """
        )

    st.markdown("".join(cards_html), unsafe_allow_html=True)


def render_bird_information(species):
    """Render Wikipedia species thumbnail and summary when available."""
    st.markdown(
        "<div class='avian-section-heading'>Species Information</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching species profile..."):
        data = get_bird_information(species)

    if data:
        thumbnail_url = (data.get("thumbnail") or {}).get("source")
        extract = data.get("extract") or "No description available."
        page_url = (
            ((data.get("content_urls") or {}).get("desktop") or {}).get("page")
        )

        with st.container(border=True):
            if thumbnail_url:
                img_col, text_col = st.columns([1, 2], gap="medium")
                with img_col:
                    st.image(thumbnail_url, use_container_width=True)
                with text_col:
                    st.markdown(
                        f"<p class='avian-info-text'>{html.escape(extract)}</p>",
                        unsafe_allow_html=True,
                    )
                    if page_url:
                        st.markdown(
                            f"<a class='avian-info-link' href='{html.escape(page_url)}' target='_blank' rel='noopener noreferrer'>Read full Wikipedia article →</a>",
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    f"<p class='avian-info-text'>{html.escape(extract)}</p>",
                    unsafe_allow_html=True,
                )
                if page_url:
                    st.markdown(
                        f"<a class='avian-info-link' href='{html.escape(page_url)}' target='_blank' rel='noopener noreferrer'>Read full Wikipedia article →</a>",
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            """
            <div class="avian-info-card">
                <p class="avian-info-text">
                    Additional species details are currently unavailable from Wikipedia.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_results(top3, modality_label, source_name, note=None):
    """Store session analytics state and render unified prediction results."""
    st.session_state.last_top3 = [
        {
            "Species": name,
            "Confidence": score
        }
        for name, score in top3
    ]
    st.session_state.last_prediction_source = source_name

    render_primary_prediction(
        species=top3[0][0],
        confidence=top3[0][1],
        modality_label=modality_label,
        note=note,
    )
    render_top3_predictions(top3)
    render_bird_information(top3[0][0])


# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs([
    "Image Prediction",
    "Audio Prediction",
    "Model Analytics"
])


# ======================================================
# IMAGE PREDICTION
# ======================================================

with tab1:
    render_image_prediction_component(
        render_upload_panel=render_upload_panel,
        render_ready_state=render_ready_state,
        render_prediction_results=render_prediction_results,
    )


# ======================================================
# AUDIO PREDICTION
# ======================================================

with tab2:
    render_audio_prediction_component(
        render_upload_panel=render_upload_panel,
        render_ready_state=render_ready_state,
        render_prediction_results=render_prediction_results,
    )


# ======================================================
# MODEL ANALYTICS
# ======================================================

with tab3:
    render_analytics_dashboard(
        render_ready_state=render_ready_state,
        render_top3_predictions=render_top3_predictions,
    )
