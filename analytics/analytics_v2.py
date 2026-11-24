"""Minimal Model Analytics page for the improved models (image v4, audio v3 or v2).

Reads only real result files exported by the training notebooks
(``reports_v2/image`` for the image model; ``reports_v3/audio`` for audio v3,
``reports_v2/audio`` for audio v2 - see ``src/active_models.py``). The original
analytics page (``analytics/analytics_ui.py``) is kept unchanged for the original models.
"""

import html
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from analytics.analytics_ui import (
    ACCENT_CORAL,
    FOREST_GREEN,
    _render_section_title,
    _render_session_prediction_section,
)

from src.active_models import AUDIO_INFO

REPORTS_V2 = "reports_v2"
REPORT_DIRS = {
    "image": os.path.join(REPORTS_V2, "image"),
    "audio": AUDIO_INFO.get("reports_dir") or os.path.join(REPORTS_V2, "audio"),
}

MODELS = {
    "image": {
        "title": "Image Model",
        "subtitle": "MobileNetV2 · 200 species · 320×320 photo input",
        "acc_key": "test_accuracy", "topk_key": "test_top5_accuracy", "topk_label": "Top-5 accuracy",
        "n_key": "test_samples", "n_label": "Test images",
        "val_col": "val_accuracy",
    },
    "audio": {
        "title": "Audio Model",
        "subtitle": f"MobileNetV2 · {AUDIO_INFO['species']} species · Mel-spectrogram input",
        "acc_key": "test_accuracy_recording", "topk_key": "test_top3_recording", "topk_label": "Top-3 accuracy",
        "n_key": "test_recordings", "n_label": "Test recordings",
        "val_col": "val_rec_acc",
    },
}


@st.cache_data(ttl=300, show_spinner=False)
def _load(kind):
    folder = REPORT_DIRS[kind]
    try:
        with open(os.path.join(folder, "test_results.json"), "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception:
        results = None
    try:
        history = pd.read_csv(os.path.join(folder, "history_all_stages.csv"))
    except Exception:
        history = None
    return results, history


def _accuracy_chart(history, val_col):
    """Training vs validation accuracy over all epochs."""
    cols = {"accuracy": "Training", val_col: "Validation"}
    df = history[["global_epoch"] + [c for c in cols if c in history.columns]].rename(columns=cols)
    df = df.melt("global_epoch", var_name="Series", value_name="Accuracy")
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("global_epoch:Q", title="Epoch"),
            y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Training", "Validation"], range=[FOREST_GREEN, ACCENT_CORAL]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("global_epoch:Q", title="Epoch"), "Series:N", alt.Tooltip("Accuracy:Q", format=".2%")],
        )
        .properties(height=220)
    )


def _render_metric_cards(items):
    """Metric tiles as single-line HTML (indented HTML would be shown as a code block)."""
    tiles = "".join(
        f'<div class="avian-metric-tile"><div class="avian-metric-label">{html.escape(label)}</div>'
        f'<div class="avian-metric-value">{html.escape(str(value))}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="avian-metric-grid">{tiles}</div>', unsafe_allow_html=True)


def _render_model(kind, render_ready_state):
    cfg = MODELS[kind]
    results, history = _load(kind)
    _render_section_title(cfg["title"], cfg["subtitle"])
    if not results:
        render_ready_state(title=f"{cfg['title']} results unavailable",
                           body=f"Expected file: {os.path.join(REPORT_DIRS[kind], 'test_results.json')}")
        return
    _render_metric_cards([
        ("Test accuracy", f"{results[cfg['acc_key']] * 100:.2f}%"),
        (cfg["topk_label"], f"{results[cfg['topk_key']] * 100:.2f}%"),
        ("Macro F1-score", f"{results['macro_f1']:.3f}"),
        (cfg["n_label"], f"{results[cfg['n_key']]:,}"),
    ])
    if history is not None and "global_epoch" in history.columns:
        st.altair_chart(_accuracy_chart(history, cfg["val_col"]), use_container_width=True)


def render_analytics_dashboard(render_ready_state, render_top3_predictions):
    st.markdown(
        """
        <div class="avian-component-header">
            <h2 class="avian-component-title">Model Analytics</h2>
            <p class="avian-component-subtitle">Test results and training progress of the image and audio models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_img, col_aud = st.columns(2, gap="large")
    with col_img:
        _render_model("image", render_ready_state)
    with col_aud:
        _render_model("audio", render_ready_state)
    st.caption("All results are measured on test data that the models did not see during training.")

    st.markdown("<div class='avian-section-spacer'></div>", unsafe_allow_html=True)
    _render_session_prediction_section(render_ready_state, render_top3_predictions)
