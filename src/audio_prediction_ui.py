"""Frontend UI component for Audio Bird Identification in Avian AI.

Connects the Streamlit audio upload and playback workspace to the backend
pipeline in ``src.audio_predictor``:
    Upload audio -> Validate & play audio -> Click Predict ->
    Sliding-window peak-energy Mel spectrogram -> Trained model predicts ->
    Render primary species card, Top-3 ranked cards, and species profile.
"""

import html
import os

import streamlit as st

from src.active_models import AUDIO_INFO, AUDIO_RESULT_NOTE, predict_audio_species, validate_audio_upload
from src import ui_extras as extras

SUPPORTED_AUDIO_FORMATS = ["WAV", "MP3", "MPEG", "OGG", "FLAC", "M4A"]


def render_audio_prediction_component(
    render_upload_panel,
    render_ready_state,
    render_prediction_results,
):
    """Render the complete Audio Bird Identification workspace.

    Frontend-to-backend communication:
    - Validates uploaded audio with ``validate_audio_upload(audio_file)`` and
      renders an embedded audio player alongside file metadata.
    - Tracks the current upload via ``(name, size)`` in ``st.session_state``
      so stale results or errors clear whenever a recording changes or is removed.
    - Waits for the user to click **Predict Bird Species** before invoking
      ``predict_audio_species(audio_bytes, filename=filename)``.
    - Passes the returned Top-3 predictions to ``render_prediction_results`` for
      consistent presentation matching Image Bird Identification.

    Args:
        render_upload_panel: Shared UI helper for the upload instructions card.
        render_ready_state: Shared UI helper for empty/ready placeholder cards.
        render_prediction_results: Shared UI helper for prediction & bird info cards.
    """
    st.markdown(
        f"""
        <div class="avian-component-header">
            <h2 class="avian-component-title">Audio Bird Identification</h2>
            <p class="avian-component-subtitle">
                Upload a bird call or song recording, listen back, and identify species across {AUDIO_INFO["species"]} bird classes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    extras.inject_css()

    left_col, right_col = st.columns([1, 1.35], gap="large")

    audio_bytes = None
    audio_filename = None

    with left_col:
        with extras.joined_panel("audio"):
            render_upload_panel(
                title="Upload Bird Audio",
                description=(
                    "Select or drag a field recording where the bird vocalization is clearly audible. "
                    "Preview playback below, then click Predict Bird Species."
                ),
                formats=SUPPORTED_AUDIO_FORMATS,
            )

            audio_file = st.file_uploader(
                "Upload Bird Audio",
                type=["wav", "mp3", "mpeg", "ogg", "flac", "m4a"],
                key="audio_upload",
                label_visibility="collapsed",
            )

        # Optional one-click samples (src/ui_extras.py, SAMPLE_BUTTONS).
        if audio_file is not None:
            extras.clear_sample("audio")
        else:
            audio_file = extras.sample_picker("audio")

        if audio_file is not None:
            file_signature = (audio_file.name, audio_file.size)
            if st.session_state.get("audio_file_signature") != file_signature:
                st.session_state.audio_file_signature = file_signature
                st.session_state.pop("audio_prediction_result", None)
                st.session_state.pop("audio_prediction_error", None)

            try:
                audio_bytes, audio_filename = validate_audio_upload(audio_file)

                # Playback control and uploaded audio details.
                st.audio(audio_file)
                size_kb = audio_file.size / 1024
                file_ext = os.path.splitext(audio_filename)[1].lstrip(".").upper() or "AUDIO"
                st.markdown(
                    f"""
                    <div class="avian-meta-strip">
                        <span><strong>File:</strong> {html.escape(audio_filename)}</span>
                        <span><strong>Format:</strong> {html.escape(file_ext)}</span>
                        <span><strong>Size:</strong> {size_kb:.1f} KB</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Explicit Predict action matching the Image Prediction workflow.
                st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)
                predict_clicked = st.button(
                    "Predict Bird Species",
                    type="primary",
                    use_container_width=True,
                    key="predict_audio_btn",
                )

                if predict_clicked:
                    with right_col:
                        with st.spinner("Processing audio and identifying bird species..."):
                            try:
                                top3 = predict_audio_species(
                                    audio_bytes,
                                    filename=audio_filename,
                                )
                                st.session_state.audio_prediction_result = top3
                                extras.record_prediction("audio", audio_filename, top3)
                                st.session_state.pop("audio_prediction_error", None)
                            except Exception as exc:
                                st.session_state.pop("audio_prediction_result", None)
                                st.session_state.audio_prediction_error = str(exc)

            except Exception as exc:
                st.session_state.pop("audio_prediction_result", None)
                st.session_state.audio_prediction_error = str(exc)
        else:
            st.session_state.pop("audio_file_signature", None)
            st.session_state.pop("audio_prediction_result", None)
            st.session_state.pop("audio_prediction_error", None)

    with right_col:
        if audio_file is None:
            extras.render_ready(
                "audio",
                render_ready_state=render_ready_state,
                title="Ready for Audio Identification",
                body=(
                    "Upload a bird audio recording on the left, preview it with the player, "
                    "and click Predict Bird Species to view the Top-3 matches and species details."
                ),
            )
        elif st.session_state.get("audio_prediction_error"):
            st.error(f"Audio prediction failed: {st.session_state.audio_prediction_error}")
        elif st.session_state.get("audio_prediction_result"):
            render_prediction_results(
                top3=st.session_state.audio_prediction_result,
                modality_label="Audio Identification",
                source_name="Audio Prediction",
                note=AUDIO_RESULT_NOTE,
            )
            # Optional waveform view (src/ui_extras.py, AUDIO_WAVEFORM).
            extras.render_audio_windows(audio_bytes, audio_filename)
        elif audio_bytes is not None:
            extras.render_ready(
                "audio",
                render_ready_state=render_ready_state,
                title="Audio Ready — Click Predict",
                body=(
                    "Your audio file is loaded and ready for playback. Click 'Predict Bird Species' "
                    "below the audio details to run identification and view the Top-3 predictions."
                ),
            )
