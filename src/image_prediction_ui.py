"""Frontend UI component for Image Bird Identification in Avian AI.

Connects the Streamlit upload/preview workflow to the image prediction backend
in ``src.image_predictor``:
    Upload image -> Preview image -> Click Predict -> Backend predicts -> Render results.
"""

import html

import streamlit as st

from src.active_models import load_uploaded_image, predict_image_species
from src import ui_extras as extras

SUPPORTED_IMAGE_FORMATS = ["JPG", "JPEG", "PNG"]


def render_image_prediction_component(
    render_upload_panel,
    render_ready_state,
    render_prediction_results,
):
    """Render the complete Image Bird Identification workspace.

    Frontend-to-backend communication:
    - Calls ``load_uploaded_image(uploaded_file)`` to safely decode and validate
      the user's upload into an RGB PIL Image for preview and inference.
    - Tracks the currently uploaded file via ``(name, size)`` in
      ``st.session_state`` so stale predictions are cleared when the user
      uploads or removes an image.
    - Waits for an explicit click on the **Predict Bird Species** button before
      invoking ``predict_image_species(image)`` from ``src.image_predictor``.
    - Passes the returned Top-3 predictions to ``render_prediction_results`` to
      render the primary species card, Top-3 ranked cards, and Wikipedia profile.

    Args:
        render_upload_panel: Shared UI helper for the upload instructions card.
        render_ready_state: Shared UI helper for empty/ready placeholder cards.
        render_prediction_results: Shared UI helper for prediction & bird info cards.
    """
    # Component heading and concise feature overview.
    st.markdown(
        """
        <div class="avian-component-header">
            <h2 class="avian-component-title">Image Bird Identification</h2>
            <p class="avian-component-subtitle">
                Upload a clear bird photograph, preview your image, and run species identification across 200 bird species.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    extras.inject_css()

    left_col, right_col = st.columns([1, 1.35], gap="large")

    image = None
    file_signature = None

    with left_col:
        with extras.joined_panel("image"):
            render_upload_panel(
                title="Upload Bird Image",
                description=(
                    "Select or drag a well-lit photograph where the bird is clearly visible. "
                    "After previewing the image below, click Predict Bird Species."
                ),
                formats=SUPPORTED_IMAGE_FORMATS,
            )

            uploaded_file = st.file_uploader(
                "Upload Bird Image",
                type=["jpg", "jpeg", "png"],
                key="image_upload",
                label_visibility="collapsed",
            )

        # Optional one-click samples (src/ui_extras.py, SAMPLE_BUTTONS).
        if uploaded_file is not None:
            extras.clear_sample("image")
        else:
            uploaded_file = extras.sample_picker("image")

        # Reset stored image prediction when the uploaded file is removed or replaced.
        if uploaded_file is not None:
            file_signature = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("image_file_signature") != file_signature:
                st.session_state.image_file_signature = file_signature
                st.session_state.pop("image_prediction_result", None)
                st.session_state.pop("image_prediction_error", None)

            try:
                image = load_uploaded_image(uploaded_file)

                # Display clean preview using native Streamlit image component.
                with st.container(key="image_preview_box"):
                    st.image(
                        image,
                        caption="Image Preview",
                        use_container_width=True,
                    )
                width, height = image.size
                size_kb = uploaded_file.size / 1024
                st.markdown(
                    f"""
                    <div class="avian-meta-strip">
                        <span><strong>File:</strong> {html.escape(uploaded_file.name)}</span>
                        <span><strong>Resolution:</strong> {width} × {height} px</span>
                        <span><strong>Size:</strong> {size_kb:.1f} KB</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Explicit Predict action required before running model inference.
                st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)
                predict_clicked = st.button(
                    "Predict Bird Species",
                    type="primary",
                    use_container_width=True,
                    key="predict_image_btn",
                )

                if predict_clicked:
                    with right_col:
                        with st.spinner("Analyzing bird image and predicting species..."):
                            try:
                                top3 = predict_image_species(image)
                                st.session_state.image_prediction_result = top3
                                extras.record_prediction("image", uploaded_file.name, top3)
                                st.session_state.pop("image_prediction_error", None)
                            except Exception as exc:
                                st.session_state.pop("image_prediction_result", None)
                                st.session_state.image_prediction_error = str(exc)

            except Exception as exc:
                st.session_state.pop("image_prediction_result", None)
                st.session_state.image_prediction_error = str(exc)
        else:
            st.session_state.pop("image_file_signature", None)
            st.session_state.pop("image_prediction_result", None)
            st.session_state.pop("image_prediction_error", None)

    with right_col:
        if uploaded_file is None:
            extras.render_ready(
                "image",
                render_ready_state=render_ready_state,
                title="Ready for Image Identification",
                body=(
                    "Upload a JPG, JPEG, or PNG bird photograph on the left, review the preview, "
                    "and click Predict Bird Species to view the Top-3 matches and species details."
                ),
            )
        elif st.session_state.get("image_prediction_error"):
            st.error(f"Image prediction failed: {st.session_state.image_prediction_error}")
        elif st.session_state.get("image_prediction_result"):
            render_prediction_results(
                top3=st.session_state.image_prediction_result,
                modality_label="Image Identification",
                source_name="Image Prediction",
            )
            # Optional Grad-CAM heatmap (src/ui_extras.py, IMAGE_HEATMAP).
            extras.render_image_heatmap(image)
        elif image is not None:
            extras.render_ready(
                "image",
                render_ready_state=render_ready_state,
                title="Image Ready — Click Predict",
                body=(
                    "Your image preview is loaded. Click 'Predict Bird Species' below the preview "
                    "to run identification and view the Top-3 predictions and bird profile."
                ),
            )
