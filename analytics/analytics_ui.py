"""Frontend UI component for the Avian AI Model Analytics Dashboard.

Presents verified model specifications, genuine training history curves and metrics
from ``reports/Training_History.xlsx``, side-by-side cross-model comparisons from
``reports/Model_Comparison_Image_vs_Audio.xlsx``, and detailed per-class evaluation
reports from ``reports/Image_Model_Per_Class_Report.xlsx`` and
``reports/Audio_Model_Per_Class_Report.xlsx``.
"""

import html

import altair as alt
import pandas as pd
import streamlit as st

from analytics.analytics_backend import load_all_analytics_data, load_models_overview

FOREST_GREEN = "#1B4332"
SAGE_GREEN = "#52796F"
ACCENT_GREEN = "#2D6A4F"
LIGHT_BG = "#E8F0EB"
ACCENT_CORAL = "#D08C5C"


def _format_metric(value, is_percent_candidate=False):
    """Format a verified float metric cleanly without fabricating values."""
    if value is None:
        return "N/A"
    try:
        val_float = float(value)
    except (ValueError, TypeError):
        return str(value)

    if is_percent_candidate and 0.0 <= val_float <= 1.0:
        return f"{val_float * 100:.2f}%"
    if is_percent_candidate:
        return f"{val_float:.2f}%"
    return f"{val_float:.4f}"


def _render_section_title(title, subtitle=None):
    """Render a consistent section heading and optional subtitle."""
    sub_html = (
        f"<p class='avian-analytics-sub'>{html.escape(subtitle)}</p>"
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div class="avian-analytics-section">
            <h3 class="avian-section-heading" style="margin-bottom: 0.2rem;">{html.escape(title)}</h3>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_cards(items):
    """Render a responsive row of clean metric cards."""
    cards = []
    for label, value in items:
        cards.append(
            f"""
            <div class="avian-metric-tile">
                <div class="avian-metric-label">{html.escape(label)}</div>
                <div class="avian-metric-value">{html.escape(str(value))}</div>
            </div>
            """
        )
    st.markdown(
        f"<div class='avian-metric-grid'>{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# Section A: Model Overview
# =====================================================================

def _render_model_overview_section(models_overview):
    """Section A: Display verified structural details for Image and Audio models."""
    _render_section_title(
        "Overall Model Overview",
        "Verified model configuration loaded directly from saved model and label files.",
    )

    col1, col2 = st.columns(2, gap="large")
    for col, key in ((col1, "image"), (col2, "audio")):
        info = models_overview[key]
        with col:
            if info["available"]:
                size_text = f"{info['file_size_mb']} MB" if info["file_size_mb"] is not None else "N/A"
                st.markdown(
                    f"""
                    <div class="avian-overview-card">
                        <div class="avian-result-kicker">{html.escape(info['display_name'])}</div>
                        <h4 class="avian-overview-name">{html.escape(info['model_name'])}</h4>
                        <div class="avian-kv-list">
                            <div class="avian-kv-row">
                                <span class="avian-kv-key">Number of Classes</span>
                                <span class="avian-kv-val">{info['num_classes']} species</span>
                            </div>
                            <div class="avian-kv-row">
                                <span class="avian-kv-key">Input Dimensions</span>
                                <span class="avian-kv-val"><code>{html.escape(info['input_shape'])}</code></span>
                            </div>
                            <div class="avian-kv-row">
                                <span class="avian-kv-key">Output Dimensions</span>
                                <span class="avian-kv-val"><code>{html.escape(info['output_shape'])}</code></span>
                            </div>
                            <div class="avian-kv-row">
                                <span class="avian-kv-key">Model File Size</span>
                                <span class="avian-kv-val">{html.escape(size_text)}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(
                    f"Could not inspect {info['display_name']}: {info['error']}"
                )


# =====================================================================
# Interactive Altair Chart Helpers
# =====================================================================

def _build_training_line_chart(df, columns, title, y_title, is_accuracy=False):
    """Build a titled Altair line chart with axis labels, point tooltips, and legend."""
    available = [col for col in columns if col in df.columns]
    if not available or "epoch" not in df.columns:
        return None

    melted = df[["epoch"] + available].melt(
        id_vars="epoch",
        value_vars=available,
        var_name="Series",
        value_name="Value",
    )

    label_map = {
        "accuracy": "Training Accuracy",
        "val_accuracy": "Validation Accuracy",
        "loss": "Training Loss",
        "val_loss": "Validation Loss",
    }
    melted["Series"] = melted["Series"].map(lambda c: label_map.get(c, c))

    color_scale = alt.Scale(
        domain=[label_map.get(c, c) for c in available],
        range=[FOREST_GREEN, ACCENT_CORAL][: len(available)],
    )

    y_scale = alt.Scale(zero=False)
    if is_accuracy:
        y_scale = alt.Scale(domain=[0.0, 1.0], clamp=True)

    chart = (
        alt.Chart(melted)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Value:Q", title=y_title, scale=y_scale),
            color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title="Metric", orient="top")),
            tooltip=[
                alt.Tooltip("epoch:Q", title="Epoch"),
                alt.Tooltip("Series:N", title="Metric"),
                alt.Tooltip("Value:Q", title="Value", format=".4f"),
            ],
        )
        .properties(title=title, height=290)
        .configure_title(color=FOREST_GREEN, fontSize=14, anchor="start")
    )
    return chart


# =====================================================================
# Section B: Image Model Analytics
# =====================================================================

def _render_image_model_history(image_history, training_runs_summary, render_ready_state):
    """Section B: Render Image Model training analytics from Training_History.xlsx."""
    _render_section_title(
        "Image Model Analytics",
        "Genuine training history curves and metrics loaded directly from reports/Training_History.xlsx.",
    )

    if image_history.get("error"):
        st.warning(image_history["error"])
        return

    df = image_history.get("df")
    summary = image_history.get("summary")
    all_runs = image_history.get("all_runs", {})

    if df is None or summary is None:
        render_ready_state(
            title="Image Model Analytics Unavailable",
            body=(
                f"No training history file was found at '{image_history['expected_path']}'. "
                "Ensure Training_History.xlsx is located inside the 'reports/' folder."
            ),
        )
        return

    st.caption(f"Data source: `{image_history['source_path']}` · Model: MobileNetV2 (200 bird species)")

    # Display primary summary metrics for the final deployed model
    best_val_acc_str = _format_metric(summary.get("best_val_accuracy"), is_percent_candidate=True)
    if summary.get("best_val_acc_epoch"):
        best_val_acc_str += f" (ep {summary['best_val_acc_epoch']})"

    best_val_loss_str = _format_metric(summary.get("best_val_loss"))
    if summary.get("best_val_loss_epoch"):
        best_val_loss_str += f" (ep {summary['best_val_loss_epoch']})"

    _render_metric_cards([
        ("Best Val Accuracy", best_val_acc_str),
        ("Best Val Loss", best_val_loss_str),
        ("Final Train Accuracy", _format_metric(summary.get("train_accuracy"), is_percent_candidate=True)),
        ("Final Val Accuracy", _format_metric(summary.get("val_accuracy"), is_percent_candidate=True)),
        ("Final Train Loss", _format_metric(summary.get("train_loss"))),
        ("Final Val Loss", _format_metric(summary.get("val_loss"))),
    ])

    # Run selection dropdown for deep exploration
    run_options = ["Final Deployed Model (23 Epochs: Stage 1 + Stage 2 + Extra)"]
    run_label_to_key = {}
    label_descriptions = {
        "IMG_12": "Final Stage 1 (Frozen, lr 1e-3, 10 ep)",
        "IMG_13": "Final Stage 2 (Fine-tune, lr 1e-5, 10 ep)",
        "IMG_14": "Final Stage 2 Extra (+3 Epochs)",
        "IMG_01": "Baseline Frozen (10 ep)",
        "IMG_03": "Fine-tune 100/54 lr1e-5 (10 ep)",
        "IMG_10": "Two-stage S2 Full (10 ep)",
    }
    for k, desc in label_descriptions.items():
        if k in all_runs:
            opt_label = f"{k}: {desc}"
            run_options.append(opt_label)
            run_label_to_key[opt_label] = k

    selected_run = st.selectbox(
        "Select Training Run to Visualize:",
        options=run_options,
        index=0,
        key="image_run_selector",
        help="Explore the full 23-epoch final model progression or examine individual training stages.",
    )

    # Determine active dataset based on selection
    if selected_run.startswith("Final"):
        active_df = df
        active_title_suffix = "Final Deployed Model (23 Epochs)"
    else:
        run_key = run_label_to_key[selected_run]
        active_df = all_runs[run_key]["df"]
        active_title_suffix = selected_run

    # Interactive Accuracy and Loss Curves
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            chart = _build_training_line_chart(
                active_df,
                ["accuracy", "val_accuracy"],
                f"Accuracy Curve across Epochs — {active_title_suffix}",
                "Accuracy",
                is_accuracy=True,
            )
            if chart:
                st.altair_chart(chart, use_container_width=True)

    with col2:
        with st.container(border=True):
            chart = _build_training_line_chart(
                active_df,
                ["loss", "val_loss"],
                f"Loss Curve across Epochs — {active_title_suffix}",
                "Loss",
                is_accuracy=False,
            )
            if chart:
                st.altair_chart(chart, use_container_width=True)

    # Epoch metrics table in an expander
    with st.expander(f"View Epoch Metrics Table ({active_title_suffix})", expanded=False):
        display_cols = [c for c in ("epoch", "Stage", "accuracy", "val_accuracy", "loss", "val_loss", "learning_rate") if c in active_df.columns]
        st.dataframe(
            active_df[display_cols].rename(columns={
                "epoch": "Epoch",
                "accuracy": "Train Accuracy",
                "val_accuracy": "Val Accuracy",
                "loss": "Train Loss",
                "val_loss": "Val Loss",
                "learning_rate": "Learning Rate",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # All runs summary table in an expander
    if training_runs_summary is not None and not training_runs_summary.empty:
        with st.expander("Summary of All 14 Image Training Runs (from Training_History.xlsx)", expanded=False):
            st.caption("Overview of all exploratory training runs recorded in the project notebook.")
            st.dataframe(training_runs_summary, use_container_width=True, hide_index=True)


# =====================================================================
# Section C: Audio Model Analytics
# =====================================================================

def _render_audio_model_history(audio_history, render_ready_state):
    """Section C: Render Audio Model training analytics from Training_History.xlsx."""
    _render_section_title(
        "Audio Model Analytics",
        "Genuine training history curves and metrics loaded directly from reports/Training_History.xlsx.",
    )

    if audio_history.get("error"):
        st.warning(audio_history["error"])
        return

    fine_df = audio_history.get("df")
    fine_summary = audio_history.get("summary")
    base_df = audio_history.get("baseline_df")
    base_summary = audio_history.get("baseline_summary")

    if fine_df is None and base_df is None:
        render_ready_state(
            title="Audio Model Analytics Unavailable",
            body=(
                f"No training history file was found at '{audio_history['expected_path']}'. "
                "Ensure Training_History.xlsx contains worksheets 'AUD_15' and 'AUD_16'."
            ),
        )
        return

    st.caption(f"Data source: `{audio_history['source_path']}` · Model: MobileNetV2 on 5s Mel spectrograms (59 species)")

    # Model variant toggle (Fine-tuned vs Baseline)
    variant_choice = st.radio(
        "Audio Model Variant:",
        options=[
            "Fine-Tuned Model (AUD_16 · 5 Epochs, Deployed)",
            "Baseline Model (AUD_15 · 10 Epochs, Frozen Backbone)",
            "Baseline vs Fine-Tuned Side-by-Side",
        ],
        horizontal=True,
        key="audio_variant_radio",
    )

    if variant_choice.startswith("Fine-Tuned"):
        active_df = fine_df
        active_summary = fine_summary
        variant_title = "Fine-Tuned Model (Final Deployed, lr 1e-5)"
    elif variant_choice.startswith("Baseline Model"):
        active_df = base_df
        active_summary = base_summary
        variant_title = "Baseline Model (Frozen Backbone, lr 1e-3)"
    else:
        active_df = None
        active_summary = fine_summary
        variant_title = "Side-by-Side Comparison"

    if active_summary:
        best_val_acc_str = _format_metric(active_summary.get("best_val_accuracy"), is_percent_candidate=True)
        if active_summary.get("best_val_acc_epoch"):
            best_val_acc_str += f" (ep {active_summary['best_val_acc_epoch']})"

        best_val_loss_str = _format_metric(active_summary.get("best_val_loss"))
        if active_summary.get("best_val_loss_epoch"):
            best_val_loss_str += f" (ep {active_summary['best_val_loss_epoch']})"

        _render_metric_cards([
            ("Best Val Accuracy", best_val_acc_str),
            ("Best Val Loss", best_val_loss_str),
            ("Final Train Accuracy", _format_metric(active_summary.get("train_accuracy"), is_percent_candidate=True)),
            ("Final Val Accuracy", _format_metric(active_summary.get("val_accuracy"), is_percent_candidate=True)),
            ("Final Train Loss", _format_metric(active_summary.get("train_loss"))),
            ("Final Val Loss", _format_metric(active_summary.get("val_loss"))),
        ])

    if active_df is not None:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            with st.container(border=True):
                chart = _build_training_line_chart(
                    active_df,
                    ["accuracy", "val_accuracy"],
                    f"Accuracy Curve across Epochs — {variant_title}",
                    "Accuracy",
                    is_accuracy=True,
                )
                if chart:
                    st.altair_chart(chart, use_container_width=True)

        with col2:
            with st.container(border=True):
                chart = _build_training_line_chart(
                    active_df,
                    ["loss", "val_loss"],
                    f"Loss Curve across Epochs — {variant_title}",
                    "Loss",
                    is_accuracy=False,
                )
                if chart:
                    st.altair_chart(chart, use_container_width=True)

        with st.expander(f"View Epoch Metrics Table ({variant_title})", expanded=False):
            display_cols = [c for c in ("epoch", "accuracy", "val_accuracy", "loss", "val_loss", "learning_rate") if c in active_df.columns]
            st.dataframe(
                active_df[display_cols].rename(columns={
                    "epoch": "Epoch",
                    "accuracy": "Train Accuracy",
                    "val_accuracy": "Val Accuracy",
                    "loss": "Train Loss",
                    "val_loss": "Val Loss",
                    "learning_rate": "Learning Rate",
                }),
                use_container_width=True,
                hide_index=True,
            )
    else:
        # Side-by-side comparison table
        st.markdown("##### Baseline vs Fine-Tuned Audio Model Performance")
        comp_rows = [
            {"Metric": "Training Strategy", "Baseline Model (AUD_15)": "Frozen Backbone (lr 1e-3)", "Fine-Tuned Model (AUD_16)": "Fine-Tuned ~30 layers (lr 1e-5)"},
            {"Metric": "Epochs Trained", "Baseline Model (AUD_15)": str(base_summary.get("epochs", "10")), "Fine-Tuned Model (AUD_16)": str(fine_summary.get("epochs", "5"))},
            {"Metric": "Best Validation Accuracy", "Baseline Model (AUD_15)": _format_metric(base_summary.get("best_val_accuracy"), is_percent_candidate=True), "Fine-Tuned Model (AUD_16)": _format_metric(fine_summary.get("best_val_accuracy"), is_percent_candidate=True)},
            {"Metric": "Best Validation Loss", "Baseline Model (AUD_15)": _format_metric(base_summary.get("best_val_loss")), "Fine-Tuned Model (AUD_16)": _format_metric(fine_summary.get("best_val_loss"))},
            {"Metric": "Final Validation Accuracy", "Baseline Model (AUD_15)": _format_metric(base_summary.get("val_accuracy"), is_percent_candidate=True), "Fine-Tuned Model (AUD_16)": _format_metric(fine_summary.get("val_accuracy"), is_percent_candidate=True)},
            {"Metric": "Final Validation Loss", "Baseline Model (AUD_15)": _format_metric(base_summary.get("val_loss")), "Fine-Tuned Model (AUD_16)": _format_metric(fine_summary.get("val_loss"))},
            {"Metric": "Final Training Accuracy", "Baseline Model (AUD_15)": _format_metric(base_summary.get("train_accuracy"), is_percent_candidate=True), "Fine-Tuned Model (AUD_16)": _format_metric(fine_summary.get("train_accuracy"), is_percent_candidate=True)},
        ]
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# =====================================================================
# Section D: Model Comparison
# =====================================================================

def _render_comparison_section(comparison, render_ready_state):
    """Section D: Render cross-model comparison with explicit scope and limitations."""
    _render_section_title(
        "Model Comparison",
        "Cross-modality comparison populated from reports/Model_Comparison_Image_vs_Audio.xlsx.",
    )

    if not comparison.get("comparable"):
        render_ready_state(
            title="Model Comparison Unavailable",
            body=(
                f"No model comparison report was found at '{comparison['expected_path']}'. "
                "Ensure Model_Comparison_Image_vs_Audio.xlsx exists inside 'reports/'."
            ),
        )
        return

    st.caption(f"Data source: `{comparison['source_path']}`")

    # STRICT REQUIREMENT: Explicit comparison limitations and independent operation disclaimer
    st.info(
        "**Important Architectural & Comparison Notice:**\n\n"
        "- **Independent Operation:** The **Image Model** (200 species) and **Audio Model** (59 species) operate strictly as "
        "independent classifiers. They do **NOT** form a fused, ensembled, or combined prediction model.\n"
        "- **Comparison Limitations:** Higher raw test accuracy in the Audio model (69.05% vs 60.63%) does **not** signify superior performance. "
        "The Audio model classifies among 59 species (random guessing baseline = 1.69%), whereas the Image model solves a 200-species problem "
        "(random guessing baseline = 0.50%). When normalized against random chance, the Image model achieves **121.3×** better than chance "
        "compared to **40.7×** for the Audio model."
    )

    # High-level comparison metric cards
    _render_metric_cards([
        ("Image Test Accuracy", "60.63% (200 species)"),
        ("Audio Test Accuracy", "69.05% (59 species)"),
        ("Image Macro F1", "60.00%"),
        ("Audio Macro F1", "64.00%"),
        ("Image vs Random Chance", "121.26× above chance"),
        ("Audio vs Random Chance", "40.74× above chance"),
    ])

    # Side-by-side comparison charts
    col1, col2 = st.columns(2, gap="large")
    with col1:
        if comparison.get("accuracy_df") is not None:
            with st.container(border=True):
                acc_chart = (
                    alt.Chart(comparison["accuracy_df"])
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=52)
                    .encode(
                        x=alt.X("Model:N", title="Model", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Accuracy:Q", title="Test Accuracy", scale=alt.Scale(domain=[0.0, 1.0])),
                        color=alt.Color("Model:N", scale=alt.Scale(range=[FOREST_GREEN, SAGE_GREEN]), legend=None),
                        tooltip=["Model:N", alt.Tooltip("Accuracy:Q", title="Accuracy", format=".2%")],
                    )
                    .properties(title="Final Test Accuracy Comparison", height=270)
                    .configure_title(color=FOREST_GREEN, fontSize=14, anchor="start")
                )
                st.altair_chart(acc_chart, use_container_width=True)

    with col2:
        if comparison.get("loss_df") is not None:
            with st.container(border=True):
                loss_chart = (
                    alt.Chart(comparison["loss_df"])
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=52)
                    .encode(
                        x=alt.X("Model:N", title="Model", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Loss:Q", title="Test Loss"),
                        color=alt.Color("Model:N", scale=alt.Scale(range=[FOREST_GREEN, SAGE_GREEN]), legend=None),
                        tooltip=["Model:N", alt.Tooltip("Loss:Q", title="Loss", format=".4f")],
                    )
                    .properties(title="Final Test Loss Comparison", height=270)
                    .configure_title(color=FOREST_GREEN, fontSize=14, anchor="start")
                )
                st.altair_chart(loss_chart, use_container_width=True)

    # Macro Precision, Recall, F1 grouped bar chart
    if comparison.get("metrics_df") is not None:
        with st.container(border=True):
            f1_chart = (
                alt.Chart(comparison["metrics_df"])
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Metric:N", title="Evaluation Metric"),
                    y=alt.Y("Value:Q", title="Score", scale=alt.Scale(domain=[0.0, 1.0])),
                    color=alt.Color("Model:N", scale=alt.Scale(range=[FOREST_GREEN, SAGE_GREEN])),
                    xOffset="Model:N",
                    tooltip=["Model:N", "Metric:N", alt.Tooltip("Value:Q", format=".2%")],
                )
                .properties(title="Macro Precision, Recall, and F1-Score Comparison", height=280)
                .configure_title(color=FOREST_GREEN, fontSize=14, anchor="start")
            )
            st.altair_chart(f1_chart, use_container_width=True)

    # Chance normalization table
    if comparison.get("chance_df") is not None:
        st.markdown("##### Task Difficulty & Random Chance Normalization")
        st.dataframe(comparison["chance_df"], use_container_width=True, hide_index=True)

    # Full Image vs Audio comparison table from workbook
    if comparison.get("comparison_table") is not None and not comparison["comparison_table"].empty:
        with st.expander("Full Specification & Comparison Matrix (from Workbook)", expanded=False):
            st.dataframe(comparison["comparison_table"], use_container_width=True, hide_index=True)

    # Candidates exploration expanders
    if comparison.get("candidates_image") is not None and not comparison["candidates_image"].empty:
        with st.expander("Image Model Candidate Variants (from Workbook)", expanded=False):
            st.dataframe(comparison["candidates_image"], use_container_width=True, hide_index=True)

    if comparison.get("candidates_audio") is not None and not comparison["candidates_audio"].empty:
        with st.expander("Audio Model Candidate Variants (from Workbook)", expanded=False):
            st.dataframe(comparison["candidates_audio"], use_container_width=True, hide_index=True)


# =====================================================================
# Section E: Evaluation Results
# =====================================================================

def _render_evaluation_section(image_eval, audio_eval, render_ready_state):
    """Section E: Render genuine evaluation reports, class ranks, and confusion analyses."""
    _render_section_title(
        "Evaluation Results",
        "Detailed test set evaluation reports from Image_Model_Per_Class_Report.xlsx and Audio_Model_Per_Class_Report.xlsx.",
    )

    has_img = image_eval.get("summary") is not None or image_eval.get("per_class_df") is not None
    has_aud = audio_eval.get("summary") is not None or audio_eval.get("per_class_df") is not None

    if not has_img and not has_aud:
        render_ready_state(
            title="Evaluation Reports Unavailable",
            body=(
                "No saved evaluation workbooks were found inside 'reports/'. "
                "Ensure Image_Model_Per_Class_Report.xlsx and Audio_Model_Per_Class_Report.xlsx are present."
            ),
        )
        return

    # ---------------- Image Model Evaluation Block ----------------
    st.markdown("#### Image Model Evaluation (200 Species)")
    if image_eval.get("source_path"):
        st.caption(f"Data source: `{image_eval['source_path']}`")

    if image_eval.get("summary"):
        s = image_eval["summary"]
        _render_metric_cards([
            ("Macro Precision", _format_metric(s.get("precision"), is_percent_candidate=True)),
            ("Macro Recall", _format_metric(s.get("recall"), is_percent_candidate=True)),
            ("Macro F1-Score", _format_metric(s.get("f1_score"), is_percent_candidate=True)),
            ("Top-1 Test Accuracy", _format_metric(s.get("top1_accuracy"), is_percent_candidate=True)),
            ("Top-5 Test Accuracy", _format_metric(s.get("top5_accuracy"), is_percent_candidate=True)),
            ("Test Loss", _format_metric(s.get("test_loss"))),
        ])

    img_tabs = st.tabs([
        "Per-Class Performance (200 Species)",
        "Strongest Classes (Top 25)",
        "Weakest Classes (Bottom 25)",
        "Confusion Analysis & Error Pairs",
    ])

    with img_tabs[0]:
        if image_eval.get("per_class_df") is not None and not image_eval["per_class_df"].empty:
            st.caption("Complete test-set precision, recall, and F1-score for all 200 species.")
            st.dataframe(image_eval["per_class_df"], use_container_width=True, hide_index=True)

    with img_tabs[1]:
        if image_eval.get("strongest_df") is not None and not image_eval["strongest_df"].empty:
            st.caption("Top 25 species with the highest test classification scores.")
            st.dataframe(image_eval["strongest_df"], use_container_width=True, hide_index=True)

    with img_tabs[2]:
        if image_eval.get("weakest_df") is not None and not image_eval["weakest_df"].empty:
            st.caption("Bottom 25 species with the lowest test classification scores.")
            st.dataframe(image_eval["weakest_df"], use_container_width=True, hide_index=True)

    with img_tabs[3]:
        # Honest confusion matrix disclosure
        st.info(image_eval["confusion_matrix_note"])
        if image_eval.get("top_confusions_df") is not None and not image_eval["top_confusions_df"].empty:
            st.markdown("##### Most Common Misclassification Pairs (Off-Diagonal Confusions)")
            st.dataframe(image_eval["top_confusions_df"], use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ---------------- Audio Model Evaluation Block ----------------
    st.markdown("#### Audio Model Evaluation (59 Species)")
    if audio_eval.get("source_path"):
        st.caption(f"Data source: `{audio_eval['source_path']}`")

    if audio_eval.get("summary"):
        s = audio_eval["summary"]
        _render_metric_cards([
            ("Macro Precision", _format_metric(s.get("precision"), is_percent_candidate=True)),
            ("Macro Recall", _format_metric(s.get("recall"), is_percent_candidate=True)),
            ("Macro F1-Score", _format_metric(s.get("f1_score"), is_percent_candidate=True)),
            ("Fine-Tuned Accuracy", _format_metric(s.get("finetuned_accuracy"), is_percent_candidate=True)),
            ("Baseline Accuracy", _format_metric(s.get("baseline_accuracy"), is_percent_candidate=True)),
            ("Test Loss", _format_metric(s.get("finetuned_loss"))),
        ])

    aud_tabs = st.tabs([
        "Per-Class Performance (59 Species)",
        "Lowest Recall Species",
        "Misclassification Examples",
        "Confusion Analysis",
    ])

    with aud_tabs[0]:
        if audio_eval.get("per_class_df") is not None and not audio_eval["per_class_df"].empty:
            st.caption("Per-class baseline evaluation report across all 59 bird sound categories.")
            st.dataframe(audio_eval["per_class_df"], use_container_width=True, hide_index=True)

    with aud_tabs[1]:
        if audio_eval.get("lowest_recall_df") is not None and not audio_eval["lowest_recall_df"].empty:
            st.caption("10 species with lowest recall on the fine-tuned audio model.")
            st.dataframe(audio_eval["lowest_recall_df"], use_container_width=True, hide_index=True)

    with aud_tabs[2]:
        if audio_eval.get("misclassifications_df") is not None and not audio_eval["misclassifications_df"].empty:
            st.caption("Documented misclassified test recordings with actual vs predicted labels and model confidence.")
            st.dataframe(audio_eval["misclassifications_df"], use_container_width=True, hide_index=True)

    with aud_tabs[3]:
        # Honest confusion matrix disclosure
        st.info(audio_eval["confusion_matrix_note"])


# =====================================================================
# Section F: Latest Session Prediction Breakdown (Unchanged)
# =====================================================================

def _render_session_prediction_section(render_ready_state, render_top3_predictions):
    """Display the latest interactive session Top-3 confidence breakdown if available."""
    _render_section_title(
        "Latest Session Prediction Breakdown",
        "Confidence distribution from your most recent interactive Image or Audio prediction in this session.",
    )

    if "last_top3" in st.session_state:
        source_label = st.session_state.get("last_prediction_source", "Latest Prediction")
        top3_df = pd.DataFrame(st.session_state.last_top3)

        with st.container(border=True):
            st.caption(f"Source: {source_label}")
            chart_data = top3_df.set_index("Species")
            st.bar_chart(
                chart_data["Confidence"],
                color=ACCENT_GREEN,
                x_label="Species",
                y_label="Confidence (%)",
            )

        render_top3_predictions([
            (row["Species"], row["Confidence"])
            for row in st.session_state.last_top3
        ])
    else:
        render_ready_state(
            title="No Recent Session Prediction",
            body=(
                "Run an identification in the Image Prediction or Audio Prediction tab "
                "to visualize its Top-3 confidence distribution here."
            ),
        )


# =====================================================================
# Main Component Entrypoint
# =====================================================================

def render_analytics_dashboard(render_ready_state, render_top3_predictions):
    """Render the complete Model Analytics Dashboard using genuine Excel reports.

    Reads model metadata, training curves, comparisons, and evaluation results
    from the ``reports/`` folder.
    """
    st.markdown(
        """
        <div class="avian-component-header">
            <h2 class="avian-component-title">Model Analytics</h2>
            <p class="avian-component-subtitle">
                Inspect verified model specifications, training history curves, cross-model comparisons, and evaluation reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models_overview = load_models_overview()
    analytics_data = load_all_analytics_data()

    # Section A: Overall Model Overview
    _render_model_overview_section(models_overview)
    st.markdown("<div class='avian-section-spacer'></div>", unsafe_allow_html=True)

    # Section B: Image Model Analytics
    _render_image_model_history(
        image_history=analytics_data["image_history"],
        training_runs_summary=analytics_data.get("training_runs_summary"),
        render_ready_state=render_ready_state,
    )
    st.markdown("<div class='avian-section-spacer'></div>", unsafe_allow_html=True)

    # Section C: Audio Model Analytics
    _render_audio_model_history(
        audio_history=analytics_data["audio_history"],
        render_ready_state=render_ready_state,
    )
    st.markdown("<div class='avian-section-spacer'></div>", unsafe_allow_html=True)

    # Section D: Model Comparison
    _render_comparison_section(
        comparison=analytics_data["comparison"],
        render_ready_state=render_ready_state,
    )
    st.markdown("<div class='avian-section-spacer'></div>", unsafe_allow_html=True)

    # Section E: Evaluation Results
    _render_evaluation_section(
        image_eval=analytics_data["image_eval"],
        audio_eval=analytics_data["audio_eval"],
        render_ready_state=render_ready_state,
    )
    st.markdown("<div class='avian-section-spacer'></div>", unsafe_allow_html=True)

    # Section F: Session Prediction Breakdown
    _render_session_prediction_section(
        render_ready_state=render_ready_state,
        render_top3_predictions=render_top3_predictions,
    )
