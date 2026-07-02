"""Streamlit UI for the DR Prediction app.

This file is the UI shell only — model loading, preprocessing, Grad-CAM,
upload handling, image-quality heuristics, and PDF rendering all live in the
`lib/` package. Translations are loaded from `translations/*.json`.
"""
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from lib.config import (
    HISTORY_LIMIT, HISTORY_PATHS, MODEL_CONFIGS, MODEL_PATHS,
    PDF_IMAGE_PAGE_LIMIT, PDF_SUMMARY_ROW_LIMIT,
    RECOMMENDED_THRESHOLDS, RESIZE_MODES,
)
from lib.i18n import LANGUAGES, get_translator
from lib.inference import predict_with_model
from lib.quality import analyze_image_quality
from lib.report import build_pdf_report
from lib.config import (DR_CLASS_NAMES, MULTICLASS_MODEL_CONFIGS,
                        MULTICLASS_MODEL_PATHS)
from lib.multiclass_inference import (class_set_to_label, compute_conformal_set,
                                       load_conformal_thresholds, predict_multiclass)
from lib.styles import CUSTOM_CSS, format_confidence_badge
from lib.uncertainty_inference import abstention_decision, ensemble_predict
from lib.uploads import load_uploaded_images


st.set_page_config(page_title="DR Prediction", page_icon="👁️", layout="wide")


if "history" not in st.session_state:
    st.session_state.history = []


def append_history(row):
    """Append a prediction row to session history with a rolling cap."""
    st.session_state.history.append(row)
    if len(st.session_state.history) > HISTORY_LIMIT:
        st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]


with st.sidebar:
    language = st.selectbox("Language / Gjuha", list(LANGUAGES.keys()))

tr = get_translator(language)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title(tr("title"))
st.write(tr("subtitle"))
st.caption(tr("disclaimer"))


@st.cache_data(show_spinner=False)
def collect_model_metadata(path, history_path=None):
    """Return file size, mtime, and the best validation metrics from training.

    The training history pickles are dicts of per-epoch metric lists. We pick
    the best (max) val_accuracy and the matching val_loss so the UI can show
    the user how good the model is without making them open the thesis.
    """
    if not os.path.isfile(path):
        return None
    metadata = {
        "size_mb": os.path.getsize(path) / (1024 * 1024),
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))),
        "best_val_accuracy": None,
        "best_val_loss": None,
        "epochs_trained": None,
    }
    if history_path and os.path.isfile(history_path):
        try:
            with open(history_path, "rb") as f:
                history = pickle.load(f)
            if isinstance(history, dict) and history.get("val_accuracy"):
                val_accs = history["val_accuracy"]
                val_losses = history.get("val_loss", [])
                best_idx = int(np.argmax(val_accs))
                metadata["best_val_accuracy"] = float(val_accs[best_idx])
                if best_idx < len(val_losses):
                    metadata["best_val_loss"] = float(val_losses[best_idx])
                metadata["epochs_trained"] = len(val_accs)
        except (OSError, pickle.UnpicklingError, ValueError):
            pass
    return metadata


# ---- Sidebar settings -------------------------------------------------------
with st.sidebar:
    st.header(tr("settings"))
    primary_model = st.selectbox(tr("primary_model"), list(MODEL_PATHS.keys()))
    compare_models = st.multiselect(
        tr("compare_models"),
        [name for name in MODEL_PATHS.keys() if name != primary_model],
        default=[],
    )
    # `key` is per-model so the slider resets to the recommended threshold
    # whenever the user switches primary model. Without this, Streamlit
    # preserves the slider value across model changes.
    threshold = st.slider(
        tr("threshold"),
        0.0,
        1.0,
        float(RECOMMENDED_THRESHOLDS.get(primary_model, 0.5)),
        0.01,
        key=f"threshold_{primary_model}",
    )
    # Store the stable RESIZE_FIT / RESIZE_CROP key, but display the localized
    # label via format_func so language changes don't break preprocessing.
    resize_mode = st.selectbox(
        tr("resize_mode"),
        options=RESIZE_MODES,
        format_func=lambda key: tr(f"resize_{key}"),
    )
    show_probs = st.checkbox(tr("show_probs"), value=True)
    show_preprocessed = st.checkbox(tr("show_preprocessed"), value=False)
    show_explain = st.checkbox(tr("show_explain"), value=False)
    # Master-thesis Phase 1/2: ensemble across all available models, plus
    # an abstention zone driven by predictive entropy / vote disagreement.
    show_uncertainty = st.checkbox(tr("show_uncertainty"), value=False)
    st.caption(tr("tip"))
    st.info(f"{tr('recommended_threshold')}: {RECOMMENDED_THRESHOLDS.get(primary_model, 0.5):.2f}")
    st.caption(tr("zip_note"))


# ---- Model status banner ----------------------------------------------------
primary_model_path = MODEL_PATHS[primary_model]
primary_history_path = HISTORY_PATHS.get(primary_model)
model_status = tr("ready") if os.path.isfile(primary_model_path) else tr("missing")
metadata = collect_model_metadata(primary_model_path, primary_history_path)

left_meta, status_col, thresh_col, acc_col = st.columns([2, 1, 1, 1])
with left_meta:
    st.write(f"{tr('model_path')}: `{primary_model_path}`")
with status_col:
    st.metric(tr("model_status"), model_status)
with thresh_col:
    st.metric(tr("recommended_threshold"), f"{RECOMMENDED_THRESHOLDS.get(primary_model, 0.5):.2f}")
with acc_col:
    if metadata and metadata.get("best_val_accuracy") is not None:
        st.metric(tr("best_val_accuracy"), f"{metadata['best_val_accuracy']:.2%}")
    else:
        st.metric(tr("best_val_accuracy"), "—")

if metadata:
    details_parts = [f"{metadata['size_mb']:.1f} MB", metadata['modified']]
    if metadata.get("epochs_trained"):
        details_parts.append(f"{metadata['epochs_trained']} {tr('epochs')}")
    if metadata.get("best_val_loss") is not None:
        details_parts.append(f"val_loss {metadata['best_val_loss']:.4f}")
    st.markdown(
        f"<div class='small-muted'><strong>{tr('model_details')}:</strong> "
        f"{' • '.join(details_parts)}</div>",
        unsafe_allow_html=True,
    )


# ---- Tabs -------------------------------------------------------------------
tabs = st.tabs([tr("predict_tab"), tr("multiclass_tab"), tr("history_tab"),
                tr("about_tab"), tr("troubleshooting_tab")])

with tabs[0]:
    st.subheader(tr("prediction"))
    st.markdown(
        "**Pipeline**: 1) Upload → 2) Quality check → 3) Preprocess → 4) Predict → 5) Review and export"
    )
    uploaded_files = st.file_uploader(
        tr("upload"),
        type=["png", "jpg", "jpeg", "zip"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        image_entries = load_uploaded_images(
            uploaded_files,
            on_warning=st.warning,
            on_error=st.error,
        )
        if not image_entries:
            st.warning("No valid images were found in the uploaded files.")
        elif not os.path.isfile(primary_model_path):
            st.error("Model file not found. Please choose a trained model.")
        else:
            progress_bar = st.progress(0, text=tr("progress"))
            total_runs = max(1, len(image_entries) * (1 + len(compare_models)))
            completed_runs = 0
            results = []
            report_entries = []

            filter_col, sort_col = st.columns(2)
            with filter_col:
                filter_choice = st.selectbox(tr("filter_results"), [tr("all"), tr("only_dr"), tr("only_nodr")])
            with sort_col:
                sort_choice = st.selectbox(tr("sort_results"), [tr("sort_risk_desc"), tr("sort_risk_asc"), tr("sort_name")])

            for entry in image_entries:
                raw_img = entry["image"]
                file_name = entry["name"]
                quality = analyze_image_quality(raw_img, tr)
                primary_result = predict_with_model(primary_model, raw_img, threshold, resize_mode, show_explain)
                completed_runs += 1
                progress_bar.progress(min(completed_runs / total_runs, 1.0), text=f"{tr('progress')} ({completed_runs}/{total_runs})")

                comparison_results = []
                for cmp_model in compare_models:
                    # Each model uses its own recommended threshold — using a
                    # single threshold for all gives a misleading view because
                    # each model's score distribution is calibrated differently.
                    cmp_threshold = RECOMMENDED_THRESHOLDS.get(cmp_model, 0.5)
                    try:
                        cmp_result = predict_with_model(cmp_model, raw_img, cmp_threshold, resize_mode, False)
                        comparison_results.append(cmp_result)
                    except FileNotFoundError:
                        comparison_results.append({
                            "model": cmp_model,
                            "prediction": "Missing model",
                            "probability_dr": np.nan,
                            "confidence": np.nan,
                            "elapsed_ms": 0,
                            "threshold": cmp_threshold,
                        })
                    completed_runs += 1
                    progress_bar.progress(min(completed_runs / total_runs, 1.0), text=f"{tr('progress')} ({completed_runs}/{total_runs})")

                result_class = "result-dr" if primary_result["prediction"] == "DR" else "result-nodr"
                card_left, card_right = st.columns([1.1, 1.2], gap="large")

                with card_left:
                    st.markdown(f"<div class='main-card {result_class}'>", unsafe_allow_html=True)
                    st.image(raw_img, caption=file_name, use_column_width=True)
                    quality_text = (
                        f"**{tr('quality')}**: {quality['rating']}  \\\n"
                        f"{tr('quality_brightness')}: {quality['brightness']:.2f}  \\\n"
                        f"{tr('quality_contrast')}: {quality['contrast']:.2f}  \\\n"
                        f"{tr('quality_blur')}: {quality['blur_score']:.1f}"
                    )
                    st.markdown(quality_text)
                    st.caption(tr("quality_disclaimer"))
                    if show_preprocessed:
                        st.image(primary_result["processed_img"], caption=tr("preprocessed"), use_column_width=True)
                    if show_explain and primary_result.get("heatmap_overlay") is not None:
                        st.image(primary_result["heatmap_overlay"], caption=tr("gradcam"), use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with card_right:
                    st.markdown(f"<div class='main-card {result_class}'>", unsafe_allow_html=True)
                    st.markdown(format_confidence_badge(primary_result["confidence"]), unsafe_allow_html=True)
                    st.write(f"**{tr('result')}:** {primary_result['prediction']}")
                    st.write(f"**{tr('confidence')}:** {primary_result['confidence']:.2%}")
                    if show_probs:
                        st.progress(float(np.clip(primary_result["probability_dr"], 0.0, 1.0)))
                        st.caption(
                            f"{tr('probability_dr')}: {primary_result['probability_dr']:.4f} • "
                            f"{tr('elapsed')}: {primary_result['elapsed_ms']} ms"
                        )
                    if show_uncertainty:
                        # Run an ensemble of all available models on this
                        # image and surface the abstention decision.
                        unc = ensemble_predict(raw_img, resize_mode,
                                               threshold=threshold)
                        if unc is not None:
                            st.markdown(f"### {tr('uncertainty')}")
                            decision = abstention_decision(unc)
                            decision_color = "salmon" if decision == "refer" else "lightgreen"
                            decision_label = (tr("uncertainty_refer") if decision == "refer"
                                              else tr("uncertainty_confident"))
                            st.markdown(
                                f"<div style='background:{decision_color}22;"
                                f"padding:10px;border-radius:8px;"
                                f"border-left:4px solid {decision_color};'>"
                                f"<strong>{tr('uncertainty_decision')}:</strong> "
                                f"{decision_label}</div>",
                                unsafe_allow_html=True,
                            )
                            ucol1, ucol2, ucol3, ucol4 = st.columns(4)
                            ucol1.metric(tr("ensemble_mean"),
                                         f"{unc['mean_prob']:.3f}")
                            ucol2.metric(tr("ensemble_std"),
                                         f"{unc['std_prob']:.3f}")
                            ucol3.metric(tr("ensemble_entropy"),
                                         f"{unc['predictive_entropy']:.3f}")
                            ucol4.metric(tr("ensemble_agreement"),
                                         f"{unc['vote_agreement']:.0%}")
                            unc_df = pd.DataFrame(unc["per_model"])
                            st.dataframe(unc_df,
                                         use_container_width=True,
                                         hide_index=True)
                    if comparison_results:
                        st.markdown(f"### {tr('comparison')}")
                        comparison_df = pd.DataFrame(
                            [
                                {
                                    "model": row["model"],
                                    "prediction": row["prediction"],
                                    "probability_dr": row.get("probability_dr"),
                                    "confidence": row.get("confidence"),
                                    "threshold": row.get("threshold"),
                                    "elapsed_ms": row.get("elapsed_ms"),
                                }
                                for row in comparison_results
                            ]
                        )
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                primary_row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file": file_name,
                    "model": primary_model,
                    "prediction": primary_result["prediction"],
                    "probability_dr": primary_result["probability_dr"],
                    "confidence": primary_result["confidence"],
                    "threshold": threshold,
                    "quality_rating": quality["rating"],
                    "brightness": quality["brightness"],
                    "contrast": quality["contrast"],
                    "blur_score": quality["blur_score"],
                    "elapsed_ms": primary_result["elapsed_ms"],
                }
                results.append(primary_row)
                append_history(primary_row)
                report_entries.append({**primary_row, "image": raw_img})

                for cmp_result in comparison_results:
                    if cmp_result.get("prediction") in {"DR", "No DR"}:
                        cmp_row = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "file": file_name,
                            "model": cmp_result["model"],
                            "prediction": cmp_result["prediction"],
                            "probability_dr": cmp_result["probability_dr"],
                            "confidence": cmp_result["confidence"],
                            "threshold": cmp_result.get("threshold", RECOMMENDED_THRESHOLDS.get(cmp_result["model"], 0.5)),
                            "quality_rating": quality["rating"],
                            "brightness": quality["brightness"],
                            "contrast": quality["contrast"],
                            "blur_score": quality["blur_score"],
                            "elapsed_ms": cmp_result["elapsed_ms"],
                        }
                        results.append(cmp_row)
                        append_history(cmp_row)

            progress_bar.empty()
            st.divider()
            st.subheader(tr("summary"))
            results_df = pd.DataFrame(results)

            if filter_choice == tr("only_dr"):
                results_df = results_df[results_df["prediction"] == "DR"]
            elif filter_choice == tr("only_nodr"):
                results_df = results_df[results_df["prediction"] == "No DR"]

            if sort_choice == tr("sort_risk_desc"):
                results_df = results_df.sort_values(by=["probability_dr", "file"], ascending=[False, True])
            elif sort_choice == tr("sort_risk_asc"):
                results_df = results_df.sort_values(by=["probability_dr", "file"], ascending=[True, True])
            else:
                results_df = results_df.sort_values(by="file")

            st.dataframe(results_df, use_container_width=True, hide_index=True)
            if not results_df.empty:
                csv_data = results_df.to_csv(index=False).encode("utf-8")
                pdf_data = build_pdf_report(results_df, report_entries, primary_model, threshold, tr)
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    st.download_button(tr("download_csv"), data=csv_data, file_name="dr_predictions.csv", mime="text/csv")
                with export_col2:
                    st.download_button(tr("download_pdf"), data=pdf_data, file_name="dr_prediction_report.pdf", mime="application/pdf")
                    if len(results_df) > PDF_SUMMARY_ROW_LIMIT or len(report_entries) > PDF_IMAGE_PAGE_LIMIT:
                        st.caption(tr("pdf_truncation_notice"))

                st.subheader(tr("batch_stats"))
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                stats_col1.metric(tr("dr_count"), int((results_df["prediction"] == "DR").sum()))
                stats_col2.metric(tr("nodr_count"), int((results_df["prediction"] == "No DR").sum()))
                stats_col3.metric("Avg confidence", f"{results_df['confidence'].mean():.2%}")
                st.bar_chart(results_df.set_index("file")["probability_dr"])
    else:
        st.info(tr("no_upload"))

with tabs[1]:
    # ---- Multi-class 5-stage grading (master-thesis Phase 3) ----
    st.subheader(tr("multiclass_tab"))
    st.markdown(tr("multiclass_intro"))

    available_mc = [n for n, c in MULTICLASS_MODEL_CONFIGS.items()
                    if os.path.isfile(c["path"])]
    if not available_mc:
        st.warning(tr("multiclass_no_models"))
    else:
        mc_model = st.selectbox(tr("multiclass_model"), available_mc,
                                key="mc_model_select")
        mc_alpha = st.select_slider(
            tr("multiclass_alpha"),
            options=[0.05, 0.10],
            value=0.10,
            format_func=lambda a: f"{a:.2f} (target {(1-a)*100:.0f}% coverage)",
        )
        mc_score = st.selectbox(
            tr("multiclass_score"),
            options=["aps", "lac"],
            format_func=lambda s: {"aps": "APS (adaptive)", "lac": "LAC (simple)"}[s],
        )
        mc_uploaded = st.file_uploader(
            tr("upload"),
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="mc_uploader",
        )

        thresholds_all = load_conformal_thresholds()
        if mc_uploaded is not None:
            from PIL import Image as PILImage
            img = PILImage.open(mc_uploaded).convert("RGB")
            try:
                result = predict_multiclass(mc_model, img, resize_mode)
            except FileNotFoundError as exc:
                st.error(f"Model file not found: {exc}")
                result = None

            if result is not None:
                col1, col2 = st.columns([1, 1.3])
                with col1:
                    st.image(img, caption=mc_uploaded.name, use_column_width=True)
                with col2:
                    st.markdown(f"### {result['predicted_class_name']}")
                    st.caption(
                        f"{tr('confidence')}: {result['confidence']:.2%} • "
                        f"{tr('elapsed')}: {result['elapsed_ms']} ms"
                    )
                    # Per-class probability bars
                    prob_df = pd.DataFrame({
                        "Class": DR_CLASS_NAMES,
                        "Probability": result["probs"],
                    })
                    st.bar_chart(prob_df.set_index("Class"))

                    # Conformal prediction set
                    model_thresholds = thresholds_all.get(mc_model, {})
                    qhat_key = f"{mc_score}_alpha_{mc_alpha:.2f}"
                    qhat = model_thresholds.get(qhat_key)
                    if qhat is None:
                        st.warning(tr("multiclass_no_threshold"))
                    else:
                        cset = compute_conformal_set(
                            np.asarray(result["probs"]), qhat, score=mc_score,
                            rng=np.random.default_rng(42),
                        )
                        cset_label = class_set_to_label(cset)
                        if len(cset) == 0:
                            decision_color = "salmon"
                        elif len(cset) == 1:
                            decision_color = "lightgreen"
                        elif len(cset) == 2:
                            decision_color = "khaki"
                        else:
                            decision_color = "salmon"
                        st.markdown(
                            f"<div style='background:{decision_color}33;"
                            f"padding:10px;border-radius:8px;"
                            f"border-left:4px solid {decision_color};'>"
                            f"<strong>{tr('multiclass_conformal_set')}:</strong> "
                            f"{cset_label}<br>"
                            f"<small>{tr('multiclass_set_size')}: {len(cset)}</small></div>",
                            unsafe_allow_html=True,
                        )
                        # Clinical hint based on set size
                        if len(cset) == 1:
                            st.info(tr("multiclass_hint_single"))
                        elif len(cset) == 2:
                            st.info(tr("multiclass_hint_two"))
                        elif len(cset) >= 3:
                            st.warning(tr("multiclass_hint_refer"))

with tabs[2]:
    st.subheader(tr("history"))
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        history_csv = history_df.to_csv(index=False).encode("utf-8")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            st.download_button(tr("history_export"), data=history_csv, file_name="dr_prediction_history.csv", mime="text/csv")
        with action_col2:
            if st.button(tr("clear_history")):
                st.session_state.history = []
                st.success(tr("history_cleared"))
    else:
        st.info(tr("session_empty"))

with tabs[3]:
    st.subheader(tr("about_tab"))
    st.write(tr("about"))
    st.write(tr("models_loaded"))
    st.write(tr("supported_formats"))
    st.markdown(f"### {tr('class_explanations')}")
    st.write(f"- {tr('class_dr')}")
    st.write(f"- {tr('class_nodr')}")
    st.markdown(f"### {tr('pipeline')}")
    st.write(tr("pipeline_steps"))

with tabs[4]:
    st.subheader(tr("troubleshooting_tab"))
    st.write(f"- {tr('trouble_missing_model')}")
    st.write(f"- {tr('trouble_missing_inputs')}")
    st.write(f"- {tr('trouble_cpu')}")
    st.write(f"- {tr('trouble_gradcam')}")
