"""Analyze MC Dropout models: T stochastic forward passes, then summarize.

Loads each MC-Dropout-trained model, runs T forward passes with dropout
left active (`training=True`), aggregates the per-pass probabilities, and
computes:
- Mean prediction and standard deviation across passes
- Predictive entropy and mutual information (BALD)
- Risk-coverage curves using these uncertainty signals
- Comparison to the deterministic prediction (T=1)

Models analyzed:
- cnn_mcd: CNN with SpatialDropout2D + Dropout
- resnet50_mcd: ResNet50 transfer with Dropout in the head

Run from project root:
    python -m master.run_mc_dropout_analysis
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from helpers import (  # noqa: E402
    build_eval_datagen, build_input_df, get_labels, read_all_images,
    stratified_split,
)
from master.uncertainty.calibration import expected_calibration_error  # noqa: E402
from master.uncertainty.ensemble import selective_accuracy  # noqa: E402
from master.uncertainty.mc_dropout import mc_predict_from_generator, summarize_mc  # noqa: E402
from master.uncertainty.plots import (  # noqa: E402
    plot_risk_coverage, plot_uncertainty_histogram,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('mcd')

OUT = PROJECT_ROOT / "master" / "results" / "mc_dropout"
OUT.mkdir(parents=True, exist_ok=True)


# Models we expect to find. Each entry: file path, preprocessing, grayscale.
MCD_MODELS = {
    "cnn_mcd": {
        "path": PROJECT_ROOT / "results" / "cnn_mcd_model.keras",
        "use_grayscale": True,
        "preprocess_arch": None,
    },
    "resnet50_mcd": {
        "path": PROJECT_ROOT / "results" / "resnet_mcd_model.keras",
        "use_grayscale": False,
        "preprocess_arch": "resnet50",
    },
}


def _get_preproc(arch_name):
    if arch_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    if arch_name == "densenet":
        from tensorflow.keras.applications.densenet import preprocess_input
        return preprocess_input
    return None


def main():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    from tensorflow.keras.models import load_model

    parser_T = int(os.environ.get("MC_T", "30"))
    logger.info("MC Dropout T = %d (override with MC_T env var)", parser_T)

    logger.info("Loading data and computing splits…")
    labels = get_labels(str(PROJECT_ROOT / "inputs" / "labels.csv"))
    images = read_all_images(str(PROJECT_ROOT / "inputs" / "images"), use_pickle=True)
    data = build_input_df(images, labels)
    _, val_df, test_df = stratified_split(
        data, test_size=0.15, val_size=0.15, random_state=123)
    y_test = test_df['label'].astype(int).values

    summary_per_model = []

    for name, cfg in MCD_MODELS.items():
        if not cfg["path"].exists():
            logger.warning("Skipping %s — file missing at %s", name, cfg["path"])
            continue
        logger.info("=== %s ===", name)
        model = load_model(str(cfg["path"]), compile=False)
        preproc = _get_preproc(cfg["preprocess_arch"])
        gen = build_eval_datagen(
            preprocessing_function=preproc,
            use_grayscale=cfg["use_grayscale"],
        ).flow_from_dataframe(
            test_df, x_col='filepath', y_col='label',
            target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False,
        )

        # Deterministic baseline (T=1, dropout off).
        det_probs = model.predict(gen, verbose=0).ravel()
        det_pred = (det_probs >= 0.5).astype(int)
        det_acc = accuracy_score(y_test, det_pred)
        det_auc = roc_auc_score(y_test, det_probs)
        logger.info("Deterministic — acc: %.4f, AUC: %.4f", det_acc, det_auc)

        # MC Dropout: T stochastic forward passes.
        logger.info("Running %d MC forward passes…", parser_T)
        mc_probs = mc_predict_from_generator(model, gen, T=parser_T,
                                             n_samples=len(test_df))
        np.save(OUT / f"{name}_mc_probs_T{parser_T}.npy", mc_probs)
        mc = summarize_mc(mc_probs)

        mc_pred = (mc["mean_prob"] >= 0.5).astype(int)
        mc_acc = accuracy_score(y_test, mc_pred)
        mc_auc = roc_auc_score(y_test, mc["mean_prob"])
        ece_det = expected_calibration_error(y_test, det_probs)
        ece_mc = expected_calibration_error(y_test, mc["mean_prob"])
        logger.info("MC Dropout — acc: %.4f, AUC: %.4f, ECE det/mc: %.4f / %.4f",
                    mc_acc, mc_auc, ece_det, ece_mc)

        # Risk-coverage curves with MC uncertainty.
        rc = {
            "MC std": selective_accuracy(y_test, mc["mean_prob"], mc["std_prob"]),
            "MC pred. entropy": selective_accuracy(y_test, mc["mean_prob"], mc["predictive_entropy"]),
            "MC mutual info": selective_accuracy(y_test, mc["mean_prob"], mc["mutual_information"]),
            "Deterministic 1-max": selective_accuracy(
                y_test, det_probs,
                1 - np.maximum(det_probs, 1 - det_probs)),
        }
        plot_risk_coverage(rc,
                           str(OUT / f"{name}_risk_coverage.png"),
                           title=f"{name} — selective accuracy (T={parser_T})")

        # Histograms.
        correct = (mc_pred == y_test).astype(int)
        plot_uncertainty_histogram(
            mc["std_prob"], correct,
            str(OUT / f"{name}_uncertainty_hist_std.png"),
            title=f"{name} — MC std (correct vs wrong)"
        )

        summary_per_model.append({
            "model": name,
            "T": parser_T,
            "deterministic_accuracy": float(det_acc),
            "deterministic_auc": float(det_auc),
            "deterministic_ece": float(ece_det),
            "mc_accuracy": float(mc_acc),
            "mc_auc": float(mc_auc),
            "mc_ece": float(ece_mc),
            "mean_std_correct": float(mc["std_prob"][correct == 1].mean()),
            "mean_std_wrong": float(mc["std_prob"][correct == 0].mean()) if (correct == 0).any() else float("nan"),
            "rc_acc_at_90pct_pred_entropy": float(
                next((acc for cov, acc in rc["MC pred. entropy"] if abs(cov - 0.9) < 1e-6), float("nan"))),
            "rc_acc_at_50pct_pred_entropy": float(
                next((acc for cov, acc in rc["MC pred. entropy"] if abs(cov - 0.5) < 1e-6), float("nan"))),
        })

    if not summary_per_model:
        logger.error("No MC Dropout models found. Train them first:")
        logger.error("  python scripts/train.py --arch cnn_mcd")
        logger.error("  python scripts/train.py --arch resnet50_mcd")
        return

    df = pd.DataFrame(summary_per_model)
    df.to_csv(OUT / "mc_dropout_summary.csv", index=False)
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary_per_model, f, indent=2)
    logger.info("Wrote summary:\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
