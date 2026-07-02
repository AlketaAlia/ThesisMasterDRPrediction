"""Phase 3 multi-class analysis: 5-stage DR grading.

Loads `cnn_5class` and `resnet50_5class`, runs inference on the held-out
multi-class test set, and reports:
- Multi-class accuracy + ordinal metrics (QWK, mean ordinal distance)
- Per-class precision / recall / F1
- Confusion matrix (saved as PNG)
- Multi-class ECE + reliability diagram
- Temperature scaling — fit on val, evaluate on test
- Multi-class conformal prediction (LAC + APS) at α = 0.10, 0.05
- Multi-class ensemble of available 5-class models

Outputs to master/results/multiclass/.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from helpers import (  # noqa: E402
    DR_CLASS_NAMES, build_eval_datagen, build_input_df, get_labels,
    read_all_images, stratified_split,
)
from master.uncertainty.calibration_mc import (  # noqa: E402
    apply_temperature_mc, expected_calibration_error_mc, ordinal_distance,
    quadratic_weighted_kappa, reliability_curve_mc, temperature_scale_mc,
)
from master.uncertainty.conformal import (  # noqa: E402
    evaluate_sets_mc, fit_threshold, predict_sets,
)
from master.uncertainty.ensemble import (  # noqa: E402
    ensemble_predictions_mc, selective_accuracy,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('mc')

OUT = PROJECT_ROOT / "master" / "results" / "multiclass"
OUT.mkdir(parents=True, exist_ok=True)

# Same convention used by train.py
MULTICLASS_MODELS = {
    "cnn_5class": {
        "path": PROJECT_ROOT / "results" / "cnn_5class_model.keras",
        "use_grayscale": True,
        "preprocess_arch": None,
    },
    "resnet50_5class": {
        "path": PROJECT_ROOT / "results" / "resnet50_5class_model.keras",
        "use_grayscale": False,
        "preprocess_arch": "resnet50",
    },
}


def _get_preproc(name):
    if name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    return None


def _predict_probs(model, df, preprocess_fn, use_grayscale, batch_size=32):
    gen = build_eval_datagen(
        preprocessing_function=preprocess_fn,
        use_grayscale=use_grayscale,
    ).flow_from_dataframe(
        df, x_col='filepath', y_col='label',
        target_size=(224, 224), batch_size=batch_size,
        class_mode='categorical', shuffle=False,
    )
    probs = model.predict(gen, verbose=0)
    return probs, gen.class_indices


def _logits_from_probs(p, eps=1e-7):
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    log_p = np.log(p)
    return log_p - log_p.mean(axis=1, keepdims=True)


def _plot_confusion(cm, save_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DR_CLASS_NAMES, yticklabels=DR_CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_reliability(mc, ac, sz, ece, save_path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    if len(mc) > 0:
        ax.bar(mc, ac, width=0.04, alpha=0.7, edgecolor='black')
        for c, a in zip(mc, ac):
            ax.plot([c, c], [c, a], color='red', alpha=0.5, linewidth=1)
    ax.set_xlabel('Confidence (max softmax)')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    from tensorflow.keras.models import load_model

    logger.info("Loading data and computing splits (multi-class)…")
    labels = get_labels(str(PROJECT_ROOT / "inputs" / "labels.csv"))
    images = read_all_images(str(PROJECT_ROOT / "inputs" / "images"), use_pickle=True)
    data = build_input_df(images, labels, multiclass=True)
    train_df, val_df, test_df = stratified_split(
        data, test_size=0.15, val_size=0.15, random_state=123)
    logger.info("Splits — train %d, val %d, test %d",
                len(train_df), len(val_df), len(test_df))

    y_test = test_df['label'].astype(int).values
    y_val = val_df['label'].astype(int).values

    per_model_test = {}
    per_model_val = {}
    summary_rows = []

    for name, cfg in MULTICLASS_MODELS.items():
        if not cfg["path"].exists():
            logger.warning("Skipping %s — file missing at %s", name, cfg["path"])
            continue
        logger.info("=== %s ===", name)
        model = load_model(str(cfg["path"]), compile=False)
        preproc = _get_preproc(cfg["preprocess_arch"])

        probs_test, class_indices = _predict_probs(model, test_df, preproc, cfg["use_grayscale"])
        probs_val, _ = _predict_probs(model, val_df, preproc, cfg["use_grayscale"])
        per_model_test[name] = probs_test
        per_model_val[name] = probs_val

        # Class indices may be sorted by label string ('0','1','2','3','4'),
        # which matches integer order — ensure consistent ordering.
        pred = probs_test.argmax(axis=1)

        acc = accuracy_score(y_test, pred)
        qwk = quadratic_weighted_kappa(y_test, pred, num_classes=5)
        odist = ordinal_distance(y_test, pred)
        ece_raw = expected_calibration_error_mc(y_test, probs_test)

        # Temperature scaling on val.
        logits_val = _logits_from_probs(probs_val)
        T = temperature_scale_mc(logits_val, y_val)
        probs_test_ts = apply_temperature_mc(probs_test, T)
        ece_ts = expected_calibration_error_mc(y_test, probs_test_ts)

        report = classification_report(
            y_test, pred, target_names=DR_CLASS_NAMES, zero_division=0,
            output_dict=True,
        )

        cm = confusion_matrix(y_test, pred, labels=list(range(5)))
        _plot_confusion(cm, str(OUT / f"{name}_confusion_matrix.png"),
                        f"{name} — Confusion matrix")
        mc_, ac_, sz_ = reliability_curve_mc(y_test, probs_test)
        _plot_reliability(mc_, ac_, sz_, ece_raw,
                          str(OUT / f"{name}_reliability_raw.png"),
                          f"{name} — raw")
        mc2, ac2, sz2 = reliability_curve_mc(y_test, probs_test_ts)
        _plot_reliability(mc2, ac2, sz2, ece_ts,
                          str(OUT / f"{name}_reliability_temp_scaled.png"),
                          f"{name} — T = {T:.2f}")

        # Conformal at α = 0.10 and 0.05.
        conformal_results = []
        for alpha in (0.10, 0.05):
            for score_type in ("lac", "aps"):
                qhat, _ = fit_threshold(probs_val, y_val,
                                        alpha=alpha, score=score_type)
                sets = predict_sets(probs_test, qhat,
                                    score=score_type,
                                    rng=np.random.default_rng(42))
                ev = evaluate_sets_mc(sets, y_test, num_classes=5)
                conformal_results.append({
                    "score": score_type,
                    "target_coverage": 1 - alpha,
                    "qhat": qhat,
                    **ev,
                })
        cf_df = pd.DataFrame(conformal_results)
        cf_df.to_csv(OUT / f"{name}_conformal.csv", index=False)

        # Save the thresholds used by the Streamlit app so it doesn't have
        # to refit on every page load. We expose APS @ α=0.10 as the default.
        thresholds_path = OUT / "app_conformal_thresholds.json"
        existing = {}
        if thresholds_path.exists():
            try:
                with open(thresholds_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        # Use the display name expected by lib/multiclass_inference.py
        if name == 'cnn_5class':
            display_key = "CNN (5-class)"
        elif name == 'resnet50_5class':
            display_key = "ResNet50 (5-class)"
        else:
            display_key = name
        existing[display_key] = {
            row["score"] + f"_alpha_{1 - row['target_coverage']:.2f}": float(row["qhat"])
            for row in conformal_results
        }
        with open(thresholds_path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info("%s — acc: %.4f, QWK: %.4f, ord-dist: %.3f, "
                    "ECE raw/ts: %.4f / %.4f (T=%.2f)",
                    name, acc, qwk, odist, ece_raw, ece_ts, T)
        logger.info("Per-class macro F1: %.4f",
                    report['macro avg']['f1-score'])

        summary_rows.append({
            "model": name,
            "test_accuracy": acc,
            "qwk": qwk,
            "ordinal_distance": odist,
            "ece_raw": ece_raw,
            "ece_ts": ece_ts,
            "temperature": T,
            "macro_f1": report['macro avg']['f1-score'],
            "weighted_f1": report['weighted avg']['f1-score'],
        })

        # Per-class detail
        per_class_rows = [{
            "model": name,
            "class": cname,
            "support": int(report[cname]['support']),
            "precision": float(report[cname]['precision']),
            "recall": float(report[cname]['recall']),
            "f1": float(report[cname]['f1-score']),
        } for cname in DR_CLASS_NAMES if cname in report]
        pd.DataFrame(per_class_rows).to_csv(
            OUT / f"{name}_per_class.csv", index=False)

    if not per_model_test:
        logger.error("No multi-class models found. Train them first.")
        return

    pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)

    # ---- Multi-class ensemble (if 2+ models loaded) -----------------------
    if len(per_model_test) >= 2:
        logger.info("Building multi-class ensemble across %d models",
                    len(per_model_test))
        member_probs = np.stack(list(per_model_test.values()))
        ens = ensemble_predictions_mc(member_probs)
        ens_pred = ens["mean_prob"].argmax(axis=1)
        ens_acc = accuracy_score(y_test, ens_pred)
        ens_qwk = quadratic_weighted_kappa(y_test, ens_pred, num_classes=5)
        ens_ece = expected_calibration_error_mc(y_test, ens["mean_prob"])
        cm_ens = confusion_matrix(y_test, ens_pred, labels=list(range(5)))
        _plot_confusion(cm_ens, str(OUT / "ensemble_confusion_matrix.png"),
                        "Ensemble (5-class) — Confusion matrix")

        rc = {
            "max-class std": selective_accuracy(y_test, ens["mean_prob"], ens["max_class_std"]),
            "predictive entropy": selective_accuracy(y_test, ens["mean_prob"], ens["predictive_entropy"]),
            "mutual information": selective_accuracy(y_test, ens["mean_prob"], ens["mutual_information"]),
            "1 - max prob": selective_accuracy(y_test, ens["mean_prob"], 1 - ens["mean_prob"].max(axis=1)),
        }
        from master.uncertainty.plots import plot_risk_coverage
        plot_risk_coverage(rc, str(OUT / "ensemble_risk_coverage.png"),
                           title="Ensemble (5-class) — selective accuracy")

        logger.info("Ensemble — acc: %.4f, QWK: %.4f, ECE: %.4f",
                    ens_acc, ens_qwk, ens_ece)

        with open(OUT / "ensemble_summary.json", "w") as f:
            json.dump({
                "n_members": len(per_model_test),
                "test_accuracy": float(ens_acc),
                "test_qwk": float(ens_qwk),
                "test_ece": float(ens_ece),
                "rc_acc_at_50pct_pred_entropy": float(rc["predictive entropy"][0, 1]),
                "rc_acc_at_90pct_pred_entropy": float(
                    next((acc for cov, acc in rc["predictive entropy"]
                          if abs(cov - 0.9) < 1e-6), float('nan'))),
            }, f, indent=2)

    logger.info("Phase 3 multi-class analysis done — see master/results/multiclass/")


if __name__ == "__main__":
    main()
