"""Phase 1 uncertainty analysis: runs everything that works on the existing
trained models without retraining.

Pipeline:
  1. Run inference on the held-out test set with each model in MODEL_CONFIGS.
  2. Compute calibration (ECE, MCE, reliability diagram) per model.
  3. Fit temperature scaling on the validation set, re-evaluate calibration.
  4. Build an ensemble; compute mean prediction, std, predictive entropy,
     mutual information, vote agreement.
  5. Compute risk-coverage curves using each uncertainty signal.
  6. Run pairwise McNemar tests between models.
  7. Save everything to master/results/: per-model and per-experiment CSVs,
     PNGs of plots, a JSON summary.

Run from project root:
    python -m master.run_uncertainty_analysis
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

# Make sibling packages importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from helpers import (  # noqa: E402
    build_eval_datagen, build_input_df, get_labels, read_all_images,
    stratified_split,
)
from lib.config import MODEL_CONFIGS  # noqa: E402
from master.uncertainty.calibration import (  # noqa: E402
    apply_temperature, expected_calibration_error,
    maximum_calibration_error, reliability_curve, temperature_scale,
)
from master.uncertainty.ensemble import ensemble_predictions, selective_accuracy  # noqa: E402
from master.uncertainty.plots import (  # noqa: E402
    plot_pairwise_pvalue_heatmap, plot_reliability_diagram,
    plot_risk_coverage, plot_uncertainty_histogram,
)
from master.uncertainty.stats_tests import (  # noqa: E402
    bootstrap_ci, mcnemar_test, pairwise_mcnemar_matrix,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('uncertainty')

RESULTS_DIR = PROJECT_ROOT / "master" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Architecture-specific preprocessing for inference. Mirrors what each model
# was trained with so probabilities are comparable.
def _arch_preproc(name):
    if name == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input, False
    if name == 'Xception':
        from tensorflow.keras.applications.xception import preprocess_input
        return preprocess_input, False
    if name == 'DenseNet121':
        from tensorflow.keras.applications.densenet import preprocess_input
        return preprocess_input, False
    if name == 'VGG16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input, False
    # CNN / CNN (Tanh+ReLU): grayscale + /255, no preprocess_input
    return None, True


def _predict_on(model, df, preprocess_fn, use_grayscale, batch_size=32):
    """Run a model over a dataframe and return P(class=1) per sample."""
    gen = build_eval_datagen(
        preprocessing_function=preprocess_fn,
        use_grayscale=use_grayscale,
    ).flow_from_dataframe(
        df, x_col='filepath', y_col='label',
        target_size=(224, 224), batch_size=batch_size,
        class_mode='binary', shuffle=False,
    )
    probs = model.predict(gen, verbose=0).ravel()
    return probs


def _logits_from_probs(probs, eps=1e-7):
    p = np.clip(probs, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    # Suppress TF noise
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    from tensorflow.keras.models import load_model

    logger.info("Loading data and computing splits…")
    labels = get_labels(str(PROJECT_ROOT / "inputs" / "labels.csv"))
    images = read_all_images(str(PROJECT_ROOT / "inputs" / "images"), use_pickle=True)
    data = build_input_df(images, labels)
    train_df, val_df, test_df = stratified_split(
        data, test_size=0.15, val_size=0.15, random_state=123)
    logger.info("Splits — train: %d, val: %d, test: %d",
                len(train_df), len(val_df), len(test_df))

    y_test = test_df['label'].astype(int).values
    y_val = val_df['label'].astype(int).values

    # ---- 1. Per-model inference -------------------------------------------
    per_model_probs_test = {}
    per_model_probs_val = {}
    per_model_summary = []

    for name, cfg in MODEL_CONFIGS.items():
        if not os.path.isfile(cfg['path']):
            logger.warning("Skipping %s — model file not found at %s",
                           name, cfg['path'])
            continue
        logger.info("Loading %s and running inference…", name)
        model = load_model(cfg['path'], compile=False)
        preproc, use_gray = _arch_preproc(name)
        probs_test = _predict_on(model, test_df, preproc, use_gray)
        probs_val = _predict_on(model, val_df, preproc, use_gray)
        per_model_probs_test[name] = probs_test
        per_model_probs_val[name] = probs_val

        pred_test = (probs_test >= cfg['threshold']).astype(int)
        acc = accuracy_score(y_test, pred_test)
        try:
            auc = roc_auc_score(y_test, probs_test)
        except ValueError:
            auc = float('nan')
        ece = expected_calibration_error(y_test, probs_test)
        mce = maximum_calibration_error(y_test, probs_test)

        # 95% CI on test accuracy via bootstrap.
        _, lo, hi = bootstrap_ci(
            lambda y, p: accuracy_score(y, (p >= cfg['threshold']).astype(int)),
            y_test, probs_test, n_boot=1000, seed=42,
        )

        # Temperature scaling on validation, re-evaluate calibration on test.
        logits_val = _logits_from_probs(probs_val)
        T = temperature_scale(logits_val, y_val)
        probs_test_ts = apply_temperature(probs_test, T)
        ece_ts = expected_calibration_error(y_test, probs_test_ts)

        per_model_summary.append({
            "model": name,
            "test_accuracy": acc,
            "test_acc_ci_lo": lo,
            "test_acc_ci_hi": hi,
            "test_auc": auc,
            "ece_raw": ece,
            "mce_raw": mce,
            "temperature": T,
            "ece_after_temp_scale": ece_ts,
            "ece_improvement": ece - ece_ts,
        })

        # Reliability diagram pre/post temperature scaling.
        mc, ac, sz = reliability_curve(y_test, probs_test)
        plot_reliability_diagram(
            mc, ac, sz, ece,
            str(RESULTS_DIR / f"reliability_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}_raw.png"),
            title=f"{name} — raw"
        )
        mc2, ac2, sz2 = reliability_curve(y_test, probs_test_ts)
        plot_reliability_diagram(
            mc2, ac2, sz2, ece_ts,
            str(RESULTS_DIR / f"reliability_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}_temp_scaled.png"),
            title=f"{name} — after temperature scaling (T={T:.2f})"
        )

    summary_df = pd.DataFrame(per_model_summary)
    summary_df.to_csv(RESULTS_DIR / "per_model_summary.csv", index=False)
    logger.info("\n%s", summary_df.to_string(index=False))

    # ---- 2. Ensemble ------------------------------------------------------
    logger.info("Building ensemble across %d models…", len(per_model_probs_test))
    member_probs_test = np.array(list(per_model_probs_test.values()))
    ens = ensemble_predictions(member_probs_test)

    ens_pred = (ens['mean_prob'] >= 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ens_pred)
    ens_auc = roc_auc_score(y_test, ens['mean_prob'])
    ens_ece = expected_calibration_error(y_test, ens['mean_prob'])
    logger.info("Ensemble — acc: %.4f, AUC: %.4f, ECE: %.4f",
                ens_acc, ens_auc, ens_ece)

    # Reliability for the ensemble.
    mc, ac, sz = reliability_curve(y_test, ens['mean_prob'])
    plot_reliability_diagram(mc, ac, sz, ens_ece,
                             str(RESULTS_DIR / "reliability_ENSEMBLE.png"),
                             title="Ensemble — raw")

    # ---- 3. Selective accuracy / risk-coverage ----------------------------
    correct_ens = (ens_pred == y_test).astype(int)
    rc_curves = {
        "std (epistemic spread)": selective_accuracy(y_test, ens['mean_prob'], ens['std_prob']),
        "predictive entropy": selective_accuracy(y_test, ens['mean_prob'], ens['predictive_entropy']),
        "mutual information": selective_accuracy(y_test, ens['mean_prob'], ens['mutual_information']),
        "1 - max prob": selective_accuracy(y_test, ens['mean_prob'], 1 - np.maximum(ens['mean_prob'], 1 - ens['mean_prob'])),
    }
    plot_risk_coverage(rc_curves,
                       str(RESULTS_DIR / "risk_coverage.png"),
                       title="Selective accuracy vs coverage (ensemble)")

    # Save the curves as CSV.
    for label, curve in rc_curves.items():
        slug = label.replace(' ', '_').replace('(', '').replace(')', '')
        pd.DataFrame(curve, columns=['coverage', 'accuracy']).to_csv(
            RESULTS_DIR / f"rc_curve_{slug}.csv", index=False)

    # Histogram of std uncertainty for correct vs wrong ensemble predictions.
    plot_uncertainty_histogram(
        ens['std_prob'], correct_ens,
        str(RESULTS_DIR / "uncertainty_hist_std.png"),
        title="Ensemble std-prob: correct vs wrong"
    )
    plot_uncertainty_histogram(
        ens['predictive_entropy'], correct_ens,
        str(RESULTS_DIR / "uncertainty_hist_entropy.png"),
        title="Predictive entropy: correct vs wrong"
    )

    # ---- 4. Pairwise McNemar ---------------------------------------------
    name_to_pred = {
        name: (probs >= MODEL_CONFIGS[name]['threshold']).astype(int)
        for name, probs in per_model_probs_test.items()
    }
    name_to_pred['ENSEMBLE'] = ens_pred
    pair_results = pairwise_mcnemar_matrix(y_test, name_to_pred)

    names = list(name_to_pred.keys())
    pmat = np.full((len(names), len(names)), np.nan)
    rows = []
    for (a, b), r in pair_results.items():
        i, j = names.index(a), names.index(b)
        pmat[i, j] = r['pvalue']
        pmat[j, i] = r['pvalue']
        rows.append({"model_a": a, "model_b": b, **r})
    np.fill_diagonal(pmat, np.nan)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "mcnemar_pairwise.csv", index=False)
    plot_pairwise_pvalue_heatmap(
        names, pmat, str(RESULTS_DIR / "mcnemar_pvalues.png"),
        title="Pairwise McNemar p-values"
    )

    # ---- 5. JSON summary --------------------------------------------------
    summary_json = {
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "models": per_model_summary,
        "ensemble": {
            "n_members": int(member_probs_test.shape[0]),
            "test_accuracy": float(ens_acc),
            "test_auc": float(ens_auc),
            "test_ece": float(ens_ece),
        },
        "risk_coverage_summary": {
            label: {f"acc_at_{int(100*cov)}pct": float(acc)
                    for cov, acc in curve}
            for label, curve in rc_curves.items()
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    logger.info("Wrote master/results/summary.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
