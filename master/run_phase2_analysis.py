"""Phase 2: Conformal prediction + OOD detection on the trained ensemble.

Uses the same val/test splits as Phase 1. Runs:
1. Split conformal (LAC + APS) at 90% and 95% target coverage, both per-
   model and on the ensemble.
2. OOD detection using DenseNet121 features:
     - In-distribution: APTOS test set
     - Out-of-distribution: noise / random inputs as a sanity OOD source
       (when no second medical dataset is available)
   Reports MSP, Energy, Mahalanobis, and cosine-centroid AUROC.

Outputs to master/results/phase2/.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from helpers import (  # noqa: E402
    build_eval_datagen, build_input_df, get_labels, read_all_images,
    stratified_split,
)
from lib.config import MODEL_CONFIGS  # noqa: E402
from master.uncertainty.conformal import (  # noqa: E402
    fit_threshold, predict_sets, evaluate_sets, softmax_scores,
)
from master.uncertainty.ood import (  # noqa: E402
    auroc_id_vs_ood, cosine_centroid_score, energy_score,
    fit_mahalanobis, fpr_at_tpr, mahalanobis_score, msp_score,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('phase2')

OUT = PROJECT_ROOT / "master" / "results" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)


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
    return None, True


def _predict_on(model, df, preprocess_fn, use_grayscale, batch_size=32):
    gen = build_eval_datagen(
        preprocessing_function=preprocess_fn,
        use_grayscale=use_grayscale,
    ).flow_from_dataframe(
        df, x_col='filepath', y_col='label',
        target_size=(224, 224), batch_size=batch_size,
        class_mode='binary', shuffle=False,
    )
    return model.predict(gen, verbose=0).ravel()


def _logits_from_probs(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ---- Conformal -------------------------------------------------------------

def run_conformal(per_model_val_probs, val_labels,
                  per_model_test_probs, test_labels,
                  alphas=(0.10, 0.05)):
    rows = []
    for alpha in alphas:
        coverage_target = 1 - alpha
        for model_name, val_p in per_model_val_probs.items():
            test_p = per_model_test_probs[model_name]
            cal_scores = softmax_scores(val_p)
            test_scores = softmax_scores(test_p)
            for score_type in ("lac", "aps"):
                qhat, _ = fit_threshold(cal_scores, val_labels,
                                        alpha=alpha, score=score_type)
                sets = predict_sets(test_scores, qhat,
                                    score=score_type,
                                    rng=np.random.default_rng(42))
                ev = evaluate_sets(sets, test_labels)
                rows.append({
                    "model": model_name,
                    "score": score_type,
                    "target_coverage": coverage_target,
                    "qhat": qhat,
                    **ev,
                })

        # Ensemble (average probabilities)
        ens_val = np.array(list(per_model_val_probs.values())).mean(axis=0)
        ens_test = np.array(list(per_model_test_probs.values())).mean(axis=0)
        cal_scores = softmax_scores(ens_val)
        test_scores = softmax_scores(ens_test)
        for score_type in ("lac", "aps"):
            qhat, _ = fit_threshold(cal_scores, val_labels,
                                    alpha=alpha, score=score_type)
            sets = predict_sets(test_scores, qhat,
                                score=score_type,
                                rng=np.random.default_rng(42))
            ev = evaluate_sets(sets, test_labels)
            rows.append({
                "model": "ENSEMBLE",
                "score": score_type,
                "target_coverage": coverage_target,
                "qhat": qhat,
                **ev,
            })
    return pd.DataFrame(rows)


# ---- OOD -------------------------------------------------------------------

def run_ood(test_df, val_df, model, feature_extractor, preprocess_fn, use_grayscale):
    """Run OOD detection comparing real fundus images vs synthetic OOD."""
    logger.info("Computing classifier probabilities + features on test ID set")
    id_probs = _predict_on(model, test_df, preprocess_fn, use_grayscale)
    id_logits = _logits_from_probs(id_probs)

    logger.info("Extracting DenseNet features on val (for fitting Mahalanobis)")
    feat_gen_val = build_eval_datagen(
        preprocessing_function=preprocess_fn,
        use_grayscale=use_grayscale,
    ).flow_from_dataframe(
        val_df, x_col='filepath', y_col='label',
        target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)
    feat_val = feature_extractor.predict(feat_gen_val, verbose=0)
    val_labels = val_df['label'].astype(int).values[:len(feat_val)]

    feat_gen_test = build_eval_datagen(
        preprocessing_function=preprocess_fn,
        use_grayscale=use_grayscale,
    ).flow_from_dataframe(
        test_df, x_col='filepath', y_col='label',
        target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)
    feat_test = feature_extractor.predict(feat_gen_test, verbose=0)

    logger.info("Generating synthetic OOD: random uniform noise images")
    rng = np.random.default_rng(0)
    n_ood = 300
    ood_imgs = rng.uniform(0.0, 1.0, size=(n_ood, 224, 224, 3)).astype(np.float32)
    if preprocess_fn is not None:
        # preprocess_input expects pixel-scale input (0-255), so rescale.
        ood_imgs = preprocess_fn(ood_imgs * 255.0)
    ood_probs = model.predict(ood_imgs, verbose=0, batch_size=32).ravel()
    ood_logits = _logits_from_probs(ood_probs)
    feat_ood = feature_extractor.predict(ood_imgs, verbose=0, batch_size=32)

    fit = fit_mahalanobis(feat_val, val_labels)
    centroid = feat_val.mean(axis=0)

    metrics = {}
    for name, id_score, ood_score in [
        ("MSP", msp_score(id_probs), msp_score(ood_probs)),
        ("Energy", -energy_score(id_logits), -energy_score(ood_logits)),
        ("Mahalanobis", mahalanobis_score(feat_test, fit),
         mahalanobis_score(feat_ood, fit)),
        ("CosineCentroid", cosine_centroid_score(feat_test, centroid),
         cosine_centroid_score(feat_ood, centroid)),
    ]:
        auroc = auroc_id_vs_ood(id_score, ood_score)
        fpr95 = fpr_at_tpr(id_score, ood_score, tpr_target=0.95)
        metrics[name] = {"auroc_id_vs_ood": auroc, "fpr_at_tpr95": fpr95}
        logger.info("OOD %s — AUROC: %.4f, FPR@TPR95: %.4f",
                    name, auroc, fpr95)
    return metrics


def main():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.applications.densenet import preprocess_input as dn_preprocess
    from tensorflow.keras.models import load_model

    logger.info("Loading data and computing splits…")
    labels = get_labels(str(PROJECT_ROOT / "inputs" / "labels.csv"))
    images = read_all_images(str(PROJECT_ROOT / "inputs" / "images"), use_pickle=True)
    data = build_input_df(images, labels)
    train_df, val_df, test_df = stratified_split(
        data, test_size=0.15, val_size=0.15, random_state=123)
    logger.info("Splits — train %d, val %d, test %d",
                len(train_df), len(val_df), len(test_df))

    y_val = val_df['label'].astype(int).values
    y_test = test_df['label'].astype(int).values

    # Run inference once per model on val + test
    per_model_val_probs = {}
    per_model_test_probs = {}
    densenet_model = None  # used as the OOD feature extractor host below

    for name, cfg in MODEL_CONFIGS.items():
        if not os.path.isfile(cfg['path']):
            logger.warning("Skipping %s — file missing", name)
            continue
        logger.info("Loading %s", name)
        model = load_model(cfg['path'], compile=False)
        preproc, use_gray = _arch_preproc(name)
        per_model_val_probs[name] = _predict_on(model, val_df, preproc, use_gray)
        per_model_test_probs[name] = _predict_on(model, test_df, preproc, use_gray)
        if name == 'DenseNet121':
            densenet_model = model

    # ---- 1. Conformal prediction --------------------------------------
    logger.info("Running conformal prediction (LAC + APS, α∈{0.05, 0.10})")
    conformal_df = run_conformal(per_model_val_probs, y_val,
                                 per_model_test_probs, y_test,
                                 alphas=(0.10, 0.05))
    conformal_df.to_csv(OUT / "conformal_results.csv", index=False)
    logger.info("Top of conformal results:\n%s",
                conformal_df.head(8).to_string(index=False))

    # ---- 2. OOD detection ---------------------------------------------
    logger.info("Building DenseNet121 feature extractor for OOD")
    base = DenseNet121(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3), pooling='avg')
    base.trainable = False

    # Use the trained DenseNet (fine-tuned head) as the classifier; the base
    # gives us 1024-D features for Mahalanobis / cosine.
    if densenet_model is None:
        logger.warning("DenseNet model not found; using ImageNet base for both")
        ood_clf = base
    else:
        ood_clf = densenet_model

    metrics = run_ood(
        test_df, val_df, ood_clf, base, dn_preprocess, use_grayscale=False,
    )
    with open(OUT / "ood_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote master/results/phase2/ood_metrics.json")

    # ---- 3. Summary JSON ----------------------------------------------
    summary = {
        "n_models": len(per_model_val_probs),
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "conformal_summary": {
            f"{row['model']}|{row['score']}|cov={row['target_coverage']:.2f}": {
                "empirical_coverage": float(row['coverage']),
                "mean_set_size": float(row['mean_size']),
                "abstain_rate": float(row['abstain_rate']),
            }
            for _, row in conformal_df.iterrows()
        },
        "ood": metrics,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Phase 2 done — see master/results/phase2/")


if __name__ == "__main__":
    main()
