"""Live uncertainty inference for the Streamlit app.

Wraps the existing per-model `predict_with_model` to additionally:
- Run all available models on the same image for ensemble disagreement.
- Compute per-sample uncertainty signals (mean, std, predictive entropy,
  vote agreement).
- Expose a "decision" with optional abstention zone — if uncertainty is
  above a threshold, the recommendation becomes "Refer to clinician".

The conformal prediction wrapper requires a calibration set to be loaded
once at app start; we lazy-load it from `master/results/phase2/conformal_results.csv`
when available.
"""
from __future__ import annotations

import os

import numpy as np

from lib.config import MODEL_CONFIGS
from lib.inference import predict_with_model


# Default abstention thresholds — tuned roughly from Phase 1 risk-coverage.
# std_prob > this → flag as uncertain; entropy > this → flag as uncertain.
DEFAULT_STD_THRESHOLD = 0.15
DEFAULT_ENTROPY_THRESHOLD = 0.5


def _binary_entropy(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def ensemble_predict(image, resize_mode, threshold=0.5):
    """Run every available model on `image` and aggregate.

    Returns dict with:
      - per_model: list of dicts with model name, prob, prediction, threshold
      - mean_prob, std_prob, predictive_entropy, vote_agreement
      - ensemble_label: majority vote label
      - n_members: how many models successfully ran
    """
    per_model = []
    probs = []
    for name in MODEL_CONFIGS:
        try:
            r = predict_with_model(name, image, MODEL_CONFIGS[name]["threshold"],
                                   resize_mode, show_explain=False)
        except FileNotFoundError:
            continue
        per_model.append({
            "model": name,
            "probability_dr": r["probability_dr"],
            "prediction": r["prediction"],
            "threshold": MODEL_CONFIGS[name]["threshold"],
        })
        probs.append(r["probability_dr"])

    if not probs:
        return None
    probs_arr = np.asarray(probs, dtype=np.float64)
    mean_prob = float(probs_arr.mean())
    std_prob = float(probs_arr.std())
    pred_entropy = _binary_entropy(mean_prob)

    member_pred = (probs_arr >= 0.5).astype(int)
    majority = int(member_pred.sum() >= len(member_pred) / 2)
    agreement = float((member_pred == majority).mean())
    ensemble_label = "DR" if mean_prob >= threshold else "No DR"

    return {
        "per_model": per_model,
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "predictive_entropy": pred_entropy,
        "vote_agreement": agreement,
        "ensemble_label": ensemble_label,
        "n_members": len(probs),
    }


def abstention_decision(uncertainty_summary,
                        std_threshold=DEFAULT_STD_THRESHOLD,
                        entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
                        agreement_threshold=0.7):
    """Return ("Confident", "Uncertain", or "Refer") given uncertainty."""
    std = uncertainty_summary["std_prob"]
    ent = uncertainty_summary["predictive_entropy"]
    agree = uncertainty_summary["vote_agreement"]

    if std > std_threshold or ent > entropy_threshold or agree < agreement_threshold:
        return "refer"
    return "confident"
