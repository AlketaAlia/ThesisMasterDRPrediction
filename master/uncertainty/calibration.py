"""Calibration metrics and post-hoc calibration methods.

A model is "calibrated" if its predicted probabilities match observed
frequencies — when it says 80% confidence, it should be right 80% of the
time. Modern deep neural networks are typically *over-confident*: a 99%
prediction is often wrong much more often than 1% of the time.

This module computes:
- Expected Calibration Error (ECE): weighted average gap between confidence
  and accuracy across confidence bins.
- Maximum Calibration Error (MCE): worst-case bin gap.
- Reliability diagrams: per-bin accuracy vs. mean confidence.
- Temperature scaling: a single-parameter post-hoc fix that often makes
  models well-calibrated without changing their predictions.

References:
- Guo et al. 2017, "On Calibration of Modern Neural Networks", ICML.
- Naeini et al. 2015, "Obtaining Well Calibrated Probabilities Using
  Bayesian Binning", AAAI.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _binarize_confidences(probs):
    """Convert sigmoid outputs to per-sample (predicted_class, confidence) pairs.

    For binary sigmoid output `p = P(class=1)`, the predicted class is 1 if
    p >= 0.5 else 0, and the confidence (probability of the predicted class)
    is `max(p, 1-p)`.
    """
    probs = np.asarray(probs).ravel()
    pred_classes = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1.0 - probs)
    return pred_classes, confidences


def expected_calibration_error(y_true, probs, n_bins=15):
    """Expected Calibration Error.

    Splits predictions into `n_bins` equal-width confidence bins, computes
    |accuracy - confidence| in each bin, weights by bin size, sums.
    Lower is better; 0 means perfectly calibrated.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    pred_classes, confidences = _binarize_confidences(probs)
    correct = (pred_classes == y_true).astype(float)

    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def maximum_calibration_error(y_true, probs, n_bins=15):
    """Maximum Calibration Error: the worst per-bin |accuracy - confidence|."""
    y_true = np.asarray(y_true).astype(int).ravel()
    pred_classes, confidences = _binarize_confidences(probs)
    correct = (pred_classes == y_true).astype(float)

    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        mce = max(mce, abs(bin_acc - bin_conf))
    return float(mce)


def reliability_curve(y_true, probs, n_bins=15):
    """Compute the (mean_confidence, accuracy, bin_size) tuples for plotting."""
    y_true = np.asarray(y_true).astype(int).ravel()
    pred_classes, confidences = _binarize_confidences(probs)
    correct = (pred_classes == y_true).astype(float)

    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    mean_conf, acc, sizes = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            mean_conf.append(confidences[in_bin].mean())
            acc.append(correct[in_bin].mean())
            sizes.append(in_bin.sum())
    return np.array(mean_conf), np.array(acc), np.array(sizes)


def temperature_scale(logits, y_true, init_T=1.0):
    """Fit a single temperature parameter T that minimizes NLL on a calibration set.

    Predictions become `sigmoid(logits / T)`. T > 1 softens confident
    predictions (makes them less peaked); T < 1 sharpens. Doesn't change the
    predicted class — only confidence.

    Args:
        logits: pre-sigmoid scores, shape (N,).
        y_true: binary labels, shape (N,).
        init_T: starting point for optimization.

    Returns the optimal temperature T (a scalar > 0).
    """
    logits = np.asarray(logits).ravel()
    y_true = np.asarray(y_true).astype(float).ravel()

    def nll(T):
        if T <= 0:
            return 1e10
        # Numerically stable binary cross-entropy from logits scaled by T.
        z = logits / T
        loss = np.mean(np.maximum(z, 0) - z * y_true + np.log1p(np.exp(-np.abs(z))))
        return loss

    result = minimize_scalar(nll, bounds=(0.05, 10.0), method='bounded')
    return float(result.x)


def apply_temperature(probs, T):
    """Apply temperature T to sigmoid probabilities.

    Equivalent to recovering the logits, dividing by T, re-applying sigmoid.
    Clips to (eps, 1-eps) for numerical stability.
    """
    eps = 1e-7
    probs = np.clip(np.asarray(probs).ravel(), eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    scaled = logits / T
    return 1.0 / (1.0 + np.exp(-scaled))
