"""Multi-class generalizations of the calibration metrics in `calibration.py`.

For K-class softmax outputs (probs shape `(N, K)`):
- Confidence per sample = max-class softmax probability.
- Predicted class = argmax.
- ECE / MCE / reliability work on (predicted_class, confidence) — same logic
  as binary, but `confidence = probs.max(axis=1)` instead of `max(p, 1-p)`.
- Temperature scaling minimizes multi-class cross-entropy.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def predicted_class_and_confidence(probs):
    """Returns (predicted_class_per_sample, confidence_per_sample)."""
    p = np.asarray(probs)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    return pred, conf


def expected_calibration_error_mc(y_true, probs, n_bins=15):
    """Multi-class ECE on max-class confidence."""
    y_true = np.asarray(y_true).astype(int).ravel()
    pred, conf = predicted_class_and_confidence(probs)
    correct = (pred == y_true).astype(float)
    bin_edges = np.linspace(1.0 / probs.shape[1], 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if not in_bin.any():
            continue
        ece += (in_bin.sum() / n) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_curve_mc(y_true, probs, n_bins=15):
    y_true = np.asarray(y_true).astype(int).ravel()
    pred, conf = predicted_class_and_confidence(probs)
    correct = (pred == y_true).astype(float)
    bin_edges = np.linspace(1.0 / probs.shape[1], 1.0, n_bins + 1)
    mean_conf, acc, sizes = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            mean_conf.append(conf[in_bin].mean())
            acc.append(correct[in_bin].mean())
            sizes.append(in_bin.sum())
    return np.array(mean_conf), np.array(acc), np.array(sizes)


def temperature_scale_mc(logits, y_true):
    """Fit single temperature T over (N, K) logits to minimize NLL."""
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true).astype(int).ravel()
    n = logits.shape[0]

    def nll(T):
        if T <= 0:
            return 1e10
        scaled = logits / T
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(n), y_true].mean()

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method='bounded')
    return float(res.x)


def apply_temperature_mc(probs, T, eps=1e-7):
    """Apply temperature to softmax probabilities by recovering logits first."""
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    logits = np.log(p) - np.log(p).mean(axis=1, keepdims=True)
    return softmax(logits / T, axis=1)


# ---- Ordinal-aware metrics for DR severity ---------------------------------

def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    """Cohen's quadratic-weighted kappa — the standard DR-grading metric.

    For ordinal labels (No DR < Mild < Moderate < Severe < PDR), QWK
    penalizes errors by their squared distance: predicting "PDR" when truth
    is "No DR" is much worse than predicting "Mild". The Kaggle APTOS
    competition was scored with this metric.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    K = num_classes
    # Confusion matrix
    O = np.zeros((K, K), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    # Weight matrix W_{i,j} = (i-j)^2 / (K-1)^2
    W = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            W[i, j] = ((i - j) ** 2) / ((K - 1) ** 2)
    # Expected matrix from histograms
    hist_t = O.sum(axis=1)
    hist_p = O.sum(axis=0)
    E = np.outer(hist_t, hist_p) / O.sum()
    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return float(1 - num / den)


def ordinal_distance(y_true, y_pred):
    """Mean absolute distance |y_true − y_pred| (0 = perfect, K-1 = worst).

    Useful complement to accuracy for ordinal targets — if the model
    misclassifies, did it miss by 1 class or by 4?
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    return float(np.abs(y_true - y_pred).mean())
