"""Out-of-distribution (OOD) detection on top of trained classifiers.

OOD detection answers: *did this input come from the same distribution as
training?* Useful clinically — if a user uploads a chest X-ray to a retina
classifier, the model should refuse to predict.

Methods implemented:
- **Maximum Softmax Probability (MSP)**: simplest baseline. OOD inputs
  often produce lower confidence. Threshold on `max(p, 1-p)`.
- **Energy score**: -log(sum exp(logits)). Lower energy = more in-
  distribution. Free of softmax saturation issues.
- **Mahalanobis distance** in feature space: fit a Gaussian to ID train
  features, score test points by Mahalanobis distance to the closest
  class-conditional mean. Class-mean + shared covariance variant from
  Lee et al. 2018.
- **Cosine distance to ID centroid**: simpler distance baseline using
  features.

References:
- Hendrycks & Gimpel 2017, "A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples in Neural Networks", ICLR.
- Liu et al. 2020, "Energy-based Out-of-distribution Detection", NeurIPS.
- Lee et al. 2018, "A Simple Unified Framework for Detecting Out-of-
  Distribution Samples and Adversarial Attacks", NeurIPS.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


# ---- Score functions --------------------------------------------------------

def msp_score(probs):
    """Max Softmax Probability — *higher = more in-distribution*.

    For binary sigmoid, this is `max(p, 1-p)`.
    """
    p = np.asarray(probs).ravel()
    return np.maximum(p, 1 - p)


def energy_score(logits, T=1.0):
    """Energy score: −T · logsumexp(logits / T). Lower = more in-distribution.

    For binary, treat the two class-logits as `(0, l)` where l = logit of
    P(class=1). logsumexp(0, l) = log(1 + exp(l)) = softplus(l).
    """
    l = np.asarray(logits).ravel()
    return -T * np.logaddexp(0.0, l / T)


def fit_mahalanobis(features_train, labels_train, eps=1e-6):
    """Fit per-class means and a shared covariance on ID training features.

    Returns dict with `means` (K, D), `inv_cov` (D, D).
    """
    features_train = np.asarray(features_train, dtype=np.float64)
    labels_train = np.asarray(labels_train).astype(int).ravel()
    classes = sorted(set(labels_train.tolist()))
    means = []
    centered = []
    for c in classes:
        Xc = features_train[labels_train == c]
        mu = Xc.mean(axis=0)
        means.append(mu)
        centered.append(Xc - mu)
    means = np.stack(means)
    centered = np.concatenate(centered, axis=0)
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    cov += np.eye(cov.shape[0]) * eps  # Tikhonov for numerical stability
    inv_cov = np.linalg.pinv(cov)
    return {"means": means, "inv_cov": inv_cov, "classes": classes}


def mahalanobis_score(features, fit):
    """Negative min-class Mahalanobis distance — *higher = more in-distribution*.

    Closer to a class mean = lower distance = higher score.
    """
    features = np.asarray(features, dtype=np.float64)
    means = fit["means"]
    inv_cov = fit["inv_cov"]
    K = means.shape[0]
    n = features.shape[0]
    dists = np.empty((n, K))
    for k in range(K):
        diff = features - means[k]
        dists[:, k] = np.einsum("nd,de,ne->n", diff, inv_cov, diff)
    return -dists.min(axis=1)


def cosine_centroid_score(features, centroid):
    """Cosine similarity to a single ID centroid — higher = more in-distribution."""
    f = np.asarray(features, dtype=np.float64)
    c = np.asarray(centroid, dtype=np.float64).ravel()
    fn = np.linalg.norm(f, axis=1) + 1e-12
    cn = np.linalg.norm(c) + 1e-12
    return (f @ c) / (fn * cn)


# ---- Evaluation helpers -----------------------------------------------------

def auroc_id_vs_ood(id_scores, ood_scores):
    """AUROC for separating ID (label=0) from OOD (label=1).

    Higher score = more in-distribution by convention; we flip for AUROC so
    1.0 means perfect OOD detection.
    """
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([np.asarray(id_scores), np.asarray(ood_scores)])
    return roc_auc_score(y, -s)  # negate so higher = more OOD


def fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95):
    """FPR (false alarm rate on ID) when 95% of OOD points are correctly flagged."""
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    # Threshold such that we flag ≤ ID points; score < t → flagged OOD.
    sorted_ood = np.sort(ood_scores)
    k = int(np.floor((1 - tpr_target) * len(ood_scores)))
    threshold = sorted_ood[k] if 0 <= k < len(ood_scores) else sorted_ood[0]
    fpr = float((id_scores < threshold).mean())
    return fpr
