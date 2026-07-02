"""Ensemble uncertainty: combine predictions from multiple models.

Uncertainty decomposition for binary ensembles:
- Predictive entropy: total uncertainty of the average prediction.
- Variance of probabilities across members: epistemic-style spread that
  indicates how much the models disagree (high = more uncertain).
- Mutual information (BALD): predictive entropy minus mean per-model
  entropy — captures epistemic uncertainty (model uncertainty), high when
  members confidently disagree.

The ensemble's accuracy is typically as good or better than the best single
member, while the disagreement signal acts as a "confidence" measure.
"""
from __future__ import annotations

import numpy as np


def _binary_entropy(p, eps=1e-12):
    """Shannon entropy in nats of a Bernoulli with parameter p.

    Clip handles p ∈ {0, 1} so log(0) doesn't fire a RuntimeWarning. The
    entropy at p=0 or p=1 is 0; clipping to (eps, 1-eps) gives entropy
    extremely close to 0, indistinguishable in practice.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def ensemble_predictions(member_probs):
    """Aggregate predictions from K models on N samples.

    Args:
        member_probs: array shape (K, N) of P(class=1) per model per sample.

    Returns dict with:
        - mean_prob:        average P(class=1), shape (N,)
        - std_prob:         std deviation across members (epistemic spread)
        - predictive_entropy: H(mean), total uncertainty
        - mean_member_entropy: mean of H(p_k), aleatoric-style
        - mutual_information: predictive_entropy - mean_member_entropy (BALD)
        - vote_agreement:   fraction of members predicting the majority class
    """
    member_probs = np.asarray(member_probs)
    if member_probs.ndim != 2:
        raise ValueError(f"expected (K, N) got {member_probs.shape}")
    K, N = member_probs.shape

    mean_prob = member_probs.mean(axis=0)
    std_prob = member_probs.std(axis=0)

    pred_entropy = _binary_entropy(mean_prob)
    member_ent = _binary_entropy(member_probs)
    mean_member_ent = member_ent.mean(axis=0)
    mutual_info = pred_entropy - mean_member_ent

    member_pred = (member_probs >= 0.5).astype(int)
    majority_class = (member_pred.sum(axis=0) >= (K / 2)).astype(int)
    agreement = ((member_pred == majority_class[None, :]).sum(axis=0) / K)

    return {
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "predictive_entropy": pred_entropy,
        "mean_member_entropy": mean_member_ent,
        "mutual_information": mutual_info,
        "vote_agreement": agreement,
    }


def _categorical_entropy(probs, eps=1e-12):
    """Shannon entropy of a categorical distribution, summed over classes."""
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def ensemble_predictions_mc(member_probs):
    """Multi-class generalization: aggregate K members' (N, C) softmax outputs.

    Args:
        member_probs: array shape (K, N, C) of class probabilities per model.

    Returns dict with:
        - mean_prob:        average per-class prob, shape (N, C)
        - max_class_std:    std of the max-prob entry across members (epistemic)
        - predictive_entropy: H(mean), total uncertainty
        - mean_member_entropy: mean of H(p_k) (aleatoric proxy)
        - mutual_information: predictive_entropy - mean_member_entropy (BALD)
        - vote_agreement:   fraction of members predicting the majority class
    """
    member_probs = np.asarray(member_probs, dtype=np.float64)
    if member_probs.ndim != 3:
        raise ValueError(f"expected (K, N, C) got {member_probs.shape}")
    K, N, C = member_probs.shape

    mean_prob = member_probs.mean(axis=0)  # (N, C)
    pred_class = mean_prob.argmax(axis=1)  # (N,)
    max_probs = member_probs.max(axis=2)   # (K, N): per-model max prob
    max_class_std = max_probs.std(axis=0)  # (N,)

    pred_entropy = _categorical_entropy(mean_prob)
    member_ent = _categorical_entropy(member_probs)  # (K, N)
    mean_member_ent = member_ent.mean(axis=0)
    mutual_info = pred_entropy - mean_member_ent

    member_pred = member_probs.argmax(axis=2)  # (K, N)
    agreement = (member_pred == pred_class[None, :]).sum(axis=0) / K

    return {
        "mean_prob": mean_prob,
        "max_class_std": max_class_std,
        "predictive_entropy": pred_entropy,
        "mean_member_entropy": mean_member_ent,
        "mutual_information": mutual_info,
        "vote_agreement": agreement,
    }


def selective_accuracy(y_true, mean_prob, uncertainty, coverage_levels=None):
    """Risk-coverage curve: accuracy when we accept only the most confident X%.

    Sort samples by ascending uncertainty, then for each coverage c in
    `coverage_levels`, take the top c-fraction least-uncertain samples and
    compute accuracy on them.

    A model that "knows what it doesn't know" should have rapidly increasing
    selective accuracy as coverage decreases — abstaining on hard cases.

    Args:
        y_true:        binary labels, shape (N,)
        mean_prob:     ensemble mean P(class=1), shape (N,)
        uncertainty:   per-sample uncertainty score (e.g., std or entropy)
        coverage_levels: list of coverages to evaluate (default 0.5..1.0)

    Returns array of (coverage, accuracy_at_coverage).
    """
    if coverage_levels is None:
        coverage_levels = np.linspace(0.5, 1.0, 11)
    y_true = np.asarray(y_true).astype(int).ravel()
    mean_prob = np.asarray(mean_prob)
    uncertainty = np.asarray(uncertainty).ravel()
    # Binary case: mean_prob is 1-D in [0, 1] → predict 1 if ≥ 0.5.
    # Multi-class: mean_prob is 2-D (N, C) → predict argmax.
    if mean_prob.ndim == 1:
        pred = (mean_prob >= 0.5).astype(int)
    else:
        pred = mean_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    order = np.argsort(uncertainty)  # most-confident first

    rows = []
    n = len(y_true)
    for c in coverage_levels:
        k = max(1, int(round(c * n)))
        idx = order[:k]
        rows.append((float(c), float(correct[idx].mean())))
    return np.array(rows)
