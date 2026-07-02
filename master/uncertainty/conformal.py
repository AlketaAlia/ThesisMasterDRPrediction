"""Split conformal prediction for binary classification.

Conformal prediction is a statistical framework that wraps any model and
produces *prediction sets* with a finite-sample coverage guarantee:
        P(y_test ∈ C(x_test)) ≥ 1 − α
under the assumption that calibration and test data are exchangeable.

For binary classification with sigmoid output, the prediction set is one of:
- {0}     — confident "No DR"
- {1}     — confident "DR"
- {0, 1}  — abstain / uncertain — refer to clinician
- {}      — both classes rejected (rare; means the model thinks neither
            class is likely; flagged as anomalous)

This module implements **split (inductive) conformal prediction**:
1. Hold out a calibration set with known labels.
2. Compute non-conformity scores α_i for each calibration sample.
3. Take the (1−α)-quantile q̂ of those scores.
4. For a test sample, include each candidate label whose non-conformity
   score is ≤ q̂.

Coverage guarantee is *marginal* (averaged over all test samples) and
exact in finite samples, distribution-free (no parametric assumptions).

References:
- Vovk et al. 2005, *Algorithmic Learning in a Random World*.
- Angelopoulos & Bates 2021, "A Gentle Introduction to Conformal Prediction
  and Distribution-Free Uncertainty Quantification", arXiv:2107.07511.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def softmax_scores(probs):
    """Convert binary sigmoid probs to a (N, 2) softmax-style matrix.

    For binary, P(class=0) = 1 - p and P(class=1) = p.
    """
    p = np.asarray(probs).ravel()
    return np.stack([1 - p, p], axis=1)


def _nonconformity_lac(scores, labels):
    """Least-Ambiguous Classifier non-conformity: 1 − P(true_class).

    Simple and well-studied; gives marginal coverage.
    """
    labels = np.asarray(labels).astype(int).ravel()
    return 1.0 - scores[np.arange(len(labels)), labels]


def _nonconformity_aps(scores, labels, randomize=True, rng=None):
    """Adaptive Prediction Sets (APS) non-conformity score.

    Sort class probabilities in decreasing order; the score for the true
    class is the cumulative probability up to and including it. Tends to
    produce smaller sets than LAC at the same coverage.
    """
    labels = np.asarray(labels).astype(int).ravel()
    n, K = scores.shape
    if rng is None:
        rng = np.random.default_rng(0)
    out = np.empty(n)
    for i in range(n):
        order = np.argsort(-scores[i])           # descending
        sorted_probs = scores[i][order]
        cumprobs = np.cumsum(sorted_probs)
        rank = int(np.where(order == labels[i])[0][0])
        if randomize:
            u = rng.uniform()
            out[i] = cumprobs[rank] - u * sorted_probs[rank]
        else:
            out[i] = cumprobs[rank]
    return out


def fit_threshold(cal_scores, cal_labels, alpha=0.1, score="lac"):
    """Compute the conformal threshold q̂ on a calibration set.

    Args:
        cal_scores: (N, 2) softmax-style probabilities on calibration set.
        cal_labels: (N,) integer labels on calibration set.
        alpha: target miscoverage (0.1 → 90% coverage).
        score: "lac" or "aps".

    Returns the empirical (1 − α) quantile (with finite-sample correction).
    """
    cal_scores = np.asarray(cal_scores)
    n = len(cal_labels)
    if score == "lac":
        nc = _nonconformity_lac(cal_scores, cal_labels)
    elif score == "aps":
        nc = _nonconformity_aps(cal_scores, cal_labels)
    else:
        raise ValueError(f"unknown score: {score!r}")
    # Finite-sample correction: ceil((n+1)(1−α)) / n
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    qhat = float(np.quantile(nc, q_level, method="higher"))
    return qhat, nc


def predict_sets(test_scores, qhat, score="lac", rng=None):
    """Return a list of prediction sets (one per test sample).

    Each entry is a tuple of class labels included in the set, e.g.
    `(1,)`, `(0,)`, `(0, 1)`, or `()`.
    """
    test_scores = np.asarray(test_scores)
    n, K = test_scores.shape
    if rng is None:
        rng = np.random.default_rng(0)
    sets = []
    for i in range(n):
        if score == "lac":
            included = [k for k in range(K) if (1 - test_scores[i, k]) <= qhat]
        elif score == "aps":
            order = np.argsort(-test_scores[i])
            sorted_probs = test_scores[i][order]
            cumprobs = np.cumsum(sorted_probs)
            included = []
            for rank, k in enumerate(order):
                u = rng.uniform()
                rand_score = cumprobs[rank] - u * sorted_probs[rank]
                if rand_score <= qhat:
                    included.append(int(k))
        else:
            raise ValueError(f"unknown score: {score!r}")
        sets.append(tuple(sorted(included)))
    return sets


def evaluate_sets_mc(prediction_sets, test_labels, num_classes):
    """Multi-class generalization of `evaluate_sets`.

    Returns coverage, mean set size, set-size distribution, and per-class
    conditional coverage (fraction of true-class points whose set contains
    that class).
    """
    test_labels = np.asarray(test_labels).astype(int).ravel()
    n = len(test_labels)
    covered = 0
    sizes = []
    conditional_covered = np.zeros(num_classes)
    conditional_total = np.zeros(num_classes)
    for s, y in zip(prediction_sets, test_labels):
        sizes.append(len(s))
        conditional_total[y] += 1
        if y in s:
            covered += 1
            conditional_covered[y] += 1
    cond_cov = np.divide(
        conditional_covered, conditional_total,
        out=np.full(num_classes, np.nan), where=conditional_total > 0,
    )
    return {
        "coverage": covered / n,
        "mean_size": float(np.mean(sizes)),
        "set_size_distribution": dict(_count_set_sizes(sizes)),
        "conditional_coverage": cond_cov.tolist(),
    }


def _count_set_sizes(sizes):
    from collections import Counter
    return Counter(sizes)


def evaluate_sets(prediction_sets, test_labels):
    """Coverage and average set size.

    Returns dict with:
      - coverage: empirical fraction of test points whose true label is in the set.
      - mean_size: average |C(x)| over test points.
      - set_size_distribution: Counter mapping set-size -> count.
      - singleton_correct_rate: among singleton sets, fraction equal to truth.
      - abstain_rate: fraction of {0, 1} sets (deferred decisions).
      - empty_rate: fraction of empty sets.
    """
    test_labels = np.asarray(test_labels).astype(int).ravel()
    n = len(test_labels)
    covered = 0
    sizes = []
    singleton_correct = 0
    n_singletons = 0
    n_full = 0
    n_empty = 0
    for s, y in zip(prediction_sets, test_labels):
        size = len(s)
        sizes.append(size)
        if y in s:
            covered += 1
        if size == 1:
            n_singletons += 1
            if s[0] == y:
                singleton_correct += 1
        elif size == 2:
            n_full += 1
        elif size == 0:
            n_empty += 1
    return {
        "coverage": covered / n,
        "mean_size": float(np.mean(sizes)),
        "set_size_distribution": dict(Counter(sizes)),
        "singleton_correct_rate": (singleton_correct / n_singletons) if n_singletons else float("nan"),
        "abstain_rate": n_full / n,
        "empty_rate": n_empty / n,
    }
