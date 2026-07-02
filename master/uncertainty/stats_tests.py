"""Statistical significance tests for comparing classifiers.

When model A scores 96.0% and model B scores 95.8% on the same test set, is
the difference real or noise? These tests answer that question.

Methods:
- McNemar test: paired test on disagreements; appropriate when both models
  predict the same examples (same test set).
- Bootstrap CI: percentile confidence interval for any metric by resampling
  the test set with replacement.
- Cohen's kappa: chance-corrected agreement between two predictions
  (or between a prediction and ground truth).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar_test(y_true, pred_a, pred_b, continuity=True):
    """McNemar's paired test on two classifiers' predictions over the same set.

    Builds a 2x2 contingency table:
                          pred_b correct   pred_b wrong
        pred_a correct        a                  b
        pred_a wrong          c                  d

    Tests H0: the marginal error rates of A and B are equal (b == c).
    Statistic: (|b - c| - 1)^2 / (b + c) with continuity correction;
    follows chi-squared with 1 df under H0.

    Returns dict with `b`, `c`, `statistic`, `pvalue`. p < 0.05 means the
    classifiers' error rates differ significantly.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    a = (np.asarray(pred_a).astype(int).ravel() == y_true)
    b_correct = (np.asarray(pred_b).astype(int).ravel() == y_true)

    b = int(np.sum(a & ~b_correct))   # A right, B wrong
    c = int(np.sum(~a & b_correct))   # A wrong, B right

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "pvalue": 1.0}

    diff = abs(b - c)
    if continuity:
        diff = max(diff - 1, 0)
    statistic = (diff ** 2) / (b + c)
    pvalue = 1.0 - stats.chi2.cdf(statistic, df=1)
    return {"b": b, "c": c, "statistic": float(statistic), "pvalue": float(pvalue)}


def pairwise_mcnemar_matrix(y_true, name_to_pred):
    """Run McNemar between every pair of models, return dict of {(A, B): result}."""
    names = list(name_to_pred.keys())
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[(a, b)] = mcnemar_test(y_true, name_to_pred[a], name_to_pred[b])
    return out


def bootstrap_ci(metric_fn, y_true, *args, n_boot=1000, alpha=0.05, seed=42):
    """Percentile bootstrap confidence interval for a paired metric.

    Args:
        metric_fn: callable(y_true_sub, *args_sub) -> scalar.
        y_true:    ground truth, shape (N,).
        *args:     additional arrays (e.g., predictions, probabilities) all
                   resampled with the same indices as y_true.
        n_boot:    number of bootstrap resamples.
        alpha:     significance level (0.05 → 95% CI).

    Returns (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    arrays = [np.asarray(a) for a in args]
    point = metric_fn(y_true, *arrays)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], *(a[idx] for a in arrays))
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def cohen_kappa(pred_a, pred_b):
    """Chance-corrected agreement between two binary predictions."""
    pred_a = np.asarray(pred_a).astype(int).ravel()
    pred_b = np.asarray(pred_b).astype(int).ravel()
    n = len(pred_a)
    po = float((pred_a == pred_b).sum() / n)
    p_a1 = pred_a.mean()
    p_b1 = pred_b.mean()
    pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if 1 - pe == 0:
        return 1.0
    return float((po - pe) / (1 - pe))
