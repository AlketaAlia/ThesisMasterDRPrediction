"""Monte Carlo Dropout inference: many forward passes with dropout left on.

If a model was trained with Dropout / SpatialDropout2D layers, these layers
are normally disabled at inference. By leaving them active and running T
forward passes, we get a distribution of predictions per sample. The mean
is the point prediction; the variance approximates the model's epistemic
(Bayesian) uncertainty (Gal & Ghahramani 2016, "Dropout as a Bayesian
Approximation").

The trick to enable dropout at inference in Keras is to call the model
with `training=True` — TensorFlow will keep dropout active.
"""
from __future__ import annotations

import numpy as np


def mc_predict(model, x, T=30, batch_size=32):
    """Run `T` stochastic forward passes and return shape (T, N) probs.

    Args:
        model: Keras model with Dropout layers.
        x: ndarray of shape (N, H, W, C) — already preprocessed inputs.
        T: number of MC samples (more is better; 30 is a reasonable default).
        batch_size: forward-pass batch size.

    Returns array shape (T, N) of P(class=1).
    """
    n = x.shape[0]
    out = np.empty((T, n), dtype=np.float32)
    for t in range(T):
        # `training=True` keeps Dropout active; we wrap in tf.function for speed.
        preds = model(x, training=True).numpy().ravel()
        out[t] = preds
    return out


def mc_predict_from_generator(model, gen, T=30, n_samples=None):
    """MC predict from a Keras data generator.

    Generators iterate once per pass; we re-iterate T times. The generator
    must be deterministic (`shuffle=False`) so samples align across passes.
    """
    if n_samples is None:
        n_samples = gen.samples
    out = np.empty((T, n_samples), dtype=np.float32)
    for t in range(T):
        gen.reset()
        preds = []
        steps = int(np.ceil(n_samples / gen.batch_size))
        for _ in range(steps):
            batch_x, _ = next(gen)
            batch_pred = model(batch_x, training=True).numpy().ravel()
            preds.append(batch_pred)
        out[t] = np.concatenate(preds)[:n_samples]
    return out


def summarize_mc(probs):
    """Aggregate (T, N) MC probabilities into per-sample summary.

    Returns dict with mean, std, predictive_entropy, mean_member_entropy,
    mutual_information — same keys as the heterogeneous ensemble module so
    downstream code is interchangeable.
    """
    eps = 1e-12
    probs = np.asarray(probs)
    mean_prob = probs.mean(axis=0)
    std_prob = probs.std(axis=0)

    def _bin_ent(p):
        p = np.clip(p, eps, 1 - eps)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    pred_entropy = _bin_ent(mean_prob)
    member_ent = _bin_ent(probs)
    mean_member_ent = member_ent.mean(axis=0)
    mutual_info = pred_entropy - mean_member_ent
    return {
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "predictive_entropy": pred_entropy,
        "mean_member_entropy": mean_member_ent,
        "mutual_information": mutual_info,
    }
