"""Multi-class (5-stage) DR grading inference for the Streamlit app.

Provides:
- `predict_multiclass(model_name, image, resize_mode)` — forward pass and
  returns the softmax probabilities, predicted class, and confidence.
- `compute_conformal_set(probs, qhat, score)` — wraps the prediction in a
  conformal prediction set using a precomputed threshold.
- `load_conformal_thresholds()` — lazy-loads thresholds fitted on the
  validation set in Phase 3.

The conformal threshold q̂ is fitted offline (Phase 3 analysis script) and
saved as JSON; the app just looks it up here. Re-fitting in-process would
require running val-set inference at every page load — too slow.
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

from lib.config import (DR_CLASS_NAMES, MULTICLASS_MODEL_CONFIGS,
                        RESIZE_CROP)


_MC_CACHE: "OrderedDict[str, tf.keras.Model]" = OrderedDict()
_MAX_MC_CACHED = 2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFORMAL_PATH = _PROJECT_ROOT / "master" / "results" / "multiclass" / "app_conformal_thresholds.json"


def _load_mc_model(path):
    if path in _MC_CACHE:
        _MC_CACHE.move_to_end(path)
        return _MC_CACHE[path]
    model = load_model(path, compile=False)
    _MC_CACHE[path] = model
    _MC_CACHE.move_to_end(path)
    while len(_MC_CACHE) > _MAX_MC_CACHED:
        _MC_CACHE.popitem(last=False)
    return model


def _arch_preprocess(arch_name):
    if arch_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    return None


def _preprocess_for_model(image, resize_mode, use_grayscale, preprocess_fn):
    image = image.convert("RGB")
    if resize_mode == RESIZE_CROP:
        image = ImageOps.fit(image, (224, 224), method=Image.BICUBIC)
    else:
        image = ImageOps.pad(image, (224, 224), method=Image.BICUBIC, color=(0, 0, 0))
    arr = np.array(image).astype("float32")
    if use_grayscale:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        arr = np.stack((gray,) * 3, axis=-1).astype("float32")
    if preprocess_fn is not None:
        arr = preprocess_fn(arr)
    else:
        arr = arr / 255.0
    return image, np.expand_dims(arr, axis=0)


def predict_multiclass(model_name, image, resize_mode):
    """Run a 5-class model on a PIL image. Returns dict with probs + label."""
    cfg = MULTICLASS_MODEL_CONFIGS[model_name]
    if not os.path.isfile(cfg["path"]):
        raise FileNotFoundError(cfg["path"])
    model = _load_mc_model(cfg["path"])
    preproc_fn = _arch_preprocess(cfg["preprocess_arch"])
    display_img, x = _preprocess_for_model(
        image, resize_mode, cfg["use_grayscale"], preproc_fn)

    start = time.time()
    probs = model.predict(x, verbose=0).ravel()
    elapsed_ms = int((time.time() - start) * 1000)

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    return {
        "model": model_name,
        "probs": probs.tolist(),
        "predicted_class": pred_class,
        "predicted_class_name": DR_CLASS_NAMES[pred_class],
        "confidence": confidence,
        "elapsed_ms": elapsed_ms,
        "processed_img": display_img,
    }


def compute_conformal_set(probs, qhat, score="lac", rng=None):
    """Construct a conformal prediction set for one sample.

    Args:
        probs: array (K,) softmax probabilities.
        qhat: precomputed conformal threshold for the chosen target coverage.
        score: "lac" or "aps".

    Returns list of class indices in the prediction set.
    """
    probs = np.asarray(probs).ravel()
    K = len(probs)
    if score == "lac":
        return [k for k in range(K) if (1 - probs[k]) <= qhat]
    if score == "aps":
        if rng is None:
            rng = np.random.default_rng(0)
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumprobs = np.cumsum(sorted_probs)
        included = []
        for rank, k in enumerate(order):
            u = rng.uniform()
            rand_score = cumprobs[rank] - u * sorted_probs[rank]
            if rand_score <= qhat:
                included.append(int(k))
        return sorted(included)
    raise ValueError(f"unknown score: {score!r}")


def load_conformal_thresholds():
    """Load precomputed conformal thresholds for all multi-class models.

    Returns dict keyed by model name. If the JSON doesn't exist, returns
    an empty dict and the caller should skip showing conformal sets.
    """
    if not _CONFORMAL_PATH.exists():
        return {}
    with open(_CONFORMAL_PATH, "r") as f:
        return json.load(f)


def class_set_to_label(class_indices):
    """Format a prediction set as a human-readable label like 'Mild, Moderate'."""
    if not class_indices:
        return "(none)"
    return ", ".join(DR_CLASS_NAMES[k] for k in class_indices)
