"""Model loading, preprocessing, prediction, and Grad-CAM helpers."""
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

from lib.config import MAX_LOADED_MODELS, MODEL_CONFIGS, RESIZE_CROP


# ---- Model cache (LRU) ------------------------------------------------------
# Each transfer model is ~100 MB; with all five plus comparison runs, RAM use
# can exceed 1 GB. Streamlit's @st.cache_resource caches forever, so we wrap
# our own LRU policy here. The cache is process-global; in a Streamlit context
# it persists across reruns within the same server process.

_MODEL_CACHE: "OrderedDict[str, tf.keras.Model]" = OrderedDict()


def load_trained_model(path):
    """Load a Keras model, evicting older ones when the cache is full.

    Keyed by absolute model path. Touching a model marks it most-recently-used;
    once we exceed MAX_LOADED_MODELS, the least-recently used model is dropped.
    """
    if path in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(path)
        return _MODEL_CACHE[path]
    model = load_model(path)
    _MODEL_CACHE[path] = model
    _MODEL_CACHE.move_to_end(path)
    while len(_MODEL_CACHE) > MAX_LOADED_MODELS:
        _MODEL_CACHE.popitem(last=False)
    return model


# ---- Preprocessing ----------------------------------------------------------

def preprocess_image(image, mode, use_grayscale=False):
    """Resize and normalize an image for model inference.

    Args:
        image: PIL Image.
        mode: one of RESIZE_FIT (padded fit) or RESIZE_CROP (center crop).
            Stable string keys, not translated UI labels.
        use_grayscale: if True, convert to grayscale and replicate to 3
            channels. The from-scratch CNN was trained this way and predicts
            incorrectly without it.

    Returns the (display_image, model_input_array) pair. The display image is
    what we show in the UI; the array is what we feed the model.
    """
    image = image.convert("RGB")
    if mode == RESIZE_CROP:
        image = ImageOps.fit(image, (224, 224), method=Image.BICUBIC)
    else:
        image = ImageOps.pad(image, (224, 224), method=Image.BICUBIC, color=(0, 0, 0))
    image_array = np.array(image).astype("float32")
    if use_grayscale:
        gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        image_array = np.stack((gray,) * 3, axis=-1).astype("float32")
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array


# ---- Grad-CAM ---------------------------------------------------------------

def find_last_conv_layer(model):
    """Return the name of the last convolution-like layer.

    Accepts Conv2D *and* SeparableConv2D / DepthwiseConv2D so Grad-CAM works
    for Xception/DenseNet, not just VGG/ResNet. Falls back to any layer whose
    output has 4 dimensions (BHWC) if no explicit conv layer is found.
    """
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name
    for layer in reversed(model.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except (AttributeError, TypeError):
            continue
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Compute a Grad-CAM heatmap. Returns None on failure (e.g. shape mismatch)."""
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-6)
        return heatmap.numpy()
    except Exception:
        return None


def overlay_heatmap(image, heatmap, alpha=0.40):
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size)
    heatmap_img = ImageOps.colorize(heatmap_img.convert("L"), black="black", white="red")
    return Image.blend(image.convert("RGB"), heatmap_img.convert("RGB"), alpha=alpha)


# ---- Prediction -------------------------------------------------------------

def predict_with_model(model_name, image, threshold, resize_mode, show_explain=False):
    """Run a single model on a single image.

    The preprocessing matches what the model was trained with — in particular,
    the from-scratch CNN was trained on grayscale-replicated images, so we
    apply the same conversion at inference. Mismatched preprocessing gives
    systematically wrong predictions (this was a real bug in the original app).
    """
    cfg = MODEL_CONFIGS[model_name]
    model_path = cfg["path"]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    model = load_trained_model(model_path)
    processed_img, input_data = preprocess_image(
        image, resize_mode, use_grayscale=cfg["use_grayscale"],
    )

    start = time.time()
    prediction = float(model.predict(input_data, verbose=0)[0][0])
    elapsed_ms = int((time.time() - start) * 1000)

    label = "DR" if prediction >= threshold else "No DR"
    confidence = prediction if label == "DR" else (1.0 - prediction)
    result = {
        "model": model_name,
        "probability_dr": prediction,
        "prediction": label,
        "confidence": confidence,
        "elapsed_ms": elapsed_ms,
        "processed_img": processed_img,
        "threshold": threshold,
    }

    if show_explain:
        last_conv_layer = find_last_conv_layer(model)
        if last_conv_layer:
            heatmap = make_gradcam_heatmap(input_data, model, last_conv_layer)
            if heatmap is not None:
                result["heatmap_overlay"] = overlay_heatmap(processed_img, heatmap)
    return result
