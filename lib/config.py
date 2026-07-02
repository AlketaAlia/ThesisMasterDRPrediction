"""Project-wide constants: model registry, resize modes, safety limits.

Keeping these in one place makes it obvious where to change a threshold,
add a new model, or relax an upload limit.
"""
import os


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


# MODEL_CONFIGS: per-architecture metadata used to keep inference consistent
# with how each model was trained.
#
# - "use_grayscale": the from-scratch CNN was trained with images converted to
#   grayscale and replicated across 3 channels. Inference must do the same,
#   otherwise predictions are systematically wrong.
# - "threshold": per-model decision threshold used in comparison runs.
MODEL_CONFIGS = {
    # All transfer-learning models point to the .keras files retrained with
    # the new pipeline (architecture-specific preprocess_input + class
    # weights + stratified split). The old .h5 files used `/255` rescaling
    # and would give ~50% accuracy if loaded with preprocess_input — they
    # are kept on disk only for thesis reproducibility.
    "ResNet50": {
        "path": os.path.join(RESULTS_DIR, "resnet_model.keras"),
        "history": os.path.join(RESULTS_DIR, "resnet_model_history.pcl"),
        "use_grayscale": False,
        "threshold": 0.50,
    },
    "Xception": {
        "path": os.path.join(RESULTS_DIR, "xception_model.keras"),
        "history": os.path.join(RESULTS_DIR, "xception_model_history.pcl"),
        "use_grayscale": False,
        "threshold": 0.45,
    },
    "DenseNet121": {
        "path": os.path.join(RESULTS_DIR, "densenet_model.keras"),
        "history": os.path.join(RESULTS_DIR, "densenet_model_history.pcl"),
        "use_grayscale": False,
        "threshold": 0.50,
    },
    # VGG16 with transfer learning (ImageNet pretrained base + custom head),
    # retrained 2026-04-30 with the new train.py pipeline. The thesis-era
    # from-scratch model still exists at `vgg16_model.h5` for reference;
    # this entry uses the new transfer-learning model which gets 95.82%
    # held-out test accuracy vs the thesis's 95.03% val.
    "VGG16": {
        "path": os.path.join(RESULTS_DIR, "model_vgg16.keras"),
        "history": os.path.join(RESULTS_DIR, "model_history_vgg16.pcl"),
        "use_grayscale": False,  # transfer learning uses vgg16.preprocess_input, not grayscale
        "threshold": 0.55,
    },
    # CNN with ReLU only — retrained 2026-04-30 with the new train.py pipeline
    # (96.00% held-out test accuracy). The thesis-era model `model.h5` still
    # exists for reference. 3 Conv2D + 3 MaxPool + Flatten + 2 Dense, all ReLU,
    # ~11.2M params, trained on grayscale-replicated 3-channel inputs.
    "CNN": {
        "path": os.path.join(RESULTS_DIR, "cnn_model.keras"),
        "history": os.path.join(RESULTS_DIR, "cnn_model_history.pcl"),
        "use_grayscale": True,
        "threshold": 0.50,
    },
    # CNN(Tanh+ReLU) variant — retrained 2026-04-30. Same shape as CNN above
    # but with tanh on the second conv block and the dense head. Reaches
    # 92.91% held-out test accuracy (vs the all-ReLU CNN at 96.00%).
    "CNN (Tanh+ReLU)": {
        "path": os.path.join(RESULTS_DIR, "cnn_tanh_model.keras"),
        "history": os.path.join(RESULTS_DIR, "cnn_tanh_model_history.pcl"),
        "use_grayscale": True,
        "threshold": 0.50,
    },
}

MODEL_PATHS = {name: cfg["path"] for name, cfg in MODEL_CONFIGS.items()}
RECOMMENDED_THRESHOLDS = {name: cfg["threshold"] for name, cfg in MODEL_CONFIGS.items()}
HISTORY_PATHS = {name: cfg["history"] for name, cfg in MODEL_CONFIGS.items()}


# ZIP upload safety limits — prevent pathological archives from exhausting RAM.
MAX_ZIP_FILES = 500
MAX_ZIP_UNCOMPRESSED_BYTES = 500 * 1024 * 1024  # 500 MB

# Resize mode keys are stable identifiers (not translated strings) so the
# preprocessing logic doesn't break when the UI language changes mid-session.
RESIZE_FIT = "fit"
RESIZE_CROP = "crop"
RESIZE_MODES = [RESIZE_FIT, RESIZE_CROP]

# PDF report layout caps.
PDF_SUMMARY_ROW_LIMIT = 12
PDF_IMAGE_PAGE_LIMIT = 10

# Cap session-state history so heavy users don't accumulate unbounded state.
HISTORY_LIMIT = 1000


# 5-class severity grading models (master-thesis Phase 3). Use sigmoid-style
# threshold semantics for binary; multi-class models use argmax + softmax
# probabilities. Class names are the standard APTOS 5-stage scale.
DR_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

MULTICLASS_MODEL_CONFIGS = {
    "ResNet50 (5-class)": {
        "path": os.path.join(RESULTS_DIR, "resnet50_5class_model.keras"),
        "history": os.path.join(RESULTS_DIR, "resnet50_5class_model_history.pcl"),
        "use_grayscale": False,
        "preprocess_arch": "resnet50",
    },
    "CNN (5-class)": {
        "path": os.path.join(RESULTS_DIR, "cnn_5class_model.keras"),
        "history": os.path.join(RESULTS_DIR, "cnn_5class_model_history.pcl"),
        "use_grayscale": True,
        "preprocess_arch": None,
    },
}

MULTICLASS_MODEL_PATHS = {n: c["path"] for n, c in MULTICLASS_MODEL_CONFIGS.items()}

# How many trained Keras models can live in memory at once. Each transfer
# model is ~100 MB; bound this to keep RAM use sensible on small laptops.
MAX_LOADED_MODELS = 2
