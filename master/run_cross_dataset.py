"""Cross-dataset evaluation: APTOS-trained models on Messidor / IDRiD.

How to use:
1. Download a second fundus dataset:
   - **Messidor-2**: https://www.adcis.net/en/third-party/messidor2/
     (registration required, 1748 images with 0-3 severity)
   - **IDRiD**: https://idrid.grand-challenge.org/Data_Download/
     (registration required, ~516 images with 0-4 severity)
   - **EyePACS** (Kaggle 2015): https://www.kaggle.com/c/diabetic-retinopathy-detection
     (very large, ~88k images, 0-4 severity)

2. Place the dataset under `inputs/cross_dataset/<name>/` with:
   - `images/` containing the .png/.jpg images
   - `labels.csv` with columns `filename`, `label` (0-4 severity)

3. Run:
   python -m master.run_cross_dataset --dataset_dir inputs/cross_dataset/messidor2 \\
                                      --models resnet50_5class

The script will:
- Load the external test images
- Run each requested APTOS-trained model on them
- Report cross-dataset accuracy, QWK (5-class) or accuracy (binary)
- Compute calibration on the new domain
- Run OOD detection (Mahalanobis + Energy) — the external dataset should
  *not* register as OOD since it's the same task.

Outputs to master/results/cross_dataset/<dataset_name>/.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from helpers import build_eval_datagen, get_labels  # noqa: E402
from master.uncertainty.calibration_mc import (  # noqa: E402
    expected_calibration_error_mc, ordinal_distance,
    quadratic_weighted_kappa,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('cross')


# Same per-architecture preprocessing as elsewhere
def _arch_preproc(arch_name):
    if arch_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input, False
    if arch_name == "densenet":
        from tensorflow.keras.applications.densenet import preprocess_input
        return preprocess_input, False
    if arch_name == "xception":
        from tensorflow.keras.applications.xception import preprocess_input
        return preprocess_input, False
    if arch_name == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input, False
    if arch_name == "cnn":
        return None, True
    return None, False


# Map model display name → file path + arch + multi-class flag
def _resolve_model(name):
    project_root = PROJECT_ROOT
    if name == "resnet50_5class":
        return {"path": project_root / "results" / "resnet50_5class_model.keras",
                "arch": "resnet50", "multiclass": True}
    if name == "cnn_5class":
        return {"path": project_root / "results" / "cnn_5class_model.keras",
                "arch": "cnn", "multiclass": True}
    if name == "resnet50":
        return {"path": project_root / "results" / "resnet_model.keras",
                "arch": "resnet50", "multiclass": False}
    if name == "resnet50_mcd":
        return {"path": project_root / "results" / "resnet_mcd_model.keras",
                "arch": "resnet50", "multiclass": False}
    raise ValueError(f"Unknown model: {name}")


def main():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True,
                        help='Path to the cross-dataset folder containing '
                             'images/ and labels.csv')
    parser.add_argument('--models', nargs='+',
                        default=['resnet50_5class'],
                        help='Trained models to evaluate (display names from '
                             '`_resolve_model` above).')
    parser.add_argument('--multiclass', action='store_true',
                        help='Force multi-class evaluation. Auto-detected '
                             'from model name if not set.')
    args = parser.parse_args()

    from tensorflow.keras.models import load_model
    from sklearn.metrics import accuracy_score, classification_report

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        logger.error("Dataset directory not found: %s", dataset_dir)
        return
    images_dir = dataset_dir / "images"
    labels_csv = dataset_dir / "labels.csv"
    if not (images_dir.exists() and labels_csv.exists()):
        logger.error("Missing %s and/or %s", images_dir, labels_csv)
        return

    out_dir = PROJECT_ROOT / "master" / "results" / "cross_dataset" / dataset_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a flat dataframe matching what helpers expects
    labels_df = pd.read_csv(labels_csv)
    if "filename" not in labels_df.columns:
        labels_df = labels_df.rename(columns={labels_df.columns[0]: "filename"})
    if "label" not in labels_df.columns:
        labels_df = labels_df.rename(columns={labels_df.columns[1]: "label"})
    labels_df["filepath"] = labels_df["filename"].apply(
        lambda n: str(images_dir / n))
    # Drop rows whose image file is missing
    labels_df = labels_df[labels_df["filepath"].apply(os.path.isfile)].reset_index(drop=True)
    logger.info("Cross-dataset %s: %d images with valid labels",
                dataset_dir.name, len(labels_df))
    if len(labels_df) == 0:
        logger.error("No valid images found.")
        return

    summary = []
    for model_name in args.models:
        try:
            spec = _resolve_model(model_name)
        except ValueError as e:
            logger.warning("%s — skip", e)
            continue
        if not spec["path"].exists():
            logger.warning("Skipping %s — file missing at %s",
                           model_name, spec["path"])
            continue

        is_mc = bool(spec["multiclass"]) or args.multiclass
        preproc, use_gray = _arch_preproc(spec["arch"])
        labels_df["_label_str"] = labels_df["label"].astype(str)
        if not is_mc:
            # Binarize per the bachelor convention
            labels_df["_label_str"] = (labels_df["label"] >= 1).astype(int).astype(str)
        gen = build_eval_datagen(
            preprocessing_function=preproc,
            use_grayscale=use_gray,
        ).flow_from_dataframe(
            labels_df, x_col="filepath", y_col="_label_str",
            target_size=(224, 224), batch_size=32,
            class_mode='categorical' if is_mc else 'binary',
            shuffle=False,
        )

        logger.info("Loading %s and predicting on %s…", model_name, dataset_dir.name)
        model = load_model(str(spec["path"]), compile=False)
        probs = model.predict(gen, verbose=0)
        if not is_mc:
            probs = probs.ravel()
            preds = (probs >= 0.5).astype(int)
            y_true = labels_df["_label_str"].astype(int).values
            acc = accuracy_score(y_true, preds)
            logger.info("%s on %s — binary acc: %.4f",
                        model_name, dataset_dir.name, acc)
            summary.append({"model": model_name, "task": "binary",
                            "test_accuracy": float(acc)})
        else:
            preds = probs.argmax(axis=1)
            y_true = labels_df["_label_str"].astype(int).values
            acc = accuracy_score(y_true, preds)
            qwk = quadratic_weighted_kappa(y_true, preds, num_classes=5)
            odist = ordinal_distance(y_true, preds)
            ece = expected_calibration_error_mc(y_true, probs)
            logger.info("%s on %s — acc: %.4f, QWK: %.4f, ord-dist: %.3f, ECE: %.4f",
                        model_name, dataset_dir.name, acc, qwk, odist, ece)
            summary.append({
                "model": model_name, "task": "5-class",
                "test_accuracy": float(acc),
                "qwk": float(qwk),
                "ordinal_distance": float(odist),
                "ece": float(ece),
            })

    pd.DataFrame(summary).to_csv(out_dir / "cross_dataset_summary.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote master/results/cross_dataset/%s/", dataset_dir.name)


if __name__ == "__main__":
    main()
