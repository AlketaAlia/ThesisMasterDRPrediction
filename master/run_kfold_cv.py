"""K-fold cross-validation for one or more architectures.

Trains each architecture K times (default K=5) on different folds of the
non-test data, holding out a fold for validation each time. Always
evaluates on the same global held-out test set so all folds report a
test metric on identical data — the variability across folds is then a
proper estimate of training variability.

Run from the project root:
    python -m master.run_kfold_cv --arch cnn --folds 5
    python -m master.run_kfold_cv --arch resnet50 --folds 5 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from helpers import (  # noqa: E402
    build_eval_datagen, build_input_df, build_train_datagen,
    compute_balanced_class_weights, get_labels, read_all_images,
    stratified_split,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('kfold')

KFOLD_RESULTS_DIR = PROJECT_ROOT / "master" / "results" / "kfold"
KFOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True,
                        help='architecture name passed to train.get_arch_spec')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--test-size', type=float, default=0.15)
    args = parser.parse_args()

    # Lazy imports — bringing in TF/Keras after argparse keeps --help fast.
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    )
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import accuracy_score, roc_auc_score

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train import get_arch_spec  # noqa: E402

    inputs_dir = PROJECT_ROOT / "inputs"

    labels = get_labels(str(inputs_dir / "labels.csv"))
    images = read_all_images(str(inputs_dir / "images"), use_pickle=True)
    data = build_input_df(images, labels)

    # Held-out test set is fixed across folds so test metrics are comparable.
    train_val_df, _, test_df = stratified_split(
        data, test_size=args.test_size, val_size=0.0, random_state=args.seed)
    logger.info("Train+val pool: %d, fixed test: %d",
                len(train_val_df), len(test_df))

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed)
    indices = np.arange(len(train_val_df))
    labels_array = train_val_df['label'].astype(int).values

    fold_rows = []
    test_probs_per_fold = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(indices, labels_array), start=1):
        logger.info("=== Fold %d/%d ===", fold_idx, args.folds)
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        spec = get_arch_spec(args.arch, str(inputs_dir),
                             (224, 224))

        train_gen = build_train_datagen(
            preprocessing_function=spec.preprocess,
            use_grayscale=spec.use_grayscale,
        ).flow_from_dataframe(
            train_df, x_col='filepath', y_col='label',
            target_size=(224, 224), batch_size=args.batch_size,
            class_mode='binary', seed=args.seed,
        )
        val_gen = build_eval_datagen(
            preprocessing_function=spec.preprocess,
            use_grayscale=spec.use_grayscale,
        ).flow_from_dataframe(
            val_df, x_col='filepath', y_col='label',
            target_size=(224, 224), batch_size=args.batch_size,
            class_mode='binary', shuffle=False,
        )
        test_gen = build_eval_datagen(
            preprocessing_function=spec.preprocess,
            use_grayscale=spec.use_grayscale,
        ).flow_from_dataframe(
            test_df, x_col='filepath', y_col='label',
            target_size=(224, 224), batch_size=args.batch_size,
            class_mode='binary', shuffle=False,
        )

        model, base_model = spec.builder()
        model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        fold_dir = KFOLD_RESULTS_DIR / f"{args.arch}_fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_ckpt = fold_dir / "best_model.keras"

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=args.patience,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=max(3, args.patience // 2),
                              min_lr=1e-6),
            ModelCheckpoint(str(model_ckpt), monitor='val_accuracy',
                            save_best_only=True, verbose=0),
        ]

        cw = compute_balanced_class_weights(train_df['label'])
        t0 = time.time()
        history = model.fit(train_gen, epochs=args.epochs,
                            validation_data=val_gen, callbacks=callbacks,
                            class_weight=cw, verbose=2)
        elapsed = time.time() - t0

        # Reload best checkpoint for evaluation.
        if model_ckpt.exists():
            best_model = load_model(str(model_ckpt), compile=False)
        else:
            best_model = model

        test_probs = best_model.predict(test_gen, verbose=0).ravel()
        y_test = test_df['label'].astype(int).values
        test_pred = (test_probs >= 0.5).astype(int)
        test_acc = accuracy_score(y_test, test_pred)
        try:
            test_auc = roc_auc_score(y_test, test_probs)
        except ValueError:
            test_auc = float('nan')

        best_val_acc = max(history.history.get('val_accuracy', [0]))
        n_epochs = len(history.history.get('val_accuracy', []))

        fold_rows.append({
            "arch": args.arch,
            "fold": fold_idx,
            "epochs_trained": n_epochs,
            "best_val_accuracy": float(best_val_acc),
            "test_accuracy": float(test_acc),
            "test_auc": float(test_auc),
            "elapsed_sec": float(elapsed),
        })

        with open(fold_dir / "history.pcl", "wb") as f:
            pickle.dump(history.history, f)
        np.save(fold_dir / "test_probs.npy", test_probs)
        test_probs_per_fold.append(test_probs)

        logger.info("Fold %d done: test_acc=%.4f, val_acc=%.4f, %ds",
                    fold_idx, test_acc, best_val_acc, int(elapsed))

    # Aggregate
    df = pd.DataFrame(fold_rows)
    df.to_csv(KFOLD_RESULTS_DIR / f"{args.arch}_fold_summary.csv", index=False)

    summary = {
        "arch": args.arch,
        "folds": args.folds,
        "test_accuracy_mean": float(df['test_accuracy'].mean()),
        "test_accuracy_std": float(df['test_accuracy'].std(ddof=1)),
        "test_auc_mean": float(df['test_auc'].mean()),
        "test_auc_std": float(df['test_auc'].std(ddof=1)),
        "best_val_accuracy_mean": float(df['best_val_accuracy'].mean()),
        "best_val_accuracy_std": float(df['best_val_accuracy'].std(ddof=1)),
    }

    # Average per-test-sample probability across folds (proxy for "robust prediction")
    fold_probs = np.stack(test_probs_per_fold)
    np.save(KFOLD_RESULTS_DIR / f"{args.arch}_fold_test_probs.npy", fold_probs)

    with open(KFOLD_RESULTS_DIR / f"{args.arch}_kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: acc = %.4f ± %.4f (std), auc = %.4f ± %.4f",
                summary['test_accuracy_mean'], summary['test_accuracy_std'],
                summary['test_auc_mean'], summary['test_auc_std'])
    logger.info("Wrote master/results/kfold/%s_kfold_summary.json", args.arch)


if __name__ == "__main__":
    main()
