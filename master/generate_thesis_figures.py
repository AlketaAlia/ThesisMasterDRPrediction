"""Render publication-quality figures from the saved analysis CSVs / JSONs.

Generates the figures the thesis chapters will reference. Output goes to
`master/thesis/figures/`. Style is plain matplotlib with consistent fonts,
no seaborn flashing, 300 DPI for print, and PDF + PNG side-by-side.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "master" / "thesis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Plain, journal-friendly defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name):
    """Save both a PNG (for previews) and PDF (for LaTeX include)."""
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def fig_phase1_accuracy_with_ci():
    """Bar chart of test accuracy with bootstrap 95% CI per model."""
    p = PROJECT_ROOT / "master" / "results" / "per_model_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df.sort_values("test_accuracy")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y = np.arange(len(df))
    means = df["test_accuracy"].values * 100
    lo = df["test_acc_ci_lo"].values * 100
    hi = df["test_acc_ci_hi"].values * 100
    err = np.array([means - lo, hi - means])
    bars = ax.barh(y, means, xerr=err, color="#4C72B0",
                   edgecolor="black", linewidth=0.5,
                   error_kw={"elinewidth": 1, "capsize": 3})
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"].values)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(85, 100)
    ax.axvline(95, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Per-model test accuracy with bootstrap 95% CI")
    _save(fig, "fig_phase1_accuracy_ci")


def fig_phase1_risk_coverage():
    """Selective accuracy curves for the 6-model ensemble."""
    files = {
        "predictive entropy": "rc_curve_predictive_entropy.csv",
        "std (epistemic)": "rc_curve_std_epistemic_spread.csv",
        "mutual information": "rc_curve_mutual_information.csv",
        "1 - max prob": "rc_curve_1_-_max_prob.csv",
    }
    fig, ax = plt.subplots(figsize=(5.5, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for (label, fname), color in zip(files.items(), colors):
        p = PROJECT_ROOT / "master" / "results" / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        ax.plot(df["coverage"], df["accuracy"] * 100,
                marker="o", color=color, label=label, linewidth=1.2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective accuracy (%)")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(96, 100)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Risk-coverage curves (binary ensemble of 6 models)")
    ax.grid(True, alpha=0.3)
    _save(fig, "fig_phase1_risk_coverage")


def fig_phase2_kfold():
    """5-fold CV accuracy bar chart with mean line."""
    p = PROJECT_ROOT / "master" / "results" / "kfold" / "resnet50_fold_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fold_acc = df["test_accuracy"] * 100
    folds = df["fold"]
    ax.bar(folds, fold_acc, color="#55A868", edgecolor="black", linewidth=0.5)
    mean_acc = fold_acc.mean()
    ax.axhline(mean_acc, color="black", linestyle="--", linewidth=1,
               label=f"Mean = {mean_acc:.2f}%")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(94, 97)
    ax.set_xticks(folds)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("ResNet50 — 5-fold CV test accuracy")
    _save(fig, "fig_phase2_kfold")


def fig_phase2_ood_aurocs():
    """OOD method comparison."""
    p = PROJECT_ROOT / "master" / "results" / "phase2" / "ood_metrics.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)
    methods = list(data.keys())
    aurocs = [data[m]["auroc_id_vs_ood"] for m in methods]
    fprs = [data[m]["fpr_at_tpr95"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    colors = ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]
    ax1.bar(methods, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("AUROC (ID vs OOD)")
    ax1.set_ylim(0.5, 1.05)
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_title("OOD detection AUROC (higher better)")
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(methods, fprs, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("FPR @ TPR=95%")
    ax2.set_ylim(0, 0.4)
    ax2.set_title("OOD false-positive rate (lower better)")
    ax2.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    _save(fig, "fig_phase2_ood")


def fig_phase3_per_class():
    """Per-class precision/recall/F1 for ResNet50 5-class."""
    p = PROJECT_ROOT / "master" / "results" / "multiclass" / "resnet50_5class_per_class.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    metrics = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    width = 0.27
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (m, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + (i - 1) * width, df[m], width, label=m,
               color=color, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("ResNet50 (5-class) — per-class precision / recall / F1")
    ax.legend(frameon=False)
    _save(fig, "fig_phase3_per_class")


def fig_phase3_confusion():
    """5-class confusion matrix as a heatmap (numeric)."""
    p = PROJECT_ROOT / "master" / "results" / "multiclass" / "resnet50_5class_per_class.csv"
    # We saved confusion_matrix as PNG only; reuse, or recompute.
    p_cm = PROJECT_ROOT / "master" / "results" / "multiclass" / "resnet50_5class_confusion_matrix.png"
    if p_cm.exists():
        # Just copy/reuse: thesis can include the existing PNG.
        return
    return


def main():
    fig_phase1_accuracy_with_ci()
    fig_phase1_risk_coverage()
    fig_phase2_kfold()
    fig_phase2_ood_aurocs()
    fig_phase3_per_class()
    print(f"Wrote figures to {OUT}/")


if __name__ == "__main__":
    main()
