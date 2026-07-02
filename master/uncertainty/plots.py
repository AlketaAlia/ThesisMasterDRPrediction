"""Plotting helpers for the uncertainty analysis."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(mean_conf, accuracy, sizes, ece, save_path,
                             title="Reliability Diagram"):
    """Reliability diagram: bar plot of accuracy vs confidence per bin.

    The diagonal is perfect calibration. Bars below diagonal = overconfident,
    above = underconfident. Bin widths reflect bin size.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1.5, label='Perfect calibration')
    if len(mean_conf) > 0:
        ax.bar(mean_conf, accuracy, width=0.04, alpha=0.7, edgecolor='black',
               label='Model')
        # Gap markers
        for c, a in zip(mean_conf, accuracy):
            ax.plot([c, c], [c, a], color='red', alpha=0.5, linewidth=1)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_risk_coverage(curves, save_path, title="Risk-Coverage Curve"):
    """Selective accuracy as coverage varies, one line per uncertainty source."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, points in curves.items():
        ax.plot(points[:, 0], points[:, 1], marker='o', label=label, alpha=0.85)
    ax.set_xlabel('Coverage (fraction of test set retained)')
    ax.set_ylabel('Selective accuracy')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_uncertainty_histogram(uncertainty, correct, save_path,
                               title="Uncertainty distribution"):
    """Histogram of uncertainty for correct vs incorrect predictions."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(uncertainty.max(), 1e-3), 30)
    ax.hist(uncertainty[correct == 1], bins=bins, alpha=0.6, label='Correct',
            color='steelblue')
    ax.hist(uncertainty[correct == 0], bins=bins, alpha=0.6, label='Wrong',
            color='salmon')
    ax.set_xlabel('Uncertainty score')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_pairwise_pvalue_heatmap(names, pvalue_matrix, save_path,
                                 title="Pairwise McNemar p-values"):
    """Heatmap of McNemar p-values between models. Cells with p<0.05 highlighted."""
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pvalue_matrix, cmap='RdYlGn', vmin=0, vmax=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = pvalue_matrix[i, j]
            text = f"{v:.3f}" if not np.isnan(v) else ""
            color = 'black' if v > 0.05 else 'white'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='p-value')
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
