"""
visualization.py
-----------------
Plotting utilities for radar signals, filtering results, CFAR detections,
and classifier evaluation metrics. All plots are saved as PNG files to the
output/ directory so they can be reviewed after the pipeline runs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_raw_vs_filtered(raw, filtered, output_path, title="Raw vs Filtered Radar Signal"):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(raw, color="tab:gray", linewidth=0.8)
    axes[0].set_title("Raw Radar Return (with noise & clutter)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(filtered, color="tab:blue", linewidth=1.0)
    axes[1].set_title("Filtered Radar Return")
    axes[1].set_xlabel("Range bin")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cfar_detection(filtered_signal, threshold_map, detections, true_targets,
                         output_path, title="CFAR Target Detection"):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(filtered_signal, color="tab:blue", label="Filtered signal", linewidth=1.0)
    ax.plot(threshold_map, color="tab:orange", linestyle="--", label="CFAR adaptive threshold")

    detected_bins = np.where(detections)[0]
    ax.scatter(detected_bins, filtered_signal[detected_bins], color="red", marker="x",
               s=60, label="CFAR detection", zorder=5)

    for t in true_targets:
        ax.axvline(t["range_bin"], color="green", alpha=0.4, linestyle=":",
                   label="True target" if t is true_targets[0] else None)

    ax.set_title(title)
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_scatter(features, labels, feature_names, output_path,
                          title="Extracted Feature Space (SNR vs Amplitude)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = np.array(labels)

    amp_idx = feature_names.index("amplitude")
    snr_idx = feature_names.index("snr")

    targets = labels == 1
    ax.scatter(features[targets, amp_idx], features[targets, snr_idx],
               color="green", label="True target", alpha=0.7)
    ax.scatter(features[~targets, amp_idx], features[~targets, snr_idx],
               color="red", label="False alarm", alpha=0.7)

    ax.set_xlabel("Amplitude")
    ax.set_ylabel("SNR (linear)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix, output_path, title="Classifier Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                   display_labels=["false_alarm", "target"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(clf, feature_names, output_path,
                             title="Feature Importance (Random Forest)"):
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(np.array(feature_names)[order], importances[order], color="tab:purple")
    ax.set_title(title)
    ax.set_ylabel("Importance")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
