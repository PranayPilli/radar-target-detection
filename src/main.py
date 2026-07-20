"""
main.py
-------
End-to-end radar target detection & signal analysis pipeline.

Run this script to:
    1. Simulate a radar scene (targets + clutter + noise)
    2. Apply digital filtering to reduce noise
    3. Run a CFAR detector to flag candidate targets
    4. Extract features for each candidate peak
    5. Train/evaluate a scikit-learn classifier to separate true targets
       from false alarms
    6. Save all plots and a CSV summary report to output/

Usage
-----
    python src/main.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow running as `python src/main.py` from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_simulation import RadarSimulator
from filtering import butterworth_lowpass_filter, compute_snr_improvement
from feature_extraction import find_candidate_peaks, extract_features
from detection import (
    cfar_detector,
    build_training_dataset,
    train_target_classifier,
    evaluate_classifier,
    train_test_split_dataset,
)
from visualization import (
    plot_raw_vs_filtered,
    plot_cfar_detection,
    plot_feature_scatter,
    plot_confusion_matrix,
    plot_feature_importance,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")


def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("RADAR TARGET DETECTION & SIGNAL ANALYSIS PIPELINE")
    print("=" * 60)

    # ---- 1. Simulate a radar scene -------------------------------------
    print("\n[1/6] Simulating radar scene...")
    sim = RadarSimulator(num_range_bins=1000, noise_power=0.05,
                          clutter_power=0.03, random_seed=7)
    clean, noisy, mask, targets = sim.generate_random_scene(num_targets=4)
    print(f"   -> Placed {len(targets)} targets at range bins: "
          f"{[t['range_bin'] for t in targets]}")

    # ---- 2. Filter the signal -------------------------------------------
    print("\n[2/6] Applying Butterworth low-pass filter...")
    filtered = butterworth_lowpass_filter(noisy, cutoff=0.15, order=4)
    snr_stats = compute_snr_improvement(noisy, filtered, noise_region_slice=slice(0, 20))
    print(f"   -> Raw SNR:      {snr_stats['raw_snr_db']:.2f} dB")
    print(f"   -> Filtered SNR: {snr_stats['filtered_snr_db']:.2f} dB")

    plot_raw_vs_filtered(noisy, filtered, os.path.join(OUTPUT_DIR, "01_raw_vs_filtered.png"))

    # ---- 3. CFAR detection ------------------------------------------------
    print("\n[3/6] Running CFAR detector...")
    detections, threshold_map = cfar_detector(filtered, num_train=15, num_guard=4,
                                               threshold_factor=6.0)
    detected_bins = np.where(detections)[0]
    print(f"   -> CFAR flagged {len(detected_bins)} range bins as detections")

    plot_cfar_detection(filtered, threshold_map, detections, targets,
                         os.path.join(OUTPUT_DIR, "02_cfar_detection.png"))

    # ---- 4. Feature extraction --------------------------------------------
    print("\n[4/6] Extracting features from candidate peaks...")
    noise_std = np.std(filtered[:20])
    peaks, _ = find_candidate_peaks(filtered, min_height=noise_std * 1.5)
    features, feature_names = extract_features(filtered, peaks, noise_std)
    print(f"   -> {len(peaks)} candidate peaks found, "
          f"{features.shape[1]} features extracted per peak")

    # ---- 5. Train ML classifier on a larger simulated dataset -------------
    print("\n[5/6] Building training dataset and training classifier...")
    print("   -> Simulating 60 additional scenes for training data...")
    X, y = build_training_dataset(sim, butterworth_lowpass_filter,
                                   extract_features, find_candidate_peaks,
                                   num_scenes=60, targets_per_scene=3)
    print(f"   -> Dataset: {X.shape[0]} samples "
          f"({int(y.sum())} targets / {int((1 - y).sum())} false alarms)")

    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)
    clf = train_target_classifier(X_train, y_train)
    report, matrix, predictions = evaluate_classifier(clf, X_test, y_test)
    print("\n   Classification report (held-out test set):")
    print("   " + report.replace("\n", "\n   "))

    plot_feature_scatter(X, y, feature_names,
                          os.path.join(OUTPUT_DIR, "03_feature_space.png"))
    plot_confusion_matrix(matrix, os.path.join(OUTPUT_DIR, "04_confusion_matrix.png"))
    plot_feature_importance(clf, feature_names,
                             os.path.join(OUTPUT_DIR, "05_feature_importance.png"))

    # ---- 6. Save summary CSV report ---------------------------------------
    print("\n[6/6] Saving CSV summary report...")
    rows = []
    for idx, peak in enumerate(peaks):
        is_true_target = bool(mask[peak])
        rows.append({
            "range_bin": int(peak),
            "range_m": round(sim.range_bin_to_meters(peak), 2),
            "amplitude": round(features[idx, 0], 4),
            "snr": round(features[idx, 1], 4),
            "width": round(features[idx, 2], 4),
            "prominence": round(features[idx, 3], 4),
            "cfar_detected": bool(detections[peak]),
            "ground_truth_target": is_true_target,
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "detection_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"   -> Report saved to {csv_path}")

    print("\n" + "=" * 60)
    print(f"DONE. All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
