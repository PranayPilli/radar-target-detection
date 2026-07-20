"""
detection.py
------------
Target detection logic:

1. CFAR (Constant False Alarm Rate) detector - the classic, industry-standard
   radar detection algorithm. It adapts its detection threshold to the local
   noise/clutter level in "training cells" surrounding a "cell under test",
   so it maintains a constant false-alarm rate even when noise power varies
   across the range profile.

2. A scikit-learn RandomForestClassifier trained on simulated labeled data,
   used to classify candidate peaks as "target" vs "false alarm" using the
   engineered features from feature_extraction.py. This demonstrates a
   machine-learning based complement to the classical CFAR approach.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def cfar_detector(signal_data, num_train=15, num_guard=4, threshold_factor=3.5):
    """
    1D Cell-Averaging CFAR detector.

    For every "cell under test" (CUT), average the power of the surrounding
    training cells (excluding guard cells immediately adjacent to the CUT,
    which might contain energy leaking from the target itself). A detection
    is declared if the CUT exceeds (local_noise_average * threshold_factor).

    Parameters
    ----------
    signal_data : np.ndarray
        The (ideally filtered) radar range profile.
    num_train : int
        Number of training cells on each side of the CUT.
    num_guard : int
        Number of guard cells on each side of the CUT.
    threshold_factor : float
        Multiplier applied to the local noise estimate to set the threshold.
        Higher = fewer false alarms, but may miss weak targets.

    Returns
    -------
    detections : np.ndarray (bool)
        True at range bins flagged as detections.
    threshold_map : np.ndarray
        The adaptive threshold value at every range bin (useful for plotting).
    """
    n = len(signal_data)
    power = signal_data ** 2
    detections = np.zeros(n, dtype=bool)
    threshold_map = np.zeros(n)

    window = num_train + num_guard

    for i in range(window, n - window):
        training_cells = np.concatenate([
            power[i - window:i - num_guard],
            power[i + num_guard + 1:i + window + 1]
        ])
        noise_level = np.mean(training_cells)
        threshold = noise_level * threshold_factor
        threshold_map[i] = np.sqrt(threshold)  # back to amplitude domain

        if power[i] > threshold:
            detections[i] = True

    return detections, threshold_map


def train_target_classifier(features_train, labels_train, random_state=42):
    """Train a RandomForest classifier to distinguish true targets from
    noise/clutter false alarms based on engineered features."""
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_state)
    clf.fit(features_train, labels_train)
    return clf


def evaluate_classifier(clf, features_test, labels_test):
    """Return a text classification report and confusion matrix."""
    predictions = clf.predict(features_test)
    report = classification_report(labels_test, predictions, target_names=["false_alarm", "target"])
    matrix = confusion_matrix(labels_test, predictions)
    return report, matrix, predictions


def build_training_dataset(simulator, filter_fn, feature_fn_extract, feature_fn_peaks,
                            num_scenes=60, targets_per_scene=3):
    """
    Generate a labeled dataset for the ML classifier by repeatedly simulating
    scenes, filtering them, extracting candidate-peak features, and labeling
    each candidate peak as a true target (1) or false alarm (0) based on
    ground truth.
    """
    all_features = []
    all_labels = []

    for _ in range(num_scenes):
        clean, noisy, mask, targets = simulator.generate_random_scene(
            num_targets=targets_per_scene
        )
        filtered = filter_fn(noisy)
        noise_std = np.std(filtered[:20])  # first 20 bins assumed target-free
        peaks, _ = feature_fn_peaks(filtered, min_height=noise_std * 1.5)
        features, _ = feature_fn_extract(filtered, peaks, noise_std)

        for idx, peak in enumerate(peaks):
            label = 1 if mask[peak] else 0
            all_features.append(features[idx])
            all_labels.append(label)

    return np.array(all_features), np.array(all_labels)


def train_test_split_dataset(features, labels, test_size=0.3, random_state=42):
    return train_test_split(features, labels, test_size=test_size,
                             random_state=random_state, stratify=labels)
