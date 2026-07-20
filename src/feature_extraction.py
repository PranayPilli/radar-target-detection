"""
feature_extraction.py
----------------------
Extracts numerical features around candidate peaks in a processed radar
signal. These features feed both the CFAR-style detector and a
scikit-learn classifier used for target identification.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths


def find_candidate_peaks(signal_data, min_height=None, min_distance=5):
    """Locate local maxima that could plausibly be target returns."""
    if min_height is None:
        min_height = np.mean(signal_data) + np.std(signal_data)
    peaks, properties = find_peaks(signal_data, height=min_height, distance=min_distance)
    return peaks, properties


def extract_features(signal_data, peak_indices, noise_std):
    """
    Build a feature vector for each candidate peak.

    Features
    --------
    amplitude        : peak amplitude
    snr              : peak amplitude relative to local noise std (linear ratio)
    width             : width of the peak at half prominence (range resolution proxy)
    prominence        : how much the peak stands out from its surroundings
    local_mean        : mean amplitude in a small window around the peak
    """
    if len(peak_indices) == 0:
        return np.empty((0, 5)), []

    widths, _, _, _ = peak_widths(signal_data, peak_indices, rel_height=0.5)

    features = []
    for i, idx in enumerate(peak_indices):
        amplitude = signal_data[idx]
        snr = amplitude / (noise_std + 1e-9)
        width = widths[i]

        lo, hi = max(0, idx - 10), min(len(signal_data), idx + 10)
        window = signal_data[lo:hi]
        local_mean = np.mean(window)
        prominence = amplitude - np.median(window)

        features.append([amplitude, snr, width, prominence, local_mean])

    return np.array(features), ["amplitude", "snr", "width", "prominence", "local_mean"]
