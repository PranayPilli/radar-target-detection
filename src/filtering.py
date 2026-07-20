"""
filtering.py
------------
Digital filtering techniques used to reduce noise and improve the quality of
a raw radar return signal before detection/feature extraction.

Implements:
    * Moving-average (boxcar) filter
    * Butterworth low-pass filter (zero-phase, via filtfilt)
    * Matched filter (correlation against the known transmit pulse shape) -
      the theoretically optimal filter for maximizing SNR of a known pulse
      in white noise.
"""

import numpy as np
from scipy import signal


def moving_average_filter(x, window_size=5):
    """Simple boxcar smoothing filter. Fast, but blurs sharp target peaks."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(x, kernel, mode="same")


def butterworth_lowpass_filter(x, cutoff=0.1, order=4):
    """
    Zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    cutoff : float
        Normalized cutoff frequency (0 < cutoff < 1, where 1 = Nyquist).
    order : int
        Filter order (higher = steeper roll-off).
    """
    b, a = signal.butter(order, cutoff, btype="low")
    return signal.filtfilt(b, a, x)


def matched_filter(x, pulse_template):
    """
    Correlate the received signal with the known transmit pulse shape.
    This is the optimal linear filter for detecting a known signal shape
    buried in additive white Gaussian noise.
    """
    template = pulse_template / np.linalg.norm(pulse_template)
    filtered = signal.correlate(x, template, mode="same")
    return filtered


def reference_pulse(width_bins=6):
    """Regenerate the same Gaussian pulse shape used by the simulator, so the
    matched filter has a template to correlate against."""
    x = np.arange(-width_bins, width_bins + 1)
    pulse = np.exp(-(x ** 2) / (2 * (width_bins / 2.5) ** 2))
    return pulse / np.max(pulse)


def compute_snr_improvement(raw_signal, filtered_signal, noise_region_slice):
    """
    Estimate the SNR (in dB) of the raw vs filtered signal, measured over a
    known noise-only region of the range profile, to quantify how much a
    filtering step improved signal quality.
    """
    def snr_db(sig):
        noise_std = np.std(sig[noise_region_slice])
        peak = np.max(np.abs(sig))
        if noise_std == 0:
            return float("inf")
        return 20 * np.log10(peak / noise_std)

    return {
        "raw_snr_db": snr_db(raw_signal),
        "filtered_snr_db": snr_db(filtered_signal),
    }
