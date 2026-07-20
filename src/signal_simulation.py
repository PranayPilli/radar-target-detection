"""
signal_simulation.py
---------------------
Simulates radar return signals for a pulsed radar system.

A radar transmits a pulse and listens for reflections ("returns").
Targets appear as amplitude peaks at a range (time) that corresponds to the
round-trip travel time of the pulse. Real radar returns are corrupted by:
    * Thermal / receiver noise (modeled as Additive White Gaussian Noise)
    * Background clutter (ground, sea, weather returns)

This module builds a synthetic range profile that mimics that behaviour so
that filtering, feature extraction, and detection algorithms can be
developed and tested without needing real radar hardware.
"""

import numpy as np


class RadarSimulator:
    """Generates synthetic radar range-profile data with configurable targets."""

    def __init__(self, num_range_bins=1000, sample_rate=1e6, noise_power=0.05,
                 clutter_power=0.02, random_seed=None):
        """
        Parameters
        ----------
        num_range_bins : int
            Number of discrete range cells in the radar sweep.
        sample_rate : float
            Sampling rate (Hz) used to convert range bins to time/range.
        noise_power : float
            Variance of the additive Gaussian receiver noise.
        clutter_power : float
            Amplitude scale of the low-frequency clutter floor.
        random_seed : int or None
            Seed for reproducibility.
        """
        self.num_range_bins = num_range_bins
        self.sample_rate = sample_rate
        self.noise_power = noise_power
        self.clutter_power = clutter_power
        self.rng = np.random.default_rng(random_seed)

        # Speed of light, used to convert range bins <-> physical range (meters)
        self.c = 3e8

    def range_bin_to_meters(self, bin_index):
        """Convert a range bin index to a physical range in meters."""
        time_per_bin = 1.0 / self.sample_rate
        return bin_index * time_per_bin * self.c / 2.0

    def _generate_pulse(self, width_bins=6):
        """Generate a normalized Gaussian-shaped transmit pulse envelope."""
        x = np.arange(-width_bins, width_bins + 1)
        pulse = np.exp(-(x ** 2) / (2 * (width_bins / 2.5) ** 2))
        return pulse / np.max(pulse)

    def _generate_clutter(self):
        """Generate slowly varying clutter across the range profile."""
        if self.clutter_power <= 0:
            return np.zeros(self.num_range_bins)
        # Smooth, low-frequency random walk to emulate ground/sea clutter
        raw = self.rng.normal(0, 1, self.num_range_bins)
        kernel = np.ones(25) / 25
        clutter = np.convolve(raw, kernel, mode="same")
        clutter = clutter / (np.max(np.abs(clutter)) + 1e-9)
        return self.clutter_power * clutter

    def generate_scene(self, targets):
        """
        Build one full radar range-profile "scene" containing targets, clutter
        and noise.

        Parameters
        ----------
        targets : list of dict
            Each dict describes one target:
                {"range_bin": int, "amplitude": float}

        Returns
        -------
        clean_signal : np.ndarray
            Ground-truth signal (targets only, no noise/clutter) - useful for
            evaluating detection performance.
        noisy_signal : np.ndarray
            The simulated received signal (targets + clutter + noise).
        target_mask : np.ndarray (bool)
            True at range bins that truly contain a target return.
        """
        clean_signal = np.zeros(self.num_range_bins)
        target_mask = np.zeros(self.num_range_bins, dtype=bool)
        pulse = self._generate_pulse()
        half_width = len(pulse) // 2

        for target in targets:
            center = target["range_bin"]
            amp = target["amplitude"]
            start = max(0, center - half_width)
            end = min(self.num_range_bins, center + half_width + 1)
            p_start = start - (center - half_width)
            p_end = p_start + (end - start)
            clean_signal[start:end] += amp * pulse[p_start:p_end]
            target_mask[max(0, center - 2):min(self.num_range_bins, center + 3)] = True

        clutter = self._generate_clutter()
        noise = self.rng.normal(0, np.sqrt(self.noise_power), self.num_range_bins)
        noisy_signal = clean_signal + clutter + noise

        return clean_signal, noisy_signal, target_mask

    def generate_random_scene(self, num_targets=3, min_amp=0.5, max_amp=1.5,
                               guard_bins=30):
        """Convenience method: place a random number of targets at random,
        well-separated range bins and generate the scene."""
        targets = []
        chosen_bins = []
        attempts = 0
        while len(targets) < num_targets and attempts < 500:
            attempts += 1
            candidate = self.rng.integers(20, self.num_range_bins - 20)
            if all(abs(candidate - b) > guard_bins for b in chosen_bins):
                chosen_bins.append(candidate)
                amplitude = self.rng.uniform(min_amp, max_amp)
                targets.append({"range_bin": int(candidate), "amplitude": float(amplitude)})

        return self.generate_scene(targets) + (targets,)
