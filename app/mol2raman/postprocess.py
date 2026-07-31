# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V. (https://wfront.nl)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Functional port of algorithms from Mol2Raman, Copyright (c) 2025
# salvasorrentino, MIT (https://github.com/salvasorrentino/Mol2Raman).
"""Deterministic post-processing for Mol2Raman predictions.

These are faithful NumPy/SciPy ports of the maths in Mol2Raman's
``utils_data_processing.py`` / ``script_prediction/final_output.py`` (MIT,
salvasorrentino). They contain no learned parameters, so they are correct and
testable on their own — independent of whether torch or the model weights are
installed.

The model emits two 800-point bands (a low/"fingerprint" band and a high/"CH"
band). ``final_output.py`` stitches them as ``first 700 of low`` + ``mean of the
100-point overlap`` + ``last 700 of high`` → 1500 points, then convolves with a
Lorentzian to turn the sparse peak picks into a continuous spectrum.
"""
from __future__ import annotations

import numpy as np
import scipy.signal


# ── Single-band operations (ported verbatim) ──────────────────────────────────

def rescale(n_out: int, arr_pred) -> np.ndarray:
    """Resample ``arr_pred`` onto ``n_out`` evenly spaced points (their ``rescale``)."""
    arr_pred = np.asarray(arr_pred, dtype=float)
    return np.interp(np.linspace(0, 1, n_out),
                     np.linspace(0, 1, len(arr_pred)), arr_pred)


def keep_peaks_prom(data, n: int) -> np.ndarray:
    """Zero everything except the ``n`` most prominent peaks (their ``keep_peaks_prom``)."""
    data = np.asarray(data, dtype=float)
    n = max(int(n), 0)
    result = np.zeros_like(data)
    if n == 0:
        return result
    peaks, _ = scipy.signal.find_peaks(data)
    if peaks.size == 0:
        return result
    prominences = scipy.signal.peak_prominences(data, peaks)[0]
    top_n = peaks[np.argsort(prominences)[-n:]]
    result[top_n] = data[top_n]
    return result


def generate_lorentzian_kernel(kernel_size: int, gamma: float) -> np.ndarray:
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    kernel = (gamma ** 2) / (x ** 2 + gamma ** 2)
    return kernel / np.sum(kernel)


def convolve_with_lorentzian(intensity, kernel) -> np.ndarray:
    return scipy.signal.convolve(np.asarray(intensity, dtype=float), kernel, mode="same")


# ── Two-band stitching (ported from final_output.process_arrays) ──────────────

def stitch_two_bands(low_band, high_band, *, band_len: int = 800,
                     keep_head: int = 700, overlap: int = 100) -> np.ndarray:
    """Combine the low and high bands the way ``final_output.py`` does.

    ``first keep_head of low`` + ``mean of the trailing/leading overlap`` +
    ``last keep_head of high``. With the defaults this yields 1500 points.
    """
    low = np.asarray(low_band, dtype=float)
    high = np.asarray(high_band, dtype=float)
    if len(low) != band_len or len(high) != band_len:
        raise ValueError(f"each band must be {band_len} points "
                         f"(got {len(low)}, {len(high)})")
    head = low[:keep_head]
    tail = high[-keep_head:]
    mean_overlap = (low[-overlap:] + high[:overlap]) / 2.0
    return np.concatenate([head, mean_overlap, tail])


def stitched_axis(start: float, stop: float, step: float) -> np.ndarray:
    """cm⁻¹ axis matching the stitched output (their ``range(501, 3500, 2)``)."""
    return np.arange(start, stop, step, dtype=float)
