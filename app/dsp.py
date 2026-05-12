"""Digital signal processing utilities for Raman spectra."""
import numpy as np
from numpy.polynomial import legendre
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter


# ── Calibration source lines (nm) ────────────────────────────────────────────

NEON_LINES = [
    585.249, 588.189, 594.483, 597.553, 603.000, 607.434,
    609.616, 614.306, 616.359, 621.728, 626.649, 630.479,
    633.443, 638.299, 640.225, 650.653, 659.895, 667.828,
    671.704, 692.947, 703.241, 717.394, 724.517, 743.890,
    747.244
]

MERCURY_ARGON_LINES = [
    404.656, 407.783, 435.833, 546.074, 576.960, 578.013,
    696.543, 706.722, 714.704, 727.294, 738.398, 750.387,
    763.511, 772.376, 794.818, 800.616, 811.531, 826.452,
    842.465, 852.144, 866.794, 912.297, 922.450
]


# ── Calibration ───────────────────────────────────────────────────────────────

def pixels_to_wavelengths(coeffs, n_pixels):
    """Apply OpenRAMAN Legendre calibration coefficients."""
    t = np.linspace(-1, 1, n_pixels)
    return legendre.legval(t, np.asarray(coeffs, dtype=float))


def power_to_legendre(coeffs):
    """Convert legacy power-basis coefficients to Legendre coefficients."""
    coeffs = list(coeffs) + [0.0] * (4 - len(coeffs))
    a0, a1, a2, a3 = coeffs[:4]
    return [
        a0 + a2 / 3.0,
        a1 + 3.0 * a3 / 5.0,
        2.0 * a2 / 3.0,
        2.0 * a3 / 5.0,
    ]


def wavelengths_to_raman(wavelengths, laser_nm):
    """Convert wavelengths (nm) to Raman shifts (cm⁻¹)."""
    return 1e7 * (1.0 / laser_nm - 1.0 / wavelengths)


def fit_calibration(pixel_positions, known_wavelengths, degree=3, n_pixels=None):
    """
    Fit OpenRAMAN-compatible Legendre calibration coefficients.

    Pixel coordinates are normalized to [-1, 1] across the full detector width.
    """
    if len(pixel_positions) < degree + 1:
        raise ValueError(f"Need at least {degree+1} points for degree-{degree} fit")
    if n_pixels is not None and n_pixels > 1:
        denom = n_pixels - 1
    else:
        denom = max(pixel_positions) if max(pixel_positions) > 0 else 1
    t = np.asarray([2 * p / denom - 1 for p in pixel_positions], dtype=float)
    coeffs = legendre.legfit(t, np.asarray(known_wavelengths, dtype=float), degree)
    return np.pad(coeffs, (0, max(0, 4 - len(coeffs))))[:4]


# ── Smoothing ─────────────────────────────────────────────────────────────────

def boxcar_smooth(y, window):
    """Simple boxcar (moving average) smoothing."""
    if window <= 1:
        return y.copy()
    k = np.ones(window) / window
    return np.convolve(y, k, mode='same')


def sgolay_smooth(y, window, order, deriv=0):
    """Savitzky-Golay smoothing / derivative."""
    if window % 2 == 0:
        window += 1
    window = max(window, order + 2)
    return savgol_filter(y, window, order, deriv=deriv)


def median_filt(y, size=3):
    return median_filter(y, size=size)


# ── Baseline ──────────────────────────────────────────────────────────────────

def baseline_schulze(y, max_iter=100):
    """
    Iterative polynomial baseline removal (Schulze method approximation):
    repeatedly fit a polynomial to the minimum-envelope of the spectrum.
    """
    b = y.copy().astype(float)
    x = np.arange(len(y))
    for _ in range(max_iter):
        coeffs = np.polyfit(x, b, 5)
        fit = np.polyval(coeffs, x)
        b = np.minimum(b, fit)
    return b


def remove_baseline(y):
    return y - baseline_schulze(y)


# ── Peak detection ────────────────────────────────────────────────────────────

def detect_peaks(y, prominence=None, distance=10, height=None, n_peaks=None):
    """
    Detect peaks in spectrum using prominence and/or height.
    prominence: minimum prominence of peaks.
    distance: minimum horizontal distance between peaks.
    height: minimum absolute height of peaks.
    n_peaks: maximum number of peaks to return (strongest first).
    """
    peaks, props = find_peaks(y, height=height, prominence=prominence, distance=distance)
    
    if n_peaks is not None and len(peaks) > n_peaks:
        # Sort by prominence if available, else height
        if 'prominences' in props:
            order = np.argsort(props['prominences'])[::-1]
        elif 'peak_heights' in props:
            order = np.argsort(props['peak_heights'])[::-1]
        else:
            order = np.arange(len(peaks))
            
        peaks = peaks[order[:n_peaks]]
        peaks = np.sort(peaks)
    return peaks


# ── Full processing pipeline ──────────────────────────────────────────────────

def process_spectrum(raw_y, blank_y=None,
                     use_blank=False, use_median=False,
                     boxcar_window=1, use_baseline=False,
                     use_sgolay=False, sg_window=11, sg_order=3, sg_deriv=0):
    """Apply full processing pipeline to raw spectrum row."""
    y = raw_y.astype(float)

    if use_blank and blank_y is not None:
        y = y - blank_y.astype(float)

    if use_median:
        y = median_filt(y)

    y = boxcar_smooth(y, boxcar_window)

    if use_baseline:
        y = remove_baseline(y)

    if use_sgolay:
        y = sgolay_smooth(y, sg_window, sg_order, sg_deriv)

    return y
