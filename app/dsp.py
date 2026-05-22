"""Digital signal processing utilities for Raman spectra."""
import numpy as np
from numpy.polynomial import legendre
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter
from scipy.optimize import differential_evolution, minimize, linear_sum_assignment


# ── Calibration source lines (nm) ────────────────────────────────────────────

NEON_LINES = [
    585.249, 588.189, 594.483, 597.553, 603.000, 607.434,
    609.616, 614.306, 616.359, 621.728, 626.649, 630.479,
    633.443, 638.299, 640.225, 650.653, 653.288, 659.895,
    667.828, 671.704, 692.947, 703.241, 717.394, 724.517,
    743.890
]

MERCURY_ARGON_LINES = [
    253.65, 296.73, 302.15, 313.16, 334.15, 365.01,
    404.66, 435.84, 546.08, 576.96, 579.07, 696.54,
    738.40, 750.39, 763.51, 772.40, 794.82, 800.62,
    811.53, 826.45, 842.46, 912.30
]


# ── Known Raman shifts for common reference samples ──────────────────────────
#
# Used by the "Calibrate from sample" workflow: the user puts a pure standard
# in the cuvette, picks it from a list, and the detected peaks are matched to
# these reference positions (cm⁻¹).
#
# Lists are sorted by Raman shift and only contain the most prominent peaks
# that are reliably resolvable on a low-resolution DIY rig. ASTM E1840 values
# are used where available (cyclohexane, polystyrene).

RAMAN_STANDARDS = {
    "Isopropanol (IPA)": [819.0, 1132.0, 1452.0, 2920.0],
    "Ethanol":            [884.0, 1053.0, 1454.0, 2929.0],
    "Cyclohexane (ASTM)": [801.3, 1028.3, 1157.6, 1266.4, 1444.4, 2852.9, 2923.8],
    "Polystyrene (ASTM)": [620.9, 1001.4, 1031.8, 1602.3, 3054.3],
    "Acetonitrile":       [379.0, 919.0, 2249.0, 2942.0],
    "Toluene":            [521.0, 786.0, 1004.0, 1031.0, 1212.0, 1604.0, 3056.0],
    "Methanol":           [1035.0, 1454.0, 2837.0, 2944.0],
}


def raman_shift_to_wavelength(shift_cm, laser_nm):
    """Convert Raman shift (cm⁻¹) → scattered wavelength (nm) for a laser."""
    shift = np.asarray(shift_cm, dtype=float)
    return 1.0 / (1.0 / float(laser_nm) - shift / 1.0e7)


# ── Calibration ───────────────────────────────────────────────────────────────

def pixels_to_wavelengths(coeffs, n_pixels):
    """Apply OpenRAMAN Legendre calibration coefficients."""
    t = np.linspace(-1, 1, n_pixels)
    return legendre.legval(t, np.asarray(coeffs, dtype=float))


def normalize_pixels(pixel_positions, n_pixels=None):
    """Normalize detector pixel positions to the OpenRAMAN [-1, 1] index."""
    pix = np.asarray(pixel_positions, dtype=float)
    if n_pixels is not None and n_pixels > 1:
        denom = n_pixels - 1
    elif pix.size and np.nanmin(pix) >= -1.0 and np.nanmax(pix) <= 1.0:
        return pix
    elif pix.size:
        denom = np.nanmax(pix) if np.nanmax(pix) > 0 else 1.0
    else:
        denom = 1.0
    return 2.0 * pix / denom - 1.0


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
    t = normalize_pixels(pixel_positions, n_pixels=n_pixels)
    coeffs = legendre.legfit(t, np.asarray(known_wavelengths, dtype=float), degree)
    return np.pad(coeffs, (0, max(0, 4 - len(coeffs))))[:4]


def calibration_pair_rms(coeffs, pixel_positions, known_wavelengths, n_pixels=None):
    """RMS residual for explicitly matched pixel/wavelength pairs."""
    t = normalize_pixels(pixel_positions, n_pixels=n_pixels)
    pred = legendre.legval(t, np.asarray(coeffs, dtype=float))
    known = np.asarray(known_wavelengths, dtype=float)
    return float(np.sqrt(np.mean((pred - known) ** 2))) if known.size else 0.0


def calibration_source_rms(coeffs, peak_positions, source_wavelengths, n_pixels=None):
    """RMS residual against a one-to-one source-line assignment."""
    t = normalize_pixels(peak_positions, n_pixels=n_pixels)
    projected = legendre.legval(t, np.asarray(coeffs, dtype=float))
    source = np.asarray(source_wavelengths, dtype=float)
    if not projected.size or not source.size:
        return 0.0
    _, residuals = assign_source_lines(projected, source)
    return float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0


def closest_source_lines(coeffs, peak_positions, source_wavelengths, n_pixels=None):
    """Return projected wavelengths and one-to-one source-line matches."""
    t = normalize_pixels(peak_positions, n_pixels=n_pixels)
    projected = legendre.legval(t, np.asarray(coeffs, dtype=float))
    source = np.asarray(source_wavelengths, dtype=float)
    if not source.size:
        return projected, np.full(projected.shape, np.nan)
    assigned, _ = assign_source_lines(projected, source)
    return projected, assigned


def assign_source_lines(projected_wavelengths, source_wavelengths):
    """
    Assign projected peak wavelengths to source lines one-to-one.

    Nearest-line matching can map several detected peaks onto the same lamp
    line, which makes ordinary sample spectra look deceptively calibratable.
    A calibration lamp has at most one detector peak for each source line, so
    automatic fitting should use unique line assignments.
    """
    projected = np.asarray(projected_wavelengths, dtype=float)
    source = np.asarray(source_wavelengths, dtype=float)
    if projected.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if source.size == 0:
        return np.full(projected.shape, np.nan), np.full(projected.shape, np.inf)
    if projected.size > source.size:
        raise ValueError(
            "Detected more calibration peaks than known source lines. "
            "Reduce Max Peaks or lower the peak sensitivity."
        )

    costs = np.abs(projected[:, None] - source[None, :])
    row_idx, col_idx = linear_sum_assignment(costs)
    assigned = np.full(projected.shape, np.nan, dtype=float)
    residuals = np.full(projected.shape, np.inf, dtype=float)
    assigned[row_idx] = source[col_idx]
    residuals[row_idx] = projected[row_idx] - source[col_idx]
    return assigned, residuals


def match_peaks_to_standard(detected_pixels, current_coeffs, expected_shifts,
                            laser_nm, n_pixels=None, search_window_cm=400.0):
    """
    Pair detected sample peaks with the closest expected Raman shifts.

    The user's *current* (possibly bad) calibration is used to project each
    detected pixel to an approximate cm⁻¹. We then find a global cm⁻¹ offset
    that aligns the detected peak pattern with the expected one, and assign
    pairs by Hungarian matching with that offset applied.

    Returns: list of (pixel, expected_shift_cm) tuples, ordered by pixel.
    Robust to extra detected peaks or missing expected peaks (no pair is
    forced if the residual exceeds the per-pair tolerance).
    """
    detected = np.asarray(detected_pixels, dtype=float)
    expected = np.asarray(sorted(expected_shifts), dtype=float)
    if detected.size == 0 or expected.size == 0:
        return []

    t = normalize_pixels(detected, n_pixels=n_pixels)
    wl = legendre.legval(t, np.asarray(current_coeffs, dtype=float))
    current_cm = wavelengths_to_raman(wl, float(laser_nm))

    # Coarse 1-D search for the best global shift between detected and expected.
    shifts = np.linspace(-search_window_cm, search_window_cm, 161)
    best_shift = 0.0
    best_cost = np.inf
    for s in shifts:
        d = current_cm + s
        # Sum of distances from each detected peak to its nearest expected peak.
        cost = float(np.sum(np.min(np.abs(d[:, None] - expected[None, :]), axis=1)))
        if cost < best_cost:
            best_cost = cost
            best_shift = float(s)

    shifted = current_cm + best_shift

    # One-to-one Hungarian assignment. Costs above tolerance get dropped.
    n_d, n_e = len(shifted), len(expected)
    if n_d <= n_e:
        cost_mat = np.abs(shifted[:, None] - expected[None, :])
        rows, cols = linear_sum_assignment(cost_mat)
        pairs_idx = [(int(r), int(c)) for r, c in zip(rows, cols)]
    else:
        cost_mat = np.abs(shifted[:, None] - expected[None, :]).T  # transpose
        rows, cols = linear_sum_assignment(cost_mat)
        pairs_idx = [(int(c), int(r)) for r, c in zip(rows, cols)]

    tol = max(80.0, search_window_cm * 0.35)
    pairs = []
    for di, ei in pairs_idx:
        if abs(shifted[di] - expected[ei]) <= tol:
            pairs.append((float(detected[di]), float(expected[ei])))
    pairs.sort(key=lambda p: p[0])
    return pairs


def diagnose_calibration(y, current_coeffs, laser_nm, n_pixels=None,
                         max_offset_cm=600.0):
    """
    Diagnose whether the current calibration plausibly matches a known
    standard. Detects the strongest peaks in ``y``, then for each entry in
    ``RAMAN_STANDARDS`` tries to align it via a single global cm⁻¹ offset.

    Returns a dict describing the best candidate, or ``None`` if no
    standard plausibly matches.

    Keys:
        name           — standard's display name
        offset_cm      — global cm⁻¹ shift required to match (≈0 means OK)
        n_matched      — number of expected peaks that found a partner
        n_expected     — total peaks the standard has
        residual_cm    — RMS residual *after* applying the offset
        verdict        — "ok" if offset and residual are small, else "off"
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.size == 0 or current_coeffs is None:
        return None
    y_clean = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
    y_clean = y_clean - np.min(y_clean)
    prom = max(float(np.max(y_clean)) * 0.03, 1.0)
    peaks, props = find_peaks(y_clean, prominence=prom, distance=8)
    if len(peaks) < 3:
        return None
    order = np.argsort(props["prominences"])[::-1]
    keep = min(len(peaks), 10)
    strong = peaks[order[:keep]]
    refined = refine_peak_positions(y_clean, strong)
    t = normalize_pixels(refined, n_pixels=n_pixels)
    wl = legendre.legval(t, np.asarray(current_coeffs, dtype=float))
    detected_cm = wavelengths_to_raman(wl, float(laser_nm))

    best = None
    shifts = np.linspace(-max_offset_cm, max_offset_cm, 121)
    for name, expected in RAMAN_STANDARDS.items():
        exp = np.asarray(expected, dtype=float)
        # Coarse offset search to align patterns.
        costs = np.array([
            np.sum(np.min(np.abs((detected_cm + s)[:, None] - exp[None, :]),
                          axis=1))
            for s in shifts
        ])
        i = int(np.argmin(costs))
        s = float(shifts[i])
        shifted = detected_cm + s
        deltas = np.min(np.abs(shifted[:, None] - exp[None, :]), axis=1)
        # A peak counts as "matched" if it lands within 20 cm⁻¹ after shift.
        matched_mask = deltas < 20.0
        n_matched = int(np.sum(matched_mask))
        if n_matched < 3:
            continue
        residual = float(np.sqrt(np.mean(deltas[matched_mask] ** 2)))
        # Score: prefer more matches, smaller residual, smaller offset.
        score = (n_matched
                 - residual / 50.0
                 - abs(s) / 200.0)
        if best is None or score > best["score"]:
            best = {
                "name": name,
                "offset_cm": s,
                "n_matched": n_matched,
                "n_expected": len(exp),
                "residual_cm": residual,
                "score": score,
            }
    if best is None:
        return None
    if abs(best["offset_cm"]) < 25.0 and best["residual_cm"] < 15.0:
        best["verdict"] = "ok"
    else:
        best["verdict"] = "off"
    best.pop("score", None)
    return best


def fit_calibration_to_source(peak_positions, source_wavelengths, model="Cubic",
                              n_pixels=None, range_min=500.0, range_max=800.0,
                              span_min=100.0, span_max=150.0,
                              distortion_min=0.0, distortion_max=10.0,
                              sampling=10, random_seed=12345,
                              endpoint_min=None, endpoint_max=None,
                              endpoint_weight=10.0):
    """
    Fit OpenRAMAN Legendre coefficients by matching detected peaks to source lines.

    Detected peak indices are projected into wavelength space and the optimizer
    minimizes the distance to a unique line from the selected calibration lamp.
    The user does not have to assign each peak manually.
    """
    peaks = np.asarray(peak_positions, dtype=float)
    source = np.asarray(source_wavelengths, dtype=float)
    if peaks.size == 0:
        raise ValueError("No calibration peaks were detected.")
    if source.size == 0:
        raise ValueError("No calibration source lines are available.")

    degree = 1 if str(model).lower().startswith("lin") else 3
    min_peaks = 2 if degree == 1 else 3
    if peaks.size < min_peaks:
        raise ValueError(f"Need at least {min_peaks} detected peaks for {model} calibration.")
    if peaks.size > source.size:
        raise ValueError(
            "Detected more calibration peaks than known source lines. "
            "Reduce Max Peaks or lower the peak sensitivity."
        )

    range_min = float(range_min)
    range_max = float(range_max)
    span_min = float(span_min)
    span_max = float(span_max)
    distortion_min = float(distortion_min)
    distortion_max = float(distortion_max)
    sampling = int(max(3, sampling))
    endpoint_weight = float(max(0.0, endpoint_weight))
    endpoint_min = None if endpoint_min is None else float(endpoint_min)
    endpoint_max = None if endpoint_max is None else float(endpoint_max)

    if range_min >= range_max:
        raise ValueError("Minimum wavelength range must be smaller than maximum range.")
    if span_min <= 0 or span_min > span_max:
        raise ValueError("Span limits must be positive and ordered.")
    if distortion_min < 0 or distortion_min > distortion_max:
        raise ValueError("Distortion limits must be non-negative and ordered.")
    if endpoint_min is not None and endpoint_max is not None and endpoint_min >= endpoint_max:
        raise ValueError("Endpoint wavelength limits must be ordered.")

    t = normalize_pixels(peaks, n_pixels=n_pixels)

    if degree == 1:
        bounds = [(range_min, range_max), (0.5 * span_min, 0.5 * span_max)]
    else:
        bounds = [
            (range_min, range_max),
            (0.5 * span_min, 0.5 * span_max),
            (-distortion_max, distortion_max),
            (-distortion_max, distortion_max),
        ]

    def as4(coeffs):
        return np.pad(np.asarray(coeffs, dtype=float), (0, max(0, 4 - len(coeffs))))[:4]

    def violation(coeffs):
        coeffs4 = as4(coeffs)
        c0, c1, c2, c3 = coeffs4
        errors = [
            max(0.0, range_min - (c0 - c1)),
            max(0.0, (c0 + c1) - range_max),
            max(0.0, 0.5 * span_min - c1),
            max(0.0, c1 - 0.5 * span_max),
        ]
        if degree == 3:
            distortion = abs(c2) + abs(c3)
            errors.extend([
                max(0.0, distortion_min - distortion),
                max(0.0, distortion - distortion_max),
            ])
        return float(sum(v * v for v in errors))

    def cost(coeffs):
        coeffs4 = as4(coeffs)
        projected = legendre.legval(t, coeffs4)
        _, residuals = assign_source_lines(projected, source)
        total = float(np.sum(np.abs(residuals)))
        if endpoint_min is not None:
            total += endpoint_weight * abs(legendre.legval(-1.0, coeffs4) - endpoint_min)
        if endpoint_max is not None:
            total += endpoint_weight * abs(legendre.legval(1.0, coeffs4) - endpoint_max)
        return total

    def objective(coeffs):
        penalty = violation(coeffs)
        if penalty:
            return 1.0e9 + 1.0e6 * penalty + cost(coeffs)
        return cost(coeffs)

    # Differential evolution gives a deterministic global search in the same
    # bounded coefficient space as the original random-start optimizer.
    result = differential_evolution(
        objective,
        bounds,
        seed=random_seed,
        popsize=max(8, min(20, sampling)),
        maxiter=max(30, sampling * 8),
        tol=1.0e-7,
        polish=False,
        updating="immediate",
        workers=1,
    )
    local = minimize(
        objective,
        result.x,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1.0e-7, "fatol": 1.0e-7},
    )

    best = local.x if local.fun <= result.fun else result.x
    if violation(best):
        raise ValueError("No calibration solution satisfied the wavelength/span constraints.")

    coeffs = as4(best)
    projected, nearest = closest_source_lines(coeffs, peaks, source, n_pixels=n_pixels)
    residuals = projected - nearest
    rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
    finite_nearest = nearest[np.isfinite(nearest)]
    unique_lines = len(np.unique(finite_nearest)) if finite_nearest.size else 0
    line_span = float(np.max(finite_nearest) - np.min(finite_nearest)) if finite_nearest.size else 0.0
    left_endpoint = float(legendre.legval(-1.0, coeffs))
    right_endpoint = float(legendre.legval(1.0, coeffs))
    return coeffs, {
        "rms": rms,
        "cost": cost(coeffs),
        "projected": projected,
        "nearest": nearest,
        "residuals": residuals,
        "max_error": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        "unique_lines": unique_lines,
        "line_span": line_span,
        "endpoint_min": endpoint_min,
        "endpoint_max": endpoint_max,
        "left_endpoint": left_endpoint,
        "right_endpoint": right_endpoint,
        "degree": degree,
    }


def refine_peak_positions(y, peaks):
    """Refine integer peak positions with a three-point quadratic fit."""
    y = np.asarray(y, dtype=float)
    refined = []
    for peak in np.asarray(peaks, dtype=int):
        if y.size < 3:
            refined.append(float(peak))
            continue
        if peak <= 0:
            pos = np.array([0, 1, 2], dtype=int)
        elif peak >= y.size - 1:
            pos = np.array([y.size - 3, y.size - 2, y.size - 1], dtype=int)
        else:
            pos = np.array([peak - 1, peak, peak + 1], dtype=int)

        x1, x2, x3 = pos.astype(float)
        y1, y2, y3 = y[pos]
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if abs(denom) < 1.0e-12:
            refined.append(float(peak))
            continue
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x1 * x1 * (y2 - y3) + x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1)) / denom
        if abs(a) < 1.0e-12:
            refined.append(float(peak))
            continue
        vertex = -b / (2.0 * a)
        if np.isfinite(vertex):
            refined.append(float(vertex))
        else:
            refined.append(float(peak))
    return np.asarray(refined, dtype=float)


def detect_calibration_peaks(y, n_peaks, sensitivity=0.8, min_relative_height=0.02):
    """
    Detect calibration lamp peaks using the original SpectrumAnalyzer strategy.

    Peaks are selected from strongest to weakest. After each peak is selected,
    the surrounding peak footprint is masked until the signal falls below the
    sensitivity threshold and starts rising again.
    """
    values = np.asarray(y, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(np.min(values))
    if min_value < 0.0:
        values = values - min_value

    n_peaks = int(max(0, n_peaks))
    sensitivity = float(np.clip(sensitivity, 0.01, 1.0))
    min_relative_height = float(max(0.0, min_relative_height))
    mask = np.ones(values.size, dtype=bool)
    peaks = []
    first_peak_value = None

    for _ in range(n_peaks):
        candidates = np.flatnonzero(mask)
        if not candidates.size:
            break

        local_idx = int(candidates[np.argmax(values[candidates])])
        peak_value = float(values[local_idx])
        if peak_value <= 0.0:
            break
        if first_peak_value is None:
            first_peak_value = peak_value
        elif min_relative_height and peak_value < first_peak_value * min_relative_height:
            break

        peaks.append(local_idx)

        reached = False
        min_seen = 0.0
        for i in range(local_idx, values.size):
            if not reached and values[i] < sensitivity * peak_value:
                reached = True
                min_seen = float(values[i])
            if reached:
                min_seen = min(min_seen, float(values[i]))
            if reached and values[i] * sensitivity > min_seen:
                break
            mask[i] = False

        reached = False
        min_seen = 0.0
        for i in range(local_idx, -1, -1):
            if not reached and values[i] < sensitivity * peak_value:
                reached = True
                min_seen = float(values[i])
            if reached:
                min_seen = min(min_seen, float(values[i]))
            if reached and values[i] * sensitivity > min_seen:
                break
            mask[i] = False

    refined = refine_peak_positions(values, peaks)
    return np.sort(refined)


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
