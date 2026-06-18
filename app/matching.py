# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V.
"""
Spectral matching: resample a query spectrum onto a common cm⁻¹ grid,
baseline-correct and L2-normalize, then score against a library matrix
using cosine similarity (Hit Quality Index).

Scores are returned in [0, 100]. They are *similarity* scores, not
probabilities — two unrelated spectra with similar broad shapes can still
score 70–80%. Treat values >85 as strong matches, 70–85 as suggestive,
and <70 as weak.
"""
from __future__ import annotations
import numpy as np
from . import dsp


DEFAULT_GRID_MIN = 100.0
DEFAULT_GRID_MAX = 3500.0
DEFAULT_GRID_STEP = 2.0


def default_grid():
    return np.arange(DEFAULT_GRID_MIN, DEFAULT_GRID_MAX + DEFAULT_GRID_STEP, DEFAULT_GRID_STEP)


def resample_to_grid(x, y, grid):
    """Linearly interpolate (x, y) onto grid. Out-of-range samples become 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    out = np.interp(grid, x, y, left=0.0, right=0.0)
    return out


def prepare_for_matching(y_on_grid, do_baseline=True):
    """Baseline-remove (if requested), clip negatives, and L2-normalize.

    NOTE: prefer :func:`prepare_query`, which baseline-corrects at the
    spectrum's native resolution *before* resampling. Running the baseline on
    an already-resampled vector is unsafe when the spectrum doesn't span the
    whole grid: ``resample_to_grid`` zero-pads outside the data range, and the
    polynomial baseline cannot follow the resulting step at the data boundary,
    leaving a large artifact peak that dominates the normalized vector.
    """
    y = np.asarray(y_on_grid, dtype=float)
    if do_baseline:
        y = y - dsp.baseline_schulze(y, max_iter=30)
    y = np.clip(y, 0.0, None)
    n = float(np.linalg.norm(y))
    if n > 0:
        y = y / n
    return y


def prepare_query(x, y, grid, do_baseline=True):
    """Baseline-correct (native resolution), resample to ``grid``, clip, L2-normalize.

    This is the matching front-end for both queries and library references.
    The baseline is removed from the *native* spectrum before resampling so
    the zero-padding outside the data range can't create a step the polynomial
    baseline fails to follow — otherwise a spurious peak survives at the data
    boundary and, after L2-normalization, dominates the cosine score (making
    every partial-range spectrum match the same handful of references).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if do_baseline:
        y = y - dsp.baseline_schulze(y, max_iter=30)
    y_grid = np.interp(grid, x, y, left=0.0, right=0.0)
    y_grid = np.clip(y_grid, 0.0, None)
    n = float(np.linalg.norm(y_grid))
    if n > 0:
        y_grid = y_grid / n
    return y_grid


def cosine_similarity(a, b):
    """Cosine similarity in [0, 1] (assumes nonnegative inputs)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def hqi(a, b):
    """Hit Quality Index in [0, 100]."""
    return 100.0 * cosine_similarity(a, b)


def score_against_matrix(query_norm, library_matrix):
    """
    Vectorized scoring of a normalized query against a library matrix.

    query_norm: shape (G,), already prepared (baseline-removed, L2-normalized).
    library_matrix: shape (N, G), each row L2-normalized.
    Returns scores in [0, 100], shape (N,).
    """
    scores = library_matrix @ query_norm
    return 100.0 * np.clip(scores, 0.0, 1.0)


def top_matches(scores, k=10):
    """Return indices of the top-k scores in descending order."""
    if k >= len(scores):
        order = np.argsort(scores)[::-1]
    else:
        partial = np.argpartition(scores, -k)[-k:]
        order = partial[np.argsort(scores[partial])[::-1]]
    return order


# ── Refinement re-rank (shift-tolerant, fingerprint-weighted) ──────────────────
# Whole-spectrum cosine is dominated by the broad C–H stretch (~2800–3000 cm⁻¹),
# which most organics share, so near-twin compounds (e.g. acetone vs. butanone)
# end up tied. A small calibration error also shifts the query's peaks off the
# references and penalizes the (sharp-banded) correct hit more than a broad wrong
# one. The refinement re-ranks the top cosine candidates with a cosine that
#   (a) takes the best score over a small ± shift, absorbing calibration error, and
#   (b) emphasizes the fingerprint region, where compounds actually differ.
# It is a re-rank only: stage-1 recall (which candidates surface) is unchanged.

FINGERPRINT_MIN = 250.0
FINGERPRINT_MAX = 1800.0
FINGERPRINT_OUTSIDE_WEIGHT = 0.35
REFINE_SHIFT_CM = 24.0


def fingerprint_weights(grid):
    """1.0 inside the fingerprint region, down-weighted outside (e.g. C–H stretch)."""
    grid = np.asarray(grid, dtype=float)
    return np.where((grid >= FINGERPRINT_MIN) & (grid <= FINGERPRINT_MAX),
                    1.0, FINGERPRINT_OUTSIDE_WEIGHT)


def refine_scores(query_x, query_y, grid, library_matrix, candidates, do_baseline=True):
    """Re-score ``candidates`` (row indices into ``library_matrix``) with a
    shift-tolerant, fingerprint-weighted cosine. Returns scores in [0, 100], one
    per candidate, in the same order as ``candidates``.

    The baseline is removed once at native resolution; each trial shift only
    re-interpolates onto the grid, so the loop stays cheap.
    """
    grid = np.asarray(grid, dtype=float)
    x = np.asarray(query_x, dtype=float)
    y = np.asarray(query_y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if do_baseline:
        y = y - dsp.baseline_schulze(y, max_iter=30)

    w = fingerprint_weights(grid)
    step = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    n_shift = int(round(REFINE_SHIFT_CM / step)) if step else 0

    q_rows = []
    for s in range(-n_shift, n_shift + 1):
        yg = np.interp(grid, x + s * step, y, left=0.0, right=0.0)
        yg = np.clip(yg, 0.0, None) * w
        nrm = np.linalg.norm(yg)
        if nrm > 0:
            yg = yg / nrm
        q_rows.append(yg)
    Q = np.vstack(q_rows)

    refs = np.asarray(library_matrix)[np.asarray(candidates, dtype=int)].astype(float) * w
    rn = np.linalg.norm(refs, axis=1, keepdims=True)
    rn[rn == 0] = 1.0
    refs = refs / rn

    sims = Q @ refs.T                       # (n_shift, n_candidates)
    best = sims.max(axis=0)
    return 100.0 * np.clip(best, 0.0, 1.0)
