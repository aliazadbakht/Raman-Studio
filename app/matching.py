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
    """Baseline-remove (if requested), clip negatives, and L2-normalize."""
    y = np.asarray(y_on_grid, dtype=float)
    if do_baseline:
        y = y - dsp.baseline_schulze(y, max_iter=30)
    y = np.clip(y, 0.0, None)
    n = float(np.linalg.norm(y))
    if n > 0:
        y = y / n
    return y


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
