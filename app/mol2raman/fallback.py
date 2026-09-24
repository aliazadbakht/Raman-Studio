# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Synthetic placeholder spectrum for when the real model can't run yet.

This is NOT a prediction. It deterministically fabricates a plausible-looking
curve from a hash of the SMILES so the "Predict from structure" UI and overlay
path are usable (and testable) before the ML stack + checkpoints are installed.
Every overlay it produces is labelled "(demo — model not installed)" so it can
never be mistaken for a real Mol2Raman result.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .predict import PredictedSpectrum


def demo_spectrum(smiles: str, *, cm_min=200.0, cm_max=3400.0, n=1600) -> PredictedSpectrum:
    seed = int(hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    cm = np.linspace(cm_min, cm_max, n)
    y = np.zeros_like(cm)
    for _ in range(rng.integers(6, 14)):
        center = rng.uniform(cm_min + 50, cm_max - 50)
        width = rng.uniform(6, 22)
        height = rng.uniform(0.2, 1.0)
        y += height * (width ** 2) / ((cm - center) ** 2 + width ** 2)
    peak = float(np.max(y)) or 1.0
    return PredictedSpectrum(
        cm=cm, intensity=y / peak, smiles=smiles, source="demo",
        label=f"Predicted (demo): {smiles}")
