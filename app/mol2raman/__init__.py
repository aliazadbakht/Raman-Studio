# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Mol2Raman integration: predict a Raman spectrum from a molecular structure.

Wraps the Mol2Raman graph-neural-network model (https://github.com/salvasorrentino/Mol2Raman,
MIT) so the analyzer can turn a SMILES / compound name / formula into a predicted
Raman spectrum and overlay it on the plot.

Public API
----------
``availability()``            – is the real model ready, or what's missing?
``resolve_to_smiles(text)``   – SMILES / name / formula → SMILES.
``predict(text, allow_demo)`` – resolve + predict, returning a PredictedSpectrum.

The heavy ML stack (torch / torch_geometric / rdkit / deepchem) and the trained
checkpoints are NOT required to import this package — only to run a real
prediction. Until they're present, ``predict(..., allow_demo=True)`` returns a
clearly-labelled synthetic curve so the workflow is usable end to end.
"""
from __future__ import annotations

from .bundle import availability, Availability, default_bundle_dir
from .predict import PredictedSpectrum, PredictionError, predict_spectrum
from .resolve import resolve_to_smiles, ResolveResult, ResolveError

__all__ = [
    "availability", "Availability", "default_bundle_dir",
    "PredictedSpectrum", "PredictionError", "predict_spectrum",
    "resolve_to_smiles", "ResolveResult", "ResolveError",
    "predict",
]


def predict(text: str, *, allow_demo: bool = True, mc_samples=None) -> PredictedSpectrum:
    """Resolve ``text`` to a structure and predict its Raman spectrum.

    Uses the real model bundle when available; otherwise, if ``allow_demo`` is
    set, returns a labelled synthetic spectrum (``source == "demo"``) so the UI
    still works. Raises :class:`PredictionError` / :class:`ResolveError`.
    """
    result = resolve_to_smiles(text)
    smiles = result.smiles
    avail = availability()
    if avail.ready:
        spec = predict_spectrum(smiles, mc_samples=mc_samples)
    elif allow_demo:
        from .fallback import demo_spectrum
        spec = demo_spectrum(smiles)
    else:
        raise PredictionError(avail.detail)

    # Carry resolution context onto the result for the overlay label / warnings.
    if result.source != "smiles":
        base = spec.label or (f"Predicted: {smiles}" if spec.source == "model"
                              else f"Predicted (demo): {smiles}")
        spec.label = f"{base}  ⟵ {result.name}"
    spec.resolve = result  # type: ignore[attr-defined]
    return spec
