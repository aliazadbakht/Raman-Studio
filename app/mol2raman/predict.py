# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Mol2Raman inference: SMILES → predicted Raman spectrum on a cm⁻¹ axis.

This orchestrates the authors' pipeline for a single molecule:

  1. featurize the molecule into a graph (``featurize.graph_from_smiles``);
  2. for each spectral band, optionally predict its peak count with a
     ``ModelPredNumPeak`` checkpoint;
  3. build the global feature vector and run the band's spectral GNN with
     MC-dropout (multiple stochastic passes, averaged);
  4. rescale, keep the N most-prominent peaks, stitch the bands, and convolve
     with a Lorentzian (see :mod:`postprocess`).

The wiring (which checkpoints, classes, bands, feature spec) is read from the
bundle ``bundle.json`` so it tracks whatever the authors ship. torch is imported
lazily; callers should gate on :func:`app.mol2raman.bundle.availability` first.

NOTE: the torch path here cannot be exercised until the ML stack and the
authors' checkpoints are installed. The control flow faithfully follows
``predict_raman_spectra.py`` + ``final_output.py``; treat the first real run as a
validation step (shapes/feature order against the supplied config).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import bundle as bundlemod
from . import postprocess as pp
from . import featurize as feat


class PredictionError(RuntimeError):
    pass


@dataclass
class PredictedSpectrum:
    cm: np.ndarray            # cm⁻¹ axis
    intensity: np.ndarray     # normalised 0..1
    smiles: str
    source: str = "model"     # "model" | "demo"
    label: str = ""           # human-readable overlay label

    def display(self) -> str:
        return self.label or f"Predicted: {self.smiles}"


# Spectral model classes we know how to construct/run. Maps descriptor names to
# the vendored architecture classes (all share the same ctor/forward shape).
def _model_class(name: str):
    from .vendor import model as M
    try:
        return getattr(M, name)
    except AttributeError as e:
        raise PredictionError(f"unknown model class in bundle descriptor: {name!r}") from e


def _build_model(cls_name, node_dim, edge_dim, n_data_points, dim_h, add_feat_size):
    cls = _model_class(cls_name)
    return cls(node_feature_size=node_dim, edge_feature_size=edge_dim,
               dim_h=int(dim_h), n_data_points=int(n_data_points),
               additional_feature_size=int(add_feat_size))


def _load_state(model, path):
    import torch
    model.load_state_dict(torch.load(path, map_location="cpu"))


def _enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def _mc_forward(model, x, glob, edge_attr, edge_index, batch_index, mc_samples):
    import torch
    preds = []
    with torch.no_grad():
        for _ in range(max(int(mc_samples), 1)):
            preds.append(model(x, glob, edge_attr, edge_index, batch_index))
    return torch.mean(torch.stack(preds, dim=0), dim=0).squeeze().cpu().numpy()


def predict_spectrum(smiles: str, *, bundle_dir=None, mc_samples=None) -> PredictedSpectrum:
    """Run the real model bundle. Raises :class:`PredictionError` if not ready."""
    avail = bundlemod.availability(bundle_dir)
    if not avail.ready:
        raise PredictionError(avail.detail)

    import torch

    bundle_dir = avail.bundle_dir
    descriptor = bundlemod.load_descriptor(bundle_dir)
    bands = descriptor.get("bands", [])
    if not bands:
        raise PredictionError("bundle.json has no 'bands'.")
    gspec = descriptor.get("global_features", {})
    mc = mc_samples if mc_samples is not None else int(descriptor.get("mc_samples", 10))

    # Featurize once; every band reuses the same graph tensors.
    data = feat.graph_from_smiles(smiles)
    x = data.x.float()
    edge_attr = data.edge_attr.float()
    edge_index = data.edge_index
    node_dim, edge_dim = x.shape[1], edge_attr.shape[1]
    batch_index = torch.zeros(x.shape[0], dtype=torch.long)
    zero_glob = torch.zeros((1, 1), dtype=torch.float32)  # for models that ignore globals

    band_preds = []
    for band in bands:
        # 1) peak count (optional)
        num_peak = float(band.get("default_num_peak", 0))
        npk_ckpt = band.get("numpeak_checkpoint")
        if npk_ckpt:
            npk_model = _build_model(
                band.get("numpeak_class", "ModelPredNumPeak"), node_dim, edge_dim,
                n_data_points=1, dim_h=band.get("numpeak_dim_h", 256), add_feat_size=12)
            _load_state(npk_model, os.path.join(bundle_dir, npk_ckpt))
            npk_model.eval(); _enable_dropout(npk_model)
            num_peak = float(np.ravel(_mc_forward(
                npk_model, x, zero_glob, edge_attr, edge_index, batch_index, mc))[0])

        # 2) global features for the spectral model
        cls_name = band.get("spectra_class", "GINEGLOBAL")
        add_feat_size = int(band.get("additional_feature_size", 4097))
        if cls_name == "GINEGLOBAL":
            gvec = feat.global_features(smiles, num_peak, gspec)
            glob = torch.from_numpy(gvec).float().reshape(1, -1)
        else:
            glob = zero_glob

        # 3) spectral model with MC dropout
        n_pts = int(band.get("n_data_points", 800))
        spec_model = _build_model(cls_name, node_dim, edge_dim, n_pts,
                                  band.get("dim_h", 256), add_feat_size)
        _load_state(spec_model, os.path.join(bundle_dir, band["spectra_checkpoint"]))
        spec_model.eval(); _enable_dropout(spec_model)
        raw = np.ravel(_mc_forward(spec_model, x, glob, edge_attr, edge_index, batch_index, mc))

        # 4) per-band post-processing
        raw = pp.rescale(n_pts, raw)
        if npk_ckpt:
            raw = pp.keep_peaks_prom(raw, round(num_peak))
        band_preds.append(raw)

    # 5) stitch + convolve
    stitch = descriptor.get("stitch", {})
    lor = descriptor.get("lorentzian", {})
    if len(band_preds) == 2:
        spectrum = pp.stitch_two_bands(
            band_preds[0], band_preds[1],
            band_len=int(stitch.get("band_len", 800)),
            keep_head=int(stitch.get("keep_head", 700)),
            overlap=int(stitch.get("overlap", 100)))
        cm = pp.stitched_axis(stitch.get("axis_start", 501),
                              stitch.get("axis_stop", 3500),
                              stitch.get("axis_step", 2))
    else:
        spectrum = band_preds[0]
        b0 = bands[0]
        cm = np.linspace(float(b0.get("cm_min", 100)), float(b0.get("cm_max", 3500)),
                         len(spectrum))

    kernel = pp.generate_lorentzian_kernel(int(lor.get("kernel_size", 600)),
                                           float(lor.get("gamma", 2.5)))
    spectrum = pp.convolve_with_lorentzian(spectrum, kernel)

    # Align axis/spectrum lengths defensively (stitch vs axis off-by-one).
    n = min(len(cm), len(spectrum))
    cm, spectrum = cm[:n], spectrum[:n]
    peak = float(np.max(spectrum)) if spectrum.size else 0.0
    if peak > 0:
        spectrum = spectrum / peak
    return PredictedSpectrum(cm=cm, intensity=spectrum, smiles=smiles, source="model")
