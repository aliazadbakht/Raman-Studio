# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Locating and describing a Mol2Raman model bundle.

A "bundle" is a directory (default ``<app>/models/mol2raman``) containing the
trained checkpoints supplied by the Mol2Raman authors plus a ``bundle.json``
descriptor that tells the adapter how to wire them up: which model class each
checkpoint is, the cm⁻¹ band each spectral model covers, how to build the
global feature vector, and the stitch/convolution parameters.

Nothing here imports torch — it only inspects the filesystem and reports what is
present, so the UI can give precise guidance ("install deps" vs "drop in the
checkpoint") before anything heavy is loaded.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

# Third-party packages required for real inference. Checked by name only.
_REQUIRED_MODULES = ("torch", "torch_geometric", "rdkit", "deepchem", "scipy")

_DESCRIPTOR_NAME = "bundle.json"


def app_root() -> str:
    # app/mol2raman/bundle.py → app/mol2raman → app → <root>
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_bundle_dir() -> str:
    override = os.environ.get("MOL2RAMAN_BUNDLE_DIR")
    if override:
        return override
    return os.path.join(app_root(), "models", "mol2raman")


@dataclass
class Availability:
    """Why prediction can or cannot run right now."""
    state: str                       # "ready" | "deps_missing" | "weights_missing"
    detail: str = ""
    missing_modules: list = field(default_factory=list)
    missing_files: list = field(default_factory=list)
    bundle_dir: str = ""

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def can_demo(self) -> bool:
        # The synthetic fallback only needs numpy/scipy, which the app already has.
        return True


def missing_modules() -> list:
    miss = []
    import importlib.util
    for name in _REQUIRED_MODULES:
        if importlib.util.find_spec(name) is None:
            miss.append(name)
    return miss


def descriptor_path(bundle_dir: Optional[str] = None) -> str:
    return os.path.join(bundle_dir or default_bundle_dir(), _DESCRIPTOR_NAME)


def load_descriptor(bundle_dir: Optional[str] = None) -> Optional[dict]:
    path = descriptor_path(bundle_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_checkpoints(descriptor: dict):
    for band in descriptor.get("bands", []):
        for key in ("spectra_checkpoint", "numpeak_checkpoint"):
            ckpt = band.get(key)
            if ckpt:
                yield ckpt


def availability(bundle_dir: Optional[str] = None) -> Availability:
    bundle_dir = bundle_dir or default_bundle_dir()
    miss = missing_modules()
    if miss:
        return Availability(
            state="deps_missing",
            detail="Prediction needs the Mol2Raman ML stack: " + ", ".join(miss),
            missing_modules=miss, bundle_dir=bundle_dir)

    descriptor = load_descriptor(bundle_dir)
    if descriptor is None:
        return Availability(
            state="weights_missing",
            detail=f"No bundle.json in {bundle_dir}. Add the authors' checkpoints "
                   f"and a descriptor (see README).",
            bundle_dir=bundle_dir)

    missing_files = [ckpt for ckpt in _iter_checkpoints(descriptor)
                     if not os.path.exists(os.path.join(bundle_dir, ckpt))]
    if missing_files:
        return Availability(
            state="weights_missing",
            detail="Bundle descriptor references checkpoints that are not present.",
            missing_files=missing_files, bundle_dir=bundle_dir)

    return Availability(state="ready", detail=f"Model bundle ready in {bundle_dir}.",
                        bundle_dir=bundle_dir)
