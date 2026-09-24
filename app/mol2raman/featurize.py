# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Functional port of the featurization in Mol2Raman, Copyright (c) 2025
# salvasorrentino, MIT (https://github.com/salvasorrentino/Mol2Raman).
"""Turn a single SMILES into the graph + global features the GNN expects.

This mirrors ``Mol2Raman/Scripts/dataset.py::MoleculeDataset.process`` for one
molecule (no on-disk dataset, no pandas): same deepchem featurizer flags, same
sanitisation, and the same global-feature construction
(``num_peak`` + Morgan + RDKit fingerprints) used by the GINEGLOBAL spectral
models. All heavy imports (rdkit / deepchem) happen inside the functions so the
rest of the app stays importable without them.
"""
from __future__ import annotations

import numpy as np


class FeaturizeError(RuntimeError):
    pass


def molecule_from_smiles(smiles: str):
    """Sanitise a SMILES the same way the training dataset did."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise FeaturizeError(f"RDKit could not parse SMILES: {smiles!r}")
    flag = Chem.SanitizeMol(mol, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def graph_from_smiles(smiles: str):
    """Return a PyG ``Data`` graph (x, edge_index, edge_attr) for one molecule."""
    import deepchem as dc
    mol = molecule_from_smiles(smiles)
    featurizer = dc.feat.MolGraphConvFeaturizer(
        use_edges=True, use_chirality=True, use_partial_charge=True)
    feats = featurizer._featurize(mol)
    data = feats.to_pyg_graph()
    if data.edge_index is None or data.edge_index.numel() == 0:
        raise FeaturizeError(
            f"{smiles!r} produced no bonds — single-atom species are not supported.")
    return data


def _fingerprint(smiles: str, *, kind: str, radius: int, n_bits: int) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise FeaturizeError(f"RDKit could not parse SMILES for fingerprint: {smiles!r}")
    if kind == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    elif kind == "rdkit":
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    else:
        raise FeaturizeError(f"unknown fingerprint kind: {kind!r}")
    return np.asarray(gen.GetFingerprint(mol), dtype=np.float64)


def global_features(smiles: str, num_peak: float, spec: dict) -> np.ndarray:
    """Build the graph-level feature vector in the order the model was trained on.

    ``spec`` comes from the bundle descriptor, e.g.::

        {"order": ["num_peak", "morgan", "rdkit"],
         "morgan_radius": 3, "morgan_bits": 2048, "rdkit_bits": 2048}

    The training pipeline placed ``num_peak`` first (the predict script reads it
    back as ``graph_level_feats[:, 0]``), followed by the fingerprints.
    """
    order = spec.get("order", ["num_peak", "morgan", "rdkit"])
    parts = []
    for item in order:
        if item == "num_peak":
            parts.append(np.array([float(num_peak)], dtype=np.float64))
        elif item == "morgan":
            parts.append(_fingerprint(smiles, kind="morgan",
                                      radius=int(spec.get("morgan_radius", 3)),
                                      n_bits=int(spec.get("morgan_bits", 2048))))
        elif item == "rdkit":
            parts.append(_fingerprint(smiles, kind="rdkit", radius=0,
                                      n_bits=int(spec.get("rdkit_bits", 2048))))
        else:
            raise FeaturizeError(f"unknown global-feature component: {item!r}")
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
