# Vendored third-party code — Mol2Raman

`model.py` in this directory is copied verbatim from the **Mol2Raman** project:

- Source: https://github.com/salvasorrentino/Mol2Raman
- File: `Scripts/script_model/model.py`
- License: MIT (see `LICENSE` in this directory) — Copyright (c) 2025 salvasorrentino
- Paper: *Mol2Raman — a graph neural network for predicting Raman spectra from SMILES*

It is bundled so the trained model architectures can be instantiated before a
checkpoint's `state_dict` is loaded. It is unmodified except for an added
attribution header. Treat it as an external dependency — do not edit it; if
Mol2Raman updates its architecture, re-copy the file and refresh the bundle
descriptor accordingly.

The rest of `app/mol2raman/` (the adapter, featurization wrapper, resolver,
post-processing port, and UI) is original work under the project's
CERN-OHL-W-2.0 license.
