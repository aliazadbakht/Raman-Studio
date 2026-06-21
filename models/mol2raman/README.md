# Mol2Raman model bundle

Drop the trained Mol2Raman checkpoints here to enable **Predict from structure**
in the Match panel (SMILES / name / formula → predicted Raman spectrum).

Until this is populated, the feature still works but produces a clearly-labelled
**demo** spectrum (synthetic, not a real prediction).

## What goes here

1. The checkpoint files (`.pth`) supplied by the Mol2Raman authors
   (https://github.com/salvasorrentino/Mol2Raman). The paper's pipeline uses
   four: two spectral models (low "fingerprint" band + high "CH" band) and two
   peak-count models. A simpler single-band checkpoint also works — use one band
   in the descriptor.
2. A `bundle.json` describing how to wire them up. Copy
   [`bundle.template.json`](bundle.template.json) to `bundle.json` and edit.

```
models/mol2raman/
├── bundle.json                       ← you create this (copy the template)
├── spectra_fingerprint_500_2100.pth  ← from the authors
├── spectra_ch_1900_3500.pth
├── numpeak_down.pth
└── numpeak_up.pth
```

You can also point elsewhere by setting `MOL2RAMAN_BUNDLE_DIR`.

## Install the ML stack (one-time)

Real prediction needs the Mol2Raman dependencies in the app's Python env:

```
pip install torch torch_geometric rdkit deepchem scipy
```

(See the upstream `requirements.txt` for exact versions. The app reports which
of these are missing in the Predict dialog.)

## Verify on first run

The adapter follows the upstream `predict_raman_spectra.py` + `final_output.py`
control flow, but a few things are defined by the authors' training artifacts,
not the public repo. On the first real prediction, confirm against their
`config.json`:

- `n_data_points` per band (template assumes 800),
- the stitch axis (`axis_start/stop/step`),
- each model's class (`GINE` vs `GINEGLOBAL` vs …),
- the **global feature order and sizes** (`num_peak` + Morgan + RDKit = 4097 in
  the template). If the model was trained with a different fingerprint
  size/order, prediction shapes will mismatch — adjust `global_features`.
