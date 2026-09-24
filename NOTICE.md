# Notices and attributions

Raman Studio

Copyright (c) 2026 Precisometer B.V. — https://precisometer.com

Licensed under CERN-OHL-W-2.0 — see [LICENSE](LICENSE).

This file lists the upstream and third-party work this project builds on. Each
item remains under its own license and copyright.

---

## 1. OpenRAMAN Spectrum Analyzer — upstream work

- Project: OpenRAMAN — https://www.open-raman.org/
- Copyright: Luc Boussemaere (The Pulsar)
- License: CERN-OHL-W v2

This repository is a **modified** work derived from the OpenRAMAN Spectrum
Analyzer, and is distributed under the same license as required by CERN-OHL-W
v2's reciprocity terms. Derived elements include the OpenRAMAN `.spc` container
format (storage objects, RLE0 encoding), the camera calibration blob layout and
checksum, and the Legendre calibration basis.

Not affiliated with, endorsed by, or supported by OpenRAMAN or The Pulsar.
"OpenRAMAN" is used only to identify the upstream project and the file formats
this software reads and writes. CERN-OHL-W v2 §8.3 grants no trademark or
trade-name rights.

## 2. Mol2Raman — vendored source

- Project: Mol2Raman — https://github.com/salvasorrentino/Mol2Raman
- Copyright: (c) 2025 salvasorrentino
- License: MIT

`app/mol2raman/vendor/model.py` is an unmodified copy of the upstream
`Scripts/script_model/model.py` (an attribution header was prepended). It stays
under MIT — see [app/mol2raman/vendor/LICENSE](app/mol2raman/vendor/LICENSE) and
[app/mol2raman/vendor/NOTICE.md](app/mol2raman/vendor/NOTICE.md).

The rest of `app/mol2raman/` is original work under this repository's license,
though `postprocess.py` and `featurize.py` are functional ports of the maths and
featurization in the upstream project.

No Mol2Raman trained checkpoints are distributed here. Obtain those from the
project authors; see [models/mol2raman/README.md](models/mol2raman/README.md).

## 3. RRUFF Project — reference data

- Source: https://rruff.info/
- Copyright: RRUFF Project / University of Arizona

The spectra in `references/builtin/` are third-party data and are **not**
covered by this repository's license. See
[references/builtin/NOTICE.md](references/builtin/NOTICE.md) for attribution,
citation, and terms.

## 4. Runtime dependencies

Installed via `pip` at setup time; not redistributed in this repository.

| Package | License |
| --- | --- |
| PySide6 (Qt for Python) | LGPL v3 |
| NumPy | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| Matplotlib | Matplotlib (PSF-based) |
| PySpin / FLIR Spinnaker SDK (optional) | Proprietary — obtain from Teledyne FLIR |
| torch, torch_geometric, rdkit, deepchem (optional) | See each project |

PySide6 is LGPL v3. Because it is installed as a separate `pip` dependency and
dynamically linked at runtime, this repository imposes no additional
obligations on you. If you build and distribute a bundled `.app` that ships Qt
binaries, LGPL v3's relinking and notice obligations then apply to that bundle.

## 5. Sample data

Files under `spectrum tests/` are original measurements made with this software
and are covered by this repository's license.
