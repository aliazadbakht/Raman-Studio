# Raman Studio

A cross-platform desktop application for acquiring, processing, calibrating, and
identifying Raman spectra. It reads and writes the file formats used by
[OpenRAMAN](https://www.open-raman.org/) spectrometers and started life as a
port of the OpenRAMAN Spectrum Analyzer, adding a Qt interface, spectral
reference-library matching, and structure-based spectrum prediction.

Copyright (c) 2026 [Wfront Principle B.V.](https://wfront.nl) and
[Precisometer B.V.](https://precisometer.com) — licensed under
[CERN-OHL-W v2](LICENSE). Portions copyright (c) Luc Boussemaere (The Pulsar),
OpenRAMAN, same license.

> **Not an official OpenRAMAN release.** This is a modified, independent port,
> not affiliated with, endorsed by, or supported by OpenRAMAN or The Pulsar.
> See [NOTICE.md](NOTICE.md) for full attributions.

## Install and run

**macOS / Linux**

```bash
./install.sh
./run.sh
```

**Windows**

```bat
install.bat
run.bat
```

The installer creates a `venv/` and installs the dependencies in
[requirements.txt](requirements.txt) (NumPy, SciPy, Matplotlib, PySide6) — all
pure-Python or wheel-distributed, on every platform.

### Camera support

Live acquisition requires a FLIR/PointGrey camera, the FLIR Spinnaker SDK, and
`PySpin` installed for the interpreter the launcher uses. Those are not
pip-installable — get them from Teledyne FLIR. **Without `PySpin` the app still
opens, plots, processes, calibrates, matches, and exports spectrum files**; only
live capture is unavailable.

The launchers add the SDK's default install location to the right environment
variable for the platform, if it exists:

| Platform | Variable | Default library path |
| --- | --- | --- |
| macOS | `DYLD_LIBRARY_PATH` | `/usr/local/lib` |
| Linux | `LD_LIBRARY_PATH` | `/opt/spinnaker/lib` |
| Windows | `PATH` | `C:\Program Files\Teledyne\Spinnaker\bin64\vs2015` |

If your SDK is somewhere else, set the variable yourself before launching —
an existing value is never overwritten.

### Choosing the interpreter

Set `RAMAN_PYTHON` to force one:

```bash
RAMAN_PYTHON=/path/to/python ./run.sh     # or: set RAMAN_PYTHON=... && run.bat
```

Otherwise `run.sh` tries, in order: `$RAMAN_PYTHON`, a Spinnaker-capable conda
env (`$RAMAN_FLIR_PYTHON`, defaulting to `/opt/anaconda3/envs/raman_flir`), the
local `venv/`, then system `python3`. `run.bat` tries `%RAMAN_PYTHON%`, the
local `venv\`, then `python` on `PATH`.

> **Platform testing:** development and testing happen on macOS. The Linux and
> Windows paths are implemented and are expected to work, but are not covered by
> routine testing — reports welcome.

## Features

- **Acquisition** — live camera feed with hardware ROI management (FLIR/PointGrey).
- **Processing** — baseline removal, Savitzky-Golay smoothing, median filtering,
  blank subtraction, peak detection.
- **Calibration** — wavelength calibration against known emission lines, with
  automatic peak-to-line assignment. Coefficients are stored in the OpenRAMAN
  Legendre basis and can be read from or written to camera user memory.
- **Matching** — score a spectrum against a reference library by cosine
  similarity (HQI) and overlay the hits.
- **Prediction** — predict a spectrum from a molecular structure.

## File support

- OpenRAMAN `.spc` read/write — spectrum data, blank data, camera UID, and
  calibration coefficients.
- CSV read/write.
- Legacy `.rspc` read/write for files made by earlier builds of this port.

## Reference library

The **Match** panel scores the current spectrum against a library of reference
spectra. A small starter set ships in `references/builtin/` so matching works on
first run. For broader coverage, use the Match panel's **Library** section:
**Download RRUFF subset…** to fetch a quality-graded bundle, or **Add spectra
from folder…** to import your own RRUFF `.txt` / two-column CSV files, then
**Rebuild library index**.

Reference spectra come from the [RRUFF database](https://rruff.info/)
(University of Arizona). **This data is third-party and is not covered by this
repository's license** — see [references/builtin/NOTICE.md](references/builtin/NOTICE.md)
for terms. If you publish results made with it, please cite:

> Lafuente B, Downs R T, Yang H, Stone N (2015) The power of databases:
> the RRUFF project. In: Highlights in Mineralogical Crystallography,
> T Armbruster and R M Danisi, eds. Berlin, Germany, W. De Gruyter,
> pp 1–30.

## Predict from structure

The Match panel can predict and overlay a Raman spectrum from a molecular
structure using [Mol2Raman](https://github.com/salvasorrentino/Mol2Raman) (MIT),
a graph neural network that predicts Raman spectra from SMILES. Input can be a
SMILES string, a compound name, or a formula.

Real predictions require the Mol2Raman ML dependencies **and** the trained
checkpoints, which are not distributed here — get them from the Mol2Raman
authors. Bundle setup is documented in
[`models/mol2raman/README.md`](models/mol2raman/README.md). Until a bundle is
installed the UI shows a clearly labelled **demo** curve, which is synthetic and
is not a prediction.

## Repository layout

```
main_qt.py            entry point
install.sh / .bat     dependency setup (macOS+Linux / Windows)
run.sh / run.bat      launcher (macOS+Linux / Windows)
app/                  application code
  gui_qt.py           main window (PySide6)
  match_panel_qt.py   library matching panel
  calibration_dialog_qt.py
  camera.py           FLIR/Spinnaker acquisition
  dsp.py              filtering, baselines, peaks, calibration fitting
  library.py          reference library indexing
  matching.py         HQI scoring
  fileio.py           CSV / .rspc I/O
  openraman_spc.py    OpenRAMAN .spc reader/writer
  mol2raman/          structure → spectrum prediction
    vendor/           vendored upstream model code (MIT)
models/mol2raman/     drop trained checkpoints here
references/builtin/   starter reference spectra (third-party data)
archive/              superseded Tkinter front end, kept for reference
```

The Tkinter front end under `archive/` is superseded — `main_qt.py` and
`app/gui_qt.py` are the current UI. The backend modules are shared.

## License

CERN-OHL-W v2 (weakly reciprocal). You may use, study, modify, and distribute
this work, including commercially; if you distribute a modified version you must
release your modified source under the same license. See [LICENSE](LICENSE) for
the full terms and [NOTICE.md](NOTICE.md) for third-party attributions.
