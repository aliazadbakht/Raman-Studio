# Raman Spectrum Analyzer for macOS

Copyright (c) 2026 Wfront Principle B.V. — licensed under [CERN-OHL-W v2](LICENSE).

Python/Tkinter macOS port of the [OpenRAMAN Spectrum Analyzer](https://www.open-raman.org/),
which is also licensed under CERN-OHL-W v2.

## Run

```bash
./install.sh
./run.sh
```

FLIR/PointGrey camera capture requires FLIR Spinnaker SDK plus `PySpin` installed
for the Python interpreter used by `run.sh`. The app can still open, plot, process,
and export spectrum files without `PySpin`.

Set `RAMAN_PYTHON=/path/to/python ./run.sh` to force a specific Python.

## File Support

- OpenRAMAN `.spc` read/write for spectrum data, blank data, camera UID, and
  calibration coefficients.
- CSV read/write.
- Legacy `.rspc` read/write for files made by earlier macOS port builds.

Calibration coefficients are stored in the OpenRAMAN Legendre basis.

## Reference Library

Spectral reference data in `references/` is sourced from the
[RRUFF database](https://rruff.info/) (University of Arizona).
Please cite:

> Lafuente B, Downs R T, Yang H, Stone N (2015) The power of databases:
> the RRUFF project. In: Highlights in Mineralogical Crystallography,
> T Armbruster and R M Danisi, eds. Berlin, Germany, W. De Gruyter,
> pp 1–30.
