# Raman Spectrum Analyzer for macOS

Python/Tkinter macOS port of the OpenRAMAN Spectrum Analyzer.

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
