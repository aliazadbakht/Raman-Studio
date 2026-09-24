# 🔬 Raman Studio v1.0.0 — Initial Public Release

We are excited to announce the initial public release of **Raman Studio** v1.0.0!

**Raman Studio** is a cross-platform desktop application for acquiring, processing, calibrating, and identifying Raman spectra. Built with PySide6 (Qt), it provides full compatibility with [OpenRAMAN](https://www.open-raman.org/) spectrometers while expanding capability with spectral library matching and structure-based spectrum prediction.

---

### ✨ Key Features

#### 📹 Hardware & Live Acquisition
* **Live Camera Feed**: Support for Teledyne FLIR / PointGrey cameras via the PySpin SDK.
* **Hardware ROI Management**: Configure sensor region-of-interest directly on the hardware.
* **Headless / Offline Mode**: Fully functional spectrum plotting, processing, calibration, and identification without FLIR hardware attached.

#### ⚙️ Spectral Processing & Analysis
* **Baseline Removal**: Iterative polynomial and baseline correction algorithms.
* **Smoothing & Noise Reduction**: Savitzky-Golay filtering and median smoothing.
* **Blank Subtraction & Peak Detection**: Automated background spectrum subtraction and peak locating.

#### 🎯 High-Precision Wavelength Calibration
* Multi-point calibration against known emission lines with automatic peak-to-line assignment.
* Stores calibration coefficients in the **OpenRAMAN Legendre polynomial basis**.
* Direct read/write support for camera user memory (flash storage).

#### 📚 Spectral Library Matching
* High-quality spectral identification using **Cosine Similarity (HQI)**.
* Included starter set in `references/builtin/`.
* One-click download of the curated **RRUFF mineral & compound database**, plus support for custom CSV / `.txt` reference imports.

#### 🧪 Structure-Based Prediction (`Mol2Raman`)
* Predict Raman spectra directly from molecular structures.

#### 💾 Open Data Standards & Interoperability
* Native read/write support for OpenRAMAN `.spc` files (containing raw data, blank data, camera UID, and calibration metadata).
* CSV and legacy `.rspc` export/import capabilities.

---

### 💻 Platform Support & Installation
* **macOS / Linux**: `./install.sh` and `./run.sh`
* **Windows**: `install.bat` and `run.bat`

Dependencies are packaged for pure-Python or wheel distribution (`PySide6`, `NumPy`, `SciPy`, `Matplotlib`).

---

### 📄 License & Attributions
Licensed under **[CERN-OHL-W v2](LICENSE)**. Copyright (c) 2026 Precisometer B.V. Base project is OpenRAMAN, portions copyright (c) Luc Boussemaere (The Pulsar), OpenRAMAN. See [`NOTICE.md`](NOTICE.md) for full attributions.
