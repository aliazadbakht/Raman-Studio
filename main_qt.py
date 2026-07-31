#!/usr/bin/env python3
# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V. (https://wfront.nl)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Raman Spectrum Analyzer for macOS — PySide6 (Qt) preview front-end.

Runs the Qt proof-of-concept window. The Tkinter app (main.py) is unchanged;
both share the same backend (camera/dsp/fileio).
"""
import os

# Auto-configure Spinnaker SDK environment variables for macOS (same as main.py)
if "DYLD_LIBRARY_PATH" not in os.environ:
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"
if "GENICAM_GENTL64_PATH" not in os.environ:
    os.environ["GENICAM_GENTL64_PATH"] = "/usr/local/lib/spinnaker-gentl"

from app.gui_qt import run

if __name__ == "__main__":
    run()
