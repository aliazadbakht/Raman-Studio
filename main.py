#!/usr/bin/env python3
# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V.
"""Raman Spectrum Analyzer for macOS"""
import os

# Auto-configure Spinnaker SDK environment variables for macOS
if "DYLD_LIBRARY_PATH" not in os.environ:
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"
if "GENICAM_GENTL64_PATH" not in os.environ:
    os.environ["GENICAM_GENTL64_PATH"] = "/usr/local/lib/spinnaker-gentl"

from app.gui import RamanApp
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = RamanApp(root)
    root.mainloop()
