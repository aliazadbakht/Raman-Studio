#!/usr/bin/env python3
# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V. (https://wfront.nl)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Raman Studio — entry point.

Raman spectrum acquisition, processing, calibration, and identification.
Runs the PySide6 (Qt) front end in app/gui_qt.py; the Tkinter front end under
archive/ is superseded and shares the same backend (camera/dsp/fileio).
"""
import os
import sys

# Point the optional FLIR Spinnaker SDK at its libraries. The variable that
# matters is different on every platform, and setting the wrong one is at best
# useless — so only touch the one this OS actually consults. Defaults are the
# stock SDK install locations; an existing value is always left alone.
_SPINNAKER_DEFAULTS = {
    "darwin": ("DYLD_LIBRARY_PATH", "/usr/local/lib", "/usr/local/lib/spinnaker-gentl"),
    "linux":  ("LD_LIBRARY_PATH", "/opt/spinnaker/lib", "/opt/spinnaker/lib/flir-gentl"),
    "win32":  ("PATH", r"C:\Program Files\Teledyne\Spinnaker\bin64\vs2015",
               r"C:\Program Files\Teledyne\Spinnaker\cti64\vs2015"),
}

for _key, (_var, _libdir, _gentl) in _SPINNAKER_DEFAULTS.items():
    if not sys.platform.startswith(_key):
        continue
    if os.path.isdir(_libdir):
        existing = os.environ.get(_var, "")
        if _libdir not in existing.split(os.pathsep):
            os.environ[_var] = f"{_libdir}{os.pathsep}{existing}" if existing else _libdir
    if "GENICAM_GENTL64_PATH" not in os.environ and os.path.isdir(_gentl):
        os.environ["GENICAM_GENTL64_PATH"] = _gentl
    break

from app.gui_qt import run

if __name__ == "__main__":
    run()
