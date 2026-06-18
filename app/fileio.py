# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V.
"""File I/O: CSV load/save and a simple .rspc binary format."""
import csv
import json
import struct
import numpy as np
import os

from . import dsp, openraman_spc


# ── CSV ───────────────────────────────────────────────────────────────────────

def load_csv(path):
    """
    Load a two-column CSV/TXT (x, y). First row may be a header.

    Also tolerates RRUFF-style ``##KEY=VALUE`` metadata lines, which are
    skipped without poisoning the x/y labels. If a ``##NAMES=`` field is
    present, it becomes the y-axis label so the spectrum is identifiable.

    Returns (x_array, y_array, x_label, y_label).
    """
    rows = []
    x_label, y_label = "x", "Intensity"
    rruff_name = None
    has_rruff_meta = False
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith("##"):
                has_rruff_meta = True
                if first.upper().startswith("##NAMES="):
                    rruff_name = first.split("=", 1)[1].strip()
                continue
            try:
                x = float(first)
                y = float(row[1].strip())
                rows.append((x, y))
            except (ValueError, IndexError):
                if len(row) >= 2 and not has_rruff_meta:
                    x_label = first
                    y_label = row[1].strip()
    if not rows:
        raise ValueError("No numeric data found in CSV")
    if has_rruff_meta:
        x_label = "Raman Shift (cm⁻¹)"
        y_label = rruff_name or "Intensity"
    xs, ys = zip(*rows)
    return np.array(xs), np.array(ys), x_label, y_label


def save_csv(path, x, y, x_label="x", y_label="Intensity"):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([x_label, y_label])
        for xi, yi in zip(x, y):
            writer.writerow([f"{xi:.6e}", f"{yi:.6e}"])


# ── OpenRAMAN .spc ────────────────────────────────────────────────────────────

def load_spc(path):
    return openraman_spc.load(path)


def save_spc(path, signal, calibration=None, blank=None, uid=""):
    openraman_spc.save(path, signal, calibration=calibration, blank=blank, uid=uid)


# ── .rspc binary format ───────────────────────────────────────────────────────
# Header: magic(4B) + version(2B) + n_pixels(4B) + has_calibration(1B)
#         + calibration_coeffs(4×8B) + has_blank(1B) + config_json_len(4B)
#         + config_json(variable) + signal_data(n_pixels×8B)
#         [+ blank_data(n_pixels×8B) if has_blank]

MAGIC = b'RSPC'
VERSION = 1


def save_rspc(path, signal, calibration=None, blank=None, config=None):
    has_cal = calibration is not None
    has_blank = blank is not None
    config = dict(config or {})
    if has_cal:
        config.setdefault("calibration_basis", "legendre")
    config_bytes = json.dumps(config or {}).encode()

    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<H', VERSION))
        f.write(struct.pack('<I', len(signal)))
        f.write(struct.pack('?', has_cal))
        if has_cal:
            coeffs = list(calibration) + [0.0] * (4 - len(calibration))
            f.write(struct.pack('<4d', *coeffs[:4]))
        else:
            f.write(struct.pack('<4d', 0, 0, 0, 0))
        f.write(struct.pack('?', has_blank))
        f.write(struct.pack('<I', len(config_bytes)))
        f.write(config_bytes)
        f.write(struct.pack(f'<{len(signal)}d', *signal))
        if has_blank:
            f.write(struct.pack(f'<{len(blank)}d', *blank))


def load_rspc(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("Not a valid .rspc file")
        version = struct.unpack('<H', f.read(2))[0]
        n = struct.unpack('<I', f.read(4))[0]
        has_cal = struct.unpack('?', f.read(1))[0]
        coeffs = list(struct.unpack('<4d', f.read(32)))
        has_blank = struct.unpack('?', f.read(1))[0]
        cfg_len = struct.unpack('<I', f.read(4))[0]
        config = json.loads(f.read(cfg_len).decode())
        signal = np.array(struct.unpack(f'<{n}d', f.read(n * 8)))
        blank = None
        if has_blank:
            blank = np.array(struct.unpack(f'<{n}d', f.read(n * 8)))

    if has_cal and config.get("calibration_basis") == "legendre":
        calibration = coeffs
    elif has_cal:
        calibration = dsp.power_to_legendre(coeffs)
    else:
        calibration = None
    return signal, calibration, blank, config


# ── Calibration Persistence ───────────────────────────────────────────────────

def save_calibration(coeffs):
    """Save calibration coefficients to a local JSON file."""
    # Save in the same directory as main.py (one level above app/)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration.json")
    try:
        with open(path, 'w') as f:
            json.dump({"basis": "legendre", "coeffs": list(coeffs)}, f)
    except Exception as e:
        print(f"Error saving calibration: {e}")


def load_calibration():
    """Load calibration coefficients from the local JSON file."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                coeffs = data.get("coeffs")
                if data.get("basis") == "power":
                    return dsp.power_to_legendre(coeffs)
                return coeffs
            # Older Mac builds saved power-basis coefficients as a bare list.
            return dsp.power_to_legendre(data)
        except Exception as e:
            print(f"Error loading calibration: {e}")
    return None
