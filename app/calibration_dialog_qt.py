# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V.
"""Wavelength calibration dialogs — PySide6 port.

Faithful port of calibration_dialog.py (Tkinter). All numerical work is reused
from app.dsp; these classes are UI shells only.
"""
import numpy as np
from numpy.polynomial import legendre
from scipy.signal import find_peaks

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QComboBox, QSlider, QSpinBox, QLineEdit, QPushButton, QLabel,
    QCheckBox, QTreeWidget, QTreeWidgetItem, QFormLayout, QGridLayout,
    QVBoxLayout, QHBoxLayout, QGroupBox, QMessageBox, QWidget,
)

from . import dsp

_CYAN, _TEXT, _MUTED = "#00d4ff", "#e0e0e0", "#888888"
_GREEN, _AMBER, _RED = "#aaffaa", "#ffdd57", "#ff6b6b"


class CalibrationDialogQt(QDialog):
    """Neon / Mercury-Argon lamp calibration with manual + auto fitting."""

    def __init__(self, parent, spectrum_y, on_solution,
                 on_load_camera=None, on_copy_to_camera=None, laser_nm=532.0):
        super().__init__(parent)
        self.setWindowTitle("Wavelength Calibration")
        self.on_solution = on_solution
        self.on_load_camera = on_load_camera
        self.on_copy_to_camera = on_copy_to_camera
        self.laser_nm = float(laser_nm)
        self.spectrum_y = spectrum_y
        self.peak_pixels = []
        self.known_wl = []
        self._auto_assigned = False
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        v = QVBoxLayout(self)

        title = QLabel("Wavelength Calibration")
        title.setStyleSheet(f"color:{_CYAN}; font-size:15px; font-weight:bold;")
        v.addWidget(title)

        tip = QLabel("💡  Easier option: close this and click 🧪 Quick Cal if you have a "
                     "pure liquid (IPA, ethanol, cyclohexane). This dialog is for "
                     "Neon / Mercury-Argon lamp spectra.")
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color:{_GREEN}; background:#0d2840; padding:6px; border-radius:3px;")
        v.addWidget(tip)

        # Source + model
        src_box = QGroupBox("Calibration source")
        src = QHBoxLayout(src_box)
        self.source_combo = QComboBox(); self.source_combo.addItems(["Neon", "Mercury-Argon"])
        self.source_combo.currentTextChanged.connect(self._on_source_change)
        self.model_combo = QComboBox(); self.model_combo.addItems(["Linear", "Cubic"])
        self.model_combo.setCurrentText("Cubic")
        src.addWidget(QLabel("Source:")); src.addWidget(self.source_combo)
        src.addSpacing(12)
        src.addWidget(QLabel("Model:")); src.addWidget(self.model_combo)
        src.addStretch(1)
        v.addWidget(src_box)

        # Sensitivity + max peaks
        sens_box = QGroupBox("Detection")
        sens = QHBoxLayout(sens_box)
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(10, 100); self.sens_slider.setValue(70)
        self.sens_slider.setFixedWidth(180)
        self.sens_label = QLabel("70%"); self.sens_label.setStyleSheet(f"color:{_CYAN};")
        self.sens_slider.valueChanged.connect(lambda v_: self.sens_label.setText(f"{v_}%"))
        self.npeaks_spin = QSpinBox(); self.npeaks_spin.setRange(2, 30)
        self.npeaks_spin.setValue(len(dsp.NEON_LINES))
        sens.addWidget(QLabel("Peak sensitivity:")); sens.addWidget(self.sens_slider)
        sens.addWidget(self.sens_label); sens.addSpacing(12)
        sens.addWidget(QLabel("Max peaks:")); sens.addWidget(self.npeaks_spin)
        sens.addStretch(1)
        v.addWidget(sens_box)

        # Advanced constraints (collapsible)
        self._adv_btn = QPushButton("▸  Advanced search constraints")
        self._adv_btn.setStyleSheet("text-align:left; background:transparent; color:#aaaaaa;")
        self._adv_btn.clicked.connect(self._toggle_advanced)
        v.addWidget(self._adv_btn)
        self._adv_box = self._build_advanced()
        self._adv_box.setVisible(False)
        v.addWidget(self._adv_box)

        # Peak table
        v.addWidget(QLabel("Matched peaks  (Pixel → Wavelength nm, residual after fit)"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Pixel", "Wavelength (nm)", "Δ (nm)", "Δ (cm⁻¹)"])
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(180)
        self.tree.itemSelectionChanged.connect(self._on_tree_select)
        v.addWidget(self.tree)

        # Add / update / remove
        edit = QHBoxLayout()
        self.pix_entry = QLineEdit(); self.pix_entry.setFixedWidth(70)
        self.wl_entry = QLineEdit(); self.wl_entry.setFixedWidth(90)
        b_add = QPushButton("Add"); b_add.clicked.connect(self._add_pair)
        b_upd = QPushButton("Update"); b_upd.clicked.connect(self._update_pair)
        b_rem = QPushButton("Remove"); b_rem.clicked.connect(self._remove_pair)
        edit.addWidget(QLabel("Pixel:")); edit.addWidget(self.pix_entry)
        edit.addWidget(QLabel("nm:")); edit.addWidget(self.wl_entry)
        edit.addWidget(b_add); edit.addWidget(b_upd); edit.addWidget(b_rem)
        edit.addStretch(1)
        v.addLayout(edit)

        # Auto-detect + camera memory
        row = QHBoxLayout()
        b_auto = QPushButton("Auto-Detect Lamp Peaks"); b_auto.clicked.connect(self._auto_detect)
        b_load = QPushButton("Hard Load from Camera"); b_load.clicked.connect(self._hard_load_from_camera)
        b_copy = QPushButton("Hard Copy to Camera"); b_copy.clicked.connect(self._hard_copy_to_camera)
        row.addWidget(b_auto); row.addStretch(1); row.addWidget(b_load); row.addWidget(b_copy)
        v.addLayout(row)

        # Quality banner
        self.quality_label = QLabel("Fit quality will appear here once you click Calibrate.")
        self.quality_label.setWordWrap(True)
        self.quality_label.setStyleSheet(f"color:{_MUTED}; background:#0d2840; padding:6px; border-radius:3px;")
        v.addWidget(self.quality_label)

        # Status
        self.status_label = QLabel("Automatic calibration requires a Neon/Hg-Ar lamp "
                                   "spectrum, not a sample spectrum.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#aaaaaa;")
        v.addWidget(self.status_label)

        # Buttons
        btns = QHBoxLayout()
        b_cal = QPushButton("Calibrate"); b_cal.clicked.connect(self._calibrate)
        b_cal.setStyleSheet(f"background:{_CYAN}; color:#000; font-weight:bold; padding:6px 18px;")
        b_close = QPushButton("Close"); b_close.clicked.connect(self.reject)
        btns.addStretch(1); btns.addWidget(b_cal); btns.addWidget(b_close)
        v.addLayout(btns)

    def _build_advanced(self):
        box = QGroupBox("Advanced search constraints")
        g = QGridLayout(box)
        self.range_min = self._spin(200, 1100, 500)
        self.range_max = self._spin(200, 1100, 800)
        self.span_min = self._spin(1, 500, 100)
        self.span_max = self._spin(1, 500, 150)
        self.distortion_max = self._spin(0, 100, 10)
        self.sampling = self._spin(3, 30, 10)
        self.lock_raman = QCheckBox("Lock Raman range"); self.lock_raman.setChecked(True)
        self.raman_min = self._spin(0, 5000, 500, step=10)
        self.raman_max = self._spin(0, 5000, 3500, step=10)
        g.addWidget(QLabel("Wavelength:"), 0, 0); g.addWidget(self.range_min, 0, 1); g.addWidget(self.range_max, 0, 2)
        g.addWidget(QLabel("Span:"), 0, 3); g.addWidget(self.span_min, 0, 4); g.addWidget(self.span_max, 0, 5)
        g.addWidget(QLabel("Distortion:"), 1, 0); g.addWidget(self.distortion_max, 1, 1)
        g.addWidget(QLabel("Search:"), 1, 3); g.addWidget(self.sampling, 1, 4)
        g.addWidget(self.lock_raman, 2, 0, 1, 2); g.addWidget(self.raman_min, 2, 2); g.addWidget(self.raman_max, 2, 3)
        return box

    def _spin(self, lo, hi, val, step=1):
        s = QSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setValue(val)
        s.setFixedWidth(70)
        return s

    def _toggle_advanced(self):
        vis = not self._adv_box.isVisible()
        self._adv_box.setVisible(vis)
        self._adv_btn.setText(("▾" if vis else "▸") + "  Advanced search constraints")

    # ── Logic (ported) ──────────────────────────────────────────────────────
    def _source_lines(self):
        return dsp.NEON_LINES if self.source_combo.currentText() == "Neon" else dsp.MERCURY_ARGON_LINES

    def _on_source_change(self, *_):
        self.npeaks_spin.setValue(len(self._source_lines()))
        self.status_label.setText(
            f"{self.source_combo.currentText()} selected. Capture that lamp spectrum "
            "before auto calibration.")

    def _auto_params(self):
        try:
            params = {
                "range_min": float(self.range_min.value()),
                "range_max": float(self.range_max.value()),
                "span_min": float(self.span_min.value()),
                "span_max": float(self.span_max.value()),
                "distortion_max": float(self.distortion_max.value()),
                "sampling": int(self.sampling.value()),
            }
            if self.lock_raman.isChecked():
                e_min = self._raman_shift_to_wavelength(float(self.raman_min.value()))
                e_max = self._raman_shift_to_wavelength(float(self.raman_max.value()))
                params.update({"endpoint_min": min(e_min, e_max),
                               "endpoint_max": max(e_min, e_max),
                               "endpoint_weight": 10.0})
            return params
        except ValueError as exc:
            raise ValueError("Calibration search parameters must be numeric.") from exc

    def _raman_shift_to_wavelength(self, shift_cm):
        denom = (1.0 / self.laser_nm) - (float(shift_cm) / 1.0e7)
        if denom <= 0:
            raise ValueError("Raman shift is too high for the current laser wavelength.")
        return 1.0 / denom

    def _add_pair(self):
        try:
            p = float(self.pix_entry.text()); w = float(self.wl_entry.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Enter valid numbers for pixel and wavelength.")
            return
        for i, pix in enumerate(self.peak_pixels):
            if abs(pix - p) < 1.0:
                self.known_wl[i] = w
                self._set_row(i, (f"{p:.1f}", f"{w:.3f}", "—", "—"), _GREEN)
                self.status_label.setText(f"Updated peak at pixel {p:.1f}")
                return
        self.peak_pixels.append(p); self.known_wl.append(w); self._auto_assigned = False
        self._append_row((f"{p:.1f}", f"{w:.3f}", "—", "—"), _GREEN)
        self.pix_entry.clear(); self.wl_entry.clear()

    def _remove_pair(self):
        idx = self._selected_index()
        if idx is None:
            return
        self.tree.takeTopLevelItem(idx)
        del self.peak_pixels[idx]; del self.known_wl[idx]

    def _update_pair(self):
        idx = self._selected_index()
        if idx is None:
            QMessageBox.information(self, "Select Row", "Select a peak from the list first.")
            return
        try:
            w = float(self.wl_entry.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Enter a valid wavelength (nm).")
            return
        self.known_wl[idx] = w; self._auto_assigned = False
        p = self.peak_pixels[idx]
        self._set_row(idx, (f"{p:.1f}", f"{w:.3f}", "—", "—"), _GREEN)
        self.status_label.setText(f"Updated Pixel {p:.1f} → {w:.3f} nm")
        self.wl_entry.clear()

    def _on_tree_select(self):
        idx = self._selected_index()
        if idx is None:
            return
        self.pix_entry.setText(f"{self.peak_pixels[idx]:.1f}")
        w = self.known_wl[idx]
        self.wl_entry.setText(f"{w:.3f}" if w > 0 else "")

    def _auto_detect(self):
        if self.spectrum_y is None:
            QMessageBox.information(self, "No Spectrum", "Load or acquire a spectrum first.")
            return
        peaks = dsp.detect_calibration_peaks(self.spectrum_y, n_peaks=self.npeaks_spin.value(),
                                             sensitivity=self.sens_slider.value() / 100.0)
        self.tree.clear(); self.peak_pixels.clear(); self.known_wl.clear()
        self._auto_assigned = False
        if len(peaks) == 0:
            self.status_label.setText("No calibration peaks detected. Increase sensitivity or exposure.")
            return
        for p in peaks:
            self.peak_pixels.append(float(p)); self.known_wl.append(0.0)
            self._append_row((f"{p:.1f}", "auto", "—", "—"), _MUTED)
        self.status_label.setText(
            f"Detected {len(peaks)} lamp candidate peaks. Click Calibrate for automatic "
            f"{self.source_combo.currentText()} source-line matching, or enter known "
            "wavelengths manually.")

    def _format_coeffs(self, coeffs):
        return ", ".join(f"{float(c):.6g}" for c in coeffs)

    def _hard_load_from_camera(self):
        if self.on_load_camera is None:
            QMessageBox.information(self, "Camera Calibration", "Camera calibration loading is not available.")
            return
        try:
            coeffs = self.on_load_camera()
        except Exception as e:
            QMessageBox.critical(self, "Camera Calibration", str(e)); return
        self.status_label.setText("Hard-loaded OpenRAMAN calibration from camera. "
                                  f"Coefficients: {self._format_coeffs(coeffs)}")

    def _hard_copy_to_camera(self):
        if self.on_copy_to_camera is None:
            QMessageBox.information(self, "Camera Calibration", "Camera calibration upload is not available.")
            return
        if QMessageBox.question(
                self, "Copy Calibration to Camera",
                "This overwrites the OpenRAMAN calibration stored in camera memory with the "
                "current Mac calibration.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            coeffs = self.on_copy_to_camera()
        except Exception as e:
            QMessageBox.critical(self, "Camera Calibration", str(e)); return
        self.status_label.setText("Hard-copied current calibration into camera memory. "
                                  f"Coefficients: {self._format_coeffs(coeffs)}")

    def _calibrate(self):
        if not self.peak_pixels:
            QMessageBox.critical(self, "Error", "Auto-detect peaks or add calibration pairs first.")
            return
        degree = 3 if self.model_combo.currentText() == "Cubic" else 1
        n_pixels = len(self.spectrum_y) if self.spectrum_y is not None else None
        valid_pix, valid_wl = [], []
        for p, w in zip(self.peak_pixels, self.known_wl):
            if w > 0:
                valid_pix.append(p); valid_wl.append(w)

        mode, rms, info = "manual", None, None
        try:
            manual_required = degree + 1
            use_auto = len(valid_pix) == 0 or self._auto_assigned
            if len(valid_pix) >= manual_required and not use_auto:
                coeffs = dsp.fit_calibration(valid_pix, valid_wl, degree=degree, n_pixels=n_pixels)
                rms = dsp.calibration_pair_rms(coeffs, valid_pix, valid_wl, n_pixels=n_pixels)
            elif use_auto:
                mode = "auto"
                coeffs, info = dsp.fit_calibration_to_source(
                    self.peak_pixels, self._source_lines(),
                    model=self.model_combo.currentText(), n_pixels=n_pixels, **self._auto_params())
                rms = info["rms_inliers"]
                for idx, nearest in enumerate(info["nearest"]):
                    self.known_wl[idx] = float(nearest) if np.isfinite(nearest) else 0.0
                self._auto_assigned = True
                self._populate_residuals_in_table(coeffs, info=info)
                self._show_quality_summary(coeffs, info=info)
                default_no = info.get("rms_inliers", 0.0) > 0.5 or info.get("n_outliers", 0) > 0
                if QMessageBox.question(
                        self, "Apply automatic calibration?", self._auto_calibration_warning(info),
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No if default_no else QMessageBox.Yes) != QMessageBox.Yes:
                    return
            else:
                QMessageBox.critical(self, "Error",
                    f"Need at least {manual_required} manual wavelength pairs for "
                    f"{self.model_combo.currentText()} fitting, or leave all wavelengths as auto.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Calibration Failed", str(e)); return

        if n_pixels:
            wl = dsp.pixels_to_wavelengths(coeffs, n_pixels)
            diffs = np.diff(wl)
            if not (np.all(diffs > 0) or np.all(diffs < 0)):
                signs = np.sign(diffs); flips = np.where(signs[:-1] != signs[1:])[0]
                bad = int(flips[0]) if len(flips) else 0
                msg = ("Calibration polynomial is non-monotonic.\n\n"
                       f"It produces wavelengths {float(np.min(wl)):.1f}–{float(np.max(wl)):.1f} nm "
                       f"but folds back on itself near pixel {bad}. The plot will show two values "
                       "per cm⁻¹ wherever it folds.\n\nTry: switch Model to Linear, or re-pick peaks "
                       "more spread out.\n\nApply this calibration anyway?")
                if QMessageBox.question(self, "Non-monotonic calibration", msg,
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                    return
        if mode != "auto":
            self._populate_residuals_in_table(coeffs, info=None)
            self._show_quality_summary(coeffs, info=None)
        self.on_solution(coeffs)
        rms_text = f", RMS {rms:.3f} nm" if rms is not None else ""
        self.status_label.setText(
            f"✓ Calibration OK ({mode}{rms_text}). Coefficients: {[round(c, 4) for c in coeffs]}")

    def _auto_calibration_warning(self, info):
        residuals = np.asarray(info.get("residuals", []), dtype=float)
        rms = float(info.get("rms_inliers", info.get("rms", 0.0)))
        max_error = float(info.get("max_error", 0.0))
        unique_lines = int(info.get("unique_lines", 0))
        line_span = float(info.get("line_span", 0.0))
        n_inliers = int(info.get("n_inliers", 0)); n_outliers = int(info.get("n_outliers", 0))
        n_lines = n_inliers + n_outliers if (n_inliers + n_outliers) else len(residuals)
        min_unique = 3 if self.model_combo.currentText() == "Linear" else 5
        issues = []
        if unique_lines < min_unique:
            issues.append(f"only {unique_lines} unique source lines were assigned")
        if line_span < 20.0:
            issues.append(f"assigned source lines span only {line_span:.1f} nm")
        if rms > 2.0:
            issues.append(f"RMS residual is {rms:.2f} nm")
        if max_error > 5.0:
            issues.append(f"largest residual is {max_error:.2f} nm")
        if n_outliers > max(1, n_inliers // 3):
            issues.append(f"{n_outliers} of {n_lines} detected peaks did not match any lamp line")
        summary = (f"{self.source_combo.currentText()} auto-fit diagnostics:\n"
                   f"  peaks: {n_lines}  ({n_inliers} matched, {n_outliers} unmatched)\n"
                   f"  unique source lines: {unique_lines}\n"
                   f"  source-line span: {line_span:.1f} nm\n"
                   f"  RMS residual: {rms:.3f} nm\n"
                   f"  largest residual: {max_error:.3f} nm\n\n"
                   "Automatic calibration must be run on a real calibration source spectrum. "
                   "If this was an IPA/sample spectrum, applying it will force sample peaks onto "
                   "lamp lines and the Raman shifts will be wrong.")
        if issues:
            summary += "\n\nThis fit is suspicious: " + "; ".join(issues) + "."
        summary += "\n\nApply this automatic calibration?"
        return summary

    def _populate_residuals_in_table(self, coeffs, info=None):
        n_pixels = len(self.spectrum_y) if self.spectrum_y is not None else None
        laser = self.laser_nm
        coeffs_arr = np.asarray(coeffs, dtype=float)
        pix_arr = np.asarray(self.peak_pixels, dtype=float)
        if pix_arr.size == 0:
            return
        t = dsp.normalize_pixels(pix_arr, n_pixels=n_pixels)
        projected_wl = legendre.legval(t, coeffs_arr)
        projected_cm = dsp.wavelengths_to_raman(projected_wl, laser)
        inlier_mask = info.get("inlier_mask") if info is not None else None
        for i, (p, known) in enumerate(zip(self.peak_pixels, self.known_wl)):
            row_proj_wl = float(projected_wl[i]); row_proj_cm = float(projected_cm[i])
            if known and known > 0:
                target_cm = float(dsp.wavelengths_to_raman(np.array([float(known)]), laser)[0])
                delta_nm = row_proj_wl - float(known); delta_cm = row_proj_cm - target_cm
                if inlier_mask is not None and i < len(inlier_mask) and not bool(inlier_mask[i]):
                    color = _RED
                elif abs(delta_nm) < 0.05:
                    color = _TEXT
                elif abs(delta_nm) < 0.3:
                    color = _AMBER
                else:
                    color = _RED
                self._set_row(i, (f"{p:.1f}", f"{float(known):.3f}", f"{delta_nm:+.3f}",
                                  f"{delta_cm:+.1f}"), color)
            else:
                self._set_row(i, (f"{p:.1f}", "unassigned", "—", "—"), _RED)

    def _show_quality_summary(self, coeffs, info=None):
        laser = self.laser_nm
        coeffs_arr = np.asarray(coeffs, dtype=float)
        wl_left = float(legendre.legval(-1.0, coeffs_arr))
        wl_right = float(legendre.legval(1.0, coeffs_arr))
        cm_left = float(dsp.wavelengths_to_raman(np.array([wl_left]), laser)[0])
        cm_right = float(dsp.wavelengths_to_raman(np.array([wl_right]), laser)[0])
        cm_lo, cm_hi = sorted((cm_left, cm_right))
        wl_lo, wl_hi = sorted((wl_left, wl_right))
        avg_wl = 0.5 * (wl_lo + wl_hi)
        if info is not None:
            rms_nm = float(info.get("rms_inliers", info.get("rms", 0.0)))
            max_err_nm = float(info.get("max_error", 0.0))
            n_in = int(info.get("n_inliers", 0)); n_out = int(info.get("n_outliers", 0))
        else:
            pairs = [(p, w) for p, w in zip(self.peak_pixels, self.known_wl) if w and w > 0]
            if pairs:
                pix = np.array([pp[0] for pp in pairs]); wl = np.array([pp[1] for pp in pairs])
                n_pixels = len(self.spectrum_y) if self.spectrum_y is not None else None
                t = dsp.normalize_pixels(pix, n_pixels=n_pixels)
                residuals = legendre.legval(t, coeffs_arr) - wl
                rms_nm = float(np.sqrt(np.mean(residuals ** 2)))
                max_err_nm = float(np.max(np.abs(residuals))); n_in = len(pairs); n_out = 0
            else:
                rms_nm = max_err_nm = 0.0; n_in = n_out = 0
        rms_cm = rms_nm * 1.0e7 / (avg_wl * avg_wl) if avg_wl > 0 else 0.0
        max_err_cm = max_err_nm * 1.0e7 / (avg_wl * avg_wl) if avg_wl > 0 else 0.0
        if rms_nm < 0.05:
            verdict, color = "✓✓ excellent fit", _GREEN
        elif rms_nm < 0.2:
            verdict, color = "✓ good fit", _GREEN
        elif rms_nm < 0.5:
            verdict, color = "⚠ acceptable — check residuals", _AMBER
        else:
            verdict, color = "✗ POOR fit — see red rows above", _RED
        out_text = f", {n_out} outlier{'s' if n_out != 1 else ''}" if n_out else ""
        self.quality_label.setText(
            f"{verdict}\nRMS = {rms_nm:.3f} nm  (~{rms_cm:.1f} cm⁻¹)    "
            f"max Δ = {max_err_nm:.3f} nm  (~{max_err_cm:.1f} cm⁻¹)    "
            f"{n_in} inlier{'s' if n_in != 1 else ''}{out_text}\n"
            f"Axis preview @ {laser:.2f} nm laser:  {wl_lo:.2f}–{wl_hi:.2f} nm   →   "
            f"{cm_lo:.0f} – {cm_hi:.0f} cm⁻¹")
        self.quality_label.setStyleSheet(f"color:{color}; background:#0d2840; padding:6px; border-radius:3px;")

    # ── QTreeWidget helpers ─────────────────────────────────────────────────
    def _append_row(self, values, color):
        item = QTreeWidgetItem([str(v) for v in values])
        for c in range(4):
            item.setForeground(c, QColor(color))
            item.setTextAlignment(c, Qt.AlignCenter)
        self.tree.addTopLevelItem(item)

    def _set_row(self, idx, values, color):
        item = self.tree.topLevelItem(idx)
        if item is None:
            return
        for c, v in enumerate(values):
            item.setText(c, str(v)); item.setForeground(c, QColor(color))

    def _selected_index(self):
        items = self.tree.selectedItems()
        if not items:
            return None
        return self.tree.indexOfTopLevelItem(items[0])


class SampleCalibrationDialogQt(QDialog):
    """One-click calibration from a known liquid (IPA, ethanol, …)."""

    def __init__(self, parent, spectrum_y, on_solution, laser_nm=532.0, current_coeffs=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate from a Known Sample")
        self.on_solution = on_solution
        self.spectrum_y = np.asarray(spectrum_y, dtype=float) if spectrum_y is not None else None
        self.laser_nm = float(laser_nm)
        self.current_coeffs = list(current_coeffs) if current_coeffs is not None else None
        self.pairs = []
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        title = QLabel("Calibrate from a Known Sample")
        title.setStyleSheet(f"color:{_CYAN}; font-size:15px; font-weight:bold;")
        v.addWidget(title)
        intro = QLabel("Put a PURE liquid in the cuvette, capture its spectrum, then pick what's "
                       "in it below. The app matches the strongest peaks to that liquid's known "
                       "Raman shifts and fits the calibration automatically.\n\n"
                       "No Neon or Mercury-Argon lamp needed.")
        intro.setWordWrap(True); v.addWidget(intro)

        top = QGroupBox(); top_l = QHBoxLayout(top)
        self.sample_combo = QComboBox(); self.sample_combo.addItems(list(dsp.RAMAN_STANDARDS.keys()))
        self.sample_combo.currentTextChanged.connect(self._on_sample_change)
        self.laser_entry = QLineEdit(f"{self.laser_nm:.2f}"); self.laser_entry.setFixedWidth(80)
        top_l.addWidget(QLabel("What's in the cuvette?")); top_l.addWidget(self.sample_combo)
        top_l.addSpacing(12); top_l.addWidget(QLabel("Laser (nm):")); top_l.addWidget(self.laser_entry)
        top_l.addStretch(1)
        v.addWidget(top)

        self.expected_label = QLabel(""); self.expected_label.setWordWrap(True)
        self.expected_label.setStyleSheet(f"color:{_MUTED};")
        v.addWidget(self.expected_label)

        b_detect = QPushButton("🔍  Detect peaks & match")
        b_detect.setStyleSheet(f"background:{_CYAN}; color:#000; font-weight:bold; padding:6px 14px;")
        b_detect.clicked.connect(self._detect_and_match)
        v.addWidget(b_detect)

        v.addWidget(QLabel("Peak assignments"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Pixel", "Current cm⁻¹", "Expected cm⁻¹", "Δ cm⁻¹"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree.setMinimumHeight(180)
        v.addWidget(self.tree)

        b_remove = QPushButton("✕ Remove selected"); b_remove.clicked.connect(self._remove_selected)
        b_remove.setStyleSheet(f"color:{_RED};")
        v.addWidget(b_remove, alignment=Qt.AlignLeft)

        self.status_label = QLabel("Capture a clean spectrum of the pure sample, then click "
                                   "'Detect peaks & match'.")
        self.status_label.setWordWrap(True); self.status_label.setStyleSheet(f"color:{_GREEN};")
        v.addWidget(self.status_label)

        actions = QHBoxLayout()
        self.apply_btn = QPushButton("✓ Apply Calibration"); self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet(f"background:{_CYAN}; color:#000; font-weight:bold; padding:6px 16px;")
        self.apply_btn.clicked.connect(self._apply)
        b_cancel = QPushButton("Cancel"); b_cancel.clicked.connect(self.reject)
        actions.addStretch(1); actions.addWidget(self.apply_btn); actions.addWidget(b_cancel)
        v.addLayout(actions)
        self._on_sample_change()

    def _on_sample_change(self, *_):
        shifts = dsp.RAMAN_STANDARDS.get(self.sample_combo.currentText(), [])
        if shifts:
            self.expected_label.setText(f"Expected peaks for {self.sample_combo.currentText()}:  "
                                        + ", ".join(f"{s:.0f}" for s in shifts) + " cm⁻¹")
        else:
            self.expected_label.setText("")

    def _laser(self):
        try:
            v = float(self.laser_entry.text())
            if not (200 < v < 2000):
                raise ValueError
            return v
        except ValueError:
            raise ValueError("Laser wavelength must be a number between 200 and 2000 nm.")

    def _detect_and_match(self):
        if self.spectrum_y is None or self.spectrum_y.size == 0:
            QMessageBox.information(self, "No spectrum",
                "Capture a spectrum of the pure sample first, then come back here.")
            return
        try:
            laser = self._laser()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid laser", str(e)); return
        expected = list(dsp.RAMAN_STANDARDS.get(self.sample_combo.currentText(), []))
        if not expected:
            QMessageBox.critical(self, "No reference", "No peak data is available for the selected sample.")
            return
        y = np.nan_to_num(self.spectrum_y, nan=0.0, posinf=0.0, neginf=0.0)
        y = y - np.min(y)
        prom = max(float(np.max(y)) * 0.03, 1.0)
        peaks, props = find_peaks(y, prominence=prom, distance=8)
        if len(peaks) == 0:
            self.status_label.setText("No peaks detected. Try a longer exposure or a more "
                                      "concentrated sample.")
            return
        order = np.argsort(props["prominences"])[::-1]
        keep = min(len(peaks), max(len(expected) + 2, 6))
        strong = peaks[order[:keep]]
        refined = dsp.refine_peak_positions(y, strong)
        n_pixels = len(y)
        coeffs = (np.array([650.0, 150.0, 0.0, 0.0]) if self.current_coeffs is None
                  else np.asarray(self.current_coeffs, dtype=float))
        self.pairs = dsp.match_peaks_to_standard(refined, coeffs, expected, laser,
                                                 n_pixels=n_pixels, search_window_cm=600.0)
        self._populate_tree()
        if len(self.pairs) < 3:
            self.status_label.setText(f"Only {len(self.pairs)} peaks could be matched — need at "
                                      "least 3 for a linear fit (4 for cubic). Try a cleaner "
                                      "spectrum or a different sample.")
            self.apply_btn.setEnabled(False)
            return
        rms_cm = self._estimate_residual()
        self.status_label.setText(f"Matched {len(self.pairs)} peaks to {self.sample_combo.currentText()}. "
                                  f"Current-calibration residual ≈ {rms_cm:.0f} cm⁻¹. Apply to fix the axis.")
        self.apply_btn.setEnabled(True)

    def _estimate_residual(self):
        if not self.pairs or self.current_coeffs is None:
            return 0.0
        pix = np.array([p[0] for p in self.pairs], dtype=float)
        exp_cm = np.array([p[1] for p in self.pairs], dtype=float)
        n = len(self.spectrum_y) if self.spectrum_y is not None else None
        t = dsp.normalize_pixels(pix, n_pixels=n)
        wl_now = legendre.legval(t, np.asarray(self.current_coeffs, dtype=float))
        cm_now = dsp.wavelengths_to_raman(wl_now, self._laser())
        return float(np.sqrt(np.mean((cm_now - exp_cm) ** 2)))

    def _populate_tree(self):
        self.tree.clear()
        if not self.pairs:
            return
        try:
            laser = self._laser()
        except ValueError:
            laser = self.laser_nm
        coeffs = np.asarray(self.current_coeffs, dtype=float) if self.current_coeffs is not None else None
        n = len(self.spectrum_y) if self.spectrum_y is not None else None
        for pix, exp_cm in self.pairs:
            if coeffs is not None:
                t = dsp.normalize_pixels([pix], n_pixels=n)
                wl_now = float(legendre.legval(t, coeffs))
                cm_now = float(dsp.wavelengths_to_raman(np.array([wl_now]), laser)[0])
                vals = (f"{pix:.1f}", f"{cm_now:+.1f}", f"{exp_cm:.1f}", f"{cm_now - exp_cm:+.1f}")
            else:
                vals = (f"{pix:.1f}", "—", f"{exp_cm:.1f}", "—")
            item = QTreeWidgetItem([str(x) for x in vals])
            for c in range(4):
                item.setTextAlignment(c, Qt.AlignCenter)
            self.tree.addTopLevelItem(item)

    def _remove_selected(self):
        rows = sorted((self.tree.indexOfTopLevelItem(i) for i in self.tree.selectedItems()), reverse=True)
        for i in rows:
            if 0 <= i < len(self.pairs):
                del self.pairs[i]; self.tree.takeTopLevelItem(i)
        if len(self.pairs) < 3:
            self.apply_btn.setEnabled(False)
            self.status_label.setText(f"{len(self.pairs)} pair(s) left — need at least 3 to fit.")

    def _apply(self):
        if len(self.pairs) < 3:
            QMessageBox.critical(self, "Not enough pairs", "Need at least 3 matched peaks to fit a calibration.")
            return
        try:
            laser = self._laser()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid laser", str(e)); return
        n_pixels = len(self.spectrum_y) if self.spectrum_y is not None else None
        pix = [p[0] for p in self.pairs]
        wl = [dsp.raman_shift_to_wavelength(p[1], laser) for p in self.pairs]
        degree = 3 if len(self.pairs) >= 4 else 1
        try:
            coeffs = dsp.fit_calibration(pix, wl, degree=degree, n_pixels=n_pixels)
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e)); return
        if n_pixels:
            diffs = np.diff(dsp.pixels_to_wavelengths(coeffs, n_pixels))
            if not (np.all(diffs > 0) or np.all(diffs < 0)):
                if QMessageBox.question(
                        self, "Suspicious fit",
                        "The fitted calibration is non-monotonic, meaning the spectrum would fold "
                        "back on itself. This usually means a peak got assigned to the wrong "
                        "reference position.\n\nApply anyway?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                    return
        rms = dsp.calibration_pair_rms(coeffs, pix, wl, n_pixels=n_pixels)
        self.on_solution(coeffs)
        self.status_label.setText(f"✓ Calibration applied (degree {degree}, RMS {rms:.3f} nm). "
                                  "The spectrum will reload.")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(800, self.accept)
