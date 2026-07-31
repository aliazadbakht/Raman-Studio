# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V. (https://wfront.nl)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Wavelength calibration dialogs."""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.signal import find_peaks
from . import dsp


# Shared color palette
_BG = "#1a1a2e"
_FRAME = "#16213e"
_ACCENT = "#0f3460"
_TEXT = "#e0e0e0"
_CYAN = "#00d4ff"
_GREEN = "#aaffaa"
_RED = "#ff6b6b"
_MUTED = "#888888"


class CalibrationDialog(tk.Toplevel):
    def __init__(self, parent, spectrum_y, on_solution,
                 on_load_camera=None, on_copy_to_camera=None, laser_nm=532.0):
        super().__init__(parent)
        self.title("Wavelength Calibration")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")
        self.on_solution = on_solution
        self.on_load_camera = on_load_camera
        self.on_copy_to_camera = on_copy_to_camera
        self.laser_nm = float(laser_nm)
        self.spectrum_y = spectrum_y
        self.peak_pixels = []
        self.known_wl = []
        self._auto_assigned = False
        self._build_ui()
        self.grab_set()

    def _build_ui(self):
        style_bg = "#1a1a2e"
        style_frame = "#16213e"
        style_accent = "#0f3460"
        style_text = "#e0e0e0"
        style_cyan = "#00d4ff"

        pad = dict(padx=10, pady=6)

        # Title
        tk.Label(self, text="Wavelength Calibration", font=("Helvetica", 14, "bold"),
                 bg=style_bg, fg=style_cyan).pack(**pad)

        # Tip banner — point users to Quick Cal for the common case
        tip = tk.Frame(self, bg="#0d2840", padx=10, pady=6)
        tip.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(tip,
                 text="💡  Easier option: close this and click 🧪 Quick Cal "
                      "if you have a pure liquid (IPA, ethanol, cyclohexane). "
                      "This dialog is for Neon / Mercury-Argon lamp spectra.",
                 bg="#0d2840", fg="#aaffaa",
                 font=("Helvetica", 9), justify="left", wraplength=520
                 ).pack(anchor="w")

        # Source selection
        src_frame = tk.Frame(self, bg=style_frame, padx=10, pady=8)
        src_frame.pack(fill="x", padx=10, pady=4)
        tk.Label(src_frame, text="Calibration Source:", bg=style_frame, fg=style_text,
                 font=("Helvetica", 10)).grid(row=0, column=0, sticky="w")
        self.source_var = tk.StringVar(value="Neon")
        src_combo = ttk.Combobox(src_frame, textvariable=self.source_var,
                                  values=["Neon", "Mercury-Argon"], state="readonly", width=18)
        src_combo.grid(row=0, column=1, padx=8)
        src_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_source_change())

        tk.Label(src_frame, text="Model:", bg=style_frame, fg=style_text,
                 font=("Helvetica", 10)).grid(row=0, column=2, padx=(14, 4))
        self.model_var = tk.StringVar(value="Cubic")
        ttk.Combobox(src_frame, textvariable=self.model_var,
                     values=["Linear", "Cubic"], state="readonly", width=10).grid(row=0, column=3)

        # Sensitivity
        sens_frame = tk.Frame(self, bg=style_frame, padx=10, pady=8)
        sens_frame.pack(fill="x", padx=10, pady=2)
        tk.Label(sens_frame, text="Peak Sensitivity:", bg=style_frame, fg=style_text,
                 font=("Helvetica", 10)).grid(row=0, column=0, sticky="w")
        self.sens_var = tk.DoubleVar(value=0.7)
        self.sens_label = tk.Label(sens_frame, text="70%", bg=style_frame, fg=style_cyan,
                                    font=("Helvetica", 10, "bold"), width=5)
        self.sens_label.grid(row=0, column=2, padx=6)
        sens_slider = ttk.Scale(sens_frame, from_=0.1, to=1.0, variable=self.sens_var,
                                 orient="horizontal", length=200,
                                 command=lambda v: self.sens_label.config(
                                     text=f"{float(v)*100:.0f}%"))
        sens_slider.grid(row=0, column=1, padx=8)

        tk.Label(sens_frame, text="Max Peaks:", bg=style_frame, fg=style_text,
                 font=("Helvetica", 10)).grid(row=0, column=3, padx=(14, 4))
        self.npeaks_var = tk.IntVar(value=len(dsp.NEON_LINES))
        ttk.Spinbox(sens_frame, from_=2, to=30, textvariable=self.npeaks_var, width=5).grid(
            row=0, column=4)

        # Advanced search constraints — hidden by default. Most users only
        # need source + sensitivity, so collapse the wavelength/span/distortion
        # controls behind a toggle.
        adv_header = tk.Frame(self, bg=style_bg)
        adv_header.pack(fill="x", padx=10, pady=(6, 0))
        self._adv_open = tk.BooleanVar(value=False)
        self._adv_btn = tk.Button(
            adv_header, text="▸  Advanced search constraints",
            command=self._toggle_advanced,
            bg=style_bg, fg="#aaaaaa", relief="flat",
            font=("Helvetica", 9), anchor="w", padx=4
        )
        self._adv_btn.pack(side="left")

        param_frame = tk.Frame(self, bg=style_frame, padx=10, pady=8)
        # NOT packed initially — _toggle_advanced controls visibility.
        self._adv_frame = param_frame
        self.range_min_var = tk.StringVar(value="500")
        self.range_max_var = tk.StringVar(value="800")
        self.span_min_var = tk.StringVar(value="100")
        self.span_max_var = tk.StringVar(value="150")
        self.distortion_max_var = tk.StringVar(value="10")
        self.sampling_var = tk.StringVar(value="10")

        tk.Label(param_frame, text="Wavelength:", bg=style_frame, fg=style_text).grid(
            row=0, column=0, sticky="w")
        ttk.Spinbox(param_frame, from_=200, to=1100, increment=1,
                    textvariable=self.range_min_var, width=6).grid(row=0, column=1, padx=(4, 2))
        ttk.Spinbox(param_frame, from_=200, to=1100, increment=1,
                    textvariable=self.range_max_var, width=6).grid(row=0, column=2, padx=(2, 8))

        tk.Label(param_frame, text="Span:", bg=style_frame, fg=style_text).grid(
            row=0, column=3, sticky="w")
        ttk.Spinbox(param_frame, from_=1, to=500, increment=1,
                    textvariable=self.span_min_var, width=6).grid(row=0, column=4, padx=(4, 2))
        ttk.Spinbox(param_frame, from_=1, to=500, increment=1,
                    textvariable=self.span_max_var, width=6).grid(row=0, column=5, padx=(2, 8))

        tk.Label(param_frame, text="Dist:", bg=style_frame, fg=style_text).grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(param_frame, from_=0, to=100, increment=1,
                    textvariable=self.distortion_max_var, width=6).grid(
                        row=1, column=1, padx=(4, 2), pady=(6, 0))
        tk.Label(param_frame, text="Search:", bg=style_frame, fg=style_text).grid(
            row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Spinbox(param_frame, from_=3, to=30, increment=1,
                    textvariable=self.sampling_var, width=6).grid(
                        row=1, column=4, padx=(4, 2), pady=(6, 0))

        self.lock_raman_range_var = tk.BooleanVar(value=True)
        self.raman_min_var = tk.StringVar(value="500")
        self.raman_max_var = tk.StringVar(value="3500")
        ttk.Checkbutton(param_frame, variable=self.lock_raman_range_var).grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        tk.Label(param_frame, text="Lock Raman:", bg=style_frame, fg=style_text).grid(
            row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Spinbox(param_frame, from_=0, to=5000, increment=10,
                    textvariable=self.raman_min_var, width=6).grid(
                        row=2, column=2, padx=(2, 4), pady=(6, 0))
        ttk.Spinbox(param_frame, from_=0, to=5000, increment=10,
                    textvariable=self.raman_max_var, width=6).grid(
                        row=2, column=3, padx=(2, 4), pady=(6, 0))

        # Peak list
        list_frame = tk.Frame(self, bg=style_bg)
        list_frame.pack(fill="both", expand=True, padx=10, pady=4)
        tk.Label(list_frame, text="Matched Peaks  (Pixel → Wavelength nm, residual after fit)",
                 bg=style_bg, fg=style_text, font=("Helvetica", 10, "bold")).pack(anchor="w")

        cols = ("pixel", "wavelength", "delta_nm", "delta_cm")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=8)
        self.tree.heading("pixel", text="Pixel")
        self.tree.heading("wavelength", text="Wavelength (nm)")
        self.tree.heading("delta_nm", text="Δ (nm)")
        self.tree.heading("delta_cm", text="Δ (cm⁻¹)")
        self.tree.column("pixel", width=70, anchor="center")
        self.tree.column("wavelength", width=140, anchor="center")
        self.tree.column("delta_nm", width=70, anchor="center")
        self.tree.column("delta_cm", width=70, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        # Row tags for color-coded fit quality
        self.tree.tag_configure("inlier",   foreground=style_text)
        self.tree.tag_configure("borderline", foreground="#ffdd57")
        self.tree.tag_configure("outlier",  foreground="#ff6b6b")
        self.tree.tag_configure("manual",   foreground="#aaffaa")
        self.tree.tag_configure("auto",     foreground="#888888")

        # Add / remove row
        edit_frame = tk.Frame(self, bg=style_bg)
        edit_frame.pack(fill="x", padx=10, pady=4)
        tk.Label(edit_frame, text="Pixel:", bg=style_bg, fg=style_text).pack(side="left")
        self.pix_entry = tk.Entry(edit_frame, width=8, bg="#0f3460", fg=style_text,
                                   insertbackground=style_cyan)
        self.pix_entry.pack(side="left", padx=4)
        tk.Label(edit_frame, text="Wavelength (nm):", bg=style_bg, fg=style_text).pack(side="left")
        self.wl_entry = tk.Entry(edit_frame, width=10, bg="#0f3460", fg=style_text,
                                  insertbackground=style_cyan)
        self.wl_entry.pack(side="left", padx=4)
        tk.Button(edit_frame, text="Add", command=self._add_pair,
                  bg=style_accent, fg=style_cyan, relief="flat", padx=8).pack(side="left", padx=4)
        tk.Button(edit_frame, text="Update", command=self._update_pair,
                  bg=style_accent, fg="#aaffaa", relief="flat", padx=8).pack(side="left", padx=4)
        tk.Button(edit_frame, text="Remove", command=self._remove_pair,
                  bg="#3a0d0d", fg="#ff6b6b", relief="flat", padx=8).pack(side="left")

        # Auto-detect peaks button
        det_frame = tk.Frame(self, bg=style_bg)
        det_frame.pack(fill="x", padx=10, pady=2)
        tk.Button(det_frame, text="Auto-Detect Lamp Peaks",
                  command=self._auto_detect, bg=style_accent, fg=style_cyan,
                  relief="flat", padx=10, pady=4).pack(side="left")

        # Camera calibration memory actions. These mirror the original
        # OpenRAMAN software's load/upload calibration behavior and bypass the
        # peak-fitting path entirely.
        cam_frame = tk.Frame(self, bg=style_bg)
        cam_frame.pack(fill="x", padx=10, pady=(4, 2))
        tk.Button(cam_frame, text="Hard Load from Camera",
                  command=self._hard_load_from_camera,
                  bg=style_accent, fg="#aaffaa", relief="flat",
                  padx=10, pady=4).pack(side="left", padx=(0, 6))
        tk.Button(cam_frame, text="Hard Copy to Camera",
                  command=self._hard_copy_to_camera,
                  bg=style_accent, fg="#ffdd57", relief="flat",
                  padx=10, pady=4).pack(side="left")

        # Fit-quality summary panel: shows fit RMS in both nm and cm⁻¹, the
        # cm⁻¹ axis preview, and inlier counts. Populated by _calibrate(),
        # and used by the user to spot a bad fit before applying.
        self._quality_frame = tk.Frame(self, bg="#0d2840", padx=10, pady=6)
        self._quality_frame.pack(fill="x", padx=10, pady=(4, 2))
        self.quality_var = tk.StringVar(
            value="Fit quality will appear here once you click Calibrate."
        )
        self.quality_label = tk.Label(self._quality_frame, textvariable=self.quality_var,
                                       bg="#0d2840", fg="#888888",
                                       font=("Helvetica", 9), justify="left",
                                       wraplength=520, anchor="w")
        self.quality_label.pack(anchor="w")

        # Status label
        self.status_var = tk.StringVar(
            value="Automatic calibration requires a Neon/Hg-Ar lamp spectrum, not a sample spectrum."
        )
        tk.Label(self, textvariable=self.status_var, bg=style_bg, fg="#aaaaaa",
                 font=("Helvetica", 9), wraplength=450).pack(padx=10, pady=2)

        # Buttons
        btn_frame = tk.Frame(self, bg=style_bg)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Calibrate", command=self._calibrate,
                  bg="#00d4ff", fg="#000000", font=("Helvetica", 11, "bold"),
                  relief="flat", padx=20, pady=6).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Close", command=self.destroy,
                  bg="#333", fg=style_text, relief="flat", padx=16, pady=6).pack(side="left")

    def _toggle_advanced(self):
        if self._adv_open.get():
            self._adv_frame.pack_forget()
            self._adv_btn.config(text="▸  Advanced search constraints")
            self._adv_open.set(False)
        else:
            # Pack right after the Advanced header so it appears in place
            self._adv_frame.pack(fill="x", padx=10, pady=2,
                                 after=self._adv_btn.master)
            self._adv_btn.config(text="▾  Advanced search constraints")
            self._adv_open.set(True)

    def _source_lines(self):
        return dsp.NEON_LINES if self.source_var.get() == "Neon" else dsp.MERCURY_ARGON_LINES

    def _on_source_change(self):
        self.npeaks_var.set(len(self._source_lines()))
        self.status_var.set(
            f"{self.source_var.get()} selected. Capture that lamp spectrum before auto calibration."
        )

    def _auto_params(self):
        try:
            params = {
                "range_min": float(self.range_min_var.get()),
                "range_max": float(self.range_max_var.get()),
                "span_min": float(self.span_min_var.get()),
                "span_max": float(self.span_max_var.get()),
                "distortion_max": float(self.distortion_max_var.get()),
                "sampling": int(float(self.sampling_var.get())),
            }
            if self.lock_raman_range_var.get():
                raman_min = float(self.raman_min_var.get())
                raman_max = float(self.raman_max_var.get())
                endpoint_min = self._raman_shift_to_wavelength(raman_min)
                endpoint_max = self._raman_shift_to_wavelength(raman_max)
                params.update({
                    "endpoint_min": min(endpoint_min, endpoint_max),
                    "endpoint_max": max(endpoint_min, endpoint_max),
                    "endpoint_weight": 10.0,
                })
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
            p = float(self.pix_entry.get())
            w = float(self.wl_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Enter valid numbers for pixel and wavelength.")
            return

        # Smart Add: If pixel already in list (within 1px), update its wavelength instead of adding new
        for i, pix in enumerate(self.peak_pixels):
            if abs(pix - p) < 1.0:
                self.known_wl[i] = w
                for item in self.tree.get_children():
                    if abs(float(self.tree.item(item, 'values')[0]) - p) < 1.0:
                        self.tree.item(item, values=(f"{p:.1f}", f"{w:.3f}", "—", "—"),
                                       tags=("manual",))
                        break
                self.status_var.set(f"Updated peak at pixel {p:.1f}")
                return

        self.peak_pixels.append(p)
        self.known_wl.append(w)
        self._auto_assigned = False
        self.tree.insert("", "end", values=(f"{p:.1f}", f"{w:.3f}", "—", "—"),
                         tags=("manual",))
        self.pix_entry.delete(0, "end")
        self.wl_entry.delete(0, "end")

    def _remove_pair(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        self.tree.delete(sel[0])
        del self.peak_pixels[idx]
        del self.known_wl[idx]

    def _update_pair(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select Row", "Select a peak from the list first.")
            return
        try:
            w = float(self.wl_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Enter a valid wavelength (nm).")
            return
        
        idx = self.tree.index(sel[0])
        self.known_wl[idx] = w
        self._auto_assigned = False
        p = self.peak_pixels[idx]
        self.tree.item(sel[0], values=(f"{p:.1f}", f"{w:.3f}", "—", "—"),
                       tags=("manual",))
        self.status_var.set(f"Updated Pixel {p:.1f} → {w:.3f} nm")
        self.wl_entry.delete(0, "end") # Clear after update

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        self.pix_entry.delete(0, "end")
        self.pix_entry.insert(0, f"{self.peak_pixels[idx]:.1f}")
        self.wl_entry.delete(0, "end")
        w = self.known_wl[idx]
        if w > 0:
            self.wl_entry.insert(0, f"{w:.3f}")

    def _auto_detect(self):
        if self.spectrum_y is None:
            messagebox.showinfo("No Spectrum", "Load or acquire a spectrum first.")
            return
        y = self.spectrum_y
        peaks = dsp.detect_calibration_peaks(
            y,
            n_peaks=self.npeaks_var.get(),
            sensitivity=self.sens_var.get(),
        )

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.peak_pixels.clear()
        self.known_wl.clear()
        self._auto_assigned = False

        if len(peaks) == 0:
            self.status_var.set("No calibration peaks detected. Increase sensitivity or exposure.")
            return

        for p in peaks:
            self.peak_pixels.append(float(p))
            self.known_wl.append(0.0)
            self.tree.insert("", "end",
                             values=(f"{p:.1f}", "auto", "—", "—"),
                             tags=("auto",))

        self.status_var.set(
            f"Detected {len(peaks)} lamp candidate peaks. Click Calibrate for automatic "
            f"{self.source_var.get()} source-line matching, or enter known wavelengths manually."
        )

    def _format_coeffs(self, coeffs):
        return ", ".join(f"{float(c):.6g}" for c in coeffs)

    def _hard_load_from_camera(self):
        if self.on_load_camera is None:
            messagebox.showinfo("Camera Calibration", "Camera calibration loading is not available.")
            return
        try:
            coeffs = self.on_load_camera()
        except Exception as e:
            messagebox.showerror("Camera Calibration", str(e))
            return
        self.status_var.set(
            "Hard-loaded OpenRAMAN calibration from camera. "
            f"Coefficients: {self._format_coeffs(coeffs)}"
        )

    def _hard_copy_to_camera(self):
        if self.on_copy_to_camera is None:
            messagebox.showinfo("Camera Calibration", "Camera calibration upload is not available.")
            return
        if not messagebox.askyesno(
            "Copy Calibration to Camera",
            "This overwrites the OpenRAMAN calibration stored in camera memory "
            "with the current Mac calibration.\n\nContinue?",
            default="no",
        ):
            return
        try:
            coeffs = self.on_copy_to_camera()
        except Exception as e:
            messagebox.showerror("Camera Calibration", str(e))
            return
        self.status_var.set(
            "Hard-copied current calibration into camera memory. "
            f"Coefficients: {self._format_coeffs(coeffs)}"
        )

    def _auto_calibration_warning(self, info):
        residuals = np.asarray(info.get("residuals", []), dtype=float)
        rms = float(info.get("rms_inliers", info.get("rms", 0.0)))
        max_error = float(info.get("max_error", 0.0))
        unique_lines = int(info.get("unique_lines", 0))
        line_span = float(info.get("line_span", 0.0))
        endpoint_min = info.get("endpoint_min")
        endpoint_max = info.get("endpoint_max")
        left_endpoint = info.get("left_endpoint")
        right_endpoint = info.get("right_endpoint")
        n_inliers = int(info.get("n_inliers", 0))
        n_outliers = int(info.get("n_outliers", 0))
        n_lines = n_inliers + n_outliers if (n_inliers + n_outliers) else len(residuals)
        min_unique = 3 if self.model_var.get() == "Linear" else 5

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
        if endpoint_min is not None and left_endpoint is not None:
            if abs(float(left_endpoint) - float(endpoint_min)) > 2.0:
                issues.append(
                    f"left endpoint is {float(left_endpoint):.1f} nm, target {float(endpoint_min):.1f} nm"
                )
        if endpoint_max is not None and right_endpoint is not None:
            if abs(float(right_endpoint) - float(endpoint_max)) > 2.0:
                issues.append(
                    f"right endpoint is {float(right_endpoint):.1f} nm, target {float(endpoint_max):.1f} nm"
                )

        summary = (
            f"{self.source_var.get()} auto-fit diagnostics:\n"
            f"  peaks: {n_lines}  ({n_inliers} matched, {n_outliers} unmatched)\n"
            f"  unique source lines: {unique_lines}\n"
            f"  source-line span: {line_span:.1f} nm\n"
            f"  RMS residual: {rms:.3f} nm\n"
            f"  largest residual: {max_error:.3f} nm\n\n"
            "Automatic calibration must be run on a real calibration source spectrum. "
            "If this was an IPA/sample spectrum, applying it will force sample peaks "
            "onto lamp lines and the Raman shifts will be wrong."
        )
        if endpoint_min is not None and endpoint_max is not None:
            summary += (
                f"\n\nEndpoint targets: {float(endpoint_min):.3f}-"
                f"{float(endpoint_max):.3f} nm"
            )
            if left_endpoint is not None and right_endpoint is not None:
                summary += (
                    f"\nFit endpoints: {float(left_endpoint):.3f}-"
                    f"{float(right_endpoint):.3f} nm"
                )
        if issues:
            summary += "\n\nThis fit is suspicious: " + "; ".join(issues) + "."
        summary += "\n\nApply this automatic calibration?"
        return summary

    def _calibrate(self):
        if not self.peak_pixels:
            messagebox.showerror("Error", "Auto-detect peaks or add calibration pairs first.")
            return

        degree = 3 if self.model_var.get() == "Cubic" else 1
        n_pixels = len(self.spectrum_y) if self.spectrum_y is not None else None

        # Wavelengths entered by the user are treated as explicit manual pairs.
        # Rows left as "auto" use the source-line matching optimizer.
        valid_pix = []
        valid_wl = []
        for p, w in zip(self.peak_pixels, self.known_wl):
            if w > 0:
                valid_pix.append(p)
                valid_wl.append(w)

        mode = "manual"
        rms = None
        info = None
        try:
            manual_required = degree + 1
            use_auto_mode = len(valid_pix) == 0 or self._auto_assigned
            if len(valid_pix) >= manual_required and not use_auto_mode:
                coeffs = dsp.fit_calibration(valid_pix, valid_wl, degree=degree, n_pixels=n_pixels)
                rms = dsp.calibration_pair_rms(coeffs, valid_pix, valid_wl, n_pixels=n_pixels)
            elif use_auto_mode:
                mode = "auto"
                coeffs, info = dsp.fit_calibration_to_source(
                    self.peak_pixels,
                    self._source_lines(),
                    model=self.model_var.get(),
                    n_pixels=n_pixels,
                    **self._auto_params(),
                )
                rms = info["rms_inliers"]
                # Populate known_wl from the (dummy-aware) assignment. Outliers
                # have nan in `nearest` — record them as 0.0 so the row still
                # shows up but doesn't pollute future manual fits.
                items = list(self.tree.get_children())
                for idx, nearest in enumerate(info["nearest"]):
                    if np.isfinite(nearest):
                        self.known_wl[idx] = float(nearest)
                    else:
                        self.known_wl[idx] = 0.0
                self._auto_assigned = True
                # Populate residuals into the table BEFORE the confirm prompt,
                # so the user sees per-peak quality while deciding to apply.
                self._populate_residuals_in_table(coeffs, info=info)
                self._show_quality_summary(coeffs, info=info)
                default_choice = "no" if (info.get("rms_inliers", 0.0) > 0.5
                                          or info.get("n_outliers", 0) > 0) else "yes"
                if not messagebox.askyesno(
                    "Apply automatic calibration?",
                    self._auto_calibration_warning(info),
                    default=default_choice,
                ):
                    return
            else:
                messagebox.showerror(
                    "Error",
                    f"Need at least {manual_required} manual wavelength pairs for "
                    f"{self.model_var.get()} fitting, or leave all wavelengths as auto."
                )
                return
        except Exception as e:
            messagebox.showerror("Calibration Failed", str(e))
            return

        # Sanity check: the resulting pixel→wavelength polynomial must be
        # strictly monotonic across the detector. A wiggly cubic fit causes
        # the plotted spectrum to fold back on itself (two y-values per cm⁻¹).
        if n_pixels:
            wl = dsp.pixels_to_wavelengths(coeffs, n_pixels)
            diffs = np.diff(wl)
            monotonic = bool(np.all(diffs > 0) or np.all(diffs < 0))
            if not monotonic:
                # Find where the slope flips sign
                signs = np.sign(diffs)
                flips = np.where(signs[:-1] != signs[1:])[0]
                bad = int(flips[0]) if len(flips) else 0
                wl_range = (float(np.min(wl)), float(np.max(wl)))
                msg = (
                    "Calibration polynomial is non-monotonic.\n\n"
                    f"It produces wavelengths in the range "
                    f"{wl_range[0]:.1f}–{wl_range[1]:.1f} nm but folds back on "
                    f"itself near pixel {bad}. The plot will show two values per "
                    f"cm⁻¹ wherever it folds.\n\n"
                    "Common causes:\n"
                    " • One peak assigned to the wrong neon/argon line\n"
                    " • Too-high polynomial degree for the points you have\n"
                    " • Points all clustered in a narrow pixel range\n\n"
                    "Try: switch Model to Linear, or re-pick peaks more spread out "
                    "and check each nm assignment.\n\n"
                    "Apply this calibration anyway?"
                )
                if not messagebox.askyesno("Non-monotonic calibration", msg, default="no"):
                    return
        # Final residual populate + quality refresh, for manual mode (auto
        # mode already did this before the confirm prompt).
        if mode != "auto":
            self._populate_residuals_in_table(coeffs, info=None)
            self._show_quality_summary(coeffs, info=None)
        self.on_solution(coeffs)
        rms_text = f", RMS {rms:.3f} nm" if rms is not None else ""
        self.status_var.set(
            f"✓ Calibration OK ({mode}{rms_text}). Coefficients: {[round(c,4) for c in coeffs]}"
        )

    # ── Residual / quality helpers ────────────────────────────────────────

    def _populate_residuals_in_table(self, coeffs, info=None):
        """
        Refresh the table's Δ(nm) and Δ(cm⁻¹) columns under ``coeffs``.

        Each row is also tagged so that the foreground color makes inliers,
        borderline peaks and outliers immediately visible — the user can
        scan the table once and spot a wrong source-line assignment that
        would otherwise hide inside a moderate overall RMS.
        """
        from numpy.polynomial import legendre
        n_pixels = len(self.spectrum_y) if self.spectrum_y is not None else None
        laser = self.laser_nm
        coeffs_arr = np.asarray(coeffs, dtype=float)

        pix_arr = np.asarray(self.peak_pixels, dtype=float)
        if pix_arr.size == 0:
            return
        t = dsp.normalize_pixels(pix_arr, n_pixels=n_pixels)
        projected_wl = legendre.legval(t, coeffs_arr)
        projected_cm = dsp.wavelengths_to_raman(projected_wl, laser)

        items = list(self.tree.get_children())
        inlier_mask = (info.get("inlier_mask") if info is not None else None)
        for i, (item, p, known) in enumerate(zip(items, self.peak_pixels, self.known_wl)):
            row_proj_wl = float(projected_wl[i])
            row_proj_cm = float(projected_cm[i])
            if known and known > 0:
                target_cm = float(dsp.wavelengths_to_raman(np.array([float(known)]), laser)[0])
                delta_nm = row_proj_wl - float(known)
                delta_cm = row_proj_cm - target_cm
                if inlier_mask is not None and i < len(inlier_mask) and not bool(inlier_mask[i]):
                    tag = "outlier"
                elif abs(delta_nm) < 0.05:
                    tag = "inlier"
                elif abs(delta_nm) < 0.3:
                    tag = "borderline"
                else:
                    tag = "outlier"
                self.tree.item(item, values=(
                    f"{p:.1f}",
                    f"{float(known):.3f}",
                    f"{delta_nm:+.3f}",
                    f"{delta_cm:+.1f}",
                ), tags=(tag,))
            else:
                # Unassigned peak (outlier from auto-fit, or never given an nm).
                self.tree.item(item, values=(
                    f"{p:.1f}", "unassigned", "—", "—",
                ), tags=("outlier",))

    def _show_quality_summary(self, coeffs, info=None):
        """Update the dark-blue quality banner above the buttons."""
        from numpy.polynomial import legendre
        laser = self.laser_nm
        coeffs_arr = np.asarray(coeffs, dtype=float)

        # cm⁻¹ axis preview: where the detector endpoints land after this fit.
        wl_left = float(legendre.legval(-1.0, coeffs_arr))
        wl_right = float(legendre.legval(1.0, coeffs_arr))
        cm_left = float(dsp.wavelengths_to_raman(np.array([wl_left]), laser)[0])
        cm_right = float(dsp.wavelengths_to_raman(np.array([wl_right]), laser)[0])
        cm_lo, cm_hi = (cm_left, cm_right) if cm_left < cm_right else (cm_right, cm_left)
        wl_lo, wl_hi = (wl_left, wl_right) if wl_left < wl_right else (wl_right, wl_left)
        avg_wl = 0.5 * (wl_lo + wl_hi)

        if info is not None:
            rms_nm = float(info.get("rms_inliers", info.get("rms", 0.0)))
            max_err_nm = float(info.get("max_error", 0.0))
            n_in = int(info.get("n_inliers", 0))
            n_out = int(info.get("n_outliers", 0))
        else:
            pairs = [(p, w) for p, w in zip(self.peak_pixels, self.known_wl) if w and w > 0]
            if pairs:
                pix = np.array([pp[0] for pp in pairs])
                wl = np.array([pp[1] for pp in pairs])
                n_pixels = len(self.spectrum_y) if self.spectrum_y is not None else None
                t = dsp.normalize_pixels(pix, n_pixels=n_pixels)
                pred = legendre.legval(t, coeffs_arr)
                residuals = pred - wl
                rms_nm = float(np.sqrt(np.mean(residuals ** 2)))
                max_err_nm = float(np.max(np.abs(residuals)))
                n_in = len(pairs)
                n_out = 0
            else:
                rms_nm = max_err_nm = 0.0
                n_in = n_out = 0

        # Convert nm RMS to approximate cm⁻¹ at the mean wavelength using
        # the local derivative of the Raman-shift formula:
        # |d(cm⁻¹)/d(wl)| = 1e7 / wl² .
        rms_cm = rms_nm * 1.0e7 / (avg_wl * avg_wl) if avg_wl > 0 else 0.0
        max_err_cm = max_err_nm * 1.0e7 / (avg_wl * avg_wl) if avg_wl > 0 else 0.0

        if rms_nm < 0.05:
            verdict, color = "✓✓ excellent fit", "#aaffaa"
        elif rms_nm < 0.2:
            verdict, color = "✓ good fit", "#aaffaa"
        elif rms_nm < 0.5:
            verdict, color = "⚠ acceptable — check residuals", "#ffdd57"
        else:
            verdict, color = "✗ POOR fit — see red rows above", "#ff6b6b"

        out_text = f", {n_out} outlier{'s' if n_out != 1 else ''}" if n_out else ""
        msg = (
            f"{verdict}\n"
            f"RMS = {rms_nm:.3f} nm  (~{rms_cm:.1f} cm⁻¹)    "
            f"max Δ = {max_err_nm:.3f} nm  (~{max_err_cm:.1f} cm⁻¹)    "
            f"{n_in} inlier{'s' if n_in != 1 else ''}{out_text}\n"
            f"Axis preview @ {laser:.2f} nm laser:  "
            f"{wl_lo:.2f}–{wl_hi:.2f} nm   →   "
            f"{cm_lo:.0f} – {cm_hi:.0f} cm⁻¹"
        )
        self.quality_var.set(msg)
        self.quality_label.config(fg=color)


class SampleCalibrationDialog(tk.Toplevel):
    """
    One-click calibration from a known liquid in the cuvette.

    The user picks what's in the sample (IPA, ethanol, cyclohexane, ...),
    the app detects the strongest peaks and pairs them with that standard's
    known Raman shifts. No lamp required. Designed to be the default path
    for users who don't have a Neon/Mercury-Argon lamp on hand.
    """

    def __init__(self, parent, spectrum_y, on_solution, laser_nm=532.0,
                 current_coeffs=None):
        super().__init__(parent)
        self.title("Calibrate from a Known Sample")
        self.configure(bg=_BG)
        self.resizable(False, False)
        self.on_solution = on_solution
        self.spectrum_y = (np.asarray(spectrum_y, dtype=float)
                           if spectrum_y is not None else None)
        self.laser_nm = float(laser_nm)
        self.current_coeffs = (list(current_coeffs)
                               if current_coeffs is not None else None)
        self.pairs = []   # list of (pixel, expected_cm) tuples

        self._build_ui()
        self.grab_set()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=12, pady=6)

        tk.Label(self, text="Calibrate from a Known Sample",
                 font=("Helvetica", 14, "bold"), bg=_BG, fg=_CYAN
                 ).pack(**pad)

        intro = (
            "Put a PURE liquid in the cuvette, capture its spectrum, then "
            "pick what's in it below. The app will match the strongest "
            "peaks to that liquid's known Raman shifts and fit the "
            "calibration automatically.\n\n"
            "No Neon or Mercury-Argon lamp needed."
        )
        tk.Label(self, text=intro, bg=_BG, fg=_TEXT, wraplength=500,
                 justify="left", font=("Helvetica", 10)
                 ).pack(padx=12, pady=(0, 8))

        # Sample picker + laser
        top = tk.Frame(self, bg=_FRAME, padx=10, pady=8)
        top.pack(fill="x", padx=12, pady=4)

        tk.Label(top, text="What's in the cuvette?", bg=_FRAME, fg=_TEXT,
                 font=("Helvetica", 10, "bold")
                 ).grid(row=0, column=0, sticky="w")
        self.sample_var = tk.StringVar(value="Isopropanol (IPA)")
        self.sample_combo = ttk.Combobox(
            top, textvariable=self.sample_var,
            values=list(dsp.RAMAN_STANDARDS.keys()),
            state="readonly", width=24)
        self.sample_combo.grid(row=0, column=1, padx=8, sticky="w")
        self.sample_combo.bind("<<ComboboxSelected>>",
                               lambda _e: self._on_sample_change())

        tk.Label(top, text="Laser (nm):", bg=_FRAME, fg=_TEXT,
                 font=("Helvetica", 10)).grid(row=0, column=2, padx=(16, 4))
        self.laser_var = tk.StringVar(value=f"{self.laser_nm:.2f}")
        tk.Entry(top, textvariable=self.laser_var, width=8,
                 bg=_ACCENT, fg=_TEXT, insertbackground=_CYAN,
                 relief="flat"
                 ).grid(row=0, column=3, padx=(0, 4))

        # Show expected peaks for the selected sample
        self.expected_var = tk.StringVar()
        tk.Label(self, textvariable=self.expected_var, bg=_BG, fg=_MUTED,
                 wraplength=500, justify="left", font=("Helvetica", 9)
                 ).pack(fill="x", padx=12, pady=(0, 6))

        # Detect button
        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill="x", padx=12, pady=4)
        tk.Button(btn_frame, text="🔍  Detect peaks & match",
                  command=self._detect_and_match,
                  bg=_CYAN, fg="#000",
                  font=("Helvetica", 11, "bold"),
                  relief="flat", padx=16, pady=6
                  ).pack(side="left")
        tk.Label(btn_frame, text="(re-run after editing assignments)",
                 bg=_BG, fg=_MUTED, font=("Helvetica", 9)
                 ).pack(side="left", padx=10)

        # Pairing table
        tbl_frame = tk.Frame(self, bg=_BG)
        tbl_frame.pack(fill="both", expand=True, padx=12, pady=4)
        tk.Label(tbl_frame, text="Peak assignments",
                 bg=_BG, fg=_TEXT, font=("Helvetica", 10, "bold"),
                 anchor="w").pack(fill="x")

        cols = ("pixel", "current_cm", "expected_cm", "delta")
        self.tree = ttk.Treeview(tbl_frame, columns=cols, show="headings",
                                  height=8, selectmode="extended")
        self.tree.heading("pixel", text="Pixel")
        self.tree.heading("current_cm", text="Current cm⁻¹")
        self.tree.heading("expected_cm", text="Expected cm⁻¹")
        self.tree.heading("delta", text="Δ cm⁻¹")
        for c, w in zip(cols, (70, 110, 110, 90)):
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=(2, 4))

        edit_row = tk.Frame(self, bg=_BG)
        edit_row.pack(fill="x", padx=12, pady=(0, 4))
        tk.Button(edit_row, text="✕ Remove selected",
                  command=self._remove_selected,
                  bg=_ACCENT, fg=_RED, relief="flat", padx=10
                  ).pack(side="left")
        tk.Label(edit_row, text="(remove any peaks that don't actually belong "
                 "to this substance)",
                 bg=_BG, fg=_MUTED, font=("Helvetica", 9)
                 ).pack(side="left", padx=8)

        # Status
        self.status_var = tk.StringVar(
            value="Capture a clean spectrum of the pure sample, then click "
                  "'Detect peaks & match'."
        )
        tk.Label(self, textvariable=self.status_var,
                 bg=_BG, fg=_GREEN, font=("Helvetica", 10),
                 wraplength=500, justify="left"
                 ).pack(fill="x", padx=12, pady=(2, 6))

        # Apply / Cancel
        action = tk.Frame(self, bg=_BG)
        action.pack(pady=10)
        self.apply_btn = tk.Button(
            action, text="✓ Apply Calibration",
            command=self._apply, state="disabled",
            bg=_CYAN, fg="#000",
            font=("Helvetica", 11, "bold"),
            relief="flat", padx=18, pady=6)
        self.apply_btn.pack(side="left", padx=8)
        tk.Button(action, text="Cancel", command=self.destroy,
                  bg="#333", fg=_TEXT, relief="flat", padx=14, pady=6
                  ).pack(side="left")

        self._on_sample_change()

    # ── Logic ─────────────────────────────────────────────────────────────

    def _on_sample_change(self):
        shifts = dsp.RAMAN_STANDARDS.get(self.sample_var.get(), [])
        if shifts:
            self.expected_var.set(
                f"Expected peaks for {self.sample_var.get()}:  "
                + ", ".join(f"{s:.0f}" for s in shifts)
                + " cm⁻¹"
            )
        else:
            self.expected_var.set("")

    def _laser_nm(self):
        try:
            v = float(self.laser_var.get())
            if not (200 < v < 2000):
                raise ValueError
            return v
        except ValueError:
            raise ValueError("Laser wavelength must be a number between 200 and 2000 nm.")

    def _detect_and_match(self):
        if self.spectrum_y is None or self.spectrum_y.size == 0:
            messagebox.showinfo("No spectrum",
                "Capture a spectrum of the pure sample first, then come back here.")
            return

        try:
            laser = self._laser_nm()
        except ValueError as e:
            messagebox.showerror("Invalid laser", str(e))
            return

        expected = list(dsp.RAMAN_STANDARDS.get(self.sample_var.get(), []))
        if not expected:
            messagebox.showerror("No reference",
                "No peak data is available for the selected sample.")
            return

        # Detect peaks with a moderate prominence threshold relative to the
        # spectrum's dynamic range.
        y = self.spectrum_y
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = y - np.min(y)
        prom = max(float(np.max(y)) * 0.03, 1.0)
        peaks, props = find_peaks(y, prominence=prom, distance=8)
        if len(peaks) == 0:
            self._set_status(
                "No peaks detected. Try a longer exposure or a more "
                "concentrated sample.", error=True)
            return

        # Sort detected peaks by prominence and keep the strongest
        # (slightly more than expected count to allow user to drop bogus ones).
        order = np.argsort(props["prominences"])[::-1]
        keep = min(len(peaks), max(len(expected) + 2, 6))
        strong = peaks[order[:keep]]
        refined = dsp.refine_peak_positions(y, strong)

        # Need a working calibration polynomial to estimate the offset. If
        # none is set, assume a generous linear default 500–800 nm.
        n_pixels = len(y)
        if self.current_coeffs is None:
            coeffs = np.array([650.0, 150.0, 0.0, 0.0])
        else:
            coeffs = np.asarray(self.current_coeffs, dtype=float)

        self.pairs = dsp.match_peaks_to_standard(
            refined, coeffs, expected, laser, n_pixels=n_pixels,
            search_window_cm=600.0,
        )
        self._populate_tree()

        if len(self.pairs) < 3:
            self._set_status(
                f"Only {len(self.pairs)} peaks could be matched — need at least 3 "
                "for a linear fit (4 for cubic). Try a cleaner spectrum or pick "
                "a different sample.", error=True)
            self.apply_btn.configure(state="disabled")
            return

        rms_cm = self._estimate_residual()
        self._set_status(
            f"Matched {len(self.pairs)} peaks to {self.sample_var.get()}. "
            f"Current-calibration residual ≈ {rms_cm:.0f} cm⁻¹. "
            f"Apply to fix the axis."
        )
        self.apply_btn.configure(state="normal")

    def _estimate_residual(self):
        if not self.pairs or self.current_coeffs is None:
            return 0.0
        pix = np.array([p[0] for p in self.pairs], dtype=float)
        exp_cm = np.array([p[1] for p in self.pairs], dtype=float)
        n = len(self.spectrum_y) if self.spectrum_y is not None else None
        t = dsp.normalize_pixels(pix, n_pixels=n)
        wl_now = np.polynomial.legendre.legval(
            t, np.asarray(self.current_coeffs, dtype=float))
        laser = self._laser_nm()
        cm_now = dsp.wavelengths_to_raman(wl_now, laser)
        return float(np.sqrt(np.mean((cm_now - exp_cm) ** 2)))

    def _populate_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if not self.pairs:
            return
        try:
            laser = self._laser_nm()
        except ValueError:
            laser = self.laser_nm

        coeffs = (np.asarray(self.current_coeffs, dtype=float)
                  if self.current_coeffs is not None else None)
        n = len(self.spectrum_y) if self.spectrum_y is not None else None
        for pix, exp_cm in self.pairs:
            if coeffs is not None:
                t = dsp.normalize_pixels([pix], n_pixels=n)
                wl_now = float(np.polynomial.legendre.legval(t, coeffs))
                cm_now = float(dsp.wavelengths_to_raman(np.array([wl_now]), laser)[0])
                delta = cm_now - exp_cm
                self.tree.insert("", "end", values=(
                    f"{pix:.1f}",
                    f"{cm_now:+.1f}",
                    f"{exp_cm:.1f}",
                    f"{delta:+.1f}",
                ))
            else:
                self.tree.insert("", "end", values=(
                    f"{pix:.1f}", "—", f"{exp_cm:.1f}", "—"))

    def _remove_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        indices = sorted((self.tree.index(s) for s in sel), reverse=True)
        for i in indices:
            del self.pairs[i]
            self.tree.delete(self.tree.get_children()[i])
        if len(self.pairs) < 3:
            self.apply_btn.configure(state="disabled")
            self._set_status(
                f"{len(self.pairs)} pair(s) left — need at least 3 to fit.",
                error=True)

    def _apply(self):
        if len(self.pairs) < 3:
            messagebox.showerror("Not enough pairs",
                "Need at least 3 matched peaks to fit a calibration.")
            return
        try:
            laser = self._laser_nm()
        except ValueError as e:
            messagebox.showerror("Invalid laser", str(e))
            return
        n_pixels = len(self.spectrum_y) if self.spectrum_y is not None else None
        pix = [p[0] for p in self.pairs]
        wl = [dsp.raman_shift_to_wavelength(p[1], laser) for p in self.pairs]
        # Cubic if we have enough points, else linear
        degree = 3 if len(self.pairs) >= 4 else 1
        try:
            coeffs = dsp.fit_calibration(pix, wl, degree=degree, n_pixels=n_pixels)
        except Exception as e:
            messagebox.showerror("Fit failed", str(e))
            return

        # Monotonicity sanity check
        if n_pixels:
            wl_full = dsp.pixels_to_wavelengths(coeffs, n_pixels)
            diffs = np.diff(wl_full)
            if not (np.all(diffs > 0) or np.all(diffs < 0)):
                if not messagebox.askyesno(
                    "Suspicious fit",
                    "The fitted calibration is non-monotonic, meaning the "
                    "spectrum would fold back on itself. This usually means "
                    "a peak got assigned to the wrong reference position.\n\n"
                    "Apply anyway?", default="no"):
                    return

        rms = dsp.calibration_pair_rms(coeffs, pix, wl, n_pixels=n_pixels)
        self.on_solution(coeffs)
        self._set_status(
            f"✓ Calibration applied (degree {degree}, RMS {rms:.3f} nm). "
            "The spectrum will reload.")
        self.after(800, self.destroy)

    def _set_status(self, msg, error=False):
        self.status_var.set(msg)
