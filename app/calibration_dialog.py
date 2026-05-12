"""Wavelength calibration dialog."""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from . import dsp


class CalibrationDialog(tk.Toplevel):
    def __init__(self, parent, spectrum_y, on_solution):
        super().__init__(parent)
        self.title("Wavelength Calibration")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")
        self.on_solution = on_solution
        self.spectrum_y = spectrum_y
        self.peak_pixels = []
        self.known_wl = []
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

        # Source selection
        src_frame = tk.Frame(self, bg=style_frame, padx=10, pady=8)
        src_frame.pack(fill="x", padx=10, pady=4)
        tk.Label(src_frame, text="Calibration Source:", bg=style_frame, fg=style_text,
                 font=("Helvetica", 10)).grid(row=0, column=0, sticky="w")
        self.source_var = tk.StringVar(value="Neon")
        src_combo = ttk.Combobox(src_frame, textvariable=self.source_var,
                                  values=["Neon", "Mercury-Argon"], state="readonly", width=18)
        src_combo.grid(row=0, column=1, padx=8)

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
        self.npeaks_var = tk.IntVar(value=10)
        ttk.Spinbox(sens_frame, from_=2, to=30, textvariable=self.npeaks_var, width=5).grid(
            row=0, column=4)

        # Peak list
        list_frame = tk.Frame(self, bg=style_bg)
        list_frame.pack(fill="both", expand=True, padx=10, pady=4)
        tk.Label(list_frame, text="Matched Peaks  (Pixel → Wavelength nm)",
                 bg=style_bg, fg=style_text, font=("Helvetica", 10, "bold")).pack(anchor="w")

        cols = ("pixel", "wavelength")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=8)
        self.tree.heading("pixel", text="Pixel")
        self.tree.heading("wavelength", text="Wavelength (nm)")
        self.tree.column("pixel", width=100, anchor="center")
        self.tree.column("wavelength", width=160, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

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
        tk.Button(det_frame, text="Auto-Detect Peaks from Spectrum",
                  command=self._auto_detect, bg=style_accent, fg=style_cyan,
                  relief="flat", padx=10, pady=4).pack(side="left")

        # Status label
        self.status_var = tk.StringVar(value="Add peak pairs and click Calibrate.")
        tk.Label(self, textvariable=self.status_var, bg=style_bg, fg="#aaaaaa",
                 font=("Helvetica", 9), wraplength=450).pack(padx=10, pady=2)

        # Buttons
        btn_frame = tk.Frame(self, bg=style_bg)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Calibrate", command=self._calibrate,
                  bg="#00d4ff", fg="#000000", font=("Helvetica", 11, "bold"),
                  relief="flat", padx=20, pady=6).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Cancel", command=self.destroy,
                  bg="#333", fg=style_text, relief="flat", padx=16, pady=6).pack(side="left")

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
                # Update treeview
                for item in self.tree.get_children():
                    if abs(float(self.tree.item(item, 'values')[0]) - p) < 1.0:
                        self.tree.item(item, values=(f"{p:.1f}", f"{w:.3f}"))
                        break
                self.status_var.set(f"Updated peak at pixel {p:.1f}")
                return

        self.peak_pixels.append(p)
        self.known_wl.append(w)
        self.tree.insert("", "end", values=(f"{p:.1f}", f"{w:.3f}"))
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
        p = self.peak_pixels[idx]
        self.tree.item(sel[0], values=(f"{p:.1f}", f"{w:.3f}"))
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
        # Map sensitivity (0.1-1.0) to prominence
        prom = (1.1 - self.sens_var.get()) * 0.05 * (np.max(y) - np.min(y))
        peaks = dsp.detect_peaks(y, prominence=prom, n_peaks=self.npeaks_var.get())
        # Get source lines
        source = self.source_var.get()
        lines = dsp.NEON_LINES if source == "Neon" else dsp.MERCURY_ARGON_LINES
        self.status_var.set(f"Detected {len(peaks)} peaks. Match each pixel to a source line below.")
        # Populate pixel column; user fills in wavelength
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.peak_pixels.clear()
        self.known_wl.clear()
        hint = ", ".join(f"{l:.1f}" for l in lines[:8]) + " …"
        self.status_var.set(f"Detected pixels: {list(peaks)}.\nSource lines (nm): {hint}")
        # Pre-fill with source lines as a starting point
        for i, p in enumerate(peaks):
            self.peak_pixels.append(p)
            wl = lines[i] if i < len(lines) else 0.0
            self.known_wl.append(wl)
            self.tree.insert("", "end", values=(f"{p:.1f}", f"{wl:.3f}" if wl > 0 else "??"))
        
        self.status_var.set(f"Detected {len(peaks)} peaks and assigned first {min(len(peaks), len(lines))} lines. Please verify and Update if wrong.")

    def _calibrate(self):
        # Filter out unassigned peaks (those with wavelength 0.0)
        valid_pix = []
        valid_wl = []
        for p, w in zip(self.peak_pixels, self.known_wl):
            if w > 0:
                valid_pix.append(p)
                valid_wl.append(w)

        if len(valid_pix) < 2:
            messagebox.showerror("Error", "Need at least 2 valid matched pairs (wavelength > 0).")
            return
        
        degree = 3 if self.model_var.get() == "Cubic" else 1
        try:
            n_pixels = len(self.spectrum_y) if self.spectrum_y is not None else None
            coeffs = dsp.fit_calibration(valid_pix, valid_wl, degree=degree, n_pixels=n_pixels)
        except Exception as e:
            messagebox.showerror("Calibration Failed", str(e))
            return
        self.on_solution(coeffs)
        self.status_var.set(f"✓ Calibration OK! Coefficients: {[round(c,4) for c in coeffs]}")
        self.after(1500, self.destroy)
