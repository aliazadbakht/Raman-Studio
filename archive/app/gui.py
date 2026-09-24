# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Main GUI for Raman Spectrum Analyzer – macOS."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, time, os, atexit
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import dsp, fileio
from .camera import Camera, CameraError, list_cameras
from .calibration_dialog import CalibrationDialog, SampleCalibrationDialog
from .match_panel import MatchPanel
from PIL import Image, ImageTk, ImageDraw


# ── Colours ───────────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
BG2       = "#16213e"
ACCENT    = "#0f3460"
CYAN      = "#00d4ff"
ORANGE    = "#e94560"
TEXT      = "#e0e0e0"
MUTED     = "#888888"

PLOT_BG   = "#0d1117"
PLOT_FG   = "#00d4ff"
GRID_COL  = "#1f2937"


class RamanApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Raman Spectrum Analyzer")
        root.configure(bg=BG)
        root.geometry("1280x820")
        root.minsize(900, 600)
        # Release the camera (Spinnaker System) while the interpreter is still
        # alive. Without this, PySpin/libusb teardown runs at exit() and aborts
        # (SIGABRT in libusb_exit) every time the app is closed.
        root.protocol("WM_DELETE_WINDOW", self.on_app_close)
        # macOS routes Cmd+Q / Dock-quit / Apple-menu-Quit through this command
        # rather than WM_DELETE_WINDOW, so intercept it too.
        try:
            root.createcommand("::tk::mac::Quit", self.on_app_close)
        except tk.TclError:
            pass
        # Last-resort net for any exit path we didn't intercept: release before
        # the C++ static destructors run at process exit.
        atexit.register(self._release_camera)

        # ── State ─────────────────────────────────────────────────────────────
        self.camera: Camera | None = None
        self.live_running = False
        self._live_thread: threading.Thread | None = None

        self.cam_view_window = None
        self.raw_signal: np.ndarray | None = None
        self.blank_signal: np.ndarray | None = None
        self.calibration = fileio.load_calibration()
        self.x_axis: np.ndarray | None = None
        self.x_label: str = "Pixel"
        # Original x-axis from a loaded CSV / RRUFF file (in whatever
        # units the file provided). Lets us match against the library
        # without a per-instrument calibration.
        self._loaded_x: np.ndarray | None = None
        self._loaded_x_label: str = ""

        self.showpeaks_var = tk.BooleanVar(value=True)
        self.peak_prom_var = tk.DoubleVar(value=5000.0)
        self.peak_dist_var = tk.IntVar(value=20)

        self._current_file: str = ""
        self._cal_health: dict | None = None

        # ── Layout ────────────────────────────────────────────────────────────
        self._build_toolbar()
        self.content = tk.Frame(root, bg=BG)
        self.content.pack(fill="both", expand=True)
        self._build_sidebar(self.content)
        self._build_match_panel(self.content)
        self._build_plot(self.content)
        self._build_statusbar()

        self._apply_ttk_style()
        self._update_axis()
        # Reflect the loaded calibration's health (e.g., placeholder warning)
        # in the status bar from the very first frame, instead of waiting for
        # the next event to refresh it.
        self._set_status("Ready")

    # ═══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_toolbar(self):
        # Flow toolbar: buttons are laid out with place() and wrap onto extra
        # rows when the window is too narrow, so nothing (e.g. Match) ever gets
        # clipped off the right edge in a non-fullscreen window.
        bar = tk.Frame(self.root, bg=ACCENT)
        bar.pack(fill="x")
        self._toolbar = bar
        self._toolbar_items = []      # (widget, left_pad) in display order
        self._toolbar_last_w = -1

        def btn(text, cmd, color=CYAN, width=9):
            b = tk.Button(bar, text=text, command=cmd, bg=ACCENT, fg=color,
                          font=("Helvetica", 9, "bold"), relief="flat",
                          activebackground=BG2, activeforeground=CYAN,
                          padx=6, pady=3, width=width)
            self._toolbar_items.append((b, 2))
            return b

        def sep():
            s = tk.Frame(bar, bg=BG2, width=2, height=22)
            self._toolbar_items.append((s, 6))

        btn("📂 Open",    self.on_open)
        btn("💾 Save",    self.on_save)
        btn("📋 Copy",    self.on_copy)
        btn("🖼 Image",   self.on_image_save)
        sep()
        self.btn_connect = btn("🔌 Connect", self.on_connect, color="#aaffaa")
        self.btn_capture = btn("📷 Capture", self.on_capture)
        self.btn_live    = btn("▶ Live",    self.on_live_toggle, color="#ffdd57")
        self.btn_cam_view = btn("📹 Live Cam", self.on_cam_view_toggle, color="#a3ffd9")
        sep()
        btn("⬛ Set Blank",   self.on_set_blank,   color="#ffaa44")
        btn("✖ Clr Blank",   self.on_clear_blank,  color="#ff6b6b", width=10)
        sep()
        btn("🧪 Quick Cal", self.on_quick_calibrate, color="#aaffaa", width=10)
        btn("📐 Calibrate",  self.on_calibrate)
        btn("⚙ Params",     self.on_params_toggle)
        sep()
        self.btn_match = btn("🔬 Match", self.on_match_toggle, color="#a3ffd9")
        sep()
        btn("❓ About",      self.on_about, color=MUTED, width=7)

        bar.bind("<Configure>", self._reflow_toolbar)
        # Lay out once after widget sizes are known.
        self.root.after_idle(self._reflow_toolbar)

    def _reflow_toolbar(self, event=None):
        """Wrap toolbar items onto as many rows as the current width needs."""
        bar = self._toolbar
        width = bar.winfo_width()
        # Re-running on a height-only change (our own configure below) would loop.
        if width <= 1 or width == self._toolbar_last_w:
            return
        self._toolbar_last_w = width

        PAD_TOP, ROW_GAP = 4, 4
        # Pass 1: greedily break the items into rows that fit the width.
        rows, cur, x = [], [], 0
        for w, lpad in self._toolbar_items:
            rw = w.winfo_reqwidth()
            if cur and x + lpad + rw > width:
                rows.append(cur)
                cur, x = [], 0
            cur.append((w, x + lpad, w.winfo_reqheight()))
            x += lpad + rw
        if cur:
            rows.append(cur)

        # Pass 2: place each row, vertically centering items within the row.
        y = PAD_TOP
        for row in rows:
            row_h = max(rh for _, _, rh in row)
            for w, ix, rh in row:
                w.place(x=ix, y=y + (row_h - rh) // 2, anchor="nw")
            y += row_h + ROW_GAP
        bar.configure(height=y - ROW_GAP + PAD_TOP)

    def _apply_ttk_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox", fieldbackground=ACCENT, background=ACCENT,
                    foreground=TEXT, selectbackground=ACCENT)
        s.configure("TScale", background=BG2, troughcolor=ACCENT)
        s.configure("TCheckbutton", background=BG2, foreground=TEXT)
        s.configure("TSpinbox", fieldbackground=ACCENT, foreground=TEXT)

    def _build_sidebar(self, parent):
        self.sidebar = tk.Frame(parent, bg=BG2, width=240)
        self.sidebar.pack(side="left", fill="y", padx=(6,0), pady=6)
        self.sidebar.pack_propagate(False)

        def section(label):
            tk.Label(self.sidebar, text=label, bg=BG2, fg=CYAN,
                     font=("Helvetica", 9, "bold")).pack(anchor="w", padx=8, pady=(10,2))
            tk.Frame(self.sidebar, bg=ACCENT, height=1).pack(fill="x", padx=8)

        def row(label, widget_factory):
            f = tk.Frame(self.sidebar, bg=BG2)
            f.pack(fill="x", padx=8, pady=2)
            tk.Label(f, text=label, bg=BG2, fg=TEXT,
                     font=("Helvetica", 9), width=16, anchor="w").pack(side="left")
            w = widget_factory(f)
            w.pack(side="left", fill="x", expand=True)
            return w

        # ── Camera ────────────────────────────────────────────────────────────
        section("Camera")
        # Exposure: log-scaled slider so 1ms..60s all get usable slider travel.
        # Plus a numeric entry for arbitrary values (FLIR cameras accept
        # several minutes; entry isn't bound to the slider's range).
        self.exposure_var = tk.DoubleVar(value=0.1)
        self._exp_slider_var = tk.DoubleVar(value=self._sec_to_slider(0.1))
        self._exp_syncing = False     # guard against feedback loops
        self.exposure_label = tk.Label(self.sidebar, text="100 ms", bg=BG2, fg=CYAN,
                                        font=("Helvetica", 8))
        exp_frame = tk.Frame(self.sidebar, bg=BG2)
        exp_frame.pack(fill="x", padx=8, pady=2)
        tk.Label(exp_frame, text="Exposure", bg=BG2, fg=TEXT,
                 font=("Helvetica", 9), width=16, anchor="w").pack(side="left")
        ttk.Scale(exp_frame, from_=0.0, to=1000.0,
                  variable=self._exp_slider_var,
                  orient="horizontal",
                  command=self._on_exposure_slider
                  ).pack(side="left", fill="x", expand=True)
        exp_entry = tk.Entry(self.sidebar, textvariable=self.exposure_var, width=8,
                              bg=ACCENT, fg=TEXT, insertbackground=CYAN,
                              justify="right")
        exp_entry.pack(anchor="e", padx=12, pady=(2, 0))
        tk.Label(self.sidebar, text="seconds (type for >60 s)",
                 bg=BG2, fg=MUTED, font=("Helvetica", 7)
                 ).pack(anchor="e", padx=12)
        exp_entry.bind("<Return>", self._on_exposure_entry)
        exp_entry.bind("<FocusOut>", self._on_exposure_entry)
        self.exposure_label.pack(anchor="e", padx=12)

        self.gain_var = tk.DoubleVar(value=0.0)
        self.gain_label = tk.Label(self.sidebar, text="0.0 dB", bg=BG2, fg=CYAN,
                                    font=("Helvetica", 8))
        row("Gain (dB)",
            lambda f: ttk.Scale(f, from_=0, to=24, variable=self.gain_var,
                                orient="horizontal",
                                command=lambda v: (self.gain_label.config(
                                    text=f"{float(v):.1f} dB"),
                                    self._on_gain(float(v)))))
        self.gain_label.pack(anchor="e", padx=12)

        self.roi_var = tk.IntVar(value=10)
        roi_spin = row("ROI rows", lambda f: ttk.Spinbox(f, from_=1, to=1000,
                                               textvariable=self.roi_var, width=7,
                                               command=self._on_roi))
        roi_spin.bind("<Return>", self._on_roi)
        roi_spin.bind("<FocusOut>", self._on_roi)

        self.avg_var = tk.IntVar(value=1)
        row("Averages", lambda f: ttk.Spinbox(f, from_=1, to=100,
                                               textvariable=self.avg_var, width=7))

        # ── Processing ────────────────────────────────────────────────────────
        section("Processing")
        self.medfilt_var = tk.BooleanVar(value=False)
        row("Median filter", lambda f: ttk.Checkbutton(f, variable=self.medfilt_var,
                                                        command=self._replot))
        self.smooth_var = tk.IntVar(value=1)
        row("Boxcar window", lambda f: ttk.Spinbox(f, from_=1, to=51,
                                                    textvariable=self.smooth_var, width=7,
                                                    command=self._replot))
        self.baseline_var = tk.BooleanVar(value=False)
        row("Baseline removal", lambda f: ttk.Checkbutton(f, variable=self.baseline_var,
                                                           command=self._replot))
        self.blank_var = tk.BooleanVar(value=True)
        row("Blank subtraction", lambda f: ttk.Checkbutton(f, variable=self.blank_var,
                                                            command=self._replot))

        section("Savitzky-Golay")
        self.sg_var = tk.BooleanVar(value=False)
        row("Enable S-G", lambda f: ttk.Checkbutton(f, variable=self.sg_var,
                                                     command=self._replot))
        self.sg_window_var = tk.IntVar(value=11)
        row("Window", lambda f: ttk.Spinbox(f, from_=3, to=101, increment=2,
                                             textvariable=self.sg_window_var, width=7,
                                             command=self._replot))
        self.sg_order_var = tk.IntVar(value=3)
        row("Order", lambda f: ttk.Spinbox(f, from_=1, to=10,
                                            textvariable=self.sg_order_var, width=7,
                                            command=self._replot))
        self.sg_deriv_var = tk.IntVar(value=0)
        row("Derivative", lambda f: ttk.Spinbox(f, from_=0, to=4,
                                                 textvariable=self.sg_deriv_var, width=7,
                                                 command=self._replot))

        # ── Axis ──────────────────────────────────────────────────────────────
        section("Axis")
        self.axis_var = tk.StringVar(value="Raman Shifts" if self.calibration else "Pixels")
        row("X-axis", lambda f: ttk.Combobox(f, textvariable=self.axis_var,
                                              values=["Pixels","Wavelengths","Raman Shifts"],
                                              state="readonly", width=14))
        self.axis_var.trace_add("write", lambda *_: self._replot())

        self.flip_x_var = tk.BooleanVar(value=False)
        row("Flip X-axis", lambda f: ttk.Checkbutton(f, variable=self.flip_x_var,
                                                      command=self._replot))

        self.laser_var = tk.DoubleVar(value=532.0)
        self.laser_entry = row("Laser (nm)", lambda f: tk.Entry(f, textvariable=self.laser_var, width=8,
                                                                 bg=ACCENT, fg=TEXT, insertbackground=CYAN))
        self.laser_entry.bind("<Return>", lambda *_: self._replot())
        self.laser_entry.bind("<FocusOut>", lambda *_: self._replot())

        # ── X-range ───────────────────────────────────────────────────────────
        # Manual X-axis limits. Empty / non-numeric = auto (matplotlib default).
        self.xmin_var = tk.StringVar(value="")
        self.xmax_var = tk.StringVar(value="")
        xr_frame = tk.Frame(self.sidebar, bg=BG2)
        xr_frame.pack(fill="x", padx=8, pady=2)
        tk.Label(xr_frame, text="X range", bg=BG2, fg=TEXT,
                 font=("Helvetica", 9), width=16, anchor="w").pack(side="left")
        self.xmin_entry = tk.Entry(xr_frame, textvariable=self.xmin_var, width=6,
                                    bg=ACCENT, fg=TEXT, insertbackground=CYAN)
        self.xmin_entry.pack(side="left", padx=(0, 2))
        tk.Label(xr_frame, text="–", bg=BG2, fg=MUTED).pack(side="left")
        self.xmax_entry = tk.Entry(xr_frame, textvariable=self.xmax_var, width=6,
                                    bg=ACCENT, fg=TEXT, insertbackground=CYAN)
        self.xmax_entry.pack(side="left", padx=(2, 0))
        for w in (self.xmin_entry, self.xmax_entry):
            w.bind("<Return>", lambda *_: self._replot())
            w.bind("<FocusOut>", lambda *_: self._replot())

        xr_btns = tk.Frame(self.sidebar, bg=BG2)
        xr_btns.pack(fill="x", padx=8, pady=(0, 4))
        tk.Button(xr_btns, text="Auto", command=self._x_auto,
                  bg=ACCENT, fg=CYAN, relief="flat",
                  font=("Helvetica", 8), padx=8, pady=1
                  ).pack(side="left", padx=2)
        tk.Button(xr_btns, text="100–3500", command=lambda: self._x_set(100, 3500),
                  bg=ACCENT, fg=TEXT, relief="flat",
                  font=("Helvetica", 8), padx=6, pady=1
                  ).pack(side="left", padx=2)
        tk.Button(xr_btns, text="200–2000", command=lambda: self._x_set(200, 2000),
                  bg=ACCENT, fg=TEXT, relief="flat",
                  font=("Helvetica", 8), padx=6, pady=1
                  ).pack(side="left", padx=2)

        # ── Peak Detection ────────────────────────────────────────────────────
        section("Peak Detection")
        row("Show peaks", lambda f: ttk.Checkbutton(f, variable=self.showpeaks_var,
                                                     command=self._replot))
        self.prom_label = tk.Label(self.sidebar, text="5000", bg=BG2, fg=CYAN,
                                    font=("Helvetica", 8))
        row("Prominence",
            lambda f: ttk.Scale(f, from_=100, to=50000, variable=self.peak_prom_var,
                                orient="horizontal",
                                command=lambda v: (self.prom_label.config(
                                    text=f"{float(v):.0f}"),
                                    self._replot())))
        self.prom_label.pack(anchor="e", padx=12)

        row("Min distance", lambda f: ttk.Spinbox(f, from_=1, to=500,
                                                   textvariable=self.peak_dist_var, width=7,
                                                   command=self._replot))

        # ── Saturation ────────────────────────────────────────────────────────
        section("Display")
        self.showsat_var = tk.BooleanVar(value=False)
        row("Show saturation", lambda f: ttk.Checkbutton(f, variable=self.showsat_var,
                                                          command=self._replot))
        self.showroi_var = tk.BooleanVar(value=False)
        row("Show ROI profile", lambda f: ttk.Checkbutton(f, variable=self.showroi_var,
                                                           command=self._replot))

    def _build_match_panel(self, parent):
        # Container that is hidden initially; on_match_toggle packs it.
        self.match_panel = MatchPanel(parent, self, width=300)
        # Not packed yet — first toggle reveals it.

    def _build_plot(self, parent):
        plot_frame = tk.Frame(parent, bg=BG)
        plot_frame.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self.fig = Figure(facecolor=PLOT_BG)
        self.ax  = self.fig.add_subplot(111)
        self._style_axes(self.ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(plot_frame, bg=PLOT_BG)
        toolbar_frame.pack(fill="x")
        self.nav = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.nav.configure(background=PLOT_BG)
        self.nav.update()

        # Crosshair / cursor
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=ACCENT, pady=2)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready  |  No camera connected  |  No calibration")
        tk.Label(bar, textvariable=self.status_var, bg=ACCENT, fg=TEXT,
                 font=("Helvetica", 9), anchor="w").pack(side="left", padx=8)
        self.cursor_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.cursor_var, bg=ACCENT, fg=CYAN,
                 font=("Helvetica", 9), anchor="e").pack(side="right", padx=8)

    def _style_axes(self, ax):
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(CYAN)
        for sp in ax.spines.values():
            sp.set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.6)

    # ═══════════════════════════════════════════════════════════════════════════
    # Toolbar actions
    # ═══════════════════════════════════════════════════════════════════════════

    def on_open(self):
        path = filedialog.askopenfilename(
            title="Open Spectrum",
            filetypes=[("OpenRAMAN SPC files","*.spc"),("CSV files","*.csv"),
                       ("RSPC files","*.rspc"),("All files","*.*")])
        if not path:
            return
        try:
            lower_path = path.lower()
            if lower_path.endswith(".spc"):
                signal, cal, blank, config = fileio.load_spc(path)
                self.raw_signal = signal
                if cal:
                    self.calibration = cal
                if blank is not None:
                    self.blank_signal = blank
                self._loaded_x = None
                self._loaded_x_label = ""
            elif lower_path.endswith(".rspc"):
                signal, cal, blank, config = fileio.load_rspc(path)
                self.raw_signal = signal
                if cal:
                    self.calibration = cal
                if blank is not None:
                    self.blank_signal = blank
                self._loaded_x = None
                self._loaded_x_label = ""
            else:
                x, y, xl, yl = fileio.load_csv(path)
                self.raw_signal = y
                self._loaded_x = np.asarray(x, dtype=float).copy()
                self._loaded_x_label = xl
                if "cm" in xl.lower() or "raman" in xl.lower() or "shift" in xl.lower():
                    self.axis_var.set("Raman Shifts")
            self._current_file = path
            self._update_axis()
            self._replot()
            self._refresh_calibration_health()
            self._set_status(f"Opened: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Open Error", str(e))

    def on_save(self):
        path = filedialog.asksaveasfilename(
            title="Save Spectrum",
            defaultextension=".spc",
            filetypes=[("OpenRAMAN SPC files","*.spc"),("CSV files","*.csv"),
                       ("RSPC files","*.rspc")])
        if not path or self.raw_signal is None:
            return
        try:
            x = self.x_axis if self.x_axis is not None else np.arange(len(self.raw_signal))
            lower_path = path.lower()
            if lower_path.endswith(".spc"):
                uid = self.camera.uid if self.camera else ""
                fileio.save_spc(path, self.raw_signal, self.calibration, self.blank_signal, uid=uid)
            elif lower_path.endswith(".rspc"):
                fileio.save_rspc(path, self.raw_signal, self.calibration, self.blank_signal)
            else:
                fileio.save_csv(path, x, self._processed_signal(), self.x_label, "Intensity")
            self._set_status(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def on_copy(self):
        if self.raw_signal is None:
            return
        x = self.x_axis if self.x_axis is not None else np.arange(len(self.raw_signal))
        lines = [f"{self.x_label},Intensity"]
        for xi, yi in zip(x, self._processed_signal()):
            lines.append(f"{xi:.5e},{yi:.5e}")
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))
        self._set_status("Spectrum copied to clipboard.")

    def on_image_save(self):
        path = filedialog.asksaveasfilename(
            title="Save Plot Image",
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            self._set_status(f"Image saved: {os.path.basename(path)}")

    def on_connect(self):
        cams = list_cameras()
        if not cams:
            messagebox.showinfo("No Camera", "No cameras found. Check connection.")
            return
        
        # Simple picker if multiple cameras
        cam_type, cam_idx = cams[0][0], cams[0][1]
        if len(cams) > 1:
            dlg = _CameraPickerDialog(self.root, cams)
            self.root.wait_window(dlg)
            if dlg.result is None:
                return
            cam_type, cam_idx = dlg.result
        
        try:
            if self.camera:
                self.camera.release()
            self.camera = Camera(type=cam_type, index=cam_idx)
            try:
                cam_cal = self.camera.get_calibration()
            except Exception as cal_err:
                cam_cal = None
                print(f"Could not load OpenRAMAN calibration from camera: {cal_err}")
            if cam_cal:
                self.calibration = cam_cal
                fileio.save_calibration(cam_cal)
                self.axis_var.set("Raman Shifts")
            self.camera.set_roi(self.roi_var.get())
            self.btn_connect.config(text="✓ Connected", fg="#00ff88")
            cal_msg = "camera calibration loaded" if cam_cal else "using local/no calibration"
            self._set_status(
                f"Connected: {self.camera.uid}  |  {self.camera.width}×{self.camera.height}px  |  {cal_msg}"
            )
        except CameraError as e:
            messagebox.showerror("Camera Error", str(e))

    def on_capture(self):
        if self.camera is None:
            messagebox.showinfo("No Camera", "Connect a camera first.")
            return
        n = self.avg_var.get()
        try:
            accumulated = None
            for _ in range(n):
                sig, sat, roi = self.camera.acquire_spectrum()
                accumulated = sig if accumulated is None else accumulated + sig
            self.raw_signal = accumulated / n
            self._sat_signal = sat
            self._roi_signal = roi
            self._loaded_x = None
            self._loaded_x_label = ""
            self._update_axis()
            self._replot()
            self._refresh_calibration_health()
            self._set_status(f"Captured ({n} avg)  |  {len(self.raw_signal)} pixels")
        except Exception as e:
            messagebox.showerror("Acquisition Error", str(e))

    def on_live_toggle(self):
        if self.live_running:
            self.live_running = False
            self.btn_live.config(text="▶ Live", fg="#ffdd57")
            self._set_status("Live stopped.")
        else:
            if self.camera is None:
                messagebox.showinfo("No Camera", "Connect a camera first.")
                return
            self.live_running = True
            self.btn_live.config(text="⏹ Stop", fg=ORANGE)
            # Drop any frame buffered with the previous settings so the first
            # live frame reflects the current exposure/gain, not a stale one.
            try:
                self.camera.flush_stream()
            except Exception:
                pass
            self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
            self._live_thread.start()

    def on_cam_view_toggle(self):
        if self.cam_view_window is not None:
            self.cam_view_window.lift()
            self.cam_view_window.focus_force()
        else:
            if self.camera is None:
                messagebox.showinfo("No Camera", "Connect a camera first.")
                return
            self.cam_view_window = CameraViewWindow(self.root, self)

    def _live_loop(self):
        while self.live_running:
            try:
                sig, sat, roi = self.camera.acquire_spectrum()
                self.raw_signal = sig
                self._sat_signal = sat
                self._roi_signal = roi
                self.root.after(0, self._replot)
            except Exception:
                self.live_running = False
                break
            time.sleep(max(0.02, self.exposure_var.get()))

    def on_set_blank(self):
        if self.raw_signal is None:
            messagebox.showinfo("No Data", "Acquire or load a spectrum first.")
            return
        self.blank_signal = self.raw_signal.copy()
        self._set_status("Blank set.")
        self._replot()

    def on_clear_blank(self):
        self.blank_signal = None
        self._set_status("Blank cleared.")
        self._replot()

    def on_calibrate(self):
        try:
            laser_nm = float(self.laser_var.get())
        except Exception:
            laser_nm = 532.0
        CalibrationDialog(
            self.root,
            self.raw_signal,
            self._on_calibration_solution,
            on_load_camera=self._hard_load_calibration_from_camera,
            on_copy_to_camera=self._hard_copy_calibration_to_camera,
            laser_nm=laser_nm,
        )

    def on_quick_calibrate(self):
        """One-click calibration from a known sample (e.g. pure IPA)."""
        if self.raw_signal is None or len(self.raw_signal) == 0:
            messagebox.showinfo(
                "No spectrum",
                "Capture a clean spectrum of a pure liquid first "
                "(IPA, ethanol, cyclohexane, etc.), then click Quick Cal."
            )
            return
        try:
            laser_nm = float(self.laser_var.get())
        except Exception:
            laser_nm = 532.0
        SampleCalibrationDialog(
            self.root,
            self.raw_signal,
            self._on_calibration_solution,
            laser_nm=laser_nm,
            current_coeffs=self.calibration,
        )

    def _on_calibration_solution(self, coeffs):
        self._apply_calibration_coeffs(
            coeffs,
            "Calibration applied locally. Calibration lamp frame cleared; capture the sample again.",
            clear_current_spectrum=True,
        )

    def _apply_calibration_coeffs(self, coeffs, msg, clear_current_spectrum=False):
        self.calibration = list(coeffs)
        fileio.save_calibration(coeffs)
        self.axis_var.set("Raman Shifts")
        if clear_current_spectrum:
            self._clear_current_spectrum()
        else:
            self._update_axis()
            self._replot()
            self._refresh_calibration_health()
        self._set_status(msg)

    def _clear_current_spectrum(self):
        self.raw_signal = None
        self.x_axis = None
        self._loaded_x = None
        self._loaded_x_label = ""
        self._current_file = ""
        self._sat_signal = None
        self._roi_signal = None
        self.ax.cla()
        self._style_axes(self.ax)
        self.ax.set_xlabel("Raman Shift (cm⁻¹)", color=TEXT, fontsize=9)
        self.ax.set_ylabel("Intensity (a.u.)", color=TEXT, fontsize=9)
        self.ax.set_title("Raman Spectrum", color=CYAN, fontsize=10, fontweight="bold")
        self.ax.text(
            0.5,
            0.5,
            "Calibration applied. Capture a sample spectrum.",
            transform=self.ax.transAxes,
            fontsize=10,
            color=TEXT,
            ha="center",
            va="center",
        )
        self.canvas.draw_idle()

    def _hard_load_calibration_from_camera(self):
        if self.camera is None:
            raise RuntimeError("Connect the camera first.")
        if not hasattr(self.camera, "get_calibration"):
            raise RuntimeError("This camera backend cannot read OpenRAMAN calibration memory.")
        coeffs = self.camera.get_calibration()
        self._apply_calibration_coeffs(
            coeffs,
            "Hard-loaded OpenRAMAN calibration from camera."
        )
        return coeffs

    def _hard_copy_calibration_to_camera(self):
        if self.camera is None:
            raise RuntimeError("Connect the camera first.")
        if self.calibration is None:
            raise RuntimeError("No local calibration is available to copy.")
        if not hasattr(self.camera, "set_calibration"):
            raise RuntimeError("This camera backend cannot write OpenRAMAN calibration memory.")
        self.camera.set_calibration(self.calibration)
        self._set_status("Hard-copied current calibration into camera memory.")
        return list(self.calibration)

    def on_match_toggle(self):
        if self.match_panel.winfo_ismapped():
            self.match_panel.pack_forget()
            self.btn_match.configure(fg="#a3ffd9")
        else:
            self.match_panel.pack(side="right", fill="y", padx=(0, 6), pady=6)
            self.btn_match.configure(fg=CYAN)
            # If we already have a spectrum, kick off a match immediately.
            if self.raw_signal is not None:
                self.match_panel.run_match()

    def query_for_matching(self):
        """
        Return (cm_x, processed_y) for the current spectrum in cm⁻¹ space,
        or (None, None) if matching isn't possible.

        Two routes get you cm⁻¹:
          1. A loaded file (CSV / RRUFF) that already supplies cm⁻¹ data.
          2. A camera capture with calibration + laser wavelength set.
        """
        if self.raw_signal is None:
            return None, None

        # Route 1: file supplied an x-axis that looks like Raman shifts.
        if self._loaded_x is not None and len(self._loaded_x) == len(self.raw_signal):
            x = self._loaded_x
            xmin, xmax = float(np.min(x)), float(np.max(x))
            looks_like_cm = (
                xmin > 30 and xmax < 6000 and (xmax - xmin) > 200
                and ("cm" in self._loaded_x_label.lower()
                     or "raman" in self._loaded_x_label.lower()
                     or "shift" in self._loaded_x_label.lower()
                     or xmin > 80)  # heuristic: pixels start at 0
            )
            if looks_like_cm:
                return x, self._processed_signal()

        # Route 2: calibrated pixel axis → cm⁻¹ via laser.
        if self.calibration is None:
            return None, None
        n = len(self.raw_signal)
        wl = dsp.pixels_to_wavelengths(self.calibration, n)
        try:
            laser = float(self.laser_var.get())
        except Exception:
            laser = 532.0
        cm = dsp.wavelengths_to_raman(wl, laser)
        return cm, self._processed_signal()

    def on_params_toggle(self):
        if self.sidebar.winfo_ismapped():
            self.sidebar.pack_forget()
        else:
            self.sidebar.pack(side="left", fill="y", padx=(6,0), pady=6,
                              before=self.canvas.get_tk_widget().master)

    def on_about(self):
        messagebox.showinfo("About",
            "Raman Spectrum Analyzer\nmacOS Edition\n\n"
            "Based on The Pulsar Engineering SpectrumAnalyzer (CERN OHL-W v2)\n"
            "macOS port: Python/Tkinter/Matplotlib")

    def on_app_close(self):
        """Shut down cleanly: stop streaming and release the camera before the
        window (and interpreter) tear down, so the Spinnaker/libusb cleanup runs
        while Python is still alive instead of aborting at process exit."""
        # Stop the Live loop and wait for the worker so no frame grab is in
        # flight when we release the camera.
        self.live_running = False
        thread = self._live_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        # Close the Live Camera view (restores hardware ROI) if it's open.
        if self.cam_view_window is not None:
            try:
                self.cam_view_window.on_close()
            except Exception:
                pass
        self._release_camera()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _release_camera(self):
        """Release the Spinnaker camera once; safe to call repeatedly."""
        cam, self.camera = self.camera, None
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════════
    # Camera callbacks
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Exposure (log-scaled slider + free-form entry) ────────────────────
    # Slider position 0..1000 maps logarithmically to 1 ms..60 s. The entry
    # box is unbounded so the user can type longer exposures directly.
    _EXP_MIN_SEC = 0.001
    _EXP_MAX_SEC = 60.0

    def _sec_to_slider(self, sec):
        import math
        sec = max(self._EXP_MIN_SEC, min(self._EXP_MAX_SEC, float(sec)))
        lo, hi = math.log10(self._EXP_MIN_SEC), math.log10(self._EXP_MAX_SEC)
        return 1000.0 * (math.log10(sec) - lo) / (hi - lo)

    def _slider_to_sec(self, pos):
        import math
        lo, hi = math.log10(self._EXP_MIN_SEC), math.log10(self._EXP_MAX_SEC)
        return 10.0 ** (lo + (float(pos) / 1000.0) * (hi - lo))

    def _on_exposure(self, v):
        if self.camera:
            self.camera.set_exposure(v)

    def _on_exposure_slider(self, pos):
        if self._exp_syncing:
            return
        sec = self._slider_to_sec(pos)
        # Snap to sensible precision so the entry doesn't show 13 digits.
        if sec < 0.01:
            sec = round(sec, 4)
        elif sec < 1.0:
            sec = round(sec, 3)
        else:
            sec = round(sec, 2)
        self._exp_syncing = True
        try:
            self.exposure_var.set(sec)
        finally:
            self._exp_syncing = False
        self._apply_exposure(sec)

    def _on_exposure_entry(self, *_):
        if self._exp_syncing:
            return
        try:
            sec = float(self.exposure_var.get())
        except (TypeError, ValueError):
            return
        if sec <= 0:
            return
        # Push back onto the slider (clamped to 1ms-60s display range);
        # if user typed >60s the slider sits at max but the entry value wins.
        self._exp_syncing = True
        try:
            self._exp_slider_var.set(self._sec_to_slider(sec))
        finally:
            self._exp_syncing = False
        self._apply_exposure(sec)

    def _apply_exposure(self, sec):
        if sec < 1.0:
            txt = f"{sec*1000:.1f} ms"
        elif sec < 60.0:
            txt = f"{sec:.2f} s"
        else:
            txt = f"{sec/60.0:.2f} min"
        self.exposure_label.config(text=txt)
        self._on_exposure(sec)

    def _on_gain(self, v):
        if self.camera:
            self.camera.set_gain(v)

    def _on_roi(self, *_):
        if self.camera:
            try:
                rows = int(self.roi_var.get())
            except (tk.TclError, ValueError):
                return
            self.camera.set_roi(rows)

    # ═══════════════════════════════════════════════════════════════════════════
    # Axis computation
    # ═══════════════════════════════════════════════════════════════════════════

    def _update_axis(self):
        if self.raw_signal is None:
            return
        n = len(self.raw_signal)
        ax_type = self.axis_var.get()

        # A CSV / RRUFF file that ships its own x-axis wins over calibration:
        # the file's units are authoritative, and re-deriving from pixels
        # would be wrong if the file isn't pixel-indexed.
        if self._loaded_x is not None and len(self._loaded_x) == n:
            self.x_axis = self._loaded_x
            self.x_label = self._loaded_x_label or "x"
            return

        if ax_type == "Pixels" or self.calibration is None:
            self.x_axis  = np.arange(n, dtype=float)
            self.x_label = "Pixel"
        elif ax_type == "Wavelengths":
            self.x_axis  = dsp.pixels_to_wavelengths(self.calibration, n)
            self.x_label = "Wavelength (nm)"
        else:  # Raman Shifts
            wl = dsp.pixels_to_wavelengths(self.calibration, n)
            try:
                laser = float(self.laser_var.get())
            except Exception:
                laser = 532.0
            self.x_axis  = dsp.wavelengths_to_raman(wl, laser)
            self.x_label = "Raman Shift (cm⁻¹)"

    # ═══════════════════════════════════════════════════════════════════════════
    # Plotting
    # ═══════════════════════════════════════════════════════════════════════════

    def _replot(self, *_):
        if self.raw_signal is None:
            return
        self._update_axis()

        y_proc = self._processed_signal()

        x = self.x_axis if self.x_axis is not None else np.arange(len(y_proc))

        self.ax.cla()
        self._style_axes(self.ax)

        # Main spectrum
        self.ax.plot(x, y_proc, color=CYAN, linewidth=1.2, label="Spectrum")

        # Match overlay (reference spectrum, scaled to fit)
        self._draw_match_overlay(x, y_proc)

        # Saturation overlay
        if self.showsat_var.get() and hasattr(self, "_sat_signal") and self._sat_signal is not None:
            s = self._sat_signal
            if len(s) == len(x):
                self.ax.plot(x, s / s.max() * y_proc.max(),
                             color=ORANGE, linewidth=0.8, alpha=0.6, label="Saturation")

        # Peak annotations
        if self.showpeaks_var.get() and self.raw_signal is not None:
            peaks = dsp.detect_peaks(y_proc, 
                                     prominence=self.peak_prom_var.get(),
                                     distance=self.peak_dist_var.get())
            if len(peaks):
                self.ax.plot(x[peaks], y_proc[peaks], "x", color=ORANGE,
                             markersize=8, markeredgewidth=1.5)
                for pk in peaks:
                    val_x = x[pk]
                    self.ax.annotate(f"{val_x:.1f}",
                                     xy=(val_x, y_proc[pk]),
                                     xytext=(0, 8), textcoords="offset points",
                                     ha="center", fontsize=7, color=ORANGE)

        self.ax.set_xlabel(self.x_label, color=TEXT, fontsize=9)
        self.ax.set_ylabel("Intensity (a.u.)", color=TEXT, fontsize=9)
        self.ax.set_title("Raman Spectrum", color=CYAN, fontsize=10, fontweight="bold")

        if self.blank_signal is not None and self.blank_var.get():
            self.ax.text(0.01, 0.97, "● Blank subtracted", transform=self.ax.transAxes,
                         fontsize=7, color="#ffaa44", va="top")

        # Non-monotonic calibration detector — when the cm⁻¹/wavelength axis
        # folds back on itself the plot appears to have "two values per x".
        # Surface that to the user instead of letting them puzzle over it.
        if self.x_axis is not None and len(self.x_axis) > 2:
            d = np.diff(self.x_axis)
            if not (np.all(d > 0) or np.all(d < 0)):
                self.ax.text(0.5, 0.97,
                             "⚠ Non-monotonic calibration — plot folds back. "
                             "Re-calibrate (try Linear model).",
                             transform=self.ax.transAxes, fontsize=9,
                             color="#ff6b6b", ha="center", va="top",
                             bbox=dict(facecolor=PLOT_BG, edgecolor="#ff6b6b",
                                       boxstyle="round,pad=0.3"))
            elif self.x_label == "Raman Shift (cm⁻¹)":
                finite_x = self.x_axis[np.isfinite(self.x_axis)]
                if finite_x.size:
                    cm_min = float(np.min(finite_x))
                    cm_max = float(np.max(finite_x))
                    if cm_min > 650 or cm_max < 3400:
                        self.ax.text(0.5, 0.92,
                                     f"⚠ Calibration covers {cm_min:.0f}–{cm_max:.0f} cm⁻¹; "
                                     "expected roughly 500–3500.",
                                     transform=self.ax.transAxes, fontsize=8,
                                     color="#ffaa44", ha="center", va="top",
                                     bbox=dict(facecolor=PLOT_BG, edgecolor="#ffaa44",
                                               boxstyle="round,pad=0.25"))
                lamp_name, lamp_count = self._calibration_lamp_signature(x, y_proc)
                if lamp_count >= 4:
                    self.ax.text(0.5, 0.86,
                                 f"⚠ {lamp_name} calibration-line pattern detected. "
                                 "Remove/turn off the calibration source and capture the sample again.",
                                 transform=self.ax.transAxes, fontsize=8,
                                 color="#ff6b6b", ha="center", va="top",
                                 bbox=dict(facecolor=PLOT_BG, edgecolor="#ff6b6b",
                                           boxstyle="round,pad=0.25"))

        # Manual X-range override
        self._apply_xrange()

        self.canvas.draw_idle()

        # Update Live Cam view if open
        if getattr(self, "cam_view_window", None) is not None:
            self.cam_view_window.update_image()

    def _apply_xrange(self):
        """Apply manual X-limits from the sidebar entries, if both are numeric."""
        should_invert = (self.x_label == "Raman Shift (cm⁻¹)") ^ self.flip_x_var.get()
        try:
            lo = float(self.xmin_var.get())
            hi = float(self.xmax_var.get())
            if lo == hi:
                raise ValueError
        except (ValueError, AttributeError):
            if self.x_label == "Raman Shift (cm⁻¹)":
                if should_invert:
                    self.ax.set_xlim(3500, 500)
                else:
                    self.ax.set_xlim(500, 3500)
            else:
                if should_invert:
                    self.ax.invert_xaxis()
            return
        if should_invert:
            self.ax.set_xlim(max(lo, hi), min(lo, hi))
        else:
            self.ax.set_xlim(min(lo, hi), max(lo, hi))

    def _x_auto(self):
        self.xmin_var.set("")
        self.xmax_var.set("")
        self._replot()

    def _x_set(self, lo, hi):
        self.xmin_var.set(str(lo))
        self.xmax_var.set(str(hi))
        self._replot()

    def _draw_match_overlay(self, x, y_proc):
        """Overlay one or more reference spectra.

        References live on a fixed cm⁻¹ grid and are L2-normalized; each is
        scaled to the sample's visible amplitude before plotting. Overlays
        are only drawn on the Raman-shift axis — that's the only axis where
        the comparison is meaningful.
        """
        if not hasattr(self, "match_panel") or not self.match_panel.winfo_ismapped():
            return
        refs = self.match_panel.selected_references()
        if not refs:
            return
        if self.x_label != "Raman Shift (cm⁻¹)":
            self.ax.text(0.99, 0.97,
                         "Switch X-axis to Raman Shifts to see overlay",
                         transform=self.ax.transAxes, fontsize=8,
                         color="#ffaa44", ha="right", va="top")
            return

        y_visible = y_proc[np.isfinite(y_proc)]
        if y_visible.size == 0:
            return
        peak = float(np.max(y_visible))
        if peak <= 0:
            return

        for _, meta, y_ref, grid, color in refs:
            ref_peak = float(np.max(y_ref)) or 1.0
            y_scaled = y_ref * (peak / ref_peak) * 0.9
            self.ax.plot(grid, y_scaled, color=color, linewidth=0.9,
                         alpha=0.7, label=f"Ref: {meta.display()}")
        self.ax.legend(loc="upper right", facecolor=PLOT_BG, edgecolor=GRID_COL,
                       labelcolor=TEXT, fontsize=8)

    def _calibration_lamp_signature(self, x, y_proc):
        """Return (source name, matched line count) for lamp-like spectra."""
        if self.x_label == "Wavelength (nm)":
            wavelengths = np.asarray(x, dtype=float)
        elif self.x_label == "Raman Shift (cm⁻¹)":
            try:
                laser = float(self.laser_var.get())
            except Exception:
                laser = 532.0
            shifts = np.asarray(x, dtype=float)
            denom = (1.0 / laser) - (shifts / 1.0e7)
            wavelengths = np.full(shifts.shape, np.nan, dtype=float)
            good = denom > 0
            wavelengths[good] = 1.0 / denom[good]
        else:
            return "", 0

        if wavelengths.size != len(y_proc):
            return "", 0
        peaks = dsp.detect_peaks(
            y_proc,
            prominence=self.peak_prom_var.get(),
            distance=self.peak_dist_var.get(),
            n_peaks=25,
        )
        if len(peaks) < 4:
            return "", 0

        peak_wavelengths = wavelengths[peaks]

        def count_matches(lines):
            lines = np.asarray(lines, dtype=float)
            used = set()
            for wl in peak_wavelengths[np.isfinite(peak_wavelengths)]:
                nearest_idx = int(np.argmin(np.abs(lines - wl)))
                if abs(lines[nearest_idx] - wl) <= 1.2:
                    used.add(nearest_idx)
            return len(used)

        neon = count_matches(dsp.NEON_LINES)
        hgar = count_matches(dsp.MERCURY_ARGON_LINES)
        if neon >= hgar:
            return "Neon", neon
        return "Mercury-Argon", hgar

    def _processed_signal(self):
        return dsp.process_spectrum(
            self.raw_signal,
            blank_y=self.blank_signal,
            use_blank=self.blank_var.get() and self.blank_signal is not None,
            use_median=self.medfilt_var.get(),
            boxcar_window=max(1, self.smooth_var.get()),
            use_baseline=self.baseline_var.get(),
            use_sgolay=self.sg_var.get(),
            sg_window=max(3, self.sg_window_var.get()),
            sg_order=max(1, self.sg_order_var.get()),
            sg_deriv=self.sg_deriv_var.get(),
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Mouse
    # ═══════════════════════════════════════════════════════════════════════════

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            self.cursor_var.set(f"x={event.xdata:.2f}  y={event.ydata:.2f}")
        else:
            self.cursor_var.set("")

    # ═══════════════════════════════════════════════════════════════════════════
    # Status
    # ═══════════════════════════════════════════════════════════════════════════

    def _set_status(self, msg):
        cam = self.camera.uid if self.camera else "No camera"
        cal = self._calibration_health_label()
        self.status_var.set(f"{msg}  |  {cam}  |  {cal}")

    def _calibration_health_label(self):
        """Status-bar text for the calibration state.

        When a recent diagnostic is available, surface a one-line hint so the
        user notices a bad calibration without having to click Match.
        """
        if not self.calibration:
            return "Uncalibrated"
        # A "Set the axis to round cm⁻¹ endpoints" placeholder cal is NOT a
        # real measurement — every peak appears in the wrong place. Flag it
        # before showing diagnostic verdicts that would assume a real fit.
        try:
            laser = float(self.laser_var.get())
        except Exception:
            laser = 532.0
        if dsp.is_default_calibration_axis(self.calibration, laser_nm=laser):
            return "⚠ Default 500–3500 axis (placeholder — Quick Cal or 📐 Calibrate to fix)"
        health = getattr(self, "_cal_health", None)
        if health is None:
            return "Calibrated"
        if health["verdict"] == "ok":
            return f"✓ Cal OK (matches {health['name']})"
        return (f"⚠ Cal off ~{abs(health['offset_cm']):.0f} cm⁻¹ "
                f"(looks like {health['name']} — try Quick Cal)")

    def _refresh_calibration_health(self):
        """Recompute the calibration diagnostic from the current spectrum.

        Called whenever raw_signal changes so the status-bar hint stays
        in sync. Failures are swallowed — the indicator is best-effort.
        """
        self._cal_health = None
        if self.raw_signal is None or not self.calibration:
            return
        try:
            laser = float(self.laser_var.get())
        except Exception:
            laser = 532.0
        try:
            self._cal_health = dsp.diagnose_calibration(
                self.raw_signal, self.calibration, laser,
                n_pixels=len(self.raw_signal),
            )
        except Exception:
            self._cal_health = None


# ── Live Camera View helper ───────────────────────────────────────────────────

class CameraViewWindow(tk.Toplevel):
    def __init__(self, parent, app_instance):
        super().__init__(parent)
        self.title("Live Camera View")
        self.configure(bg=BG)
        self.app = app_instance
        self.geometry("680x540")
        self.resizable(True, True)

        # Label to display the image
        self.img_label = tk.Label(self, bg=BG)
        self.img_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Control Frame for options
        ctrl_frame = tk.Frame(self, bg=BG)
        ctrl_frame.pack(fill="x", side="bottom", pady=(0, 5))

        # Checkbox to toggle full sensor view
        self.full_sensor_var = tk.BooleanVar(value=False)
        self.chk_full = ttk.Checkbutton(ctrl_frame, text="Show Full Sensor (Reset Hardware ROI)",
                                         variable=self.full_sensor_var,
                                         command=self.on_full_sensor_toggle)
        self.chk_full.pack(pady=2)

        # Status/info label
        self.info_var = tk.StringVar(value="Start 'Live' mode to see the camera feed.")
        tk.Label(ctrl_frame, textvariable=self.info_var, bg=BG, fg=TEXT,
                 font=("Helvetica", 10)).pack(fill="x", pady=2)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_image()

    def on_full_sensor_toggle(self):
        if self.app.camera is None:
            return
        try:
            if self.full_sensor_var.get():
                # Show full sensor: reset offsets to 0 and size to max
                self.app.camera.set_hardware_roi()
            else:
                # Restore original cropped ROI
                self.app.camera.restore_hardware_roi()
            # Trigger a replot/refresh to fetch the new frame dimensions
            self.app._replot()
        except Exception as e:
            messagebox.showerror("ROI Error", f"Failed to change hardware ROI: {e}")
            self.full_sensor_var.set(not self.full_sensor_var.get())

    def update_image(self):
        if not self.winfo_exists():
            return

        # Get last frame from camera
        gray = None
        if self.app.camera and getattr(self.app.camera, "last_frame", None) is not None:
            gray = self.app.camera.last_frame

        if gray is not None:
            h, w = gray.shape

            # Normalize to 0-255 for display
            g_min, g_max = float(gray.min()), float(gray.max())
            if g_max > g_min:
                img_data = ((gray - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
            else:
                img_data = np.zeros_like(gray, dtype=np.uint8)

            img = Image.fromarray(img_data)

            # Draw ROI bounds
            try:
                roi_rows = int(self.app.roi_var.get())
            except Exception:
                roi_rows = 0

            if roi_rows > 0 and roi_rows < h:
                cy = h // 2
                r = roi_rows // 2
                # Draw red box indicating the ROI region
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                draw.rectangle([0, max(0, cy - r), w - 1, min(h - 1, cy + r)], outline="#ff6b6b", width=2)

            # Resize image to fit window width dynamically
            window_width = self.winfo_width()
            if window_width < 100:
                window_width = 640
            
            target_width = max(320, window_width - 20)
            scale = target_width / w
            target_height = int(h * scale)
            
            img_resized = img.resize((target_width, target_height), Image.Resampling.BILINEAR)

            self.photo = ImageTk.PhotoImage(image=img_resized)
            self.img_label.configure(image=self.photo)
            
            # Show saturating pixels percent or count
            sat_count = int(np.sum(gray >= 254 if g_max <= 255 else gray >= 4094))
            sat_msg = f"  |  Saturated: {sat_count} px" if sat_count > 0 else ""
            self.info_var.set(
                f"Resolution: {w}x{h} px  |  ROI Rows: {roi_rows}  |  Range: {int(g_min)}-{int(g_max)}{sat_msg}"
            )
        else:
            if self.app.live_running:
                self.info_var.set("Waiting for first frame...")
            else:
                self.info_var.set("Camera idle. Click 'Live' to start feed.")

    def on_close(self):
        if self.full_sensor_var.get() and self.app.camera is not None:
            try:
                self.app.camera.restore_hardware_roi()
            except Exception:
                pass
        self.app.cam_view_window = None
        self.destroy()


# ── Camera picker helper ───────────────────────────────────────────────────────

class _CameraPickerDialog(tk.Toplevel):
    def __init__(self, parent, camera_info):
        super().__init__(parent)
        self.title("Select Camera")
        self.configure(bg=BG)
        self.result = None
        self.resizable(False, False)
        
        tk.Label(self, text="Choose camera:", bg=BG, fg=TEXT,
                 font=("Helvetica", 11)).pack(padx=20, pady=10)
        
        self.selection = tk.StringVar(value=f"{camera_info[0][0]}:{camera_info[0][1]}")
        
        for ctype, cidx, cname in camera_info:
            tk.Radiobutton(self, text=cname, variable=self.selection, 
                           value=f"{ctype}:{cidx}",
                           bg=BG, fg=TEXT, selectcolor=ACCENT,
                           activebackground=BG).pack(anchor="w", padx=20)
        
        tk.Button(self, text="OK", command=self._ok,
                  bg=CYAN, fg="#000", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=16, pady=4).pack(pady=12)
        self.grab_set()

    def _ok(self):
        ctype, cidx = self.selection.get().split(":")
        self.result = (ctype, int(cidx))
        self.destroy()
