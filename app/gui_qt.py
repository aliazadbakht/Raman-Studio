# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V. (https://wfront.nl)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""PySide6 (Qt) GUI for Raman Studio.

The current front end, reusing every backend module unchanged (camera, dsp,
fileio, library, matching). The Tkinter front end under archive/ is superseded.
Run via main_qt.py.
"""
import os
os.environ.setdefault("QT_API", "pyside6")

import threading
import time
import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QAction, QFont, QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QLineEdit, QSpinBox,
    QCheckBox, QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QSplitter, QScrollArea, QMessageBox, QFileDialog, QToolBar,
    QDialog,
)

from . import dsp, fileio
from .camera import Camera, CameraError, list_cameras
from .calibration_dialog_qt import CalibrationDialogQt, SampleCalibrationDialogQt
from .match_panel_qt import MatchPanelQt

# ── Palette ─────────────────────────────────────────────────────────────────
BG, BG2, ACCENT = "#1a1a2e", "#16213e", "#0f3460"
CYAN, ORANGE, TEXT, MUTED = "#00d4ff", "#e94560", "#e0e0e0", "#888888"
PLOT_BG, GRID_COL = "#0d1117", "#1f2937"

STYLE = f"""
QMainWindow, QWidget, QDialog {{ background:{BG}; color:{TEXT}; }}
QGroupBox {{ border:1px solid {ACCENT}; border-radius:4px; margin-top:10px; font-weight:bold; }}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; color:{CYAN}; }}
QPushButton {{ background:{ACCENT}; color:{CYAN}; border:none; padding:5px 12px;
               border-radius:3px; font-weight:bold; }}
QPushButton:hover {{ background:{BG2}; }}
QPushButton:disabled {{ color:{MUTED}; }}
QToolBar {{ background:{ACCENT}; border:none; padding:3px; spacing:2px; }}
QToolButton {{ color:{CYAN}; background:transparent; padding:5px 9px; border-radius:3px; font-weight:bold; }}
QToolButton:hover {{ background:{BG2}; }}
QLineEdit, QSpinBox, QComboBox {{ background:{ACCENT}; color:{TEXT}; border:1px solid {BG2};
    border-radius:3px; padding:2px 4px; }}
QListWidget, QTreeWidget {{ background:{PLOT_BG}; color:{TEXT}; border:1px solid {ACCENT}; }}
QHeaderView::section {{ background:{ACCENT}; color:{TEXT}; padding:3px; border:none; }}
QSlider::groove:horizontal {{ background:{ACCENT}; height:4px; border-radius:2px; }}
QSlider::handle:horizontal {{ background:{CYAN}; width:14px; margin:-6px 0; border-radius:7px; }}
QCheckBox {{ color:{TEXT}; }}
QStatusBar {{ background:{ACCENT}; color:{TEXT}; }}
QStatusBar QLabel {{ color:{TEXT}; }}
QLabel {{ color:{TEXT}; }}
QScrollArea {{ border:none; }}
QDockWidget {{ color:{CYAN}; titlebar-close-icon:none; }}
QDockWidget::title {{ background:{ACCENT}; padding:5px; }}
"""


class _Emitter(QObject):
    frame_ready = Signal()
    error = Signal(str)


class RamanQtWindow(QMainWindow):
    _EXP_MIN_SEC, _EXP_MAX_SEC = 0.001, 60.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raman Studio")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(STYLE)

        # ── State ───────────────────────────────────────────────────────────
        self.camera = None
        self.live_running = False
        self._live_thread = None
        self.cam_view_window = None
        self.raw_signal = None
        self.blank_signal = None
        self._sat_signal = None
        self._roi_signal = None
        self.calibration = fileio.load_calibration()
        self.x_axis = None
        self.x_label = "Pixel"
        self._loaded_x = None
        self._loaded_x_label = ""
        self._cal_health = None
        self._current_file = ""
        self._exp_syncing = False

        self._emitter = _Emitter()
        self._emitter.frame_ready.connect(self.replot)
        self._emitter.error.connect(lambda m: self._set_status(f"Live error: {m}"))

        self._build_toolbar()
        self._build_body()
        self._build_match_dock()
        self._build_statusbar()
        self._set_status("Ready")

    # ════════════════════════════════════════════════════════════════════════
    # UI construction
    # ════════════════════════════════════════════════════════════════════════
    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(text, slot):
            a = QAction(text, self); a.triggered.connect(slot); tb.addAction(a); return a

        act("📂 Open", self.on_open)
        act("💾 Save", self.on_save)
        act("📋 Copy", self.on_copy)
        act("🖼 Image", self.on_image_save)
        tb.addSeparator()
        self.act_connect = act("🔌 Connect", self.on_connect)
        self.act_capture = act("📷 Capture", self.on_capture)
        self.act_live = act("▶ Live", self.on_live_toggle)
        self.act_cam_view = act("📹 Live Cam", self.on_cam_view_toggle)
        tb.addSeparator()
        act("⬛ Set Blank", self.on_set_blank)
        act("✖ Clr Blank", self.on_clear_blank)
        tb.addSeparator()
        act("🧪 Quick Cal", self.on_quick_calibrate)
        act("📐 Calibrate", self.on_calibrate)
        act("⚙ Params", self.on_params_toggle)
        tb.addSeparator()
        act("🔬 Match", self.on_match_toggle)
        tb.addSeparator()
        act("❓ About", self.on_about)

    def _build_body(self):
        self.splitter = QSplitter(Qt.Horizontal)
        self.sidebar = self._build_sidebar()
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self._build_plot())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([280, 1000])
        self.setCentralWidget(self.splitter)

    def _build_sidebar(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedWidth(290)
        panel = QWidget(); v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8); v.setSpacing(8)

        # ── Camera ──
        cam_box = QGroupBox("Camera"); cam = QFormLayout(cam_box)
        self.exp_slider = QSlider(Qt.Horizontal); self.exp_slider.setRange(0, 1000)
        self.exp_slider.setValue(int(self._sec_to_slider(0.1)))
        self.exp_slider.valueChanged.connect(self._on_exposure_slider)
        self.exp_entry = QLineEdit("0.1"); self.exp_entry.setFixedWidth(70)
        self.exp_entry.editingFinished.connect(self._on_exposure_entry)
        self.exp_label = QLabel("100 ms"); self.exp_label.setStyleSheet(f"color:{CYAN};")
        erow = QHBoxLayout(); erow.addWidget(self.exp_slider, 1); erow.addWidget(self.exp_entry)
        cam.addRow("Exposure", self._wrap(erow)); cam.addRow("", self.exp_label)
        self.gain_slider = QSlider(Qt.Horizontal); self.gain_slider.setRange(0, 24)
        self.gain_label = QLabel("0.0 dB"); self.gain_label.setStyleSheet(f"color:{CYAN};")
        self.gain_slider.valueChanged.connect(self._on_gain)
        cam.addRow("Gain (dB)", self.gain_slider); cam.addRow("", self.gain_label)
        self.roi_spin = QSpinBox(); self.roi_spin.setRange(1, 1000); self.roi_spin.setValue(10)
        self.roi_spin.valueChanged.connect(self._on_roi)
        cam.addRow("ROI rows", self.roi_spin)
        self.avg_spin = QSpinBox(); self.avg_spin.setRange(1, 100); self.avg_spin.setValue(1)
        cam.addRow("Averages", self.avg_spin)
        v.addWidget(cam_box)

        # ── Processing ──
        proc_box = QGroupBox("Processing"); proc = QFormLayout(proc_box)
        self.chk_median = self._check(proc, "Median filter", False)
        self.boxcar_spin = QSpinBox(); self.boxcar_spin.setRange(1, 51); self.boxcar_spin.setValue(1)
        self.boxcar_spin.valueChanged.connect(self.replot)
        proc.addRow("Boxcar window", self.boxcar_spin)
        self.chk_baseline = self._check(proc, "Baseline removal", False)
        self.chk_blank = self._check(proc, "Blank subtraction", True)
        v.addWidget(proc_box)

        # ── Savitzky-Golay ──
        sg_box = QGroupBox("Savitzky-Golay"); sg = QFormLayout(sg_box)
        self.chk_sg = self._check(sg, "Enable S-G", False)
        self.sg_window = QSpinBox(); self.sg_window.setRange(3, 101); self.sg_window.setSingleStep(2)
        self.sg_window.setValue(11); self.sg_window.valueChanged.connect(self.replot)
        sg.addRow("Window", self.sg_window)
        self.sg_order = QSpinBox(); self.sg_order.setRange(1, 10); self.sg_order.setValue(3)
        self.sg_order.valueChanged.connect(self.replot)
        sg.addRow("Order", self.sg_order)
        self.sg_deriv = QSpinBox(); self.sg_deriv.setRange(0, 4); self.sg_deriv.setValue(0)
        self.sg_deriv.valueChanged.connect(self.replot)
        sg.addRow("Derivative", self.sg_deriv)
        v.addWidget(sg_box)

        # ── Axis ──
        axis_box = QGroupBox("Axis"); axis = QFormLayout(axis_box)
        self.axis_combo = QComboBox(); self.axis_combo.addItems(["Pixels", "Wavelengths", "Raman Shifts"])
        self.axis_combo.setCurrentText("Raman Shifts" if self.calibration else "Pixels")
        self.axis_combo.currentTextChanged.connect(self.replot)
        axis.addRow("X-axis", self.axis_combo)
        self.chk_flip_x = self._check(axis, "Flip X-axis", False)
        self.laser_entry = QLineEdit("532.0"); self.laser_entry.editingFinished.connect(self.replot)
        axis.addRow("Laser (nm)", self.laser_entry)
        v.addWidget(axis_box)

        # ── X-range ──
        xr_box = QGroupBox("X range"); xr = QVBoxLayout(xr_box)
        rr = QHBoxLayout()
        self.xmin_entry = QLineEdit(); self.xmin_entry.setPlaceholderText("min")
        self.xmax_entry = QLineEdit(); self.xmax_entry.setPlaceholderText("max")
        self.xmin_entry.editingFinished.connect(self.replot)
        self.xmax_entry.editingFinished.connect(self.replot)
        rr.addWidget(self.xmin_entry); rr.addWidget(QLabel("–")); rr.addWidget(self.xmax_entry)
        xr.addLayout(rr)
        br = QHBoxLayout()
        b_auto = QPushButton("Auto"); b_auto.clicked.connect(self._x_auto)
        b1 = QPushButton("100–3500"); b1.clicked.connect(lambda: self._x_set(100, 3500))
        b2 = QPushButton("200–2000"); b2.clicked.connect(lambda: self._x_set(200, 2000))
        br.addWidget(b_auto); br.addWidget(b1); br.addWidget(b2)
        xr.addLayout(br)
        v.addWidget(xr_box)

        # ── Peak detection ──
        peak_box = QGroupBox("Peak Detection"); peak = QFormLayout(peak_box)
        self.chk_peaks = self._check(peak, "Show peaks", True)
        self.prom_slider = QSlider(Qt.Horizontal); self.prom_slider.setRange(100, 50000)
        self.prom_slider.setValue(5000); self.prom_slider.valueChanged.connect(self.replot)
        peak.addRow("Prominence", self.prom_slider)
        self.dist_spin = QSpinBox(); self.dist_spin.setRange(1, 500); self.dist_spin.setValue(20)
        self.dist_spin.valueChanged.connect(self.replot)
        peak.addRow("Min distance", self.dist_spin)
        v.addWidget(peak_box)

        # ── Display ──
        disp_box = QGroupBox("Display"); disp = QFormLayout(disp_box)
        self.chk_sat = self._check(disp, "Show saturation", False)
        self.chk_roi = self._check(disp, "Show ROI profile", False)
        v.addWidget(disp_box)

        v.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _build_plot(self):
        container = QWidget(); lay = QVBoxLayout(container); lay.setContentsMargins(6, 6, 6, 6)
        self.fig = Figure(facecolor=PLOT_BG)
        self.ax = self.fig.add_subplot(111)
        self._style_axes(self.ax)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.nav = NavigationToolbar(self.canvas, container)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.nav)
        return container

    def _build_match_dock(self):
        self.match_dock = MatchPanelQt(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.match_dock)
        self.match_dock.hide()

    def _build_statusbar(self):
        self.cursor_label = QLabel("")
        self.cursor_label.setStyleSheet(f"color:{CYAN};")
        self.statusBar().addPermanentWidget(self.cursor_label)
        self.statusBar().showMessage("Ready")

    # ── small helpers ──
    def _wrap(self, layout):
        w = QWidget(); layout.setContentsMargins(0, 0, 0, 0); w.setLayout(layout); return w

    def _check(self, form, label, default):
        cb = QCheckBox(); cb.setChecked(default); cb.stateChanged.connect(self.replot)
        form.addRow(label, cb); return cb

    def _style_axes(self, ax):
        ax.set_facecolor(PLOT_BG); ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.6)

    # ════════════════════════════════════════════════════════════════════════
    # Exposure controls
    # ════════════════════════════════════════════════════════════════════════
    def _sec_to_slider(self, sec):
        import math
        sec = max(self._EXP_MIN_SEC, min(self._EXP_MAX_SEC, float(sec)))
        lo, hi = math.log10(self._EXP_MIN_SEC), math.log10(self._EXP_MAX_SEC)
        return 1000.0 * (math.log10(sec) - lo) / (hi - lo)

    def _slider_to_sec(self, pos):
        import math
        lo, hi = math.log10(self._EXP_MIN_SEC), math.log10(self._EXP_MAX_SEC)
        return 10.0 ** (lo + (float(pos) / 1000.0) * (hi - lo))

    def _on_exposure_slider(self, pos):
        if self._exp_syncing:
            return
        sec = self._slider_to_sec(pos)
        sec = round(sec, 4 if sec < 0.01 else 3 if sec < 1.0 else 2)
        self._exp_syncing = True; self.exp_entry.setText(str(sec)); self._exp_syncing = False
        self._apply_exposure(sec)

    def _on_exposure_entry(self):
        if self._exp_syncing:
            return
        try:
            sec = float(self.exp_entry.text())
        except ValueError:
            return
        if sec <= 0:
            return
        self._exp_syncing = True; self.exp_slider.setValue(int(self._sec_to_slider(sec)))
        self._exp_syncing = False
        self._apply_exposure(sec)

    def _apply_exposure(self, sec):
        txt = f"{sec*1000:.1f} ms" if sec < 1.0 else (f"{sec:.2f} s" if sec < 60 else f"{sec/60:.2f} min")
        self.exp_label.setText(txt)
        if self.camera:
            self.camera.set_exposure(sec)

    def _on_gain(self, val):
        self.gain_label.setText(f"{float(val):.1f} dB")
        if self.camera:
            self.camera.set_gain(float(val))

    def _on_roi(self, *_):
        if self.camera:
            self.camera.set_roi(int(self.roi_spin.value()))

    def _exposure_seconds(self):
        try:
            return float(self.exp_entry.text())
        except ValueError:
            return 0.1

    def _laser_nm(self):
        try:
            return float(self.laser_entry.text())
        except ValueError:
            return 532.0

    # ════════════════════════════════════════════════════════════════════════
    # Camera actions
    # ════════════════════════════════════════════════════════════════════════
    def on_connect(self):
        cams = list_cameras()
        if not cams:
            QMessageBox.information(self, "No Camera", "No cameras found. Check connection.")
            return
        cam_type, cam_idx = cams[0][0], cams[0][1]
        try:
            if self.camera:
                self.camera.release()
            self.camera = Camera(type=cam_type, index=cam_idx)
            try:
                cam_cal = self.camera.get_calibration()
            except Exception:
                cam_cal = None
            if cam_cal:
                self.calibration = cam_cal
                fileio.save_calibration(cam_cal)
                self.axis_combo.setCurrentText("Raman Shifts")
            self.camera.set_roi(int(self.roi_spin.value()))
            self.act_connect.setText("✓ Connected")
            cal_msg = "camera calibration loaded" if cam_cal else "using local/no calibration"
            self._set_status(f"Connected: {self.camera.uid}  |  "
                             f"{self.camera.width}×{self.camera.height}px  |  {cal_msg}")
        except CameraError as e:
            QMessageBox.critical(self, "Camera Error", str(e))

    def on_capture(self):
        if self.camera is None:
            QMessageBox.information(self, "No Camera", "Connect a camera first.")
            return
        n = int(self.avg_spin.value())
        try:
            acc = None
            for _ in range(n):
                sig, sat, roi = self.camera.acquire_spectrum()
                acc = sig if acc is None else acc + sig
            self.raw_signal = acc / n
            self._sat_signal = sat; self._roi_signal = roi
            self._loaded_x = None; self._loaded_x_label = ""
            self.replot()
            self._refresh_calibration_health()
            self._set_status(f"Captured ({n} avg)  |  {len(self.raw_signal)} pixels")
        except Exception as e:
            QMessageBox.critical(self, "Acquisition Error", str(e))

    def on_live_toggle(self):
        if self.live_running:
            self.live_running = False
            self.act_live.setText("▶ Live")
            self._set_status("Live stopped.")
            return
        if self.camera is None:
            QMessageBox.information(self, "No Camera", "Connect a camera first.")
            return
        self.live_running = True
        self.act_live.setText("⏹ Stop")
        # Drop frames buffered with previous settings (stop→change→live fix).
        try:
            self.camera.flush_stream()
        except Exception:
            pass
        self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._live_thread.start()

    def _live_loop(self):
        while self.live_running:
            try:
                sig, sat, roi = self.camera.acquire_spectrum()
                self.raw_signal = sig; self._sat_signal = sat; self._roi_signal = roi
                self._emitter.frame_ready.emit()
            except Exception as e:
                self.live_running = False
                self._emitter.error.emit(str(e))
                break
            time.sleep(max(0.02, self._exposure_seconds()))

    def on_cam_view_toggle(self):
        if self.cam_view_window is not None:
            self.cam_view_window.raise_(); self.cam_view_window.activateWindow()
            return
        if self.camera is None:
            QMessageBox.information(self, "No Camera", "Connect a camera first.")
            return
        self.cam_view_window = CameraViewWindowQt(self)
        self.cam_view_window.show()

    def on_set_blank(self):
        if self.raw_signal is None:
            QMessageBox.information(self, "No Data", "Acquire or load a spectrum first.")
            return
        self.blank_signal = self.raw_signal.copy()
        self._set_status("Blank set."); self.replot()

    def on_clear_blank(self):
        self.blank_signal = None
        self._set_status("Blank cleared."); self.replot()

    # ════════════════════════════════════════════════════════════════════════
    # Calibration
    # ════════════════════════════════════════════════════════════════════════
    def on_calibrate(self):
        CalibrationDialogQt(
            self, self.raw_signal, self._on_calibration_solution,
            on_load_camera=self._hard_load_calibration_from_camera,
            on_copy_to_camera=self._hard_copy_calibration_to_camera,
            laser_nm=self._laser_nm()).show()

    def on_quick_calibrate(self):
        if self.raw_signal is None or len(self.raw_signal) == 0:
            QMessageBox.information(self, "No spectrum",
                "Capture a clean spectrum of a pure liquid first (IPA, ethanol, cyclohexane, "
                "etc.), then click Quick Cal.")
            return
        SampleCalibrationDialogQt(
            self, self.raw_signal, self._on_calibration_solution,
            laser_nm=self._laser_nm(), current_coeffs=self.calibration).show()

    def _on_calibration_solution(self, coeffs):
        self._apply_calibration_coeffs(
            coeffs, "Calibration applied locally. Calibration lamp frame cleared; "
            "capture the sample again.", clear_current_spectrum=True)

    def _apply_calibration_coeffs(self, coeffs, msg, clear_current_spectrum=False):
        self.calibration = list(coeffs)
        fileio.save_calibration(coeffs)
        self.axis_combo.setCurrentText("Raman Shifts")
        if clear_current_spectrum:
            self._clear_current_spectrum()
        else:
            self.replot(); self._refresh_calibration_health()
        self._set_status(msg)

    def _clear_current_spectrum(self):
        self.raw_signal = None; self.x_axis = None
        self._loaded_x = None; self._loaded_x_label = ""
        self._current_file = ""; self._sat_signal = None; self._roi_signal = None
        self.ax.cla(); self._style_axes(self.ax)
        self.ax.set_xlabel("Raman Shift (cm⁻¹)", color=TEXT, fontsize=9)
        self.ax.set_ylabel("Intensity (a.u.)", color=TEXT, fontsize=9)
        self.ax.text(0.5, 0.5, "Calibration applied. Capture a sample spectrum.",
                     transform=self.ax.transAxes, fontsize=10, color=TEXT, ha="center", va="center")
        self.canvas.draw_idle()

    def _hard_load_calibration_from_camera(self):
        if self.camera is None:
            raise RuntimeError("Connect the camera first.")
        coeffs = self.camera.get_calibration()
        self._apply_calibration_coeffs(coeffs, "Hard-loaded OpenRAMAN calibration from camera.")
        return coeffs

    def _hard_copy_calibration_to_camera(self):
        if self.camera is None:
            raise RuntimeError("Connect the camera first.")
        if self.calibration is None:
            raise RuntimeError("No local calibration is available to copy.")
        self.camera.set_calibration(self.calibration)
        self._set_status("Hard-copied current calibration into camera memory.")
        return list(self.calibration)

    # ════════════════════════════════════════════════════════════════════════
    # Match / Params toggles
    # ════════════════════════════════════════════════════════════════════════
    def on_match_toggle(self):
        if self.match_dock.isVisible():
            self.match_dock.hide()
        else:
            self.match_dock.show()
            if self.raw_signal is not None:
                self.match_dock.run_match()

    def on_params_toggle(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def query_for_matching(self):
        """Return (cm_x, processed_y) for the current spectrum, or (None, None)."""
        if self.raw_signal is None:
            return None, None
        if self._loaded_x is not None and len(self._loaded_x) == len(self.raw_signal):
            x = self._loaded_x
            xmin, xmax = float(np.min(x)), float(np.max(x))
            looks_like_cm = (xmin > 30 and xmax < 6000 and (xmax - xmin) > 200
                             and ("cm" in self._loaded_x_label.lower()
                                  or "raman" in self._loaded_x_label.lower()
                                  or "shift" in self._loaded_x_label.lower()
                                  or xmin > 80))
            if looks_like_cm:
                return x, self._processed_signal()
        if self.calibration is None:
            return None, None
        n = len(self.raw_signal)
        wl = dsp.pixels_to_wavelengths(self.calibration, n)
        cm = dsp.wavelengths_to_raman(wl, self._laser_nm())
        return cm, self._processed_signal()

    def on_about(self):
        QMessageBox.information(self, "About",
            "Raman Studio\n\n"
            "Copyright (c) 2026 Wfront Principle B.V. (https://wfront.nl)\n"
            "Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)\n"
            "Licensed under CERN-OHL-W v2.\n\n"
            "A modified, independent port of the OpenRAMAN Spectrum Analyzer\n"
            "(https://www.open-raman.org/), copyright (c) Luc Boussemaere\n"
            "(The Pulsar), also under CERN-OHL-W v2. Not an official OpenRAMAN\n"
            "release and not affiliated with or endorsed by OpenRAMAN.\n\n"
            "PySide6 / NumPy / SciPy / Matplotlib")

    # ════════════════════════════════════════════════════════════════════════
    # File actions
    # ════════════════════════════════════════════════════════════════════════
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Spectrum", "", "Spectra (*.spc *.csv *.rspc);;All files (*.*)")
        if not path:
            return
        try:
            low = path.lower()
            if low.endswith(".spc"):
                sig, cal, blank, _ = fileio.load_spc(path)
                self.raw_signal = sig; self._loaded_x = None; self._loaded_x_label = ""
                if cal:
                    self.calibration = cal
                if blank is not None:
                    self.blank_signal = blank
            elif low.endswith(".rspc"):
                sig, cal, blank, _ = fileio.load_rspc(path)
                self.raw_signal = sig; self._loaded_x = None; self._loaded_x_label = ""
                if cal:
                    self.calibration = cal
                if blank is not None:
                    self.blank_signal = blank
            else:
                x, y, xl, yl = fileio.load_csv(path)
                self.raw_signal = y
                self._loaded_x = np.asarray(x, dtype=float).copy()
                self._loaded_x_label = xl
                if any(k in xl.lower() for k in ("cm", "raman", "shift")):
                    self.axis_combo.setCurrentText("Raman Shifts")
            self._current_file = path
            self.replot(); self._refresh_calibration_health()
            self._set_status(f"Opened: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Open Error", str(e))

    def on_save(self):
        if self.raw_signal is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Spectrum", "", "OpenRAMAN SPC (*.spc);;CSV (*.csv);;RSPC (*.rspc)")
        if not path:
            return
        try:
            x = self.x_axis if self.x_axis is not None else np.arange(len(self.raw_signal))
            low = path.lower()
            if low.endswith(".spc"):
                uid = self.camera.uid if self.camera else ""
                fileio.save_spc(path, self.raw_signal, self.calibration, self.blank_signal, uid=uid)
            elif low.endswith(".rspc"):
                fileio.save_rspc(path, self.raw_signal, self.calibration, self.blank_signal)
            else:
                fileio.save_csv(path, x, self._processed_signal(), self.x_label, "Intensity")
            self._set_status(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def on_copy(self):
        if self.raw_signal is None:
            return
        x = self.x_axis if self.x_axis is not None else np.arange(len(self.raw_signal))
        lines = [f"{self.x_label},Intensity"]
        for xi, yi in zip(x, self._processed_signal()):
            lines.append(f"{xi:.5e},{yi:.5e}")
        QApplication.clipboard().setText("\n".join(lines))
        self._set_status("Spectrum copied to clipboard.")

    def on_image_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot Image", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            self._set_status(f"Image saved: {os.path.basename(path)}")

    # ════════════════════════════════════════════════════════════════════════
    # Axis / processing / plotting
    # ════════════════════════════════════════════════════════════════════════
    def _update_axis(self):
        if self.raw_signal is None:
            return
        n = len(self.raw_signal)
        ax_type = self.axis_combo.currentText()
        if self._loaded_x is not None and len(self._loaded_x) == n:
            self.x_axis = self._loaded_x; self.x_label = self._loaded_x_label or "x"
            return
        if ax_type == "Pixels" or self.calibration is None:
            self.x_axis = np.arange(n, dtype=float); self.x_label = "Pixel"
        elif ax_type == "Wavelengths":
            self.x_axis = dsp.pixels_to_wavelengths(self.calibration, n)
            self.x_label = "Wavelength (nm)"
        else:
            wl = dsp.pixels_to_wavelengths(self.calibration, n)
            self.x_axis = dsp.wavelengths_to_raman(wl, self._laser_nm())
            self.x_label = "Raman Shift (cm⁻¹)"

    def _processed_signal(self):
        return dsp.process_spectrum(
            self.raw_signal, blank_y=self.blank_signal,
            use_blank=self.chk_blank.isChecked() and self.blank_signal is not None,
            use_median=self.chk_median.isChecked(),
            boxcar_window=max(1, int(self.boxcar_spin.value())),
            use_baseline=self.chk_baseline.isChecked(),
            use_sgolay=self.chk_sg.isChecked(),
            sg_window=max(3, int(self.sg_window.value())),
            sg_order=max(1, int(self.sg_order.value())),
            sg_deriv=int(self.sg_deriv.value()))

    def replot(self):
        if self.raw_signal is None:
            return
        self._update_axis()
        y = self._processed_signal()
        x = self.x_axis if self.x_axis is not None else np.arange(len(y))

        self.ax.cla(); self._style_axes(self.ax)
        self.ax.plot(x, y, color=CYAN, linewidth=1.2, label="Spectrum")
        self._draw_match_overlay(x, y)

        if self.chk_sat.isChecked() and self._sat_signal is not None:
            s = self._sat_signal
            if len(s) == len(x) and s.max() > 0:
                self.ax.plot(x, s / s.max() * y.max(), color=ORANGE, linewidth=0.8,
                             alpha=0.6, label="Saturation")

        if self.chk_peaks.isChecked():
            peaks = dsp.detect_peaks(y, prominence=float(self.prom_slider.value()),
                                     distance=int(self.dist_spin.value()))
            if len(peaks):
                self.ax.plot(x[peaks], y[peaks], "x", color=ORANGE, markersize=8, markeredgewidth=1.5)
                for pk in peaks:
                    self.ax.annotate(f"{x[pk]:.1f}", xy=(x[pk], y[pk]), xytext=(0, 8),
                                     textcoords="offset points", ha="center", fontsize=7, color=ORANGE)

        self.ax.set_xlabel(self.x_label, color=TEXT, fontsize=9)
        self.ax.set_ylabel("Intensity (a.u.)", color=TEXT, fontsize=9)
        self.ax.set_title("Raman Spectrum", color=CYAN, fontsize=10, fontweight="bold")

        if self.blank_signal is not None and self.chk_blank.isChecked():
            self.ax.text(0.01, 0.97, "● Blank subtracted", transform=self.ax.transAxes,
                         fontsize=7, color="#ffaa44", va="top")

        self._draw_calibration_warnings(x, y)

        self._apply_xrange()
        self.canvas.draw_idle()

        if self.cam_view_window is not None:
            self.cam_view_window.update_image()

    def _draw_match_overlay(self, x, y_proc):
        if not self.match_dock.isVisible():
            return
        refs = self.match_dock.selected_references()
        if not refs:
            return
        if self.x_label != "Raman Shift (cm⁻¹)":
            self.ax.text(0.99, 0.97, "Switch X-axis to Raman Shifts to see overlay",
                         transform=self.ax.transAxes, fontsize=8, color="#ffaa44", ha="right", va="top")
            return
        y_visible = y_proc[np.isfinite(y_proc)]
        if y_visible.size == 0:
            return
        peak = float(np.max(y_visible))
        if peak <= 0:
            return
        for _, meta, y_ref, grid, color in refs:
            ref_peak = float(np.max(y_ref)) or 1.0
            self.ax.plot(grid, y_ref * (peak / ref_peak) * 0.9, color=color, linewidth=0.9,
                         alpha=0.7, label=f"Ref: {meta.display()}")
        self.ax.legend(loc="upper right", facecolor=PLOT_BG, edgecolor=GRID_COL,
                       labelcolor=TEXT, fontsize=8)

    def _draw_calibration_warnings(self, x, y_proc):
        if self.x_axis is None or len(self.x_axis) <= 2:
            return
        d = np.diff(self.x_axis)
        if not (np.all(d > 0) or np.all(d < 0)):
            self.ax.text(0.5, 0.97, "⚠ Non-monotonic calibration — plot folds back. "
                         "Re-calibrate (try Linear model).", transform=self.ax.transAxes,
                         fontsize=9, color="#ff6b6b", ha="center", va="top",
                         bbox=dict(facecolor=PLOT_BG, edgecolor="#ff6b6b", boxstyle="round,pad=0.3"))
            return
        if self.x_label != "Raman Shift (cm⁻¹)":
            return
        finite_x = self.x_axis[np.isfinite(self.x_axis)]
        if finite_x.size:
            cm_min, cm_max = float(np.min(finite_x)), float(np.max(finite_x))
            if cm_min > 650 or cm_max < 3400:
                self.ax.text(0.5, 0.92, f"⚠ Calibration covers {cm_min:.0f}–{cm_max:.0f} cm⁻¹; "
                             "expected roughly 500–3500.", transform=self.ax.transAxes, fontsize=8,
                             color="#ffaa44", ha="center", va="top",
                             bbox=dict(facecolor=PLOT_BG, edgecolor="#ffaa44", boxstyle="round,pad=0.25"))
        lamp_name, lamp_count = self._calibration_lamp_signature(x, y_proc)
        if lamp_count >= 4:
            self.ax.text(0.5, 0.86, f"⚠ {lamp_name} calibration-line pattern detected. "
                         "Remove/turn off the calibration source and capture the sample again.",
                         transform=self.ax.transAxes, fontsize=8, color="#ff6b6b", ha="center", va="top",
                         bbox=dict(facecolor=PLOT_BG, edgecolor="#ff6b6b", boxstyle="round,pad=0.25"))

    def _calibration_lamp_signature(self, x, y_proc):
        if self.x_label == "Wavelength (nm)":
            wavelengths = np.asarray(x, dtype=float)
        elif self.x_label == "Raman Shift (cm⁻¹)":
            shifts = np.asarray(x, dtype=float)
            denom = (1.0 / self._laser_nm()) - (shifts / 1.0e7)
            wavelengths = np.full(shifts.shape, np.nan, dtype=float)
            good = denom > 0
            wavelengths[good] = 1.0 / denom[good]
        else:
            return "", 0
        if wavelengths.size != len(y_proc):
            return "", 0
        peaks = dsp.detect_peaks(y_proc, prominence=float(self.prom_slider.value()),
                                 distance=int(self.dist_spin.value()), n_peaks=25)
        if len(peaks) < 4:
            return "", 0
        peak_wavelengths = wavelengths[peaks]

        def count_matches(lines):
            lines = np.asarray(lines, dtype=float); used = set()
            for wl in peak_wavelengths[np.isfinite(peak_wavelengths)]:
                nearest = int(np.argmin(np.abs(lines - wl)))
                if abs(lines[nearest] - wl) <= 1.2:
                    used.add(nearest)
            return len(used)

        neon = count_matches(dsp.NEON_LINES); hgar = count_matches(dsp.MERCURY_ARGON_LINES)
        return ("Neon", neon) if neon >= hgar else ("Mercury-Argon", hgar)

    def _apply_xrange(self):
        should_invert = (self.x_label == "Raman Shift (cm⁻¹)") ^ self.chk_flip_x.isChecked()
        try:
            lo = float(self.xmin_entry.text()); hi = float(self.xmax_entry.text())
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
        self.xmin_entry.clear(); self.xmax_entry.clear(); self.replot()

    def _x_set(self, lo, hi):
        self.xmin_entry.setText(str(lo)); self.xmax_entry.setText(str(hi)); self.replot()

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            self.cursor_label.setText(f"x={event.xdata:.2f}  y={event.ydata:.2f}")
        else:
            self.cursor_label.setText("")

    # ════════════════════════════════════════════════════════════════════════
    # Status / calibration health
    # ════════════════════════════════════════════════════════════════════════
    def _set_status(self, msg):
        cam = self.camera.uid if self.camera else "No camera"
        self.statusBar().showMessage(f"{msg}   |   {cam}   |   {self._calibration_health_label()}")

    def _calibration_health_label(self):
        if not self.calibration:
            return "Uncalibrated"
        if dsp.is_default_calibration_axis(self.calibration, laser_nm=self._laser_nm()):
            return "⚠ Default 500–3500 axis (placeholder — Quick Cal or 📐 Calibrate to fix)"
        health = self._cal_health
        if health is None:
            return "Calibrated"
        if health["verdict"] == "ok":
            return f"✓ Cal OK (matches {health['name']})"
        return (f"⚠ Cal off ~{abs(health['offset_cm']):.0f} cm⁻¹ "
                f"(looks like {health['name']} — try Quick Cal)")

    def _refresh_calibration_health(self):
        self._cal_health = None
        if self.raw_signal is None or not self.calibration:
            return
        try:
            self._cal_health = dsp.diagnose_calibration(
                self.raw_signal, self.calibration, self._laser_nm(), n_pixels=len(self.raw_signal))
        except Exception:
            self._cal_health = None

    # ════════════════════════════════════════════════════════════════════════
    # Clean shutdown
    # ════════════════════════════════════════════════════════════════════════
    def closeEvent(self, event):
        self.live_running = False
        t = self._live_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        if self.cam_view_window is not None:
            try:
                self.cam_view_window.close()
            except Exception:
                pass
        cam, self.camera = self.camera, None
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass
        super().closeEvent(event)


# ── Live Camera View ────────────────────────────────────────────────────────
class CameraViewWindowQt(QDialog):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Live Camera View")
        self.resize(680, 540)
        v = QVBoxLayout(self)
        self.img_label = QLabel("Start 'Live' mode to see the camera feed.")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(320, 240)
        v.addWidget(self.img_label, 1)
        self.full_sensor_chk = QCheckBox("Show Full Sensor (Reset Hardware ROI)")
        self.full_sensor_chk.stateChanged.connect(self.on_full_sensor_toggle)
        v.addWidget(self.full_sensor_chk)
        self.info_label = QLabel("Start 'Live' mode to see the camera feed.")
        self.info_label.setStyleSheet(f"color:{TEXT};")
        v.addWidget(self.info_label)
        self.update_image()

    def on_full_sensor_toggle(self, _state):
        if self.app.camera is None:
            return
        try:
            if self.full_sensor_chk.isChecked():
                self.app.camera.set_hardware_roi()
            else:
                self.app.camera.restore_hardware_roi()
            self.app.replot()
        except Exception as e:
            QMessageBox.critical(self, "ROI Error", f"Failed to change hardware ROI: {e}")
            self.full_sensor_chk.blockSignals(True)
            self.full_sensor_chk.setChecked(not self.full_sensor_chk.isChecked())
            self.full_sensor_chk.blockSignals(False)

    def update_image(self):
        gray = None
        if self.app.camera and getattr(self.app.camera, "last_frame", None) is not None:
            gray = self.app.camera.last_frame
        if gray is None:
            self.info_label.setText("Waiting for first frame…" if self.app.live_running
                                    else "Camera idle. Click 'Live' to start feed.")
            return
        h, w = gray.shape
        gmin, gmax = float(gray.min()), float(gray.max())
        if gmax > gmin:
            img8 = ((gray - gmin) / (gmax - gmin) * 255.0).astype(np.uint8)
        else:
            img8 = np.zeros_like(gray, dtype=np.uint8)
        img8 = np.ascontiguousarray(img8)
        qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8).copy()
        pix = QPixmap.fromImage(qimg)

        try:
            roi_rows = int(self.app.roi_spin.value())
        except Exception:
            roi_rows = 0
        if 0 < roi_rows < h:
            cy = h // 2; r = roi_rows // 2
            top = max(0, cy - r); bot = min(h - 1, cy + r)
            painter = QPainter(pix)
            pen = QPen(QColor("#ff6b6b")); pen.setWidth(2); painter.setPen(pen)
            painter.drawRect(0, top, w - 1, bot - top)
            painter.end()

        target_w = max(320, self.img_label.width() - 4)
        self.img_label.setPixmap(pix.scaledToWidth(target_w, Qt.SmoothTransformation))

        sat_count = int(np.sum(gray >= 254 if gmax <= 255 else gray >= 4094))
        sat_msg = f"  |  Saturated: {sat_count} px" if sat_count > 0 else ""
        self.info_label.setText(f"Resolution: {w}x{h} px  |  ROI Rows: {roi_rows}  |  "
                                f"Range: {int(gmin)}-{int(gmax)}{sat_msg}")

    def closeEvent(self, event):
        if self.full_sensor_chk.isChecked() and self.app.camera is not None:
            try:
                self.app.camera.restore_hardware_roi()
            except Exception:
                pass
        self.app.cam_view_window = None
        super().closeEvent(event)


def run():
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 10))
    app.setStyleSheet(STYLE)
    win = RamanQtWindow()
    win.show()
    sys.exit(app.exec())
