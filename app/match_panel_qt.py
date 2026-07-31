# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V. (https://wfront.nl)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Match panel — PySide6 port of match_panel.py.

A dockable side panel that scores the current spectrum against a reference
library and overlays selected hits on the main plot. All scoring/library work
is reused from app.library / app.matching.
"""
import os
import threading

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QDialog, QLineEdit, QMessageBox,
    QFileDialog, QRadioButton, QButtonGroup,
)

from . import library as libmod
from . import library_download
from . import mol2raman

_BG, _BG_DARK, _ACCENT = "#16213e", "#1a1a2e", "#0f3460"
_CYAN, _TEXT, _MUTED = "#00d4ff", "#e0e0e0", "#888888"

_OVERLAY_COLORS = ["#e94560", "#a78bfa", "#34d399", "#fbbf24",
                   "#60a5fa", "#f472b6", "#fb923c", "#22d3ee"]


class MatchPanelQt(QDockWidget):
    """Right-side dock for spectrum identification."""

    # Thread → GUI marshaling.
    _status_sig = Signal(str)
    _loaded_sig = Signal(object, object)        # (library|None, error|None)
    _download_sig = Signal(object, object)      # (count|None, error|None)

    def __init__(self, app):
        super().__init__("Match Library")
        self.app = app
        self.library = None
        self._results = None
        self._busy = False
        self._pinned = []
        self._predicted = []        # list[mol2raman.PredictedSpectrum] overlays
        self._pending_on_done = None
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._build()
        self._status_sig.connect(self.status_label.setText)
        self._loaded_sig.connect(self._loaded)
        self._download_sig.connect(self._download_done)

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)

        self.status_label = QLabel("Library not loaded")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color:{_MUTED};")
        v.addWidget(self.status_label)

        self.run_btn = QPushButton("🔍 Match current spectrum")
        self.run_btn.setStyleSheet(f"background:{_ACCENT}; color:{_CYAN}; font-weight:bold; padding:5px;")
        self.run_btn.clicked.connect(self.run_match)
        v.addWidget(self.run_btn)

        hint = QLabel("Requires calibration (Raman shift axis).")
        hint.setStyleSheet(f"color:{_MUTED};")
        v.addWidget(hint)

        v.addWidget(QLabel("Top matches (HQI %)"))
        self.listbox = QListWidget()
        self.listbox.currentRowChanged.connect(self._on_select)
        v.addWidget(self.listbox, 1)

        self.detail_label = QLabel("—")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet(f"color:{_CYAN};")
        v.addWidget(self.detail_label)

        self.overlay_chk = QCheckBox("Overlay reference on plot")
        self.overlay_chk.setChecked(True)
        self.overlay_chk.stateChanged.connect(lambda *_: self.app.replot())
        v.addWidget(self.overlay_chk)

        v.addWidget(QLabel("Pinned overlays"))
        self.pinned_listbox = QListWidget()
        self.pinned_listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self.pinned_listbox.setMaximumHeight(90)
        v.addWidget(self.pinned_listbox)
        prow = QHBoxLayout()
        b_unpin = QPushButton("✕ Remove"); b_unpin.clicked.connect(self._unpin_selected)
        b_clear = QPushButton("Clear all"); b_clear.clicked.connect(self._unpin_all)
        prow.addWidget(b_unpin); prow.addWidget(b_clear); prow.addStretch(1)
        v.addLayout(prow)

        v.addWidget(QLabel("Predict from structure"))
        self.predict_btn = QPushButton("🧪  Predict spectrum from structure…")
        self.predict_btn.setStyleSheet("text-align:left; padding:3px;")
        self.predict_btn.clicked.connect(self.on_predict_structure)
        v.addWidget(self.predict_btn)
        self.predicted_listbox = QListWidget()
        self.predicted_listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self.predicted_listbox.setMaximumHeight(70)
        v.addWidget(self.predicted_listbox)
        prow2 = QHBoxLayout()
        b_pred_rm = QPushButton("✕ Remove"); b_pred_rm.clicked.connect(self._remove_predicted)
        b_pred_clr = QPushButton("Clear all"); b_pred_clr.clicked.connect(self._clear_predicted)
        prow2.addWidget(b_pred_rm); prow2.addWidget(b_pred_clr); prow2.addStretch(1)
        v.addLayout(prow2)

        v.addWidget(QLabel("Library"))
        for label, cmd in (
            ("📚  Browse all references…", self.on_browse_library),
            ("⬇  Download RRUFF subset…", self.on_download_rruff),
            ("📁  Add spectra from folder…", self.on_import_folder),
            ("🔄  Rebuild library index", self.on_rebuild_index),
            ("📂  Open references folder", self.on_open_refs_folder),
        ):
            b = QPushButton(label)
            b.setStyleSheet("text-align:left; padding:3px;")
            b.clicked.connect(cmd)
            v.addWidget(b)

        self.setWidget(w)

    # ── Public API used by the main window ─────────────────────────────────
    def ensure_library_loaded(self, on_done=None):
        if self.library is not None:
            if on_done is not None:
                on_done()
            return
        if self._busy:
            return
        self._busy = True
        self._pending_on_done = on_done
        self.status_label.setText("Loading library…")
        self.run_btn.setEnabled(False)

        def worker():
            try:
                lib = libmod.load_or_build_default(
                    progress=lambda i, n, name: self._status_sig.emit(f"Indexing {i}/{n}: {name}"))
            except Exception as e:
                self._loaded_sig.emit(None, str(e)); return
            self._loaded_sig.emit(lib, None)

        threading.Thread(target=worker, daemon=True).start()

    def selected_references(self):
        """(idx, meta, y_on_grid, grid, color) for every overlay to draw."""
        refs = []
        # Predicted-from-structure overlays are user-added explicitly, so they
        # show regardless of the library "Overlay reference" checkbox.
        for spec in self._predicted:
            color = _OVERLAY_COLORS[len(refs) % len(_OVERLAY_COLORS)]
            refs.append((None, spec, spec.intensity, spec.cm, color))
        if self.library is None or not self.overlay_chk.isChecked():
            return refs
        used = set()
        for i in self._pinned:
            if 0 <= i < len(self.library.entries):
                color = _OVERLAY_COLORS[len(refs) % len(_OVERLAY_COLORS)]
                refs.append((i, self.library.entries[i], self.library.matrix[i],
                             self.library.grid, color))
                used.add(i)
        if self._results is not None:
            row = self.listbox.currentRow()
            if row >= 0:
                _, top_idx = self._results
                if row < len(top_idx):
                    i = int(top_idx[row])
                    if i not in used:
                        color = _OVERLAY_COLORS[len(refs) % len(_OVERLAY_COLORS)]
                        refs.append((i, self.library.entries[i], self.library.matrix[i],
                                     self.library.grid, color))
        return refs

    def run_match(self):
        x, y = self.app.query_for_matching()
        if x is None:
            QMessageBox.information(self, "Match",
                "Matching requires a calibrated spectrum.\nRun Calibrate first, then try again.")
            return
        self.ensure_library_loaded(on_done=lambda: self._do_match(x, y))

    # ── Internals ──────────────────────────────────────────────────────────
    def _do_match(self, x, y):
        if self.library is None or self.library.is_empty():
            QMessageBox.information(self, "Match",
                "Reference library is empty.\n\nUse 'Download RRUFF subset…' to get started, "
                "or drop RRUFF .txt / two-column CSV files into the references folder and click "
                "'Rebuild library index'.")
            return
        try:
            scores, top = self.library.match(x, y, do_baseline=True, k=15)
        except Exception as e:
            QMessageBox.critical(self, "Match error", str(e)); return
        self._results = (scores, top)
        self._populate_listbox(scores, top)
        self.app.replot()

    def _populate_listbox(self, scores, top_idx):
        self.listbox.clear()
        for i in top_idx:
            m = self.library.entries[int(i)]
            self.listbox.addItem(f"{scores[int(i)]:5.1f}   {m.display()}")
        if len(top_idx) > 0:
            self.listbox.setCurrentRow(0)

    def _on_select(self, row):
        if row < 0 or self._results is None:
            return
        _, top = self._results
        if row >= len(top):
            return
        m = self.library.entries[int(top[row])]
        parts = [m.display()]
        if m.source:
            parts.append(f"Source: {m.source}" + (f" ({m.source_id})" if m.source_id else ""))
        if m.laser_nm:
            parts.append(f"Reference laser: {m.laser_nm:.0f} nm")
        self.detail_label.setText("\n".join(parts))
        self.app.replot()

    def _loaded(self, lib, err):
        self._busy = False
        self.run_btn.setEnabled(True)
        on_done, self._pending_on_done = self._pending_on_done, None
        if err:
            self.status_label.setText(f"Load failed: {err}")
            self.library = None
            return
        self.library = lib
        if lib.is_empty():
            self.status_label.setText("Library empty — download a bundle to begin.")
        else:
            self.status_label.setText(f"Library loaded — {len(lib)} references.")
        if on_done is not None:
            on_done()

    # ── Library management ─────────────────────────────────────────────────
    def on_download_rruff(self):
        if self._busy:
            return
        dlg = _RruffPickerDialogQt(self)
        if dlg.exec() != QDialog.Accepted or dlg.result_key is None:
            return
        key = dlg.result_key
        self._busy = True
        self.run_btn.setEnabled(False)
        self.status_label.setText("Starting download…")
        dest = libmod.default_library_dir()

        def worker():
            try:
                count = library_download.download_rruff_bundle(
                    key, dest,
                    progress=lambda stage, cur, tot, msg: self._status_sig.emit(
                        f"{stage}: {cur}/{tot}  {os.path.basename(msg)[:32]}" if tot
                        else f"{stage}: {cur} bytes"),
                    processed_only=True)
            except Exception as e:
                self._download_sig.emit(None, str(e)); return
            self._download_sig.emit(count, None)

        threading.Thread(target=worker, daemon=True).start()

    def _download_done(self, count, err):
        self._busy = False
        self.run_btn.setEnabled(True)
        if err:
            self.status_label.setText("Download failed.")
            QMessageBox.critical(self, "RRUFF download", err)
            return
        self.status_label.setText(f"Downloaded {count} files. Rebuilding index…")
        libmod.invalidate_cache()
        self.library = None
        self.ensure_library_loaded()

    def on_import_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Add reference spectra from folder")
        if not path:
            return
        import shutil
        dest = libmod.default_library_dir()
        os.makedirs(dest, exist_ok=True)
        copied = 0
        for root, _, names in os.walk(path):
            for n in names:
                if n.lower().endswith((".txt", ".csv", ".tsv")):
                    src = os.path.join(root, n); dst = os.path.join(dest, n)
                    if os.path.abspath(src) == os.path.abspath(dst):
                        continue
                    try:
                        shutil.copy2(src, dst); copied += 1
                    except OSError:
                        pass
        self.status_label.setText(f"Imported {copied} files. Rebuilding…")
        libmod.invalidate_cache()
        self.library = None
        self.ensure_library_loaded()

    def on_rebuild_index(self):
        libmod.invalidate_cache()
        self.library = None
        self.ensure_library_loaded()

    def on_browse_library(self):
        if self._busy:
            return
        if self.library is None:
            self.ensure_library_loaded(on_done=self._open_browser)
            return
        self._open_browser()

    def _open_browser(self):
        if self.library is None or self.library.is_empty():
            QMessageBox.information(self, "Browse",
                "Library is empty. Download a bundle or drop reference files into the "
                "references folder first.")
            return
        _LibraryBrowserDialogQt(self).exec()

    def on_open_refs_folder(self):
        d = libmod.default_library_dir()
        os.makedirs(d, exist_ok=True)
        try:
            import subprocess, sys
            if sys.platform == "darwin":
                subprocess.Popen(["open", d])
            elif sys.platform == "win32":
                os.startfile(d)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            QMessageBox.critical(self, "Open folder", str(e))

    # ── Pinned overlays ────────────────────────────────────────────────────
    def pin_indices(self, indices):
        added = 0
        for i in indices:
            i = int(i)
            if 0 <= i < len(self.library.entries) and i not in self._pinned:
                self._pinned.append(i); added += 1
        if added:
            self._refresh_pinned_list(); self.app.replot()
        return added

    def _refresh_pinned_list(self):
        self.pinned_listbox.clear()
        if self.library is None:
            return
        for k, i in enumerate(self._pinned):
            color = _OVERLAY_COLORS[k % len(_OVERLAY_COLORS)]
            item = QListWidgetItem(f"●  {self.library.entries[i].display()}")
            item.setForeground(QColor(color))
            self.pinned_listbox.addItem(item)

    def _unpin_selected(self):
        rows = sorted((self.pinned_listbox.row(i) for i in self.pinned_listbox.selectedItems()),
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._pinned):
                del self._pinned[r]
        self._refresh_pinned_list(); self.app.replot()

    def _unpin_all(self):
        if not self._pinned:
            return
        self._pinned.clear()
        self._refresh_pinned_list(); self.app.replot()

    # ── Predicted-from-structure overlays ──────────────────────────────────
    def on_predict_structure(self):
        _PredictDialogQt(self).exec()

    def add_predicted(self, spec):
        """Add a mol2raman.PredictedSpectrum as a plot overlay."""
        self._predicted.append(spec)
        self._refresh_predicted_list()
        self.app.replot()

    def _refresh_predicted_list(self):
        self.predicted_listbox.clear()
        for k, spec in enumerate(self._predicted):
            color = _OVERLAY_COLORS[k % len(_OVERLAY_COLORS)]
            item = QListWidgetItem(f"●  {spec.display()}")
            item.setForeground(QColor(color))
            self.predicted_listbox.addItem(item)

    def _remove_predicted(self):
        rows = sorted((self.predicted_listbox.row(i)
                       for i in self.predicted_listbox.selectedItems()), reverse=True)
        for r in rows:
            if 0 <= r < len(self._predicted):
                del self._predicted[r]
        self._refresh_predicted_list(); self.app.replot()

    def _clear_predicted(self):
        if not self._predicted:
            return
        self._predicted.clear()
        self._refresh_predicted_list(); self.app.replot()


class _RruffPickerDialogQt(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Download RRUFF bundle")
        self.result_key = None
        v = QVBoxLayout(self)
        v.addWidget(QLabel("Choose a RRUFF bundle to download:"))
        self._group = QButtonGroup(self)
        first = True
        for key, (fname, size, descr) in library_download.list_rruff_bundles():
            rb = QRadioButton(f"{key}   ({size})  —  {descr}")
            rb.setProperty("bundle_key", key)
            if first:
                rb.setChecked(True); first = False
            self._group.addButton(rb)
            v.addWidget(rb)
        row = QHBoxLayout()
        b_cancel = QPushButton("Cancel"); b_cancel.clicked.connect(self.reject)
        b_ok = QPushButton("Download"); b_ok.clicked.connect(self._ok)
        b_ok.setStyleSheet(f"background:{_CYAN}; color:#000; font-weight:bold;")
        row.addStretch(1); row.addWidget(b_cancel); row.addWidget(b_ok)
        v.addLayout(row)

    def _ok(self):
        btn = self._group.checkedButton()
        if btn is not None:
            self.result_key = btn.property("bundle_key")
        self.accept()


class _LibraryBrowserDialogQt(QDialog):
    """Searchable picker over the full library; selections pin as overlays."""
    _MAX_VISIBLE = 2000

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.setWindowTitle(f"Browse references — {len(panel.library)} entries")
        self.resize(560, 620)
        self._all_items = []
        for i, m in enumerate(panel.library.entries):
            blob = f"{m.name} {m.formula} {m.source} {m.source_id}".lower()
            self._all_items.append((i, m.display(), blob))
        self._filtered = list(self._all_items)

        v = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Search:"))
        self.search = QLineEdit()
        self.search.textChanged.connect(self._on_search)
        top.addWidget(self.search, 1)
        v.addLayout(top)

        self.count_label = QLabel(""); self.count_label.setStyleSheet(f"color:{_MUTED};")
        v.addWidget(self.count_label)

        self.listbox = QListWidget()
        self.listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self.listbox.itemDoubleClicked.connect(lambda *_: self._pin())
        v.addWidget(self.listbox, 1)

        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color:{_MUTED};")
        v.addWidget(self.status_label)

        row = QHBoxLayout()
        b_pin = QPushButton("📌 Pin selection"); b_pin.clicked.connect(self._pin)
        b_pin.setStyleSheet(f"background:{_CYAN}; color:#000; font-weight:bold;")
        b_clear = QPushButton("Unpin all"); b_clear.clicked.connect(self._clear_all)
        b_close = QPushButton("Close"); b_close.clicked.connect(self.accept)
        row.addWidget(b_pin); row.addWidget(b_clear); row.addStretch(1); row.addWidget(b_close)
        v.addLayout(row)

        self._populate()
        self._refresh_status()

    def _on_search(self, _text):
        q = self.search.text().lower().strip()
        if not q:
            self._filtered = list(self._all_items)
        else:
            terms = q.split()
            self._filtered = [it for it in self._all_items if all(t in it[2] for t in terms)]
        self._populate()

    def _populate(self):
        self.listbox.clear()
        for _, name, _blob in self._filtered[:self._MAX_VISIBLE]:
            self.listbox.addItem(name)
        total = len(self._filtered); shown = min(total, self._MAX_VISIBLE)
        if total > self._MAX_VISIBLE:
            self.count_label.setText(f"Showing first {shown} of {total} matches — refine your search.")
        else:
            self.count_label.setText(f"{total} matching")

    def _pin(self):
        rows = [self.listbox.row(i) for i in self.listbox.selectedItems()]
        indices = [self._filtered[r][0] for r in rows if r < len(self._filtered)]
        if not indices:
            return
        added = self.panel.pin_indices(indices)
        self.status_label.setText("Already pinned." if added == 0 else "")
        if added:
            self._refresh_status(added)

    def _clear_all(self):
        self.panel._unpin_all()
        self._refresh_status()

    def _refresh_status(self, just_added=0):
        n = len(self.panel._pinned)
        if n == 0:
            self.status_label.setText("No references pinned.")
            return
        names = [self.panel.library.entries[i].name for i in self.panel._pinned[:3]]
        suffix = "" if n <= 3 else f"  (+{n - 3} more)"
        prefix = f"Added {just_added}. " if just_added else ""
        self.status_label.setText(f"{prefix}{n} pinned: " + ", ".join(names) + suffix)


class _PredictDialogQt(QDialog):
    """Predict a Raman spectrum from a SMILES / compound name / formula.

    Shows the prediction in its own preview plot (works with no spectrum loaded)
    and can push it onto the main plot as an overlay via the parent panel.
    """
    _done_sig = Signal(object, object)   # (PredictedSpectrum|None, error_str|None)

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self._spec = None
        self.setWindowTitle("Predict spectrum from structure")
        self.resize(640, 520)

        v = QVBoxLayout(self)

        avail = mol2raman.availability()
        if avail.ready:
            banner = "✓ Mol2Raman model ready — real predictions enabled."
            banner_col = "#34d399"
        else:
            extra = ("Install the ML stack, then add the checkpoints (see "
                     "models/mol2raman/README.md). " if avail.state == "deps_missing"
                     else "Add the authors' checkpoints to models/mol2raman/. ")
            banner = "⚠ " + avail.detail + "\n" + extra + "Showing a DEMO spectrum (not a real prediction)."
            banner_col = "#fbbf24"
        self.banner = QLabel(banner)
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(f"color:{banner_col};")
        v.addWidget(self.banner)

        row = QHBoxLayout()
        row.addWidget(QLabel("Structure:"))
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("SMILES (CC(=O)Oc1ccccc1C(=O)O), name (aspirin), or formula (C9H8O4)")
        self.entry.returnPressed.connect(self._predict)
        row.addWidget(self.entry, 1)
        self.go_btn = QPushButton("Predict")
        self.go_btn.setStyleSheet(f"background:{_CYAN}; color:#000; font-weight:bold;")
        self.go_btn.clicked.connect(self._predict)
        row.addWidget(self.go_btn)
        v.addLayout(row)

        self.status = QLabel("Enter a structure and press Predict.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{_MUTED};")
        v.addWidget(self.status)

        self.fig = Figure(facecolor=_BG_DARK, figsize=(5, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self._style_ax()
        v.addWidget(self.canvas, 1)

        brow = QHBoxLayout()
        self.overlay_btn = QPushButton("➕ Overlay on main plot")
        self.overlay_btn.clicked.connect(self._overlay)
        self.overlay_btn.setEnabled(False)
        b_close = QPushButton("Close"); b_close.clicked.connect(self.accept)
        brow.addWidget(self.overlay_btn); brow.addStretch(1); brow.addWidget(b_close)
        v.addLayout(brow)

        self._done_sig.connect(self._on_done)

    def _style_ax(self):
        self.ax.set_facecolor(_BG_DARK)
        for spine in self.ax.spines.values():
            spine.set_color(_MUTED)
        self.ax.tick_params(colors=_MUTED, labelsize=8)
        self.ax.set_xlabel("Raman Shift (cm⁻¹)", color=_TEXT, fontsize=9)
        self.ax.set_ylabel("Intensity (a.u.)", color=_TEXT, fontsize=9)
        self.ax.grid(True, color=_ACCENT, linestyle="--", linewidth=0.5, alpha=0.5)

    def _predict(self):
        text = self.entry.text().strip()
        if not text:
            return
        self.go_btn.setEnabled(False)
        self.overlay_btn.setEnabled(False)
        self.status.setText("Predicting…  (resolving structure, running model)")

        def worker():
            try:
                spec = mol2raman.predict(text, allow_demo=True)
            except Exception as e:
                self._done_sig.emit(None, str(e)); return
            self._done_sig.emit(spec, None)

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, spec, err):
        self.go_btn.setEnabled(True)
        if err:
            self.status.setText(f"Could not predict: {err}")
            return
        self._spec = spec
        self.overlay_btn.setEnabled(True)

        res = getattr(spec, "resolve", None)
        bits = [f"SMILES: {spec.smiles}"]
        if res is not None and res.name and res.name != spec.smiles:
            bits.append(f"({res.name})")
        if spec.source == "demo":
            bits.append("— DEMO output, not a real prediction")
        if res is not None and getattr(res, "ambiguous", False):
            bits.append("\n⚠ " + res.note)
        self.status.setText("  ".join(bits))

        self.ax.cla(); self._style_ax()
        col = "#fbbf24" if spec.source == "demo" else _CYAN
        self.ax.plot(spec.cm, spec.intensity, color=col, linewidth=1.2)
        self.ax.set_title(spec.display(), color=col, fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _overlay(self):
        if self._spec is None:
            return
        self.panel.add_predicted(self._spec)
        self.status.setText("Added to main plot overlay. "
                            "(Switch the X-axis to Raman Shifts to see it.)")
