# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""
Match panel: side panel that scores the current spectrum against a
reference library and shows the top hits with HQI scores. Selecting a hit
overlays its reference spectrum on the main plot.
"""
from __future__ import annotations
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from . import library as libmod
from . import library_download


# Colour constants are imported from gui at runtime to keep this module
# independent of the live theme.
_BG = "#16213e"
_BG_DARK = "#1a1a2e"
_ACCENT = "#0f3460"
_CYAN = "#00d4ff"
_ORANGE = "#e94560"
_TEXT = "#e0e0e0"
_MUTED = "#888888"

# Distinct colors cycled through for stacked overlays. First entry matches
# the legacy single-overlay color so behavior is unchanged when nothing
# extra is pinned.
_OVERLAY_COLORS = [
    "#e94560",  # red/orange (legacy)
    "#a78bfa",  # violet
    "#34d399",  # green
    "#fbbf24",  # amber
    "#60a5fa",  # blue
    "#f472b6",  # pink
    "#fb923c",  # orange
    "#22d3ee",  # cyan
]


class MatchPanel(tk.Frame):
    """
    Right-side panel for spectrum identification. Owned by RamanApp; calls
    back into the app to trigger re-plots and read the current spectrum.
    """

    def __init__(self, parent, app, width=300):
        super().__init__(parent, bg=_BG, width=width)
        self.app = app
        self.pack_propagate(False)

        self.library = None
        self._results = None   # (scores ndarray, top_idx ndarray)
        self._busy = False
        self._pinned: list[int] = []  # library indices pinned for overlay

        self._build()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build(self):
        header = tk.Frame(self, bg=_ACCENT)
        header.pack(fill="x")
        tk.Label(header, text="🔬  Match Library", bg=_ACCENT, fg=_CYAN,
                 font=("Helvetica", 10, "bold"), pady=6).pack(side="left", padx=8)
        tk.Button(header, text="✕", bg=_ACCENT, fg=_TEXT, relief="flat",
                  font=("Helvetica", 10, "bold"), command=self.app.on_match_toggle
                  ).pack(side="right", padx=4)

        # Status / library size
        self.status_var = tk.StringVar(value="Library not loaded")
        tk.Label(self, textvariable=self.status_var, bg=_BG, fg=_MUTED,
                 font=("Helvetica", 8), anchor="w").pack(fill="x", padx=8, pady=(8, 2))

        # Run button
        self.run_btn = tk.Button(self, text="🔍 Match current spectrum",
                                  bg=_ACCENT, fg=_CYAN, relief="flat",
                                  font=("Helvetica", 10, "bold"),
                                  activebackground=_BG_DARK,
                                  activeforeground=_CYAN,
                                  command=self.run_match, pady=4)
        self.run_btn.pack(fill="x", padx=8, pady=(4, 2))

        # Hint
        tk.Label(self, text="Requires calibration (Raman shift axis).",
                 bg=_BG, fg=_MUTED, font=("Helvetica", 8)
                 ).pack(anchor="w", padx=8, pady=(0, 6))

        # Results listbox
        tk.Label(self, text="Top matches (HQI %)", bg=_BG, fg=_TEXT,
                 font=("Helvetica", 9, "bold"), anchor="w"
                 ).pack(fill="x", padx=8)
        list_frame = tk.Frame(self, bg=_BG)
        list_frame.pack(fill="both", expand=True, padx=8, pady=(2, 6))

        self.listbox = tk.Listbox(list_frame, bg=_BG_DARK, fg=_TEXT,
                                   selectbackground=_ACCENT, selectforeground=_CYAN,
                                   relief="flat", font=("Menlo", 10),
                                   activestyle="none", exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Detail box
        self.detail_var = tk.StringVar(value="—")
        tk.Label(self, textvariable=self.detail_var, bg=_BG, fg=_CYAN,
                 font=("Helvetica", 9), anchor="w", justify="left",
                 wraplength=280
                 ).pack(fill="x", padx=8, pady=(0, 6))

        # Overlay toggle
        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Overlay reference on plot",
                        variable=self.overlay_var,
                        command=self.app._replot).pack(anchor="w", padx=8)

        # Pinned overlays (multiple references stacked on the plot)
        pinned_box = tk.Frame(self, bg=_BG)
        pinned_box.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(pinned_box, text="Pinned overlays", bg=_BG, fg=_MUTED,
                 font=("Helvetica", 8, "bold"), anchor="w"
                 ).pack(fill="x")
        self.pinned_listbox = tk.Listbox(
            pinned_box, bg=_BG_DARK, fg=_TEXT,
            selectbackground=_ACCENT, selectforeground=_CYAN,
            relief="flat", font=("Menlo", 9), height=4,
            activestyle="none", exportselection=False)
        self.pinned_listbox.pack(fill="x", pady=(2, 2))
        pinned_btns = tk.Frame(pinned_box, bg=_BG)
        pinned_btns.pack(fill="x")
        tk.Button(pinned_btns, text="✕ Remove", bg=_BG, fg=_TEXT,
                  relief="flat", font=("Helvetica", 8),
                  activebackground=_ACCENT, activeforeground=_CYAN,
                  command=self._unpin_selected).pack(side="left")
        tk.Button(pinned_btns, text="Clear all", bg=_BG, fg=_TEXT,
                  relief="flat", font=("Helvetica", 8),
                  activebackground=_ACCENT, activeforeground=_CYAN,
                  command=self._unpin_all).pack(side="left", padx=4)

        # Library management
        tk.Frame(self, bg=_ACCENT, height=1).pack(fill="x", padx=8, pady=8)
        tk.Label(self, text="Library", bg=_BG, fg=_CYAN,
                 font=("Helvetica", 9, "bold"), anchor="w"
                 ).pack(fill="x", padx=8)

        for label, cmd in (
            ("📚  Browse all references…", self.on_browse_library),
            ("⬇  Download RRUFF subset…", self.on_download_rruff),
            ("📁  Add spectra from folder…", self.on_import_folder),
            ("🔄  Rebuild library index",   self.on_rebuild_index),
            ("📂  Open references folder",  self.on_open_refs_folder),
        ):
            tk.Button(self, text=label, bg=_BG, fg=_TEXT, relief="flat",
                      font=("Helvetica", 9), anchor="w", padx=8, pady=2,
                      activebackground=_ACCENT, activeforeground=_CYAN,
                      command=cmd).pack(fill="x", padx=8, pady=1)

    # ── Public API ────────────────────────────────────────────────────────

    def ensure_library_loaded(self, on_done=None):
        """Load the library on a background thread if not already loaded."""
        if self.library is not None:
            if on_done is not None:
                on_done()
            return
        if self._busy:
            return
        self._busy = True
        self.status_var.set("Loading library…")
        self.run_btn.configure(state="disabled")

        def worker():
            try:
                lib = libmod.load_or_build_default(
                    progress=lambda i, n, name: self._post_status(
                        f"Indexing {i}/{n}: {name}"))
            except Exception as e:
                self._post(lambda: self._loaded(None, str(e), on_done))
                return
            self._post(lambda: self._loaded(lib, None, on_done))

        threading.Thread(target=worker, daemon=True).start()

    def selected_references(self):
        """Return all references to overlay, as a list of
        (idx, ReferenceMeta, y_on_grid, grid, color) tuples.

        Combines explicitly pinned references with the currently-highlighted
        top-match hit (if any and not already pinned). The overlay master
        toggle suppresses the whole list when off.
        """
        if self.library is None or not self.overlay_var.get():
            return []
        refs = []
        used = set()
        for i in self._pinned:
            if 0 <= i < len(self.library.entries):
                color = _OVERLAY_COLORS[len(refs) % len(_OVERLAY_COLORS)]
                refs.append((i, self.library.entries[i],
                             self.library.matrix[i], self.library.grid, color))
                used.add(i)
        if self._results is not None:
            sel = self.listbox.curselection()
            if sel:
                _, top_idx = self._results
                rank = sel[0]
                if rank < len(top_idx):
                    i = int(top_idx[rank])
                    if i not in used:
                        color = _OVERLAY_COLORS[len(refs) % len(_OVERLAY_COLORS)]
                        refs.append((i, self.library.entries[i],
                                     self.library.matrix[i],
                                     self.library.grid, color))
        return refs

    def run_match(self):
        x, y = self.app.query_for_matching()
        if x is None:
            messagebox.showinfo("Match",
                "Matching requires a calibrated spectrum.\n"
                "Run Calibrate first, then try again.")
            return

        self.ensure_library_loaded(on_done=lambda: self._do_match(x, y))

    # ── Internals ─────────────────────────────────────────────────────────

    def _do_match(self, x, y):
        if self.library is None or self.library.is_empty():
            messagebox.showinfo("Match",
                "Reference library is empty.\n\n"
                "Use 'Download RRUFF subset…' to get started, or drop your own "
                "RRUFF .txt / two-column CSV files into the references folder "
                "and click 'Rebuild library index'.")
            return
        try:
            scores, top = self.library.match(x, y, do_baseline=True, k=15)
        except Exception as e:
            messagebox.showerror("Match error", str(e))
            return

        self._results = (scores, top)
        self._populate_listbox(scores, top)
        self.app._replot()

    def _populate_listbox(self, scores, top_idx):
        self.listbox.delete(0, "end")
        for rank, i in enumerate(top_idx):
            m = self.library.entries[int(i)]
            label = f"{scores[int(i)]:5.1f}   {m.display()}"
            self.listbox.insert("end", label)
        if len(top_idx) > 0:
            self.listbox.selection_set(0)
            self._on_select(None)

    def _on_select(self, _event):
        sel = self.listbox.curselection()
        if not sel or self._results is None:
            return
        _, top = self._results
        i = int(top[sel[0]])
        m = self.library.entries[i]
        parts = [m.display()]
        if m.source:
            parts.append(f"Source: {m.source}" + (f" ({m.source_id})" if m.source_id else ""))
        if m.laser_nm:
            parts.append(f"Reference laser: {m.laser_nm:.0f} nm")
        self.detail_var.set("\n".join(parts))
        self.app._replot()

    # ── Background-thread helpers ─────────────────────────────────────────

    def _post(self, fn):
        self.app.root.after(0, fn)

    def _post_status(self, text):
        self._post(lambda: self.status_var.set(text))

    def _loaded(self, lib, err, on_done):
        self._busy = False
        self.run_btn.configure(state="normal")
        if err:
            self.status_var.set(f"Load failed: {err}")
            self.library = None
            return
        self.library = lib
        if lib.is_empty():
            self.status_var.set("Library empty — download a bundle to begin.")
        else:
            self.status_var.set(f"Library loaded — {len(lib)} references.")
        if on_done is not None:
            on_done()

    # ── Library management actions ────────────────────────────────────────

    def on_download_rruff(self):
        if self._busy:
            return
        dlg = _RruffPickerDialog(self.app.root)
        self.app.root.wait_window(dlg)
        if dlg.result is None:
            return
        key = dlg.result

        self._busy = True
        self.run_btn.configure(state="disabled")
        self.status_var.set("Starting download…")

        dest = libmod.default_library_dir()

        def worker():
            try:
                count = library_download.download_rruff_bundle(
                    key, dest,
                    progress=lambda stage, cur, tot, msg: self._post_status(
                        f"{stage}: {cur}/{tot}  {os.path.basename(msg)[:32]}"
                            if tot else f"{stage}: {cur} bytes"),
                    processed_only=True)
            except Exception as e:
                self._post(lambda: self._download_done(None, str(e)))
                return
            self._post(lambda: self._download_done(count, None))

        threading.Thread(target=worker, daemon=True).start()

    def _download_done(self, count, err):
        self._busy = False
        self.run_btn.configure(state="normal")
        if err:
            self.status_var.set("Download failed.")
            messagebox.showerror("RRUFF download", err)
            return
        self.status_var.set(f"Downloaded {count} files. Rebuilding index…")
        libmod.invalidate_cache()
        self.library = None
        self.ensure_library_loaded()

    def on_import_folder(self):
        path = filedialog.askdirectory(title="Add reference spectra from folder")
        if not path:
            return
        # Copy/link by reference: just symlink/copy into the references dir.
        # Simpler: just point the library scanner at this folder by adding a
        # symlink. To avoid surprises, we copy file paths via the rebuild.
        import shutil
        dest = libmod.default_library_dir()
        os.makedirs(dest, exist_ok=True)
        copied = 0
        for root, _, names in os.walk(path):
            for n in names:
                if n.lower().endswith((".txt", ".csv", ".tsv")):
                    src = os.path.join(root, n)
                    dst = os.path.join(dest, n)
                    if os.path.abspath(src) == os.path.abspath(dst):
                        continue
                    try:
                        shutil.copy2(src, dst)
                        copied += 1
                    except OSError:
                        pass
        self.status_var.set(f"Imported {copied} files. Rebuilding…")
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
            messagebox.showinfo("Browse",
                "Library is empty. Download a bundle or drop reference "
                "files into the references folder first.")
            return
        _LibraryBrowserDialog(self.app.root, self)

    # ── Pinned overlay helpers ────────────────────────────────────────────

    def pin_indices(self, indices):
        """Add library indices to the pinned-overlay set (de-duped)."""
        added = 0
        for i in indices:
            i = int(i)
            if 0 <= i < len(self.library.entries) and i not in self._pinned:
                self._pinned.append(i)
                added += 1
        if added:
            self._refresh_pinned_list()
            self.app._replot()
        return added

    def _refresh_pinned_list(self):
        self.pinned_listbox.delete(0, "end")
        if self.library is None:
            return
        for k, i in enumerate(self._pinned):
            color = _OVERLAY_COLORS[k % len(_OVERLAY_COLORS)]
            m = self.library.entries[i]
            self.pinned_listbox.insert("end", f"●  {m.display()}")
            self.pinned_listbox.itemconfig(
                self.pinned_listbox.size() - 1, fg=color)

    def _unpin_selected(self):
        sel = list(self.pinned_listbox.curselection())
        if not sel:
            return
        for rank in sorted(sel, reverse=True):
            if 0 <= rank < len(self._pinned):
                del self._pinned[rank]
        self._refresh_pinned_list()
        self.app._replot()

    def _unpin_all(self):
        if not self._pinned:
            return
        self._pinned.clear()
        self._refresh_pinned_list()
        self.app._replot()

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
            messagebox.showerror("Open folder", str(e))


class _RruffPickerDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Download RRUFF bundle")
        self.configure(bg=_BG_DARK)
        self.result = None
        self.resizable(False, False)

        tk.Label(self, text="Choose a RRUFF bundle to download:",
                 bg=_BG_DARK, fg=_TEXT, font=("Helvetica", 11)
                 ).pack(padx=20, pady=(14, 8))

        self.selection = tk.StringVar(value="fair_oriented")
        for key, (fname, size, descr) in library_download.list_rruff_bundles():
            label = f"{key}   ({size})  —  {descr}"
            tk.Radiobutton(self, text=label, variable=self.selection,
                           value=key, bg=_BG_DARK, fg=_TEXT,
                           selectcolor=_ACCENT, activebackground=_BG_DARK,
                           anchor="w", font=("Helvetica", 9)
                           ).pack(fill="x", padx=20, anchor="w")

        btns = tk.Frame(self, bg=_BG_DARK)
        btns.pack(pady=12)
        tk.Button(btns, text="Cancel", command=self.destroy,
                  bg=_ACCENT, fg=_TEXT, relief="flat", padx=12, pady=3
                  ).pack(side="left", padx=4)
        tk.Button(btns, text="Download", command=self._ok,
                  bg=_CYAN, fg="#000", relief="flat",
                  font=("Helvetica", 10, "bold"), padx=14, pady=3
                  ).pack(side="left", padx=4)
        self.grab_set()

    def _ok(self):
        self.result = self.selection.get()
        self.destroy()


class _LibraryBrowserDialog(tk.Toplevel):
    """Searchable picker over the full reference library. Selected entries
    are pinned to the main plot as overlays."""

    _MAX_VISIBLE = 2000  # cap list size so 5k-entry libraries stay responsive

    def __init__(self, parent, panel):
        super().__init__(parent)
        self.panel = panel
        self.title(f"Browse references — {len(panel.library)} entries")
        self.configure(bg=_BG_DARK)
        self.geometry("560x620")
        self.minsize(420, 400)

        # Pre-compute (index, display, search_blob) for cheap filtering.
        self._all_items = []
        for i, m in enumerate(panel.library.entries):
            blob = f"{m.name} {m.formula} {m.source} {m.source_id}".lower()
            self._all_items.append((i, m.display(), blob))
        self._filtered = list(self._all_items)

        # Search row
        top = tk.Frame(self, bg=_BG_DARK)
        top.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(top, text="Search:", bg=_BG_DARK, fg=_TEXT,
                 font=("Helvetica", 10)).pack(side="left")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search)
        ent = tk.Entry(top, textvariable=self.search_var, bg=_BG, fg=_TEXT,
                       insertbackground=_TEXT, relief="flat",
                       font=("Helvetica", 10))
        ent.pack(side="left", fill="x", expand=True, padx=(6, 0), ipady=3)
        ent.focus_set()

        self.count_var = tk.StringVar()
        tk.Label(self, textvariable=self.count_var, bg=_BG_DARK, fg=_MUTED,
                 font=("Helvetica", 9), anchor="w"
                 ).pack(fill="x", padx=10, pady=(0, 2))

        # Listbox + scrollbar
        frame = tk.Frame(self, bg=_BG_DARK)
        frame.pack(fill="both", expand=True, padx=10, pady=4)
        self.listbox = tk.Listbox(
            frame, bg=_BG_DARK, fg=_TEXT,
            selectbackground=_ACCENT, selectforeground=_CYAN,
            relief="flat", font=("Menlo", 10),
            selectmode="extended", activestyle="none",
            exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, orient="vertical",
                           command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<Double-Button-1>", lambda e: self._pin())

        self.status_var = tk.StringVar()
        tk.Label(self, textvariable=self.status_var, bg=_BG_DARK, fg=_MUTED,
                 font=("Helvetica", 9), anchor="w", wraplength=540,
                 justify="left").pack(fill="x", padx=10, pady=(2, 0))

        # Action row
        btns = tk.Frame(self, bg=_BG_DARK)
        btns.pack(fill="x", padx=10, pady=(6, 10))
        tk.Button(btns, text="📌 Pin selection", command=self._pin,
                  bg=_CYAN, fg="#000", relief="flat", padx=12, pady=4,
                  font=("Helvetica", 10, "bold")
                  ).pack(side="left")
        tk.Button(btns, text="Unpin all", command=self._clear_all,
                  bg=_ACCENT, fg=_TEXT, relief="flat", padx=10, pady=4
                  ).pack(side="left", padx=6)
        tk.Button(btns, text="Close", command=self.destroy,
                  bg=_ACCENT, fg=_TEXT, relief="flat", padx=10, pady=4
                  ).pack(side="right")

        self._populate()
        self._refresh_status()

        self.bind("<Return>", lambda e: self._pin())
        self.bind("<Escape>", lambda e: self.destroy())
        self.transient(parent)
        self.grab_set()

    def _on_search(self, *_):
        q = self.search_var.get().lower().strip()
        if not q:
            self._filtered = list(self._all_items)
        else:
            terms = q.split()
            self._filtered = [it for it in self._all_items
                              if all(t in it[2] for t in terms)]
        self._populate()

    def _populate(self):
        self.listbox.delete(0, "end")
        for _, name, _blob in self._filtered[:self._MAX_VISIBLE]:
            self.listbox.insert("end", name)
        total = len(self._filtered)
        shown = min(total, self._MAX_VISIBLE)
        if total > self._MAX_VISIBLE:
            self.count_var.set(f"Showing first {shown} of {total} matches "
                               f"— refine your search to narrow down.")
        else:
            self.count_var.set(f"{total} matching")

    def _pin(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        indices = [self._filtered[r][0] for r in sel]
        added = self.panel.pin_indices(indices)
        if added == 0:
            self.status_var.set("Already pinned.")
        else:
            self._refresh_status(added)

    def _clear_all(self):
        self.panel._unpin_all()
        self._refresh_status()

    def _refresh_status(self, just_added=0):
        n = len(self.panel._pinned)
        if n == 0:
            self.status_var.set("No references pinned.")
            return
        names = [self.panel.library.entries[i].name
                 for i in self.panel._pinned[:3]]
        suffix = "" if n <= 3 else f"  (+{n - 3} more)"
        prefix = f"Added {just_added}. " if just_added else ""
        self.status_var.set(f"{prefix}{n} pinned: " + ", ".join(names) + suffix)
