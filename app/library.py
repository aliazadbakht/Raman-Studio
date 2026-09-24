# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""
Reference spectrum library.

A library is a collection of pre-normalized reference spectra resampled
onto a common cm⁻¹ grid, plus the metadata describing each entry. The
matrix form (N references × G grid points) lets us score a query against
the whole library with one matrix multiply.

Supported on-disk source formats:
  * RRUFF ``.txt``  — ``##KEY=VALUE`` header lines followed by
    ``wavenumber, intensity`` rows. Both "Processed" and "RAW" files are
    accepted; Processed are preferred because their baselines are already
    removed.
  * CSV / plain ``.txt`` two-column ``cm⁻¹, intensity``. Optional header.

Libraries are cached as ``.npz`` for fast subsequent loads.
"""
from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np

from . import matching


# ── Filesystem locations ──────────────────────────────────────────────────────

def app_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_library_dir():
    """Where reference source files (RRUFF/CSV) live."""
    return os.path.join(app_root(), "references")


def default_cache_path():
    """Where the compiled library matrix lives."""
    return os.path.join(app_root(), "references", "library_cache.npz")


# ── Metadata ──────────────────────────────────────────────────────────────────

@dataclass
class ReferenceMeta:
    name: str
    formula: str = ""
    source: str = ""
    source_id: str = ""
    laser_nm: float = 0.0
    n_points: int = 0
    cm_min: float = 0.0
    cm_max: float = 0.0
    path: str = ""

    def display(self) -> str:
        if self.formula:
            return f"{self.name}  [{self.formula}]"
        return self.name


# ── Parsers ───────────────────────────────────────────────────────────────────

_RRUFF_HEADER = re.compile(r"^##([^=]+)=(.*)$")


def parse_rruff_txt(path):
    """Return (cm, intensity, ReferenceMeta) from a RRUFF .txt file."""
    headers = {}
    xs, ys = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = _RRUFF_HEADER.match(line)
            if m:
                headers[m.group(1).strip().upper()] = m.group(2).strip()
                continue
            parts = [p.strip() for p in re.split(r"[,\s]+", line) if p.strip()]
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue

    if not xs:
        raise ValueError(f"No numeric data in RRUFF file: {path}")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    name = headers.get("NAMES", "") or _clean_filename_stem(path)
    formula = _strip_rruff_markup(headers.get("IDEAL CHEMISTRY", ""))
    laser = _parse_float(headers.get("RAMAN WAVELENGTH", "")) or 0.0
    rid = headers.get("RRUFFID", "")

    meta = ReferenceMeta(
        name=name,
        formula=formula,
        source="RRUFF",
        source_id=rid,
        laser_nm=laser,
        n_points=len(x),
        cm_min=float(x.min()),
        cm_max=float(x.max()),
        path=path,
    )
    return x, y, meta


def parse_csv(path):
    """Return (cm, intensity, ReferenceMeta) from a CSV/TSV.

    Two-column files are treated as ``cm⁻¹, intensity``. For files with
    three or more numeric columns (e.g. Pulsar DIY exports of the form
    ``wavenumber, raw spectrum, baseline-corrected spectrum``), the LAST
    column is used as the intensity — downstream columns are assumed to
    have additional processing applied.
    """
    xs, ys = [], []
    intensity_col = None
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in re.split(r"[,\s\t]+", line.strip()) if p.strip()]
            if len(parts) < 2:
                continue
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue
            if intensity_col is None:
                intensity_col = len(nums) - 1 if len(nums) >= 3 else 1
            if len(nums) <= intensity_col:
                continue
            xs.append(nums[0])
            ys.append(nums[intensity_col])
    if not xs:
        raise ValueError(f"No numeric data in CSV: {path}")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    meta = ReferenceMeta(
        name=_clean_filename_stem(path),
        source="CSV",
        n_points=len(x),
        cm_min=float(x.min()),
        cm_max=float(x.max()),
        path=path,
    )
    return x, y, meta


def _strip_rruff_markup(s: str) -> str:
    # RRUFF uses underscores around subscripts/superscripts: Al_2_O_3_
    return s.replace("_", "")


def _clean_filename_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    # RRUFF filenames look like "Quartz__R040031__Raman__514__..."
    # take the first chunk before the double underscore
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def _parse_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ── Library ───────────────────────────────────────────────────────────────────

@dataclass
class Library:
    grid: np.ndarray
    matrix: np.ndarray  # (N, G), each row baseline-removed and L2-normalized
    entries: list = field(default_factory=list)

    def __len__(self):
        return len(self.entries)

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def score(self, query_x, query_y, do_baseline=True):
        """
        Score a raw query spectrum (any cm⁻¹ range) against the library.
        Returns (scores, query_prepared_on_grid).
        """
        q_norm = matching.prepare_query(query_x, query_y, self.grid, do_baseline=do_baseline)
        scores = matching.score_against_matrix(q_norm, self.matrix)
        return scores, q_norm

    def match(self, query_x, query_y, do_baseline=True, k=15, refine=True, refine_pool=40):
        """Score a raw query spectrum and return ``(scores, top_indices)``.

        Stage 1 is a fast whole-spectrum cosine over the entire library (this is
        what decides which references surface). With ``refine``, the top
        ``refine_pool`` candidates are then re-ranked with a shift-tolerant,
        fingerprint-weighted cosine (see :func:`matching.refine_scores`) — which
        separates near-twin compounds and tolerates small calibration offsets.
        The returned ``scores`` array carries the refined values for the re-ranked
        candidates so the displayed HQI matches the ordering.
        """
        q_norm = matching.prepare_query(query_x, query_y, self.grid, do_baseline=do_baseline)
        base = matching.score_against_matrix(q_norm, self.matrix)
        if not refine or self.is_empty():
            return base, matching.top_matches(base, k=k)
        pool = matching.top_matches(base, k=min(refine_pool, len(base)))
        refined = matching.refine_scores(query_x, query_y, self.grid, self.matrix,
                                         pool, do_baseline=do_baseline)
        scores = base.copy()
        scores[pool] = refined
        return scores, matching.top_matches(scores, k=k)

    def reference_on_grid(self, idx) -> np.ndarray:
        return self.matrix[idx]

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta_json = json.dumps([asdict(m) for m in self.entries])
        np.savez_compressed(
            path,
            grid=self.grid,
            matrix=self.matrix.astype(np.float32),
            meta=np.array(meta_json),
        )

    @classmethod
    def load(cls, path) -> "Library":
        data = np.load(path, allow_pickle=False)
        grid = data["grid"]
        matrix = data["matrix"].astype(np.float32)
        meta_list = json.loads(str(data["meta"]))
        entries = [ReferenceMeta(**m) for m in meta_list]
        return cls(grid=grid, matrix=matrix, entries=entries)

    @classmethod
    def empty(cls, grid=None) -> "Library":
        if grid is None:
            grid = matching.default_grid()
        return cls(grid=grid, matrix=np.zeros((0, len(grid)), dtype=np.float32), entries=[])

    # ── Building ───────────────────────────────────────────────────────────

    @classmethod
    def from_directory(cls, directory, grid=None, progress=None) -> "Library":
        """
        Walk ``directory`` recursively, parse every supported file, and build
        a library. ``progress`` is an optional callable(i, n, name) used for
        UI feedback.
        """
        if grid is None:
            grid = matching.default_grid()

        files = _scan_library_files(directory)
        rows = []
        entries = []
        for i, path in enumerate(files):
            try:
                x, y, meta = _parse_any(path)
            except Exception:
                continue
            y_prep = matching.prepare_query(x, y, grid, do_baseline=True)
            if not np.any(y_prep > 0):
                continue
            rows.append(y_prep.astype(np.float32))
            entries.append(meta)
            if progress is not None:
                progress(i + 1, len(files), meta.name)

        if rows:
            matrix = np.vstack(rows)
        else:
            matrix = np.zeros((0, len(grid)), dtype=np.float32)
        return cls(grid=grid, matrix=matrix, entries=entries)


# ── File discovery ────────────────────────────────────────────────────────────

_SUPPORTED_EXTS = (".txt", ".csv", ".tsv")


def _scan_library_files(directory):
    """Return a deterministic list of files to ingest, skipping RAW RRUFF
    duplicates when a Processed counterpart exists."""
    if not os.path.isdir(directory):
        return []

    candidates = []
    for root, _, names in os.walk(directory):
        for n in names:
            if n.lower().endswith(_SUPPORTED_EXTS):
                candidates.append(os.path.join(root, n))

    # Prefer "Raman_Data_Processed" over "Raman_Data_RAW" for the same RRUFFID.
    by_key = {}
    for p in sorted(candidates):
        key = _rruff_dedup_key(p)
        prefer_processed = "Raman_Data_RAW" not in p
        if key not in by_key:
            by_key[key] = (p, prefer_processed)
        else:
            cur_path, cur_pref = by_key[key]
            if prefer_processed and not cur_pref:
                by_key[key] = (p, prefer_processed)
    return [v[0] for v in by_key.values()]


def _rruff_dedup_key(path):
    """Group a Processed/RAW pair for the same RRUFF sample."""
    base = os.path.basename(path)
    m = re.match(r"^([^_]+)__([A-Z0-9-]+)__.*$", base)
    if m:
        # name + sample id + orientation chunk
        # (orientation chunk = first three "__"-separated fields after id)
        parts = base.split("__")
        return tuple(parts[:5])
    return (path,)  # unique


def _parse_any(path):
    if path.lower().endswith(".txt") and _looks_like_rruff(path):
        return parse_rruff_txt(path)
    return parse_csv(path)


def _looks_like_rruff(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(3):
                line = f.readline()
                if line.startswith("##"):
                    return True
                if not line:
                    return False
    except OSError:
        return False
    return False


# ── Convenience ───────────────────────────────────────────────────────────────

def load_or_build_default(progress=None) -> Library:
    """
    Load the cached library if present; otherwise scan the default library
    directory and build (and cache) one. Returns an empty Library if no
    references are available yet.
    """
    cache = default_cache_path()
    if os.path.exists(cache):
        try:
            return Library.load(cache)
        except Exception:
            pass

    src = default_library_dir()
    if not os.path.isdir(src):
        return Library.empty()

    lib = Library.from_directory(src, progress=progress)
    if len(lib) > 0:
        try:
            lib.save(cache)
        except Exception:
            pass
    return lib


def invalidate_cache():
    cache = default_cache_path()
    if os.path.exists(cache):
        try:
            os.remove(cache)
        except OSError:
            pass
