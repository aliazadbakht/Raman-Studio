"""
Downloaders for public Raman reference libraries.

RRUFF is the primary source — publicly downloadable as quality-graded zip
bundles (https://rruff.info/zipped_data_files/raman/). The ROD project on
solsa.crystallography.net is a secondary aggregator that re-publishes
RRUFF-derived spectra alongside Crystallography Open Database entries.
"""
from __future__ import annotations
import os
import io
import zipfile
import urllib.request


RRUFF_BASE = "https://www.rruff.net/zipped_data_files/raman/"

RRUFF_BUNDLES = {
    # name: (filename, approximate_size_str, description)
    "fair_oriented":       ("fair_oriented.zip",       "271 KB",  "Tiny starter set — handful of minerals."),
    "unrated_unoriented":  ("unrated_unoriented.zip",  "12 MB",   "Small, miscellaneous quality."),
    "poor_unoriented":     ("poor_unoriented.zip",     "33 MB",   "Lower-quality but broader coverage."),
    "unrated_oriented":    ("unrated_oriented.zip",    "39 MB",   "Oriented samples, unrated."),
    "fair_unoriented":     ("fair_unoriented.zip",     "59 MB",   "Fair-quality, broad coverage."),
    "excellent_oriented":  ("excellent_oriented.zip",  "77 MB",   "High-quality oriented samples."),
    "lr_raman":            ("LR-Raman.zip",            "227 MB",  "Low-resolution Raman."),
    "excellent_unoriented":("excellent_unoriented.zip","229 MB",  "RECOMMENDED — high-quality, broad coverage."),
}


def list_rruff_bundles():
    return list(RRUFF_BUNDLES.items())


def download_rruff_bundle(bundle_key, dest_dir, progress=None,
                          processed_only=True) -> int:
    """
    Download and extract a RRUFF zip bundle into ``dest_dir``.

    Args:
        bundle_key: key from RRUFF_BUNDLES.
        dest_dir: target directory for the extracted .txt files.
        progress: optional callable(stage, current, total, message) where
            stage is "download" or "extract".
        processed_only: if True, skip ``Raman_Data_RAW`` files and only
            extract baseline-corrected ``Raman_Data_Processed`` ones.

    Returns the number of files extracted.
    """
    if bundle_key not in RRUFF_BUNDLES:
        raise ValueError(f"Unknown RRUFF bundle: {bundle_key}")
    filename, _, _ = RRUFF_BUNDLES[bundle_key]
    url = RRUFF_BASE + filename

    os.makedirs(dest_dir, exist_ok=True)

    buf = io.BytesIO()
    req = urllib.request.Request(url, headers={"User-Agent": "RamanSpectrumAnalyzer"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        chunk = 64 * 1024
        while True:
            block = resp.read(chunk)
            if not block:
                break
            buf.write(block)
            read += len(block)
            if progress is not None:
                progress("download", read, total, filename)

    buf.seek(0)
    extracted = 0
    with zipfile.ZipFile(buf) as zf:
        members = zf.namelist()
        if processed_only:
            members = [m for m in members if "Raman_Data_RAW" not in m]
        for i, m in enumerate(members):
            if m.endswith("/"):
                continue
            out_name = os.path.basename(m)
            out_path = os.path.join(dest_dir, out_name)
            with zf.open(m) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            extracted += 1
            if progress is not None:
                progress("extract", i + 1, len(members), out_name)

    return extracted


# ── ROD (placeholder) ─────────────────────────────────────────────────────────
# ROD spectra live at https://solsa.crystallography.net/rod/ in per-entry
# pages without a bulk-download zip. A scraper is feasible but requires
# respecting robots.txt and rate limiting; left as a follow-up so the first
# release ships with the RRUFF path working end-to-end.

def download_rod_placeholder(*_, **__):
    raise NotImplementedError(
        "ROD (Raman Open Database) bulk download isn't wired up yet. "
        "For now, drop ROD CSV exports into the references/ directory and "
        "rebuild the library."
    )
