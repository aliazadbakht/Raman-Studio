# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V. (https://wfront.nl)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
"""Resolve free-text input (SMILES, compound name, or formula) to a SMILES.

The Mol2Raman model takes a SMILES — a full molecular structure. A bare
chemical *formula* (e.g. ``C8H9NO2``) does not define one molecule (many
isomers share it), so the user request "give the formula" is handled as:

  * looks like SMILES  → validate (RDKit if available) and use as-is;
  * looks like a name  → PubChem name → canonical SMILES;
  * looks like formula → PubChem "fastformula" → first/most-common match,
    flagged as ambiguous so the UI can warn.

Network lookups use the standard library (``urllib``) with a short timeout, so
this module adds no dependency and degrades gracefully offline.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_TIMEOUT = 8
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


class ResolveError(RuntimeError):
    pass


@dataclass
class ResolveResult:
    smiles: str
    name: str = ""
    source: str = ""          # "smiles" | "pubchem-name" | "pubchem-formula"
    ambiguous: bool = False
    note: str = ""


_SMILES_CHARS_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#/\\%.]+$")


def _valid_smiles(text: str) -> bool:
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        # RDKit not installed — permissive structural heuristic (good enough for
        # the offline demo path; the bundled env has RDKit for the real check).
        if " " in text or len(text) < 2:
            return False
        return bool(_SMILES_CHARS_RE.match(text))


def _looks_like_formula(text: str) -> bool:
    # Formula = element-symbol/count runs with at least one explicit count digit.
    # Requiring a digit avoids catching all-caps SMILES like "CCO"/"CCC".
    return bool(_FORMULA_RE.match(text)) and bool(re.search(r"\d", text))


def _http_get(url: str) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RamanAnalyzer/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return None


def _pubchem_smiles_for_cid(cid: str) -> Optional[str]:
    url = f"{_PUBCHEM}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    body = _http_get(url)
    if not body:
        return None
    try:
        props = json.loads(body)["PropertyTable"]["Properties"]
        return props[0].get("CanonicalSMILES") if props else None
    except (KeyError, IndexError, ValueError):
        return None


def _resolve_name(name: str) -> Optional[ResolveResult]:
    url = (f"{_PUBCHEM}/compound/name/{urllib.parse.quote(name)}"
           f"/property/CanonicalSMILES,Title/JSON")
    body = _http_get(url)
    if not body:
        return None
    try:
        props = json.loads(body)["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, ValueError):
        return None
    smi = props.get("CanonicalSMILES")
    if not smi:
        return None
    return ResolveResult(smiles=smi, name=props.get("Title", name), source="pubchem-name")


def _resolve_formula(formula: str) -> Optional[ResolveResult]:
    url = f"{_PUBCHEM}/compound/fastformula/{urllib.parse.quote(formula)}/cids/JSON"
    body = _http_get(url)
    if not body:
        return None
    try:
        cids = json.loads(body)["IdentifierList"]["CID"]
    except (KeyError, IndexError, ValueError):
        return None
    if not cids:
        return None
    smi = _pubchem_smiles_for_cid(str(cids[0]))
    if not smi:
        return None
    return ResolveResult(
        smiles=smi, name=formula, source="pubchem-formula", ambiguous=True,
        note=f"{len(cids)} compounds share formula {formula}; showing the most "
             f"common match (CID {cids[0]}). Enter a name or SMILES to be exact.")


def resolve_to_smiles(text: str) -> ResolveResult:
    """Best-effort resolution. Raises :class:`ResolveError` if nothing matches."""
    text = (text or "").strip()
    if not text:
        raise ResolveError("Enter a SMILES, compound name, or formula.")

    # A formula check must precede the SMILES check: "CO" is valid as both, but
    # an all-caps element string is far more likely meant as a formula.
    if _looks_like_formula(text):
        res = _resolve_formula(text)
        if res is not None:
            return res
        # Fall through — maybe it was actually a SMILES like "CCO".

    if _valid_smiles(text):
        return ResolveResult(smiles=text, name=text, source="smiles")

    res = _resolve_name(text)
    if res is not None:
        return res

    res = _resolve_formula(text)
    if res is not None:
        return res

    raise ResolveError(
        f"Could not resolve {text!r} to a structure. Check spelling, or paste a "
        f"SMILES string. (Name/formula lookup needs an internet connection.)")
