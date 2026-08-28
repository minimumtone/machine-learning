"""Unit tests for the MP transfer snapshot format and integrity check."""
from __future__ import annotations

import gzip
import json

import pytest

from scripts.build_mp_transfer_db import (
    _records_sha256,
    load_snapshot,
    save_snapshot,
)

ENTRIES = [{"entry_id": "mp-1", "formula": "Ni3Al", "chemsys": "Al-Ni",
            "nelements": 2, "crystal_system": "Cubic",
            "spacegroup_symbol": "Pm-3m", "energy_per_atom": -5.0,
            "energy_above_hull": 0.0, "band_gap": 0.0, "volume": 45.0,
            "lattice_a": 3.57, "lattice_b": 3.57, "lattice_c": 3.57,
            "is_stable": True}]
RATIOS = [{"entry_id": "mp-1", "element": "Al", "atomic_fraction": 0.25},
          {"entry_id": "mp-1", "element": "Ni", "atomic_fraction": 0.75}]
ELEMENTS = [{"symbol": "Al", "atomic_number": 13, "name": "Aluminum"},
            {"symbol": "Ni", "atomic_number": 28, "name": "Nickel"}]


def test_sha256_is_deterministic_and_content_sensitive():
    d1 = _records_sha256(ENTRIES, RATIOS, ELEMENTS)
    d2 = _records_sha256(ENTRIES, RATIOS, ELEMENTS)
    assert d1 == d2
    changed = [dict(ENTRIES[0], energy_above_hull=0.001)]
    assert _records_sha256(changed, RATIOS, ELEMENTS) != d1


def test_save_and_load_snapshot_roundtrip(tmp_path):
    path = tmp_path / "snap.json.gz"
    save_snapshot(ENTRIES, RATIOS, ELEMENTS, {"source": "unit test"},
                  path=path)
    snap = load_snapshot(path)
    assert snap["entries"] == ENTRIES
    assert snap["ratios"] == RATIOS
    assert snap["elements"] == ELEMENTS
    meta = snap["_meta"]
    assert meta["n_entries"] == 1
    assert meta["n_ratios"] == 2
    assert meta["n_elements"] == 2
    assert meta["records_sha256"] == _records_sha256(ENTRIES, RATIOS,
                                                     ELEMENTS)


def test_save_snapshot_canonicalizes_record_order(tmp_path):
    """Records saved in arbitrary (fetch) order must hash identically to
    the guard's canonical order (entry_id / (entry_id, element) / symbol)."""
    entries2 = ENTRIES + [dict(ENTRIES[0], entry_id="mp-0", formula="Co3Ti",
                               chemsys="Co-Ti")]
    ratios2 = RATIOS + [{"entry_id": "mp-0", "element": "Co",
                         "atomic_fraction": 0.75},
                        {"entry_id": "mp-0", "element": "Ti",
                         "atomic_fraction": 0.25}]
    path = tmp_path / "snap.json.gz"
    save_snapshot(list(reversed(entries2)), list(reversed(ratios2)),
                  list(reversed(ELEMENTS)), {"source": "unit test"},
                  path=path)
    snap = load_snapshot(path)
    canonical_entries = sorted(entries2, key=lambda e: e["entry_id"])
    canonical_ratios = sorted(ratios2,
                              key=lambda r: (r["entry_id"], r["element"]))
    assert snap["entries"] == canonical_entries
    assert snap["ratios"] == canonical_ratios
    assert snap["elements"] == ELEMENTS
    assert snap["_meta"]["records_sha256"] == _records_sha256(
        canonical_entries, canonical_ratios, ELEMENTS)


def test_load_snapshot_rejects_tampered_records(tmp_path):
    path = tmp_path / "snap.json.gz"
    save_snapshot(ENTRIES, RATIOS, ELEMENTS, {"source": "unit test"},
                  path=path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        snap = json.load(f)
    snap["entries"][0]["energy_above_hull"] = 0.5
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(snap, f)
    with pytest.raises(ValueError, match="integrity"):
        load_snapshot(path)


def test_load_snapshot_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path / "missing.json.gz")
