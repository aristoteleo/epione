"""Network-free tests for ``epi.datasets.archr_hematopoiesis``."""
from __future__ import annotations

import re

import pooch
import pytest

import epione as epi
from epione.datasets import _archr


def test_registry_pins_three_samples():
    assert list(_archr.ARCHR_HEME_FRAGMENTS) == [
        "scATAC_BMMC_R1", "scATAC_CD34_BMMC_R1", "scATAC_PBMC_R1",
    ]
    for h in _archr.ARCHR_HEME_FRAGMENTS.values():
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", h)
    assert _archr.ARCHR_HEME_BASE_URL.startswith("https://")


def test_fetch_uses_pinned_urls_and_hashes(tmp_path, monkeypatch):
    calls = []

    def fake_retrieve(url, known_hash, fname, path, progressbar):
        calls.append((url, known_hash, fname, path))
        out = tmp_path / "HemeFragments" / fname
        out.write_bytes(b"")
        (tmp_path / "HemeFragments" / (fname + ".tbi")).write_bytes(b"")
        return str(out)

    monkeypatch.setattr(pooch, "retrieve", fake_retrieve)
    frags = epi.datasets.archr_hematopoiesis(tmp_path)

    assert list(frags) == list(_archr.ARCHR_HEME_FRAGMENTS)
    for (url, known_hash, fname, path), (sample, h) in zip(
        calls, _archr.ARCHR_HEME_FRAGMENTS.items()
    ):
        assert url == _archr.ARCHR_HEME_BASE_URL + f"{sample}.fragments.tsv.gz"
        assert known_hash == h
        assert path == str(tmp_path / "HemeFragments")
        assert frags[sample].name == fname


def test_fetch_builds_missing_index(tmp_path, monkeypatch):
    pysam = pytest.importorskip("pysam")
    indexed = []

    def fake_retrieve(url, known_hash, fname, path, progressbar):
        out = tmp_path / "HemeFragments" / fname
        out.write_bytes(b"")
        return str(out)

    monkeypatch.setattr(pooch, "retrieve", fake_retrieve)
    monkeypatch.setattr(pysam, "tabix_index", lambda p, preset, force: indexed.append((p, preset)))
    epi.datasets.archr_hematopoiesis(tmp_path)
    assert len(indexed) == 3 and all(preset == "bed" for _, preset in indexed)
