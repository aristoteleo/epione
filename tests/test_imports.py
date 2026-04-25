"""Smoke test: every public submodule imports cleanly and exposes its
advertised public symbols. Catches dependency-regression bugs where a
library dep got bumped and broke a transitive import.
"""
from __future__ import annotations

import importlib

import pytest


SUBMODULES = [
    "epione",
    "epione.bulk",
    "epione.bulk.atac",
    "epione.bulk.hic",
    "epione.single",
    "epione.single.atac",
    "epione.single.hic",
    "epione.upstream",
    "epione.io",
    "epione.core",
    "epione.data",
    "epione.datasets",
    "epione.tl",
    "epione.pl",
    "epione.pp",
    "epione.utils",
]


@pytest.mark.parametrize("mod", SUBMODULES)
def test_submodule_imports(mod):
    importlib.import_module(mod)


def test_key_public_symbols_resolve():
    """Every symbol we document as ``epi.<module>.<name>`` in tutorials
    and docstrings must resolve after ``import epione``."""
    import epione as epi

    expected = {
        # bulk
        "bulk.bigwig",
        "bulk.plot_matrix",
        "bulk.footprint_archr",
        "bulk.find_motifs_genome",
        "bulk.run_homer_motifs",
        "bulk.gene_expression_from_bigwigs",
        # upstream (renamed from align in v0.4)
        "upstream.check_tools",
        "upstream.tool_path",
        "upstream.build_env",
        "upstream.resolve_executable",
        "upstream.ATAC_TOOLS",
        "upstream.RNA_TOOLS",
        "upstream.MOTIF_TOOLS",
        "upstream.pairs_from_bam",
        "upstream.pairs_to_cool",
        "upstream.HIC_TOOLS",
        # bulk.hic + single.hic (Phase 1 Hi-C migration in PR 1)
        "bulk.hic.balance_cool",
        "bulk.hic.plot_contact_matrix",
        "single.hic.impute_cells",
        "single.hic.embedding",
        # utils
        "utils.get_gene_annotation",
        "utils.filter_distal_peaks",
        "utils.classify_peaks_by_overlap",
        "utils.distance_to_nearest_peak",
        "utils.convert_gff_to_gtf",
        "utils.merge_peaks",
        # tl
        "tl.differential_peaks",
        "tl.iterative_lsi",
        "tl.find_marker_features",
        "tl.compute_deviations",
        "tl.peak_to_gene",
        "tl.coaccessibility",
        # pl
        "pl.volcano",
        "pl.ma_plot",
        "pl.cumulative_distance",
        "pl.homer_motif_table",
    }
    missing = []
    for path in expected:
        attr, *rest = path.split(".")
        obj = getattr(epi, attr, None)
        for piece in rest:
            obj = getattr(obj, piece, None) if obj is not None else None
        if obj is None:
            missing.append(path)
    assert not missing, f"missing public symbols: {missing}"


def test_no_omicverse_imports_in_epione():
    """``epione`` must not depend on ``omicverse`` — per project
    convention. Regression guard: grep the loaded module files."""
    import epione
    from pathlib import Path

    root = Path(epione.__file__).parent
    offenders = []
    for f in root.rglob("*.py"):
        text = f.read_text(errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "import omicverse" in stripped or "from omicverse" in stripped:
                offenders.append(f"{f.relative_to(root.parent)}: {stripped}")
    assert not offenders, "omicverse imports leaked in:\n  " + "\n  ".join(offenders)
