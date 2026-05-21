"""Tests for epione.utils.annotate_peaks and peak_feature_enrichment."""
import numpy as np
import pandas as pd

import epione as epi
from epione.core._sampling import annotate_peaks, peak_feature_enrichment


_GTF = """\
chr1\tTEST\tgene\t1000\t9000\t.\t+\t.\tgene_id "G1"; gene_type "protein_coding"; gene_name "G1";
chr1\tTEST\texon\t1000\t1200\t.\t+\t.\tgene_id "G1"; gene_type "protein_coding"; gene_name "G1";
chr1\tTEST\texon\t8000\t9000\t.\t+\t.\tgene_id "G1"; gene_type "protein_coding"; gene_name "G1";
chr2\tTEST\tgene\t5000\t6000\t.\t-\t.\tgene_id "G2"; gene_type "protein_coding"; gene_name "G2";
chr2\tTEST\texon\t5000\t6000\t.\t-\t.\tgene_id "G2"; gene_type "protein_coding"; gene_name "G2";
"""


def _write_gtf(tmp_path):
    p = tmp_path / "mini.gtf"
    p.write_text(_GTF)
    return str(p)


def test_annotate_peaks_assigns_four_classes(tmp_path):
    gtf = _write_gtf(tmp_path)
    # G1: chr1 1000-9000 (+), TSS=1000, promoter=[0,3000]; exons 1000-1200, 8000-9000.
    peaks = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr1"],
        "start": [500, 8500, 5000, 20000],
        "end":   [700, 8600, 5100, 20100],
    })
    # 500-700 in promoter window; 8500 in exon; 5000 in gene body, no exon,
    # >2kb from TSS -> intron; 20000 outside any gene -> intergenic.
    ann = annotate_peaks(peaks, gtf)
    assert list(ann["feature"]) == ["promoter", "exon", "intron", "intergenic"]
    # original columns preserved, one column added
    assert set(peaks.columns).issubset(ann.columns)
    assert "feature" in ann.columns


def test_annotate_peaks_strand_aware_promoter(tmp_path):
    gtf = _write_gtf(tmp_path)
    # G2 on chr2 is '-' strand, gene 5000-6000 -> TSS at the END (6000).
    # promoter = TSS +/- 2kb = [4000, 8000].
    peaks = pd.DataFrame({"chrom": ["chr2", "chr2"],
                          "start": [7000, 9000], "end": [7100, 9100]})
    ann = annotate_peaks(peaks, gtf)
    # 7000 is within [4000,8000] of the minus-strand TSS -> promoter;
    # 9000 is past the promoter and outside the gene -> intergenic.
    assert list(ann["feature"]) == ["promoter", "intergenic"]


def test_annotate_peaks_precedence_is_configurable(tmp_path):
    gtf = _write_gtf(tmp_path)
    # A peak in exon 8000-9000 is also inside the gene body (intron-eligible).
    peaks = pd.DataFrame({"chrom": ["chr1"], "start": [8400], "end": [8600]})
    assert annotate_peaks(peaks, gtf)["feature"].iloc[0] == "exon"
    demoted = annotate_peaks(
        peaks, gtf, precedence=("promoter", "intron", "exon", "intergenic"))
    assert demoted["feature"].iloc[0] == "intron"


def test_annotate_peaks_empty(tmp_path):
    gtf = _write_gtf(tmp_path)
    empty = pd.DataFrame({"chrom": [], "start": [], "end": []})
    ann = annotate_peaks(empty, gtf)
    assert len(ann) == 0 and "feature" in ann.columns


def test_annotate_peaks_rejects_unknown_precedence(tmp_path):
    gtf = _write_gtf(tmp_path)
    peaks = pd.DataFrame({"chrom": ["chr1"], "start": [500], "end": [700]})
    try:
        annotate_peaks(peaks, gtf, precedence=("promoter", "enhancer"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown precedence class")


def test_peak_feature_enrichment_detects_promoter_signal(tmp_path):
    gtf = _write_gtf(tmp_path)
    rng = np.random.default_rng(0)
    # Foreground: mostly promoter-proximal (40 promoter + 10 intergenic).
    fg_start = np.concatenate([rng.integers(400, 800, 40),
                               rng.integers(20000, 30000, 10)])
    fg = pd.DataFrame({"chrom": ["chr1"] * 50, "start": fg_start})
    fg["end"] = fg["start"] + 150
    # Background: mostly intergenic (8 promoter + 42 intergenic) -> bg_frac > 0
    # so the enrichment ratio is finite.
    bg_start = np.concatenate([rng.integers(400, 800, 8),
                               rng.integers(20000, 30000, 42)])
    bg = pd.DataFrame({"chrom": ["chr1"] * 50, "start": bg_start})
    bg["end"] = bg["start"] + 150
    enr = peak_feature_enrichment(fg, bg, gtf)
    assert set(enr["feature"]) == {"promoter", "exon", "intron", "intergenic"}
    prom = enr.set_index("feature").loc["promoter"]
    assert prom["enrichment"] > 1.0           # finite ratio, promoter-enriched
    assert prom["pvalue"] < 0.05
    # FDR column present and in [0, 1]
    assert ((enr["fdr"] >= 0) & (enr["fdr"] <= 1)).all()


def test_functions_exposed_on_utils():
    assert hasattr(epi.utils, "annotate_peaks")
    assert hasattr(epi.utils, "peak_feature_enrichment")
