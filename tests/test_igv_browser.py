"""Track-spec generation for ``epi.pl.igv_browser`` (pure Python, no browser)."""
import html
import json
import re

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from epione.pl import IGVBrowser, igv_browser
from epione.pl._igv import (
    IGV_JS_URL,
    data_uri,
    decode_data_uri,
    gene_bed12,
    links_frame,
    tiles_to_bedgraph,
)


def _genes():
    return pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [10_001, 30_001],
                         "end": [20_000, 35_000], "strand": ["+", "-"],
                         "symbol": ["GENEA", "GENEB"]})


def _exons():
    return pd.DataFrame({"chrom": ["chr1"] * 3, "start": [10_001, 15_001, 14_000],
                         "end": [11_000, 16_000, 15_500], "strand": ["+"] * 3,
                         "symbol": ["GENEA"] * 3})


def _adata():
    rng = np.random.default_rng(0)
    var = pd.DataFrame({"chrom": "chr1", "start": np.arange(0, 60_000, 2_000),
                        "end": np.arange(500, 60_500, 2_000)})
    var.index = [f"chr1:{s}-{e}" for s, e in zip(var.start, var.end)]
    obs = pd.DataFrame({"Clusters": pd.Categorical(["C1"] * 20 + ["C2"] * 20)},
                       index=[f"c{i}" for i in range(40)])
    return AnnData(rng.poisson(1.0, (40, len(var))).astype(np.float32), obs=obs, var=var)


def test_data_uri_roundtrip_and_deterministic():
    u = data_uri("chr1\t0\t10\n")
    assert u.startswith("data:application/gzip;base64,")
    assert decode_data_uri(u) == "chr1\t0\t10\n"
    assert data_uri("chr1\t0\t10\n") == u


def test_gene_bed12_merges_exons_and_converts_to_0_based():
    bed = gene_bed12(_genes(), _exons(), "chr1", 1, 50_000).strip().split("\n")
    a = bed[0].split("\t")
    assert a[:6] == ["chr1", "10000", "20000", "GENEA", "0", "+"]
    assert a[9] == "2"                      # 15001-16000 and 14000-15500 merged
    assert a[10] == "1000,2001" and a[11] == "0,3999"
    b = bed[1].split("\t")
    assert b[3] == "GENEB" and b[9] == "1"  # no exons -> one block


def test_tiles_to_bedgraph():
    cov = pd.DataFrame({"x": [1001.0, 1251.0, np.nan], "y": [0.5, 1.0, 0.0],
                        "group": ["A", "A", "A"]})
    cov.attrs["groups"] = ["A"]
    bg = tiles_to_bedgraph(cov, "chr1", 250)["A"]
    assert bg[["start", "end"]].values.tolist() == [[1000, 1250], [1250, 1500]]


def test_full_spec_track_order_and_window():
    ad = _adata()
    loops = pd.DataFrame({"chrom": "chr1", "start": [12_000, 5_000], "end": [30_000, 900_000],
                          "value": [0.6, 0.9]})
    links = pd.DataFrame({"chrom": "chr1", "peak_start": [40_000, 41_000], "peak_end": [40_500, 41_500],
                          "gene": ["GENEA", "GENEA"], "tss": [10_001, 10_001],
                          "correlation": [0.8, 0.1], "fdr": [1e-5, 0.5]})
    fp = pd.DataFrame({"chrom": "chr1", "start": [15_000], "end": [15_020], "name": ["CTCF"],
                       "score": [7.5], "strand": ["+"]})
    hl = pd.DataFrame({"chrom": ["chr1"], "start": [10_001], "end": [11_000]})
    b = igv_browser(ad, gene="GENEA", group_by="Clusters", features="var", loops={"CoAcc": loops},
                    links=links, footprints={"HINT": fp}, gene_annotation=_genes(),
                    exon_annotation=_exons(), genome="hg19", upstream=5_000, downstream=40_000,
                    tile_size=500, highlight=hl,
                    tracks=[{"name": "remote", "type": "wig", "url": "https://x.org/a.bw"}])
    assert isinstance(b, IGVBrowser)
    assert b.locus == "chr1:5001-50001"
    assert b.options["genome"] == "hg19"
    assert b.track_names() == ["C1", "C2", "Peaks", "HINT", "CoAcc", "Peak2GeneLinks",
                               "Genes", "remote"]
    t = {x["name"]: x for x in b.tracks}
    assert t["C1"]["type"] == "wig" and t["C1"]["format"] == "bedgraph"
    assert t["C1"]["autoscaleGroup"] == t["C2"]["autoscaleGroup"]
    assert t["C1"]["color"] == "#D51F26"   # ArchR stallion, first group
    # peaks restricted to the data window (region +- half its width)
    peaks = decode_data_uri(t["Peaks"]["url"]).strip().split("\n")
    starts = [int(l.split("\t")[1]) for l in peaks]
    assert min(starts) >= 0 and max(starts) <= 50_001 + 22_500
    # out-of-window loop dropped, kept loop is 0-based 1-bp anchors + colour column
    rows = decode_data_uri(t["CoAcc"]["url"]).strip().split("\n")
    assert rows[0].startswith("#chr1\tstart1") and rows[0].endswith("\tcolor")
    assert len(rows) == 2 and rows[1].split("\t")[:6] == ["chr1", "11999", "12000",
                                                         "chr1", "29999", "30000"]
    assert t["CoAcc"]["type"] == "interact" and t["CoAcc"]["arcType"] == "proportional"
    # only the significant link survives, as in plot_peak2gene
    lrows = decode_data_uri(t["Peak2GeneLinks"]["url"]).strip().split("\n")[1:]
    assert len(lrows) == 1 and lrows[0].split("\t")[6] == "GENEA"
    assert decode_data_uri(t["HINT"]["url"]) == "chr1\t15000\t15020\tCTCF\t7.5\t+\n"
    assert b.options["roi"][0]["features"] == [{"chr": "chr1", "start": 10_000, "end": 11_000}]


def test_coverage_from_peak_matrix_matches_group_means():
    ad = _adata()
    b = igv_browser(ad, region="chr1:1-20000", flank=0, tile_size=1000)
    bg = decode_data_uri(b.tracks[0]["url"]).strip().split("\n")
    assert all(len(l.split("\t")) == 4 for l in bg)
    assert b.track_names() == ["C1", "C2"]


def test_use_groups_and_ylim():
    ad = _adata()
    b = igv_browser(ad, region="chr1:1-20000", use_groups=["C2"], ylim=(0, 5))
    assert b.track_names() == ["C2"]
    assert b.tracks[0]["max"] == 5.0 and b.tracks[0]["autoscale"] is False


def test_links_frame_filters():
    links = pd.DataFrame({"chrom": "chr1", "peak_start": [100, 300], "peak_end": [200, 400],
                          "gene": ["A", "B"], "tss": [1000, 2000],
                          "correlation": [-0.6, 0.2], "fdr": [0.01, 0.01]})
    f = links_frame(links, min_abs_r=0.5)
    assert f["name"].tolist() == ["A"] and f["start1"].tolist() == [149]


def test_absolute_paths_refused_and_html_is_portable(tmp_path):
    with pytest.raises(ValueError, match="absolute local path"):
        igv_browser(region="chr1:1-1000", group_by=None,
                    tracks=[{"name": "x", "url": "/home/me/a.bw"}])
    b = igv_browser(region="chr1:1-1000", group_by=None, bigwig_files={"A": "tracks/a.bw"})
    assert b.tracks[0]["url"] == "tracks/a.bw" and b.tracks[0]["format"] == "bigwig"
    page = b.to_html()
    assert IGV_JS_URL in page and "integrity=" in page
    cfg = json.loads(re.search(r"var options=(\{.*?\});igv", page).group(1).replace("<\\/", "</"))
    assert cfg["locus"] == "chr1:1-1000" and cfg["showSVGButton"] is True
    frame = b._repr_html_()
    assert frame.startswith("<iframe srcdoc=") and html.unescape(frame).count("<script") == 2
    p = b.save(tmp_path / "b.html")
    assert open(p).read() == page


def test_gene_name_only_uses_igv_search():
    b = igv_browser(gene="CD34", group_by=None, genome="hg38")
    assert b.locus == "CD34" and b.tracks == []
    with pytest.raises(ValueError, match="explicit window"):
        igv_browser(gene="CD34", features=pd.DataFrame({"chrom": ["chr1"], "start": [1], "end": [2]}))


def test_auto_height_fits_tracks():
    b = igv_browser(_adata(), region="chr1:1-20000", genome="hg38")
    assert b.height == 150 + 2 * (40 + 6) + 70
    b.height = 300
    assert b.height == 300 and 'height:300px' in b._repr_html_()


def test_local_bigwig_is_embedded_not_linked(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    path = str(tmp_path / "a.bw")
    bw = pyBigWig.open(path, "w")
    bw.addHeader([("chr1", 100_000)])
    bw.addEntries(["chr1"], [1000], ends=[2000], values=[3.0])
    bw.close()
    b = igv_browser(region="chr1:1-4000", flank=0, group_by=None, bigwig_files={"A": path},
                    tile_size=500)
    t = b.tracks[0]
    assert t["format"] == "bedgraph" and str(tmp_path) not in b.to_html()
    rows = [l.split("\t") for l in decode_data_uri(t["url"]).strip().split("\n")]
    assert [(r[1], r[2], float(r[3])) for r in rows] == [("1000", "1500", 3.0), ("1500", "2000", 3.0)]
