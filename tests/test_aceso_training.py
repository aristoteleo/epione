"""ACESO training code: the pieces that can be checked without a GPU.

The trainings themselves (stage 1 on scGPT, stage 2 on AlphaGenome) need GPUs, the upstream sources and
a multi-terabyte store, and are exercised by the runs in the analysis repository. What is tested here is
what decides the data the models see and what a user touches directly: the store contract, the exonic
masks the RNA head pools over, the named within-type axes, path resolution, the configuration objects
(no flag may silently change its default) and the paper figures.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from epione.models import aceso
from epione.models.aceso import _sites


def test_site_resolution_and_error(tmp_path, monkeypatch):
    f = tmp_path / "hg38.fa"
    f.write_text(">chr1\nACGT\n")
    monkeypatch.setenv("HG38_FA", str(f))
    _sites._cache.clear()
    assert aceso.site_path("hg38_fa") == str(f)

    _sites._cache.clear()
    monkeypatch.setenv("HG38_FA", str(tmp_path / "missing.fa"))
    monkeypatch.setenv("EPIONE_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="hg38_fa"):
        aceso.site_path("hg38_fa")
    assert aceso.site_path("hg38_fa", required=False) is None


def test_register_site(tmp_path):
    p = tmp_path / "panel.tsv"
    p.write_text("gene\n")
    aceso.register_site("test_panel", [str(tmp_path / "nope.tsv"), str(p)])
    assert aceso.site_path("test_panel") == str(p)


def _exon_npz(path):
    np.savez_compressed(
        path,
        symbols=np.array(["GENE1", "GENE2"]), chroms=np.array(["chr1", "chr1"]),
        strands=np.array(["+", "-"]),
        #        GENE1: 1000-1100 and 2000-2100      GENE2: 50000-50100
        starts=np.array([1000, 2000, 50000], np.int64),
        ends=np.array([1100, 2100, 50100], np.int64),
        offsets=np.array([0, 2, 3], np.int64))
    return str(path)


def test_exon_masks_window_is_exonic_only(tmp_path):
    em = aceso.ExonMasks(_exon_npz(tmp_path / "e.npz"), gene_list=["GENE2", "GENE1"], bin_size=32)
    cols, M = em.window("chr1", 0, 32 * 100)                      # 3,200 bp: only GENE1 overlaps
    assert cols.tolist() == [1]                                   # the caller's column order, not the file's
    assert M.shape == (1, 100)
    assert M[0, 1000 // 32:-(-1100 // 32)].all()                  # the exon's bins
    assert M[0, :1000 // 32].sum() == 0 and M.sum() < 100         # and nothing else
    cols, M = em.window("chr1", 10_000, 10_000 + 32 * 100)        # no gene here
    assert len(cols) == 0 and M.shape == (0, 100)
    cols, M = em.window("chrZ", 0, 3_200)                         # unknown contig
    assert len(cols) == 0


def test_exon_masks_drops_genes_absent_from_the_store(tmp_path):
    em = aceso.ExonMasks(_exon_npz(tmp_path / "e.npz"), gene_list=["GENE2"], bin_size=32)
    cols, M = em.window("chr1", 0, 32 * 100)
    assert len(cols) == 0                                         # GENE1 has no column to be scored against


def _tiny_store(tmp_path, n=40):
    """A store with the contract's three files: cells.tsv, rna.h5ad and splits.json."""
    ad = pytest.importorskip("anndata")
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    genes = ["MYH7", "MYL2", "NPPA", "MYL7", "MKI67", "CDKN1A", "GAPDH"]
    X = sp.csr_matrix(rng.poisson(3, size=(n, len(genes))).astype(np.int32))
    cells = pd.DataFrame({"cell_id": [f"c{i}" for i in range(n)],
                          "corpus": "test", "sample": np.repeat(["s1", "s2"], n // 2),
                          "organ": np.where(np.arange(n) < n // 2, "Heart", "Liver"),
                          "cell_type": "cardiomyocyte", "stage_or_age": "PCW10",
                          "n_umi": np.asarray(X.sum(1)).ravel(), "n_genes": np.diff(X.indptr)})
    cells.to_csv(tmp_path / "cells.tsv", sep="\t", index=False)
    ad.AnnData(X, var=pd.DataFrame(index=genes)).write_h5ad(tmp_path / "rna.h5ad")
    json.dump({"train": ["s1"], "val": ["s2"], "test": []}, open(tmp_path / "splits.json", "w"))
    np.save(tmp_path / "z_epiagent.npy", rng.normal(size=(n, 8)).astype(np.float32))
    return tmp_path


def test_build_axes_masks_cells_outside_the_axis_tissue(tmp_path):
    store = _tiny_store(tmp_path)
    aceso.build_axes(store, log=lambda *a: None)
    z = np.load(store / "axes.npz", allow_pickle=False)
    names = list(z["names"])
    assert names == ["chamber", "cycle"]
    heart = np.arange(len(z["mask"])) < 20
    assert z["mask"][heart, names.index("chamber")].all()
    assert not z["mask"][~heart, names.index("chamber")].any()    # chamber is undefined outside the heart
    assert (z["score"][~heart, names.index("chamber")] == 0).all()  # and never given a fabricated value
    assert z["mask"][:, names.index("cycle")].all()               # cell cycle is defined everywhere
    assert abs(z["score"][heart, names.index("chamber")].std() - 1) < 0.2   # standardised in its tissue


def test_training_store_reads_the_contract(tmp_path):
    store = aceso.TrainingStore(_tiny_store(tmp_path))
    assert len(store.cells) == 40
    assert store.genes[:2] == ["MYH7", "MYL2"]
    assert store.splits["val"] == ["s2"]
    assert store.z.shape == (40, 8)
    assert store.fragments == {}
    s = store.summary()
    assert int(s["cells"].sum()) == 40


def test_training_store_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="cells.tsv"):
        aceso.TrainingStore(tmp_path).cells


@pytest.mark.parametrize("name,flag,field,default", [
    ("V1EncoderConfig", "--panel-mode", "panel_mode", "mixed"),
    ("V1EncoderConfig", "--tau", "tau", 0.07),
    ("V1DecoderConfig", "--within-pool", "within_pool", None),
    ("V2EncoderConfig", "--lr", "lr", None),
    ("V2DecoderConfig", "--bin", "bin", 32),
    ("V2DecoderConfig", "--window", "window", 524288),
])
def test_config_defaults_match_the_released_runs(name, flag, field, default):
    cfg = getattr(aceso, name)()
    assert hasattr(cfg, field), f"{name} lost {flag}"
    if default is not None:
        assert getattr(cfg, field) == default


def test_v2_decoder_default_heads_are_atac_and_rna():
    assert aceso.V2DecoderConfig().heads == "atac,rna"


def test_plot_tracks_draws_every_row_and_names_the_window(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    aceso.paper_style()
    rng = np.random.default_rng(0)
    tracks = {"measured": rng.poisson(1, 200).astype(float), "predicted": rng.random(200)}
    fig, axes = aceso.plot_tracks(tracks, chrom="chr14", start=23_000_000, show=False)
    assert len(axes) >= len(tracks)
    texts = [t.get_text() for ax in axes for t in ax.texts] + [t.get_text() for t in fig.texts]
    assert "measured" in texts and "predicted" in texts
    assert any("chr14" in t for t in texts)
    out = tmp_path / "t"
    aceso.save_figure(fig, str(out), formats=("svg",))
    svg = (tmp_path / "t.svg").read_text()
    assert 'font-family="Arial"' in svg or "font-family: Arial" in svg


def test_read_training_log_rejects_a_file_that_is_not_one(tmp_path):
    (tmp_path / "train.log").write_text("starting run\n")
    with pytest.raises(ValueError, match="no JSON records"):
        aceso.read_training_log(tmp_path)
    (tmp_path / "train.log").write_text(json.dumps({"step": 20, "loss": 1.0}) + "\n")
    df = aceso.read_training_log(tmp_path)
    assert df["step"].tolist() == [20]


@pytest.mark.skipif(not os.environ.get("ACESO_STORE"), reason="needs a built store in $ACESO_STORE")
def test_real_store_rows_line_up():
    store = aceso.TrainingStore(os.environ["ACESO_STORE"])
    assert store.z.shape[0] == len(store.cells)
