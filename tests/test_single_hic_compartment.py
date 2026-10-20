"""Unit tests for epione.single.hic compartments (synthetic data, no network).

Parity with scHiCluster / HCEA is in tests/parity/schicluster_compartment.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

cooler = pytest.importorskip("cooler")

import epione as epi  # noqa: E402

epi.pl  # noqa: B018  (import order: utils before single)
from epione.single import hic  # noqa: E402

RES = 100_000
SIZES = pd.Series({"chrA": 4_050_000, "chrB": 3_020_000})


def _write_fasta(path, rng):
    # A bins (even 400 kb blocks) are CpG rich, B bins CpG poor.
    with open(path, "w") as f:
        for chrom, size in SIZES.items():
            seq = []
            for start in range(0, size, RES):
                a = (start // 400_000) % 2 == 0
                p = [0.15, 0.35, 0.35, 0.15] if a else [0.35, 0.15, 0.15, 0.35]
                s = "".join(rng.choice(list("ACGT"), size=RES, p=p))
                if start == 0:
                    s = "N" * 500 + s[500:]
                seq.append(s)
            seq = "".join(seq)[:size]
            f.write(f">{chrom}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")
    import pyfaidx
    pyfaidx.Faidx(str(path))


def _compartment_of(chrom, pos):
    return (pos // 400_000) % 2


def _write_cell(path, rng, n=6000):
    rows = []
    while len(rows) < n:
        chrom = rng.choice(SIZES.index)
        p1 = int(rng.integers(1, SIZES[chrom]))
        # contacts prefer the same compartment and short distances
        d = int(rng.exponential(600_000)) + 5_000
        p2 = p1 + d if rng.random() < 0.5 else p1 - d
        if not (0 < p2 < SIZES[chrom]):
            continue
        if _compartment_of(chrom, p1) != _compartment_of(chrom, p2) and rng.random() < 0.7:
            continue
        rows.append(f"0\t{chrom}\t{p1}\t0\t0\t{chrom}\t{p2}\t1\n")
    with gzip.open(path, "wt") as f:
        f.writelines(rows)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("comp")
    fasta = d / "toy.fa"
    _write_fasta(fasta, rng)
    cells = []
    for i in range(6):
        p = d / f"cell{i}.3C.contact.tsv.gz"
        _write_cell(p, rng)
        cells.append(p)
    sizes_path = d / "sizes.tsv"
    SIZES.to_csv(sizes_path, sep="\t", header=False)
    return {"dir": d, "fasta": fasta, "cells": cells, "sizes": sizes_path}


def test_cpg_ratio_matches_definition(toy):
    cpg = hic.cpg_ratio(toy["fasta"], chrom_sizes=toy["sizes"], resolution=RES)
    assert list(cpg.columns) == ["chrom", "start", "end", "cpg_ratio"]
    assert len(cpg) == 41 + 31
    import pyfaidx
    seq = str(pyfaidx.Fasta(str(toy["fasta"]))["chrA"][0:RES]).upper()
    expect = seq.count("CG") / (RES - seq.count("N"))
    assert cpg.loc[0, "cpg_ratio"] == expect
    a = cpg[(cpg["start"] // 400_000) % 2 == 0]["cpg_ratio"].mean()
    b = cpg[(cpg["start"] // 400_000) % 2 == 1]["cpg_ratio"].mean()
    assert a > 2 * b


def test_scool_scores_and_tsv_mode_agree(toy):
    cpg = hic.cpg_ratio(toy["fasta"], chrom_sizes=toy["sizes"], resolution=RES)
    scool = toy["dir"] / "toy.scool"
    hic.contacts_to_scool(toy["cells"], scool, chrom_sizes=toy["sizes"], resolution=RES)
    ad_cool = hic.load_scool_cells(scool)
    ad_tsv = hic.load_contact_tables(toy["cells"], chrom_sizes=toy["sizes"], resolution=RES)
    s_cool = hic.compartment_scores(ad_cool, cpg, dtype=np.float64, progress=False,
                                    calc_strength=True)
    s_tsv = hic.compartment_scores(ad_tsv, cpg, dtype=np.float64, progress=False)
    assert s_cool.shape == (6, 72)
    # all toy contacts are > 2.5 kb apart and unique, so both inputs give the same matrix
    order = [s_tsv.obs_names.get_loc(c) for c in s_cool.obs_names]
    np.testing.assert_allclose(s_cool.X, s_tsv.X[order], rtol=0, atol=1e-12)
    is_a = ((s_cool.var["start"] // 400_000) % 2 == 0).to_numpy()
    assert s_cool.X[:, is_a].mean() > s_cool.X[:, ~is_a].mean()
    st = s_cool.obsm["compartment_strength"]
    assert list(st.columns) == ["AA", "BB", "AB"]
    per = s_cool.uns["compartment_strength_per_chrom"]["values"]
    np.testing.assert_allclose(st.to_numpy(), per.sum(axis=1))


def test_merge_cells_sums_pixels(toy):
    ad = hic.load_contact_tables(toy["cells"], chrom_sizes=toy["sizes"], resolution=RES,
                                 obs=pd.DataFrame({"g": ["x", "x", "x", "y", "y", "y"]}))
    out = hic.merge_cells(ad, "g", toy["dir"] / "pb", overwrite=True)
    scool = toy["dir"] / "toy.scool"
    if not scool.exists():
        hic.contacts_to_scool(toy["cells"], scool, chrom_sizes=toy["sizes"], resolution=RES)
    total = None
    for c in ad.obs_names[:3]:
        m = cooler.Cooler(f"{scool}::/cells/{c}").matrix(balance=False, sparse=True)[:]
        total = m if total is None else total + m
    got = cooler.Cooler(out["x"]).matrix(balance=False, sparse=True)[:]
    assert (got != total).nnz == 0
    out2 = hic.merge_cells(ad, "g", toy["dir"] / "pb2", double_diagonal=True, overwrite=True)
    got2 = cooler.Cooler(out2["x"]).matrix(balance=False, sparse=True)[:]
    np.testing.assert_array_equal(got2.diagonal(), 2 * total.diagonal())


def test_model_projection_saddle(toy):
    cpg = hic.cpg_ratio(toy["fasta"], chrom_sizes=toy["sizes"], resolution=RES)
    ad = hic.load_contact_tables(toy["cells"], chrom_sizes=toy["sizes"], resolution=RES,
                                 obs=pd.DataFrame({"g": ["all"] * 6}))
    pb = hic.merge_cells(ad, "g", toy["dir"] / "pb_all", overwrite=True)["all"]
    model = hic.compartment_model(pb, cpg, n_quantiles=4, ws=1)
    assert {"chrom", "start", "end", "keep", "loading", "pc"} <= set(model.columns)
    proj = hic.project_compartments({"all": pb}, model)
    assert len(proj) == int(model["keep"].sum())
    is_a = ((proj["start"] // 400_000) % 2 == 0).to_numpy()
    # sign fixed by CpG: A bins positive on average
    assert proj.loc[is_a, "all"].mean() > 0 > proj.loc[~is_a, "all"].mean()
    sad = hic.compartment_saddle(pb, model, n_quantiles=4, ws=1)
    assert sad["saddle"].shape == (4, 4)
    assert sad["strength"]["strength"] > 1
    assert set(sad["strength"].index) == {"BB", "AA", "BA", "AB", "strength"}
