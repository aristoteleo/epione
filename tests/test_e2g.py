"""Synthetic tests for epi.tl._e2g: peak-free insertion counting (overlapping intervals), distance ranks,
spline basis, and an end-to-end e2g_links run in which a planted active element near a gene outranks a
background element at the same distance."""
import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from epione.tl._e2g import _ns_basis, _pairs, count_insertions, e2g_links


def _frags(tmp_path, rows):
    f = tmp_path / "f.tsv.gz"
    with gzip.open(f, "wt") as fh:
        for r in sorted(rows, key=lambda r: (r[0], r[1])):
            fh.write("\t".join(map(str, r)) + "\n")
    return str(f)


def test_count_insertions_overlapping_intervals(tmp_path):
    rows = [("chr1", 100, 200, "A", 1), ("chr1", 150, 400, "B", 1), ("chr1", 1000, 1100, "A", 1),
            ("chr2", 5, 50, "B", 1), ("chrM", 1, 10, "A", 1)]
    f = _frags(tmp_path, rows)
    el = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1", "chr2"], "start": [90, 140, 0, 0], "end": [210, 400, 2000, 10]})
    C, tot = count_insertions([(f, {"A": "a", "B": "b"})], ["a", "b"], el, n_jobs=1)
    C = C.toarray()
    # insertions: A: 100, 199, 1000, 1099 (chrM ignored); B: 150, 399, 5, 49
    np.testing.assert_array_equal(C[0], [2, 1, 4, 0])
    np.testing.assert_array_equal(C[1], [1, 2, 2, 1])
    np.testing.assert_array_equal(tot, [4, 4])


def test_pairs_rank():
    el = pd.DataFrame({"chrom": ["chr1"], "start": [10_000], "end": [10_500]})
    genes = pd.DataFrame({"gene_name": ["g1", "g2", "g3"], "chrom": "chr1", "tss": [30_000, 9_000, 600_000]})
    ei, gi, d, r = _pairs(el, genes, 1_000_000)
    got = dict(zip(genes.gene_name.to_numpy()[gi], r))
    assert got == {"g2": 1, "g1": 2, "g3": 3}


def test_ns_basis_linear_beyond_boundary_knots():
    x = np.linspace(0, 10, 101)
    B = _ns_basis(x, [2, 4, 6, 8])
    assert B.shape == (101, 3)
    tail = B[x >= 8]                      # natural spline: linear beyond the last knot
    np.testing.assert_allclose(np.diff(tail, n=2, axis=0), 0, atol=1e-8)


def test_e2g_links_ranks_active_element(tmp_path):
    rng = np.random.default_rng(0)
    cells = [f"c{i}" for i in range(40)]
    rows = []
    for c in cells:
        for s in rng.integers(0, 2_000_000, 400):            # background
            rows.append(("chr1", int(s), int(s) + 200, c, 1))
        for s in rng.integers(1_050_000, 1_050_400, 30):     # active element 50 kb from g1
            rows.append(("chr1", int(s), int(s) + 100, c, 1))
    f = _frags(tmp_path, rows)
    peaks = ["chr1:1049900-1050600", "chr1:1500000-1500500"]
    X = sp.csr_matrix(np.column_stack([np.full(40, 60.0), np.full(40, 1.0)]))
    ad = AnnData(X=X, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=peaks))
    ann = pd.DataFrame({"gene_name": ["g1", "g2"], "chrom": "chr1", "start": [1_000_000, 1_400_000],
                        "end": [1_010_000, 1_410_000], "strand": "+"})
    el = pd.DataFrame({"chrom": "chr1", "start": [1_050_000, 950_000], "end": [1_050_500, 950_500]})
    out = e2g_links(ad, gene_annotation=ann, model="atac", elements=el, fragments=[(f, None)], n_jobs=1, verbose=False)
    s = out.set_index(["element", "gene"])["score"]
    assert s[("chr1:1050000-1050500", "g1")] > s[("chr1:950000-950500", "g1")]
    assert ((out.score > 0) & (out.score < 1)).all()


def _toy_multiome(seed=0, n=300):
    rng = np.random.default_rng(seed)
    grp = np.repeat([0, 1, 2], n // 3)
    peaks = [f"chr1:{s}-{s + 500}" for s in range(100_000, 2_100_000, 20_000)]
    lam = rng.gamma(2.0, 1.0, size=(3, len(peaks)))
    X = rng.poisson(lam[grp]).astype(np.float32)
    obs = pd.DataFrame({"n_fragment": X.sum(1) * 3 + 500, "ct": grp.astype(str)}, index=[f"c{i}" for i in range(n)])
    ad = AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=peaks))
    ad.obsm["X_birmt"] = np.column_stack([grp + rng.normal(0, 0.1, n), rng.normal(0, 1, (n, 4))])
    genes = [f"g{i}" for i in range(8)]
    G = rng.poisson(np.exp(0.5 * lam[grp][:, ::12][:, :8])).astype(np.float32)
    rn = AnnData(X=sp.csr_matrix(G), obs=obs.copy(), var=pd.DataFrame(index=genes))
    ann = pd.DataFrame({"gene_name": genes, "chrom": "chr1", "start": np.arange(8) * 240_000 + 150_000,
                        "end": np.arange(8) * 240_000 + 160_000, "strand": "+"})
    return ad, rn, ann


def test_link_peaks_to_genes_default_schema_and_dispatch():
    import epione as epi
    ad, rn, ann = _toy_multiome()
    lk = epi.tl.link_peaks_to_genes(ad, rna=rn, gene_annotation=ann, n_metacells=50, k_neighbors=10, verbose=False)
    for c in ("peak", "gene", "chrom", "peak_start", "peak_end", "tss", "distance", "score", "correlation", "fdr"):
        assert c in lk.columns
    np.testing.assert_allclose(lk.fdr, 1 - lk.score, rtol=1e-5)
    assert ad.uns["peak_to_gene_params"]["model"] == "full" and ad.uns["peak_to_gene"] is not None
    assert (np.abs(lk.distance) <= 250_000).all()
    # ATAC-only model without RNA; correlation is NaN
    la = epi.tl.link_peaks_to_genes(ad, gene_annotation=ann, inplace=False, verbose=False)
    assert la.correlation.isna().all() and ad.uns["peak_to_gene_params"]["model"] == "full"
    # explicit ArchR port unchanged
    p2g = epi.tl.link_peaks_to_genes(ad, rna=rn, gene_annotation=ann, method="peak_to_gene", n_metacells=50,
                                     k_neighbors=10, inplace=False, verbose=False)
    ref = epi.tl.peak_to_gene(ad, rna=rn, gene_annotation=ann, use_rep="X_birmt", n_metacells=50, k_neighbors=10,
                              seed=1, inplace=False, verbose=False)
    pd.testing.assert_frame_equal(p2g.reset_index(drop=True), ref.reset_index(drop=True))


def test_link_peaks_to_genes_works_with_plot_peak2gene():
    import matplotlib
    matplotlib.use("Agg")
    import epione as epi
    ad, rn, ann = _toy_multiome(1)
    lk = epi.tl.link_peaks_to_genes(ad, rna=rn, gene_annotation=ann, n_metacells=50, k_neighbors=10, verbose=False)
    g = lk.sort_values("score").gene.iloc[-1]
    fig, _ = epi.pl.plot_peak2gene(ad, gene=g, fdr_thresh=1.0, gene_annotation=ann, show=False)
    assert fig is not None
