"""epione.grn.pando: plumbing, normalisations and the exactness of near-target scanning."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import epione as epi
from epione.grn import seurat_lognormalize, signac_tfidf


def test_normalisations_match_signac_and_seurat_formulas():
    counts = sp.csr_matrix(np.array([[0, 2, 1], [3, 0, 0], [1, 1, 4]], dtype=float))
    tf = signac_tfidf(counts).toarray()
    cell = counts.toarray().sum(1, keepdims=True)
    peak = counts.toarray().sum(0, keepdims=True)
    expect = np.log1p(counts.toarray() / cell * (3 / peak) * 1e4)
    expect[counts.toarray() == 0] = 0
    np.testing.assert_allclose(tf, expect, rtol=1e-14)
    ln = seurat_lognormalize(counts).toarray()
    np.testing.assert_allclose(ln, np.log1p(counts.toarray() / cell * 1e4), rtol=1e-14)


def test_grn_namespace_reexports_scenic():
    assert epi.grn.scenic is epi.scenic


def _synthetic(tmp_path, n_cells=120, seed=0):
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    chrom = "".join(alphabet[rng.integers(0, 4, 60000)])
    # plant GATA-like sites in a few peaks so motifs exist
    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + "\n".join(chrom[i:i + 60] for i in range(0, len(chrom), 60)) + "\n")
    starts = np.arange(1000, 58000, 900)
    peaks = [f"chr1:{s}-{s + 400}" for s in starts]
    genes = [f"G{i}" for i in range(12)] + ["GATA1", "TAL1", "SPI1", "CEBPA"]
    ann = []
    for i, g in enumerate(genes):
        s = 2000 + i * 3500
        ann.append(("chr1", s, s + 600, "+", f"T{i}", g, f"E{i}", "protein_coding", "exon"))
        ann.append(("chr1", s + 1500, s + 1800, "+", f"T{i}", g, f"E{i}", "protein_coding", "exon"))
    ann = pd.DataFrame(ann, columns=["seqnames", "start", "end", "strand", "tx_id", "gene_name",
                                     "gene_id", "gene_biotype", "type"])
    z = rng.normal(size=(n_cells, 3))
    rna_counts = rng.poisson(np.exp(0.5 + z @ rng.normal(size=(3, len(genes))) * 0.6))
    atac_counts = rng.poisson(np.exp(-0.5 + z @ rng.normal(size=(3, len(peaks))) * 0.6))
    import anndata as ad

    cells = [f"c{i}" for i in range(n_cells)]
    rna = ad.AnnData(sp.csr_matrix(rna_counts.astype(float)), obs=pd.DataFrame(index=cells),
                     var=pd.DataFrame(index=genes))
    atac = ad.AnnData(sp.csr_matrix(atac_counts.astype(float)), obs=pd.DataFrame(index=cells),
                      var=pd.DataFrame(index=peaks))
    return rna, atac, ann, str(fasta), genes[:12]


def test_near_target_scan_equals_full_scan(tmp_path):
    pp = pytest.importorskip("pypando")
    rna, atac, ann, fasta, targets = _synthetic(tmp_path)
    kw = dict(genome=fasta, annotation=ann, genes=targets, method="glm", tf_cor=0.0, upstream=5000,
              find_modules=False, key_added=None, verbose=False)
    full = epi.grn.pando(rna, atac, scan="all", **kw)
    near = epi.grn.pando(rna, atac, scan="near_targets", **kw)
    a = pp.coef(full).sort_values(["target", "tf", "region"]).reset_index(drop=True)
    b = pp.coef(near).sort_values(["target", "tf", "region"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(pp.gof(full).reset_index(drop=True), pp.gof(near).reset_index(drop=True))


def test_pando_wrapper_stores_tables(tmp_path):
    pytest.importorskip("pypando")
    rna, atac, ann, fasta, targets = _synthetic(tmp_path, seed=1)
    epi.grn.pando(rna, atac, genome=fasta, annotation=ann, genes=targets, tf_cor=0.0, upstream=5000,
                  find_modules=True, nvar_thresh=0, rsq_thresh=0.0, verbose=False)
    out = rna.uns["pando"]
    assert {"coefs", "gof", "modules", "params"} <= set(out)
    assert len(out["gof"]) > 0
