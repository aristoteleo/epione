"""Matrix-free iterative LSI: equality with the materialising implementation
and a memory-regression test.

* The streamed TF-logIDF operator gives bit-identical products to the
  materialised SciPy matrix, and ``iterative_lsi`` gives bit-identical
  outputs with ``EPIONE_LSI_STREAM=0`` (the previous implementation, kept as
  a switch) and the default streaming path.
* On a synthetic tile matrix the extra memory ``iterative_lsi`` allocates
  stays under a quarter of the input CSR, whereas the materialising path
  (copies of the feature subset and its TF-logIDF values) exceeds it.
"""
import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData


def _random_tiles(n_cells, n_feat, nnz_per_cell, n_groups=3, seed=0):
    """Binary-ish cell x tile CSR (int32 counts, sorted indices) with group
    structure, built without Python loops over cells."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n_cells) % n_groups
    k = rng.integers(int(nnz_per_cell * 0.6), int(nnz_per_cell * 1.4), size=n_cells)
    rows = np.repeat(np.arange(n_cells), k)
    # half shared (skewed popularity), half group-specific
    pop = (rng.random(rows.size) ** 2 * n_feat).astype(np.int64)   # skewed
    own = (labels[rows] * (n_feat // n_groups)
           + rng.integers(0, n_feat // n_groups, size=rows.size))
    cols = np.where(rng.random(rows.size) < 0.5, pop, own)
    X = sp.csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)),
                      shape=(n_cells, n_feat))
    X.sum_duplicates()
    X.data[:] = rng.integers(1, 4, size=X.nnz).astype(np.int32)
    X.sort_indices()
    ad = AnnData(X=X, obs={"group": labels.astype(str),
                           "n_fragment": np.asarray(X.sum(axis=1)).ravel()})
    return ad


@pytest.fixture(params=[1, 3])
def n_jobs(request):
    return request.param


def _lsi(ad, monkeypatch, stream, **kw):
    import epione as ep
    monkeypatch.setenv("EPIONE_LSI_STREAM", "1" if stream else "0")
    ad = ad.copy()
    args = dict(n_components=15, iterations=3, var_features=1500, total_features=6000,
                resolution=0.3, sample_cells_pre=600, depth_col="n_fragment",
                seed=0, verbose=False)
    args.update(kw)
    ep.tl.iterative_lsi(ad, **args)
    return ad


def test_stream_products_match_scipy(n_jobs):
    from epione.tl._iterative_lsi import _tf_logidf
    from epione.tl._lsi_kernels import BinaryCSR, csr_select
    from epione.tl._lsi_stream import TfidfStream
    ad = _random_tiles(700, 3000, 120, seed=1)
    X = ad.X
    Xb = BinaryCSR(X.indptr, X.indices, X.shape)
    rng = np.random.default_rng(0)
    cols = np.sort(rng.choice(X.shape[1], 1800, replace=False))
    for rows in (None, rng.choice(X.shape[0], 400, replace=False)):
        ref = _tf_logidf(csr_select(Xb, cols, rows=rows))
        op = TfidfStream(Xb, cols, rows=rows, n_jobs=n_jobs, block_nnz=997)
        assert len(op.blocks) > 3
        M = op.materialize()
        assert np.array_equal(M.indptr, ref.indptr)
        assert np.array_equal(M.indices, ref.indices)
        assert np.array_equal(M.data, ref.data)
        Q = rng.normal(size=(op.shape[1], 7)).astype(np.float32)
        P = rng.normal(size=(op.shape[0], 7)).astype(np.float32)
        assert np.array_equal(op @ Q, ref @ Q)
        assert np.array_equal(op.T @ P, ref.T @ P)            # csc_matvecs order
        assert np.array_equal(P.T @ op, P.T @ ref)
        assert np.array_equal(Q.T @ op.T, Q.T @ ref.T)
        assert np.array_equal(op.col_sums,
                              np.asarray(csr_select(Xb, cols, rows=rows).sum(axis=0)).ravel())


@pytest.mark.parametrize("block", [None, 1500])
def test_iterative_lsi_stream_bit_identical(monkeypatch, n_jobs, block):
    import epione.tl._lsi_stream as ls
    if block:
        monkeypatch.setattr(ls, "_BLOCK_NNZ", block)
    ad = _random_tiles(1500, 8000, 300, seed=2)
    old = _lsi(ad, monkeypatch, False, n_jobs=n_jobs)
    new = _lsi(ad, monkeypatch, True, n_jobs=n_jobs)
    assert np.array_equal(old.obsm["X_iterative_lsi"], new.obsm["X_iterative_lsi"])
    assert np.array_equal(old.varm["X_iterative_lsi_loadings"],
                          new.varm["X_iterative_lsi_loadings"])
    uo, un = old.uns["X_iterative_lsi"], new.uns["X_iterative_lsi"]
    for k in ("stdev", "feature_sums", "features_final", "depth_cor", "kept_dims",
              "scale_mean", "scale_sd"):
        assert np.array_equal(uo[k], un[k]), k
    assert np.array_equal(uo["iterations"]["n_clusters"], un["iterations"]["n_clusters"])


def test_iterative_lsi_stream_float_binarised_input(monkeypatch):
    """A float32 0/1 input (``_binarize`` output) also takes the stream."""
    ad = _random_tiles(900, 5000, 200, seed=3)
    old = _lsi(ad, monkeypatch, False, iterations=2)
    ad.X = ad.X.astype(np.float32)
    ad.X.data[:] = 1.0
    new = _lsi(ad, monkeypatch, True, iterations=2)
    assert np.array_equal(old.obsm["X_iterative_lsi"], new.obsm["X_iterative_lsi"])


def test_iterative_lsi_stream_torch(monkeypatch):
    pytest.importorskip("torch")
    ad = _random_tiles(900, 5000, 200, seed=4)
    kw = dict(iterations=1, sample_cells_pre=None)
    old = _lsi(ad, monkeypatch, False, device="torch-cpu", **kw)
    new = _lsi(ad, monkeypatch, True, device="torch-cpu", **kw)
    cpu = _lsi(ad, monkeypatch, True, **kw)
    # every block streamed host -> device for each product
    monkeypatch.setenv("EPIONE_GPU_MEMORY_GB", "0.0001")
    streamed = _lsi(ad, monkeypatch, True, device="torch-cpu", **kw)
    # the tolerance of tests/test_gpu_torch.py (torch vs CPU, one round)
    for x in (old, cpu, streamed):
        assert list(x.uns["X_iterative_lsi"]["kept_dims"]) == \
            list(new.uns["X_iterative_lsi"]["kept_dims"])
        assert _abs_cos(x.obsm["X_iterative_lsi"], new.obsm["X_iterative_lsi"]).min() > 0.999


def _abs_cos(A, B):
    return np.abs((A * B).sum(0) / np.linalg.norm(A, axis=0) / np.linalg.norm(B, axis=0))


def test_tile_gene_product_matches_full():
    from epione.tl._gene_score import _tile_gene_product
    ad = _random_tiles(500, 4000, 150, seed=5)
    X = ad.X
    rng = np.random.default_rng(0)
    W = sp.random(4000, 300, density=0.01, format="csr", dtype=np.float32, random_state=0)
    Xt = X.astype(np.float32)
    np.clip(Xt.data, 0, 2, out=Xt.data)
    ref = (Xt @ W).astype(np.float32).toarray()
    for nj in (1, 3):
        got = _tile_gene_product(X, W, 2, n_jobs=nj, block_nnz=5000)
        assert np.array_equal(ref, got)


def _peak_extra_bytes(fn):
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak - base


def test_iterative_lsi_memory_regression(monkeypatch):
    """Peak extra memory of iterative_lsi < 25% of the input CSR.

    The input is 8k cells x 30k tiles with ~3k nnz per cell (~180 MB of
    CSR, int32 indices + int32 counts). The materialising implementation
    copies the selected cells x features (int32 indices + float32 ones) and
    their TF-logIDF values -- 12 B/nnz, more than the 8 B/nnz input -- which
    this budget catches. The intermediate kNN + Leiden is replaced by a
    fixed labelling: its memory scales with the number of cells, not with
    the matrix, and is not what this test is about.
    """
    import epione as ep
    import epione.tl._iterative_lsi as il
    import epione.tl._lsi_stream as ls
    monkeypatch.setattr(ls, "_BLOCK_NNZ", 1 << 16)
    monkeypatch.setattr(il, "_knn_leiden", lambda emb, **kw: (
        np.argmax(np.abs(emb[:, :4]), axis=1).astype(str)))
    ad = _random_tiles(8000, 30000, 3000, seed=6)
    X = ad.X
    csr_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    budget = 0.25 * csr_bytes
    kw = dict(n_components=20, iterations=2, var_features=3000, total_features=25000,
              sample_cells_pre=2000, depth_col="n_fragment", seed=0, n_jobs=2,
              verbose=False)

    monkeypatch.setenv("EPIONE_LSI_STREAM", "1")
    extra_new = _peak_extra_bytes(lambda: ep.tl.iterative_lsi(ad, **kw))
    emb_new = ad.obsm["X_iterative_lsi"].copy()
    monkeypatch.setenv("EPIONE_LSI_STREAM", "0")
    extra_old = _peak_extra_bytes(lambda: ep.tl.iterative_lsi(ad, **kw))
    assert np.array_equal(emb_new, ad.obsm["X_iterative_lsi"])
    assert extra_old > budget, (extra_old, budget)     # the test can fail
    assert extra_new < budget, (
        f"iterative_lsi allocated {extra_new / 2**20:.0f} MiB at peak, budget "
        f"{budget / 2**20:.0f} MiB (input CSR {csr_bytes / 2**20:.0f} MiB)")
