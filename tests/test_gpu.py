"""GPU backend helpers that need no torch: device resolution, batching plans,
and the rapids-singlecell dispatch (mocked). Runs on CI."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import scipy.sparse as sp

from epione.core import _gpu


# ---------------------------------------------------------------------------
# device resolution (no torch needed)
# ---------------------------------------------------------------------------

def test_resolve_cpu_never_imports_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)   # import torch -> ImportError
    assert _gpu.resolve_device("cpu") == "cpu"
    assert _gpu.resolve_device(None) == "cpu"
    assert _gpu.resolve_device("CPU") == "cpu"
    assert not _gpu.is_torch_device("cpu")


def test_resolve_auto_without_torch_is_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert _gpu.resolve_device("auto") == "cpu"


def test_resolve_cuda_without_torch_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"epione\[gpu\]"):
        _gpu.resolve_device("cuda")


def test_resolve_unknown_device():
    with pytest.raises(ValueError, match="unknown device"):
        _gpu.resolve_device("tpu")


def test_resolve_cuda_unavailable_raises():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA available")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        _gpu.resolve_device("cuda")
    assert _gpu.resolve_device("auto") == "cpu"


def test_memory_budget_env(monkeypatch):
    monkeypatch.setenv("EPIONE_GPU_MEMORY_GB", "1.5")
    assert _gpu.memory_budget(None) == int(1.5 * 2**30)
    monkeypatch.delenv("EPIONE_GPU_MEMORY_GB")
    assert _gpu.memory_budget(None) == 4 * 2**30


# ---------------------------------------------------------------------------
# batching plans (no torch needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,bpr,mx", [(0, 8, 100), (1, 8, 100), (10, 8, 100),
                                      (1000, 13, 999), (7, 1000, 10)])
def test_row_blocks_cover_and_bound(n, bpr, mx):
    blocks = _gpu.row_blocks(n, bpr, mx)
    covered = np.concatenate([np.arange(b.start, b.stop) for b in blocks]) if blocks else np.array([], int)
    assert np.array_equal(covered, np.arange(n))
    for b in blocks:
        assert (b.stop - b.start) * bpr <= mx or b.stop - b.start == 1


def test_csr_row_blocks_bound_nnz():
    rng = np.random.default_rng(0)
    X = sp.random(500, 80, density=0.1, format="csr", random_state=1)
    X[7] = 1.0  # a dense row larger than the budget gets its own block
    X = X.tocsr()
    blocks = _gpu.csr_row_blocks(X.indptr, max_nnz=60)
    covered = np.concatenate([np.arange(b.start, b.stop) for b in blocks])
    assert np.array_equal(covered, np.arange(500))
    for b in blocks:
        nnz = X.indptr[b.stop] - X.indptr[b.start]
        assert nnz <= 60 or b.stop - b.start == 1
    rb = _gpu.csr_row_blocks(X.indptr, max_nnz=10**9, max_rows=33)
    assert all(b.stop - b.start <= 33 for b in rb)
    assert _gpu.csr_row_blocks(np.array([0]), 5) == []
    del rng


# ---------------------------------------------------------------------------
# rapids-singlecell dispatch (mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_rsc(monkeypatch):
    calls = {}
    rsc = types.ModuleType("rapids_singlecell")
    rsc.pp = types.SimpleNamespace(neighbors=lambda adata, **kw: calls.setdefault("neighbors", kw))
    rsc.tl = types.SimpleNamespace(umap=lambda adata, **kw: calls.setdefault("umap", kw),
                                   leiden=lambda adata, **kw: calls.setdefault("leiden", kw))
    monkeypatch.setitem(sys.modules, "rapids_singlecell", rsc)
    return calls


@pytest.fixture
def peaks():
    import anndata as ad
    a = ad.AnnData(sp.csr_matrix(np.eye(20, dtype=np.float32)))
    return a


def test_rapids_dispatch(fake_rsc, peaks):
    import epione as epi
    epi.pp.neighbors(peaks, n_neighbors=7, use_rep="X", backend="rapids", metric="cosine")
    assert fake_rsc["neighbors"]["algorithm"] == "brute"
    assert fake_rsc["neighbors"]["n_neighbors"] == 7
    assert fake_rsc["neighbors"]["metric"] == "cosine"
    epi.tl.umap(peaks, backend="rapids", random_state=3, min_dist=0.3)
    assert fake_rsc["umap"]["random_state"] == 3 and fake_rsc["umap"]["min_dist"] == 0.3
    epi.tl.clusters(peaks, method="leiden", backend="rapids", key_added="cl",
                    leiden_kwargs=dict(resolution=0.7, random_state=2, flavor="igraph",
                                       n_iterations=2, directed=False))
    kw = fake_rsc["leiden"]
    assert kw["resolution"] == 0.7 and kw["key_added"] == "cl" and kw["n_iterations"] == 2
    assert "flavor" not in kw and "directed" not in kw


def test_rapids_neighbors_drops_self_entries(monkeypatch):
    import anndata as ad
    import epione as epi
    a = ad.AnnData(sp.csr_matrix(np.eye(4, dtype=np.float32)))

    def fake_neighbors(adata, **kw):
        D = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 3], [0, 0, 3, 0]], float))
        D = D + sp.csr_matrix((np.zeros(4), (np.arange(4), np.arange(4))), shape=(4, 4))
        D = sp.csr_matrix((np.r_[D.data, np.zeros(4)],
                           (np.r_[D.tocoo().row, np.arange(4)], np.r_[D.tocoo().col, np.arange(4)])),
                          shape=(4, 4))
        adata.obsp["distances"] = D
    rsc = types.ModuleType("rapids_singlecell")
    rsc.pp = types.SimpleNamespace(neighbors=fake_neighbors)
    monkeypatch.setitem(sys.modules, "rapids_singlecell", rsc)
    epi.pp.neighbors(a, backend="rapids", n_neighbors=3, use_rep="X")
    D = a.obsp["distances"].tocoo()
    assert not np.any(D.row == D.col)
    assert D.nnz == 6


def test_rapids_missing_message(monkeypatch, peaks):
    import epione as epi
    monkeypatch.setitem(sys.modules, "rapids_singlecell", None)
    with pytest.raises(ImportError, match="rapids-singlecell"):
        epi.pp.neighbors(peaks, backend="rapids")
    with pytest.raises(ValueError, match="method='leiden'"):
        epi.tl.clusters(peaks, method="louvain", backend="rapids")
