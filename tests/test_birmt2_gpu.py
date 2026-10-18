"""GPU path of BiRMT-2 (``epi.tl.birmt(device='cuda')``): CPU-vs-GPU
equivalence on a synthetic binary matrix (rows shorter than a summation
segment: bit-identical; longer rows: round-off, same rank), the
memory-lean view layout, out-of-sample low-degree cells, streamed blocks
within a small device budget, and the CPU fallback.

The CUDA tests are skipped when no CUDA device is visible.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData


def _cuda_ok():
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        from epione.tl._birmt2_gpu import gpu_unavailable_reason
        return gpu_unavailable_reason(torch.device("cuda:0")) is None
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _cuda_ok(), reason="needs CUDA (+ NVRTC)")


def _matrix(m=1500, n=6000, r=4, seed=0, n_low=0, n_rare=200):
    """Binary cells x features with a planted rank-r structure, optional
    near-empty cells (Sinkhorn-infeasible: out of sample) and rare
    features (dropped by min_cells, exercising the column remap)."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, r, m)
    ft = rng.integers(0, r, n)
    P = np.where(lab[:, None] == ft[None, :], 0.12, 0.03)
    Y = rng.random((m, n)) < P
    Y[:, :n_rare] = rng.random((m, n_rare)) < 0.001
    for i in range(n_low):
        Y[i] = False
        Y[i, rng.choice(np.arange(n_rare, n), 2, replace=False)] = True
    X = sp.csr_matrix(Y.astype(np.float32))
    X.data[:] = rng.integers(1, 4, X.nnz)          # counts: the pattern is what matters
    return X


def _run(X, monkeypatch=None, **kw):
    import epione as epi
    ad = AnnData(X.copy())
    epi.tl.birmt(ad, **kw)
    return ad


@needs_cuda
@pytest.mark.parametrize("layout", ["compact", "view"])
@pytest.mark.parametrize("n_low", [0, 7])
def test_gpu_bit_identical_to_cpu(monkeypatch, layout, n_low):
    monkeypatch.setenv("EPIONE_BIRMT_PATTERN", layout)
    X = _matrix(n_low=n_low)
    a = _run(X, n_jobs=2)
    b = _run(X, n_jobs=2, device="cuda")
    ua, ub = a.uns["X_birmt"], b.uns["X_birmt"]
    assert ub["device"].startswith("cuda") and ua["device"] == "cpu"
    assert ua["levels"][0]["rank"] == ub["levels"][0]["rank"] >= 2
    assert ua["levels"][0]["alpha"] == ub["levels"][0]["alpha"]
    assert ua["sinkhorn_iter"] == ub["sinkhorn_iter"]
    np.testing.assert_array_equal(a.obsm["X_birmt"], b.obsm["X_birmt"])


@needs_cuda
def test_gpu_streamed_blocks_within_budget(monkeypatch):
    """Tiny blocks, nothing resident: every product streams the index
    arrays through the staging slots; the result is unchanged and the
    device peak stays within the budget."""
    import torch
    from epione.tl import _birmt2, _birmt2_gpu
    X = _matrix(m=2000, n=8000, seed=3, n_low=3)
    Y, _ = _birmt2._filter_columns(X, 15)
    ref = _birmt2.fast_denoise_binary(Y, seed=0, n_jobs=2)
    budget = 256 << 20
    monkeypatch.setattr(_birmt2_gpu, "_BLOCK_NNZ", (50_000, 50_000))
    for cap in (0, 1_000_000):          # all streamed / partly resident
        monkeypatch.setattr(_birmt2_gpu, "_RESIDENT_CAP", cap)
        got = _birmt2.fast_denoise_binary(Y, seed=0, n_jobs=2, device=torch.device("cuda:0"),
                                          gpu_budget=budget)
        op = got.pop("op")
        assert len(op.R.blocks) > 4 and len(op.C.blocks) > 4
        if cap == 0:
            assert not op.R.resident and not op.C.resident
        else:
            assert 0 < len(op.R.resident) + len(op.C.resident) < len(op.R.blocks) + len(op.C.blocks)
        assert got["gpu_peak_bytes"] <= budget
        op.close()
        assert got["rank"] == ref["rank"]
        np.testing.assert_array_equal(got["U"], ref["U"])
        np.testing.assert_array_equal(got["singular_values"], ref["singular_values"])
        np.testing.assert_array_equal(got["row_scale"], ref["row_scale"])


@needs_cuda
def test_gpu_operator_products_match_cpu():
    import torch
    from epione.tl import _birmt2
    from epione.tl._birmt2_gpu import CudaBinaryBiwhitened
    X = _matrix(m=800, n=3000, seed=5)
    Y, _ = _birmt2._filter_columns(X, 3)
    Yt = Y.transpose()
    rng = np.random.default_rng(0)
    x, y = rng.random(Y.shape[0]) + 0.5, rng.random(Y.shape[1]) + 0.5
    cpu = _birmt2.BinaryBiwhitened(Y, x, y, Yt)
    gpu = CudaBinaryBiwhitened(Y, Yt, torch.device("cuda:0"))
    gpu.set_scaling(x, y)
    for b in (1, 16, 45):
        V = rng.standard_normal((Y.shape[1], b))
        U = rng.standard_normal((Y.shape[0], b))
        np.testing.assert_array_equal(gpu.matmat(V), cpu.matmat(V))
        np.testing.assert_array_equal(gpu.rmatmat(U), cpu.rmatmat(U))
        np.testing.assert_array_equal(gpu.normal_matmat(U), cpu.matmat(cpu.rmatmat(U)))
    xs, ys, it = gpu.sinkhorn()
    xc, yc, itc = _birmt2.sinkhorn_binary(Y, Yt)
    assert it == itc
    np.testing.assert_array_equal(xs, xc)
    np.testing.assert_array_equal(ys, yc)
    gpu.close()


def test_cpu_fallback_without_cuda(monkeypatch):
    """device='cuda' without a usable GPU warns and gives the CPU result."""
    from epione.tl import _birmt2
    monkeypatch.setattr(_birmt2, "_resolve_device", lambda d: None)
    X = _matrix(m=600, n=2500, seed=2)
    a = _run(X, n_jobs=1)
    b = _run(X, n_jobs=1, device="cuda")
    assert b.uns["X_birmt"]["device"] == "cpu"
    np.testing.assert_array_equal(a.obsm["X_birmt"], b.obsm["X_birmt"])


def test_resolve_device_falls_back(monkeypatch):
    from epione.tl import _birmt2
    from epione.core import _gpu
    assert _birmt2._resolve_device("cpu") is None

    def boom(d):
        raise RuntimeError("no CUDA")
    monkeypatch.setattr(_gpu, "resolve_device", boom)
    assert _birmt2._resolve_device("cuda") is None


def test_version1_rejects_device():
    import epione as epi
    with pytest.raises(ValueError):
        epi.tl.birmt(AnnData(_matrix(m=200, n=800)), version=1, device="cuda")


@needs_cuda
def test_gpu_segmented_long_rows(monkeypatch):
    """Rows longer than a summation segment are summed per segment (fixed,
    layout-independent order): same rank, values within float64 round-off
    of the CPU, and bit-identical between resident and streamed layouts."""
    import torch
    from epione.tl import _birmt2, _birmt2_gpu
    monkeypatch.setattr(_birmt2_gpu, "_SEG", 16)
    X = _matrix(m=1500, n=6000, seed=4)
    Y, _ = _birmt2._filter_columns(X, 12)
    ref = _birmt2.fast_denoise_binary(Y, seed=0, n_jobs=2)
    dev = torch.device("cuda:0")
    got = []
    for cap, blk in ((None, (1 << 22, 1 << 26)), (0, (40_000, 40_000))):
        monkeypatch.setattr(_birmt2_gpu, "_RESIDENT_CAP", cap)
        monkeypatch.setattr(_birmt2_gpu, "_BLOCK_NNZ", blk)
        lv = _birmt2.fast_denoise_binary(Y, seed=0, n_jobs=2, device=dev)
        lv.pop("op").close()
        got.append(lv)
    for lv in got:
        assert lv["rank"] == ref["rank"] and lv["sinkhorn_iter"] == ref["sinkhorn_iter"]
        np.testing.assert_allclose(lv["singular_values"], ref["singular_values"], rtol=1e-10)
        np.testing.assert_allclose(lv["row_scale"], ref["row_scale"], rtol=1e-10)
        # same subspace: |U_gpu^T U_cpu| ~ identity for well-separated values
        k = 3
        M = np.abs(lv["U"][:, :k].T @ ref["U"][:, :k])
        np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-8)
    np.testing.assert_array_equal(got[0]["U"], got[1]["U"])
