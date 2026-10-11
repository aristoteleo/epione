"""Fast LDA sampler for :func:`epione.scenic.run_topic_models` (``method="parallel"``).

Approximate distributed collapsed Gibbs sampling (AD-LDA; Newman *et al.*,
JMLR 2009), partitioned by **regions** instead of cells:

* Regions are split into contiguous blocks with about the same number of
  tokens, one block per task. A block owns the topic counts of its regions
  (``n_wk``), so those are always exact and need no synchronisation.
* Cell-topic counts ``n_dk`` and topic totals ``n_k`` are shared. During a
  sweep each block works on its own int32 copy of the cell × topic counts
  (small for single-cell data: ~200 kB at 2,400 cells × 20 topics), taken at
  the start of the sweep; the blocks' changes are summed into the shared counts
  after every sweep. No copy of the regions × topics matrix is ever made.
* Tokens are visited region by region, so the region's count row and the
  cells × topics table stay in cache, and the per-token weights
  ``(n_wk + β)/(n_k + Vβ) · (n_dk + α)`` are a dense, vectorisable multiply over
  topics: the region factor is cached per region and the two entries touched
  by a token update are refreshed in O(1). (For scATAC data SparseLDA's bucket split saves little: a cell
  has ~10⁴ tokens, so ``n_dk`` is non-zero for nearly every topic, and a dense
  multiply over cached rows is cheaper than walking sparse lists.)

Results are reproducible for a given ``random_state`` and number of blocks.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from numba import njit


@njit(cache=True, inline="always")
def _next_u01(state):
    """splitmix64 step on ``state[0]``; uniform double in [0, 1)."""
    state[0] += np.uint64(0x9E3779B97F4A7C15)
    z = state[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return float(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, nogil=True)
def _init_counts(TW, TD, ZS, nwz, ndz, nz):
    for i in range(TW.shape[0]):
        z = ZS[i]
        nwz[TW[i], z] += 1
        ndz[TD[i], z] += 1
        nz[z] += 1


@njit(cache=True, nogil=True, fastmath=True)
def _sweep_block(TD, ZS, word_ptr, w0, w1, nwz, ndz, lndz, nz, dnz, alpha, beta, vbeta, state):
    """One sweep over the tokens of regions ``w0..w1`` (region-major order).

    ``ndz`` and ``nz`` are the counts at the start of the sweep (read only). The
    block works on its own copy ``lndz`` of the cell-topic counts and on
    ``dnz`` (its change of the topic totals); ``nwz`` rows of the block are
    owned and updated in place.
    """
    K = nz.shape[0]
    D = ndz.shape[0]
    for d in range(D):
        for k in range(K):
            lndz[d, k] = ndz[d, k]
    nk = np.empty(K)
    inv = np.empty(K)
    wpart = np.empty(K)
    p = np.empty(K)
    for k in range(K):
        nk[k] = nz[k] + dnz[k]
        inv[k] = 1.0 / (nk[k] + vbeta)
    for w in range(w0, w1):
        if word_ptr[w] == word_ptr[w + 1]:
            continue
        for k in range(K):
            wpart[k] = (nwz[w, k] + beta) * inv[k]
        for i in range(word_ptr[w], word_ptr[w + 1]):
            d = TD[i]
            z = ZS[i]
            nwz[w, z] -= 1
            lndz[d, z] -= 1
            nk[z] -= 1.0
            dnz[z] -= 1
            inv[z] = 1.0 / (nk[z] + vbeta)
            wpart[z] = (nwz[w, z] + beta) * inv[z]
            c = 0.0
            for k in range(K):
                v = wpart[k] * (lndz[d, k] + alpha)
                p[k] = v
                c += v
            u = _next_u01(state) * c
            z_new = K - 1
            for k in range(K - 1):
                u -= p[k]
                if u < 0.0:
                    z_new = k
                    break
            ZS[i] = z_new
            nwz[w, z_new] += 1
            lndz[d, z_new] += 1
            nk[z_new] += 1.0
            dnz[z_new] += 1
            inv[z_new] = 1.0 / (nk[z_new] + vbeta)
            wpart[z_new] = (nwz[w, z_new] + beta) * inv[z_new]


@njit(cache=True, nogil=True)
def _merge_doc_counts(ndz, lndz_list, nz, dnz_list):
    """ndz += sum of every block's change (its copy minus the start-of-sweep counts)."""
    T = len(lndz_list)
    for d in range(ndz.shape[0]):
        for k in range(ndz.shape[1]):
            base = ndz[d, k]
            acc = base
            for t in range(T):
                acc += lndz_list[t][d, k] - base
            ndz[d, k] = acc
    for t in range(T):
        dn = dnz_list[t]
        for k in range(nz.shape[0]):
            nz[k] += dn[k]
            dn[k] = 0


def default_threads() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        return os.cpu_count() or 1


def region_major_tokens(binary):
    """Token arrays (region, cell) in region-major order, plus the region pointer."""
    import scipy.sparse as sp

    csc = sp.csc_matrix(binary)
    csc.sort_indices()
    counts = csc.data.astype(np.int64)
    per_entry_w = np.repeat(np.arange(csc.shape[1]), np.diff(csc.indptr))
    keep = counts > 0
    TW = np.repeat(per_entry_w[keep], counts[keep]).astype(np.int32)
    TD = np.repeat(csc.indices[keep], counts[keep]).astype(np.int32)
    word_ptr = np.concatenate(([0], np.cumsum(np.bincount(TW, minlength=csc.shape[1])))).astype(
        np.int64)
    return TW, TD, word_ptr


def _blocks(word_ptr: np.ndarray, n_blocks: int):
    W = word_ptr.shape[0] - 1
    n_blocks = max(1, min(n_blocks, W))
    targets = np.linspace(0, word_ptr[-1], n_blocks + 1)[1:-1]
    cuts = np.searchsorted(word_ptr, targets)
    bounds = np.unique(np.concatenate(([0], cuts, [W])))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def fit_parallel(TW, TD, word_ptr, n_docs, n_topics, n_iter, alpha, eta, random_state,
                 pool: ThreadPoolExecutor, n_blocks: int):
    """Region-partitioned AD-LDA; returns ``(nzw, ndz, nz)`` in ``lda``'s layout."""
    from numba.typed import List

    W = word_ptr.shape[0] - 1
    K = int(n_topics)
    ss = np.random.SeedSequence(int(random_state))
    init_seed, block_seed = ss.spawn(2)
    ZS = np.random.default_rng(init_seed).integers(0, K, TW.shape[0], dtype=np.int32)
    nwz = np.zeros((W, K), dtype=np.int32)
    ndz = np.zeros((n_docs, K), dtype=np.int32)
    nz = np.zeros(K, dtype=np.int64)
    _init_counts(TW, TD, ZS, nwz, ndz, nz)

    blocks = _blocks(word_ptr, n_blocks)
    lndz = List([np.zeros((n_docs, K), dtype=np.int32) for _ in blocks])
    dnz = List([np.zeros(K, dtype=np.int64) for _ in blocks])
    states = [np.array([s.generate_state(1, np.uint64)[0]], dtype=np.uint64)
              for s in block_seed.spawn(len(blocks))]
    beta = float(eta)
    vbeta = beta * W
    for _ in range(n_iter):
        futs = [pool.submit(_sweep_block, TD, ZS, word_ptr, a, b, nwz, ndz, lndz[p], nz, dnz[p],
                            float(alpha), beta, vbeta, states[p])
                for p, (a, b) in enumerate(blocks)]
        for f in futs:
            f.result()
        _merge_doc_counts(ndz, lndz, nz, dnz)
    return np.ascontiguousarray(nwz.T), ndz, nz.astype(np.int32)
