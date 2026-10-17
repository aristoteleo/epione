"""Generative (likelihood-ratio) doublet detection for scATAC-seq.

:func:`bayes_doublets` scores every droplet by the posterior probability that
its accessibility profile is a mixture of two singlet-type profiles rather
than one (multinomial mixture likelihood, marginalised over the mixing
fraction), with the doublet rate calibrated by EM from a Poisson-loading
prior. Optional evidence terms: library size (sum of two singlet libraries)
and read-overlap copy number (AMULET principle, :func:`count_overlap_loci`).
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np

from ..utils import console  # noqa: F401

__all__ = ["bayes_doublets", "count_overlap_loci", "loading_doublet_rate"]


# ---------------------------------------------------------------------------
# Read-overlap (copy-number) evidence
# ---------------------------------------------------------------------------

_SEX_MITO = re.compile(r"(^|_)(chr)?(M|MT|X|Y)$", re.IGNORECASE)

_OVL_KERNEL = {}


def _overlap_kernel():
    if _OVL_KERNEL:
        return _OVL_KERNEL["f"]
    import numba as nb

    @nb.njit(nogil=True, cache=False)
    def sweep(key_row, chrom, start, end, min_depth, out_row, out_chrom, out_pos):
        """Fragments sorted by (row, chrom, start). For every (row, chrom) run,
        count maximal intervals covered by >= ``min_depth`` fragments.
        Writes one (row, chrom, pos) per interval; returns their number."""
        n = start.shape[0]
        heap = np.empty(64, dtype=np.int64)   # min-heap of active ends
        n_out = 0
        i = 0
        while i < n:
            r = key_row[i]
            c = chrom[i]
            hs = 0
            in_region = False
            while i < n and key_row[i] == r and chrom[i] == c:
                s = start[i]
                # pop ends <= s (half-open fragments)
                while hs > 0 and heap[0] <= s:
                    hs -= 1
                    last = heap[hs]
                    # sift down
                    j = 0
                    while True:
                        l = 2 * j + 1
                        if l >= hs:
                            break
                        m = l
                        if l + 1 < hs and heap[l + 1] < heap[l]:
                            m = l + 1
                        if heap[m] < last:
                            heap[j] = heap[m]
                            j = m
                        else:
                            break
                    if hs > 0:
                        heap[j] = last
                in_region = in_region and hs >= min_depth
                # push end
                if hs == heap.shape[0]:
                    nh = np.empty(2 * hs, dtype=np.int64)
                    nh[:hs] = heap
                    heap = nh
                e = end[i]
                j = hs
                hs += 1
                while j > 0:
                    p = (j - 1) // 2
                    if heap[p] > e:
                        heap[j] = heap[p]
                        j = p
                    else:
                        break
                heap[j] = e
                if hs >= min_depth and not in_region:
                    if n_out == out_row.shape[0]:
                        return -1
                    out_row[n_out] = r
                    out_chrom[n_out] = c
                    out_pos[n_out] = s
                    n_out += 1
                    in_region = True
                i += 1
        return n_out

    _OVL_KERNEL["f"] = sweep
    return sweep


def _collect_fragments(fragments, barcodes, chrom_names_keep=None, n_jobs=None):
    """Stream ``fragments`` and keep (row, chrom, start, end) of ``barcodes``.
    Returns arrays and the list of chromosome names (codes index it)."""
    from ._fragment_io import BarcodeLookup, map_fragment_chunks

    lookup = BarcodeLookup([str(b) for b in barcodes])
    chrom_index: dict = {}
    chrom_list: list = []

    def fn(ch):
        rows = lookup(ch.barcode_names)[ch.barcode]
        keep = rows >= 0
        return (rows[keep].astype(np.int32), ch.chrom[keep], list(ch.chrom_names),
                ch.start[keep].astype(np.int64), ch.end[keep].astype(np.int64))

    R, C, S, E = [], [], [], []
    for rows, cc, names, s, e in map_fragment_chunks(fragments, fn, n_jobs=n_jobs):
        if rows.size == 0:
            continue
        lut = np.empty(len(names), dtype=np.int32)
        for k, nm in enumerate(names):
            if nm not in chrom_index:
                chrom_index[nm] = len(chrom_list)
                chrom_list.append(nm)
            lut[k] = chrom_index[nm]
        R.append(rows); C.append(lut[cc]); S.append(s); E.append(e)
    if not R:
        z = np.zeros(0, np.int64)
        return z.astype(np.int32), z.astype(np.int32), z, z, chrom_list
    return (np.concatenate(R), np.concatenate(C), np.concatenate(S),
            np.concatenate(E), chrom_list)


def count_overlap_loci(
    fragments,
    barcodes: Sequence[str],
    *,
    min_depth: int = 3,
    exclude_chrom_pattern: Optional[str] = r"(^|_)(chr)?(M|MT|X|Y)$",
    n_jobs: Optional[int] = None,
):
    """Loci covered by ``>= min_depth`` fragments of one barcode.

    A diploid nucleus has two copies of every autosomal base, so at most two
    Tn5 fragments of one nucleus can overlap any base; a locus covered by
    three or more fragments implies more than two genome copies in the
    droplet (Thibodeau et al. 2021, AMULET). For every cell this sweeps its
    fragments in genomic order (min-heap of active fragment ends) and records
    each maximal interval with coverage ``>= min_depth``.

    Returns
    -------
    dict with ``row`` (cell index into ``barcodes``), ``chrom`` (code into
    ``chrom_names``), ``pos`` (interval start), ``chrom_names`` and
    ``n_fragments`` (fragments per cell on the kept chromosomes).
    """
    rows, chrom, start, end, names = _collect_fragments(fragments, barcodes, n_jobs=n_jobs)
    n_cells = len(barcodes)
    if exclude_chrom_pattern:
        pat = re.compile(exclude_chrom_pattern, re.IGNORECASE)
        bad = np.array([bool(pat.search(c)) for c in names] + [False], dtype=bool)
        keep = ~bad[chrom]
        rows, chrom, start, end = rows[keep], chrom[keep], start[keep], end[keep]
    nfrag = np.bincount(rows, minlength=n_cells).astype(np.int64)
    if rows.size == 0:
        z = np.zeros(0, np.int64)
        return dict(row=z, chrom=z, pos=z, chrom_names=names, n_fragments=nfrag)
    # sort by (row, chrom, start): 1 composite key when it fits in 63 bits
    cb = max(1, int(np.ceil(np.log2(max(len(names), 2)))))
    sb = int(np.ceil(np.log2(int(start.max()) + 2)))
    rb = int(np.ceil(np.log2(n_cells + 1)))
    if rb + cb + sb <= 63:
        key = (rows.astype(np.int64) << (cb + sb)) | (chrom.astype(np.int64) << sb) | start
        order = np.argsort(key, kind="stable")
        del key
    else:
        order = np.lexsort((start, chrom, rows))
    rows, chrom, start, end = rows[order], chrom[order], start[order], end[order]
    del order
    rows = rows.astype(np.int64); chrom = chrom.astype(np.int64)
    cap = max(1024, rows.size // 16)
    while True:
        o_r = np.empty(cap, np.int64); o_c = np.empty(cap, np.int64); o_p = np.empty(cap, np.int64)
        n_out = _overlap_kernel()(rows, chrom, start, end, int(min_depth), o_r, o_c, o_p)
        if n_out >= 0:
            break
        cap = min(rows.size + 1, cap * 4)
    return dict(row=o_r[:n_out].copy(), chrom=o_c[:n_out].copy(), pos=o_p[:n_out].copy(),
                chrom_names=names, n_fragments=nfrag)


# ---------------------------------------------------------------------------
# Composition + library-size likelihood
# ---------------------------------------------------------------------------

_LOG2PI = float(np.log(2.0 * np.pi))
_PAIR_KERNEL = {}


def _pair_kernel():
    """numba kernel: per-cell doublet marginal likelihood over candidate pairs."""
    if _PAIR_KERNEL:
        return _PAIR_KERNEL["f"]
    import numba as nb

    @nb.njit(nogil=True, cache=False)
    def _comp(indptr, indices, data, A, Asum, W, eps, g, ni, i, k, l, al):
        """sum_j x_ij log((1-ε)(α q_kj + (1-α) q_lj) + ε g_j) with leave-one-out
        type profiles q_k = (A_k - W_ik x_i) / (Asum_k - W_ik n_i) and a
        background (ambient / non-specific) component g of weight ε."""
        dk = Asum[k] - W[i, k] * ni
        dl = Asum[l] - W[i, l] * ni
        s = 0.0
        for t in range(indptr[i], indptr[i + 1]):
            j = indices[t]
            x = data[t]
            pk = max(A[k, j] - W[i, k] * x, 0.0) / dk
            pl = max(A[l, j] - W[i, l] * x, 0.0) / dl
            s += x * np.log((1.0 - eps) * (al * pk + (1.0 - al) * pl) + eps * g[j] + 1e-300)
        return s

    @nb.njit(parallel=True, nogil=True, cache=False)
    def loo_singlet(indptr, indices, data, A, Asum, W, eps, g, out_L):
        """out_L[i, k] = sum_j x_ij log((1-ε) q^{(-i)}_kj + ε g_j) (leave-one-out)."""
        n = indptr.shape[0] - 1
        K = A.shape[0]
        for i in nb.prange(n):
            ni = 0.0
            for t in range(indptr[i], indptr[i + 1]):
                ni += data[t]
            for k in range(K):
                d = Asum[k] - W[i, k] * ni
                s = 0.0
                for t in range(indptr[i], indptr[i + 1]):
                    j = indices[t]
                    x = data[t]
                    s += x * np.log((1.0 - eps) * max(A[k, j] - W[i, k] * x, 0.0) / d
                                    + eps * g[j] + 1e-300)
                out_L[i, k] = s

    @nb.njit(parallel=True, nogil=True, cache=False)
    def pair_marginal(indptr, indices, data, A, Asum, W, eps, g, L, first, n_pairs, coarse, logw, t_grid,
                      log_alpha, log_1malpha, logN, mu, sig, use_lib, log_prior_t,
                      log_kappa, homotypic, out_logD, out_best, out_bestmix):
        """For cell i:

        1. partner search -- for every first parent k in ``first[i]`` (its top
           singlet types) and every other type l, the composition
           log-likelihood max over the coarse α grid ``coarse``; the
           ``n_pairs`` best (k, l) pairs are kept, plus the homotypic pairs
           (k, k) of the first parents;
        2. for every kept pair the marginal

            log ∫ Mult(x_i; α p_k + (1-α) p_l) · g_kl(α) dt,   α = sigmoid(t),

           g_kl(α) = φ_k(log αN) φ_l(log (1-α)N) (library sizes of the two
           cells: the change of variables N_a = αN, N_b = (1-α)N has Jacobian
           N, and dα = α(1-α) dt cancels the 1/(N_a N_b) of the two log-normal
           densities) or, without library evidence, the uniform prior on α
           written in t (``log_prior_t``), on the uniform t grid;
        3. logsumexp over kept pairs with prior weights ``logw[k, l]``.
        """
        n = indptr.shape[0] - 1
        M = first.shape[1]
        K = A.shape[0]
        G = t_grid.shape[0]
        dt = t_grid[1] - t_grid[0]
        for i in nb.prange(n):
            ni = 0.0
            for t in range(indptr[i], indptr[i + 1]):
                ni += data[t]
            # ---- stage 1: partner search
            sc = np.full(M * K, -np.inf)
            for u in range(M):
                k = first[i, u]
                for l in range(K):
                    if l == k:
                        continue
                    dup = False
                    for u2 in range(u):          # (k, l) already seen as (l, k)
                        if first[i, u2] == l:
                            dup = True
                    if dup:
                        continue
                    b = -np.inf
                    for c in range(coarse.shape[0]):
                        v = _comp(indptr, indices, data, A, Asum, W, eps, g, ni, i, k, l, coarse[c])
                        if v > b:
                            b = v
                    sc[u * K + l] = b + logw[k, l]
            order = np.argsort(-sc)
            npk = min(n_pairs, M * K)
            # ---- stage 2: marginal over kept pairs (+ homotypic of first parents)
            acc_m = -np.inf
            acc_s = 0.0
            best = -np.inf
            best_p = -1
            vals = np.empty(G)
            nq = npk + M if homotypic else npk
            for q in range(nq):
                if q < npk:
                    idx = order[q]
                    if not np.isfinite(sc[idx]):
                        continue
                    k = first[i, idx // K]
                    l = idx % K
                else:
                    k = first[i, q - npk]
                    l = k
                for a in range(G):
                    if k == l:
                        comp = L[i, k]
                    else:
                        comp = _comp(indptr, indices, data, A, Asum, W, eps, g, ni, i, k, l, np.exp(log_alpha[a]))
                    if use_lib:
                        za = (log_alpha[a] + logN[i] - log_kappa - mu[k]) / sig[k]
                        zb = (log_1malpha[a] + logN[i] - log_kappa - mu[l]) / sig[l]
                        lib = (-0.5 * za * za - np.log(sig[k]) - 0.5 * zb * zb
                               - np.log(sig[l]) - _LOG2PI)
                    else:
                        lib = log_prior_t[a]
                    vals[a] = comp + lib
                mx = vals.max()
                tot = 0.0
                for a in range(G):
                    tot += np.exp(vals[a] - mx)
                lp = mx + np.log(tot * dt) + logw[k, l]
                if lp > best and k != l:
                    best = lp
                    best_p = k * 100000 + l
                if lp > acc_m:
                    acc_s = acc_s * np.exp(acc_m - lp) + 1.0
                    acc_m = lp
                else:
                    acc_s += np.exp(lp - acc_m)
            out_logD[i] = acc_m + np.log(acc_s)
            out_best[i] = best_p
            out_bestmix[i] = best

    _PAIR_KERNEL["f"] = pair_marginal
    _PAIR_KERNEL["loo"] = loo_singlet
    return pair_marginal


def _logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _lsi(Xb, n_comps, seed):
    """TF-IDF (log(1 + 1e4·tf)·idf) + truncated SVD; drops the component most
    correlated with depth (|r| > 0.5), L2-normalises rows."""
    import scipy.sparse as sp
    from sklearn.utils.extmath import randomized_svd
    n = Xb.shape[0]
    df = np.asarray(Xb.sum(0)).ravel()
    idf = np.log1p(n / np.maximum(df, 1.0))
    rs = np.asarray(Xb.sum(1)).ravel()
    T = (sp.diags(1.0 / np.maximum(rs, 1.0)) @ Xb @ sp.diags(idf)).tocsr().astype(np.float32)
    T.data = np.log1p(T.data * 1e4)
    U, S, _ = randomized_svd(T, n_comps + 1, n_iter=7, random_state=seed)
    Z = U * S
    lr = np.log(np.maximum(rs, 1.0))
    cor = np.array([abs(np.corrcoef(Z[:, c], lr)[0, 1]) for c in range(Z.shape[1])])
    cor = np.nan_to_num(cor)
    drop = int(np.argmax(cor)) if cor.max() > 0.5 else Z.shape[1] - 1
    Z = np.delete(Z, drop, axis=1)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    return Z


def _knn(Z, k, seed, n_jobs):
    if Z.shape[0] > 50000:
        from pynndescent import NNDescent
        idx, _ = NNDescent(Z, n_neighbors=k + 1, random_state=seed, n_jobs=n_jobs).neighbor_graph
        return idx[:, 1:]
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]


def _leiden(Z, k, resolution, seed, n_jobs):
    import igraph as ig
    import leidenalg
    ind = _knn(Z, k, seed, n_jobs)
    n = Z.shape[0]
    edges = np.column_stack([np.repeat(np.arange(n), ind.shape[1]), ind.ravel()])
    g = ig.Graph(n=n, edges=edges, directed=False).simplify()
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution, seed=seed)
    lab = np.asarray(part.membership)
    return lab


def loading_doublet_rate(n_barcodes: int, effective_droplets: float = 62500.0) -> float:
    """Expected heterotypic+homotypic doublet fraction among non-empty
    droplets from Poisson loading.

    With λ cells per droplet, the fraction of non-empty droplets holding
    ≥ 2 cells is ρ = 1 - λe^{-λ} / (1 - e^{-λ}). λ is solved from the observed
    number of non-empty barcodes, n = D (1 - e^{-λ}), with D effective
    droplets. ``D = 62,500`` reproduces the 10x Chromium rule of thumb of
    ≈ 0.8 % multiplets per 1,000 recovered cells at low loading (ρ ≈ λ/2).
    """
    frac = min(float(n_barcodes) / float(effective_droplets), 0.999)
    lam = -np.log1p(-frac)
    if lam <= 0:
        return 0.0
    return float(1.0 - lam * np.exp(-lam) / (1.0 - np.exp(-lam)))


def _overlap_model(m, logN, r, gamma=None, n_iter=30):
    """Read-overlap (copy-number) likelihood ratio log P(m|D) - log P(m|S).

    m_i (loci covered by >= 3 fragments) is Poisson with depth-dependent
    rates: singlets of type k, log λ_k(N) = a_k + b_S log N (type-specific
    intercepts absorb aneuploid cell types), doublets
    log λ_D(N) = a_D + b_D log N. Rates are weighted Poisson-GLM fits with
    weights (1 - r_i) γ_ik and r_i (Newton-Raphson); the singlet likelihood
    is the γ-mixture over types."""
    from scipy.special import gammaln
    n = m.size
    if gamma is None:
        gamma = np.ones((n, 1))
    K = gamma.shape[1]
    x = logN - logN.mean()

    def glm(Xd, w, y):
        beta = np.zeros(Xd.shape[1])
        beta[:-1] = np.log(max((w * y).sum() / max(w.sum(), 1e-9), 1e-3))
        beta[-1] = 1.0
        for _ in range(n_iter):
            eta = np.clip(Xd @ beta, -30, 30)
            mu = np.exp(eta)
            g = Xd.T @ (w * (y - mu))
            H = (Xd * (w * mu)[:, None]).T @ Xd + 1e-6 * np.eye(Xd.shape[1])
            step = np.linalg.solve(H, g)
            beta += step
            if np.abs(step).max() < 1e-8:
                break
        return beta

    # singlets: stack (cell, type) rows with weights (1 - r) γ
    ws = ((1.0 - r)[:, None] * gamma).ravel()
    Xs = np.zeros((n * K, K + 1))
    Xs[np.arange(n * K), np.tile(np.arange(K), n)] = 1.0
    Xs[:, K] = np.repeat(x, K)
    keep = ws > 1e-6
    bS = glm(Xs[keep], ws[keep], np.repeat(m, K)[keep])
    lamS = np.exp(np.clip(bS[:K][None, :] + bS[K] * x[:, None], -30, 30))      # n x K
    bD = glm(np.column_stack([np.ones(n), x]), r, m)
    lamD = np.exp(np.clip(bD[0] + bD[1] * x, -30, 30))
    lpS = m[:, None] * np.log(lamS) - lamS - gammaln(m + 1)[:, None]
    lpS = _logsumexp(lpS + np.log(np.maximum(gamma, 1e-300)), 1)
    lpD = m * np.log(lamD) - lamD - gammaln(m + 1)
    return lpD - lpS, (bS, bD)


def _smoothed_profiles(S, ncell, pseudocount):
    """Posterior-mean multinomial profiles: counts plus ``pseudocount`` average
    cells of the global profile (Dirichlet prior centred on the pooled
    profile)."""
    tot = S.sum(1, keepdims=True)
    g = S.sum(0) / max(S.sum(), 1e-12)
    P = S + pseudocount * (tot / np.maximum(ncell, 1e-9)[:, None]) * g[None, :]
    return P / np.maximum(P.sum(1, keepdims=True), 1e-300)


def _cluster_mixture_fraction(Xb, lab, *, pseudocount, n_pairs, coarse, t_grid, min_size=10,
                              min_adv_per_fragment=None):
    """For every cluster k, the fraction of its cells' log-likelihood advantage
    of the own profile over the best *other* single type that is recovered
    by the doublet model built from the other types:

        R_k = mean_i [log P(x_i | D, -k) - log P(x_i | S, -k)]
              / mean_i [log P(x_i | k)    - log P(x_i | S, -k)],

    i over cells of k. A population of doublets of two types is explained
    almost entirely by their mixture (R_k ≈ 1); a genuine type is not
    (R_k ≪ 1). Also returns the own-profile advantage per fragment,
    adv_k = mean_i [log P(x_i | k) - log P(x_i | S, -k)] / mean_i n_i.
    R is only computed (else NaN) for clusters of >= ``min_size`` cells and,
    if given, adv_k >= ``min_adv_per_fragment``."""
    import scipy.sparse as sp
    K = int(lab.max()) + 1
    n = Xb.shape[0]
    H = sp.csr_matrix((np.ones(n), (lab, np.arange(n))), shape=(K, n))
    S_all = np.asarray((H @ Xb).todense())
    nk = np.bincount(lab, minlength=K).astype(float)
    P_all = _smoothed_profiles(S_all, nk, pseudocount)
    logP = np.log(P_all)
    g = S_all.sum(0) + 1.0
    g /= g.sum()
    pi = nk / nk.sum()
    log_alpha = -np.logaddexp(0.0, -t_grid)
    log_1malpha = -np.logaddexp(0.0, t_grid)
    log_prior_t = log_alpha + log_1malpha
    kern = _pair_kernel()
    R = np.full(K, np.nan)
    adv = np.full(K, np.nan)
    gz = np.zeros(Xb.shape[1])
    for k in range(K):
        idx = np.flatnonzero(lab == k)
        if idx.size < min_size or K < 3:
            continue
        Xk = Xb[idx]
        Lall = np.asarray(Xk @ logP.T)
        keep = np.flatnonzero(np.arange(K) != k)
        L = np.ascontiguousarray(Lall[:, keep])
        pik = pi[keep] / pi[keep].sum()
        logS = _logsumexp(L + np.log(pik), 1)
        # own profile evaluated leave-one-out (a cell never predicts itself)
        Ak = np.ascontiguousarray(S_all[k][None, :] + pseudocount * (S_all.sum() / n) * g[None, :])
        Lk = np.empty((idx.size, 1))
        _PAIR_KERNEL["loo"](Xk.indptr, Xk.indices, Xk.data, Ak, Ak.sum(1), np.ones((idx.size, 1)),
                            0.0, gz, Lk)
        own = Lk[:, 0] - logS
        adv[k] = np.mean(own) / max(Xk.sum() / idx.size, 1.0)
        if min_adv_per_fragment is not None and not adv[k] >= min_adv_per_fragment:
            continue
        logw = np.log(pik)[:, None] + np.log(pik)[None, :] + np.log(2.0) * (1 - np.eye(keep.size))
        first = np.argsort(-(L + np.log(pik)), axis=1)[:, :min(3, keep.size)].astype(np.int64)
        A = np.ascontiguousarray(P_all[keep]); Asum = np.ones(keep.size)
        W = np.zeros((idx.size, keep.size))
        logD = np.empty(idx.size); best = np.empty(idx.size, np.int64); bm = np.empty(idx.size)
        kern(Xk.indptr, Xk.indices, Xk.data, A, Asum, W, 0.0, gz, L, first, int(n_pairs),
             np.asarray(coarse, float), logw, t_grid, log_alpha, log_1malpha, np.zeros(idx.size),
             np.zeros(keep.size), np.ones(keep.size), False, log_prior_t, 0.0, False, logD, best, bm)
        R[k] = np.mean(logD - logS) / max(np.mean(own), 1e-9)
    return R, adv


def _fit_bayes(Xb, N, *, m=None, n_comps=30, resolution=1.0, n_neighbors=15, n_candidates=5,
               t_max=4.0, n_grid=17, pseudocount=5.0, use_library=True, use_overlap=True,
               rho0=0.1, prior_strength=0.0, n_iter=3, n_pairs=6, coarse=(0.2, 0.5, 0.8),
               homotypic=True, fit_kappa=True, prune_doublet_types=True, prune_min_fraction=0.95,
               prune_min_adv=0.1, loo=True,
               calibrate_rho=True, background="auto", eps_grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), min_sigma=0.15, seed=0, n_jobs=8,
               labels=None, verbose=True):
    """EM fit of the singlet/doublet generative model on binarised counts."""
    import scipy.sparse as sp
    Xb = sp.csr_matrix(Xb, dtype=np.float64)
    n, F = Xb.shape
    logN = np.log(np.maximum(np.asarray(N, dtype=np.float64), 1.0))
    if labels is None:
        Z = _lsi(Xb, n_comps, seed)
        labels = _leiden(Z, n_neighbors, resolution, seed, n_jobs)
    lab = np.asarray(labels)
    t_grid = np.linspace(-t_max, t_max, n_grid)
    log_alpha = -np.logaddexp(0.0, -t_grid)
    log_1malpha = -np.logaddexp(0.0, t_grid)
    log_prior_t = log_alpha + log_1malpha        # uniform α: dα = α(1-α) dt
    K0 = int(lab.max()) + 1
    dbl_type = np.zeros(K0, dtype=bool)
    if prune_doublet_types:
        R, _adv = _cluster_mixture_fraction(Xb, lab, pseudocount=pseudocount, n_pairs=n_pairs,
                                            coarse=coarse, t_grid=t_grid,
                                            min_adv_per_fragment=prune_min_adv)
        dbl_type = np.nan_to_num(R, nan=0.0) >= prune_min_fraction
    kept = np.flatnonzero(~dbl_type)
    remap = np.full(K0, -1)
    remap[kept] = np.arange(kept.size)
    K = kept.size
    gamma = np.zeros((n, K))
    ok = remap[lab] >= 0
    gamma[np.flatnonzero(ok), remap[lab[ok]]] = 1.0
    r = np.where(ok, rho0, 1.0)                  # cells of doublet-type clusters
    rho = rho0
    log_kappa = 0.0
    M = min(n_candidates, K)
    # pooled profile of all cells (+1 per feature): centre of the Dirichlet prior
    gpool = np.asarray(Xb.sum(0)).ravel() + 1.0
    gpool /= gpool.sum()
    kern = _pair_kernel()
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(12)
    gh_w = gh_w / gh_w.sum()
    hist = []
    for it in range(n_iter + 1):
        w = gamma * (1.0 - r)[:, None]
        wk = np.maximum(w.sum(0), 1e-9)
        S = np.asarray((sp.csr_matrix(w.T) @ Xb).todense())          # K x F
        # Dirichlet prior: `pseudocount` average cells of the pooled profile
        A = S + pseudocount * (S.sum() / max(w.sum(), 1e-9)) * gpool[None, :]
        Asum = A.sum(1)
        W = np.ascontiguousarray(w) if loo else np.zeros_like(w)
        pi = wk / wk.sum()
        mu = (w * logN[:, None]).sum(0) / wk
        var = (w * (logN[:, None] - mu) ** 2).sum(0) / wk
        sig = np.maximum(np.sqrt(var), min_sigma)
        L = np.empty((n, K))
        if background == "auto":
            # ε by maximum (singlet-marginal) likelihood on a grid; a subsample
            # of cells suffices for one scalar
            sub = np.random.default_rng(seed).choice(n, min(n, 3000), replace=False)
            sub.sort()
            Xs = Xb[sub]
            Ls = np.empty((sub.size, K))
            best_ll = -np.inf
            for e in eps_grid:
                _PAIR_KERNEL["loo"](Xs.indptr, Xs.indices, Xs.data, A, Asum,
                                    np.ascontiguousarray(W[sub]), float(e), gpool, Ls)
                ll = _logsumexp(Ls + np.log(pi), 1).sum()
                if ll > best_ll:
                    best_ll, eps = ll, float(e)
        else:
            eps = float(background)
        _PAIR_KERNEL["loo"](Xb.indptr, Xb.indices, Xb.data, A, Asum, W, eps, gpool, L)
        if use_library:
            lib = -0.5 * ((logN[:, None] - mu) / sig) ** 2 - np.log(sig) - 0.5 * _LOG2PI
        else:
            lib = np.zeros((n, K))
        sing = L + lib + np.log(pi)
        logS = _logsumexp(sing, 1)
        first = np.argsort(-(L + np.log(pi)), axis=1)[:, :M].astype(np.int64)
        logw = np.log(pi)[:, None] + np.log(pi)[None, :] + np.log(2.0) * (1 - np.eye(K))
        logD = np.empty(n)
        best = np.empty(n, dtype=np.int64)
        bestmix = np.empty(n)
        kern(Xb.indptr, Xb.indices, Xb.data, A, Asum, W, eps, gpool, L, first, int(n_pairs),
             np.asarray(coarse, float),
             logw, t_grid, log_alpha, log_1malpha, logN, mu, sig, bool(use_library), log_prior_t,
             float(log_kappa), bool(homotypic), logD, best, bestmix)
        llr = logD - logS
        if use_overlap and m is not None:
            ovl, _ = _overlap_model(np.asarray(m, float), logN, r, gamma)
        else:
            ovl = np.zeros(n)
        logit = np.log(rho) - np.log1p(-rho) + llr + ovl
        r = 1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700)))
        a = prior_strength * rho0
        b = prior_strength * (1 - rho0)
        rho = float(np.clip((r.sum() + a) / (n + a + b), 1e-4, 0.75))
        gamma = np.exp(sing - logS[:, None])
        if use_library and fit_kappa:
            # library sub-additivity: N_doublet = κ (N_a + N_b); κ by moment
            # matching E[log N | D] over the doublet posterior, with
            # E[log(N_a + N_b)] of the best pair by Gauss-Hermite quadrature.
            Xa = mu[:, None] + sig[:, None] * gh_x[None, :]
            msum = np.einsum("klab,a,b->kl", np.logaddexp(Xa[:, None, :, None], Xa[None, :, None, :]),
                             gh_w, gh_w)
            hk, hl = best // 100000, best % 100000
            okb = best >= 0
            if okb.any() and r[okb].sum() > 1:
                log_kappa = float(np.sum(r[okb] * (logN[okb] - msum[hk[okb], hl[okb]])) / r[okb].sum())
                log_kappa = float(np.clip(log_kappa, np.log(0.25), np.log(2.0)))
        hist.append((rho, float(np.exp(log_kappa)), eps))
        if verbose:
            console.level2(f"[bayes_doublets] iter {it}: rho={rho:.4f}, kappa={np.exp(log_kappa):.3f}, K={K}")
    if calibrate_rho:
        # EM for the doublet rate with the likelihood ratios fixed:
        # ρ <- mean_i P(D | x_i, ρ), under the Beta prior of strength `prior_strength`
        a = prior_strength * rho0
        b = prior_strength * (1 - rho0)
        for _it in range(200):
            if use_overlap and m is not None and _it < 20:
                # the overlap rates are re-fitted with the current responsibilities
                ovl, _ = _overlap_model(np.asarray(m, float), logN, r, gamma)
            ev = llr + ovl
            r = 1.0 / (1.0 + np.exp(-np.clip(np.log(rho) - np.log1p(-rho) + ev, -700, 700)))
            new_rho = float(np.clip((r.sum() + a) / (n + a + b), 1e-4, 0.75))
            if abs(new_rho - rho) < 1e-7 and _it >= 20:
                rho = new_rho
                break
            rho = new_rho
        logit = np.log(rho) - np.log1p(-rho) + ev
        r = 1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700)))
    return dict(bestmix_llr=bestmix - L.max(1), prob=r, llr=llr, ovl=ovl, logit=logit, rho=rho, labels=lab, gamma=gamma,
                best_pair=best, pi=pi, mu=mu, sig=sig, rho_hist=hist,
                doublet_types=np.flatnonzero(dbl_type), kappa=float(np.exp(log_kappa)), eps=eps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bayes_doublets(
    adata,
    *,
    layer: Optional[str] = None,
    n_features: int = 100000,
    n_comps: int = 30,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_candidates: int = 3,
    n_pairs: int = 6,
    pseudocount: float = 5.0,
    n_iter: int = 0,
    use_library: bool = False,
    homotypic: bool = False,
    use_overlap: bool = False,
    fragments=None,
    expected_doublet_rate: Optional[float] = None,
    prior_strength: float = 0.0,
    prune_doublet_types: bool = True,
    loo: bool = False,
    t_max: float = 4.0,
    n_grid: int = 17,
    random_state: int = 0,
    n_jobs: int = 8,
    key_added: str = "bayes_doublet",
    inplace: bool = True,
):
    r"""Generative (likelihood-ratio) doublet detection for scATAC-seq.

    Each droplet is either a singlet of type :math:`k` or a doublet of two
    different types :math:`(k, l)`. With binarised counts :math:`x_i` over the
    ``n_features`` most frequently accessible features:

    * **Singlet**: :math:`P(x_i \mid k) = \mathrm{Mult}(x_i; p_k)` with type
      profiles :math:`p_k` = smoothed type centroids (counts plus
      ``pseudocount`` average cells of the pooled profile, i.e. the posterior
      mean under a Dirichlet prior centred on the pooled profile).
    * **Doublet**: the profile is a mixture of two singlet profiles with an
      unknown fraction :math:`\alpha` of fragments from the first cell,

      .. math::
         P(x_i \mid k, l) = \int_0^1 \mathrm{Mult}(x_i;\, \alpha p_k + (1-\alpha) p_l)\,
         d\alpha = \int \mathrm{Mult}(\cdot)\, \sigma(t)(1-\sigma(t))\, dt,

      evaluated on a uniform grid in :math:`t = \mathrm{logit}\,\alpha`.
    * **Posterior**:

      .. math::
         P(D \mid x_i) = \frac{\rho \sum_{k<l} 2\pi_k\pi_l P(x_i \mid k, l)}
         {\rho \sum_{k<l} 2\pi_k\pi_l P(x_i \mid k, l) + (1 - \rho) \sum_k \pi_k P(x_i \mid k)}.

      ``obs[f"{key_added}_score"]`` is the log posterior odds
      (log ρ/(1-ρ) + log Bayes factor), the ranking score.

    Types are Leiden clusters of an LSI embedding. For every cell the doublet
    partners are found by scanning all types for each of its
    ``n_candidates`` best singlet types (coarse α grid); the ``n_pairs``
    best pairs enter the marginal likelihood. A cluster that is itself a
    doublet population is removed from the singlet types: its cells' log-
    likelihood advantage of the own profile (evaluated leave-one-out) over the
    best other type is recovered for >= 95 % by the doublet model built from
    the other types, and that advantage is >= 0.1 nats per fragment (clearly
    distinct parents; :func:`_cluster_mixture_fraction`; ``prune_doublet_types``). The doublet
    rate :math:`\rho` starts from the Poisson-loading expectation
    (:func:`loading_doublet_rate`, or ``expected_doublet_rate``) and is
    re-estimated by EM, :math:`\rho \leftarrow \bar P(D \mid x_i, \rho)`, so the
    probabilities are calibrated to the data.

    Optional evidence (off by default: neither improved ranking on the
    development data, GSE162690 cell-line mixtures and 10x hgmm barnyard):

    * ``use_library`` -- library size as the (sub-additive) sum of two
      singlet libraries, :math:`N = \kappa(N_a + N_b)`,
      :math:`\alpha = N_a/(N_a+N_b)`, with log-normal type libraries, folded
      into the same α integral;
    * ``use_overlap`` -- AMULET's copy-number evidence (Thibodeau et al.
      2021): loci covered by >= 3 fragments of one barcode
      (:func:`count_overlap_loci`, re-reads the fragment file) modelled as
      Poisson with type- and depth-dependent singlet rates and a doublet rate;
    * ``n_iter > 0`` -- EM refinement of the type profiles with singlet
      weights (``loo`` evaluates them leave-one-out).

    References: DePasquale et al. 2019 (DoubletDecon); Germain et al. 2021
    (scDblFinder); Kang et al. 2018 (demuxlet); Thibodeau et al. 2021
    (AMULET); Bloom 2018 (Poisson loading); Dempster, Laird & Rubin 1977 (EM).

    Parameters
    ----------
    adata
        Cells × tiles (or peaks) counts (``adata.X`` or ``layer``), e.g. from
        :func:`epione.pp.add_tile_matrix`.
    n_features
        Most frequently accessible features used.
    n_comps, resolution, n_neighbors
        LSI dimensions, Leiden resolution and kNN size of the singlet types.
    n_candidates, n_pairs
        First parents per cell and kept doublet pairs per cell.
    pseudocount
        Dirichlet prior strength of the type profiles, in average cells.
    n_iter, loo
        EM iterations refining the type profiles (0 = off), leave-one-out.
    use_library, homotypic, use_overlap, fragments
        Optional evidence terms (see above); ``homotypic`` adds same-type
        pairs to the doublet hypothesis.
    expected_doublet_rate
        Initial doublet rate; default from Poisson loading.
    prior_strength
        Beta prior strength (pseudo-cells) keeping ρ near its initial value.
    prune_doublet_types
        Remove clusters that are doublet populations from the singlet types.
    key_added
        Writes ``obs[f"{key_added}_probability"]`` (posterior),
        ``obs[f"{key_added}_score"]`` (log posterior odds),
        ``obs[f"{key_added}_class"]``, ``obs[f"{key_added}_pair"]`` (most
        likely parent types), ``obs[f"{key_added}_type"]`` and
        ``uns[key_added]`` (ρ, settings).

    Returns
    -------
    ``None`` if ``inplace`` else a DataFrame with the same columns.
    """
    import pandas as pd
    import scipy.sparse as sp
    X = adata.layers[layer] if layer is not None else adata.X
    X = sp.csr_matrix(X)
    N = np.asarray(X.sum(1)).ravel().astype(np.float64)
    df = np.bincount(X.indices, minlength=X.shape[1])
    nf = min(int(n_features), int((df > 0).sum()))
    feat = np.sort(np.argsort(-df, kind="stable")[:nf])
    Xb = X[:, feat].tocsr().astype(np.float64)
    Xb.data[:] = 1.0
    Xb.eliminate_zeros()
    m = None
    if use_overlap:
        from ._doublets import _fragments
        frag = _fragments(adata, fragments)
        bcs = [str(b).split("#")[-1] for b in adata.obs_names]
        ov = count_overlap_loci(frag, bcs, n_jobs=n_jobs)
        m = np.bincount(ov["row"], minlength=adata.n_obs).astype(np.float64)
    rho0 = (float(expected_doublet_rate) if expected_doublet_rate is not None
            else max(loading_doublet_rate(adata.n_obs), 0.01))
    console.level1(f"[bayes_doublets] {adata.n_obs:,} cells, {Xb.shape[1]:,} features, rho0={rho0:.3f}")
    res = _fit_bayes(Xb, N, m=m, n_comps=n_comps, resolution=resolution, n_neighbors=n_neighbors,
                     n_candidates=n_candidates, n_pairs=n_pairs, t_max=t_max, n_grid=n_grid,
                     pseudocount=pseudocount, use_library=use_library, use_overlap=use_overlap,
                     rho0=rho0, prior_strength=prior_strength, n_iter=n_iter, homotypic=homotypic,
                     loo=loo, background=0.0, calibrate_rho=True,
                     prune_doublet_types=prune_doublet_types, seed=random_state, n_jobs=n_jobs,
                     verbose=True)
    bp = res["best_pair"]
    pair = np.where(bp >= 0, pd.Series(bp // 100000).astype(str) + "+" + pd.Series(bp % 100000).astype(str), "")
    out = pd.DataFrame({
        f"{key_added}_probability": res["prob"],
        f"{key_added}_score": res["logit"],
        f"{key_added}_class": pd.Categorical(np.where(res["prob"] > 0.5, "doublet", "singlet"),
                                             categories=["singlet", "doublet"]),
        f"{key_added}_pair": pair,
        f"{key_added}_type": res["gamma"].argmax(1).astype(str),
    }, index=adata.obs_names)
    if m is not None:
        out[f"{key_added}_overlap_loci"] = m
    console.level2(f"[bayes_doublets] rho={res['rho']:.3f}, kappa={res['kappa']:.3f}, "
                   f"called {int((res['prob'] > 0.5).sum()):,} doublets")
    if not inplace:
        return out
    for c in out.columns:
        adata.obs[c] = out[c].values
    adata.uns[key_added] = dict(rho=res["rho"], kappa=res["kappa"], n_types=int(res["gamma"].shape[1]),
                                n_features=int(Xb.shape[1]), use_library=use_library,
                                use_overlap=use_overlap, homotypic=homotypic, n_iter=n_iter)
