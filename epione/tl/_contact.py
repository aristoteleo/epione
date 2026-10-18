"""
Polymer-contact priors for enhancer–gene links and gene activity.

Principle
---------
The contact probability of two loci separated by ``s`` base pairs decays as
a power law,

.. math::
    P(s) \\propto s^{-\\gamma},

with :math:`\\gamma \\approx 1` for the fractal (crumpled) globule
(Lieberman-Aiden et al. 2009 Science 326:289; Mirny 2011 Chromosome Res
19:37) and loop-extrusion corrections between ~10 kb and ~1 Mb (Fudenberg et
al. 2016 Cell Rep 15:2038). The Activity-by-Contact (ABC) model (Fulco et al.
2019 Nat Genet 51:1664; Nasser et al. 2021 Nature 593:238) predicts the effect
of element ``e`` on gene ``g`` as activity × contact, normalised over all
elements near the gene:

.. math::
    \\mathrm{ABC}_{eg} = \\frac{a_e\\, C(s_{eg})}{\\sum_{e'} a_{e'}\\, C(s_{e'g})}.

This module provides

* :func:`powerlaw_contact` — the contact kernel
  :math:`C(s) = (\\max(s, s_0)/s_0)^{-\\gamma}`;
* :func:`fit_contact_gamma` — the exponent :math:`\\gamma` estimated from the
  data (decay of metacell co-accessibility with distance) or from a Hi-C
  ``.cool`` (decay of the balanced expected contact frequency);
* :func:`contact_links` — enhancer–gene scores that combine the contact
  prior, ABC normalisation and a **partial** correlation between peak
  accessibility and gene expression across metacells, estimated with the
  graphical lasso (Friedman, Hastie & Tibshirani 2008 Biostatistics 9:432)
  conditional on the other local peaks and on global cell-state factors;
* :func:`contact_gene_score` — a per-cell gene activity score
  :math:`\\sum_t x_{ct}\\, C(s_{tg})` that replaces the fixed exponential
  decay of ArchR's gene score with the power-law contact prior.

None of these functions changes the behaviour of
:func:`epione.tl.peak_to_gene` or :func:`epione.tl.add_gene_score_matrix`.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ..utils import console

#: ABC-Enhancer-Gene-Prediction reference power law (``hic_gamma`` /
#: ``hic_scale`` fitted to averaged Hi-C; Fulco et al. 2019).
ABC_GAMMA = 1.024238616787792
ABC_SCALE = 5.9594510043736655


# ---------------------------------------------------------------------------
# Contact kernel and exponent estimation
# ---------------------------------------------------------------------------

def powerlaw_contact(distance, gamma: float = ABC_GAMMA, s0: float = 5000.0) -> np.ndarray:
    """Power-law contact prior :math:`C(s) = (\\max(|s|, s_0)/s_0)^{-\\gamma}`.

    ``C(s) = 1`` for ``|s| <= s0`` (the short-range plateau where the
    polymer picture breaks down and ABC clips the distance), then decays as
    ``s^-gamma``. The ABC code uses ``exp(scale - gamma * log(s + 1))`` with
    ``s`` clipped at 5 kb; up to a constant factor (which cancels in every
    ranking and in the ABC ratio) this is the same function.

    Arguments:
        distance: genomic separations in bp (any shape; sign ignored).
        gamma: decay exponent (``ABC_GAMMA`` = 1.0242 by default).
        s0: short-range floor in bp.

    Returns:
        ``float64`` array of the same shape, values in ``(0, 1]``.
    """
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    d = np.maximum(np.abs(np.asarray(distance, dtype=np.float64)), float(s0))
    return (d / float(s0)) ** (-float(gamma))


def _metacell_membership(emb: np.ndarray, n_metacells: int, k_neighbors: int,
                         seed: int) -> sp.csr_matrix:
    """``(n_metacells × n_cells)`` 0/1 membership matrix of the
    peak_to_gene metacells (uniform anchors + cosine k-NN)."""
    from ._peak_to_gene import _build_metacell_indices
    idx = _build_metacell_indices(emb, n_metacells, k_neighbors, seed)
    n_mc, k = idx.shape
    M = sp.csr_matrix((np.ones(idx.size, dtype=np.float64),
                       (np.repeat(np.arange(n_mc), k), idx.ravel())),
                      shape=(n_mc, emb.shape[0]))
    M.sum_duplicates()
    M.data[:] = 1.0
    return M


def _log_cpm(S: np.ndarray, scale: float = 1e6) -> np.ndarray:
    """``log1p(scale * S / rowsum)`` for a dense ``(n_mc × features)``."""
    S = np.asarray(S, dtype=np.float64)
    tot = S.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(S * (scale / tot))


def _standardise(M: np.ndarray) -> np.ndarray:
    Z = M - M.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, ddof=1, keepdims=True)
    sd[~(sd > 0)] = np.inf
    return Z / sd


def _peak_table(var_names) -> pd.DataFrame:
    from ._peak_to_gene import _parse_peak_names
    df = _parse_peak_names(var_names).reset_index(drop=True)
    df["center"] = ((df["start"] + df["end"]) // 2).astype(np.int64)
    df["peak_idx"] = np.arange(len(df), dtype=np.int64)
    return df


def fit_contact_gamma(
    adata: Optional[AnnData] = None,
    *,
    hic: Optional[str] = None,
    use_rep: str = "X_iterative_lsi",
    layer: Optional[str] = None,
    n_metacells: int = 500,
    k_neighbors: int = 100,
    seed: int = 0,
    d_min: float = 5_000,
    d_max: float = 1_000_000,
    n_bins: int = 14,
    n_pairs: int = 200_000,
    clip: tuple = (0.25, 2.0),
    metacells: Optional[sp.csr_matrix] = None,
    verbose: bool = True,
) -> dict:
    """Estimate the contact-decay exponent :math:`\\gamma`.

    **From scATAC (default).** On metacells (same construction as
    :func:`epione.tl.peak_to_gene`) the Pearson correlation of log-CPM
    accessibility between two peaks is averaged in ``n_bins`` log-spaced
    distance bins between ``d_min`` and ``d_max`` (``n_pairs`` random
    same-chromosome pairs per bin), and the mean correlation of
    ``n_pairs`` random inter-chromosomal pairs (no contact) is subtracted.
    The excess co-accessibility :math:`\\bar r(s) - \\bar r_{trans}` is taken
    as proportional to contact frequency, and

    .. math::
        \\log(\\bar r(s) - \\bar r_{trans}) = c - \\gamma \\log s

    is fitted by ordinary least squares over the bins with positive excess.

    **From Hi-C** (``hic=`` path to a balanced ``.cool`` / ``.mcool::…``).
    The cis expected (``cooltools.expected_cis``, balanced mean per
    diagonal, pooled over chromosome arms weighted by valid pixels) is fitted
    in log-log space over ``[d_min, d_max]``. Needs ``cooler`` and
    ``cooltools``.

    Arguments:
        adata: cells × peaks AnnData with ``obsm[use_rep]`` (scATAC route).
        hic: path/URI of a balanced cool (Hi-C route; ``adata`` ignored).
        use_rep, n_metacells, k_neighbors, seed: metacell construction.
        layer: count layer (default ``adata.X``).
        d_min, d_max, n_bins: distance range and bin count of the fit.
        n_pairs: sampled pairs per bin (and for the trans background).
        clip: bounds applied to the fitted exponent.
        metacells: optional precomputed ``(n_mc × n_cells)`` membership.

    Returns:
        ``dict`` with ``gamma`` (clipped), ``gamma_raw``, ``intercept``,
        ``source`` and the per-bin ``curve`` DataFrame (``distance``,
        ``mean_r`` or ``expected``, ``excess``, ``n``).
    """
    edges = np.exp(np.linspace(np.log(d_min), np.log(d_max), n_bins + 1))
    if hic is not None:
        return _fit_gamma_hic(hic, edges, clip)
    if adata is None:
        raise ValueError("pass `adata` (scATAC route) or `hic` (Hi-C route)")
    rng = np.random.default_rng(seed)
    if metacells is None:
        if use_rep not in adata.obsm:
            raise KeyError(f"adata.obsm[{use_rep!r}] not found")
        metacells = _metacell_membership(np.asarray(adata.obsm[use_rep]),
                                         n_metacells, k_neighbors, seed)
    X = adata.layers[layer] if layer else adata.X
    S = metacells @ (X if sp.issparse(X) else np.asarray(X))
    S = S.toarray() if sp.issparse(S) else np.asarray(S)
    Z = _standardise(_log_cpm(S)).astype(np.float32)
    n_mc = Z.shape[0]
    pk = _peak_table(adata.var_names)
    # global coordinate with a huge gap between chromosomes
    codes, uniq = pd.factorize(pk["chrom"])
    key = codes.astype(np.int64) * np.int64(10**11) + pk["center"].to_numpy(np.int64)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    n = len(skey)

    def corr(i, j):
        out = np.empty(len(i), dtype=np.float64)
        step = 200_000
        for a in range(0, len(i), step):
            b = min(a + step, len(i))
            out[a:b] = (Z[:, i[a:b]] * Z[:, j[a:b]]).sum(axis=0) / (n_mc - 1)
        return out

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = 3 * n_pairs
        a = rng.integers(0, n, size=m)
        d = np.exp(rng.uniform(np.log(lo), np.log(hi), size=m))
        sgn = rng.choice([-1, 1], size=m)
        tgt = skey[a] + (sgn * d).astype(np.int64)
        b = np.clip(np.searchsorted(skey, tgt), 0, n - 1)
        b2 = np.clip(b - 1, 0, n - 1)
        b = np.where(np.abs(skey[b2] - tgt) < np.abs(skey[b] - tgt), b2, b)
        dist = np.abs(skey[b] - skey[a])
        ok = (b != a) & (dist >= lo) & (dist < hi)
        a, b, dist = a[ok][:n_pairs], b[ok][:n_pairs], dist[ok][:n_pairs]
        if len(a) == 0:
            rows.append((np.sqrt(lo * hi), np.nan, 0))
            continue
        r = corr(order[a], order[b])
        r = r[np.isfinite(r)]
        rows.append((float(np.exp(np.mean(np.log(dist)))), float(np.mean(r)) if len(r) else np.nan, len(r)))
    # trans background
    a = rng.integers(0, n, size=3 * n_pairs)
    b = rng.integers(0, n, size=3 * n_pairs)
    ok = codes[order[a]] != codes[order[b]]
    a, b = a[ok][:n_pairs], b[ok][:n_pairs]
    rb = corr(order[a], order[b])
    bg = float(np.mean(rb[np.isfinite(rb)])) if len(rb) else 0.0
    curve = pd.DataFrame(rows, columns=["distance", "mean_r", "n"])
    curve["excess"] = curve["mean_r"] - bg
    res = _loglog_fit(curve["distance"].to_numpy(), curve["excess"].to_numpy(), clip)
    res.update(source="coaccessibility", background_r=bg, curve=curve)
    if verbose:
        console.info(f"[fit_contact_gamma] gamma = {res['gamma']:.3f} "
                     f"(raw {res['gamma_raw']:.3f}; trans r = {bg:.4f}; "
                     f"{res['n_bins_used']} bins)")
    return res


def _loglog_fit(x, y, clip) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        raise ValueError("fewer than 3 distance bins with positive signal; cannot fit gamma")
    lx, ly = np.log(x[ok]), np.log(y[ok])
    slope, intercept = np.polyfit(lx, ly, 1)
    g = -float(slope)
    return dict(gamma=float(np.clip(g, *clip)), gamma_raw=g, intercept=float(intercept),
                n_bins_used=int(ok.sum()))


def _fit_gamma_hic(hic, edges, clip) -> dict:
    try:
        import cooler
        from cooltools.api.expected import expected_cis
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("fitting gamma from Hi-C needs `cooler` and `cooltools` "
                          "(pip install epione[hic])") from e
    clr = cooler.Cooler(str(hic))
    exp = expected_cis(clr, ignore_diags=0, chunksize=1_000_000)
    col = "balanced.avg" if "balanced.avg" in exp.columns else "count.avg"
    binsize = int(clr.binsize)
    exp = exp[np.isfinite(exp[col]) & (exp["n_valid"] > 0)].copy()
    exp["s"] = exp["dist"].astype(np.int64) * binsize
    exp["w"] = exp[col] * exp["n_valid"]
    g = exp.groupby("s").agg(w=("w", "sum"), n=("n_valid", "sum"))
    s = g.index.to_numpy(float)
    e = (g["w"] / g["n"]).to_numpy(float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        if m.any():
            rows.append((float(np.exp(np.mean(np.log(s[m])))), float(np.mean(e[m])), int(m.sum())))
    curve = pd.DataFrame(rows, columns=["distance", "expected", "n"])
    curve["excess"] = curve["expected"]
    res = _loglog_fit(curve["distance"], curve["excess"], clip)
    res.update(source="hic", curve=curve)
    return res


# ---------------------------------------------------------------------------
# Partial correlation via the graphical lasso
# ---------------------------------------------------------------------------

def _glasso_partial_corr(Y: np.ndarray, alpha: float, max_iter: int = 200,
                         tol: float = 1e-4) -> np.ndarray:
    """Partial correlations of column 0 of ``Y`` with every other column.

    ``Y`` (n × m) is standardised; the empirical correlation matrix ``R`` is
    passed to the graphical lasso,

    .. math::
        \\hat\\Theta = \\arg\\max_{\\Theta \\succ 0}\\;
            \\log\\det\\Theta - \\operatorname{tr}(R\\Theta) - \\alpha \\|\\Theta\\|_{1,\\text{off}},

    and the partial correlation is
    :math:`\\rho_{0j} = -\\Theta_{0j}/\\sqrt{\\Theta_{00}\\Theta_{jj}}`.
    If the solver fails (ill-conditioned ``R``), ``R`` is shrunk towards the
    identity (``0.9 R + 0.1 I``, then ``0.5 R + 0.5 I``) and the fit retried;
    a ridge inverse is the last resort.
    """
    from sklearn.covariance import graphical_lasso
    n, m = Y.shape
    R = (Y.T @ Y) / (n - 1)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    Theta = None
    for lam in (0.0, 0.1, 0.5):
        Rs = (1 - lam) * R + lam * np.eye(m)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, Theta = graphical_lasso(Rs, alpha=alpha, max_iter=max_iter, tol=tol,
                                           mode="cd")
            if np.all(np.isfinite(Theta)):
                break
            Theta = None
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            Theta = None
    if Theta is None:
        Theta = np.linalg.inv(R + max(alpha, 1e-3) * np.eye(m))
    d = np.sqrt(np.clip(np.diag(Theta), 1e-12, None))
    return -Theta[0, 1:] / (d[0] * d[1:])


_GL_KERNELS: dict = {}


def _glasso_kernels():
    """nogil numba graphical lasso (Friedman, Hastie & Tibshirani 2008, block
    coordinate descent), the same objective and update order as
    ``sklearn.covariance.graphical_lasso(mode="cd")``: the diagonal is not
    penalised, the start is ``0.95·R`` with the diagonal of ``R``, each row is
    a lasso on the Gram matrix ``W_11`` solved by cyclic coordinate descent,
    and the outer loop stops when the duality gap is below ``tol``."""
    if _GL_KERNELS:
        return _GL_KERNELS
    import numba
    jit = numba.njit(cache=True, nogil=True, fastmath={"reassoc", "contract", "arcp", "nsz"})  # no nnan/ninf: finiteness checks stay valid

    @jit
    def _inv_spd(A):
        """Inverse of a symmetric positive-definite matrix (Gauss-Jordan, no LAPACK call)."""
        m = A.shape[0]
        W = A.copy()
        Inv = np.eye(m)
        for c in range(m):
            piv = W[c, c]
            for k in range(m):
                W[c, k] /= piv
                Inv[c, k] /= piv
            for r in range(m):
                if r != c:
                    f = W[r, c]
                    if f != 0.0:
                        for k in range(m):
                            W[r, k] -= f * W[c, k]
                            Inv[r, k] -= f * Inv[c, k]
        return Inv

    @jit
    def glasso(S, alpha, max_iter, tol, enet_max_iter, enet_tol, tol_c=1e-4, shrink=0.95):
        m = S.shape[0]
        C = S * shrink
        for i in range(m):
            C[i, i] = S[i, i]
        P = _inv_spd(C)
        w = np.zeros(m)
        grad = np.zeros(m)
        act = np.empty(m, dtype=np.int64)
        it = 0
        for it in range(max_iter):
            dc_max = 0.0
            for j in range(m):
                # row j: lasso min 0.5 w'C_{-j,-j} w - S_{-j,j}'w + alpha |w|_1 in full index
                # space with w[j] = 0 (no sub-matrix copies); warm start from P[:, j]
                den = P[j, j] + 1000.0 * 2.220446049250313e-16
                for a in range(m):
                    w[a] = -P[a, j] / den
                w[j] = 0.0
                for a in range(m):
                    grad[a] = S[j, a]
                for bb in range(m):
                    wb = w[bb]
                    if wb != 0.0:
                        for a in range(m):
                            grad[a] -= C[bb, a] * wb     # C symmetric: row access
                full = True
                n_act = 0
                for _ in range(enet_max_iter):
                    w_max = 0.0
                    d_w_max = 0.0
                    n_loop = m if full else n_act
                    for t in range(n_loop):
                        aa = t if full else act[t]
                        if aa == j or C[aa, aa] == 0.0:
                            continue
                        w_old = w[aa]
                        tmp = grad[aa] + C[aa, aa] * w_old
                        if tmp > alpha:
                            wn = (tmp - alpha) / C[aa, aa]
                        elif tmp < -alpha:
                            wn = (tmp + alpha) / C[aa, aa]
                        else:
                            wn = 0.0
                        if wn != w_old:
                            dlt = wn - w_old
                            w[aa] = wn
                            for bb in range(m):
                                grad[bb] -= C[aa, bb] * dlt
                            if abs(dlt) > d_w_max:
                                d_w_max = abs(dlt)
                        if abs(wn) > w_max:
                            w_max = abs(wn)
                    if full:
                        n_act = 0
                        for aa in range(m):
                            if w[aa] != 0.0:
                                act[n_act] = aa
                                n_act += 1
                    if w_max == 0.0 or d_w_max / w_max < enet_tol:
                        if full:
                            break
                        full = True
                    else:
                        full = False
                dot = 0.0
                for aa in range(m):
                    dot += C[aa, j] * w[aa]
                pjj = 1.0 / (C[j, j] - dot)
                for aa in range(m):
                    if aa != j:
                        P[aa, j] = -pjj * w[aa]
                        P[j, aa] = -pjj * w[aa]
                P[j, j] = pjj
                # new covariance row: C_{-j,j} = C_{-j,-j} w  (= S_{-j,j} - grad, the lasso residual)
                for aa in range(m):
                    if aa != j:
                        nv = S[j, aa] - grad[aa]
                        d = abs(nv - C[aa, j])
                        if d > dc_max:
                            dc_max = d
                        C[aa, j] = nv
                        C[j, aa] = nv
            gap = -float(m)
            l1 = 0.0
            for aa in range(m):
                for bb in range(m):
                    gap += S[aa, bb] * P[aa, bb]
                    if aa != bb:
                        l1 += abs(P[aa, bb])
            gap += alpha * l1
            if abs(gap) < tol or dc_max < tol_c:
                break
        return P, it + 1

    @jit
    def pcor_block(g0, g1, Z, gene_ptr, cols, out_ptr, n_conf, alpha, max_iter, tol, out, iters):
        n = Z.shape[1]
        for g in range(g0, g1):
            c0, c1 = gene_ptr[g], gene_ptr[g + 1]
            m = c1 - c0
            if m < 2:
                continue
            # S = X X' / (n - 1) with plain loops: no BLAS call inside the threaded kernel
            # (a multi-threaded BLAS here oversubscribes the cores ~25x)
            S = np.empty((m, m))
            for a in range(m):
                ra = Z[cols[c0 + a]]
                for b in range(a, m):
                    rb = Z[cols[c0 + b]]
                    acc = 0.0
                    for r in range(n):
                        acc += ra[r] * rb[r]
                    S[a, b] = acc / (n - 1)
                    S[b, a] = S[a, b]
            for a in range(m):
                S[a, a] = 1.0
            # a gene uncorrelated with every other variable is its own glasso component
            # (Witten, Friedman & Simon 2011): rho = 0 exactly, no fit needed
            iso = True
            for a in range(1, m):
                if abs(S[0, a]) > alpha:
                    iso = False
                    break
            if iso:
                continue
            P, nit = glasso(S, alpha, max_iter, tol, 100, 1e-4)
            iters[g] = nit
            ok = True
            for a in range(m):
                for b in range(m):
                    if not np.isfinite(P[a, b]):
                        ok = False
            if not ok:
                for a in range(out_ptr[g], out_ptr[g + 1]):
                    out[a] = np.nan
                continue
            d0 = np.sqrt(max(P[0, 0], 1e-12))
            n_pk = out_ptr[g + 1] - out_ptr[g]
            for a in range(n_pk):
                out[out_ptr[g] + a] = -P[0, 1 + a] / (d0 * np.sqrt(max(P[1 + a, 1 + a], 1e-12)))

    @jit
    def mc_sums(r0, r1, mptr, midx, indptr, indices, data, out):
        for i in range(r0, r1):
            for t in range(mptr[i], mptr[i + 1]):
                c = midx[t]
                for q in range(indptr[c], indptr[c + 1]):
                    out[i, indices[q]] += data[q]

    @jit
    def log_std_cols(c0, c1, S, rowscale, out):
        """out[j, :] = standardise(log1p(S[:, j] * rowscale)) (ddof 1; zero variance -> 0);
        output is variables x observations."""
        n = S.shape[0]
        for j in range(c0, c1):
            mu = 0.0
            for i in range(n):
                v = np.log1p(S[i, j] * rowscale[i])
                out[j, i] = v
                mu += v
            mu /= n
            ss = 0.0
            for i in range(n):
                d = out[j, i] - mu
                ss += d * d
            sd = np.sqrt(ss / (n - 1))
            if sd > 0:
                for i in range(n):
                    out[j, i] = (out[j, i] - mu) / sd
            else:
                for i in range(n):
                    out[j, i] = 0.0

    @jit
    def pair_dot(a0, a1, rows_a, rows_b, Z, out):
        n = Z.shape[1]
        for q in range(a0, a1):
            ra = Z[rows_a[q]]
            rb = Z[rows_b[q]]
            acc = 0.0
            for r in range(n):
                acc += ra[r] * rb[r]
            out[q] = acc / (n - 1)

    @jit
    def colsum_rows(b0, b1, rows, indptr, indices, data, out2d, blk):
        acc = out2d[blk]
        for t in range(b0, b1):
            r = rows[t]
            for q in range(indptr[r], indptr[r + 1]):
                acc[indices[q]] += data[q]

    @jit
    def abc_den(c, a, tss, window, gamma, s0, out):
        n = len(c)
        for i in range(len(tss)):
            t = tss[i]
            lo = np.searchsorted(c, t - window, side="left")
            hi = np.searchsorted(c, t + window, side="right")
            acc = 0.0
            for k in range(lo, hi):
                d = abs(c[k] - t)
                if d < s0:
                    d = s0
                acc += a[k] * (d / s0) ** (-gamma)
            out[i] = acc


    _GL_KERNELS.update(glasso=glasso, pcor_block=pcor_block, abc_den=abc_den, mc_sums=mc_sums,
                       log_std_cols=log_std_cols, pair_dot=pair_dot,
                       colsum_rows=colsum_rows)
    return _GL_KERNELS


def _threaded(fn, n, n_jobs, *args):
    """Run ``fn(b0, b1, *args)`` over ``range(n)`` in contiguous blocks on threads."""
    from concurrent.futures import ThreadPoolExecutor
    from ..pp._fragment_io import resolve_n_jobs
    nj = resolve_n_jobs(n_jobs)
    edges = np.linspace(0, n, min(n, 4 * nj) + 1).astype(np.int64)
    blocks = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]
    if nj > 1 and len(blocks) > 1:
        with ThreadPoolExecutor(max_workers=nj) as ex:
            list(ex.map(lambda b: fn(b[0], b[1], *args), blocks))
    else:
        for b in blocks:
            fn(b[0], b[1], *args)


def _colsum_rows(X, rows, n_jobs=None):
    """Column sums of the CSR rows ``rows`` (threaded numba, private per-block accumulators)."""
    from concurrent.futures import ThreadPoolExecutor
    from ..pp._fragment_io import resolve_n_jobs
    k = _glasso_kernels()
    X = X.tocsr()
    rows = np.asarray(rows, dtype=np.int64)
    nj = resolve_n_jobs(n_jobs)
    edges = np.linspace(0, len(rows), nj + 1).astype(np.int64)
    acc = np.zeros((nj, X.shape[1]), dtype=np.float64)
    ip = X.indptr.astype(np.int64)

    def run(b):
        k["colsum_rows"](edges[b], edges[b + 1], rows, ip, X.indices, X.data, acc, b)
    with ThreadPoolExecutor(max_workers=nj) as ex:
        list(ex.map(run, range(nj)))
    return acc.sum(axis=0)


def _metacell_log_std(X, M, scale, out=None, n_jobs=None, totals=None):
    """Standardised ``log1p(scale * S / rowsum(S))`` of the metacell sums ``S = M @ X``
    (``M`` the 0/1 membership CSR), returned TRANSPOSED (features × n_mc, float64), threaded numba;
    written into ``out`` when given."""
    k = _glasso_kernels()
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    M = M.tocsr()
    S = np.zeros((M.shape[0], X.shape[1]), dtype=np.float64)
    _threaded(k["mc_sums"], M.shape[0], n_jobs, M.indptr.astype(np.int64), M.indices.astype(np.int64),
              X.indptr.astype(np.int64), X.indices, X.data, S)
    tot = S.sum(axis=1) if totals is None else np.asarray(totals, dtype=np.float64)
    rowscale = np.where(tot > 0, scale / np.where(tot > 0, tot, 1.0), 0.0)
    if out is None:
        out = np.empty((S.shape[1], S.shape[0]), dtype=np.float64)
    _threaded(k["log_std_cols"], S.shape[1], n_jobs, S, rowscale, out)
    return out


def _pcor_rows(ZT, gene_rows, peak_lists, conf_rows, alpha, n_jobs=None, max_iter=200, tol=1e-4):
    """Partial correlations for many genes (numba, threads over genes) from one matrix ``ZT``
    (variables × observations, standardised rows): gene ``i`` = row ``gene_rows[i]``, its peaks =
    rows ``peak_lists[i]``, confounders = rows ``conf_rows`` (shared). Returns concatenated ρ."""
    from concurrent.futures import ThreadPoolExecutor
    from ..pp._fragment_io import resolve_n_jobs
    k = _glasso_kernels()
    Kc = len(conf_rows)
    sizes = np.array([1 + len(pl) + Kc for pl in peak_lists], dtype=np.int64)
    gene_ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    cols = np.empty(gene_ptr[-1], dtype=np.int64)
    out_ptr = np.r_[0, np.cumsum([len(pl) for pl in peak_lists])].astype(np.int64)
    conf_rows = np.asarray(conf_rows, dtype=np.int64)
    for i, (gr, pl) in enumerate(zip(gene_rows, peak_lists)):
        a = gene_ptr[i]
        cols[a] = gr
        cols[a + 1:a + 1 + len(pl)] = pl
        cols[a + 1 + len(pl):gene_ptr[i + 1]] = conf_rows
    out = np.zeros(out_ptr[-1])
    n_genes = len(peak_lists)
    iters = np.zeros(n_genes, dtype=np.int64)
    nj = resolve_n_jobs(n_jobs)
    cost = np.cumsum(sizes.astype(np.float64) ** 3)
    n_blk = max(1, min(n_genes, 8 * nj))
    edges = np.searchsorted(cost, np.linspace(0, cost[-1], n_blk + 1)[1:-1]) if n_genes else []
    bounds = np.unique(np.r_[0, edges, n_genes]).astype(np.int64)

    def run(b):
        k["pcor_block"](bounds[b], bounds[b + 1], ZT, gene_ptr, cols, out_ptr, Kc, float(alpha),
                        int(max_iter), float(tol), out, iters)
    if nj > 1 and len(bounds) > 2:
        with ThreadPoolExecutor(max_workers=nj) as ex:
            list(ex.map(run, range(len(bounds) - 1)))
    else:
        for b in range(len(bounds) - 1):
            run(b)
    for i in range(n_genes):
        if not np.isfinite(out[out_ptr[i]:out_ptr[i + 1]]).all():
            F = ZT[conf_rows].T if Kc else None
            out[out_ptr[i]:out_ptr[i + 1]] = _pcor_one_gene(
                ZT[gene_rows[i]], ZT[np.asarray(peak_lists[i], dtype=np.int64)].T, F, alpha)
    _pcor_rows.last_iters = iters
    return out


def _pcor_all_genes(Zg, Zp, F, gene_cols, peak_lists, alpha, n_jobs=None, max_iter=200, tol=1e-4):
    """Column-major convenience wrapper of :func:`_pcor_rows` (``Zg`` n × genes, ``Zp`` n × peaks,
    ``F`` n × K confounders)."""
    n = Zg.shape[0]
    Kc = 0 if F is None else F.shape[1]
    ZT = np.vstack([np.nan_to_num(Zg).T, np.nan_to_num(Zp).T] +
                   ([np.nan_to_num(_standardise(F)).T] if Kc else [])).astype(np.float64)
    off = Zg.shape[1]
    conf = np.arange(off + Zp.shape[1], off + Zp.shape[1] + Kc, dtype=np.int64)
    out = _pcor_rows(np.ascontiguousarray(ZT), list(gene_cols),
                     [off + np.asarray(p, dtype=np.int64) for p in peak_lists], conf, alpha,
                     n_jobs=n_jobs, max_iter=max_iter, tol=tol)
    _pcor_all_genes.last_iters = _pcor_rows.last_iters
    return out


def _pcor_one_gene(y, P, F, alpha):
    """Return partial corr of y with each column of P given P and F."""
    keep = np.isfinite(P).all(axis=0) & (P.std(axis=0) > 0)
    out = np.zeros(P.shape[1], dtype=np.float64)
    if not np.isfinite(y).all() or y.std() == 0 or keep.sum() == 0:
        return out
    cols = [y[:, None], P[:, keep]]
    if F is not None and F.shape[1]:
        cols.append(F)
    Y = _standardise(np.hstack(cols))
    Y[~np.isfinite(Y)] = 0.0
    rho = _glasso_partial_corr(Y, alpha)
    out[keep] = rho[: int(keep.sum())]
    return out


# ---------------------------------------------------------------------------
# Gene annotation helper
# ---------------------------------------------------------------------------

def _genes_table(gene_annotation) -> pd.DataFrame:
    from ._peak_to_gene import _resolve_gene_annotation
    g = _resolve_gene_annotation(gene_annotation)
    return g.reset_index(drop=True)


def _pairs_within(peaks: pd.DataFrame, genes: pd.DataFrame, max_distance: int):
    """All (peak, gene) pairs with |peak centre − TSS| <= max_distance.

    Returns int arrays ``(peak_idx, gene_row, distance)`` where ``gene_row``
    indexes ``genes`` and ``distance = centre − TSS`` (signed)."""
    pi, gi, di = [], [], []
    pk_by = {c: d for c, d in peaks.groupby("chrom", sort=False)}
    for chrom, g in genes.groupby("chrom", sort=False):
        p = pk_by.get(chrom)
        if p is None:
            continue
        p = p.sort_values("center")
        c = p["center"].to_numpy(np.int64)
        idx = p["peak_idx"].to_numpy(np.int64)
        tss = g["tss"].to_numpy(np.int64)
        lo = np.searchsorted(c, tss - max_distance, side="left")
        hi = np.searchsorted(c, tss + max_distance, side="right")
        n = hi - lo
        if n.sum() == 0:
            continue
        rows = np.repeat(g.index.to_numpy(np.int64), n)
        off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        cols = np.repeat(lo, n) + off
        pi.append(idx[cols])
        gi.append(rows)
        di.append(c[cols] - np.repeat(tss, n))
    if not pi:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    return np.concatenate(pi), np.concatenate(gi), np.concatenate(di)


# ---------------------------------------------------------------------------
# Enhancer–gene links
# ---------------------------------------------------------------------------

def _abc_denominators(peaks: pd.DataFrame, genes: pd.DataFrame, act: np.ndarray, gamma: float,
                      s0: float, window: int) -> np.ndarray:
    """``sum_{p: |c_p - TSS_g| <= window} a_p C(|c_p - TSS_g|)`` for every gene (numba,
    no pair materialisation)."""
    _den = _glasso_kernels()["abc_den"]
    out = np.zeros(len(genes))
    pk_by = {ch: d for ch, d in peaks.groupby("chrom", sort=False)}
    for ch, g in genes.groupby("chrom", sort=False):
        p = pk_by.get(ch)
        if p is None:
            continue
        p = p.sort_values("center")
        res = np.zeros(len(g))
        _den(p["center"].to_numpy(np.float64), act[p["peak_idx"].to_numpy()].astype(np.float64),
             g["tss"].to_numpy(np.float64), float(window), float(gamma), float(s0), res)
        out[g.index.to_numpy()] = res
    return out


_LINK_METHODS = ("abc_pcor", "abc", "pcor_contact", "pcor", "contact", "pearson_contact")


def contact_links(
    adata: AnnData,
    *,
    rna: Optional[AnnData] = None,
    gene_annotation=None,
    method: str = "abc_pcor",
    groupby: Optional[str] = None,
    target: Optional[Union[str, Sequence[str]]] = None,
    genes: Optional[Sequence[str]] = None,
    gamma: Union[float, str, dict] = ABC_GAMMA,
    s0: float = 5_000,
    max_distance: int = 1_000_000,
    abc_window: int = 5_000_000,
    corr_window: int = 250_000,
    n_confounders: int = 0,
    glasso_alpha: float = 0.05,
    eps: float = 0.05,
    use_rep: str = "X_iterative_lsi",
    n_metacells: int = 500,
    k_neighbors: int = 100,
    seed: int = 0,
    layer: Optional[str] = None,
    rna_layer: Optional[str] = None,
    n_jobs: Optional[int] = None,
    solver: Literal["numba", "sklearn"] = "numba",
    confounders: Literal["glasso", "residualize"] = "glasso",
    key_added: str = "contact_links",
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Enhancer–gene links from a polymer-contact prior (ABC) and partial
    correlation.

    For every peak ``p`` and gene ``g`` with ``|centre_p − TSS_g| ≤
    max_distance``:

    * **contact** :math:`C_{pg} = (\\max(s_{pg}, s_0)/s_0)^{-\\gamma}`
      (:func:`powerlaw_contact`), ``s`` = peak centre to TSS;
    * **activity** :math:`a_p` = pseudobulk accessibility (CPM) of the
      cells in ``target`` (all cells when ``target`` is ``None``);
    * **ABC** :math:`a_p C_{pg} / \\sum_{p': s_{p'g} \\le W} a_{p'} C_{p'g}`
      with ``W = abc_window`` (Fulco et al. 2019; Nasser et al. 2021);
    * **pcor** :math:`\\rho_{pg}` — partial correlation across metacells
      between the log-normalised expression of ``g`` and the log-CPM
      accessibility of ``p``, conditional on the other peaks within
      ``corr_window`` of the TSS and on the top ``n_confounders`` principal
      components of the metacell-mean embedding (global cell-state axes that
      otherwise make every cell-type-specific peak correlate with every
      cell-type-specific gene). The precision matrix is the graphical-lasso
      estimate on the correlation scale (penalty ``glasso_alpha``; Friedman,
      Hastie & Tibshirani 2008); :math:`\\rho_{pg} = -\\Theta_{gp}/
      \\sqrt{\\Theta_{gg}\\Theta_{pp}}`. Pairs beyond ``corr_window`` get
      :math:`\\rho = 0`.

    ``score`` by ``method``:

    ``"abc_pcor"``  :math:`\\mathrm{ABC}_{pg}\\,(\\varepsilon + \\max(\\rho_{pg}, 0))`
    ``"abc"``       :math:`\\mathrm{ABC}_{pg}`
    ``"pcor_contact"`` :math:`\\max(\\rho_{pg}, 0)\\, C_{pg}`
    ``"pcor"``      :math:`\\rho_{pg}`
    ``"contact"``   :math:`C_{pg}`
    ``"pearson_contact"`` :math:`\\max(r_{pg}, 0)\\, C_{pg}` (plain metacell
    Pearson ``r``; the no-glasso ablation)

    Parameters
    ----------
    adata
        Cells × peaks counts; ``var_names`` like ``chr:start-end``;
        ``obsm[use_rep]`` (correlation methods and ``gamma="fit"``).
    rna
        Cells × genes counts with the same ``obs_names`` (correlation
        methods).
    gene_annotation
        DataFrame ``gene_name, chrom, start, end[, strand]`` or a GTF path.
    groupby, target
        ``adata.obs[groupby]`` value(s) of the cell group whose accessibility
        defines the activity (ABC is cell-type specific).
    genes
        Restrict the output to these gene names (each gene is computed
        independently, so this is exact).
    gamma
        Contact exponent: a number, ``"fit"`` (:func:`fit_contact_gamma` on
        ``adata``, same metacells) or a result dict of
        :func:`fit_contact_gamma`.
    s0, max_distance, abc_window, corr_window
        Contact floor, output window, ABC denominator window and correlation
        window (bp).
    n_confounders, glasso_alpha, eps
        Global factors conditioned on, graphical-lasso penalty, and the
        floor added to the correlation evidence in ``"abc_pcor"``.
    use_rep, n_metacells, k_neighbors, seed
        Metacells as in :func:`epione.tl.peak_to_gene`.
    n_jobs
        Threads for the per-gene graphical-lasso fits (``None`` = the CPU
        budget, as Epione's fragment tools).
    confounders
        ``"glasso"``: the confounder factors enter the graphical lasso as
        (penalised) variables; ``"residualize"``: they are projected out of
        every variable first (exact, unpenalised conditioning), and the
        graphical lasso runs on the gene and its local peaks only (smaller
        problems, faster).
    solver
        ``"numba"`` (default): Epione's nogil numba graphical lasso, threads
        over genes; ``"sklearn"``: ``sklearn.covariance.graphical_lasso``
        per gene (reference implementation, same objective; slower).

    Returns
    -------
    DataFrame with ``peak, gene, chrom, peak_start, peak_end, tss, distance,
    contact, activity, abc, pcor, pearson, score``; stored in
    ``adata.uns[key_added]`` when ``inplace``.
    """
    if method not in _LINK_METHODS:
        raise ValueError(f"method must be one of {_LINK_METHODS}")
    if gene_annotation is None:
        raise ValueError("`gene_annotation` is required")
    needs_corr = method in ("abc_pcor", "pcor_contact", "pcor", "pearson_contact")
    needs_abc = method in ("abc_pcor", "abc")

    peaks = _peak_table(adata.var_names)
    gdf = _genes_table(gene_annotation)
    if rna is not None:
        rna_names = pd.Index(rna.var_names.astype(str))
        gdf = gdf[gdf["gene_name"].isin(rna_names)]
    if genes is not None:
        gdf = gdf[gdf["gene_name"].isin(set(genes))]
    gdf = gdf.reset_index(drop=True)
    X = adata.layers[layer] if layer else adata.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()

    import time as _time
    _t0 = _time.time()
    _tick = (lambda msg: console.info(f"[contact_links] {msg} ({_time.time() - _t0:.1f}s)")) if verbose else (lambda msg: None)
    # --- metacells (shared by gamma fit and correlations) ---
    M = None
    if needs_corr or (isinstance(gamma, str) and gamma == "fit"):
        if use_rep not in adata.obsm:
            raise KeyError(f"adata.obsm[{use_rep!r}] not found – run iterative_lsi first")
        emb = np.asarray(adata.obsm[use_rep])
        M = _metacell_membership(emb, n_metacells, k_neighbors, seed)

    _tick("metacells")
    # --- gamma ---
    gamma_info = None
    if isinstance(gamma, str):
        if gamma != "fit":
            raise ValueError("gamma must be a number, 'fit' or a fit_contact_gamma() dict")
        gamma_info = fit_contact_gamma(adata, layer=layer, metacells=M, seed=seed,
                                       verbose=verbose)
        g_val = gamma_info["gamma"]
    elif isinstance(gamma, dict):
        gamma_info = gamma
        g_val = float(gamma["gamma"])
    else:
        g_val = float(gamma)

    # --- activity ---
    if groupby is not None and target is not None:
        tg = [target] if isinstance(target, str) else list(target)
        mask = adata.obs[groupby].astype(str).isin([str(t) for t in tg]).to_numpy()
        if mask.sum() == 0:
            raise ValueError(f"no cells with {groupby} in {tg}")
    else:
        mask = np.ones(adata.n_obs, dtype=bool)
    pb = np.asarray(X[mask].sum(axis=0)).ravel().astype(np.float64)
    act = pb * (1e6 / max(pb.sum(), 1.0))

    _tick("activity")
    # --- pairs within the output window ---
    p_idx, g_row, dist = _pairs_within(peaks, gdf, max_distance)
    contact = powerlaw_contact(dist, g_val, s0)
    if verbose:
        console.info(f"[contact_links] {len(peaks):,} peaks | {len(gdf):,} genes | "
                     f"{len(p_idx):,} pairs ≤ {max_distance:,} bp | gamma = {g_val:.3f}")

    _tick("pairs")
    # --- ABC denominators over abc_window ---
    abc = np.zeros(len(p_idx))
    if needs_abc or method == "abc_pcor":
        denom = _abc_denominators(peaks, gdf, act, g_val, s0, abc_window)
        den = denom[g_row]
        abc = np.where(den > 0, act[p_idx] * contact / np.where(den > 0, den, 1.0), 0.0)

    _tick("ABC")
    # --- correlations ---
    pcor = np.zeros(len(p_idx))
    pear = np.zeros(len(p_idx))
    if needs_corr:
        if rna is None:
            raise ValueError("correlation methods need `rna`")
        if list(rna.obs_names) != list(adata.obs_names):
            raise ValueError("rna.obs_names must equal adata.obs_names (same order)")
        R = rna.layers[rna_layer] if rna_layer else rna.X
        R = sp.csr_matrix(R) if not sp.issparse(R) else R.tocsr()
        gcol = pd.Index(rna.var_names.astype(str)).get_indexer(gdf["gene_name"])
        n_mc = M.shape[0]
        n_g, n_p = len(gdf), X.shape[1]
        F = None
        if n_confounders > 0:
            E = (M @ emb) / np.asarray(M.sum(axis=1))
            E = np.asarray(E) - np.asarray(E).mean(axis=0)
            U, s, _ = np.linalg.svd(E, full_matrices=False)
            k = min(n_confounders, U.shape[1])
            F = U[:, :k] * s[:k]
        Kc = 0 if F is None else F.shape[1]
        # one matrix, variables x metacells: [genes | peaks | confounders], standardised rows
        ZT = np.empty((n_g + n_p + Kc, n_mc), dtype=np.float64)
        ZT[:n_g] = _metacell_log_std(R, M, 1e4, n_jobs=n_jobs)[gcol]       # log-CP10k genes
        _metacell_log_std(X, M, 1e6, out=ZT[n_g:n_g + n_p], n_jobs=n_jobs)  # log-CPM peaks
        if Kc:
            ZT[n_g + n_p:] = np.nan_to_num(_standardise(F)).T
        conf_rows = np.arange(n_g + n_p, n_g + n_p + Kc, dtype=np.int64)
        _tick("aggregation + normalisation")
        within = np.abs(dist) <= corr_window
        wi = np.flatnonzero(within)
        kk = _glasso_kernels()
        pv = np.zeros(len(wi))
        _threaded(kk["pair_dot"], len(wi), n_jobs, g_row[wi].astype(np.int64),
                  (n_g + p_idx[wi]).astype(np.int64), ZT, pv)
        pear[wi] = np.nan_to_num(pv)
        _tick("pearson")
        if method != "pearson_contact":
            order = np.argsort(g_row[wi], kind="stable")
            wi_sorted = wi[order]
            gr = g_row[wi_sorted]
            brk = np.flatnonzero(np.r_[True, gr[1:] != gr[:-1], True])
            slices = [wi_sorted[a:b] for a, b in zip(brk[:-1], brk[1:])]
            if verbose:
                console.info(f"[contact_links] graphical lasso for {len(slices):,} genes "
                             f"(alpha = {glasso_alpha}, K = {Kc})")
            if Kc and confounders == "residualize":
                # condition on the confounders exactly (unpenalised): project them out of every
                # variable, then the graphical lasso runs on [gene, local peaks] only
                Q, _ = np.linalg.qr(np.column_stack([np.ones(n_mc), F]))
                V = ZT[:n_g + n_p]
                V = V - (V @ Q) @ Q.T
                ZT = np.ascontiguousarray(np.nan_to_num(_standardise(V.T)).T)
                conf_rows = np.zeros(0, dtype=np.int64)
            gene_rows = [int(g_row[sl[0]]) for sl in slices]
            peak_lists = [(n_g + p_idx[sl]).astype(np.int64) for sl in slices]
            if solver == "numba":
                rho = _pcor_rows(ZT, gene_rows, peak_lists, conf_rows, glasso_alpha, n_jobs=n_jobs)
            else:
                Fu = ZT[conf_rows].T if len(conf_rows) else None
                args = [(ZT[g], ZT[pl].T) for g, pl in zip(gene_rows, peak_lists)]
                if n_jobs and n_jobs != 1:
                    from joblib import Parallel, delayed
                    res = Parallel(n_jobs=n_jobs)(
                        delayed(_pcor_one_gene)(y, P, Fu, glasso_alpha) for y, P in args)
                else:
                    res = [_pcor_one_gene(y, P, Fu, glasso_alpha) for y, P in args]
                rho = np.concatenate(res) if res else np.zeros(0)
            pcor[np.concatenate(slices) if slices else np.zeros(0, np.int64)] = rho

    _tick("correlations")
    if method == "abc_pcor":
        score = abc * (eps + np.maximum(pcor, 0.0))
    elif method == "abc":
        score = abc
    elif method == "pcor_contact":
        score = np.maximum(pcor, 0.0) * contact
    elif method == "pcor":
        score = pcor
    elif method == "contact":
        score = contact
    else:
        score = np.maximum(pear, 0.0) * contact

    names = np.asarray(adata.var_names).astype(str)
    out = pd.DataFrame({
        "peak": names[p_idx],
        "gene": gdf["gene_name"].to_numpy()[g_row],
        "chrom": peaks["chrom"].to_numpy()[p_idx],
        "peak_start": peaks["start"].to_numpy()[p_idx],
        "peak_end": peaks["end"].to_numpy()[p_idx],
        "tss": gdf["tss"].to_numpy()[g_row],
        "distance": dist,
        "contact": contact.astype(np.float32),
        "activity": act[p_idx].astype(np.float32),
        "abc": abc.astype(np.float32),
        "pcor": pcor.astype(np.float32),
        "pearson": pear.astype(np.float32),
        "score": score.astype(np.float64),
    })
    if inplace:
        adata.uns[key_added] = out
        adata.uns[key_added + "_params"] = dict(
            method=method, gamma=float(g_val), s0=float(s0), max_distance=int(max_distance),
            abc_window=int(abc_window), corr_window=int(corr_window),
            n_confounders=int(n_confounders), glasso_alpha=float(glasso_alpha), eps=float(eps),
            use_rep=use_rep, n_metacells=int(n_metacells), k_neighbors=int(k_neighbors),
            seed=int(seed), groupby=groupby,
            target=None if target is None else (target if isinstance(target, str) else list(target)),
            gamma_source=None if gamma_info is None else gamma_info.get("source"))
    return out


# ---------------------------------------------------------------------------
# Gene activity
# ---------------------------------------------------------------------------

def _contact_weights_chrom(centres: np.ndarray, genes_chr: pd.DataFrame, *, gamma: float,
                           s0: float, window: int, distance: str, compete: bool,
                           n_features: int) -> sp.csr_matrix:
    """``(n_genes_chr × n_features)`` contact weights for one chromosome.

    ``centres`` must be sorted. ``distance="tss"``: s = |centre − TSS|;
    ``"body"``: s = 0 inside the gene body, else the gap to the nearest
    body edge. Features farther than ``window`` from the TSS (``"tss"``) or
    from the body (``"body"``) get weight 0. ``compete`` divides every
    feature's weights by their sum over genes (each feature's accessibility is
    shared among the genes it contacts in proportion to contact)."""
    tss = genes_chr["tss"].to_numpy(np.int64)
    if distance == "tss":
        lo_b, hi_b = tss, tss
    elif distance == "body":
        lo_b = genes_chr["start"].to_numpy(np.int64)
        hi_b = genes_chr["end"].to_numpy(np.int64)
    else:
        raise ValueError("distance must be 'tss' or 'body'")
    i0 = np.searchsorted(centres, lo_b - window, side="left")
    i1 = np.searchsorted(centres, hi_b + window, side="right")
    n = np.maximum(i1 - i0, 0)
    tot = int(n.sum())
    if tot == 0:
        return sp.csr_matrix((len(genes_chr), n_features), dtype=np.float32)
    g_rep = np.repeat(np.arange(len(genes_chr)), n)
    cols = i0[g_rep] + (np.arange(tot) - np.repeat(np.cumsum(n) - n, n))
    c = centres[cols]
    s = np.maximum(0, np.maximum(lo_b[g_rep] - c, c - hi_b[g_rep]))
    w = powerlaw_contact(s, gamma, s0)
    if compete:
        colsum = np.bincount(cols, weights=w, minlength=n_features)
        w = w / colsum[cols]
    return sp.csr_matrix((w.astype(np.float32), cols, np.r_[0, np.cumsum(n)]),
                         shape=(len(genes_chr), n_features))


def contact_gene_score(
    adata: AnnData,
    gene_anno: pd.DataFrame,
    *,
    gamma: Union[float, dict] = 1.0,
    s0: float = 500,
    window: int = 100_000,
    distance: Literal["tss", "body"] = "body",
    compete: bool = True,
    tile_size: int = 500,
    use_x: Union[bool, Literal["auto"]] = "auto",
    fragment_file: Optional[str] = None,
    ceiling: Optional[int] = 4,
    scale_to: float = 10_000.0,
    log_transform: bool = False,
    exclude_chr: tuple = ("chrY", "chrM"),
    key_added: str = "contact_gene_score",
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> AnnData:
    """Per-cell gene activity from a power-law contact prior.

    .. math::
        \\mathrm{GS}_{cg} = \\sum_t \\min(x_{ct}, \\kappa)\\; w_{tg},\\qquad
        w_{tg} = C(s_{tg}) = (\\max(s_{tg}, s_0)/s_0)^{-\\gamma}
        \\;\\; (s_{tg} \\le W),

    then each cell is scaled to ``scale_to``. ``x`` are Tn5 insertion counts
    in ``tile_size`` tiles (from the fragment file, chromosome by chromosome,
    never materialising the genome-wide tile matrix) or the interval features
    of ``adata.X`` (tiles or peaks), ``κ`` = ``ceiling``. ``s`` is the
    distance of the feature centre to the TSS (``distance="tss"``) or to the
    gene body (``"body"``). With ``compete=True`` each feature's weights are
    divided by their sum over genes, so a feature's accessibility is shared
    among the genes it contacts in proportion to contact (the ABC idea on the
    element side, replacing ArchR's hard gene-boundary rule).

    Compared with :func:`epione.tl.add_gene_score_matrix` (ArchR:
    :math:`e^{-|x|/5000} + e^{-1}` within gene-boundary-capped 100 kb
    domains) the weight is a polymer-physics contact probability with a
    heavier tail and no hard boundaries.

    Arguments:
        adata: AnnData from :func:`epione.pp.import_fragments` (fragment
            path) or with interval features in ``adata.X``.
        gene_anno: ``gene_name, chrom, start, end, strand``.
        gamma: contact exponent, or a :func:`fit_contact_gamma` dict.
        s0, window: contact floor and maximum distance (bp).
        distance: ``"tss"`` or ``"body"``.
        compete: share each feature among genes in proportion to contact.
        tile_size: tile width on the fragment path.
        use_x: ``"auto"`` prefers fragments when available; ``True`` forces
            ``adata.X``; ``False`` forces fragments.
        fragment_file: explicit fragment file (overrides ``uns['files']``).
        ceiling: per-cell per-feature count cap (ArchR uses 4).
        scale_to: per-cell depth normalisation target.
        log_transform: ``log2(x + 1)`` at the end.
        key_added: ``obsm`` key; gene names in ``uns[key_added + '_gene_names']``.
        n_jobs: threads for fragment parsing.

    Returns:
        ``adata`` with ``obsm[key_added]`` (cells × genes, float32).
    """
    from ._gene_score import _normalise_genes, _adata_has_fragments
    g_val = float(gamma["gamma"]) if isinstance(gamma, dict) else float(gamma)
    genes = _normalise_genes(gene_anno, exclude_chr=exclude_chr)
    n_cells, n_genes = adata.n_obs, len(genes)
    score = np.zeros((n_cells, n_genes), dtype=np.float32)
    has_frag = fragment_file is not None or _adata_has_fragments(adata)
    use_frag = has_frag if use_x == "auto" else (not use_x)
    kw = dict(gamma=g_val, s0=s0, window=window, distance=distance, compete=compete)

    store = row_map = None
    if use_frag and fragment_file is None:
        from ..pp._fragment_store import get_fragment_store
        store = get_fragment_store(adata)
        if store is not None:
            row_map = store.covering_row_map(list(map(str, adata.obs_names)))
    if row_map is not None:
        # in-memory fragment store (epi.pp.import_fragments): one nogil per-cell pass with the
        # store kernel of add_gene_score_matrix, contact weights instead of ArchR weights
        from ..pp._fragment_store import gene_score as _store_gene_score
        weights = _contact_store_weights(store, genes, tile_size, kw, n_jobs)
        score = _store_gene_score(store, row_map, n_genes, weights, tile_size=tile_size,
                                  ceiling=ceiling, n_jobs=n_jobs)
        source = "fragment store"
    elif use_frag:
        frag = fragment_file or adata.uns["files"]["fragments"]
        for chrom, starts, matGS in _iter_chrom_tile_counts(
                frag, list(map(str, adata.obs_names)), set(genes["chrom"]), tile_size, n_jobs):
            g_chr = genes[genes["chrom"] == chrom]
            if len(g_chr) == 0:
                continue
            if ceiling is not None and ceiling > 0:
                np.clip(matGS.data, 0, ceiling, out=matGS.data)
            W = _contact_weights_chrom(starts + tile_size // 2, g_chr,
                                       n_features=len(starts), **kw)
            score[:, g_chr["idx"].to_numpy()] = np.asarray((W @ matGS).todense()).T
        source = "fragments"
    else:
        X = adata.X
        X = sp.csr_matrix(X, dtype=np.float32) if not sp.issparse(X) else X.tocsr().astype(np.float32)
        if ceiling is not None and ceiling > 0:
            X = X.copy()
            np.clip(X.data, 0, ceiling, out=X.data)
        feats = _peak_table(adata.var_names)
        Xc = X.tocsc()
        for chrom, f in feats.groupby("chrom", sort=False):
            g_chr = genes[genes["chrom"] == chrom]
            if len(g_chr) == 0:
                continue
            f = f.sort_values("center")
            W = _contact_weights_chrom(f["center"].to_numpy(np.int64), g_chr,
                                       n_features=len(f), **kw)
            sub = Xc[:, f["peak_idx"].to_numpy()]
            score[:, g_chr["idx"].to_numpy()] = np.asarray((sub @ W.T).todense())
        source = "adata.X"

    rs = score.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    score *= (scale_to / rs).astype(np.float32)
    if log_transform:
        np.log2(score + 1, out=score)
    adata.obsm[key_added] = score
    adata.uns[key_added + "_gene_names"] = genes["gene_name"].tolist()
    adata.uns[key_added + "_params"] = dict(
        gamma=g_val, s0=float(s0), window=int(window), distance=distance,
        compete=bool(compete), tile_size=int(tile_size), source=source,
        ceiling=int(ceiling) if ceiling else 0, scale_to=float(scale_to),
        log_transform=bool(log_transform))
    if verbose:
        console.info(f"[contact_gene_score] obsm[{key_added!r}] = {score.shape} "
                     f"(gamma = {g_val:.3f}, window = {window:,}, {source})")
    return adata


def _contact_store_weights(store, genes, tile_size, kw, n_jobs):
    """``(base, n_tiles, Wp, Wg, Wv)`` tile → gene contact weights in the layout of
    :func:`epione.pp._fragment_store.build_gene_weights` (CSR by global tile;
    tile grid of each chromosome = 0 … its last populated tile)."""
    from ..pp._fragment_store import _kernels, _run_blocks
    k = _kernels()
    n_chrom = len(store.chrom_names)
    res_l = []

    def cmt(r0, r1, indptr, chrom, start, lens, lpos, lend, ts):
        r = np.full(n_chrom, -1, dtype=np.int64)
        k["chrom_max_tile"](r0, r1, indptr, chrom, start, lens, lpos, lend, ts, r)
        res_l.append(r)

    _run_blocks(cmt, len(store.indptr) - 1, n_jobs, store.indptr, store.chrom, store.start,
                *store.end_args, int(tile_size))
    max_tile = np.vstack(res_l).max(axis=0) if res_l else np.full(n_chrom, -1)
    base = np.full(n_chrom, -1, dtype=np.int64)
    code = {c: i for i, c in enumerate(store.chrom_names)}
    rows_t, rows_g, vals = [], [], []
    off = 0
    for chrom, g_chr in genes.groupby("chrom", sort=False):
        c = code.get(chrom)
        if c is None or max_tile[c] < 0:
            continue
        n_t = int(max_tile[c]) + 1
        base[c] = off
        W = _contact_weights_chrom(np.arange(n_t, dtype=np.int64) * tile_size + tile_size // 2,
                                   g_chr, n_features=n_t, **kw).tocoo()
        rows_t.append(W.col.astype(np.int64) + off)
        rows_g.append(g_chr["idx"].to_numpy(np.int64)[W.row])
        vals.append(W.data.astype(np.float32))
        off += n_t
    n_tiles = off
    if rows_t:
        t_all = np.concatenate(rows_t); g_all = np.concatenate(rows_g); v_all = np.concatenate(vals)
        order = np.argsort(t_all, kind="stable")
        Wg, Wv = g_all[order], v_all[order]
        Wp = np.zeros(n_tiles + 1, dtype=np.int64)
        np.cumsum(np.bincount(t_all, minlength=n_tiles), out=Wp[1:])
    else:
        Wg = np.zeros(0, np.int64); Wv = np.zeros(0, np.float32)
        Wp = np.zeros(n_tiles + 1, dtype=np.int64)
    return base, n_tiles, Wp, Wg, Wv


def _iter_chrom_tile_counts(fragment_file, obs_keys, chroms, tile_size, n_jobs):
    """Stream a fragment file and yield ``(chrom, tile_starts, matGS)`` per
    chromosome, ``matGS`` = CSR ``(n_populated_tiles × n_cells)`` Tn5
    insertion counts (fragment start and end − 1, as ArchR / Epione's gene
    score). The file must be grouped by chromosome (tabix order)."""
    from ..pp._fragment_io import BarcodeLookup, map_fragment_chunks, sorted_reduce
    lookup = BarcodeLookup(obs_keys)
    n_cells = len(obs_keys)

    def per_chunk(ch):
        runs = []
        if len(ch) == 0:
            return runs
        cell_all = lookup(ch.barcode_names)[ch.barcode]
        brk = np.flatnonzero(ch.chrom[1:] != ch.chrom[:-1]) + 1
        for a, b in zip(np.r_[0, brk], np.r_[brk, len(ch)]):
            name = ch.chrom_names[ch.chrom[a]]
            if name not in chroms:
                runs.append((name, None, None))
                continue
            cell = cell_all[a:b]
            m = cell >= 0
            cell = cell[m]
            ts = ch.start[a:b][m] // tile_size
            te = (ch.end[a:b][m] - 1) // tile_size
            two = te != ts
            t = np.concatenate([ts, te[two]])
            c = np.concatenate([cell, cell[two]])
            if len(t) == 0:
                runs.append((name, None, None))
                continue
            key_max = (max(int(t.max()), 0) + 1) * max(n_cells, 1)
            k, w = sorted_reduce(t * n_cells + c, np.ones(len(t), np.int64), key_max)
            runs.append((name, k, w))
        return runs

    def build(parts):
        if len(parts) == 1:
            k, w = parts[0]
        else:
            ks = np.concatenate([p[0] for p in parts])
            k, w = sorted_reduce(ks, np.concatenate([p[1] for p in parts]),
                                 int(ks.max()) + 1, runs=True)
        tile = k // n_cells
        cell = k - tile * n_cells
        rs = np.flatnonzero(np.r_[True, tile[1:] != tile[:-1]])
        starts = tile[rs] * tile_size
        mat = sp.csr_matrix((w.astype(np.float32), cell, np.r_[rs, len(k)]),
                            shape=(len(starts), n_cells))
        return starts, mat

    pending, done, current = {}, set(), None
    for runs in map_fragment_chunks(fragment_file, per_chunk, n_jobs=n_jobs):
        for name, k, w in runs:
            if name != current:
                if current is not None and pending.get(current):
                    yield (current, *build(pending.pop(current)))
                if current is not None:
                    done.add(current)
                if name in done:
                    raise RuntimeError(f"{fragment_file}: fragments of {name!r} are not "
                                       "contiguous; the file must be sorted by chromosome")
                current = name
            if k is not None and len(k):
                pending.setdefault(name, []).append((k, w))
    if current is not None and pending.get(current):
        yield (current, *build(pending.pop(current)))
