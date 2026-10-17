r"""Per-cell TF activity from an empirical-Bayes conditional count model.

chromVAR (Schep *et al.* 2017) scores a motif in a cell by comparing the
cell's reads in the motif's peaks with the reads in ``n`` resampled sets of
background peaks matched on GC content and mean accessibility.
:func:`tf_activity` replaces the resampling by conditioning, and adds
motif-occupancy weights and empirical-Bayes shrinkage.

Model
-----
For cell :math:`c`, peak :math:`i` and motif :math:`m` with counts
:math:`y_{ic}` and depth :math:`n_c`,

.. math::

    y_{ic} \sim \mathrm{Poisson}(\mu_{ic}), \qquad
    \log \mu_{ic} = \log n_c + \log \pi_i + f_c(b_i) + \gamma_{cm} w_{im},

where :math:`\pi_i` is the peak's share of all reads (the offset chromVAR
uses for its expectation), :math:`b_i` is the peak's *bias stratum* -- a
cell of a quantile grid over peak covariates (GC content, log mean
accessibility, optionally a Tn5-bias summary) -- and :math:`f_c(b)` is a
free per-cell rate for each stratum. :math:`f_c` absorbs every cell-specific
bias that acts through those covariates (the reason chromVAR draws matched
background peaks). :math:`w_{im}` is the motif annotation: binary match, or
the thermodynamic occupancy of the peak's sites (below), and
:math:`\gamma_{cm}` is the cell's TF activity, a log fold change of the
motif's peaks relative to bias-matched peaks of the same cell.

Conditioning on the cell's read count per stratum,
:math:`Y_{cb} = \sum_{i \in b} y_{ic}`, removes :math:`f_c` exactly: within a
stratum the reads are multinomial over its peaks with probabilities
:math:`\pi_i e^{\gamma w_{im}} / \sum_{j \in b} \pi_j e^{\gamma w_{jm}}`.
The efficient score for :math:`\gamma_{cm}` at :math:`\gamma = 0` and its
information are therefore closed form,

.. math::

    T_{cm} = \sum_i w_{im} y_{ic}, \qquad
    U_{cm} = T_{cm} - \sum_b Y_{cb}\, q^{(1)}_{bm}, \qquad
    I_{cm} = \sum_b Y_{cb}\, \bigl(q^{(2)}_{bm} - (q^{(1)}_{bm})^2\bigr),

    q^{(k)}_{bm} = \frac{\sum_{i \in b} \pi_i w_{im}^k}{\sum_{i \in b} \pi_i},

and the one-step (IRLS-from-zero) estimate is
:math:`\hat\gamma_{cm} = U_{cm} / I_{cm}` with standard error
:math:`I_{cm}^{-1/2}`. The whole fit is three sparse products --
``Y @ W`` (cells x motifs), ``Y @ S`` (cells x strata) and two
strata x motifs tables -- so it costs one pass over the non-zeros of ``Y``
per motif hit and never resamples or densifies the count matrix.

Overdispersion and calibration
------------------------------
Reads are not independent draws (paired Tn5 insertions, peak-level
over-dispersion, co-accessibility), so the score z
:math:`z^0_{cm} = U_{cm} / \sqrt{I_{cm}}` is wider than N(0, 1), by a
motif-specific factor. ``null="empirical"`` estimates, per motif, Efron's
empirical null :math:`N(\delta_m, \sigma_m^2)` from the bulk of the cells
(median and MAD; Efron 2004, 2010 ch. 6) and reports
:math:`z_{cm} = (z^0_{cm} - \delta_m) / \sigma_m`; the standard error of
:math:`\hat\gamma` is inflated by :math:`\sigma_m` accordingly.

Peak sensitivities (rank-1 refinement)
--------------------------------------
Not every peak carrying a motif depends on the TF. The default
``peak_sensitivity="eb"`` extends the effect to
:math:`\gamma_{cm}\beta_{im} w_{im}`, estimates each peak's sensitivity
:math:`\beta_{im}` by a one-step Poisson regression of the peak's counts on
the cells' first-pass (leave-one-out) activities, shrinks the
sensitivities of each motif to their mean with a normal prior fitted by
moments (James-Stein / Efron & Morris 1973), and re-scores the cells with
:math:`w_{im}\beta_{im}` -- one alternating step of a bilinear (rank-1)
Poisson factor model restricted to the motif's peaks. Standard errors use
the quasi-Poisson dispersion of the conditional null (McCullagh & Nelder
1989), about 2 for paired Tn5 insertions.

Empirical-Bayes shrinkage
-------------------------
Per-cell estimates are noisy and their precision varies with depth. For
each motif, ``shrinkage="ash"`` puts a unimodal prior on the effects,
:math:`\gamma_{cm} \sim g_m = \pi_0 \delta_0 + \sum_k \pi_k N(0, s_k^2)`
on a fixed geometric grid of scales, fits the mixture weights by maximum
marginal likelihood over all cells with EM (adaptive shrinkage; Stephens
2017), and reports the posterior mean and standard deviation of each
:math:`\gamma_{cm}`. Posterior means borrow strength across cells: shallow
cells are pulled towards zero in proportion to their noise, which is what
the likelihood-ratio ranking of cells calls for when depth varies.

Occupancy weights
-----------------
:func:`add_motif_occupancy` scans peaks with MOODS (the scanner pychromvar
and chromVAR's motifmatchr use) and weights each peak by the probability
that at least one of its sites is bound under a two-state thermodynamic
model (Berg & von Hippel 1987; Granek & Clarke 2005),

.. math::

    w_{im} = 1 - \prod_{s \in i} \bigl(1 - \sigma(S_s - t_m - \Delta)\bigr),

with :math:`S_s` the site's natural-log PWM log-odds score, :math:`t_m` the
score at the match p-value threshold and :math:`\Delta` a chemical-potential
offset.

References
----------
Schep, A. N. *et al.* chromVAR. *Nat Methods* 14, 975-978 (2017).
Efron, B. Large-scale simultaneous hypothesis testing: the choice of a null
hypothesis. *JASA* 99, 96-104 (2004); *Large-Scale Inference*, CUP (2010).
Stephens, M. False discovery rates: a new deal. *Biostatistics* 18,
275-294 (2017).
Berg, O. G. & von Hippel, P. H. Selection of DNA binding sites by regulatory
proteins. *J Mol Biol* 193, 723-750 (1987).
Granek, J. A. & Clarke, N. D. Explicit equilibrium modeling of transcription-
factor binding and gene regulation. *Genome Biol* 6, R87 (2005).
Efron, B. & Morris, C. Stein's estimation rule and its competitors -- an
empirical Bayes approach. *JASA* 68, 117-130 (1973).
McCullagh, P. & Nelder, J. A. *Generalized Linear Models*, 2nd ed. (1989).
Rao, C. R. Large sample tests of statistical hypotheses concerning several
parameters. *Proc Camb Phil Soc* 44, 50-57 (1948) (score test).
"""
from __future__ import annotations

import os
from typing import Dict, Literal, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [tf_activity] {msg}", flush=True)


try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is a core dependency
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @njit(parallel=True, cache=True, boundscheck=False)
    def _nb_csr_csr_dense(xp, xi, xd, wp, wi, wd, n_cols, out):
        """``out = X @ W`` for CSR ``X`` (cells x peaks) and CSR ``W``
        (peaks x motifs), dense float64 output; parallel over cells. Cost
        is ``sum_{(c,i) in nnz(X)} nnz(W[i])``."""
        n_rows = xp.shape[0] - 1
        for c in prange(n_rows):
            row = np.zeros(n_cols, dtype=np.float64)
            for k in range(xp[c], xp[c + 1]):
                i = xi[k]
                v = np.float64(xd[k])
                for q in range(wp[i], wp[i + 1]):
                    row[wi[q]] += v * wd[q]
            out[c, :] = row

    @njit(parallel=True, cache=True, boundscheck=False)
    def _nb_csr_group_sum(xp, xi, xd, group, n_groups, out):
        """``out[c, g] = sum_{i: group[i] = g} X[c, i]`` (cells x strata)."""
        n_rows = xp.shape[0] - 1
        for c in prange(n_rows):
            for k in range(xp[c], xp[c + 1]):
                out[c, group[xi[k]]] += np.float64(xd[k])

    @njit(parallel=True, cache=True, boundscheck=False)
    def _nb_ash(ghat, se, grid, prior0, max_iter, tol, pm_out, psd_out,
                w_out):
        """Adaptive shrinkage per motif (columns), EM on mixture weights.

        Prior ``g = sum_k pi_k N(0, grid_k^2)`` (``grid[0] = 0`` is the
        point mass); likelihood ``ghat ~ N(gamma, se^2)``. ``prior0`` is
        the Dirichlet pseudo-count on the null component (ashr's
        ``nullweight``; 1 = no penalty)."""
        n, m = ghat.shape
        K = grid.shape[0]
        for j in prange(m):
            pi = np.full(K, 1.0 / K)
            lik = np.empty((n, K))
            for c in range(n):
                s2 = se[c, j] * se[c, j]
                x = ghat[c, j]
                for k in range(K):
                    v = grid[k] * grid[k] + s2
                    lik[c, k] = np.exp(-0.5 * x * x / v) / np.sqrt(v)
            prev = -np.inf
            for it in range(max_iter):
                acc = np.zeros(K)
                ll = 0.0
                for c in range(n):
                    tot = 0.0
                    for k in range(K):
                        tot += pi[k] * lik[c, k]
                    if tot <= 0.0:
                        continue
                    ll += np.log(tot)
                    for k in range(K):
                        acc[k] += pi[k] * lik[c, k] / tot
                acc[0] += prior0 - 1.0
                s = 0.0
                for k in range(K):
                    s += acc[k]
                for k in range(K):
                    pi[k] = acc[k] / s
                if ll - prev < tol * (abs(ll) + 1.0):
                    break
                prev = ll
            for k in range(K):
                w_out[j, k] = pi[k]
            for c in range(n):
                s2 = se[c, j] * se[c, j]
                x = ghat[c, j]
                tot = 0.0
                m1 = 0.0
                m2 = 0.0
                for k in range(K):
                    r = pi[k] * lik[c, k]
                    tot += r
                    if grid[k] > 0.0:
                        g2 = grid[k] * grid[k]
                        pv = g2 * s2 / (g2 + s2)
                        pmk = pv * x / s2
                        m1 += r * pmk
                        m2 += r * (pv + pmk * pmk)
                if tot > 0.0:
                    m1 /= tot
                    m2 /= tot
                pm_out[c, j] = m1
                var = m2 - m1 * m1
                psd_out[c, j] = np.sqrt(var) if var > 0.0 else 0.0


if _HAS_NUMBA:
    @njit(parallel=True, cache=True, boundscheck=False)
    def _nb_peak_scores(cp, ci, cd, wp, wi, wd, strata, scale, Ycb, G, invV0,
                        u_out, l_out):
        """Per (peak i, motif m) in nnz(W), over the cells with y_ic > 0:

            u_out[q] = sum_c y_ic G[c, m]
            l_out[q] = sum_c (y_ic^2 - 2 y_ic mu_ic) invV0[c, m]

        with ``mu_ic = Ycb[c, strata[i]] * scale[i]``; ``cp/ci/cd`` is the
        CSC (peak-major) count matrix, ``q`` indexes W's CSR data."""
        n_peaks = cp.shape[0] - 1
        for i in prange(n_peaks):
            b = strata[i]
            s = scale[i]
            for k in range(cp[i], cp[i + 1]):
                c = ci[k]
                y = np.float64(cd[k])
                mu = Ycb[c, b] * s
                for q in range(wp[i], wp[i + 1]):
                    m = wi[q]
                    u_out[q] += y * G[c, m]
                    l_out[q] += (y * y - 2.0 * y * mu) * invV0[c, m]


def _peak_sensitivity(X, W, strata, pi, P_b, Ycb, U, V0, groups=None, tau_scale=1.0,
                      dispersion=1.0):
    r"""Empirical-Bayes peak sensitivities :math:`\beta_{im}` (rank-1 step).

    Extends the model to :math:`\gamma_{cm} \beta_{im} w_{im}`: not every
    motif peak responds equally to the TF. Given the first-pass activities
    :math:`\tilde\gamma_{cm} = \hat\gamma_{cm} - \mathrm{median}_c`, the
    one-step score for :math:`\beta_{im}` (Poisson regression of the peak
    on the cells' activities, offsets from the conditional null) is

    .. math::

        U_{im} = \sum_c \tilde\gamma_{cm}^{(-i)} (y_{ic} - \mu_{ic}),\qquad
        I_{im} = \sum_c \tilde\gamma_{cm}^2 \mu_{ic},

    with :math:`\mu_{ic} = Y_{c b_i} \pi_i / P_{b_i}` and the leave-one-out
    activity :math:`\tilde\gamma^{(-i)} = \tilde\gamma - w_{im}
    (y_{ic} - \mu_{ic}) / V_{cm}` (removes the peak's own contribution, so
    a peak is not rewarded for correlating with itself). The slopes
    :math:`\hat\beta = U / I` (s.e. :math:`I^{-1/2}`) are shrunk per motif
    to their precision-weighted mean :math:`\beta_0` with a normal prior
    whose variance is the method-of-moments estimate
    :math:`\tau^2 = \max(0, \mathrm{var}(\hat\beta) - \overline{se^2})`
    (James-Stein / Efron-Morris), clipped at 0 and divided by
    :math:`\beta_0` (only relative sensitivities matter). Motifs with
    :math:`\beta_0 \le 0` or :math:`\tau^2 = 0` keep unit weights.
    """
    n_cells, n_peaks = X.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(V0 > 0, U / V0, 0.0)
        invV0 = np.where(V0 > 0, 1.0 / V0, 0.0)
    if groups is None:
        G = g - np.median(g, axis=0)
    else:  # centre within cell groups: sensitivities from within-group co-variation
        G = np.empty_like(g)
        for lab in np.unique(groups):
            sel = groups == lab
            G[sel] = g[sel] - np.median(g[sel], axis=0)
    scale = np.where(P_b[strata] > 0, pi / P_b[strata], 0.0)
    Xc = X.tocsc()
    u = np.zeros(W.nnz); l = np.zeros(W.nnz)
    _nb_peak_scores(Xc.indptr, Xc.indices, Xc.data, W.indptr, W.indices, W.data,
                    strata, scale, Ycb, G, invV0, u, l)
    del Xc
    rows = np.repeat(np.arange(n_peaks), np.diff(W.indptr))
    cols = W.indices
    bq = strata[rows]; sq = scale[rows]
    GY = Ycb.T @ G                       # strata x motifs: sum_c Y_cb G_cm
    G2 = Ycb.T @ (G * G)                 # sum_c Y_cb G_cm^2
    H = (Ycb * Ycb).T @ invV0            # sum_c Y_cb^2 / V_cm
    U_i = u - sq * GY[bq, cols]
    L_i = l + sq * sq * H[bq, cols]
    U_i = U_i - W.data * L_i
    I_i = sq * G2[bq, cols]
    beta = np.ones(W.nnz)
    ok = I_i > 0
    bh = np.zeros(W.nnz); se2 = np.full(W.nnz, np.inf)
    bh[ok] = U_i[ok] / I_i[ok]; se2[ok] = dispersion / I_i[ok]
    n_m = W.shape[1]
    for m in range(n_m):
        sel = np.flatnonzero((cols == m) & ok)
        if len(sel) < 10:
            continue
        w = 1.0 / se2[sel]
        b0 = np.sum(w * bh[sel]) / np.sum(w)
        tau2 = tau_scale * max(np.var(bh[sel]) - np.mean(se2[sel]), 0.0)
        if b0 <= 0 or tau2 <= 0:
            continue
        post = b0 + tau2 / (tau2 + se2[sel]) * (bh[sel] - b0)
        beta[sel] = np.maximum(post, 0.0) / b0
    return beta


if _HAS_NUMBA:
    @njit(parallel=True, cache=True, boundscheck=False)
    def _nb_pearson_nnz(xp, xi, xd, strata, scale, Ycb, out):
        """``out[c] = sum_{i: y_ic > 0} y_ic^2 / mu_ic`` with
        ``mu_ic = Ycb[c, strata[i]] * scale[i]`` (CSR, parallel over cells)."""
        n_rows = xp.shape[0] - 1
        for c in prange(n_rows):
            acc = 0.0
            for k in range(xp[c], xp[c + 1]):
                i = xi[k]
                mu = Ycb[c, strata[i]] * scale[i]
                if mu > 0.0:
                    y = np.float64(xd[k])
                    acc += y * y / mu
            out[c] = acc


def quasi_dispersion(X, strata, pi, P_b, Ycb) -> float:
    r"""Quasi-Poisson dispersion of the conditional null (McCullagh & Nelder
    1989, sec. 4.5): Pearson :math:`X^2 / (N - p)` with
    :math:`X^2 = \sum_{c,i} (y_{ic} - \mu_{ic})^2 / \mu_{ic}
    = \sum_{y_{ic} > 0} y_{ic}^2 / \mu_{ic} - \sum y_{ic}` (the fitted
    :math:`\mu` reproduce each cell's stratum totals), ``N`` cells x peaks
    and ``p`` cells x strata. Paired Tn5 insertions alone give about 2."""
    n_cells, n_peaks = X.shape
    scale = np.where(P_b[strata] > 0, pi / P_b[strata], 0.0)
    acc = np.zeros(n_cells)
    _nb_pearson_nnz(X.indptr, X.indices, X.data, strata.astype(np.int64), scale, Ycb, acc)
    x2 = acc.sum() - float(X.sum())
    dof = n_cells * n_peaks - n_cells * Ycb.shape[1]
    return max(x2 / max(dof, 1), 1e-12)


def _csr_at_csr_dense(X: sp.csr_matrix, W: sp.csr_matrix) -> np.ndarray:
    out = np.zeros((X.shape[0], W.shape[1]), dtype=np.float64)
    if _HAS_NUMBA:
        # no dtype copies of X (it can hold 10^9 non-zeros); numba
        # specialises on whatever integer / float types the CSR carries.
        _nb_csr_csr_dense(X.indptr, X.indices, X.data, W.indptr, W.indices,
                          W.data.astype(np.float64), W.shape[1], out)
    else:  # pragma: no cover
        out[:] = (X @ W).toarray()
    return out


def _group_sum(X: sp.csr_matrix, group: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros((X.shape[0], n_groups), dtype=np.float64)
    if _HAS_NUMBA:
        _nb_csr_group_sum(X.indptr, X.indices, X.data, group.astype(np.int64),
                          n_groups, out)
    else:  # pragma: no cover
        S = sp.csr_matrix((np.ones(len(group)), (np.arange(len(group)), group)),
                          shape=(len(group), n_groups))
        out[:] = (X @ S).toarray()
    return out


# ---------------------------------------------------------------------------
# Bias strata
# ---------------------------------------------------------------------------

def bias_strata(covariates: np.ndarray, n_bins: Sequence[int]) -> np.ndarray:
    """Assign each peak to a cell of a quantile grid over its covariates.

    Parameters
    ----------
    covariates
        ``(n_peaks, k)`` array (e.g. GC content, log10 mean accessibility).
    n_bins
        Number of quantile bins per covariate (length ``k``).

    Returns
    -------
    ``(n_peaks,)`` int64 stratum id, compacted to ``0 .. n_strata - 1``.
    Ties are broken by rank so that every bin holds about the same number
    of peaks even for discrete covariates.
    """
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov[:, None]
    if cov.shape[1] != len(n_bins):
        raise ValueError("need one bin count per covariate")
    n = cov.shape[0]
    sid = np.zeros(n, dtype=np.int64)
    for j, nb in enumerate(n_bins):
        nb = int(nb)
        if nb <= 1:
            continue
        r = np.empty(n, dtype=np.int64)
        r[np.argsort(cov[:, j], kind="stable")] = np.arange(n)
        sid = sid * nb + np.minimum(r * nb // n, nb - 1)
    _, sid = np.unique(sid, return_inverse=True)
    return sid.astype(np.int64)


def _gc_from_sequences(seqs: Sequence[str]) -> np.ndarray:
    gc = np.empty(len(seqs))
    for i, s in enumerate(seqs):
        s = s.upper()
        n = len(s) - s.count("N")
        gc[i] = (s.count("G") + s.count("C")) / n if n > 0 else 0.5
    return gc


def _peak_sequences(adata: AnnData, genome_fasta: str) -> list:
    """Upper-case peak sequences from ``var`` (chrom/start/end columns or
    ``chrom:start-end`` / ``chrom-start-end`` names; 0-based half-open)."""
    from pysam import FastaFile
    if {"chrom", "start", "end"} <= set(adata.var.columns):
        ch = adata.var["chrom"].astype(str).to_numpy()
        st = adata.var["start"].astype(np.int64).to_numpy()
        en = adata.var["end"].astype(np.int64).to_numpy()
    else:
        import re
        parts = [re.split(r"[:\-_]", str(v)) for v in adata.var_names]
        ch = np.array(["-".join(p[:-2]) if len(p) > 3 else p[0] for p in parts])
        st = np.array([int(p[-2]) for p in parts], dtype=np.int64)
        en = np.array([int(p[-1]) for p in parts], dtype=np.int64)
    fa = FastaFile(genome_fasta)
    try:
        return [fa.fetch(c, int(s), int(e)).upper() for c, s, e in zip(ch, st, en)]
    finally:
        fa.close()


# ---------------------------------------------------------------------------
# Motif occupancy
# ---------------------------------------------------------------------------

_SCAN_STATE: Dict[str, object] = {}


def _scan_init(matrices, thresholds):
    import MOODS.scan
    import MOODS.tools
    sc = MOODS.scan.Scanner(7)
    sc.set_motifs(matrices=matrices, bg=MOODS.tools.flat_bg(4),
                  thresholds=thresholds)
    _SCAN_STATE["scanner"] = sc
    _SCAN_STATE["n"] = len(matrices) // 2


def _scan_chunk(args):
    start, seqs = args
    sc = _SCAN_STATE["scanner"]
    n = _SCAN_STATE["n"]
    pk, mo, sc_out = [], [], []
    for off, s in enumerate(seqs):
        res = sc.scan(s)
        for j in range(2 * n):
            r = res[j]
            if len(r):
                m = j if j < n else j - n
                for hit in r:
                    pk.append(start + off)
                    mo.append(m)
                    sc_out.append(hit.score)
    return (np.asarray(pk, dtype=np.int64), np.asarray(mo, dtype=np.int32),
            np.asarray(sc_out, dtype=np.float32))


def scan_motif_sites(
    seqs: Sequence[str],
    pwms,
    *,
    p_value: float = 5e-5,
    site_p_value: Optional[float] = None,
    pseudocounts: float = 1e-4,
    n_jobs: int = -1,
    chunk: int = 2000,
) -> dict:
    """MOODS scan returning every site (both strands) above a threshold.

    Matrices, pseudocounts, the even background and the p-value threshold
    are those of :func:`pychromvar.match_motif` (and chromVAR's
    motifmatchr), so ``score >= thr_match`` reproduces the shared binary
    annotation. ``site_p_value`` (default: ``p_value``) admits weaker
    sites for occupancy weighting.

    Returns
    -------
    dict with ``peak``, ``motif``, ``score`` (natural-log log-odds) arrays,
    ``thr_match`` / ``thr_site`` per motif and the motif ``names``.
    """
    import MOODS.tools
    site_p_value = p_value if site_p_value is None else float(site_p_value)
    if site_p_value < p_value:
        raise ValueError("site_p_value must be >= p_value")
    bg = MOODS.tools.flat_bg(4)
    n = len(pwms)
    mats, thr_scan = [None] * (2 * n), [None] * (2 * n)
    thr_match = np.empty(n)
    thr_site = np.empty(n)
    names = []
    for i, m in enumerate(pwms):
        counts = tuple(tuple(m.counts[b]) for b in "ACGT")
        mats[i] = MOODS.tools.log_odds(counts, bg, pseudocounts)
        mats[i + n] = MOODS.tools.reverse_complement(mats[i])
        thr_match[i] = MOODS.tools.threshold_from_p(mats[i], bg, p_value)
        thr_site[i] = MOODS.tools.threshold_from_p(mats[i], bg, site_p_value)
        thr_scan[i] = thr_scan[i + n] = thr_site[i]
        names.append(f"{getattr(m, 'matrix_id', i)}.{getattr(m, 'name', i)}")
    jobs = [(s, list(seqs[s:s + chunk])) for s in range(0, len(seqs), chunk)]
    n_jobs = (os.cpu_count() or 1) if n_jobs is None or n_jobs < 0 else max(1, n_jobs)
    if n_jobs == 1 or len(jobs) == 1:
        _scan_init(mats, thr_scan)
        parts = [_scan_chunk(j) for j in jobs]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(n_jobs, initializer=_scan_init, initargs=(mats, thr_scan)) as pool:
            parts = pool.map(_scan_chunk, jobs, chunksize=1)
    return dict(
        peak=np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64),
        motif=np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int32),
        score=np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, np.float32),
        thr_match=thr_match, thr_site=thr_site, names=np.asarray(names, dtype=object),
        n_peaks=len(seqs),
    )


def occupancy_weights(
    sites: dict,
    *,
    offset: float = 0.0,
    temperature: float = 1.0,
    matched_only: bool = True,
    binary: bool = False,
) -> sp.csr_matrix:
    r"""Peak x motif weights from scanned sites.

    ``binary=True`` gives the 0/1 match matrix (any site with score
    :math:`\ge t_m`). Otherwise

    .. math::

        w_{im} = 1 - \prod_s \bigl(1 - \sigma((S_s - t_m - \Delta)/T)\bigr)

    over the peak's sites, with ``offset`` :math:`\Delta` (nats) and
    ``temperature`` :math:`T`. With ``matched_only`` the weights are kept
    only on peaks that carry a match, so the annotation (which peaks count
    for a motif) is the binary one and only the weights differ.
    """
    pk, mo, sc = sites["peak"], sites["motif"].astype(np.int64), sites["score"].astype(np.float64)
    tm = np.asarray(sites["thr_match"], dtype=np.float64)
    n_peaks, n_motifs = int(sites["n_peaks"]), len(tm)
    key = pk * n_motifs + mo
    is_match = sc >= tm[mo] - 1e-9
    if binary:
        u = np.unique(key[is_match])
        return sp.csr_matrix((np.ones(len(u), dtype=np.float32),
                              (u // n_motifs, u % n_motifs)), shape=(n_peaks, n_motifs))
    p = 1.0 / (1.0 + np.exp(-(sc - tm[mo] - offset) / temperature))
    lq = np.log1p(-np.minimum(p, 1 - 1e-12))
    u, inv = np.unique(key, return_inverse=True)
    w = 1.0 - np.exp(np.bincount(inv, weights=lq, minlength=len(u)))
    if matched_only:
        keep = np.zeros(len(u), dtype=bool)
        keep[np.unique(inv[is_match])] = True
        u, w = u[keep], w[keep]
    return sp.csr_matrix((w.astype(np.float32), (u // n_motifs, u % n_motifs)),
                         shape=(n_peaks, n_motifs))


def add_motif_occupancy(
    adata: AnnData,
    genome_fasta: str,
    pwms,
    *,
    p_value: float = 5e-5,
    site_p_value: float = 1e-4,
    offset: float = 0.0,
    temperature: float = 1.0,
    matched_only: bool = True,
    key_added: str = "motif",
    gc_key: str = "gc_bias",
    n_jobs: int = -1,
    verbose: bool = True,
) -> AnnData:
    """Scan peaks and store binary matches and occupancy weights.

    Writes ``adata.varm[key_added]`` (binary peak x motif CSR, the same
    annotation as :func:`epione.tl.add_motif_matrix` with the same
    ``p_value``), ``adata.varm[key_added + '_occupancy']`` (weights, see
    :func:`occupancy_weights`), ``adata.uns[key_added + '_names']`` and,
    if missing, ``adata.var[gc_key]`` (GC fraction of the peak sequence).
    """
    from ._motif_matrix import _coerce_pwms_to_biopython
    pwms = _coerce_pwms_to_biopython(pwms)
    _console(f"reading {adata.n_vars:,} peak sequences", verbose)
    seqs = _peak_sequences(adata, genome_fasta)
    if gc_key not in adata.var.columns:
        adata.var[gc_key] = _gc_from_sequences(seqs)
    _console(f"MOODS scan of {len(pwms)} motifs (site p < {site_p_value:g})", verbose)
    sites = scan_motif_sites(seqs, pwms, p_value=p_value,
                             site_p_value=site_p_value, n_jobs=n_jobs)
    adata.varm[key_added] = occupancy_weights(sites, binary=True)
    adata.varm[f"{key_added}_occupancy"] = occupancy_weights(
        sites, offset=offset, temperature=temperature, matched_only=matched_only)
    adata.uns[f"{key_added}_names"] = sites["names"]
    adata.uns[f"{key_added}_occupancy_params"] = dict(
        p_value=p_value, site_p_value=site_p_value, offset=offset,
        temperature=temperature, matched_only=matched_only)
    _console(f"{adata.varm[key_added].nnz:,} peak-motif matches", verbose)
    return adata


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def tf_activity_core(
    X,
    W,
    strata: np.ndarray,
    *,
    null: Literal["empirical", "theoretical"] = "empirical",
    shrinkage: Literal["ash", "none"] = "ash",
    peak_sensitivity: Literal["none", "eb"] = "none",
    sensitivity_groups: Optional[np.ndarray] = None,
    sensitivity_tau_scale: float = 1.0,
    dispersion: Literal["quasi", "poisson"] = "poisson",
    ash_grid_mult: float = np.sqrt(2.0),
    ash_nullweight: float = 10.0,
    ash_max_iter: int = 500,
    verbose: bool = True,
) -> dict:
    """Array-level fit behind :func:`tf_activity`.

    Parameters
    ----------
    X
        ``(n_cells, n_peaks)`` counts (sparse or dense).
    W
        ``(n_peaks, n_motifs)`` motif weights (binary or occupancy).
    strata
        ``(n_peaks,)`` bias-stratum id per peak (:func:`bias_strata`).

    Returns
    -------
    dict of ``(n_cells, n_motifs)`` arrays: ``z0`` (raw score z),
    ``z`` (calibrated z), ``gamma`` (one-step log fold change), ``se``,
    and with shrinkage ``post_mean`` / ``post_sd`` / ``post_z``; plus
    per-motif ``null_mean`` / ``null_sd`` and the ``ash_weights``.
    """
    X = sp.csr_matrix(X)
    X.sum_duplicates()
    W = sp.csr_matrix(W, dtype=np.float64, copy=True)
    strata = np.asarray(strata, dtype=np.int64)
    n_cells, n_peaks = X.shape
    if W.shape[0] != n_peaks or strata.shape[0] != n_peaks:
        raise ValueError("X, W and strata disagree on the number of peaks")
    n_str = int(strata.max()) + 1
    peak_tot = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    pi = peak_tot / max(peak_tot.sum(), 1.0)
    # strata x motifs tables of the multinomial null
    P_b = np.bincount(strata, weights=pi, minlength=n_str)
    S = sp.csr_matrix((pi, (strata, np.arange(n_peaks))), shape=(n_str, n_peaks))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(P_b > 0, 1.0 / P_b, 0.0)
    _console(f"{n_cells:,} cells x {n_peaks:,} peaks x {W.shape[1]:,} motifs; "
             f"{n_str} bias strata", verbose)
    Ycb = _group_sum(X, strata, n_str)

    def _score(Wk):
        Q1 = (sp.diags(inv) @ (S @ Wk)).toarray()
        Q2 = (sp.diags(inv) @ (S @ Wk.multiply(Wk))).toarray()
        Qv = np.maximum(Q2 - Q1 * Q1, 0.0)
        T = _csr_at_csr_dense(X, Wk)
        return T - Ycb @ Q1, Ycb @ Qv

    U, V0 = _score(W)
    phi = quasi_dispersion(X, strata, pi, P_b, Ycb) if dispersion == "quasi" else 1.0
    if dispersion == "quasi":
        _console(f"quasi-Poisson dispersion {phi:.3f}", verbose)
    beta = None
    if peak_sensitivity == "eb":
        _console("peak sensitivities (rank-1 refinement, EB-shrunk)", verbose)
        beta = _peak_sensitivity(X, W, strata, pi, P_b, Ycb, U, V0,
                                 groups=sensitivity_groups, tau_scale=sensitivity_tau_scale,
                                 dispersion=phi)
        # fresh index arrays: scipy shares them between converted matrices,
        # and eliminate_zeros() would otherwise rewrite the caller's W
        W = sp.csr_matrix((W.data * beta, W.indices.copy(), W.indptr.copy()),
                          shape=W.shape)
        W.eliminate_zeros()
        U, V0 = _score(W)
    elif peak_sensitivity != "none":
        raise ValueError(f"unknown peak_sensitivity {peak_sensitivity!r}")
    with np.errstate(divide="ignore", invalid="ignore"):
        z0 = np.where(V0 > 0, U / np.sqrt(V0), 0.0)
    if null == "empirical":
        d0 = np.median(z0, axis=0)
        s0 = 1.4826 * np.median(np.abs(z0 - d0), axis=0)
        s0 = np.where(s0 > 0, s0, 1.0)
    elif null == "theoretical":
        d0 = np.zeros(z0.shape[1]); s0 = np.full(z0.shape[1], np.sqrt(phi))
    else:
        raise ValueError(f"unknown null {null!r}")
    z = (z0 - d0) / s0
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where(V0 > 0, s0 / np.sqrt(V0), np.inf)
        gamma = np.where(V0 > 0, z * se, 0.0)
    out = dict(z0=z0.astype(np.float32), z=z.astype(np.float32),
               gamma=gamma.astype(np.float32), se=se.astype(np.float32),
               null_mean=d0, null_sd=s0, n_strata=n_str)
    if beta is not None:
        out["peak_sensitivity"] = beta
    if shrinkage == "ash":
        _console("adaptive shrinkage (EM over cells, per motif)", verbose)
        se_f = np.where(np.isfinite(se), se, 1e6)
        smin = max(np.nanmin(se_f[se_f < 1e5]) / 10.0, 1e-8)
        smax = 2.0 * np.sqrt(max(np.max(gamma ** 2 - se_f ** 2), 0.0))
        smax = max(smax, 8.0 * smin)
        n_grid = int(np.ceil(np.log(smax / smin) / np.log(ash_grid_mult))) + 1
        grid = np.concatenate([[0.0], smin * ash_grid_mult ** np.arange(n_grid)])
        pm = np.empty_like(gamma); psd = np.empty_like(gamma)
        wts = np.empty((gamma.shape[1], len(grid)))
        _nb_ash(np.ascontiguousarray(gamma), np.ascontiguousarray(se_f), grid,
                float(ash_nullweight), int(ash_max_iter), 1e-8, pm, psd, wts)
        with np.errstate(divide="ignore", invalid="ignore"):
            pz = np.where(psd > 0, pm / psd, 0.0)
        out.update(post_mean=pm.astype(np.float32), post_sd=psd.astype(np.float32),
                   post_z=pz.astype(np.float32), ash_weights=wts, ash_grid=grid)
    elif shrinkage != "none":
        raise ValueError(f"unknown shrinkage {shrinkage!r}")
    return out


def tf_activity(
    adata: AnnData,
    *,
    motif_key: str = "motif",
    weights_key: Optional[str] = None,
    gc_key: str = "gc_bias",
    covariates: Sequence[str] = (),
    n_bins: Sequence[int] = (30, 30),
    genome_fasta: Optional[str] = None,
    peak_sensitivity: Literal["eb", "none"] = "eb",
    groupby: Optional[str] = None,
    dispersion: Literal["quasi", "poisson"] = "quasi",
    null: Literal["theoretical", "empirical"] = "theoretical",
    shrinkage: Literal["ash", "none"] = "ash",
    score: Literal["post_mean", "post_z", "z", "gamma"] = "post_mean",
    layer: Optional[str] = None,
    key_added: str = "tf_activity",
    verbose: bool = True,
) -> AnnData:
    """Per-cell TF activity by an empirical-Bayes conditional count model.

    See the module docstring for the model. Uses the motif annotation in
    ``adata.varm[motif_key]`` (:func:`epione.tl.add_motif_matrix` or
    :func:`add_motif_occupancy`) and needs no background peaks.

    Parameters
    ----------
    adata
        Cells x peaks counts in ``X`` (or ``layers[layer]``).
    motif_key
        Binary peak x motif annotation in ``varm``.
    weights_key
        Optional ``varm`` key with peak x motif weights (e.g.
        ``'motif_occupancy'``); default uses ``motif_key``.
    gc_key
        GC column in ``var``; computed from ``genome_fasta`` if missing.
    covariates
        Extra ``var`` columns (e.g. a per-peak Tn5-bias summary) used, after
        GC and log10 mean accessibility, to define the bias strata.
    n_bins
        Quantile bins per covariate, in the order (GC, log accessibility,
        *covariates*). The strata are the cells of the product grid.
    peak_sensitivity
        ``'eb'`` (default): one rank-1 refinement step that estimates how
        strongly each motif peak follows the motif's activity across cells,
        shrinks these sensitivities by empirical Bayes and re-scores with
        them (:func:`_peak_sensitivity`); ``'none'``: all motif peaks
        weigh the same, as in chromVAR.
    groupby
        Optional ``obs`` column (sample / batch / cell type). Sensitivities
        are then learnt from co-variation *within* these groups, so that
        differences between cell types do not define which peaks count.
    dispersion
        ``'quasi'`` (default): quasi-Poisson dispersion of the conditional
        null (:func:`quasi_dispersion`; ~2 for paired Tn5 insertions)
        scales all standard errors; ``'poisson'``: dispersion 1.
    null
        ``'theoretical'`` (default; N(0, dispersion)) or ``'empirical'``
        (Efron's per-motif empirical null) calibration of the score z.
    shrinkage
        ``'ash'`` (adaptive shrinkage of the per-cell log fold changes,
        default) or ``'none'``.
    score
        What goes to ``adata.obsm[key_added]``: ``'post_mean'`` (default;
        shrunken log fold change), ``'post_z'`` (posterior mean / posterior
        sd), ``'z'`` (calibrated score z) or ``'gamma'`` (one-step log fold
        change).

    Defaults were chosen on the GSE274113 CRISPR-KO benchmark (Fig. 2c
    task 5) before evaluation on independent perturbation screens.

    Returns
    -------
    ``adata`` with ``obsm[key_added]`` (cells x motifs, the chosen score),
    ``obsm[key_added + '_z']``, ``obsm[key_added + '_lfc']`` (one-step log
    fold change) and, with shrinkage, ``obsm[key_added + '_post_mean']``
    and ``obsm[key_added + '_post_sd']``; ``uns[key_added + '_names']`` and
    ``uns[key_added + '_params']``.
    """
    if motif_key not in adata.varm:
        raise KeyError(f"adata.varm[{motif_key!r}] not found; run "
                       "epione.tl.add_motif_matrix or add_motif_occupancy first.")
    W = adata.varm[weights_key if weights_key is not None else motif_key]
    W = sp.csr_matrix(W, dtype=np.float64)
    X = adata.layers[layer] if layer is not None else adata.X
    X = sp.csr_matrix(X)
    if gc_key not in adata.var.columns:
        if genome_fasta is None:
            raise ValueError(f"adata.var[{gc_key!r}] missing; pass genome_fasta=")
        adata.var[gc_key] = _gc_from_sequences(_peak_sequences(adata, genome_fasta))
    peak_tot = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    cov = [adata.var[gc_key].to_numpy(np.float64), np.log10(peak_tot + 1.0)]
    cov += [adata.var[c].to_numpy(np.float64) for c in covariates]
    if len(n_bins) != len(cov):
        raise ValueError(f"n_bins needs {len(cov)} entries (GC, log accessibility"
                         f"{', ' + ', '.join(covariates) if covariates else ''})")
    strata = bias_strata(np.column_stack(cov), n_bins)
    groups = None if groupby is None else adata.obs[groupby].astype(str).to_numpy()
    res = tf_activity_core(X, W, strata, null=null, shrinkage=shrinkage,
                           peak_sensitivity=peak_sensitivity, sensitivity_groups=groups,
                           dispersion=dispersion, verbose=verbose)
    if score in ("post_z", "post_mean") and shrinkage == "none":
        raise ValueError(f"score={score!r} needs shrinkage='ash'")
    adata.obsm[key_added] = res[score]
    adata.obsm[f"{key_added}_z"] = res["z"]
    adata.obsm[f"{key_added}_lfc"] = res["gamma"]
    if shrinkage == "ash":
        adata.obsm[f"{key_added}_post_mean"] = res["post_mean"]
        adata.obsm[f"{key_added}_post_sd"] = res["post_sd"]
    names = adata.uns.get(f"{motif_key}_names", [f"M{j}" for j in range(W.shape[1])])
    adata.uns[f"{key_added}_names"] = np.asarray(names, dtype=object)
    adata.uns[f"{key_added}_params"] = dict(
        motif_key=motif_key, weights_key=weights_key, gc_key=gc_key,
        covariates=list(covariates), n_bins=[int(b) for b in n_bins], null=null,
        peak_sensitivity=peak_sensitivity, groupby=groupby, dispersion=dispersion,
        shrinkage=shrinkage, score=score, layer=layer, n_strata=int(res["n_strata"]))
    _console(f"done. obsm[{key_added!r}] = {res[score].shape} ({score})", verbose)
    return adata
