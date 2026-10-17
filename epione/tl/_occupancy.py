"""Thermodynamic TF occupancy: a per-site binding score from ATAC-seq.

Which motif matches in open chromatin does a factor actually occupy?
Footprint scores (PRINT, HINT-ATAC, TOBIAS) look for the dip in Tn5 cuts
that a bound protein leaves. On ChIP-seq ground truth that dip turns out to
carry little information beyond accessibility itself: on ENCODE K562 the
plain count of Tn5 insertions within 100 bp of a site ranks ChIP-bound
sites as well as TOBIAS does and far better than PRINT. This module starts
from the physics of binding instead and treats accessibility and motif
affinity as two factors of one occupancy.

Model
-----
**Site occupancy.** A motif site *i* is in one of three states: wrapped
in a nucleosome, free, or bound by the factor. In the grand-canonical
ensemble with a free-energy gain ``ΔG_i`` for binding and chemical
potential ``μ`` (log free-factor concentration) the statistical weights
are ``w_N``, ``1`` and ``q_i = exp((μ − ΔG_i)/kT)`` (Berg & von Hippel
1987; Granek & Clarke 2005; nucleosome competition: Raveh-Sadka *et al.*
2009, Wasson & Hartemink 2009). The probability that the site is free of
nucleosomes -- what ATAC measures -- is ``a_i = (1 + q_i)/(w_N + 1 + q_i)``
and the occupancy factorises::

    P(bound_i) = a_i * q_i / (1 + q_i)        ≈ a_i * q_i     (q_i << 1)

i.e. accessibility gates an ordinary Fermi-Dirac binding curve.

**Affinity from the PWM.** ``−ΔG_i/kT = β s_i + const`` with ``s_i`` the
motif log-odds score in nats (Berg & von Hippel; Stormo & Fields 1998;
Zhao, Granas & Stormo 2009). ``β = 1`` would hold if the matrix were an
equilibrium Boltzmann sample of sites; count-based PWMs overstate energy
differences (Foat, Morozov & Bussemaker 2006; Zhao *et al.* 2009), and
``β`` acts as an inverse effective temperature. Default ``β = 0.5``.

**Accessibility from Tn5.** ``a_i`` is estimated from the insertions within
``±window`` bp of the site centre, with insertions from sub-nucleosomal
fragments (≤ ``short_max`` bp) given extra weight ``short_weight``: those
fragments come from nucleosome-free DNA (the V-plot of Henikoff *et al.*
2011; NucleoATAC, Schep *et al.* 2015), whereas long fragments also span
nucleosomes. ``log a_i = log(1 + n_i + λ n_i^short)`` up to a constant.

**What a ChIP peak sees.** ChIP resolves occupancy only to the size of its
fragments, so the evidence that a factor sits "at" site *i* is the
occupancy summed over the factor's sites within ``±radius`` bp -- the
partition-function sum of the neighbourhood, ``1 − Π_j (1 − P_j) ≈ Σ_j P_j``
in the dilute limit. The score is therefore::

    score_i = log Σ_{j : |c_j − c_i| ≤ radius} exp(log a_j + β s_j)

computed per motif (sites of the same factor only). It equals the log
posterior odds of occupancy of the neighbourhood up to the per-factor
constant ``−μ``, which does not change the ranking of a factor's sites.

There is no per-site fit: the score needs no labels, no training and no
bias model, and runs in one pass over the fragments. The four constants
(``β``, ``window``, ``short_weight``, ``radius``) were chosen on K562
(ENCODE scATAC + 188 K562 ChIP sets) and frozen before evaluation on held-
out cell types; see the Epione paper's Methods for that protocol.

What it does not do
-------------------
It does not look for a footprint. A per-site protection likelihood
(beta-binomial core-vs-flank against a Tn5 bias model, EM-fitted per
factor) was tested during design and did not improve, or worsened, the
ranking of ChIP-bound sites; see :func:`epione.tl.tf_binding_score` (PRINT)
and :func:`epione.tl.multi_scale_footprint_region` for footprints.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "OCCUPANCY_DEFAULTS",
    "OccupancyResult",
    "neighbourhood_logsumexp",
    "occupancy_from_counts",
    "site_insertion_counts",
    "tf_occupancy",
]

#: Constants of the model, frozen on K562 (see module docstring).
OCCUPANCY_DEFAULTS = dict(beta=0.5, window=50, short_weight=2.0, short_max=120,
                          radius=150)


# ---------------------------------------------------------------------------
# Maths
# ---------------------------------------------------------------------------

def neighbourhood_logsumexp(
    keys: Sequence,
    centres: np.ndarray,
    values: np.ndarray,
    radius: int,
) -> np.ndarray:
    """``log Σ_j exp(values_j)`` over sites ``j`` with the same key and
    ``|centre_j − centre_i| ≤ radius``, for every site ``i``.

    Computed exactly per neighbourhood (a gather plus ``reduceat``), not
    as a difference of cumulative sums, which loses every small term next
    to a large one.

    Parameters
    ----------
    keys
        Group label per site (e.g. ``motif|chrom``); only sites with equal
        keys are neighbours.
    centres
        Integer site centres.
    values
        Per-site log weights.
    radius
        Neighbourhood half-width in bp (``0`` returns ``values``).
    """
    values = np.asarray(values, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.int64)
    out = np.empty(len(values), dtype=np.float64)
    if len(values) == 0:
        return out
    if radius <= 0:
        out[:] = values
        return out
    codes, _ = pd.factorize(pd.Series(np.asarray(keys)), sort=False)
    order = np.lexsort((centres, codes))
    k_s, c_s, v_s = codes[order], centres[order], values[order]
    # neighbour bounds within the sorted (key, centre) order
    # key boundaries: search on a combined coordinate key*BIG + centre
    big = int(c_s.max()) + 2 * int(radius) + 2
    comb = k_s.astype(np.int64) * big + c_s
    lo = np.searchsorted(comb, comb - radius, side="left")
    hi = np.searchsorted(comb, comb + radius, side="right")
    cnt = hi - lo
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    j = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(starts, cnt))
    v = v_s[j]
    mx = np.maximum.reduceat(v, starts)
    res = np.log(np.add.reduceat(np.exp(v - np.repeat(mx, cnt)), starts)) + mx
    out[order] = res
    return out


def occupancy_from_counts(
    sites: pd.DataFrame,
    n_all: np.ndarray,
    n_short: np.ndarray,
    *,
    score_key: str = "score",
    motif_key: str = "motif",
    score_base: float = 2.0,
    beta: float = OCCUPANCY_DEFAULTS["beta"],
    short_weight: float = OCCUPANCY_DEFAULTS["short_weight"],
    radius: int = OCCUPANCY_DEFAULTS["radius"],
):
    """Occupancy scores from per-site insertion counts.

    Parameters
    ----------
    sites
        One row per (motif, site): ``chrom``, ``start``, ``end``,
        ``motif_key`` and ``score_key`` (PWM log-odds in log base
        ``score_base``; MOODS / Epione motif scans report log2).
    n_all, n_short
        ``[n_groups, n_sites]`` (or ``[n_sites]``) insertions within
        ``±window`` of each site centre, from all fragments and from
        sub-nucleosomal fragments.

    Returns
    -------
    (score, log_weight)
        Both ``[n_groups, n_sites]``: the neighbourhood score and the
        per-site log weight ``log(1 + n + λ n_short) + β s``.
    """
    n_all = np.atleast_2d(np.asarray(n_all, dtype=np.float64))
    n_short = np.atleast_2d(np.asarray(n_short, dtype=np.float64))
    if n_all.shape != n_short.shape or n_all.shape[1] != len(sites):
        raise ValueError("n_all / n_short must be [n_groups, n_sites] matching sites")
    s_nats = np.asarray(sites[score_key], dtype=np.float64) * np.log(score_base)
    centre = ((np.asarray(sites["start"]) + np.asarray(sites["end"])) // 2).astype(np.int64)
    keys = (pd.Series(np.asarray(sites[motif_key]).astype(str))
            + "|" + pd.Series(np.asarray(sites["chrom"]).astype(str))).to_numpy()
    logw = np.log1p(n_all + short_weight * n_short) + beta * s_nats[None, :]
    score = np.vstack([neighbourhood_logsumexp(keys, centre, lw, radius) for lw in logw])
    return score, logw


# ---------------------------------------------------------------------------
# Counting insertions around sites
# ---------------------------------------------------------------------------

def _merge_windows(chrom: np.ndarray, lo: np.ndarray, hi: np.ndarray, gap: int = 0):
    """Merge per-site windows into disjoint fetch intervals per chromosome."""
    df = pd.DataFrame({"chrom": chrom, "lo": lo, "hi": hi}).sort_values(["chrom", "lo"])
    out = []
    for c, g in df.groupby("chrom", sort=False):
        a, b = g.lo.to_numpy(), g.hi.to_numpy()
        bmax = np.maximum.accumulate(b)
        new = np.concatenate([[True], a[1:] > bmax[:-1] + gap])
        idx = np.flatnonzero(new)
        ends = np.append(idx[1:], len(a)) - 1
        out.append(pd.DataFrame({"chrom": c, "start": a[idx], "end": bmax[ends]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["chrom", "start", "end"])


def _count_chunk(args):
    """Worker: insertion tracks over merged intervals, then window sums."""
    (fragment_file, bc_group, n_groups, intervals, site_chrom, site_centre,
     window, short_max, shift) = args
    import pysam

    tbx = pysam.TabixFile(fragment_file)
    contigs = set(tbx.contigs)
    n_all = np.zeros((n_groups, len(site_centre)), dtype=np.float64)
    n_short = np.zeros_like(n_all)
    by_chrom = pd.Series(np.arange(len(site_centre))).groupby(np.asarray(site_chrom)).indices
    for c, s, e in intervals:
        if c not in contigs:
            continue
        L = e - s
        trk = np.zeros((2, n_groups, L), dtype=np.float64)
        rows, cols, short = [], [], []
        for line in tbx.fetch(c, max(0, s - 1), e):
            p = line.split("\t", 4)
            gi = bc_group.get(p[3])
            if gi is None:
                continue
            a, b = int(p[1]), int(p[2])
            sh = (b - a) <= short_max
            x, y = a + shift[0], b + shift[1]
            if s <= x < e:
                rows.append(gi); cols.append(x - s); short.append(sh)
            if s <= y < e:
                rows.append(gi); cols.append(y - s); short.append(sh)
        if rows:
            rows = np.asarray(rows); cols = np.asarray(cols); short = np.asarray(short)
            np.add.at(trk[0], (rows, cols), 1.0)
            np.add.at(trk[1], (rows[short], cols[short]), 1.0)
        cs = np.concatenate([np.zeros((2, n_groups, 1)), np.cumsum(trk, axis=-1)], axis=-1)
        ix = by_chrom.get(c)
        if ix is None:
            continue
        cen = site_centre[ix]
        m = (cen - window >= s) & (cen + window < e)
        ix, cen = ix[m], cen[m]
        a = cen - window - s
        b = cen + window + 1 - s
        n_all[:, ix] = cs[0][:, b] - cs[0][:, a]
        n_short[:, ix] = cs[1][:, b] - cs[1][:, a]
    return n_all, n_short


def site_insertion_counts(
    fragment_file: str,
    barcodes: Sequence[str],
    groups: Sequence[int],
    n_groups: int,
    chrom: np.ndarray,
    centre: np.ndarray,
    *,
    window: int = OCCUPANCY_DEFAULTS["window"],
    short_max: int = OCCUPANCY_DEFAULTS["short_max"],
    shift=(0, -1),
    n_jobs: int = 1,
    chunk_size: int = 2000,
):
    """Tn5 insertions within ``±window`` bp of each site, per cell group.

    Parameters
    ----------
    fragment_file
        Tabix-indexed fragments; insertions at ``start + shift[0]`` and
        ``end + shift[1]`` (default ``s`` and ``e − 1``, ArchR's
        convention -- a 1-bp convention difference is immaterial for
        100-bp window counts).
    barcodes, groups
        Cell barcodes and their group index (``-1`` = unused cell).
    chrom, centre
        Site coordinates (unique sites; duplicates are fine).
    short_max
        Fragments of at most this length count as sub-nucleosomal.

    Returns
    -------
    (n_all, n_short)
        ``[n_groups, n_sites]`` float arrays.
    """
    chrom = np.asarray(chrom).astype(str)
    centre = np.asarray(centre, dtype=np.int64)
    bc_group = {b: int(g) for b, g in zip(barcodes, groups) if int(g) >= 0}
    iv = _merge_windows(chrom, centre - window, centre + window + 1)
    iv_list = list(iv.itertuples(index=False, name=None))
    chunks = [iv_list[i:i + chunk_size] for i in range(0, len(iv_list), chunk_size)]
    # sites belonging to each chunk (by chromosome and coordinate range)
    jobs = []
    for ch in chunks:
        cs = {}
        for c, s, e in ch:
            cs.setdefault(c, []).append((s, e))
        sel = np.zeros(len(centre), dtype=bool)
        for c, spans in cs.items():
            m = chrom == c
            lo, hi = min(s for s, _ in spans), max(e for _, e in spans)
            sel |= m & (centre - window >= lo) & (centre + window < hi)
        ix = np.flatnonzero(sel)
        jobs.append((ix, (fragment_file, bc_group, n_groups, ch, chrom[ix], centre[ix],
                          int(window), int(short_max), tuple(shift))))
    n_all = np.zeros((n_groups, len(centre)))
    n_short = np.zeros_like(n_all)
    if n_jobs == 1 or len(jobs) == 1:
        results = map(_count_chunk, [j[1] for j in jobs])
    else:
        ex = ProcessPoolExecutor(n_jobs)
        results = ex.map(_count_chunk, [j[1] for j in jobs])
    for (ix, _), (a, s) in zip(jobs, results):
        n_all[:, ix] = a
        n_short[:, ix] = s
    if n_jobs != 1 and len(jobs) > 1:
        ex.shutdown()
    return n_all, n_short


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class OccupancyResult:
    """Per-site TF occupancy scores.

    Attributes
    ----------
    groups
        Cell groups, in row order of :attr:`score`.
    sites
        The input sites (one row per motif x site) with ``center``.
    score
        ``[n_groups, n_sites]`` log neighbourhood occupancy (log posterior
        odds up to a per-motif constant); rank sites of one motif by it.
    log_weight
        ``[n_groups, n_sites]`` per-site term ``log a_i + β s_i``.
    n_all, n_short
        ``[n_groups, n_sites]`` insertions in ``±window`` (all /
        sub-nucleosomal fragments).
    params
        The model constants used.
    """

    groups: List[str]
    sites: pd.DataFrame
    score: np.ndarray
    log_weight: np.ndarray
    n_all: np.ndarray
    n_short: np.ndarray
    params: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Long table: one row per (site, group)."""
        df = self.sites.loc[self.sites.index.repeat(len(self.groups))].copy()
        df["group"] = np.tile(self.groups, len(self.sites))
        df["occupancy_score"] = self.score.T.reshape(-1)
        df["log_weight"] = self.log_weight.T.reshape(-1)
        df["n_insertions"] = self.n_all.T.reshape(-1)
        df["n_short"] = self.n_short.T.reshape(-1)
        return df.reset_index(drop=True)


def tf_occupancy(
    adata,
    sites: pd.DataFrame,
    *,
    groupby: Optional[str] = None,
    motif_key: str = "motif",
    score_key: str = "score",
    score_base: float = 2.0,
    beta: float = OCCUPANCY_DEFAULTS["beta"],
    window: int = OCCUPANCY_DEFAULTS["window"],
    short_weight: float = OCCUPANCY_DEFAULTS["short_weight"],
    short_max: int = OCCUPANCY_DEFAULTS["short_max"],
    radius: int = OCCUPANCY_DEFAULTS["radius"],
    min_cells_per_group: int = 10,
    n_jobs: int = 1,
    verbose: bool = True,
) -> OccupancyResult:
    """Thermodynamic occupancy score for every motif site.

    ``score_i = log Σ_{j ∈ N(i)} (1 + n_j + λ n_j^short) · exp(β s_j)``
    over the same motif's sites within ``radius`` bp: accessibility-gated
    Boltzmann occupancy (module docstring) summed over the neighbourhood a
    ChIP peak resolves.

    Parameters
    ----------
    adata
        AnnData with ``uns['files']['fragments']`` (see
        :func:`epione.pp.import_fragments`).
    sites
        Motif sites, one row per (motif, site): ``chrom``, ``start``,
        ``end``, ``motif_key`` (motif / TF name) and ``score_key`` (PWM
        log-odds, log base ``score_base``). E.g. the output of a MOODS scan.
    groupby
        ``obs`` column for pseudobulks; ``None`` = all cells as one group.
    beta
        Inverse effective temperature of the PWM energy (per nat).
    window, short_weight, short_max
        Accessibility estimate: insertions within ``±window`` bp, with
        fragments ≤ ``short_max`` bp weighted ``1 + short_weight``.
    radius
        Neighbourhood (bp) over which occupancy is summed; ``0`` = the
        site alone.
    n_jobs
        Worker processes for reading fragments.

    Returns
    -------
    :class:`OccupancyResult`

    Examples
    --------
    >>> res = epi.tl.tf_occupancy(adata, motif_hits, motif_key="tf",
    ...                           score_key="pwm_score", n_jobs=8)
    >>> res.to_frame().sort_values("occupancy_score").tail()
    """
    from ..core import console

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise ValueError("adata.uns['files']['fragments'] missing -- run "
                         "epi.pp.import_fragments first")
    st = pd.DataFrame(sites).copy()
    for col in ("chrom", "start", "end", motif_key, score_key):
        if col not in st.columns:
            raise ValueError(f"sites needs a {col!r} column")
    st = st.reset_index(drop=True)
    st["center"] = (st["start"].astype(np.int64) + st["end"].astype(np.int64)) // 2

    if groupby is None:
        names = ["all"]
        gidx = np.zeros(adata.n_obs, dtype=int)
    else:
        labels = adata.obs[groupby].astype(str).to_numpy()
        vc = pd.Series(labels).value_counts()
        names = sorted(g for g, n in vc.items() if n >= min_cells_per_group)
        if not names:
            raise ValueError(f"no {groupby!r} group has >= {min_cells_per_group} cells")
        pos = {g: i for i, g in enumerate(names)}
        gidx = np.array([pos.get(x, -1) for x in labels])

    # count once per unique (chrom, centre)
    key = st["chrom"].astype(str) + ":" + st["center"].astype(str)
    uniq, inv = np.unique(key.to_numpy(), return_inverse=True)
    first = pd.Series(np.arange(len(st))).groupby(inv).first().to_numpy()
    n_all_u, n_short_u = site_insertion_counts(
        adata.uns["files"]["fragments"], list(map(str, adata.obs_names)), gidx,
        len(names), st["chrom"].to_numpy()[first], st["center"].to_numpy()[first],
        window=window, short_max=short_max, n_jobs=n_jobs)
    n_all, n_short = n_all_u[:, inv], n_short_u[:, inv]

    score, logw = occupancy_from_counts(
        st, n_all, n_short, score_key=score_key, motif_key=motif_key,
        score_base=score_base, beta=beta, short_weight=short_weight, radius=radius)
    if verbose:
        console.level1(
            f"tf_occupancy: {len(st):,} sites ({len(uniq):,} unique), "
            f"{st[motif_key].nunique()} motifs x {len(names)} groups")
    return OccupancyResult(
        groups=names, sites=st, score=score, log_weight=logw,
        n_all=n_all, n_short=n_short,
        params=dict(beta=beta, window=window, short_weight=short_weight,
                    short_max=short_max, radius=radius, score_base=score_base),
    )
