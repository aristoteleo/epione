"""Multi-scale footprint dispersion score (scPrinter-inspired).

For each (scale, position) around a motif centre, run an *edge-vs-centre
binomial test* on the aggregated Tn5 insertion track:

.. math::

    n_{\\text{centre}}(s, p) &= \\sum_{i \\in [p - s/2,\\ p + s/2]} \\text{signal}_g[i] \\\\
    n_{\\text{edge}}(s, p)   &= \\sum_{i \\in [p - s,\\ p - s/2) \\cup (p + s/2,\\ p + s]} \\text{signal}_g[i] \\\\
    \\pi_0(s, p)             &= \\frac{\\sum \\text{bias}_{\\text{centre}}}{\\sum \\text{bias}_{\\text{centre}} + \\sum \\text{bias}_{\\text{edge}}} \\\\
    F(s, p)                  &= -\\log_{10}\\,P\\!\\left(X \\leq n_{\\text{centre}} \\,\\middle|\\, X \\sim \\text{Binomial}(n_{\\text{total}}, \\pi_0)\\right)

High :math:`F` = centre is depleted relative to bias expectation → the
site looks protected (i.e. a footprint).

Output shape ``(n_groups, n_scales, n_positions)`` — same layout as
scPrinter's ``footprintsadata.obsm[key]``, ready for a seaborn heatmap.

This uses the **aggregate** cut track from :func:`epione.tl.get_footprints`
and the **aggregate** hexamer-bias track. That makes it statistically
equivalent to running a per-site test and then summing the cut/bias
vectors — all linear ops commute with the aggregation. The non-linearity
only enters in the final :math:`-\\log_{10}(\\cdot)`, applied once at the
end. This matches scPrinter's per-pseudobulk scoring approach; the
remaining fidelity gap is the k-mer-vs-CNN bias model and the null
calibration (see ``null=`` below).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ._footprint import Footprint


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultiScaleFootprint:
    """Output of :func:`multi_scale_footprint`.

    Attributes
    ----------
    motif
        Motif name inherited from the source ``Footprint``.
    groups
        Group labels (same order as ``dispersion``'s first axis).
    scales
        1D array of scale sizes (bp).
    positions
        1D array of genomic offsets — ``-flank..flank``.
    dispersion
        ``(n_groups, n_scales, n_positions)`` array. Positive values
        indicate centre depletion (footprint); negative values indicate
        centre enrichment.
    raw_F
        ``(n_groups, n_scales, n_positions)`` un-null-corrected
        ``-log10 P(X ≤ n_centre)`` — useful for debugging the bias model.
    n_sites
        Sites that went into the aggregate signal (per group).
    flank
        Half-window around motif centre (bp).
    null
        Short name of the null-correction strategy used.
    """
    motif: str
    groups: List[str]
    scales: np.ndarray
    positions: np.ndarray
    dispersion: np.ndarray
    raw_F: np.ndarray
    n_sites: Dict[str, int]
    flank: int
    null: str


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _sliding_sum(x: np.ndarray, width: int) -> np.ndarray:
    """Rolling window sum of ``x`` with output length equal to
    ``len(x)`` — centred, zero-padded at the edges.

    Equivalent to ``np.convolve(x, np.ones(width), mode='same')`` but
    faster via ``cumsum``.
    """
    x = np.asarray(x, dtype=np.float64)
    if width <= 1:
        return x.copy()
    pad_left = width // 2
    pad_right = width - pad_left - 1
    xp = np.concatenate([np.zeros(pad_left), x, np.zeros(pad_right)])
    c = np.cumsum(xp, dtype=np.float64)
    # sum over window ending at i = c[i + width - 1] - c[i - 1]
    return c[width - 1:] - np.concatenate([[0.0], c[:-width]])


def _edge_center_sums(
    track: np.ndarray, scale: int, edge_factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """For each position p, return
    ``(centre_sum, edge_sum, centre_width, edge_width)`` where

    centre = track[p - s/2 : p + s/2]                 width = s
    edge   = track[p - ef·s/2 : p - s/2) ∪
             (p + s/2 : p + ef·s/2]                    width ≈ (ef - 1) · s

    ``edge_factor`` controls how wide the edge windows are relative to
    the centre. 3.0 is the scPrinter-like default: centre is a narrow
    TF-scale dip; edge is the broader accessibility dome flanking it.
    ``edge_factor = 2.0`` gives symmetric half-half windows (used by
    TOBIAS-style tests).
    """
    track = np.asarray(track, dtype=np.float64)
    s = int(scale)
    if s < 2:
        raise ValueError(f"scale must be >= 2, got {s}")

    centre_w = s
    total_w = int(round(edge_factor * s))
    # Ensure total_w > centre_w so edge has positive width.
    total_w = max(total_w, centre_w + 2)
    if total_w % 2 == 0:
        total_w += 1   # keep it odd so it's symmetric around p

    total = _sliding_sum(track, total_w)
    centre = _sliding_sum(track, centre_w)
    edge = total - centre
    edge_w = total_w - centre_w
    return centre, edge, centre_w, edge_w


def multi_scale_footprint(
    footprint: Union[Footprint, Dict[str, Footprint]],
    *,
    motif: Optional[str] = None,
    scales: Sequence[int] = tuple(range(5, 101, 5)),
    statistic: str = "amplitude",
    null: str = "outer_flank",
    outer_flank_fraction: float = 0.2,
    min_total_insertions: int = 5,
    signed: bool = True,
    pseudocount: float = 1.0,
    edge_factor: float = 3.0,
) -> MultiScaleFootprint:
    """Multi-scale edge-vs-centre binomial dispersion score.

    Parameters
    ----------
    footprint
        A single :class:`Footprint` or a ``{motif: Footprint}`` mapping
        (in which case ``motif=`` disambiguates).
    scales
        Scale sizes (bp). Small scales (5–30 bp) capture single-TF
        protection, medium (30–80) TF complex / half-nucleosome, large
        (>80) nucleosome.
    statistic
        Scoring function for the centre-vs-edge contrast:

        * ``'amplitude'`` *(default)* —
          ``edge_mean(normalizedSignal) − centre_mean(normalizedSignal)``.
          Operates on the already bias-corrected, per-group
          flank-normalised signal, so (a) per-group amplitude
          differences (e.g. Erythroid >> Mono for GATA1) are
          preserved, (b) the shared hexamer bias is cancelled, and
          (c) centre-vs-edge contrast picks up the footprint shape.
          Positive = centre depleted (footprint).
        * ``'log2_ratio'`` — ``-log2((n_centre + ε) /
          (n_total · π₀ + ε))``. Scale-invariant across cell counts;
          measures *shape* only (amplitude cancels). Use this when
          you want to ask "does the bias-corrected shape look like a
          footprint?" independent of how many cells you have.
        * ``'log2_odds'`` — ``-log2((n_centre / n_edge + ε) /
          (π₀ / (1-π₀) + ε))``; shape-only variant closer to the
          standard TOBIAS footprint score.
        * ``'pvalue_log10'`` — ``-log10 P(X ≤ n_centre | Binom(n_total,
          π₀))``. The proper statistical-test flavour, but scales with
          ``sqrt(n_total)`` so cell-count imbalance dominates the
          colour scale — use only when group sizes are matched.

    null
        Null-correction strategy for the raw :math:`F` score:

        * ``'none'``        — return raw :math:`F`.
        * ``'outer_flank'`` — subtract, per (group, scale), the mean
          :math:`F` over the outer-most ``outer_flank_fraction`` of the
          position window. Assumes no TF binding at the window edges,
          which is correct for a properly sized ``flank`` (default
          ±250 bp around the motif, a range where TF occupancy is
          already back to background).
    outer_flank_fraction
        Fraction of the position axis (on each side) used as the local
        null when ``null='outer_flank'``. Default 0.2 means the outer
        20 % on each side → 40 % of the window informs the baseline.
    min_total_insertions
        Positions where ``n_total < this`` are set to 0 dispersion
        (underpowered binomial test; prevents noisy points from
        dominating the heatmap).
    signed
        If ``True`` (default), flip the sign of positions where the
        centre is *enriched* (not depleted) relative to bias. Result
        is then ``+`` for footprint, ``-`` for anti-footprint
        (e.g. pioneer-TF-dependent nucleosome eviction).
        If ``False``, always return the raw ``-log10 binom.cdf`` value.

    Returns
    -------
    :class:`MultiScaleFootprint`
        With ``.dispersion[group, scale, position]`` ready to plot.
    """
    from scipy.stats import binom

    # Resolve a single Footprint.
    if isinstance(footprint, Footprint):
        fp = footprint
    else:
        if motif is None:
            if len(footprint) == 1:
                motif = next(iter(footprint))
            else:
                raise ValueError(f"pass motif= one of {list(footprint)}")
        fp = footprint[motif]

    if fp.Tn5Bias is None:
        raise ValueError(
            "Footprint has no Tn5Bias track — recompute with "
            "normalize != 'None' and a genome/bias_table"
        )

    # ``fp.signal`` is the per-site MEAN (counts / n_sites) — multiply
    # back by n_sites to get aggregate counts that the binomial test needs.
    sig_mean = np.asarray(fp.signal, dtype=np.float64)        # (G, W)
    n_sites_arr = np.asarray([fp.n_sites[g] for g in fp.groups],
                              dtype=np.float64)[:, None]
    sig = sig_mean * n_sites_arr                              # aggregate counts (G, W)
    bias = np.asarray(fp.Tn5Bias, dtype=np.float64)           # (W,)

    # For the 'amplitude' statistic we use fp.normalizedSignal directly
    # (already bias-subtracted + per-group flank-normalised).
    norm_sig = (np.asarray(fp.normalizedSignal, dtype=np.float64)
                if fp.normalizedSignal is not None else None)

    n_groups, window = sig.shape
    scales_arr = np.asarray(list(scales), dtype=np.int32)
    n_scales = len(scales_arr)

    raw_F = np.zeros((n_groups, n_scales, window), dtype=np.float64)
    # Mask — True where the full scale window fits inside ±flank.
    pos_axis = fp.positions.astype(np.int64)

    for si, s in enumerate(scales_arr):
        s = int(s)
        # Total half-window extent for range-validity check.
        half_w = int(round(edge_factor * s)) // 2
        valid_pos = np.abs(pos_axis) + half_w <= fp.flank

        # Centre / edge sums vectorised over the full position axis.
        per_group = [_edge_center_sums(sig[g], s, edge_factor=edge_factor)
                     for g in range(n_groups)]
        sig_centre = np.stack([x[0] for x in per_group])
        sig_edge = np.stack([x[1] for x in per_group])
        centre_w = per_group[0][2]
        edge_w = per_group[0][3]
        n_total = sig_centre + sig_edge

        bias_centre, bias_edge, _, _ = _edge_center_sums(
            bias, s, edge_factor=edge_factor
        )
        bias_total = bias_centre + bias_edge
        with np.errstate(divide="ignore", invalid="ignore"):
            pi0 = np.where(bias_total > 0,
                            bias_centre / np.maximum(bias_total, 1e-12),
                            0.5)
        pi0 = np.clip(pi0, 1e-6, 1.0 - 1e-6)
        pi_b = np.broadcast_to(pi0, n_total.shape)

        if statistic == "amplitude":
            if norm_sig is None:
                raise ValueError(
                    "statistic='amplitude' requires fp.normalizedSignal; "
                    "recompute get_footprints with normalize='Subtract' "
                    "and a genome/bias_table"
                )
            # Centre / edge MEANS (not sums) of the already bias-
            # corrected, flank-normalised signal. Edge − centre so
            # positive values = centre depletion = footprint.
            per_group_ns = [_edge_center_sums(norm_sig[g], s,
                                               edge_factor=edge_factor)
                             for g in range(n_groups)]
            ns_centre = np.stack([x[0] for x in per_group_ns]) / centre_w
            ns_edge = np.stack([x[1] for x in per_group_ns]) / edge_w
            F = ns_edge - ns_centre

        elif statistic == "log2_ratio":
            # -log2( (n_c + eps) / (n_total * pi0 + eps) )
            #   > 0 → centre depleted vs bias expectation → footprint
            exp_centre = n_total * pi_b
            with np.errstate(divide="ignore", invalid="ignore"):
                F = -np.log2((sig_centre + pseudocount) /
                              (exp_centre + pseudocount))
            # For this statistic, "signed" is already baked in — always
            # positive for footprints, negative for anti-footprints.
            # The `signed` flag becomes a no-op.

        elif statistic == "log2_odds":
            # -log2( (n_c / n_e + eps) / (pi0 / (1-pi0) + eps) )
            #   > 0 → centre depleted vs bias-predicted odds → footprint
            with np.errstate(divide="ignore", invalid="ignore"):
                obs_odds = (sig_centre + pseudocount) / (sig_edge + pseudocount)
                exp_odds = pi_b / (1.0 - pi_b)
                F = -np.log2(obs_odds / np.maximum(exp_odds, 1e-12))

        elif statistic == "pvalue_log10":
            # Binomial P(X <= n_c). Integer n_total required by scipy.
            n_tot_int = np.rint(n_total).astype(np.int64)
            k_obs = np.rint(sig_centre).astype(np.int64)
            k_obs = np.minimum(k_obs, n_tot_int)
            with np.errstate(divide="ignore", invalid="ignore"):
                cdf = binom.cdf(k_obs, n_tot_int, pi_b)
                cdf = np.clip(cdf, 1e-300, 1.0)
                F = -np.log10(cdf)
            if signed:
                expected_centre = n_total * pi0
                F = F * np.where(sig_centre <= expected_centre, 1.0, -1.0)
        else:
            raise ValueError(f"unknown statistic: {statistic!r}")

        # Mask low-count and out-of-range positions.
        low_power = n_total < min_total_insertions
        out_of_range = ~valid_pos[None, :]
        F = np.where(low_power | out_of_range, 0.0, F)
        raw_F[:, si, :] = F

    # ---- null correction ----
    if null == "none":
        dispersion = raw_F.copy()
    elif null == "outer_flank":
        # Per (group, scale), subtract mean F in the outermost
        # `outer_flank_fraction` of positions on each side. These
        # regions are far from the motif centre and should have no
        # systematic footprint signal — their mean F is the local
        # null expectation.
        k = max(int(window * outer_flank_fraction), 1)
        left = slice(0, k)
        right = slice(window - k, window)
        null_F = 0.5 * (raw_F[:, :, left].mean(axis=-1) +
                        raw_F[:, :, right].mean(axis=-1))
        dispersion = raw_F - null_F[..., None]
    else:
        raise ValueError(f"unknown null strategy: {null!r}")

    return MultiScaleFootprint(
        motif=fp.motif,
        groups=list(fp.groups),
        scales=scales_arr,
        positions=fp.positions.copy(),
        dispersion=dispersion.astype(np.float32),
        raw_F=raw_F.astype(np.float32),
        n_sites=dict(fp.n_sites),
        flank=fp.flank,
        null=null,
    )


def footprint_score(
    msfp: MultiScaleFootprint,
    *,
    position_radius: int = 5,
    scale_range: tuple = (20, 40),
) -> Dict[str, float]:
    """Mean dispersion in the TF-protection zone — one scalar per group.

    Higher = stronger footprint. Use to rank cell types by TF binding:
    canonical lineages (e.g. Erythroid for GATA1) land at the top.

    Parameters
    ----------
    msfp
        Output of :func:`multi_scale_footprint`.
    position_radius
        Half-width (bp) of the position zone centred on the motif.
        Default 5 = ±5 bp around the PWM centre, tight enough to
        capture single-TF protection.
    scale_range
        ``(min_scale, max_scale)`` in bp — the scales where a
        single TF footprint is expected to dominate. Default (20, 40)
        avoids very fine scales (dominated by sequence bias residuals)
        and very coarse scales (dominated by nucleosome structure).

    Returns
    -------
    ``{group: score}`` dictionary. Ordering preserves the group axis
    of ``msfp`` (not sorted by score).
    """
    lo, hi = scale_range
    pos_mask = np.abs(msfp.positions) <= position_radius
    scale_mask = (msfp.scales >= lo) & (msfp.scales <= hi)
    out = {}
    for i, g in enumerate(msfp.groups):
        zone = msfp.dispersion[i][scale_mask[:, None] & pos_mask[None, :]]
        out[g] = float(np.nanmean(zone))
    return out


__all__ = ["multi_scale_footprint", "MultiScaleFootprint", "footprint_score"]
