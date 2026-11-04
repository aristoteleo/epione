"""Figures for HINT-ATAC differential footprinting.

``rgt-hint differential`` writes one PDF per motif (the normalised
footprint profile per condition, with a sequence logo) and two summary
plots (TF activity per motif; activity fold change vs p-value). These are
the same views drawn from a :class:`~epione.tl.HintDifferentialResult`,
following ``epione.pl`` conventions: return ``(fig, axes)`` and, with
``show=True``, display and close the figure in a notebook.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ._print_plot import GROUP_COLORS, INK, INK2, MUTED, _style_axis

__all__ = ["plot_hint_differential", "plot_hint_profile"]


def _show(fig, show):
    if show:
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display
            display(fig)
            plt.close(fig)
        except Exception:
            pass


def plot_hint_profile(
    result,
    motifs: Sequence[str],
    *,
    genome=None,
    colors: Optional[Sequence[str]] = None,
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
):
    """Normalised footprint profile of each motif, one line per condition.

    The profile is the bias-corrected (or raw) cut signal summed over all
    binding sites of the motif and normalised by the condition factor and
    the number of sites, centred on the site. The shaded band marks the
    motif; the protection score compares it with the two flanks of the same
    width.

    Parameters
    ----------
    result
        :class:`~epione.tl.HintDifferentialResult`.
    motifs
        Motif names (index of ``result.stats``); a substring is accepted if
        it matches one motif.
    genome
        If given, a base-frequency strip of the sites (what upstream turns
        into a logo) is drawn under each profile.
    """
    import matplotlib.pyplot as plt

    names = []
    for m in ([motifs] if isinstance(motifs, str) else motifs):
        if m in result.profiles:
            names.append(m)
            continue
        hits = [k for k in result.profiles if m.upper() in k.upper()]
        if len(hits) != 1:
            raise KeyError(f"{m!r} matches {len(hits)} motifs")
        names.append(hits[0])
    colors = list(colors or GROUP_COLORS)
    n = len(names)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    rows_per = 2 if genome is not None else 1
    if figsize is None:
        figsize = (3.3 * ncols, (2.6 if genome is None else 3.1) * nrows)
    fig = plt.figure(figsize=figsize)
    hr = ([4, 1] if genome is not None else [1]) * nrows
    gs = fig.add_gridspec(nrows * rows_per, ncols, height_ratios=hr, hspace=0.45, wspace=0.3)
    W = result.window_size
    x = np.arange(-(W // 2), W - W // 2)
    axes = []
    fasta = None
    if genome is not None:
        import pysam
        from ._tn5_bias import _fasta_path
        fasta = pysam.FastaFile(_fasta_path(genome))
    for k, name in enumerate(names):
        r, c = divmod(k, ncols)
        ax = fig.add_subplot(gs[r * rows_per, c])
        L = result.motif_len[name]
        ax.axvspan(-(L // 2), L - L // 2, color="#e1e0d9", lw=0, zorder=0)
        prof = result.profiles[name]
        for i, cond in enumerate(result.conditions):
            ax.plot(x, prof[i], color=colors[i % len(colors)], lw=1.1, label=cond)
        num = int(result.stats.loc[name, "Num"]) if name in result.stats.index else 0
        title = name
        if "TF_Activity" in result.stats.columns:
            title += f"\nactivity {result.stats.loc[name, 'TF_Activity']:+.3f}, n={num}"
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlim(x[0], x[-1])
        _style_axis(ax)
        if c == 0:
            ax.set_ylabel("normalised cuts", fontsize=8, color=INK2)
        if k == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        axes.append(ax)
        if fasta is not None:
            from ._hint import _site_pwm
            lax = fig.add_subplot(gs[r * rows_per + 1, c], sharex=ax)
            pwm = _site_pwm(fasta, result.sites[name], W)
            p = (pwm + 1).div((pwm + 1).sum(axis=1), axis=0)
            ic = 2 + (p * np.log2(p)).sum(axis=1)
            bottom = np.zeros(W)
            for base, col in zip("ACGT", ("#1baf7a", "#2a78d6", "#eda100", "#e34948")):
                h = (p[base] * ic).values
                lax.bar(x, h, bottom=bottom, width=1.0, color=col, lw=0, label=base)
                bottom += h
            lax.set_yticks([])
            _style_axis(lax, left=False)
            lax.set_xlabel("distance to site centre (bp)", fontsize=8, color=INK2)
            axes.append(lax)
        else:
            ax.set_xlabel("distance to site centre (bp)", fontsize=8, color=INK2)
    _show(fig, show)
    return fig, axes


def plot_hint_differential(
    result,
    *,
    fdr: float = 0.05,
    label_top: int = 12,
    figsize: Tuple[float, float] = (9.0, 3.8),
    show: bool = True,
):
    """TF activity change per motif, and its significance.

    Left: every motif's activity difference (second condition minus the
    first), ranked; motifs with ``P_values < fdr`` are coloured by the
    condition they are more active in. Right: log2 fold change of the
    activity score against -log10 p (upstream's volcano plot).
    """
    import matplotlib.pyplot as plt

    st = result.stats
    if "TF_Activity" not in st.columns:
        raise ValueError("TF activity is defined for two conditions")
    c1, c2 = result.conditions[:2]
    col1, col2 = GROUP_COLORS[1], GROUP_COLORS[0]
    s = st.sort_values("TF_Activity")
    sig = s["P_values"] < fdr
    colors = np.where(~sig, "#c3c2b7", np.where(s["TF_Activity"] > 0, col2, col1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]})
    a1.scatter(np.arange(len(s)), s["TF_Activity"], c=colors, s=14, lw=0)
    a1.axhline(0, color=MUTED, lw=0.6)
    lab = s[sig].reindex(s[sig]["TF_Activity"].abs().sort_values(ascending=False).index)[:label_top]
    pos = {n: i for i, n in enumerate(s.index)}
    for name, row in lab.iterrows():
        short = name.split(".", 2)[-1] if name.count(".") >= 2 else name
        a1.annotate(short, (pos[name], row["TF_Activity"]), fontsize=7, color=INK2,
                    xytext=(3, 0), textcoords="offset points", va="center")
    a1.set_xlabel("motifs, ranked", fontsize=8, color=INK2)
    a1.set_ylabel(f"TF activity  ({c1} ← → {c2})", fontsize=8, color=INK2)
    a1.set_xticks([])
    _style_axis(a1)
    with np.errstate(divide="ignore"):
        y = -np.log10(st["P_values"])
    a2.scatter(st["log2FC"], y, s=14, lw=0,
               c=np.where(st["P_values"] < fdr, np.where(st["TF_Activity"] > 0, col2, col1), "#c3c2b7"))
    a2.axhline(-np.log10(fdr), color=MUTED, lw=0.6, ls="--")
    a2.set_xlabel("log2 fold change of activity score", fontsize=8, color=INK2)
    a2.set_ylabel("-log10 p", fontsize=8, color=INK2)
    _style_axis(a2)
    a1.set_title(f"HINT differential footprinting, {c1} vs {c2}", fontsize=9.5, color=INK,
                 loc="left")
    fig.tight_layout()
    _show(fig, show)
    return fig, (a1, a2)
