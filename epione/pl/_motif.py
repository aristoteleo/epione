"""Paper-style motif-enrichment visualisations.

Currently hosts :func:`homer_motif_table`, which renders HOMER's
``findMotifsGenome.pl`` output as the familiar Rank | Logo | TF | P-value
table seen in Wang 2025 Fig 3b (and many ChIP / CUT&RUN papers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _tf_from_motif_name(name: str) -> str:
    """HOMER motif names look like ``OTX2(Homeobox)/Photoreceptors-Otx2-ChIP-Seq/Homer``.
    Return the leading TF identifier, uppercased."""
    import re
    return re.split(r"[(/]", name)[0].strip().upper()


def _load_homer_pwm(motif_file: Path) -> Optional[pd.DataFrame]:
    """Read one ``knownN.motif`` PWM file. Returns an (L, 4) dataframe with
    ACGT columns, or ``None`` when the file is missing."""
    if not motif_file.exists():
        return None
    rows: list[list[float]] = []
    with motif_file.open() as fh:
        fh.readline()  # HOMER header line
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith(">"):
                continue
            rows.append([float(x) for x in ln.split()])
    if not rows:
        return None
    return pd.DataFrame(np.asarray(rows), columns=list("ACGT"))


def homer_motif_table(
    homer_outdir: Union[str, Path],
    *,
    top_n: int = 5,
    collapse_per_tf: bool = True,
    title: str = "Motif enrichment",
    figsize: Optional[Tuple[float, float]] = None,
    width_ratios: Sequence[float] = (0.6, 2.6, 2.0),
    logo_color_scheme: str = "classic",
    logo_ylim: Tuple[float, float] = (0, 2.0),
    column_headers: Sequence[str] = ("Rank", "Logo", "TF            P value"),
    suptitle_fontsize: int = 11,
    row_fontsize: int = 11,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Render HOMER ``knownResults`` as a Rank | Logo | TF | P-value table.

    Consumes the directory that ``findMotifsGenome.pl`` writes (containing
    ``knownResults.txt`` plus a ``knownResults/known<i>.motif`` PWM per
    motif) and produces the paper-style table used throughout ChIP / CUT&RUN
    papers, with one row per top TF.

    Arguments:
        homer_outdir: path that contains ``knownResults.txt`` and the
            ``knownResults/`` sub-folder of ``known{i}.motif`` files.
        top_n: how many rows to show. HOMER sorts by log-p internally; this
            function resorts by ``Log P-value`` (more negative = stronger)
            for safety, then picks the first ``top_n``.
        collapse_per_tf: if True (default), keep only the strongest hit per
            TF identifier so the table shows distinct families rather than
            several near-duplicate rows for the same factor (HOMER often
            ranks many ``OTX2-ChIP`` variants at the top).
        title: figure suptitle.
        figsize: optional ``(width, height)``. When None, height scales
            with ``top_n`` (~0.9 in per row plus margin).
        width_ratios: relative widths of the three columns.
        logo_color_scheme: any logomaker colour scheme (``'classic'``,
            ``'chemistry'``, ``'NajafabadiEtAl2017'``, …).
        logo_ylim: y-axis limits for the information-content logos. The
            conventional cap is 2 bits per position.
        column_headers: override the three column-header strings.
        suptitle_fontsize, row_fontsize: text sizes.

    Returns:
        ``(fig, axes, top)`` — the figure, a ``(top_n, 3)`` axes array, and
        the ``top`` DataFrame with columns ``Motif Name``, ``TF``,
        ``P-value``, ``log10P``.

    Example:
        >>> import epione as epi
        >>> fig, axes, top = epi.pl.homer_motif_table(
        ...     '/tmp/homer_out', top_n=5,
        ...     title='Motif enrichment of 4C OTX2 peaks')
    """
    try:
        import logomaker  # lazy: not every install has it
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "logomaker is required for homer_motif_table; install with "
            "`pip install logomaker`"
        ) from e

    outdir = Path(homer_outdir)
    known_tsv = outdir / "knownResults.txt"
    if not known_tsv.exists():
        raise FileNotFoundError(f"{known_tsv} not found; run HOMER first")

    kr_orig = pd.read_csv(known_tsv, sep="\t")
    kr = kr_orig.copy()
    kr["log10P"] = kr["Log P-value"] / np.log(10)
    kr = kr.sort_values("log10P").reset_index(drop=True)  # most negative first
    kr["TF"] = kr["Motif Name"].apply(_tf_from_motif_name)

    # Preserve mapping from motif name -> HOMER's 1-based rank (which is
    # the file-name index for known{i}.motif) regardless of re-sorting.
    idx_of_name = {n: i for i, n in enumerate(kr_orig["Motif Name"])}

    if collapse_per_tf:
        seen: set[str] = set()
        picks: list[pd.Series] = []
        for _, row in kr.iterrows():
            if row["TF"] in seen:
                continue
            seen.add(row["TF"])
            picks.append(row)
            if len(picks) == top_n:
                break
        top = pd.DataFrame(picks).reset_index(drop=True)
    else:
        top = kr.head(top_n).reset_index(drop=True)

    pwms = [_load_homer_pwm(outdir / "knownResults" / f"known{idx_of_name[r['Motif Name']]+1}.motif")
            for _, r in top.iterrows()]

    # Figure layout.
    n = len(top)
    if figsize is None:
        figsize = (5.5, 0.9 * n + 0.6)
    fig, axes = plt.subplots(
        nrows=n, ncols=3, figsize=figsize,
        gridspec_kw={"width_ratios": list(width_ratios),
                     "hspace": 0.25, "wspace": 0.1},
    )
    if n == 1:
        axes = np.asarray([axes])
    fig.suptitle(title, y=0.98, fontsize=suptitle_fontsize, weight="bold")

    # Column headers (placed above the first row).
    for ax, header in zip(axes[0], column_headers):
        ax.annotate(header, xy=(0.5, 1.18), xycoords="axes fraction",
                    ha="center", va="bottom",
                    fontsize=suptitle_fontsize - 1, weight="bold")

    for i in range(n):
        r = top.iloc[i]
        ax_rank, ax_logo, ax_tf = axes[i]
        for ax in (ax_rank, ax_logo, ax_tf):
            ax.set_axis_off()

        ax_rank.text(0.5, 0.5, str(i + 1),
                     ha="center", va="center", fontsize=row_fontsize + 2)

        pwm = pwms[i]
        if pwm is not None:
            eps = 1e-9
            info = (2 + (pwm * np.log2(pwm + eps)).sum(axis=1)).clip(lower=0)
            lm_df = pwm.multiply(info, axis=0)
            logomaker.Logo(lm_df, ax=ax_logo, color_scheme=logo_color_scheme,
                           show_spines=False, shade_below=0.0,
                           fade_below=0.0, width=0.95)
            ax_logo.set_xticks([]); ax_logo.set_yticks([])
            ax_logo.set_ylim(*logo_ylim)

        ax_tf.text(0.02, 0.5, str(r["TF"]), ha="left", va="center",
                   fontsize=row_fontsize, fontstyle="italic")
        exp_str = f"{int(r['log10P']):,}"
        ax_tf.text(0.98, 0.5, f"$10^{{{exp_str}}}$",
                   ha="right", va="center", fontsize=row_fontsize)

    plt.subplots_adjust(top=0.85, bottom=0.05)
    return fig, axes, top
