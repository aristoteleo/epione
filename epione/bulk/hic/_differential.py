"""Differential TADs between two Hi-C matrices (HiCExplorer ``hicDifferentialTAD``).

Port of ``hicexplorer/hicDifferentialTAD.py`` (deeptools/HiCExplorer 3.7.6,
GPL-3.0, Copyright (c) Fidel Ramírez, Joachim Wolff and contributors) for
``.cool`` inputs. For each TAD, three Wilcoxon rank-sum tests compare target
and control pixels: within the TAD (intra-TAD), between the TAD and its left
neighbour, and between the TAD and its right neighbour.

HiCExplorer splits each chromosome's TADs across ``--threads`` workers and
handles the edges of every chunk differently, so its output depends on the
thread count. This port reproduces ``--threads 1``, including its reuse of the
previous TAD's right-boundary index in the last TAD's left inter-TAD test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ._hicmatrix import HiCMat


def _region(path, chrom, start, end):
    return HiCMat.from_cool(str(path), region=f"{chrom}:{int(start)}-{int(end)}")


def differential_tads(
    target_cool: Union[str, Path],
    control_cool: Union[str, Path],
    domains: Union[pd.DataFrame, str, Path],
    *,
    p_value: float = 0.05,
    mode: str = "all",
    mode_reject: str = "one",
) -> pd.DataFrame:
    """Test every TAD for a change in contacts between two conditions.

    Arguments:
        target_cool, control_cool: ``.cool`` files with the same bins.
        domains: TAD domains -- a DataFrame with ``chrom``, ``start``, ``end``
            (e.g. :func:`tad_domains` output) or a BED file path, sorted.
        p_value: rejection threshold for each test.
        mode: which tests decide rejection: ``"intra-TAD"`` alone,
            ``"left-inter-TAD"`` or ``"right-inter-TAD"`` (each together with
            the intra-TAD test), or ``"all"`` three.
        mode_reject: reject when ``"one"`` of the selected tests rejects, or
            only when ``"all"`` of them do.

    Returns:
        ``DataFrame`` with one row per tested TAD: ``chrom``, ``start``,
        ``end``, the three statistics and p-values (``NaN`` where a neighbour
        is missing), and ``rejected`` (``True`` = differential TAD).
    """
    from scipy.stats import ranksums

    if mode not in ("intra-TAD", "left-inter-TAD", "right-inter-TAD", "all"):
        raise ValueError("mode must be 'intra-TAD', 'left-inter-TAD', 'right-inter-TAD' or 'all'")
    if mode_reject not in ("all", "one"):
        raise ValueError("mode_reject must be 'all' or 'one'")
    if isinstance(domains, (str, Path)):
        dom = pd.read_csv(domains, sep="\t", header=None)
        dom = dom.iloc[:, :6] if dom.shape[1] >= 6 else dom
        rows = dom.values.tolist()
    else:
        rows = domains[["chrom", "start", "end"]].values.tolist()
    per_chrom = []
    prev = None
    for r in rows:
        if prev is None or r[0] != prev:
            per_chrom.append([r])
            prev = r[0]
        else:
            per_chrom[-1].append(r)

    recs = []
    for chrom_list in per_chrom:
        n = len(chrom_list)
        # HiCExplorer recomputes the right-boundary index only when a right
        # neighbour exists, so the last TAD's left inter-TAD slice reuses the
        # index left over from the previous TAD (per chromosome). Reproduced.
        rt = rc = None
        for i, row in enumerate(chrom_list):
            if i - 1 >= 0:
                chromosom, start = chrom_list[i - 1][0], chrom_list[i - 1][1]
            else:
                chromosom, start = chrom_list[i][0], chrom_list[i][1]
            end = chrom_list[i + 1][2] if i + 1 < n else chrom_list[i][2]
            t = _region(target_cool, row[0], row[1], row[2]).matrix.toarray().flatten()
            c = _region(control_cool, row[0], row[1], row[2]).matrix.toarray().flatten()
            ht = _region(target_cool, chromosom, start, end)
            hc = _region(control_cool, chromosom, start, end)
            mt, mc = ht.matrix, hc.matrix
            lt = ht.getRegionBinRange(str(chromosom), row[1], row[1])[0]
            lc = hc.getRegionBinRange(str(chromosom), row[1], row[1])[0]
            olt = olc = 0
            orc = ort = -1
            if i + 1 < n:
                rt = ht.getRegionBinRange(str(chromosom), row[2], row[2])[0]
                rc = hc.getRegionBinRange(str(chromosom), row[2], row[2])[0]
            sl = sr = stl = str_ = None
            if i - 1 >= 0 and i + 1 < n:
                a = mt[olt:lt, lt:rt].toarray().flatten()
                b = mc[olc:lc, lc:rc].toarray().flatten()
                stl, sl = ranksums(a, b)
                a = mt[lt:rt, rt:ort].toarray().flatten()
                b = mc[lc:rc, rc:orc].toarray().flatten()
                str_, sr = ranksums(a, b)
            elif i - 1 < 0 and i + 1 < n:
                a = mt[lt:rt, rt:ort].toarray().flatten()
                b = mc[lc:rc, rc:orc].toarray().flatten()
                str_, sr = ranksums(a, b)
            elif i - 1 > 0 and i + 1 >= n:
                a = mt[olt:lt, lt:rt].toarray().flatten()
                b = mc[olc:lc, lc:rc].toarray().flatten()
                stl, sl = ranksums(a, b)
            st, s = ranksums(t, c)

            def _acc(sig, stat):
                if sig is None or np.isnan(sig):
                    return 0, np.nan, np.nan
                return (1 if sig <= p_value else 0), sig, stat

            al, pl, xl = _acc(sl, stl)
            ar, pr, xr = _acc(sr, str_)
            ai, pi, xi = _acc(s, st)
            recs.append((row[0], int(row[1]), int(row[2]), xl, pl, xr, pr, xi, pi, al, ar, ai))
    out = pd.DataFrame(recs, columns=["chrom", "start", "end",
                                      "stat_left_inter", "pvalue_left_inter",
                                      "stat_right_inter", "pvalue_right_inter",
                                      "stat_intra", "pvalue_intra",
                                      "_acc_left", "_acc_right", "_acc_intra"])
    left = out["_acc_left"].to_numpy() == 1
    right = out["_acc_right"].to_numpy() == 1
    intra = out["_acc_intra"].to_numpy() == 1
    comb = np.logical_and if mode_reject == "all" else np.logical_or
    if mode == "intra-TAD":
        rej = intra
    elif mode == "left-inter-TAD":
        rej = comb(left, intra)
    elif mode == "right-inter-TAD":
        rej = comb(intra, right)
    else:
        rej = comb(comb(left, right), intra)
    out["rejected"] = rej
    return out.drop(columns=["_acc_left", "_acc_right", "_acc_intra"])
