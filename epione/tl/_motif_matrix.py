"""Peak × motif binary matrix via MOODS PWM scan.

Port of ArchR ``addMotifAnnotations`` (which wraps Bioconductor's
``motifmatchr``). For each peak, scan the sequence in its window against
a collection of position weight matrices (PWMs) and record a hit when the
max log-odds score exceeds the p-value threshold. Output is a sparse
binary ``(n_peaks, n_motifs)`` matrix stored in ``adata.varm``.
"""
from __future__ import annotations

import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [motif_matrix] {msg}", flush=True)


def _parse_peak_coords(
    var_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse peak names of form ``chrom:start-end`` (also accepts
    ``chrom-start-end`` or ``chrom_start_end``)."""
    import re
    chroms, starts, ends = [], [], []
    pat = re.compile(r"(?P<chrom>[^\s:_-]+(?:[._][^\s:_-]+)?)[:_-](?P<s>\d+)[-_](?P<e>\d+)$")
    for n in var_names:
        m = pat.match(str(n))
        if m is None:
            raise ValueError(
                f"cannot parse peak name {n!r}; expected 'chrom:start-end'."
            )
        chroms.append(m.group("chrom"))
        starts.append(int(m.group("s")))
        ends.append(int(m.group("e")))
    return (np.array(chroms, dtype=object),
            np.array(starts, dtype=np.int64),
            np.array(ends, dtype=np.int64))


def _fetch_peak_sequences(
    fasta_path: str,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    window: Optional[int] = None,
) -> List[bytes]:
    """Fetch upper-case DNA bytes for each peak (or ``window``-bp window
    centered on each peak). Out-of-bounds is clipped."""
    import pyfaidx
    fa = pyfaidx.Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    chrom_len = {k: len(fa[k]) for k in fa.keys()}
    seqs: List[bytes] = []
    miss = 0
    for c, s, e in zip(chroms, starts, ends):
        c = str(c)
        if c not in chrom_len:
            seqs.append(b"")
            miss += 1
            continue
        L = chrom_len[c]
        if window is None:
            lo, hi = int(s), int(e)
        else:
            cx = (int(s) + int(e)) // 2
            lo = cx - window // 2
            hi = lo + window
        lo = max(0, lo)
        hi = min(L, hi)
        s = fa[c][lo:hi]
        # pyfaidx(as_raw=True) returns a str; encode to ASCII bytes.
        if isinstance(s, str):
            seqs.append(s.upper().encode("ascii", errors="ignore"))
        else:
            seqs.append(bytes(s).upper())
    if miss:
        _console(f"{miss:,}/{len(chroms):,} peaks had unknown chrom; stored empty sequences")
    return seqs


def _pwms_from_jaspar(
    release: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: Sequence[str] = ("vertebrates",),
) -> Dict[str, np.ndarray]:
    """Return ``{name: 4xL probability matrix}`` with rows ACGT."""
    from pyjaspar import jaspardb
    jdb = jaspardb(release=release)
    motifs = jdb.fetch_motifs(collection=collection, tax_group=list(tax_group))
    out: Dict[str, np.ndarray] = {}
    for m in motifs:
        counts = m.counts
        arr = np.array([list(counts["A"]), list(counts["C"]),
                         list(counts["G"]), list(counts["T"])], dtype=np.float64)
        arr = arr / arr.sum(axis=0, keepdims=True)     # columns sum to 1
        # JASPAR IDs like "MA0079.4"; attach TF name
        tf_name = getattr(m, "name", m.matrix_id)
        out[f"{m.matrix_id}_{tf_name}"] = arr
    return out


def _build_moods_matrices_and_thresholds(
    pwms: Dict[str, np.ndarray],
    bg: Sequence[float],
    pvalue: float,
    pseudo: float = 0.8,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Convert probability matrices → log-odds + p-value threshold per motif."""
    import MOODS.tools as tools
    mats = []
    thrs = []
    lbg = np.log(np.asarray(bg, dtype=np.float64))
    for name, pm in pwms.items():
        # log-odds with pseudo-count
        pm_eff = (pm + pseudo / 4.0) / (1.0 + pseudo)
        mat = np.log(pm_eff) - lbg[:, None]           # 4 × L
        mats.append(mat)
        thrs.append(tools.threshold_from_p(mat.tolist(), list(bg), pvalue))
    return mats, np.asarray(thrs, dtype=np.float64)


def add_motif_matrix(
    adata: AnnData,
    genome_fasta: str,
    pwms: Optional[Dict[str, np.ndarray]] = None,
    *,
    motif_db: str = "JASPAR2024",
    motif_collection: str = "CORE",
    motif_tax_group: Sequence[str] = ("vertebrates",),
    window: Optional[int] = None,
    pvalue: float = 5e-5,
    bg: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    key_added: str = "motif",
    verbose: bool = True,
) -> AnnData:
    """Scan peaks against PWMs; store peak × motif binary matrix.

    Parameters
    ----------
    adata
        Peak-matrix ``AnnData`` where ``var_names`` are ``chrom:start-end``.
    genome_fasta
        Path to a reference FASTA matching the peak coordinates (e.g.
        hg19.fa, GRCh38.fa).
    pwms
        Mapping ``name → 4×L probability matrix`` (rows ACGT, columns
        sum to 1). If ``None``, motifs are fetched from JASPAR via
        :mod:`pyjaspar`.
    window
        If given, scan a ``window``-bp window centered on each peak.
        Otherwise use the peak interval as-is.
    pvalue
        Per-motif log-odds p-value threshold passed to MOODS
        (``5e-5`` is chromVAR / motifmatchr default).
    key_added
        Prefix under which results are stored:
        - ``adata.varm[key_added]`` — ``scipy.sparse.csr_matrix`` of
          shape ``(n_peaks, n_motifs)``, dtype bool.
        - ``adata.uns[key_added + '_names']`` — motif names in column
          order.
    """
    import MOODS.scan
    if pwms is None:
        _console(f"fetching motifs: {motif_db}/{motif_collection}/{motif_tax_group}", verbose)
        pwms = _pwms_from_jaspar(motif_db, motif_collection, motif_tax_group)
    motif_names = list(pwms.keys())
    n_motifs = len(motif_names)
    _console(f"{n_motifs:,} motifs", verbose)

    chroms, starts, ends = _parse_peak_coords(adata.var_names)
    _console(f"fetching sequences for {len(chroms):,} peaks", verbose)
    seqs = _fetch_peak_sequences(genome_fasta, chroms, starts, ends, window=window)

    _console(f"building log-odds + thresholds (pvalue={pvalue:.0e})", verbose)
    mats, thrs = _build_moods_matrices_and_thresholds(pwms, bg, pvalue)

    scanner = MOODS.scan.Scanner(7)
    scanner.set_motifs([m.tolist() for m in mats], list(bg), list(thrs))

    _console(f"scanning {len(seqs):,} peak sequences", verbose)
    rows: List[int] = []
    cols: List[int] = []
    log_every = max(1, len(seqs) // 20)
    for i, seq in enumerate(seqs):
        if not seq:
            continue
        # MOODS returns a list of lists (one per motif) of Match objects.
        hits = scanner.scan(seq.decode("ascii", errors="ignore"))
        for j, ms in enumerate(hits):
            if ms:
                rows.append(i)
                cols.append(j)
        if verbose and (i + 1) % log_every == 0:
            _console(f"  scanned {i + 1:,} / {len(seqs):,}", verbose)

    # Deduplicate (a motif can hit a peak multiple times, we keep binary)
    rows_arr = np.asarray(rows, dtype=np.int32)
    cols_arr = np.asarray(cols, dtype=np.int32)
    data = np.ones(len(rows_arr), dtype=np.bool_)
    M = sp.coo_matrix(
        (data, (rows_arr, cols_arr)),
        shape=(adata.n_vars, n_motifs),
    ).tocsr()
    M.sum_duplicates()
    # Binarize (duplicates summed → anything >=1 becomes True)
    M = (M > 0).astype(np.bool_)

    adata.varm[key_added] = M
    adata.uns[f"{key_added}_names"] = np.asarray(motif_names, dtype=object)
    adata.uns[f"{key_added}_params"] = dict(
        db=motif_db, collection=motif_collection,
        tax_group=list(motif_tax_group),
        pvalue=float(pvalue), window=int(window or 0), bg=list(map(float, bg)),
    )

    hits_per_motif = np.asarray(M.sum(axis=0)).ravel()
    _console(
        f"{M.nnz:,} (peak, motif) hits | median hits per motif: "
        f"{int(np.median(hits_per_motif)):,}",
        verbose,
    )
    return adata
