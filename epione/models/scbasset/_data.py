"""scBasset inputs: peak sequences, binary targets and the peak split.

Follows calico/scBasset (Apache-2.0) ``scbasset/utils.py``:
``make_bed_seqs_from_df`` (1,344 bp centred on the peak midpoint, N-padded
past chromosome ends), ``dna_1hot_2vec`` (A/C/G/T -> 0/1/2/3, any other base
drawn uniformly from 0-3) and ``split_train_test_val`` (90/5/5, seed 10).
"""
from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEQ_LEN = 1344

_LUT = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i


def _fasta(genome):
    """A ``pyfaidx.Fasta`` from a path, a ``pyfaidx.Fasta`` or an
    :class:`epione.core.genome.Genome` (``epi.data.hg38`` and friends)."""
    import pyfaidx

    fasta = getattr(genome, "fasta", genome)
    if callable(fasta):
        fasta = fasta()
    if isinstance(fasta, pyfaidx.Fasta):
        return fasta
    return pyfaidx.Fasta(str(fasta), as_raw=True, sequence_always_upper=True)


_COORD = re.compile(r"^(.+?)[:_-](\d+)[-_](\d+)$")


def peak_table(var: pd.DataFrame, chrom_key: Optional[str] = None, start_key: Optional[str] = None,
               end_key: Optional[str] = None) -> pd.DataFrame:
    """``chr``/``start``/``end`` of every peak.

    Uses the given ``var`` columns, else the first present of
    ``chr``/``chrom``/``seqnames`` (+ ``start``/``chromStart``, ``end``/``chromEnd``),
    else parses ``var_names`` such as ``chr1:100-600`` or ``chr1-100-600``.
    """
    def pick(given, names):
        if given is not None:
            return given
        for n in names:
            if n in var.columns:
                return n
        return None

    ck = pick(chrom_key, ("chr", "chrom", "seqnames", "chromosome"))
    sk = pick(start_key, ("start", "chromStart"))
    ek = pick(end_key, ("end", "chromEnd"))
    if ck is not None and sk is not None and ek is not None:
        return pd.DataFrame({"chr": var[ck].astype(str).to_numpy(),
                             "start": var[sk].astype(np.int64).to_numpy(),
                             "end": var[ek].astype(np.int64).to_numpy()})
    rows = []
    for name in var.index.astype(str):
        m = _COORD.match(name)
        if m is None:
            raise ValueError(f"cannot read peak coordinates from var name {name!r}; "
                             "pass chrom_key/start_key/end_key")
        rows.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return pd.DataFrame(rows, columns=["chr", "start", "end"])


def peak_sequences(peaks: pd.DataFrame, genome, seq_len: int = SEQ_LEN, seed: int = 0,
                   return_n_mask: bool = False):
    """Integer-coded sequences ``(n_peaks, seq_len)`` int8, as scBasset's preprocessing.

    The window starts at ``(start + end) // 2 - seq_len // 2``; positions past
    either chromosome end are ``N``. Bases other than A/C/G/T get a uniform
    random code (scBasset draws them with Python's unseeded ``random``; here
    a ``numpy`` generator seeded with ``seed``).

    Returns:
        codes, and with ``return_n_mask=True`` also the boolean mask of the
        randomised (non-ACGT) positions.
    """
    fa = _fasta(genome)
    n = len(peaks)
    codes = np.empty((n, seq_len), dtype=np.int8)
    chrom = peaks["chr"].astype(str).to_numpy()
    mid = (peaks["start"].to_numpy(np.int64) + peaks["end"].to_numpy(np.int64)) // 2
    s0 = mid - seq_len // 2
    for i in range(n):
        start, end = int(s0[i]), int(s0[i]) + seq_len
        rec = fa[chrom[i]]
        L = len(rec)
        left = max(0, -start)
        seq = "N" * left + str(rec[max(0, start):min(end, L)])
        seq = seq + "N" * (seq_len - len(seq))
        codes[i] = _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    nmask = codes == 4
    if nmask.any():
        rng = np.random.default_rng(seed)
        codes[nmask] = rng.integers(0, 4, size=int(nmask.sum()), dtype=np.int8)
    return (codes, nmask) if return_n_mask else codes


def encode_sequences(seqs: Sequence[str], seq_len: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """Integer codes for a list of equal-length sequences (non-ACGT random, seeded)."""
    seqs = [s.upper() for s in seqs]
    L = seq_len or len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("all sequences must have the same length")
    codes = _LUT[np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)].reshape(len(seqs), L).copy()
    nmask = codes == 4
    if nmask.any():
        codes[nmask] = np.random.default_rng(seed).integers(0, 4, size=int(nmask.sum()), dtype=np.int8)
    return codes


def binarize(X) -> sp.csr_matrix:
    """Peaks x cells binary CSR (``uint8``) from a cells x peaks count matrix.

    scBasset's data generator sets a target to 1 for every stored entry; stored
    zeros are dropped first so only non-zero counts are positive.
    """
    m = sp.csr_matrix(X, copy=True)
    m.eliminate_zeros()
    m = m.T.tocsr()
    m.data = np.ones_like(m.data, dtype=np.uint8)
    m.sort_indices()
    return m.astype(np.uint8)


def split_train_test_val(n: int, seed: int = 10, train_ratio: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(train, test, val)`` peak indices, identical to scBasset's
    ``split_train_test_val(np.arange(n))`` (legacy ``np.random`` stream)."""
    ids = np.arange(n)
    rs = np.random.RandomState(seed)
    test_val = rs.choice(ids, int(len(ids) * (1 - train_ratio)), replace=False)
    train = np.setdiff1d(ids, test_val)
    val = rs.choice(test_val, int(len(test_val) / 2), replace=False)
    test = np.setdiff1d(test_val, val)
    return train, test, val


def read_fasta(path) -> list:
    """Sequences of a FASTA file, in order."""
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs
