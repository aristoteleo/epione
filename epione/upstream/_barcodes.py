"""Cell-barcode handling for single-cell ATAC FASTQs.

Pure Python / numpy, no aligner needed:

* :func:`load_whitelist` reads a barcode whitelist (plain or gzipped, one
  barcode per line; extra columns and a ``-1`` GEM-well suffix are ignored).
* :func:`detect_barcode_orientation` decides whether the barcode read has to
  be reverse-complemented to match the whitelist. The i5 index read (the 10x
  ATAC barcode) comes out in either orientation depending on the sequencer's
  i5 workflow, so this is checked on the data rather than assumed.
* :class:`BarcodeCorrector` corrects raw barcodes against the whitelist with
  the scheme 10x Genomics describes for Cell Ranger (ATAC): a barcode that is
  on the whitelist is kept; otherwise every whitelist barcode at Hamming
  distance 1 is a candidate, weighted by a prior (how often that whitelist
  barcode was seen exactly in the whole run, plus a pseudocount) times the
  probability of a sequencing error at the differing base (from its Phred
  quality). The best candidate is accepted when its posterior probability is
  at least ``confidence`` (0.975 by default).
* :func:`barcode_translation_table` builds an ATAC -> gene-expression barcode
  map for 10x Multiome, where line *i* of the ATAC whitelist corresponds to
  line *i* of the GEX whitelist.
"""
from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_BASES = "ACGT"


def revcomp(seq: str) -> str:
    """Reverse complement of a DNA string (``N`` stays ``N``)."""
    return seq.translate(_COMP)[::-1]


def _open_text(path: Union[str, Path]):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def load_whitelist(path: Union[str, Path]) -> List[str]:
    """Read a barcode whitelist, keeping file order.

    One barcode per line; anything after the first whitespace or comma is
    ignored, as is a trailing ``-<n>`` GEM-well suffix. Empty lines and
    ``#`` comments are skipped. Duplicates are dropped (first occurrence kept).
    """
    out: List[str] = []
    seen = set()
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bc = line.replace(",", " ").split()[0].split("-")[0].upper()
            if bc not in seen:
                seen.add(bc)
                out.append(bc)
    if not out:
        raise ValueError(f"whitelist {path} is empty")
    return out


def barcode_translation_table(
    source_whitelist: Union[str, Path, Sequence[str]],
    target_whitelist: Union[str, Path, Sequence[str]],
) -> Dict[str, str]:
    """Pair two whitelists line by line (10x Multiome ATAC -> GEX barcodes).

    In a 10x Multiome run the ATAC library is barcoded with
    ``737K-arc-v1`` (ATAC version) and the gene-expression library with
    ``737K-arc-v1`` (GEX version). The two files have the same length and
    line *i* of one is the same gel bead as line *i* of the other; Cell
    Ranger ARC reports ATAC fragments under the GEX barcode. Passing both
    files here gives the map used by ``fastq_to_fragments(...,
    barcode_translation=...)``.
    """
    src = load_whitelist(source_whitelist) if isinstance(source_whitelist, (str, Path)) else list(source_whitelist)
    dst = load_whitelist(target_whitelist) if isinstance(target_whitelist, (str, Path)) else list(target_whitelist)
    if len(src) != len(dst):
        raise ValueError(
            f"whitelists differ in length ({len(src)} vs {len(dst)}); "
            "a positional translation needs files of equal length"
        )
    return dict(zip(src, dst))


def load_barcode_translation(path: Union[str, Path]) -> Dict[str, str]:
    """Read a two-column (``from<TAB>to``, or comma / space separated) map."""
    out: Dict[str, str] = {}
    with _open_text(path) as fh:
        for line in fh:
            parts = line.strip().replace(",", " ").split()
            if len(parts) >= 2 and not parts[0].startswith("#"):
                out[parts[0].split("-")[0].upper()] = parts[1]
    if not out:
        raise ValueError(f"barcode translation {path} has no two-column lines")
    return out


def orient_barcode(seq: str, qual: str, orientation: str, start: int, length: int) -> Tuple[str, str]:
    """Cut the barcode out of a barcode-read sequence.

    The read is first put into whitelist orientation (reverse-complemented
    when ``orientation == "reverse"``, with the qualities reversed to match),
    then ``[start, start + length)`` is taken. For a 24-bp Multiome i5 read
    (8-bp spacer + 16-bp barcode) that is ``start=8, length=16`` in either
    orientation.
    """
    if orientation == "reverse":
        seq = revcomp(seq)
        qual = qual[::-1]
    elif orientation != "forward":
        raise ValueError("orientation must be 'forward' or 'reverse'")
    return seq[start:start + length], qual[start:start + length]


def detect_barcode_orientation(
    barcode_reads: Iterable[str],
    whitelist: Union[set, Dict[str, int], Sequence[str]],
    *,
    start: int = 0,
    length: int = 16,
    min_fraction: float = 0.05,
) -> Tuple[str, float, float]:
    """Pick the barcode-read orientation that matches the whitelist best.

    Returns ``(orientation, frac_forward, frac_reverse)`` where the fractions
    are exact-whitelist hits among the reads given (a few hundred thousand
    are plenty). Raises :class:`ValueError` when neither orientation
    reaches ``min_fraction``, which usually means the wrong whitelist or
    read layout.
    """
    wl = whitelist if isinstance(whitelist, (set, dict)) else set(whitelist)
    n = fwd = rev = 0
    for seq in barcode_reads:
        n += 1
        if seq[start:start + length] in wl:
            fwd += 1
        if revcomp(seq)[start:start + length] in wl:
            rev += 1
    if n == 0:
        raise ValueError("no barcode reads to detect the orientation from")
    f_fwd, f_rev = fwd / n, rev / n
    if max(f_fwd, f_rev) < min_fraction:
        raise ValueError(
            f"only {f_fwd:.1%} (forward) / {f_rev:.1%} (reverse-complement) of "
            f"barcode reads match the whitelist at [{start}:{start + length}]; "
            "check the whitelist, the chemistry and which file holds the barcode"
        )
    return ("forward" if f_fwd >= f_rev else "reverse"), f_fwd, f_rev


class BarcodeCorrector:
    """Whitelist barcode correction with an abundance prior.

    Parameters
    ----------
    whitelist
        Barcode sequences (all the same length).
    confidence
        Minimum posterior probability of the best Hamming-1 candidate.
    pseudocount
        Added to every whitelist barcode's exact-match count to form the
        prior, so a barcode never seen exactly can still be a candidate.
    min_error_prob
        Floor on the per-base error probability taken from the quality
        string (Phred 33 by default).
    qual_offset
        ASCII offset of the quality string.

    Usage: feed every raw barcode of the run to :meth:`count` (this sets the
    prior), then call :meth:`correct` / :meth:`correct_many`.
    """

    def __init__(
        self,
        whitelist: Sequence[str],
        *,
        confidence: float = 0.975,
        pseudocount: float = 1.0,
        min_error_prob: float = 0.0005,
        qual_offset: int = 33,
    ) -> None:
        wl = [str(b).upper() for b in whitelist]
        if not wl:
            raise ValueError("whitelist is empty")
        lengths = {len(b) for b in wl}
        if len(lengths) != 1:
            raise ValueError(f"whitelist barcodes have mixed lengths {sorted(lengths)}")
        self.length = lengths.pop()
        self.whitelist = wl
        self.index: Dict[str, int] = {b: i for i, b in enumerate(wl)}
        self.counts = np.zeros(len(wl), dtype=np.int64)
        self.confidence = float(confidence)
        self.pseudocount = float(pseudocount)
        self.min_error_prob = float(min_error_prob)
        self.qual_offset = int(qual_offset)
        # Phred -> error probability, floored.
        q = np.arange(0, 94, dtype=np.float64)
        self._perr = np.maximum(self.min_error_prob, 10.0 ** (-q / 10.0))
        self._neighbors: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # -- prior --------------------------------------------------------
    def count(self, barcodes: Iterable[str]) -> int:
        """Add exact whitelist hits from ``barcodes`` to the prior; returns
        the number of hits added."""
        c = Counter(barcodes)
        hits = 0
        idx = self.index
        for bc, n in c.items():
            i = idx.get(bc)
            if i is not None:
                self.counts[i] += n
                hits += n
        return hits

    def set_counts(self, counts: Union[np.ndarray, Dict[str, int]]) -> None:
        """Set the prior directly (array aligned with the whitelist, or a
        ``{barcode: count}`` dict)."""
        if isinstance(counts, dict):
            arr = np.zeros(len(self.whitelist), dtype=np.int64)
            for bc, n in counts.items():
                i = self.index.get(bc)
                if i is not None:
                    arr[i] = n
            self.counts = arr
        else:
            arr = np.asarray(counts, dtype=np.int64)
            if arr.shape != (len(self.whitelist),):
                raise ValueError("counts must align with the whitelist")
            self.counts = arr.copy()

    # -- correction ---------------------------------------------------
    def _candidates(self, seq: str) -> Tuple[np.ndarray, np.ndarray]:
        """Hamming-1 whitelist neighbours of ``seq``: (positions, wl indices)."""
        hit = self._neighbors.get(seq)
        if hit is not None:
            return hit
        pos: List[int] = []
        ids: List[int] = []
        idx = self.index
        for p in range(len(seq)):
            head, cur, tail = seq[:p], seq[p], seq[p + 1:]
            for b in _BASES:
                if b == cur:
                    continue
                i = idx.get(head + b + tail)
                if i is not None:
                    pos.append(p)
                    ids.append(i)
        hit = (np.asarray(pos, dtype=np.int64), np.asarray(ids, dtype=np.int64))
        if len(self._neighbors) > 2_000_000:
            self._neighbors.clear()
        self._neighbors[seq] = hit
        return hit

    def posterior(self, seq: str, qual: str) -> Tuple[Optional[str], float]:
        """Best candidate and its posterior probability (``(None, 0.0)``
        when there is none). An exact whitelist hit returns ``(seq, 1.0)``."""
        seq = seq.upper()
        if seq in self.index:
            return seq, 1.0
        if len(seq) != self.length:
            return None, 0.0
        pos, ids = self._candidates(seq)
        if pos.size == 0:
            return None, 0.0
        q = np.frombuffer(qual.encode("ascii"), dtype=np.uint8).astype(np.int64) - self.qual_offset
        q = np.clip(q, 0, 93)
        like = (self.counts[ids] + self.pseudocount) * self._perr[q[pos]]
        tot = like.sum()
        if tot <= 0:
            return None, 0.0
        k = int(np.argmax(like))
        return self.whitelist[ids[k]], float(like[k] / tot)

    def correct(self, seq: str, qual: str) -> Optional[str]:
        """Corrected barcode, or ``None`` if it cannot be assigned."""
        bc, p = self.posterior(seq, qual)
        if bc is None:
            return None
        if p >= self.confidence:
            return bc
        return None

    def correct_many(self, seqs: Sequence[str], quals: Sequence[str]) -> List[Optional[str]]:
        return [self.correct(s, q) for s, q in zip(seqs, quals)]


__all__ = [
    "revcomp",
    "load_whitelist",
    "barcode_translation_table",
    "load_barcode_translation",
    "orient_barcode",
    "detect_barcode_orientation",
    "BarcodeCorrector",
]
