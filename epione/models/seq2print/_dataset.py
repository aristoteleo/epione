"""Training windows for seq2PRINT, read straight from fragments.

scPrinter trains seq2PRINT from bigWigs: it first exports one insertion
bigWig per cell group (``export_bigwigs``), pairs each with a genome-wide
bias bigWig, and reads windows from those. Here a window is built from
the same inputs Epione already uses for footprinting -- an
:class:`~epione._insertions.InsertionReader` over the fragments and a
:class:`~epione.tl.Tn5Bias` -- so there is no export step and no
genome-wide bias file.

Geometry and augmentation are scPrinter's (``seq2PRINTDataset``):

* DNA: ``dna_len`` bp (1,840 by default) centred on the peak summit;
* signal: ``signal_window`` bp (1,000) centred on the summit, two
  channels -- insertions and bias -- from which the footprint target is
  computed on the fly (its first and last 100 bp only serve as flanks);
* jitter: one offset in ``[0, 2 * max_jitter)`` shifts DNA and signal
  together;
* reverse complement with probability 0.5: DNA flips channels and
  positions, signal flips positions.

Each peak is read once, on construction, and cached as compact arrays
(one byte per base of sequence), so an epoch costs no I/O.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

__all__ = ["Seq2PRINTDataset", "summits_from_peaks"]

# A, C, G, T -> one-hot rows; anything else (N, IUPAC) all-zero, as in
# scPrinter's DNAmapping.
_LUT = np.zeros((256, 4), dtype=np.float32)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b), _i] = 1.0


def summits_from_peaks(peaks) -> pd.DataFrame:
    """``chrom, summit`` from a peak table with a ``summit`` column, or
    the midpoint of ``start``/``end``."""
    df = pd.DataFrame(peaks).copy()
    low = {c.lower(): c for c in df.columns}
    chrom = next((low[c] for c in ("chrom", "chromosome", "seqnames") if c in low), None)
    if chrom is None:
        raise ValueError("peaks need a chrom column")
    if "summit" in low:
        summit = df[low["summit"]].astype(int)
    elif "start" in low and "end" in low:
        summit = (df[low["start"]].astype(int) + df[low["end"]].astype(int)) // 2
    else:
        raise ValueError("peaks need a summit, or start and end")
    return pd.DataFrame({"chrom": df[chrom].astype(str).to_numpy(),
                         "summit": summit.to_numpy()})


class Seq2PRINTDataset(torch.utils.data.Dataset):
    """seq2PRINT training/evaluation windows.

    Parameters
    ----------
    peaks
        Peak table (``chrom`` + ``summit``, or ``start``/``end``).
    fasta
        Indexed reference FASTA.
    reader
        :class:`~epione._insertions.InsertionReader` over the fragments.
        Its ``shift`` must match ``bias.shift`` (checked).
    bias
        :class:`~epione.tl.Tn5Bias`; PRINT's, for the targets to be
        PRINT's.
    groups
        Cell-index arrays, one per cell state. ``None``: one group of all
        cells (a bulk model). With several groups each item is a
        ``(peak, group)`` pair and also returns the group index, for
        LoRA fine-tuning.
    dna_len
        Input length; take it from the model (``model.dna_len``).
    signal_window, max_jitter, reverse_complement
        scPrinter's defaults: 1000, 128, True.
    min_coverage
        Drop peaks with fewer insertions (summed over groups, in the
        signal window). ``'auto'`` reproduces scPrinter's filter --
        between ``max(q_0.0001, 10)`` and ``q_0.9999`` of coverage.
    seed
        Seeds jitter / reverse complement. Call :meth:`set_epoch` each
        epoch for fresh, still reproducible, augmentation.
    """

    def __init__(
        self,
        peaks,
        *,
        fasta,
        reader,
        bias,
        groups: Optional[Sequence[Sequence[int]]] = None,
        dna_len: int = 1840,
        signal_window: int = 1000,
        max_jitter: int = 128,
        reverse_complement: bool = True,
        min_coverage: Union[str, float, None] = "auto",
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        import pysam
        from ..._insertions import groups_to_matrix

        if tuple(reader.shift) != tuple(bias.shift):
            raise ValueError(
                f"reader cut convention {reader.shift} does not match the bias "
                f"model's {bias.shift}; build the reader with shift=bias.shift"
            )
        if dna_len % 2 or signal_window % 2:
            raise ValueError("dna_len and signal_window must be even")
        self.dna_len, self.signal_window = int(dna_len), int(signal_window)
        self.max_jitter = int(max_jitter)
        self.reverse_complement = bool(reverse_complement)
        self.seed, self._epoch = int(seed), 0
        self._augment = True

        self.n_groups = 1 if groups is None else len(groups)
        G = None if groups is None else groups_to_matrix(groups, reader.n_cells)

        fa = pysam.FastaFile(str(fasta))
        summits = summits_from_peaks(peaks)
        J = self.max_jitter
        dna_half, sig_half = self.dna_len // 2 + J, self.signal_window // 2 + J

        keep, seqs, ins, bia = [], [], [], []
        for i, (c, s) in enumerate(zip(summits["chrom"], summits["summit"])):
            if c not in fa.references:
                continue
            n = fa.get_reference_length(c)
            if s - dna_half < 0 or s + dna_half > n:
                continue                        # scPrinter drops edge peaks too
            seq = fa.fetch(c, s - dna_half, s + dna_half).upper()
            seqs.append(np.frombuffer(seq.encode("ascii"), dtype=np.uint8))
            x = reader.group_insertions(c, s - sig_half, s + sig_half, G)
            ins.append(x.astype(np.float32))
            bia.append(np.asarray(bias.track(c, s - sig_half, s + sig_half), np.float32))
            keep.append(i)
        reader.close()
        if not keep:
            raise ValueError("no peak fits inside its chromosome with the requested windows")

        self.peaks = summits.iloc[keep].reset_index(drop=True)
        self._seq = np.stack(seqs)                  # [P, dna_len + 2J] uint8
        self._ins = np.stack(ins)                   # [P, G, signal + 2J]
        self._bias = np.stack(bia)                  # [P, signal + 2J]

        cov = self._ins[:, :, J:J + self.signal_window].sum((1, 2))
        if min_coverage == "auto":
            lo = max(np.quantile(cov, 1e-4), 10)
            ok = (cov >= lo) & (cov <= np.quantile(cov, 0.9999))
        elif min_coverage is None:
            ok = np.ones(len(cov), bool)
        else:
            ok = cov >= float(min_coverage)
        if not ok.all():
            self.peaks = self.peaks.loc[ok].reset_index(drop=True)
            self._seq, self._ins, self._bias = self._seq[ok], self._ins[ok], self._bias[ok]
        if verbose:
            from ...core import console
            console.level1(
                f"Seq2PRINTDataset: {len(self.peaks):,} peaks x {self.n_groups} group(s) "
                f"({len(summits) - len(keep)} at contig edges, {int((~ok).sum())} "
                f"outside coverage bounds dropped)"
            )

    @property
    def coverage(self) -> np.ndarray:
        """``[n_peaks, n_groups]`` insertions in each centred signal window."""
        J = self.max_jitter
        return self._ins[:, :, J:J + self.signal_window].sum(-1)

    # -- sampling -------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation stream (call once per epoch)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.peaks) * self.n_groups

    def window(self, idx: int, *, jitter: Optional[int] = None, rc: bool = False):
        """One item with explicit augmentation (``jitter`` defaults to the
        centred window). Returns ``(X [4, dna_len], y [2, signal], group)``."""
        p, g = divmod(int(idx), self.n_groups)
        j = self.max_jitter if jitter is None else int(jitter)
        X = _LUT[self._seq[p, j:j + self.dna_len]].T
        y = np.stack([self._ins[p, g, j:j + self.signal_window],
                      self._bias[p, j:j + self.signal_window]])
        if rc:
            X = X[::-1, ::-1]        # complement = reversed ACGT channels
            y = y[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(X)),
                torch.from_numpy(np.ascontiguousarray(y)), g)

    def __getitem__(self, idx: int):
        if not self._augment:
            jitter, rc = None, False
        elif self.max_jitter or self.reverse_complement:
            rng = np.random.default_rng((self.seed, self._epoch, int(idx)))
            jitter = int(rng.integers(2 * self.max_jitter)) if self.max_jitter else 0
            rc = self.reverse_complement and rng.random() > 0.5
        else:
            jitter, rc = 0, False
        X, y, g = self.window(idx, jitter=jitter, rc=rc)
        if self.n_groups == 1:
            return X, y
        return X, y, torch.tensor(g, dtype=torch.long)

    def centred(self) -> "Seq2PRINTDataset":
        """A view of the same cached windows with no jitter or reverse
        complement, for evaluation."""
        view = object.__new__(type(self))
        view.__dict__.update(self.__dict__)
        view._augment = False
        return view

    def split(self, chromosomes: Sequence[str]) -> "Seq2PRINTDataset":
        """Subset to peaks on ``chromosomes`` (e.g. one side of a fold)."""
        m = self.peaks["chrom"].isin(list(chromosomes)).to_numpy()
        view = object.__new__(type(self))
        view.__dict__.update(self.__dict__)
        view.peaks = self.peaks.loc[m].reset_index(drop=True)
        view._seq, view._ins, view._bias = self._seq[m], self._ins[m], self._bias[m]
        return view

