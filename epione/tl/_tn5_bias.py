"""Per-base Tn5 sequence bias as one pluggable object.

Footprinting compares observed Tn5 cuts with the cuts expected from the
enzyme's sequence preference alone, so every footprint method needs "the
expected-cut track for this region". Epione had that logic inlined twice
(a hexamer lookup in ``get_footprints`` and again in
``multi_scale_footprint_region``) and had no way to swap in a better
model. :class:`Tn5Bias` is that missing seam:

``track(chrom, start, end) -> float32[end - start]``

Four implementations:

:class:`PrintBias`
    The PRINT/scPrinter CNN (Hu *et al.*, *Nature* 2025), 0.1 MB, run on
    demand for just the bases asked for. scPrinter itself ships this as a
    precomputed genome-wide file (10.9 GB for mm10) and loads the whole
    genome into memory on every call.
:class:`PrecomputedBias`
    scPrinter's precomputed ``*_bias_v2.h5`` files, read lazily per
    region if you already have one.
:class:`KmerBias`
    Epione's existing hexamer enrichment table
    (:func:`~epione.tl.compute_tn5_bias_table`).
:class:`ChromBPNetBias`
    ChromBPNet's convolutional bias model (Pampari *et al.*), trained per
    dataset on GC-matched background or taken from an ENCODE release, run
    in PyTorch (:mod:`epione.models.chrombpnet`).

Each model carries the cut convention it was trained under (``shift``,
see :mod:`epione._insertions`). The PRINT models were fitted to cuts at a
fragment's ``start`` and ``end``; the hexamer table counts ``start`` and
``end - 1``. Reading insertions with a different convention from the one
the bias assumes misaligns the two tracks by a base, so footprinting code
should read insertions with ``bias.shift`` rather than a hard-coded one.

Exactness
---------
:class:`PrintBias` reproduces scPrinter's ``predict_genome_tn5_bias``
bit for bit. That takes more than running the CNN on ``[s-50, e+50)``:
CPU convolution kernels are chosen by input length, so a region run on
its own differs from the genome-wide run by ~1e-6. scPrinter splits each
chromosome into overlapping chunks whose layout depends only on the
chromosome length; ``exact=True`` (default) recomputes that layout and
runs the region through the very same chunks. ``exact=False`` runs one
forward pass per region instead -- faster for long regions or on GPU,
identical up to float32 rounding.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .._insertions import SHIFT_ARCHR, SHIFT_PRINT

__all__ = [
    "ChromBPNetBias",
    "KmerBias",
    "PrecomputedBias",
    "PrintBias",
    "Tn5Bias",
    "tn5_bias",
]

# scPrinter's genome-wide inference constants (genome.py,
# predict_genome_tn5_bias): N-padding on each side, and the default
# chunk size whose layout exact mode must reproduce.
_CONTEXT_RADIUS = 50
_PRINT_BATCH_SIZE = 1000


def _fasta_path(genome) -> str:
    """Accept a :class:`~epione.core.genome.Genome` or a FASTA path."""
    fasta = getattr(genome, "fasta", genome)
    if callable(fasta):          # Genome.fasta may be a lazy fetcher
        fasta = fasta()
    if fasta is None:
        raise ValueError("a genome / FASTA path is required for this bias model")
    return str(fasta)


class Tn5Bias:
    """Base class: expected Tn5 cuts per base from sequence alone.

    Subclasses implement :meth:`track` and set :attr:`shift` to the cut
    convention the model was trained under.
    """

    #: Short name, shown in result objects and plot titles.
    name: str = "tn5"
    #: ``(shift_start, shift_end)`` cut convention; see module docstring.
    shift: Tuple[int, int] = SHIFT_PRINT

    def track(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Bias for ``[start, end)``, a float array of length
        ``end - start`` (float32 for the PRINT models, which is what
        scPrinter stores; float64 for the hexamer table)."""
        raise NotImplementedError

    def __getstate__(self):
        # File handles and torch modules do not survive a fork/pickle;
        # subclasses reopen them lazily.
        state = self.__dict__.copy()
        for k in ("_fa", "_h5", "_model", "_cache"):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(shift={self.shift})"


# ---------------------------------------------------------------------------
# PRINT CNN, on demand
# ---------------------------------------------------------------------------

def _valid_batch_size(seq_len: int, num_batch: int, stride: int) -> int:
    """scPrinter ``genome.valid_batch_size``: smallest chunking whose last
    chunk is still longer than the receptive field."""
    init_bs = (seq_len + num_batch + stride) // num_batch
    lengths = [min(start + init_bs, seq_len) - start
               for start in range(0, seq_len, init_bs - stride)]
    if min(lengths) < stride + 1:
        return _valid_batch_size(seq_len, num_batch - 1, stride)
    return init_bs


# Matches scPrinter's utils.DNAmapping: A,C,G,T one-hot, anything else 0.
_ONEHOT = np.zeros((128, 4), dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    _ONEHOT[ord(_b), _i] = 1


class PrintBias(Tn5Bias):
    """PRINT's convolutional Tn5 bias model, evaluated on demand.

    Parameters
    ----------
    genome
        :class:`~epione.core.genome.Genome` or FASTA path (indexed).
    model
        Path to a TorchScript bias model. Default: the published v2 model
        (``Tn5_NN_model_py_v2.pt``), downloaded once and cached.
    device
        Torch device. Exact mode is only guaranteed on the device the
        reference was produced on (CPU).
    exact
        Reproduce scPrinter's chunked genome-wide inference bit for bit
        (see module docstring). ``False`` runs one pass per region.
    cache_chunks
        How many computed chunks to keep, so overlapping regions do not
        rerun the CNN.
    """

    name = "print"
    shift = SHIFT_PRINT

    def __init__(
        self,
        genome,
        *,
        model: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        exact: bool = True,
        cache_chunks: int = 256,
    ) -> None:
        self.fasta = _fasta_path(genome)
        self.model_path = str(model) if model is not None else None
        self.device = device
        self.exact = bool(exact)
        self.cache_chunks = int(cache_chunks)

    # -- lazy resources ----------------------------------------------------

    def _fasta(self):
        if getattr(self, "_fa", None) is None:
            import pysam
            self._fa = pysam.FastaFile(self.fasta)
        return self._fa

    def _net(self):
        if getattr(self, "_model", None) is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "PrintBias runs the PRINT CNN and needs torch. Install "
                    "torch, or use PrecomputedBias with scPrinter's bias "
                    "file, or KmerBias."
                ) from exc
            from ._print import fetch_print_model
            path = self.model_path or fetch_print_model("tn5_bias")
            self._model = torch.jit.load(path, map_location="cpu").to(self.device).eval()
        return self._model

    def _chunk_cache(self) -> "OrderedDict":
        if getattr(self, "_cache", None) is None:
            self._cache = OrderedDict()
        return self._cache

    # -- inference ---------------------------------------------------------

    def _forward(self, seq: str) -> np.ndarray:
        """CNN on a padded sequence -> bias for its central len-100 bases,
        with scPrinter's output transform."""
        import torch
        codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        onehot = _ONEHOT[codes].T  # (4, L) int8, as DNA_one_hot builds it
        x = torch.from_numpy(onehot).to(self.device).float()[None]
        with torch.no_grad():
            b = self._net()(x).detach().cpu().numpy().reshape(-1)
        return np.power(10, (b - 0.5) * 2) - 0.01

    def _padded_slice(self, chrom: str, lo: int, hi: int, chrom_len: int) -> str:
        """Sequence of the N-padded chromosome over ``[lo, hi)`` in padded
        coordinates (padded = 50 N + chrom + 50 N)."""
        g_lo, g_hi = lo - _CONTEXT_RADIUS, hi - _CONTEXT_RADIUS
        left = max(0, -g_lo)
        right = max(0, g_hi - chrom_len)
        core = ""
        a, b = max(g_lo, 0), min(g_hi, chrom_len)
        if b > a:
            core = self._fasta().fetch(chrom, a, b).upper()
        return "N" * left + core + "N" * right

    def track(self, chrom: str, start: int, end: int) -> np.ndarray:
        if end <= start:
            raise ValueError(f"empty interval {chrom}:{start}-{end}")
        fa = self._fasta()
        if chrom not in fa.references:
            raise KeyError(f"{chrom!r} not in {self.fasta}")
        chrom_len = fa.get_reference_length(chrom)

        if not self.exact:
            seq = self._padded_slice(chrom, start, end + 2 * _CONTEXT_RADIUS,
                                     chrom_len)
            return self._forward(seq).astype(np.float32, copy=False)

        # Reproduce scPrinter's chunk layout for this chromosome.
        seq_len = chrom_len + 2 * _CONTEXT_RADIUS
        stride = 2 * _CONTEXT_RADIUS
        num_batch = max(seq_len // _PRINT_BATCH_SIZE, 1)
        bs = _valid_batch_size(seq_len, num_batch, stride)
        step = bs - stride

        out = np.zeros(end - start, dtype=np.float32)
        cache = self._chunk_cache()
        # Chunk k (padded start k*step) yields genome [k*step, k*step+len-100).
        k0 = max(0, (start - bs + stride) // step)
        k = k0
        while True:
            c0 = k * step
            if c0 >= seq_len or c0 >= end:
                break
            c_len = min(bs, seq_len - c0)
            g0, g1 = c0, c0 + c_len - stride
            if g1 > start:
                key = (chrom, c0)
                b = cache.get(key)
                if b is None:
                    b = self._forward(self._padded_slice(chrom, c0, c0 + c_len,
                                                         chrom_len))
                    cache[key] = b
                    if len(cache) > self.cache_chunks:
                        cache.popitem(last=False)
                else:
                    cache.move_to_end(key)
                a, z = max(start, g0), min(end, g1)
                if z > a:
                    out[a - start:z - start] = b[a - g0:z - g0]
            k += 1
        # Positions past the chromosome end have no bias.
        return out


# ---------------------------------------------------------------------------
# scPrinter's precomputed files
# ---------------------------------------------------------------------------

class PrecomputedBias(Tn5Bias):
    """scPrinter's precomputed genome-wide bias (``*_bias_v2.h5``).

    One float array per chromosome; only the requested slice is read, so
    the file is never loaded whole. Positions beyond a chromosome are 0.
    """

    name = "print-precomputed"
    shift = SHIFT_PRINT

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = str(path)

    def _file(self):
        if getattr(self, "_h5", None) is None:
            import h5py
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def track(self, chrom: str, start: int, end: int) -> np.ndarray:
        h5 = self._file()
        if chrom not in h5:
            raise KeyError(f"{chrom!r} not in {self.path}")
        ds = h5[chrom]
        out = np.zeros(end - start, dtype=np.float32)
        a, b = max(start, 0), min(end, ds.shape[0])
        if b > a:
            out[a - start:b - start] = ds[a:b]
        return out


# ---------------------------------------------------------------------------
# Epione's hexamer table
# ---------------------------------------------------------------------------

class KmerBias(Tn5Bias):
    """Hexamer (k-mer) enrichment bias from
    :func:`~epione.tl.compute_tn5_bias_table`.

    This is the lookup that ``get_footprints`` and the original
    ``multi_scale_footprint_region`` inlined; positions whose k-mer
    contains ``N`` get a neutral 1.0.

    Parameters
    ----------
    genome
        :class:`~epione.core.genome.Genome` or FASTA path.
    table
        ``float[4**k]`` from :func:`compute_tn5_bias_table`. If ``None``,
        pass ``fragment_file`` and it is estimated on construction.
    fragment_file
        Used only to estimate ``table``.
    kmer_length
        k. Must match ``table`` if given.
    """

    name = "kmer"
    # compute_tn5_bias_table counts cuts at s and e-1.
    shift = SHIFT_ARCHR

    def __init__(
        self,
        genome,
        table: Optional[np.ndarray] = None,
        *,
        fragment_file: Optional[Union[str, Path]] = None,
        kmer_length: int = 6,
    ) -> None:
        self.fasta = _fasta_path(genome)
        self.k = int(kmer_length)
        if table is None:
            if fragment_file is None:
                raise ValueError("KmerBias needs a table or a fragment_file")
            from ._footprint import compute_tn5_bias_table
            table = compute_tn5_bias_table(str(fragment_file), self.fasta,
                                           kmer_length=self.k)
        table = np.asarray(table, dtype=np.float64)
        if table.shape != (4 ** self.k,):
            raise ValueError(
                f"table has shape {table.shape}; expected ({4 ** self.k},) "
                f"for kmer_length={self.k}"
            )
        self.table = table

    def _fasta(self):
        if getattr(self, "_fa", None) is None:
            import pysam
            self._fa = pysam.FastaFile(self.fasta)
        return self._fa

    def track(self, chrom: str, start: int, end: int) -> np.ndarray:
        from ._footprint import _codes_to_kmers, _seq_to_codes
        k, half = self.k, self.k // 2
        lo, hi = start - half, end + (k - half) - 1
        fa = self._fasta()
        n = fa.get_reference_length(chrom)
        left, right = max(0, -lo), max(0, hi - n)
        seq = "N" * left + fa.fetch(chrom, max(lo, 0), min(hi, n)).upper() + "N" * right
        km = _codes_to_kmers(_seq_to_codes(seq), k)
        # float64, like the lookup this replaces: rounding to float32 here
        # would perturb every downstream score by ~1e-5.
        return np.where(km >= 0, self.table[np.clip(km, 0, None)], 1.0)


class ChromBPNetBias(Tn5Bias):
    """ChromBPNet's Tn5 bias model as a per-base expected-cut track.

    The bias model (a small BPNet trained on GC-matched background regions,
    see :mod:`epione.models.chrombpnet`) predicts a 1,000-bp cut profile and
    a total count from 2,114 bp of sequence. ``track`` tiles the requested
    interval with 1,000-bp output windows, runs each on its centred 2,114-bp
    input (N-padded past chromosome ends) and returns the expected cuts
    ``softmax(profile) * exp(log_counts)`` per base.

    Parameters
    ----------
    genome
        :class:`~epione.core.genome.Genome` or FASTA path.
    model
        A trained bias :class:`~epione.models.chrombpnet.BPNet`; a path to an
        official ``bias_scaled`` ``.h5``; or an ENCODE release name from
        :data:`epione.models.chrombpnet.PRETRAINED` (``'encode_k562'``...).
    fold
        ENCODE fold for a release name.
    device, batch_size
        Torch device and windows per forward pass.

    The cut convention is ``(0, 0)``: the official pipeline builds its
    training bigWig from 10x fragments with cuts at ``start`` and ``end``.
    """

    name = "chrombpnet"
    shift = SHIFT_PRINT

    def __init__(self, genome, model, *, fold: int = 0, device: str = "cpu",
                 batch_size: int = 64) -> None:
        self.fasta = _fasta_path(genome)
        self.model_spec = model
        self.fold = int(fold)
        self.device = device
        self.batch_size = int(batch_size)

    def _fasta(self):
        if getattr(self, "_fa", None) is None:
            import pysam
            self._fa = pysam.FastaFile(self.fasta)
        return self._fa

    def _net(self):
        if getattr(self, "_model", None) is None:
            from epione.models import chrombpnet as cbp
            spec = self.model_spec
            if isinstance(spec, (str, Path)) and str(spec) in cbp.PRETRAINED:
                m = cbp.load_pretrained(str(spec), "bias", self.fold)
            elif isinstance(spec, (str, Path)):
                m = cbp.load_keras_h5(spec)
            else:
                m = spec
            if hasattr(m, "accessibility"):   # a full ChromBPNet: take its bias
                m = m.bias
            self._model = m.to(self.device).eval()
        return self._model

    def _seq(self, chrom: str, lo: int, hi: int, chrom_len: int) -> str:
        a, b = max(lo, 0), min(hi, chrom_len)
        core = self._fasta().fetch(chrom, a, b).upper() if b > a else ""
        return "N" * (a - lo) + core + "N" * (hi - b)

    def windows(self, chrom: str, start: int, end: int):
        """Output-window starts covering ``[start, end)``."""
        out_len = self._net().output_len
        return list(range(start, end, out_len))

    def track(self, chrom: str, start: int, end: int) -> np.ndarray:
        if end <= start:
            raise ValueError(f"empty interval {chrom}:{start}-{end}")
        from epione.models.chrombpnet import one_hot_encode, predict, softmax
        fa = self._fasta()
        if chrom not in fa.references:
            raise KeyError(f"{chrom!r} not in {self.fasta}")
        net = self._net()
        L, O = net.input_len, net.output_len
        pad = (L - O) // 2
        clen = fa.get_reference_length(chrom)
        starts = self.windows(chrom, start, end)
        seqs = [self._seq(chrom, w - pad, w - pad + L, clen) for w in starts]
        x = np.transpose(one_hot_encode(seqs), (0, 2, 1)).astype(np.float32)
        prof, cnt = predict(net, x, batch_size=self.batch_size, device=self.device)
        exp_cuts = softmax(prof.astype(np.float64)) * np.exp(cnt.astype(np.float64))
        return exp_cuts.reshape(-1)[: end - start].astype(np.float32)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ChromBPNetBias(model={self.model_spec!r}, shift={self.shift})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def tn5_bias(
    genome=None,
    *,
    method: str = "print",
    path: Optional[Union[str, Path]] = None,
    fragment_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Tn5Bias:
    """Build a Tn5 bias model.

    Parameters
    ----------
    genome
        :class:`~epione.core.genome.Genome` or FASTA path.
    method
        ``'print'`` -- PRINT CNN on demand (needs torch);
        ``'precomputed'`` -- scPrinter's bias file at ``path``;
        ``'kmer'`` -- hexamer table, estimated from ``fragment_file``
        unless ``table=`` is passed;
        ``'chrombpnet'`` -- ChromBPNet's bias model (``model=`` a trained
        BPNet, an official ``.h5`` or an ENCODE release name; needs torch).
    **kwargs
        Forwarded to the model class.

    Examples
    --------
    >>> bias = epi.tl.tn5_bias(epi.data.mm10)                # PRINT CNN
    >>> bias.track("chr19", 41_829_000, 41_830_000).shape
    (1000,)
    >>> hint = epi.tl.tn5_bias("genome.fa", method="hint")   # HINT-ATAC

    ``method='hint'`` builds :class:`~epione.tl.HintBias`, HINT-ATAC's
    strand-specific 8-mer tables (pass ``table_f=`` / ``table_r=`` to use
    an ``rgt-hint estimation`` pair instead).
    """
    method = method.lower()
    if method == "hint":
        from ._hint import HintBias
        return HintBias(genome, **kwargs)
    if method == "print":
        return PrintBias(genome, **kwargs)
    if method in ("precomputed", "h5"):
        if path is None:
            raise ValueError("method='precomputed' needs path=")
        return PrecomputedBias(path)
    if method in ("kmer", "hexamer"):
        return KmerBias(genome, fragment_file=fragment_file, **kwargs)
    if method == "chrombpnet":
        model = kwargs.pop("model", None) or path
        if model is None:
            raise ValueError("method='chrombpnet' needs model= (BPNet, .h5 path or "
                             "an ENCODE release name such as 'encode_k562')")
        return ChromBPNetBias(genome, model, **kwargs)
    raise ValueError(
        f"unknown bias method {method!r}; use 'print', 'precomputed', 'kmer' or 'chrombpnet'"
    )
