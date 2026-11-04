"""Run gReLU's tooling on Epione's sequence models (and the reverse).

gReLU (Lal *et al.*, Genentech; MIT) wraps sequence models in a Lightning
module with one prediction API, and builds variant-effect prediction, ISM,
attributions (captum / tangermeme DeepLIFT-SHAP) and motif analyses on top
of it. :func:`wrap` turns an Epione model into such a module, so all of that
works on ChromBPNet and seq2PRINT models unchanged; the helper functions
here are thin, exact pass-throughs that also accept native gReLU models and
plain FASTA paths.

gReLU is optional (``pip install epione[grelu]``) and imported lazily.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "HEADS",
    "wrap",
    "predict_variant_effects",
    "ism",
    "attributions",
    "pseudobulk_anndata",
]


def _grelu():
    try:
        import grelu  # noqa: F401
        import grelu.lightning
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("this needs gReLU: pip install 'epione[grelu]' (grelu>=1.1)") from exc
    return grelu


class _no_lightning_logs:
    """gReLU predicts through ``pl.Trainer(logger=None)``, which Lightning
    reads as "default TensorBoard logger" and writes ``lightning_logs/`` into
    the working directory. Inside this context ``logger=None`` becomes
    ``logger=False`` and checkpointing is off; predictions are unaffected."""

    def __enter__(self):
        import pytorch_lightning as pl

        self._pl = pl
        self._orig = orig = pl.Trainer.__init__

        def init(trainer, *args, **kwargs):
            if kwargs.get("logger", None) is None:
                kwargs["logger"] = False
            kwargs.setdefault("enable_checkpointing", False)
            orig(trainer, *args, **kwargs)

        pl.Trainer.__init__ = init
        return self

    def __exit__(self, *exc):
        self._pl.Trainer.__init__ = self._orig
        return False


# ---------------------------------------------------------------------------
# heads: Epione model output -> (N, tasks, length) tensor
# ---------------------------------------------------------------------------

def _kind(model) -> str:
    if hasattr(model, "accessibility") and hasattr(model, "bias"):
        return "chrombpnet"
    if hasattr(model, "count_dense") and hasattr(model, "prof_conv"):
        return "bpnet"
    if hasattr(model, "profile_cnn_model") and hasattr(model, "dna_len"):
        return "seq2print"
    return "module"


#: Available heads per model kind. Log-scale heads (log counts, log
#: coverage, logits) suit ``compare_func="subtract"`` (log fold change).
HEADS = {
    "bpnet": ("counts", "profile", "profile_logits", "expected_counts"),
    "chrombpnet": ("counts", "profile", "profile_logits", "expected_counts"),
    "seq2print": ("coverage", "footprints"),
    "module": ("output",),
}
_DEFAULT_HEAD = {"bpnet": "counts", "chrombpnet": "counts", "seq2print": "coverage",
                 "module": "output"}


def _seq_len(model, kind) -> Optional[int]:
    if kind in ("bpnet", "chrombpnet"):
        return int(model.input_len)
    if kind == "seq2print":
        return int(model.dna_len)
    return getattr(model, "input_len", None)


def _adapter(model, head: Union[str, Callable], cells=None):
    import torch
    from torch import nn

    kind = _kind(model)
    if callable(head) and not isinstance(head, str):
        fn = head
    else:
        if head not in HEADS[kind]:
            raise ValueError(f"head {head!r} not available for a {kind} model; "
                             f"choose from {HEADS[kind]}")
        fn = None

    class EpioneHead(nn.Module):
        """gReLU-facing view of an Epione model: ``(N, 4, L) -> (N, T, L')``."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            if kind == "seq2print":
                out = self.m(x, cells=cells) if cells is not None else self.m(x)
            else:
                out = self.m(x)
            if fn is not None:
                y = fn(out)
                return y if y.ndim == 3 else y.reshape(y.shape[0], -1, 1)
            if kind in ("bpnet", "chrombpnet"):
                prof, cnt = out
                if head == "counts":
                    return cnt[:, :, None]
                if head == "profile_logits":
                    return prof[:, None, :]
                probs = torch.softmax(prof, dim=-1)
                if head == "profile":
                    return probs[:, None, :]
                return (probs * torch.exp(cnt))[:, None, :]  # expected_counts
            if kind == "seq2print":
                fp, cov = out
                if head == "coverage":
                    return cov.reshape(-1, 1, 1)
                return fp  # (N, modes, L)
            y = out
            return y if y.ndim == 3 else y.reshape(y.shape[0], -1, 1)

    return EpioneHead(model)


def wrap(model, *, head: Union[str, Callable, None] = None, tasks: Optional[Sequence[str]] = None,
         seq_len: Optional[int] = None, cells=None):
    """Wrap an Epione sequence model as a gReLU ``LightningModel``.

    Parameters
    ----------
    model
        A :class:`epione.models.chrombpnet.BPNet` / ``ChromBPNet``, a
        :class:`epione.models.seq2print.Seq2PRINT`, or any ``nn.Module`` that
        maps one-hot ``(N, 4, L)`` to a tensor. A gReLU ``LightningModel`` is
        returned unchanged.
    head
        Which output gReLU sees (see :data:`HEADS`), or a callable applied to
        the model's raw output. Defaults: log counts (ChromBPNet), log
        coverage (seq2PRINT).
    tasks
        Task names (one per output track). Default: the head name.
    seq_len
        Input length; read from the model when it records one.
    cells
        For a seq2PRINT model with cell-state LoRA: the cell/state index (or
        indices) to condition on.

    The wrapper is for inference and interpretation; it applies no output
    activation (gReLU's ``task="regression"``, ``loss="mse"``). Train and
    save Epione models with Epione's own functions.
    """
    grelu = _grelu()
    from grelu.lightning import LightningModel

    if isinstance(model, LightningModel):
        return model
    kind = _kind(model)
    head = head or _DEFAULT_HEAD[kind]
    adapter = _adapter(model, head, cells=cells)
    seq_len = seq_len or _seq_len(model, kind)
    if seq_len is None:
        raise ValueError("seq_len= is required for this model")
    if tasks is None:
        import torch
        with torch.no_grad():
            p = next(model.parameters())
            n_t = adapter(torch.zeros(1, 4, seq_len, dtype=p.dtype, device=p.device)).shape[1]
        name = head if isinstance(head, str) else "output"
        tasks = [name] if n_t == 1 else [f"{name}_{i}" for i in range(n_t)]
    lm = LightningModel(
        model_params={"model_type": "ConvModel", "n_tasks": len(tasks)},
        train_params={"task": "regression", "loss": "mse"},
    )
    lm.model = adapter  # replace the placeholder network
    lm.data_params = {"tasks": {"name": list(tasks)},
                      "train": {"seq_len": int(seq_len), "bin_size": 1}}
    lm.model_params["crop_len"] = 0
    lm.model_params["epione_kind"] = kind
    lm.model_params["epione_head"] = head if isinstance(head, str) else "callable"
    return lm.eval()


def _genome_arg(genome) -> str:
    """gReLU takes a genomepy name or a FASTA path; accept an Epione Genome.

    For a FASTA path gReLU's ``CustomGenome`` needs ``<fasta>.fai`` and a
    ``<fasta>.sizes`` file. The sizes file is written from the index if it
    is missing -- next to the FASTA when that directory is writable, else in
    Epione's cache behind symlinks to the FASTA and its index.
    """
    import os

    fasta = getattr(genome, "fasta", genome)
    if callable(fasta):
        fasta = fasta()
    fasta = str(fasta)
    if not os.path.isfile(fasta):
        return fasta  # a genomepy name
    fai = fasta + ".fai"
    if not os.path.isfile(fai):
        import pysam
        pysam.faidx(fasta)
    if os.path.isfile(fasta + ".sizes"):
        return fasta
    sizes = "".join("\t".join(line.split("\t")[:2]) + "\n" for line in open(fai))
    try:
        with open(fasta + ".sizes", "w") as f:
            f.write(sizes)
        return fasta
    except OSError:
        import hashlib

        import pooch
        d = Path(os.environ.get("EPIONE_DATA_DIR") or pooch.os_cache("epione")) / "grelu_genomes"
        d = d / hashlib.md5(os.path.abspath(fasta).encode()).hexdigest()[:12]
        d.mkdir(parents=True, exist_ok=True)
        link = d / Path(fasta).name
        for src, dst in ((fasta, link), (fai, Path(str(link) + ".fai"))):
            if not dst.exists():
                os.symlink(os.path.abspath(src), dst)
        (Path(str(link) + ".sizes")).write_text(sizes)
        return str(link)


def predict_variant_effects(model, variants: pd.DataFrame, genome, *, head=None,
                            seq_len: Optional[int] = None, compare_func="subtract",
                            rc: bool = False, max_seq_shift: int = 0, devices="cpu",
                            batch_size: int = 64, num_workers: int = 1,
                            return_ad: bool = True, **wrap_kwargs):
    """Predict variant effects with any Epione or gReLU sequence model.

    ``variants`` has columns ``chrom``, ``pos`` (1-based, as gReLU expects),
    ``ref``, ``alt``. ``genome`` is a FASTA path (indexed) or a genomepy
    name. The model is wrapped with :func:`wrap` if needed and
    :func:`grelu.variant.predict_variant_effects` does the work, so the
    result is identical to calling gReLU directly. With the default
    log-scale heads, ``compare_func="subtract"`` gives the log fold change
    alt vs ref.
    """
    _grelu()
    from grelu.variant import predict_variant_effects as _pve

    lm = wrap(model, head=head, seq_len=seq_len, **wrap_kwargs)
    with _no_lightning_logs():
        return _pve(variants, lm, devices=devices, seq_len=seq_len, batch_size=batch_size,
                    num_workers=num_workers, genome=_genome_arg(genome), rc=rc,
                    max_seq_shift=max_seq_shift, compare_func=compare_func,
                    return_ad=return_ad)


def ism(model, seqs, *, genome=None, head=None, start_pos: int = 0,
        end_pos: Optional[int] = None, compare_func="subtract", devices="cpu",
        batch_size: int = 64, return_df: bool = True, **wrap_kwargs):
    """In-silico mutagenesis through gReLU's ``ISM_predict`` (exact
    pass-through). ``seqs``: a sequence string, one-hot array or intervals
    with ``genome``."""
    _grelu()
    from grelu.interpret.score import ISM_predict

    lm = wrap(model, head=head, **wrap_kwargs)
    with _no_lightning_logs():
        return ISM_predict(seqs, lm, genome=None if genome is None else _genome_arg(genome),
                           start_pos=start_pos, end_pos=end_pos, compare_func=compare_func,
                           devices=devices, batch_size=batch_size, return_df=return_df)


def attributions(model, seqs, *, genome=None, head=None, method: str = "deepshap",
                 device="cpu", n_shuffles: int = 20, seed: Optional[int] = 0,
                 hypothetical: bool = False, **wrap_kwargs) -> np.ndarray:
    """Base-level attributions through gReLU's ``get_attributions``
    (``deepshap``, ``saliency``, ``inputxgradient``,
    ``integratedgradients``). Returns ``(N, 4, L)``."""
    _grelu()
    from grelu.interpret.score import get_attributions

    lm = wrap(model, head=head, **wrap_kwargs)
    with _no_lightning_logs():
        return get_attributions(lm, seqs, genome=None if genome is None else _genome_arg(genome),
                                device=device, method=method, n_shuffles=n_shuffles,
                                seed=seed, hypothetical=hypothetical)


def pseudobulk_anndata(adata, groupby: str, *, layer: Optional[str] = None,
                       peak_coords: Optional[pd.DataFrame] = None, min_cells: int = 1):
    """Groups x peaks pseudobulk AnnData in gReLU's training layout.

    ``adata`` is cells x peaks (counts). Cells are summed per ``groupby``
    category; ``var`` gets ``chrom``/``start``/``end`` (parsed from
    ``chr:start-end`` / ``chr_start_end`` names unless ``peak_coords`` is
    given), so the result feeds ``grelu.data.dataset.AnnDataSeqDataset``
    directly (tasks = groups, intervals = peaks).
    """
    import anndata as ad
    import scipy.sparse as sp

    X = adata.layers[layer] if layer else adata.X
    labels = adata.obs[groupby].astype("category")
    cats = [c for c in labels.cat.categories if (labels == c).sum() >= min_cells]
    rows = []
    for c in cats:
        m = (labels == c).to_numpy()
        s = X[m].sum(axis=0)
        rows.append(np.asarray(s).ravel())
    Y = np.vstack(rows).astype(np.float32)
    if peak_coords is None:
        names = pd.Index(adata.var_names).astype(str)
        parts = names.str.replace(":", "_", regex=False).str.replace("-", "_", regex=False).str.rsplit("_", n=2)
        peak_coords = pd.DataFrame({"chrom": parts.str[0], "start": parts.str[1].astype(int),
                                    "end": parts.str[2].astype(int)}, index=adata.var_names)
    var = peak_coords.copy()
    var.index = adata.var_names.astype(str)
    obs = pd.DataFrame({"n_cells": [int((labels == c).sum()) for c in cats]},
                       index=pd.Index([str(c) for c in cats], name="name"))
    return ad.AnnData(X=Y, obs=obs, var=var)
