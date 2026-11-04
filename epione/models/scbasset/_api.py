"""High-level scBasset model: fit, cell embedding, denoised accessibility,
motif-injection TF activity (calico/scBasset ``scbasset/utils.py``)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from ._data import SEQ_LEN, binarize, encode_sequences, read_fasta, split_train_test_val

MOTIF_URL = "https://storage.googleapis.com/scbasset_tutorial_data/Homo_sapiens_motif_fasta.tar.gz"
MOTIF_MD5 = "md5:3c736cc3897314c79062053af2d49361"


def motif_library(motif_dir: Optional[os.PathLike] = None, progressbar: bool = False) -> Path:
    """The scBasset human motif-injection library (CIS-BP motifs injected into
    dinucleotide-shuffled peaks; 306 MB), downloaded once and md5-checked.

    Returns the folder holding ``shuffled_peaks.fasta`` and ``shuffled_peaks_motifs/``.
    Pass ``motif_dir`` to use an already extracted copy.
    """
    if motif_dir is not None:
        p = Path(motif_dir)
        if (p / "shuffled_peaks.fasta").exists():
            return p
        if (p / "Homo_sapiens_motif_fasta" / "shuffled_peaks.fasta").exists():
            return p / "Homo_sapiens_motif_fasta"
    import pooch

    cache = Path(motif_dir) if motif_dir is not None else Path(pooch.os_cache("epione")) / "scbasset_motifs"
    pooch.retrieve(MOTIF_URL, known_hash=MOTIF_MD5, fname="Homo_sapiens_motif_fasta.tar.gz",
                   path=str(cache), processor=pooch.Untar(extract_dir="."), progressbar=progressbar)
    return cache / "Homo_sapiens_motif_fasta"


class ScBasset:
    """A trained (or trainable) scBasset model on one peak set.

    Arguments:
        codes: ``(n_peaks, 1344)`` int8 sequence codes (:func:`peak_sequences`).
        X: cells x peaks counts (binarised internally).
        bottleneck: size of the bottleneck / cell embedding (32 in the paper).
        split_seed: seed of scBasset's 90/5/5 train/test/val peak split.
    """

    def __init__(self, codes: np.ndarray, X, bottleneck: int = 32, split_seed: int = 10,
                 train_ratio: float = 0.9):
        from ._model import ScBassetNet

        self.codes = np.ascontiguousarray(codes, dtype=np.int8)
        self.Y = binarize(X)
        if self.Y.shape[0] != self.codes.shape[0]:
            raise ValueError(f"{self.codes.shape[0]} sequences for {self.Y.shape[0]} peaks")
        self.n_peaks, self.n_cells = self.Y.shape
        self.train_idx, self.test_idx, self.val_idx = split_train_test_val(
            self.n_peaks, seed=split_seed, train_ratio=train_ratio)
        self.net = ScBassetNet(self.n_cells, bottleneck=bottleneck, seq_len=self.codes.shape[1])
        self.history: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, epochs: int = 1000, batch_size: int = 128, lr: float = 0.01, patience: int = 50,
            min_delta: float = 1e-6, seed: int = 0, device: str = "auto", verbose: bool = True,
            log_every: int = 10) -> "ScBasset":
        """Train with the recipe of ``scbasset_train.py`` (defaults are its defaults)."""
        import torch
        from ._model import ScBassetNet, train

        torch.manual_seed(seed)
        # re-create the network so the initialisation follows ``seed``
        self.net = ScBassetNet(self.n_cells, bottleneck=self.net.bottleneck, seq_len=self.codes.shape[1])
        self.history = train(self.net, self.codes, self.Y, self.train_idx, self.val_idx, epochs=epochs,
                             batch_size=batch_size, lr=lr, patience=patience, min_delta=min_delta,
                             seed=seed, device=device, verbose=verbose, log_every=log_every)
        return self

    def load_keras_weights(self, path) -> "ScBasset":
        """Load an official ``best_model.h5`` (no TensorFlow needed)."""
        from ._model import load_keras_h5

        load_keras_h5(self.net, path)
        return self

    # -------------------------------------------------------------- outputs
    def cell_embedding(self) -> np.ndarray:
        """``(n_cells, bottleneck)``: the final layer's kernel (``get_cell_embedding``)."""
        return self.net.final.weight.detach().cpu().numpy().copy()

    def intercept(self) -> np.ndarray:
        """``(n_cells,)``: the final layer's bias, which tracks sequencing depth."""
        return self.net.final.bias.detach().cpu().numpy().copy()

    def predict(self, idx=None, codes=None, normalize: bool = False, sigmoid: bool = True,
                batch_size: int = 256, device: str = "auto") -> np.ndarray:
        """``(n_peaks, n_cells)`` predicted accessibility of the model's peaks
        (``idx`` selects some) or of new sequences ``codes``.

        ``normalize=True`` drops the cell intercept (``imputation_Y_normalize``,
        sigmoid scale), giving depth-corrected, denoised accessibility.
        """
        from ._model import predict_logits

        z = predict_logits(self.net, self.codes if codes is None else codes, idx=idx,
                           batch_size=batch_size, device=device, normalize=normalize)
        from scipy.special import expit

        return expit(z) if sigmoid else z

    def impute(self, normalize: bool = True, **kw) -> np.ndarray:
        """Denoised accessibility, cells x peaks (dense float32)."""
        return self.predict(normalize=normalize, **kw).T.astype(np.float32)

    def test_auroc(self, device: str = "auto") -> np.ndarray:
        """Per-cell ROC AUC on the held-out test peaks (sklearn, eval mode)."""
        from sklearn.metrics import roc_auc_score

        p = self.predict(idx=self.test_idx, device=device)
        y = self.Y[self.test_idx].toarray()
        out = np.full(self.n_cells, np.nan)
        for j in range(self.n_cells):
            if 0 < y[:, j].sum() < y.shape[0]:
                out[j] = roc_auc_score(y[:, j], p[:, j])
        return out

    def motif_score(self, tf: str, motif_dir: Optional[os.PathLike] = None, seed: int = 0,
                    device: str = "auto") -> np.ndarray:
        """TF activity per cell by motif injection (``motif_score``): mean
        depth-normalised sigmoid accessibility of shuffled peaks carrying the
        motif minus that of the shuffled background, z-scored over cells."""
        lib = motif_library(motif_dir)
        fa_m = lib / "shuffled_peaks_motifs" / f"{tf}.fasta"
        if not fa_m.exists():
            raise ValueError(f"{tf} is not in the motif library ({lib})")
        cm = encode_sequences(read_fasta(fa_m), seed=seed)
        cb = encode_sequences(read_fasta(lib / "shuffled_peaks.fasta"), seed=seed)
        pm = self.predict(codes=cm, normalize=True, device=device)
        pb = self.predict(codes=cb, normalize=True, device=device)
        s = pm.mean(0) - pb.mean(0)
        return (s - s.mean()) / s.std()

    @staticmethod
    def available_motifs(motif_dir: Optional[os.PathLike] = None) -> list:
        lib = motif_library(motif_dir)
        return sorted(p.stem for p in (lib / "shuffled_peaks_motifs").glob("*.fasta"))

    # ----------------------------------------------------------- save/load
    def save(self, path) -> None:
        import torch

        torch.save({"state_dict": self.net.state_dict(), "bottleneck": self.net.bottleneck,
                    "n_cells": self.n_cells, "history": self.history}, path)

    def load(self, path) -> "ScBasset":
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ck["state_dict"])
        self.history = ck.get("history", {})
        return self
