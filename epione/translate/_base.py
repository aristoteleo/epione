"""Method registry, the common model interface and :func:`fit`.

Each method lives in its own module (``_<method>.py``) that defines a
:class:`TranslationModel` subclass decorated with :func:`register`. The
registry maps a method name to that module, so torch and the method's code
are imported only when the method is used.
"""
from __future__ import annotations

import importlib
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ._prep import check_paired, resolve_split

# name -> module that registers it (imported on first use)
_METHOD_MODULES = {
    "scbutterfly": "._scbutterfly",
    "babel": "._babel",
    "scpair": "._scpair",
}
_REGISTRY: dict = {}

DIRECTIONS = ("rna2atac", "atac2rna")


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.method = name
        return cls

    return deco


def available_methods() -> list[str]:
    """Names accepted by :func:`fit`."""
    return sorted(_METHOD_MODULES)


def get_method(name: str):
    key = name.lower()
    if key not in _METHOD_MODULES:
        raise ValueError(f"unknown method {name!r}; available: {available_methods()}")
    if key not in _REGISTRY:
        try:
            importlib.import_module(_METHOD_MODULES[key], __package__)
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"method {name!r} needs PyTorch; install it with `pip install epione[translate]`"
            ) from e
    return _REGISTRY[key]


def _device(device):
    import torch

    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TranslationModel:
    """A trained RNA <-> ATAC translator.

    Created by :func:`fit`; :meth:`predict` translates new cells. Subclasses
    implement ``_fit``, ``_predict``, ``_state`` and ``_from_state``.
    """

    method: str = ""
    directions: tuple = DIRECTIONS
    default_sparsify: Optional[float] = 1e-4

    rna_features: pd.Index
    atac_features: pd.Index
    history: dict

    # ------------------------------------------------------------------
    def predict(
        self,
        adata: AnnData,
        direction: Optional[str] = None,
        *,
        device=None,
        sparsify="auto",
    ) -> AnnData:
        """Translate the cells in ``adata`` to the other modality.

        Parameters
        ----------
        adata
            Preprocessed profiles of the source modality (same preprocessing as
            for training). Features are matched to the training features by
            name; missing ones are an error.
        direction
            ``"rna2atac"`` or ``"atac2rna"``. Inferred from ``adata``'s features
            when omitted.
        sparsify
            Values below this are set to zero and the result is stored sparse.
            ``"auto"`` uses the method's convention (scButterfly: 1e-4, as it
            returns its predictions; BABEL: dense). ``None`` keeps a dense array.

        Returns
        -------
        AnnData with ``adata.obs`` and the target modality's features.
        """
        if direction is None:
            direction = self._infer_direction(adata)
        if direction not in self.directions:
            raise ValueError(f"{self.method} supports {self.directions}, not {direction!r}")
        src = self.rna_features if direction == "rna2atac" else self.atac_features
        dst = self.atac_features if direction == "rna2atac" else self.rna_features
        missing = src.difference(adata.var_names)
        if len(missing):
            raise ValueError(f"{len(missing)} training features are missing from adata, e.g. {list(missing[:3])}")
        X = adata[:, src].X if not adata.var_names.equals(src) else adata.X
        X = sp.csr_matrix(X, dtype=np.float32)
        Y = self._predict(X, direction, _device(device))
        if sparsify == "auto":
            sparsify = self.default_sparsify
        if sparsify is not None:
            Y = np.where(Y < sparsify, 0.0, Y).astype(np.float32)
            Y = sp.csr_matrix(Y)
        out = AnnData(X=Y, obs=adata.obs.copy(), var=pd.DataFrame(index=dst.copy()))
        out.uns["translate"] = {"method": self.method, "direction": direction}
        return out

    def _infer_direction(self, adata):
        n_r = len(self.rna_features.intersection(adata.var_names))
        n_a = len(self.atac_features.intersection(adata.var_names))
        if n_r == len(self.rna_features) and n_r >= n_a:
            return "rna2atac"
        if n_a == len(self.atac_features):
            return "atac2rna"
        raise ValueError("cannot tell the modality of adata from its features; pass direction=")

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the trained model (weights, configuration, feature names)."""
        import torch

        torch.save({"method": self.method, "rna_features": list(self.rna_features),
                    "atac_features": list(self.atac_features), "history": self.history,
                    "state": self._state()}, path)

    def __repr__(self):
        n_r = len(getattr(self, "rna_features", []))
        n_a = len(getattr(self, "atac_features", []))
        return f"<{type(self).__name__} {self.method}: {n_r} genes <-> {n_a} peaks>"


def load(path: str) -> TranslationModel:
    """Load a model written by :meth:`TranslationModel.save`."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    cls = get_method(blob["method"])
    obj = cls._from_state(blob["state"], len(blob["rna_features"]))
    obj.rna_features = pd.Index(blob["rna_features"])
    obj.atac_features = pd.Index(blob["atac_features"])
    obj.history = blob["history"]
    return obj


def fit(
    rna: AnnData,
    atac: AnnData,
    method: str = "scbutterfly",
    *,
    direction: str = "both",
    train_idx: Optional[Sequence[int]] = None,
    val_idx: Optional[Sequence[int]] = None,
    device=None,
    seed: int = 0,
    fast: bool = False,
    verbose: bool = False,
    threads: Optional[int] = None,
    **kwargs,
) -> TranslationModel:
    """Train a cross-modality translator on paired RNA / ATAC profiles.

    Parameters
    ----------
    rna, atac
        Preprocessed profiles of the same cells (see :func:`prepare`).
    method
        One of :func:`available_methods`.
    direction
        ``"rna2atac"``, ``"atac2rna"`` or ``"both"``. Methods that learn both
        directions jointly (scButterfly) train the same model either way; the
        value sets the default for :meth:`TranslationModel.predict`.
    train_idx, val_idx
        Row indices for training and early-stopping validation. By default all
        cells are used for training, with a random 20% held out for validation.
    device
        ``"cuda"``, ``"cpu"`` or ``"auto"`` (default).
    seed
        Seeds the split, the weight initialisation and the training RNG.
    fast
        Use the method's validated fast path (see the method's documentation).
    threads
        CPU threads for torch.
    **kwargs
        Method hyper-parameters.
    """
    import torch

    check_paired(rna, atac)
    if direction not in ("both",) + DIRECTIONS:
        raise ValueError(f"direction must be 'both', 'rna2atac' or 'atac2rna', not {direction!r}")
    if threads:
        torch.set_num_threads(int(threads))
    cls = get_method(method)
    model = cls(fast=fast, **kwargs)
    tr, va = resolve_split(rna.n_obs, train_idx, val_idx, seed=seed)
    model.rna_features = rna.var_names.copy()
    model.atac_features = atac.var_names.copy()
    model.direction = direction
    model._fit(rna, atac, tr, va, _device(device), seed, verbose)
    return model
