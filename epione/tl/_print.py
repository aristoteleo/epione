"""PRINT multi-scale footprinting and TF binding scores (scPrinter).

PRINT (Hu *et al.*, *Nature* 2025) asks, at every base and every scale,
whether Tn5 cut a window *less* than its sequence bias predicts relative
to the flanks -- the signature of something bound there. Small scales
pick out transcription factors, large ones nucleosomes. scPrinter
(Buenrostro lab) brought it to single-cell data and added a model that
turns the multi-scale footprint around a site into a binding probability.

What this module takes from scPrinter, and what it leaves
---------------------------------------------------------
The numbers are scPrinter's: its window geometry, its learned null
(the "dispersion model"), its smoothing, its TF-binding model and the
published weights. They are reproduced exactly -- see ``Exactness``.

The machinery around them is not. scPrinter keeps state in a ``printer``
object over a backed AnnData plus a ``_supp/`` side-car directory, passes
models to worker processes through ``globals()``, and needs snapATAC2's
genome-wide insertion matrix and a 10.9 GB precomputed bias file (mm10).
Here the same computation is three plain inputs -- an insertion track
from :class:`~epione._insertions.InsertionReader`, a bias track from any
:class:`~epione.tl.Tn5Bias`, and a model loaded once -- and results come
back as dataclasses.

Exactness
---------
Given the same insertion and bias tracks, the scores match scPrinter's
``regionFootprintScore`` / ``getRegionBindingScore`` bit for bit when
torch is installed: window sums are taken in float64 exactly as
``rz_conv`` does, the dispersion MLP runs through the same TorchScript
ops in float32, and the p-value / -log10 / smoothing steps keep
scPrinter's order and NaN/inf handling. Without torch the MLP falls back
to numpy. That is not *guaranteed* bit-identical -- BLAS may sum a matrix
product in a different order depending on shape and threading -- but the
network is tiny (5 -> 32 -> 4) and on the CPUs tested the two agree
exactly.

Weights come from scPrinter's Zenodo record and are cached by pooch;
set ``EPIONE_DATA_DIR`` to choose where.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "DispersionModel",
    "PRINT_SCALES",
    "TFBindingResult",
    "fetch_print_model",
    "print_footprint_pvalues",
    "print_footprint_score",
    "tf_binding_score",
]

#: scPrinter's default footprint radii (``np.arange(2, 101)``).
PRINT_SCALES = tuple(range(2, 101))

# ---------------------------------------------------------------------------
# Pretrained weights
# ---------------------------------------------------------------------------

_ZENODO = "https://zenodo.org/records/15224770/files/"
_MODELS = {
    # key: (filename, md5)
    "dispersion": ("dispersion_model_py_v2.h5", "md5:341a011610c7db9a1418917e359d3169"),
    "tn5_bias": ("Tn5_NN_model_py_v2.pt", "md5:5ea6c26fcd54ad2aff3d9705ed54dd3c"),
    "tf_binding": ("TFBS_model_py.pt", "md5:5bd79a9c4f3374241a6f4341eb39fe2c"),
}
_registry = None


def fetch_print_model(key: str) -> str:
    """Path to a published PRINT model, downloading it on first use.

    Parameters
    ----------
    key
        ``'dispersion'`` (0.7 MB), ``'tn5_bias'`` (0.1 MB) or
        ``'tf_binding'`` (0.7 MB).
    """
    global _registry
    if key not in _MODELS:
        raise KeyError(f"unknown PRINT model {key!r}; one of {sorted(_MODELS)}")
    import pooch
    if _registry is None:
        _registry = pooch.create(
            path=pooch.os_cache("epione") / "print",
            base_url=_ZENODO,
            env="EPIONE_DATA_DIR",
            registry={name: md5 for name, md5 in _MODELS.values()},
        )
    return _registry.fetch(_MODELS[key][0])


# ---------------------------------------------------------------------------
# Window sums -- scPrinter's rz_conv / footprintWindowSum
# ---------------------------------------------------------------------------

def _rz_conv(a: np.ndarray, n: int) -> np.ndarray:
    """Centred running sum of width ``2n`` (scPrinter ``utils.rz_conv``).

    The zero padding is float64, so the result is float64 whatever the
    input dtype -- the same promotion scPrinter gets, and the reason its
    window sums never accumulate float32 rounding.
    """
    if n == 0:
        return a
    shapes = np.array(a.shape)
    shapes[-1] = n
    a = np.concatenate([np.zeros(shapes), a, np.zeros(shapes)], axis=-1)
    ret = np.nancumsum(a, axis=-1)
    return ret[..., n * 2:] - ret[..., : -n * 2]


def _window_sums(x: np.ndarray, radius: int, flank_radius: int):
    """``(left_flank, centre, right_flank)`` sums at every position.

    Centre spans ``2 * radius`` bases; each flank spans
    ``2 * (flank_radius // 2)`` bases and sits right against the centre.
    (So a "flank radius" of R gives flanks about R wide, not 2R.)
    """
    half = int(flank_radius / 2)
    width = x.shape[-1]
    shift = half + radius
    shapes = list(x.shape)
    shapes[-1] = shift
    left = _rz_conv(np.concatenate([np.zeros(shapes), x], axis=-1), half)[..., :width]
    right = _rz_conv(np.concatenate([x, np.zeros(shapes)], axis=-1), half)[..., shift:]
    centre = _rz_conv(x, radius)
    return left, centre, right


# ---------------------------------------------------------------------------
# Dispersion model -- the learned null
# ---------------------------------------------------------------------------

_mlp_jit = None


def _torch_mlp():
    """scPrinter's ``predictDispersion_jit``, compiled once."""
    global _mlp_jit
    if _mlp_jit is None:
        import torch
        import torch.nn.functional as F

        @torch.jit.script
        def predict(x, w0, b0, w1, b1):
            x = F.relu(F.linear(x, w0, b0))
            x = F.linear(x, w1, b1)
            return x

        _mlp_jit = predict
    return _mlp_jit


@dataclass
class DispersionModel:
    """PRINT's background model of the centre/(centre+flank) cut ratio.

    For each footprint radius a small MLP (5 -> 32 -> 4) maps the bias
    sums in the three windows and the local coverage to the expected
    mean and spread of the cut ratio on each side. A footprint is a
    ratio well below that expectation; this learned null is what
    separates PRINT from a plain binomial test, which ignores the
    overdispersion of real Tn5 data.

    Load with :meth:`load`; index by radius.
    """

    params: Dict[int, dict]
    version: str = ""
    path: str = ""

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "DispersionModel":
        """Read a dispersion model (default: the published v2)."""
        import h5py

        path = str(path) if path is not None else fetch_print_model("dispersion")

        def read(node):  # scPrinter utils.load_entire_hdf5
            if isinstance(node, h5py.Dataset):
                return node[()]
            return {k: read(v) for k, v in node.items()}

        with h5py.File(path, "r") as f:
            raw = read(f)
        version = raw.pop("version", b"")
        if isinstance(version, bytes):
            version = version.decode()
        params = {}
        for k, v in raw.items():
            w = v["modelWeights"]
            v["weights"] = [np.asarray(w[n], dtype=np.float32)
                            for n in ("ELT1", "ELT2", "ELT3", "ELT4")]
            params[int(k)] = v
        return cls(params=params, version=str(version), path=path)

    @property
    def scales(self) -> List[int]:
        return sorted(self.params)

    def __getitem__(self, radius: int) -> dict:
        try:
            return self.params[int(radius)]
        except KeyError:
            raise KeyError(
                f"no dispersion model for radius {radius}; "
                f"available {self.scales[0]}..{self.scales[-1]}"
            ) from None

    def predict(self, radius: int, features_scaled: np.ndarray) -> np.ndarray:
        """Run the MLP for one radius on already-standardised features."""
        w = self[radius]["weights"]
        try:
            import torch
        except ImportError:
            x = features_scaled.astype(np.float32)
            x = np.maximum(x @ w[0].T + w[1], 0)
            return x @ w[2].T + w[3]
        with torch.no_grad():
            return _torch_mlp()(
                torch.from_numpy(features_scaled).float(),
                *[torch.from_numpy(a) for a in w],
            ).numpy()


# ---------------------------------------------------------------------------
# Footprint p-values and scores
# ---------------------------------------------------------------------------

def print_footprint_pvalues(
    insertions: np.ndarray,
    bias: np.ndarray,
    radius: int,
    model: DispersionModel,
    *,
    flank_radius: Optional[int] = None,
) -> np.ndarray:
    """One-sided p-value of a footprint at every position, one radius.

    Port of scPrinter ``footprint.footprintScoring`` (``return_pval``).

    Parameters
    ----------
    insertions
        ``[n_groups, L]`` observed cuts.
    bias
        ``[L]`` expected cuts from a :class:`~epione.tl.Tn5Bias`.
    radius
        Footprint radius R (centre is 2R bp).
    model
        :class:`DispersionModel`; must be fitted with the same bias
        model family (PRINT) or its null is meaningless.
    flank_radius
        Defaults to ``radius``, as scPrinter does.

    Returns
    -------
    np.ndarray
        ``[n_groups, L]`` p-values. Positions with no cuts on either
        side are 1.
    """
    from scipy.stats import norm

    flank_radius = radius if flank_radius is None else flank_radius
    disp = model[radius]
    insertions = np.asarray(insertions)
    n_groups = len(insertions)

    b_left, b_centre, b_right = _window_sums(bias, radius, flank_radius)
    i_left, i_centre, i_right = _window_sums(insertions, radius, flank_radius)
    total_left = i_centre + i_left
    total_right = i_centre + i_right

    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        features = np.stack(
            [
                np.array([b_left] * n_groups),
                np.array([b_right] * n_groups),
                np.array([b_centre] * n_groups),
                np.log10(total_left),
                np.log10(total_right),
            ],
            axis=-1,
        )
        scaled = (features - disp["featureMean"]) / disp["featureSD"]
        pred = model.predict(radius, scaled)
        pred = pred * disp["targetSD"]
        pred = pred + disp["targetMean"]

        mean_l, sd_l = pred[..., 0], pred[..., 1]
        mean_r, sd_r = pred[..., 2], pred[..., 3]
        sd_l[sd_l < 0] = 0
        sd_r[sd_r < 0] = 0
        ratio_l = i_centre / total_left
        ratio_r = i_centre / total_right

        p_l = norm.cdf(ratio_l, mean_l, sd_l)
        p_l[np.isnan(p_l)] = 1
        p_r = norm.cdf(ratio_r, mean_r, sd_r)
        p_r[np.isnan(p_r)] = 1
    p = np.maximum(p_l, p_r)
    p[(total_left < 1) | (total_right < 1)] = 1
    return p


def _pvalue_to_score(p: np.ndarray, smooth_radius: int) -> np.ndarray:
    """scPrinter ``regionFootprintScore`` post-processing.

    -log10 p, a ``2s`` max-filter, then a ``2s`` running mean. Only p=0
    (score inf) is capped at 20; tiny non-zero p keep their real score.
    """
    from scipy.ndimage import maximum_filter

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.array(p, copy=True)
        p[np.isnan(p)] = 1
        s = -np.log10(p)
        s[np.isnan(s)] = 0
        s[np.isinf(s)] = 20
        if smooth_radius > 0:
            size = [0] * s.ndim
            size[-1] = 2 * smooth_radius
            s = maximum_filter(s, tuple(size), origin=-1)
            s = _rz_conv(s, smooth_radius) / (2 * smooth_radius)
        s[np.isnan(s)] = 0
        s[np.isinf(s)] = 20
    return s


def print_footprint_score(
    insertions: np.ndarray,
    bias: np.ndarray,
    model: DispersionModel,
    *,
    scales: Sequence[int] = PRINT_SCALES,
    smooth_radius: Optional[Union[int, str]] = "auto",
) -> np.ndarray:
    """Multi-scale PRINT footprint score of a region.

    Parameters
    ----------
    insertions
        ``[n_groups, L]`` observed cuts.
    bias
        ``[L]`` expected cuts.
    model
        :class:`DispersionModel`.
    scales
        Footprint radii. Default 2..100.
    smooth_radius
        ``'auto'`` -> ``int(R / 2)`` per radius (scPrinter's default in
        ``regionFootprintScore`` and its TF binding path); an int fixes
        it for every radius (scPrinter's ``get_footprint_score`` uses 5);
        0 turns smoothing off.

    Returns
    -------
    np.ndarray
        ``[n_groups, n_scales, L]`` smoothed -log10 p.
    """
    insertions = np.asarray(insertions)
    out = None
    for si, r in enumerate(scales):
        sr = (int(r / 2) if smooth_radius in ("auto", None)
              else int(smooth_radius))
        s = _pvalue_to_score(
            print_footprint_pvalues(insertions, bias, int(r), model), sr)
        if out is None:
            out = np.zeros((s.shape[0], len(scales), s.shape[-1]))
        out[:, si, :] = s
    return out


# ---------------------------------------------------------------------------
# TF binding score
# ---------------------------------------------------------------------------

@dataclass
class TFBindingResult:
    """Per-site TF binding probabilities from PRINT footprints.

    Attributes
    ----------
    groups
        Cell groups, in row order of :attr:`score`.
    sites
        One row per scored site: ``chrom``, ``center``, ``strand``,
        ``region`` (``chrom:start-end`` of the region it was scored in)
        and ``position`` (offset from that region's start).
    score
        ``[n_groups, n_sites]`` binding probability in [0, 1].
    regions
        The regions scored, with per-group insertion and bias tracks
        kept in :attr:`tracks` for plotting.
    """

    groups: List[str]
    sites: pd.DataFrame
    score: np.ndarray
    regions: pd.DataFrame
    tracks: Dict[str, dict] = field(default_factory=dict, repr=False)
    model_version: str = ""
    bias_model: str = ""

    def to_frame(self) -> pd.DataFrame:
        """Long table: one row per (site, group)."""
        df = self.sites.loc[self.sites.index.repeat(len(self.groups))].copy()
        df["group"] = np.tile(self.groups, len(self.sites))
        df["score"] = self.score.T.reshape(-1)
        return df.reset_index(drop=True)


def _load_tfbs_model(path=None):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError("tf_binding_score needs torch for the PRINT model") from exc
    path = str(path) if path is not None else fetch_print_model("tf_binding")
    model = torch.jit.load(path, map_location="cpu").eval()
    return model, [int(s) for s in model.scales], str(model.version)


def _tile_sites(start: int, end: int, tile: int) -> pd.DataFrame:
    """scPrinter's tiling: ``pyranges.window(tile)`` with the last tile
    widened by one base."""
    starts = np.arange(start, end, tile)
    ends = np.minimum(starts + tile, end)
    ends[-1] += 1
    return pd.DataFrame({"Start": starts, "End": ends, "Strand": "+"})


def _region_binding_score(
    insertions, bias, region_start, sites, scales, disp, model,
    context_radius=100,
):
    """Port of scPrinter ``TFBS.getRegionBindingScore`` for one region.

    Returns ``(relative positions, keep mask, scores [n_groups, n_kept])``.
    """
    import torch

    width = len(bias)
    fp = None
    for si, r in enumerate(scales):
        s = _pvalue_to_score(
            print_footprint_pvalues(insertions, bias, r, disp), int(r / 2))
        if fp is None:
            fp = np.zeros((s.shape[0], len(scales), s.shape[1]), dtype="float32")
        fp[:, si, :] = s

    rel = (np.array(sites["Start"] + sites["End"]) * 0.5 - int(region_start)).astype("int")
    keep = (rel > context_radius) & (rel <= (width - context_radius))
    rel_k = rel[keep]
    if len(rel_k) == 0:
        return rel, keep, np.zeros((fp.shape[0], 0), dtype=np.float32)

    first = rel_k - context_radius - 1
    span = 2 * context_radius + 1
    l, m, n = fp.shape
    s0, s1, s2 = fp.strides
    windows = np.lib.stride_tricks.as_strided(
        fp, shape=(l, m, n - span + 1, span), strides=(s0, s1, s2, s2)
    )[:, :, first]
    minus = np.isin(np.asarray(sites["Strand"])[keep], ["-", "-1", -1])
    windows[:, :, minus] = windows[:, :, minus][..., ::-1]
    data = np.transpose(windows, axes=[0, 2, 1, 3]).reshape(
        (windows.shape[0], windows.shape[2], -1))
    with torch.no_grad():
        score = model(torch.from_numpy(data).float()).numpy()
    return rel, keep, score[..., 0]


def tf_binding_score(
    adata,
    regions,
    *,
    groupby: str,
    genome=None,
    bias=None,
    sites: Optional[pd.DataFrame] = None,
    tile: int = 10,
    context_radius: int = 100,
    min_cells_per_group: int = 10,
    model=None,
    dispersion: Optional[DispersionModel] = None,
    keep_tracks: bool = True,
    verbose: bool = True,
) -> TFBindingResult:
    """PRINT TF binding probability for every site in some regions.

    Footprints are computed over each whole region (a peak, typically
    ~1 kb) at the model's six scales, then the 201-bp multi-scale
    footprint around each site is scored by PRINT's pretrained binding
    model. Sites closer than ``context_radius`` to a region edge are
    skipped, as in scPrinter.

    Parameters
    ----------
    adata
        AnnData with ``uns['files']['fragments']`` and ``obs[groupby]``.
    regions
        DataFrame with ``chrom``, ``start``, ``end`` -- the regions in
        which to score (e.g. peaks).
    groupby
        ``obs`` column defining pseudobulks.
    genome
        Genome or FASTA for the default PRINT bias.
    bias
        A :class:`~epione.tl.Tn5Bias`. Defaults to the PRINT CNN on
        ``genome``. The binding model was trained on PRINT-bias
        footprints, so a different bias family will give uncalibrated
        scores.
    sites
        Motif sites (``chrom``, ``start``, ``end``, optional ``strand``).
        If omitted, each region is tiled every ``tile`` bp -- an
        unbiased scan for any bound factor.
    tile, context_radius
        As in scPrinter.

    Returns
    -------
    :class:`TFBindingResult`
    """
    from .._insertions import InsertionReader
    from ._tn5_bias import PrintBias, Tn5Bias
    from ..core import console

    if bias is None:
        bias = PrintBias(genome)
    if not isinstance(bias, Tn5Bias):
        raise TypeError("bias must be a Tn5Bias (see epi.tl.tn5_bias)")
    disp = dispersion if dispersion is not None else DispersionModel.load()
    net, scales, version = _load_tfbs_model(model)

    groups, names, reader = _groups_and_reader(
        adata, groupby, min_cells_per_group, shift=bias.shift)
    regs = _as_intervals(regions)
    site_df = None if sites is None else _as_intervals(sites, need_strand=True)

    rows, scores, tracks = [], [], {}
    for _, r in regs.iterrows():
        chrom, s, e = r["chrom"], int(r["start"]), int(r["end"])
        rid = f"{chrom}:{s}-{e}"
        ins = reader.group_insertions(chrom, s, e, groups)
        b = bias.track(chrom, s, e)
        if site_df is None:
            sub = _tile_sites(s, e, tile)
        else:
            m = ((site_df["chrom"] == chrom) & (site_df["start"] >= s)
                 & (site_df["end"] <= e))
            sub = site_df.loc[m].rename(
                columns={"start": "Start", "end": "End", "strand": "Strand"})
            if sub.empty:
                continue
        rel, keep, sc = _region_binding_score(
            ins, b, s, sub.reset_index(drop=True), scales, disp, net,
            context_radius=context_radius)
        if sc.shape[1] == 0:
            continue
        kept = sub.reset_index(drop=True).loc[keep]
        rows.append(pd.DataFrame({
            "chrom": chrom,
            "center": s + rel[keep],
            "strand": np.asarray(kept["Strand"]),
            "region": rid,
            "position": rel[keep],
        }))
        scores.append(sc)
        if keep_tracks:
            tracks[rid] = {"insertions": ins, "bias": b, "start": s, "end": e}

    if not rows:
        raise ValueError("no site lies at least context_radius from a region edge")
    site_table = pd.concat(rows, ignore_index=True)
    score = np.concatenate(scores, axis=1)
    if verbose:
        console.level1(
            f"tf_binding_score: {score.shape[1]:,} sites in {len(tracks) or len(regs)} "
            f"region(s) x {len(names)} groups"
        )
    return TFBindingResult(
        groups=names, sites=site_table, score=score, regions=regs,
        tracks=tracks, model_version=version, bias_model=bias.name,
    )


# ---------------------------------------------------------------------------
# Shared helpers (also used by multi_scale_footprint_region)
# ---------------------------------------------------------------------------

def _groups_and_reader(adata, groupby, min_cells_per_group, *, shift):
    """Group cells by ``obs[groupby]`` and open a reader with ``shift``."""
    from .._insertions import InsertionReader, groups_to_matrix

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise ValueError(
            "adata.uns['files']['fragments'] missing -- run "
            "epi.pp.import_fragments first"
        )
    labels = adata.obs[groupby].astype(str).to_numpy()
    names = sorted(g for g, n in pd.Series(labels).value_counts().items()
                   if n >= min_cells_per_group)
    if not names:
        raise ValueError(f"no {groupby!r} group has >= {min_cells_per_group} cells")
    members = [np.flatnonzero(labels == g) for g in names]
    reader = InsertionReader(
        adata.uns["files"]["fragments"], list(map(str, adata.obs_names)),
        chrom_sizes=dict(adata.uns.get("reference_sequences", {}) or {}),
        shift=shift,
    )
    return groups_to_matrix(members, reader.n_cells), names, reader


def _as_intervals(df, need_strand: bool = False) -> pd.DataFrame:
    """Normalise a region/site table to ``chrom, start, end[, strand]``."""
    df = pd.DataFrame(df).copy()
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    for want, cands in (("chrom", ("chrom", "chromosome", "seqnames", "seqname")),
                        ("start", ("start",)), ("end", ("end",)),
                        ("strand", ("strand",))):
        for c in cands:
            if c in lower:
                ren[lower[c]] = want
                break
    df = df.rename(columns=ren)
    for c in ("chrom", "start", "end"):
        if c not in df.columns:
            raise ValueError(f"interval table needs a {c!r} column")
    if "strand" not in df.columns:
        df["strand"] = "+"
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["chrom"] = df["chrom"].astype(str)
    cols = ["chrom", "start", "end"] + (["strand"] if need_strand or "strand" in df else [])
    return df[cols].reset_index(drop=True)
