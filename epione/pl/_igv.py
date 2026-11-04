"""Interactive genome browser (igv.js) for Epione objects.

``epi.pl.igv_browser`` is the interactive counterpart of
:func:`epione.pl.plot_browser_track` (ArchR ``plotBrowserTrack``) and
:func:`epione.pl.plot_peak2gene`. It takes the same inputs -- an AnnData with
cell groups, fragment files or BigWigs, peaks, co-accessibility / peak-to-gene
links, gene models -- plus motif sites, footprints and Hi-C contacts, and
returns an :class:`IGVBrowser` that renders with `igv.js
<https://github.com/igvteam/igv.js>`_ (MIT) in Jupyter, in the Sphinx docs and
as a standalone HTML file.

All track data for the requested window (plus ``flank`` bp on each side) is
embedded in the page as gzipped base64 ``data:`` URIs, so the output never
refers to local files and works offline except for igv.js itself (loaded from
a pinned CDN URL) and the reference genome named by ``genome``.
"""
from __future__ import annotations

import base64
import copy
import gzip
import html
import json
import os
import re
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._browser_track import (
    _overlaps,
    _parse_region,
    _standardise_intervals,
    archr_palette_discrete,
    browser_track_coverage,
    gene_region,
    loops_from_coaccessibility,
)

IGV_JS_VERSION = "3.8.10"
IGV_JS_URL = f"https://cdn.jsdelivr.net/npm/igv@{IGV_JS_VERSION}/dist/igv.min.js"
# Subresource integrity of the pinned igv.min.js (jsdelivr's sha256).
IGV_JS_SRI = "sha256-h7mRripyCXgXLiad1RJ8SV5W31rzEUTf8ZyDEk4vV0Q="

# ArchR loop colours (``_browser_track._LOOP_COLORS``), used for arc colour.
_LOOP_COLORS = ["#E6E7E8", "#3A97FF", "#8816A7", "#000000"]
_GENE_COLOR = "#1C3D7A"
_PEAK_COLOR = "#D51F26"

Intervals = Union[pd.DataFrame, str, os.PathLike]


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def data_uri(text: str) -> str:
    """Gzip + base64 ``data:`` URI that igv.js decodes in the browser.

    ``mtime=0`` keeps the bytes (and hence notebooks / HTML files)
    reproducible.
    """
    raw = gzip.compress(text.encode("utf-8"), mtime=0)
    return "data:application/gzip;base64," + base64.b64encode(raw).decode("ascii")


def decode_data_uri(uri: str) -> str:
    """Inverse of :func:`data_uri` (used by tests and for inspection)."""
    head, payload = uri.split(",", 1)
    raw = base64.b64decode(payload)
    if "gzip" in head:
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def _is_remote(path) -> bool:
    return isinstance(path, str) and re.match(r"^(https?|s3|gs|ftp)://", path) is not None


def _check_url(url: str) -> str:
    """Remote URLs and relative paths pass; absolute local paths are refused
    so they never get baked into a shared page."""
    if url.startswith("data:") or _is_remote(url):
        return url
    if os.path.isabs(url) or url.startswith("~") or url.startswith("file:"):
        raise ValueError(
            f"refusing to embed the absolute local path {url!r} in the browser page; "
            "pass a remote URL, a path relative to where the HTML will be served, "
            "or let Epione embed the data (the default for local files)"
        )
    return url


def _hex_to_rgb(c: str) -> str:
    import matplotlib.colors as mcolors
    r, g, b = (int(round(255 * v)) for v in mcolors.to_rgb(c))
    return f"{r},{g},{b}"


def _value_colors(values: np.ndarray, vmin: float, vmax: float) -> List[str]:
    """ArchR loop palette mapped onto ``values`` (``r,g,b`` strings)."""
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("archr_loop", _LOOP_COLORS)
    span = (vmax - vmin) if vmax > vmin else 1.0
    out = []
    for v in np.clip((np.asarray(values, float) - vmin) / span, 0.0, 1.0):
        r, g, b, _ = cmap(float(v))
        out.append(f"{int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))}")
    return out


def _fmt(v: float) -> str:
    return f"{float(v):.6g}"


# ---------------------------------------------------------------------------
# Interval readers
# ---------------------------------------------------------------------------

def _read_intervals(src: Intervals, chrom: str, start: int, end: int) -> pd.DataFrame:
    """BED path / DataFrame -> 0-based ``chrom/start/end[/name/score/strand]``
    restricted to the window. DataFrames follow Epione's 1-based ``start``
    convention only when they carry ``Start``/``start`` from pyranges/BED
    readers; we treat every input as BED-like (0-based start)."""
    if isinstance(src, pd.DataFrame):
        df = src
    else:
        path = os.fspath(src)
        comp = "gzip" if path.endswith(".gz") else None
        df = pd.read_csv(path, sep="\t", header=None, comment="#", compression=comp)
        df = df.rename(columns={0: "chrom", 1: "start", 2: "end", 3: "name",
                                4: "score", 5: "strand"})
        df = df[[c for c in ["chrom", "start", "end", "name", "score", "strand"] if c in df]]
    out = _standardise_intervals(df)
    for src_cols, dst in [(["name", "motif", "TF", "tf", "id"], "name"),
                          (["score", "tc", "footprint_score", "value"], "score")]:
        if dst not in out:
            for c in src_cols:
                if c in df.columns:
                    out[dst] = df[c].to_numpy()
                    break
    if "symbol" in out and "name" not in out:
        out["name"] = out["symbol"]
    return _overlaps(out, chrom, start, end).reset_index(drop=True)


def _bed_text(df: pd.DataFrame) -> str:
    lines = []
    has_name = "name" in df
    has_score = "score" in df
    has_strand = "strand" in df
    for row in df.itertuples(index=False):
        d = row._asdict()
        cols = [str(d["chrom"]), str(int(d["start"])), str(int(d["end"]))]
        if has_name or has_score or has_strand:
            cols.append(str(d["name"]) if has_name else ".")
            if has_score or has_strand:
                sc = d.get("score") if has_score else None
                cols.append(_fmt(sc) if sc is not None and pd.notna(sc) else "0")
                if has_strand:
                    st = str(d["strand"])
                    cols.append(st if st in ("+", "-") else ".")
        lines.append("\t".join(cols))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Track-spec builders (pure Python; each returns igv.js track dicts)
# ---------------------------------------------------------------------------

def coverage_tracks(
    coverage: Mapping[str, pd.DataFrame],
    *,
    palette: Optional[Mapping[str, str]] = None,
    height: int = 40,
    ylim: Optional[Tuple[float, float]] = None,
    autoscale_group: str = "coverage",
) -> List[Dict[str, Any]]:
    """One igv.js ``wig`` track per group from ``{group: bedGraph frame}``.

    Each frame holds 0-based ``chrom/start/end/value``. All tracks share one
    y-scale (igv ``autoscaleGroup``), as ArchR's ``plotBrowserTrack`` does,
    unless ``ylim`` fixes it.
    """
    if palette is None:
        palette = archr_palette_discrete(list(coverage))
    tracks = []
    for g, df in coverage.items():
        text = "".join(f"{c}\t{int(s)}\t{int(e)}\t{_fmt(v)}\n"
                       for c, s, e, v in df[["chrom", "start", "end", "value"]].itertuples(index=False)
                       if v != 0)
        t = {"name": str(g), "type": "wig", "format": "bedgraph", "url": data_uri(text),
             "color": palette.get(str(g), palette.get(g, "#3D3D3D")), "height": int(height)}
        if ylim is not None:
            t.update(min=float(ylim[0]), max=float(ylim[1]), autoscale=False)
        else:
            t.update(autoscale=True, autoscaleGroup=autoscale_group)
        tracks.append(t)
    return tracks


def tiles_to_bedgraph(cov: pd.DataFrame, chrom: str, tile_size: int) -> Dict[str, pd.DataFrame]:
    """:func:`browser_track_coverage` long table -> ``{group: bedGraph}``.

    ArchR tiles are 1-based (``x`` is the first base of the tile), so the
    0-based bedGraph interval is ``[x - 1, x - 1 + tile_size)``.
    """
    cov = cov[np.isfinite(cov["x"].to_numpy(float))]
    groups = cov.attrs.get("groups") or list(pd.unique(cov["group"]))
    out = {}
    for g in groups:
        sub = cov[cov["group"] == g].drop_duplicates("x").sort_values("x")
        s = sub["x"].to_numpy(np.int64) - 1
        out[str(g)] = pd.DataFrame({"chrom": chrom, "start": np.maximum(s, 0),
                                    "end": s + int(tile_size), "value": sub["y"].to_numpy(float)})
    return out


def _bigwig_bedgraph(path: str, chrom: str, start: int, end: int, bin_size: int) -> pd.DataFrame:
    import pyBigWig
    with pyBigWig.open(os.fspath(path)) as bw:
        size = bw.chroms().get(chrom)
        if size is None:
            return pd.DataFrame(columns=["chrom", "start", "end", "value"])
        lo, hi = max(0, start), min(int(size), end)
        n = max(1, (hi - lo) // bin_size)
        vals = np.asarray(bw.stats(chrom, lo, hi, nBins=n, type="mean"), dtype=object)
    vals = np.array([0.0 if v is None else float(v) for v in vals])
    edges = np.linspace(lo, hi, n + 1).astype(np.int64)
    return pd.DataFrame({"chrom": chrom, "start": edges[:-1], "end": edges[1:], "value": vals})


def _peak_matrix_bedgraph(adata, group_by, chrom, start, end, bin_size) -> Dict[str, pd.DataFrame]:
    from ._peak2gene import _coverage_from_peaks
    var = _var_intervals(adata)
    n_bins = max(1, (end - start) // bin_size)
    cov, _ = _coverage_from_peaks(adata, group_by, chrom, start, end, var, n_bins=n_bins)
    if cov is None:
        return {}
    edges = np.linspace(start, end, n_bins + 1).astype(np.int64)
    return {str(g): pd.DataFrame({"chrom": chrom, "start": edges[:-1], "end": edges[1:],
                                  "value": np.asarray(v, float)}) for g, v in cov.items()}


def _var_intervals(adata) -> pd.DataFrame:
    var = adata.var
    if {"chrom", "start", "end"}.issubset(var.columns):
        return var[["chrom", "start", "end"]].reset_index(drop=True)
    pk = pd.Series(adata.var_names.astype(str)).str.extract(r"^([^:]+)[:_-](\d+)[-_](\d+)$")
    if pk.isna().any().any():
        raise ValueError("adata.var needs chrom/start/end columns or var_names like 'chr1:100-200'")
    return pd.DataFrame({"chrom": pk[0], "start": pk[1].astype(np.int64), "end": pk[2].astype(np.int64)})


def annotation_track(name: str, df: pd.DataFrame, *, color: Optional[str] = None,
                     height: Optional[int] = None, display_mode: str = "COLLAPSED",
                     **extra) -> Dict[str, Any]:
    """igv.js ``annotation`` track from a 0-based interval frame."""
    t = {"name": name, "type": "annotation", "format": "bed", "url": data_uri(_bed_text(df)),
         "displayMode": display_mode}
    if color is not None:
        t["color"] = color
    if height is not None:
        t["height"] = int(height)
    t.update(extra)
    return t


def gene_bed12(gene_annotation: pd.DataFrame, exon_annotation: Optional[pd.DataFrame],
               chrom: str, start: int, end: int) -> str:
    """BED12 gene models for the window (one row per gene, exons as blocks).

    Gene/exon frames use Epione's 1-based closed coordinates (``gene_region``
    / ``plot_browser_track`` convention), converted here to 0-based BED.
    """
    g = _standardise_intervals(gene_annotation)
    if "strand" not in g:
        g["strand"] = "."
    g = _overlaps(g, chrom, start, end)
    ex = None
    if exon_annotation is not None:
        ex = _standardise_intervals(exon_annotation)
        ex = _overlaps(ex, chrom, start, end)
    lines = []
    for row in g.sort_values(["start", "end"]).itertuples(index=False):
        gs, ge = int(row.start) - 1, int(row.end)
        name = str(getattr(row, "symbol", "."))
        strand = str(row.strand) if str(row.strand) in ("+", "-") else "."
        blocks = []
        if ex is not None and "symbol" in ex:
            e = ex[(ex["symbol"].astype(str) == name) & (ex["start"] >= int(row.start))
                   & (ex["end"] <= int(row.end))]
            iv = sorted((int(a) - 1, int(b)) for a, b in zip(e["start"], e["end"]))
            for a, b in iv:  # merge overlapping exons of all transcripts
                if blocks and a <= blocks[-1][1]:
                    blocks[-1][1] = max(blocks[-1][1], b)
                else:
                    blocks.append([a, b])
        if not blocks:
            blocks = [[gs, ge]]
        sizes = ",".join(str(b - a) for a, b in blocks)
        starts = ",".join(str(a - gs) for a, _ in blocks)
        lines.append("\t".join([chrom, str(gs), str(ge), name, "0", strand, str(gs), str(ge),
                                "0", str(len(blocks)), sizes, starts]))
    return "\n".join(lines) + "\n"


def _loops_frame(loops, min_correlation: float, resolution: Optional[int]) -> pd.DataFrame:
    """Any supported loop table -> BEDPE frame
    ``chrom1 start1 end1 chrom2 start2 end2 name score``.

    Accepts ``plot_browser_track``-style loops (``chrom/start/end/value``,
    positions are anchor centres), raw :func:`epione.tl.coaccessibility`
    output (``peak1_center/peak2_center/correlation``), and BEDPE-like
    frames (``chrom1/start1/end1/chrom2/start2/end2`` [``score``/``value``]),
    e.g. Epione's Hi-C loop callers.
    """
    df = loops
    if {"peak1_center", "peak2_center"}.issubset(df.columns):
        df = loops_from_coaccessibility(df, min_correlation=min_correlation, resolution=resolution)
    if {"chrom1", "start1", "end1", "chrom2", "start2", "end2"}.issubset(df.columns):
        score = None
        for c in ("score", "value", "correlation", "obs", "count", "balanced"):
            if c in df.columns:
                score = df[c].to_numpy(float)
                break
        out = df[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()
        out["name"] = df["name"].astype(str).to_numpy() if "name" in df else "."
        out["score"] = score if score is not None else 1.0
        return out.reset_index(drop=True)
    s = _standardise_intervals(df)
    val = s["value"].to_numpy(float) if "value" in s else np.ones(len(s))
    a, b = s["start"].to_numpy(np.int64), s["end"].to_numpy(np.int64)
    return pd.DataFrame({"chrom1": s["chrom"], "start1": a - 1, "end1": a,
                         "chrom2": s["chrom"], "start2": b - 1, "end2": b,
                         "name": s["symbol"].astype(str) if "symbol" in s else ".",
                         "score": val})


def links_frame(links: pd.DataFrame, *, fdr_thresh: float = 0.05,
                min_abs_r: float = 0.0, genes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """:func:`epione.tl.peak_to_gene` table -> BEDPE (peak centre <-> TSS),
    filtered exactly as :func:`epione.pl.plot_peak2gene` does."""
    sub = links[(links["fdr"] <= fdr_thresh) & (links["correlation"].abs() >= min_abs_r)]
    if genes is not None:
        sub = sub[sub["gene"].isin(list(genes))]
    pc = ((sub["peak_start"] + sub["peak_end"]) // 2).to_numpy(np.int64)
    tss = sub["tss"].to_numpy(np.int64)
    return pd.DataFrame({"chrom1": sub["chrom"].astype(str).to_numpy(), "start1": pc - 1, "end1": pc,
                         "chrom2": sub["chrom"].astype(str).to_numpy(), "start2": tss - 1, "end2": tss,
                         "name": sub["gene"].astype(str).to_numpy(),
                         "score": sub["correlation"].to_numpy(float)})


def interaction_track(name: str, bedpe: pd.DataFrame, chrom: str, start: int, end: int, *,
                      color_by_value: bool = True, limits: Optional[Tuple[float, float]] = None,
                      color: Optional[str] = None, height: int = 80,
                      arc_type: str = "proportional", **extra) -> Dict[str, Any]:
    """igv.js ``interact`` (arc) track from a BEDPE frame, restricted to arcs
    with both anchors in the window. Arc colour follows ArchR's loop palette
    over ``limits`` (default: the score range); arc height is the score."""
    keep = ((bedpe["chrom1"].astype(str) == chrom) & (bedpe["chrom2"].astype(str) == chrom)
            & (np.minimum(bedpe["start1"], bedpe["start2"]) >= start)
            & (np.maximum(bedpe["end1"], bedpe["end2"]) <= end))
    sub = bedpe[keep]
    score = sub["score"].to_numpy(float)
    header = "#chr1\tstart1\tend1\tchr2\tstart2\tend2\tname\tscore\tstrand1\tstrand2"
    cols = None
    if color_by_value and color is None and len(sub):
        lo, hi = limits if limits is not None else (float(np.nanmin(score)), float(np.nanmax(score)))
        cols = _value_colors(score, lo, hi)
        header += "\tcolor"
    lines = [header]
    for i, r in enumerate(sub.itertuples(index=False)):
        row = [str(r.chrom1), str(int(r.start1)), str(int(r.end1)), str(r.chrom2),
               str(int(r.start2)), str(int(r.end2)), str(r.name), _fmt(r.score), ".", "."]
        if cols is not None:
            row.append(cols[i])
        lines.append("\t".join(row))
    t = {"name": name, "type": "interact", "format": "bedpe",
         "url": data_uri("\n".join(lines) + "\n"), "arcType": arc_type,
         "valueColumn": "score", "height": int(height), "showBlocks": True}
    if color is not None:
        t["color"] = color
    t.update(extra)
    return t


def cooler_bedpe(cool, chrom: str, start: int, end: int, *, resolution: Optional[int] = None,
                 balance: bool = True, min_distance: Optional[int] = None,
                 top_n: int = 2000) -> pd.DataFrame:
    """Contacts of a ``.cool`` / ``.mcool`` in the window as BEDPE.

    igv.js has no ``.cool`` reader, so the strongest ``top_n`` off-diagonal
    pixels (``>= min_distance``, default two bins) are drawn as arcs.
    """
    import cooler
    if isinstance(cool, (str, os.PathLike)):
        uri = os.fspath(cool)
        if uri.endswith(".mcool") and "::" not in uri:
            if resolution is None:
                raise ValueError("pass hic_resolution for a .mcool file")
            uri = f"{uri}::resolutions/{int(resolution)}"
        cool = cooler.Cooler(uri)
    binsize = int(cool.binsize)
    region = (chrom, max(0, int(start)), min(int(cool.chromsizes[chrom]), int(end)))
    px = cool.matrix(balance=balance, as_pixels=True, join=True).fetch(region)
    col = "balanced" if balance and "balanced" in px else "count"
    px = px[np.isfinite(px[col].to_numpy(float))]
    md = 2 * binsize if min_distance is None else int(min_distance)
    px = px[(px["start2"] - px["start1"]) >= md]
    px = px.nlargest(int(top_n), col)
    out = px[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()
    out["name"] = "."
    out["score"] = px[col].to_numpy(float)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Browser object
# ---------------------------------------------------------------------------

class IGVBrowser:
    """An igv.js browser configuration with Jupyter / HTML rendering.

    Attributes:
        options: the ``igv.createBrowser`` options (``genome`` or
            ``reference``, ``locus``, ``tracks``, ``roi`` ...). Edit freely
            before rendering.
        height: iframe height in pixels for notebook display; ``None``
            sizes it to fit all tracks.
    """

    def __init__(self, options: Dict[str, Any], *, height: Optional[int] = None,
                 title: str = "Epione IGV", igv_js_url: str = IGV_JS_URL):
        self.options = options
        self._height = None if height is None else int(height)
        self.title = title
        self.igv_js_url = igv_js_url

    @property
    def height(self) -> int:
        """iframe height: explicit, or toolbar + ruler + every track."""
        if self._height is not None:
            return self._height
        h = 150 + sum(int(t.get("height", 50)) + 6 for t in self.tracks)
        if "genome" in self.options:
            h += 70  # the genome's own gene track
        return h

    @height.setter
    def height(self, value: Optional[int]):
        self._height = None if value is None else int(value)

    @property
    def tracks(self) -> List[Dict[str, Any]]:
        return self.options.get("tracks", [])

    @property
    def locus(self):
        return self.options.get("locus")

    def track_names(self) -> List[str]:
        return [t.get("name", "") for t in self.tracks]

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.options)

    def to_json(self) -> str:
        return json.dumps(self.options, separators=(",", ":"))

    def to_html(self) -> str:
        """Standalone HTML page (igv.js from the pinned CDN, data inline).

        igv.js' own toolbar has a camera button that saves the current view
        as SVG or PNG.
        """
        sri = (f' integrity="{IGV_JS_SRI}" crossorigin="anonymous"'
               if self.igv_js_url == IGV_JS_URL else "")
        div = "igv-" + uuid.uuid5(uuid.NAMESPACE_URL, self.to_json()).hex[:12]
        cfg = self.to_json().replace("</", "<\\/")
        return (
            "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(self.title)}</title>"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<script src=\"{self.igv_js_url}\"{sri}></script>"
            "<style>body{margin:0;padding:4px;font-family:sans-serif;background:#fff}"
            ".epi-igv-err{color:#b00;font-size:13px}</style></head><body>"
            f"<div id=\"{div}\"></div>"
            "<script>(function(){"
            f"var el=document.getElementById('{div}');"
            "if(typeof igv==='undefined'){el.innerHTML='<p class=\"epi-igv-err\">igv.js could not be "
            "loaded (offline?).</p>';return;}"
            f"var options={cfg};"
            "igv.createBrowser(el,options).then(function(b){window.igvBrowser=b;"
            "document.body.setAttribute('data-igv-ready','1');})"
            ".catch(function(e){el.innerHTML='<p class=\"epi-igv-err\">'+e+'</p>';});"
            "})();</script></body></html>\n"
        )

    def save(self, path: Union[str, os.PathLike]) -> str:
        """Write :meth:`to_html` to ``path`` and return the path."""
        path = os.fspath(path)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_html())
        return path

    def _repr_html_(self) -> str:
        # An iframe isolates igv.js (and its CSS) from the notebook / Sphinx
        # page, and lets several browsers coexist on one page.
        doc = html.escape(self.to_html(), quote=True)
        return (f'<iframe srcdoc="{doc}" style="width:100%;height:{self.height}px;border:0" '
                f'title="{html.escape(self.title)}" loading="lazy"></iframe>')

    def show(self):
        """Display in Jupyter (same as leaving the object last in a cell)."""
        from IPython.display import HTML, display
        display(HTML(self._repr_html_()))

    def __repr__(self) -> str:
        return (f"IGVBrowser(locus={self.locus!r}, tracks={self.track_names()!r})")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def igv_browser(
    adata=None,
    region=None,
    *,
    gene: Optional[str] = None,
    fragments=None,
    bigwig_files: Optional[Mapping[str, str]] = None,
    coverage: Optional[Union[pd.DataFrame, Mapping[str, pd.DataFrame]]] = None,
    group_by: Optional[str] = "Clusters",
    sample_key: Optional[str] = "Sample",
    barcode_sep: Optional[str] = "#",
    use_groups: Optional[Sequence[str]] = None,
    features=None,
    loops=None,
    links=None,
    links_key: str = "peak_to_gene",
    fdr_thresh: float = 0.05,
    min_abs_r: float = 0.0,
    min_correlation: float = 0.5,
    resolution: Optional[int] = 1000,
    motifs: Optional[Union[Intervals, Mapping[str, Intervals]]] = None,
    footprints: Optional[Union[Intervals, Mapping[str, Intervals]]] = None,
    hic=None,
    hic_resolution: Optional[int] = None,
    hic_balance: bool = True,
    hic_top_n: int = 2000,
    gene_annotation: Optional[pd.DataFrame] = None,
    exon_annotation: Optional[pd.DataFrame] = None,
    genome: Union[str, Mapping[str, Any], None] = "hg38",
    upstream: int = 50_000,
    downstream: int = 50_000,
    flank: Optional[int] = None,
    tile_size: int = 250,
    max_cells: int = 500,
    min_cells: int = 25,
    norm_method: str = "ReadsInTSS",
    highlight: Optional[pd.DataFrame] = None,
    ylim: Optional[Tuple[float, float]] = None,
    palette: Optional[Mapping[str, str]] = None,
    cells: Optional[Sequence[str]] = None,
    seed: Optional[int] = 1,
    rng: str = "archr",
    count_ends: bool = False,
    track_height: int = 40,
    tracks: Optional[Sequence[Mapping[str, Any]]] = None,
    title: str = "",
    height: Optional[int] = None,
    igv_js_url: str = IGV_JS_URL,
) -> IGVBrowser:
    """Interactive igv.js genome browser built from Epione objects.

    The interactive counterpart of :func:`plot_browser_track` and
    :func:`plot_peak2gene`: the shared arguments mean the same thing, so a
    static call can be switched to an interactive one by changing the
    function name. Tracks are drawn top to bottom in this order: per-group
    coverage, peaks, motif sites, footprints, co-accessibility / loops,
    peak-to-gene links, Hi-C, genes, then any raw ``tracks``.

    Data for ``region`` plus ``flank`` bp on each side is embedded in the
    page; panning further shows empty tracks (re-run with a new region).

    Arguments:
        adata: AnnData with cell groups in ``obs`` (for coverage), peaks in
            ``var`` and, optionally, peak-to-gene links in ``uns``.
        region: ``"chr:start-end"`` or ``(chrom, start, end)``, 1-based closed.
        gene: alternatively a gene symbol. With ``gene_annotation`` the
            window is the TSS ``- upstream`` / ``+ downstream`` (as in
            ``plot_browser_track``); without it, igv.js searches the name in
            the ``genome`` annotation and only remote / annotation tracks are
            shown.
        fragments: tabix-indexed fragment file or ``{sample: path}``; per-group
            coverage is computed exactly as in ``plot_browser_track``
            (ArchR ``.groupRegionSumArrows``).
        bigwig_files: ``{label: path_or_url}``, e.g. from
            ``epi.single.pseudobulk_with_fragments``. Local files are read
            over the window in ``tile_size`` bins and embedded; remote URLs
            (and relative paths) are streamed by igv.js.
        coverage: precomputed :func:`browser_track_coverage` output or
            ``{group: DataFrame(chrom, start, end, value)}`` (0-based).
            Without ``fragments``/``bigwig_files``/``coverage`` but with
            ``group_by`` and a peak matrix, coverage comes from the peak
            matrix (``plot_peak2gene``'s default).
        group_by, sample_key, barcode_sep, use_groups: as in
            ``plot_browser_track``. Pass ``group_by=None`` for no coverage.
        features: peaks as DataFrame or ``{name: DataFrame}``; ``"var"`` or
            ``True`` uses ``adata.var``.
        loops: co-accessibility (``epi.tl.coaccessibility`` output or
            ``loops_from_coaccessibility`` output), Hi-C loops (BEDPE frame)
            or ``{name: frame}``; drawn as arcs coloured by value.
        links: ``epi.tl.peak_to_gene`` table, or ``True`` for
            ``adata.uns[links_key]``. Filtered with ``fdr_thresh`` and
            ``min_abs_r`` as in ``plot_peak2gene``.
        min_correlation, resolution: applied to raw co-accessibility tables.
        motifs, footprints: BED paths or DataFrames (or ``{name: ...}``) of
            motif sites, HINT footprints (``epi.tl.hint_footprints``) or
            TOBIAS/BINDetect bound sites (``*_bound.bed``). 0-based BED.
        hic: ``.hic`` URL (rendered natively by igv.js), a local ``.cool`` /
            ``.mcool`` path or ``cooler.Cooler`` (top ``hic_top_n`` contacts in
            the window as arcs; needs ``cooler``), or a BEDPE frame.
        gene_annotation, exon_annotation: gene / exon tables
            (``chrom/start/end/strand/symbol``, 1-based); embedded as a BED12
            gene track. igv.js also shows the ``genome``'s own annotation.
        genome: igv.js genome id (``"hg38"``, ``"hg19"``, ``"mm10"``,
            ``"mm39"`` ...), an igv ``reference`` dict, or ``None``.
        flank: bp embedded beyond the region on each side (default: half the
            region width).
        tile_size, max_cells, min_cells, norm_method, cells, seed, rng,
            count_ends: coverage parameters of ``plot_browser_track``.
        highlight: intervals (1-based) shown as igv regions of interest.
        ylim: fixed ``(min, max)`` for coverage tracks (absolute values; the
            static plot's quantile semantics do not apply). Default: shared
            autoscale across groups.
        palette: ``{group: colour}``; default ArchR ``paletteDiscrete``.
        track_height: height of each coverage track in px.
        tracks: extra raw igv.js track configs appended at the end. ``url``
            values must be remote or relative.
        title: page title.
        height: iframe height in notebooks / docs (default: fit all tracks).
        igv_js_url: igv.js script URL (default pinned jsdelivr build); point
            it at a local copy for offline use.

    Returns:
        :class:`IGVBrowser`. Display it in Jupyter, ``.save("x.html")`` for a
        standalone page, inspect ``.options`` / ``.tracks``.
    """
    # ---- window ------------------------------------------------------------
    locus = None
    if region is None:
        if gene is None:
            raise ValueError("pass `region` or `gene`")
        if gene_annotation is not None:
            region = gene_region(gene_annotation, gene, upstream, downstream)
        else:
            locus = str(gene)
    win = None
    if region is not None:
        chrom, start, end = _parse_region(region)
        fl = (end - start) // 2 if flank is None else int(flank)
        win = (chrom, max(0, start - 1 - fl), end + fl)  # 0-based half-open data window
        locus = f"{chrom}:{start}-{end}"

    out_tracks: List[Dict[str, Any]] = []

    def need_window(what):
        if win is None:
            raise ValueError(f"{what} needs an explicit window: pass `region`, or `gene` "
                             "together with `gene_annotation`")
        return win

    # ---- coverage ----------------------------------------------------------
    bedgraphs: Dict[str, pd.DataFrame] = {}
    remote_bw: Dict[str, str] = {}
    if coverage is not None:
        c, s, e = need_window("coverage")
        if isinstance(coverage, pd.DataFrame) and {"x", "y", "group"}.issubset(coverage.columns):
            bedgraphs = tiles_to_bedgraph(coverage, c, tile_size)
        else:
            bedgraphs = {str(k): v for k, v in dict(coverage).items()}
    elif bigwig_files is not None:
        c, s, e = need_window("bigwig_files")
        for label, path in bigwig_files.items():
            p = os.fspath(path)
            if _is_remote(p) or not os.path.exists(p):
                remote_bw[str(label)] = _check_url(p)
            else:
                bedgraphs[str(label)] = _bigwig_bedgraph(p, c, s, e, tile_size)
    elif fragments is not None and adata is not None and group_by is not None:
        c, s, e = need_window("fragments")
        cov = browser_track_coverage(
            adata, (c, s + 1, e), fragments=fragments, group_by=group_by, sample_key=sample_key,
            barcode_sep=barcode_sep, use_groups=use_groups, tile_size=tile_size,
            norm_method=norm_method, min_cells=min_cells, max_cells=max_cells, cells=cells,
            seed=seed, rng=rng, count_ends=count_ends)
        bedgraphs = tiles_to_bedgraph(cov, c, tile_size)
    elif adata is not None and group_by is not None and group_by in adata.obs and win is not None:
        c, s, e = win
        bedgraphs = _peak_matrix_bedgraph(adata, group_by, c, s, e, tile_size)
    if use_groups is not None and bedgraphs:
        bedgraphs = {str(g): bedgraphs[str(g)] for g in use_groups if str(g) in bedgraphs}
    labels = list(bedgraphs) + list(remote_bw)
    if labels:
        if palette is None:
            src = (adata.obs[group_by].dropna().astype(str).unique()
                   if adata is not None and group_by in getattr(adata, "obs", {}) else labels)
            palette = archr_palette_discrete(list(src) + [l for l in labels if l not in set(src)])
        palette = {str(k): v for k, v in palette.items()}
        out_tracks += coverage_tracks(bedgraphs, palette=palette, height=track_height, ylim=ylim)
        for label, url in remote_bw.items():
            t = {"name": label, "type": "wig", "format": "bigwig", "url": url,
                 "color": palette.get(label, "#3D3D3D"), "height": int(track_height)}
            t.update(dict(min=float(ylim[0]), max=float(ylim[1]), autoscale=False) if ylim is not None
                     else dict(autoscale=True, autoscaleGroup="coverage"))
            out_tracks.append(t)

    # ---- peaks -------------------------------------------------------------
    if features is True or (isinstance(features, str) and features == "var"):
        features = {"Peaks": _var_intervals(adata)}
    if features is not None:
        c, s, e = need_window("features")
        fdict = {"Peaks": features} if isinstance(features, pd.DataFrame) else dict(features)
        for name, df in fdict.items():
            # Epione peak tables (adata.var) are 0-based BED starts already.
            sub = _overlaps(_standardise_intervals(df), c, s, e)[["chrom", "start", "end"]]
            out_tracks.append(annotation_track(str(name), sub, color=_PEAK_COLOR, height=25))

    # ---- motif sites / footprints -----------------------------------------
    for kind, obj, col in (("Motifs", motifs, "#208A42"), ("Footprints", footprints, "#89288F")):
        if obj is None:
            continue
        c, s, e = need_window(kind.lower())
        d = {kind: obj} if not isinstance(obj, Mapping) else dict(obj)
        for name, src in d.items():
            out_tracks.append(annotation_track(str(name), _read_intervals(src, c, s, e),
                                               color=col, display_mode="EXPANDED"))

    # ---- co-accessibility / loops -----------------------------------------
    if loops is not None:
        c, s, e = need_window("loops")
        ldict = {"Loops": loops} if isinstance(loops, pd.DataFrame) else dict(loops)
        frames = {k: _loops_frame(v, min_correlation, resolution) for k, v in ldict.items()}
        allv = np.concatenate([f["score"].to_numpy(float) for f in frames.values()] or [np.ones(1)])
        lim = (float(np.nanmin(allv)), float(np.nanmax(allv))) if len(allv) else None
        for name, f in frames.items():
            out_tracks.append(interaction_track(str(name), f, c, s, e, limits=lim))

    # ---- peak-to-gene links -----------------------------------------------
    if links is True:
        links = adata.uns[links_key]
    if links is not None:
        c, s, e = need_window("links")
        if not isinstance(links, pd.DataFrame):
            links = pd.DataFrame(links)
        f = links_frame(links, fdr_thresh=fdr_thresh, min_abs_r=min_abs_r)
        out_tracks.append(interaction_track("Peak2GeneLinks", f, c, s, e))

    # ---- Hi-C ----------------------------------------------------------------
    if hic is not None:
        if isinstance(hic, str) and hic.split("?")[0].endswith(".hic"):
            t = {"name": "Hi-C", "type": "interact", "format": "hic", "url": _check_url(hic),
                 "arcType": "proportional", "height": 100}
            if hic_resolution is not None:
                t["resolution"] = int(hic_resolution)
            out_tracks.append(t)
        else:
            c, s, e = need_window("hic")
            if isinstance(hic, pd.DataFrame):
                f = _loops_frame(hic, min_correlation, resolution)
            else:
                f = cooler_bedpe(hic, c, s, e, resolution=hic_resolution, balance=hic_balance,
                                 top_n=hic_top_n)
            out_tracks.append(interaction_track("Hi-C", f, c, s, e, height=100))

    # ---- genes ---------------------------------------------------------------
    if gene_annotation is not None and win is not None:
        c, s, e = win
        out_tracks.append({"name": "Genes", "type": "annotation", "format": "bed",
                           "url": data_uri(gene_bed12(gene_annotation, exon_annotation, c, s + 1, e)),
                           "displayMode": "EXPANDED", "color": _GENE_COLOR, "height": 70,
                           "searchable": True})

    # ---- raw tracks ------------------------------------------------------------
    for t in tracks or ():
        t = dict(t)
        for k in ("url", "indexURL"):
            if k in t:
                t[k] = _check_url(str(t[k]))
        out_tracks.append(t)

    options: Dict[str, Any] = {"locus": locus, "tracks": out_tracks, "showSVGButton": True}
    if isinstance(genome, str):
        options["genome"] = genome
    elif isinstance(genome, Mapping):
        options["reference"] = dict(genome)
    if highlight is not None:
        h = _standardise_intervals(highlight)
        options["roi"] = [{"name": "highlight", "color": "rgba(255,205,0,0.25)",
                           "features": [{"chr": str(r.chrom), "start": int(r.start) - 1,
                                         "end": int(r.end)} for r in h.itertuples(index=False)]}]
    return IGVBrowser(options, height=height, title=title or f"Epione IGV {locus}",
                      igv_js_url=igv_js_url)
