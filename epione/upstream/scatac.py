"""Single-cell ATAC FASTQ -> fragments (``epi.upstream.fastq_to_fragments``).

Three aligners share one interface:

``"chromap"`` (default)
    chromap aligns, corrects barcodes against the whitelist, trims adapters,
    removes duplicates per cell and applies the Tn5 shift in one pass
    (``chromap --preset atac``). Epione builds the index if needed, works
    out the barcode-read orientation and offset, and turns the output into a
    sorted, tabix-indexed ``fragments.tsv.gz`` (optionally translating
    Multiome ATAC barcodes to GEX barcodes).

``"bwa-mem2"`` / ``"bowtie2"``
    Epione reads the barcode read, corrects it against the whitelist itself
    (:class:`~epione.upstream.BarcodeCorrector`: exact match, else Hamming-1
    candidates weighted by whitelist abundance and base quality), writes
    ``CR``/``CY``/``CB`` tags into the read comment so the aligner copies
    them into the BAM, streams the read pairs into the aligner (through
    ``fastp`` adapter trimming for bowtie2), and builds fragments from the
    BAM with :func:`~epione.upstream.bam_to_fragments` (proper pairs, MAPQ,
    Tn5 shift, per-cell duplicate removal).

Read layouts: the 10x scATAC FASTQs from ``cellranger-atac mkfastq`` /
``bcl2fastq`` are ``R1`` (genomic), ``R2`` (16-bp barcode, the i5 index
read), ``R3`` (genomic); Multiome ATAC has a 24-bp ``R2`` (8-bp spacer +
16-bp barcode). FASTQs written as ``R1``/``R2`` + ``I2`` (barcode in the
index file) are recognised too, and :class:`ReadLayout` describes anything
else.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ._barcodes import (
    BarcodeCorrector,
    barcode_translation_table,
    detect_barcode_orientation,
    load_barcode_translation,
    load_whitelist,
    orient_barcode,
    revcomp,
)
from ._common import CommandError, resolve_tool
from ._env import build_env
from ._fragments import (
    barcode_table,
    collect_fragments,
    dedup_fragments,
    write_fragments,
)
from .reference import ensure_aligner_index, ensure_fasta_index, ensure_fasta_unzipped

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# read layouts
# ---------------------------------------------------------------------------
@dataclass
class ReadLayout:
    """Which FASTQ holds what.

    ``read1`` / ``read2`` / ``barcode`` are FASTQ roles (the ``R1``, ``R2``,
    ``R3``, ``I1``, ``I2`` field of Illumina file names, or the keys of the
    ``fastqs`` dict). The barcode is ``[bc_start, bc_start + bc_length)`` of
    the barcode read *after* putting it in whitelist orientation;
    ``bc_orientation`` is ``"forward"``, ``"reverse"`` (reverse-complement)
    or ``"auto"`` (decided from the first reads).
    """

    read1: str = "R1"
    read2: str = "R3"
    barcode: str = "R2"
    bc_start: int = 0
    bc_length: int = 16
    bc_orientation: str = "auto"


#: Built-in layouts. ``"atac"``: 10x Single Cell ATAC (v1 / v1.1 / v2),
#: 16-bp barcode read. ``"multiome"``: 10x Multiome ATAC, 24-bp barcode read
#: with an 8-bp spacer. The ``-i2`` variants take the barcode from an ``I2``
#: file with genomic reads in ``R1`` / ``R2``.
CHEMISTRIES: Dict[str, ReadLayout] = {
    "atac": ReadLayout("R1", "R3", "R2", 0, 16),
    "multiome": ReadLayout("R1", "R3", "R2", 8, 16),
    "atac-i2": ReadLayout("R1", "R2", "I2", 0, 16),
    "multiome-i2": ReadLayout("R1", "R2", "I2", 8, 16),
}

_FASTQ_RE = re.compile(
    r"^(?P<sample>.+?)(?:_S\d+)?(?:_L(?P<lane>\d{3}))?_(?P<role>R1|R2|R3|I1|I2)(?:_\d{3})?\.f(?:ast)?q(?:\.gz)?$"
)


def find_fastqs(fastq_dir: PathLike, sample: Optional[str] = None) -> Dict[str, List[str]]:
    """Group Illumina-named FASTQs in ``fastq_dir`` by read role.

    Recognises ``<sample>_S1_L001_R1_001.fastq.gz`` and shorter variants
    (``_S1`` / ``_L001`` / ``_001`` optional). Returns ``{role: [files sorted
    by lane]}``. With several samples in the folder, ``sample`` is required.
    """
    fastq_dir = Path(fastq_dir)
    hits: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    for p in sorted(fastq_dir.iterdir()):
        m = _FASTQ_RE.match(p.name)
        if not m:
            continue
        hits.setdefault(m["sample"], {}).setdefault(m["role"], []).append((m["lane"] or "", str(p)))
    if not hits:
        raise FileNotFoundError(f"no Illumina-named FASTQs in {fastq_dir}")
    if sample is None:
        if len(hits) > 1:
            raise ValueError(f"several samples in {fastq_dir}: {sorted(hits)}; pass sample=")
        sample = next(iter(hits))
    if sample not in hits:
        raise FileNotFoundError(f"sample {sample!r} not in {fastq_dir} (found {sorted(hits)})")
    return {role: [f for _, f in sorted(v)] for role, v in hits[sample].items()}


def _fastqs_from_list(paths: Sequence[PathLike]) -> Dict[str, List[str]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for p in map(str, paths):
        m = _FASTQ_RE.match(Path(p).name)
        if not m:
            raise ValueError(f"cannot tell the read role of {p}; pass fastqs as a dict {{'R1': [...], ...}}")
        out.setdefault(m["role"], []).append((m["lane"] or "", p))
    return {k: [f for _, f in sorted(v)] for k, v in out.items()}


def _resolve_layout(chemistry: Union[str, ReadLayout, Mapping], roles: Sequence[str]) -> ReadLayout:
    if isinstance(chemistry, ReadLayout):
        return chemistry
    if isinstance(chemistry, Mapping):
        return ReadLayout(**dict(chemistry))
    if chemistry not in ("atac", "multiome") and chemistry not in CHEMISTRIES:
        raise ValueError(f"unknown chemistry {chemistry!r}; use one of {sorted(CHEMISTRIES)} or a ReadLayout")
    layout = CHEMISTRIES[chemistry]
    # bcl2fastq without the 10x base mask: barcode in I2, genomic in R1/R2.
    if chemistry in ("atac", "multiome") and "R3" not in roles and "I2" in roles:
        layout = CHEMISTRIES[chemistry + "-i2"]
    return layout


# ---------------------------------------------------------------------------
# FASTQ streaming
# ---------------------------------------------------------------------------
class _FastqStream:
    """Line-wise FASTQ reader; gzip decompression runs in a child process."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._proc = None
        if path.endswith(".gz"):
            gz = shutil.which("pigz") or shutil.which("gzip")
            if gz:
                self._proc = subprocess.Popen([gz, "-dc", path], stdout=subprocess.PIPE, bufsize=1 << 20)
                self._fh = self._proc.stdout
            else:
                self._fh = gzip.open(path, "rb")
        else:
            self._fh = open(path, "rb")

    def __iter__(self) -> Iterator[Tuple[bytes, bytes, bytes]]:
        rl = self._fh.readline
        while True:
            h = rl()
            if not h:
                return
            s = rl().rstrip(b"\r\n")
            rl()
            q = rl().rstrip(b"\r\n")
            yield h, s, q

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._proc is not None:
                self._proc.wait()


def _read_name(header: bytes) -> bytes:
    n = header[1:].split(None, 1)[0]
    if n.endswith(b"/1") or n.endswith(b"/2") or n.endswith(b"/3"):
        n = n[:-2]
    return n


def _sample_barcode_reads(files: Sequence[str], n: int = 200_000) -> List[str]:
    out: List[str] = []
    for f in files:
        st = _FastqStream(f)
        try:
            for _, s, _ in st:
                out.append(s.decode())
                if len(out) >= n:
                    return out
        finally:
            st.close()
    return out


def _fit_layout(layout: ReadLayout, read_len: int) -> ReadLayout:
    """Shift the barcode window to the end of a barcode read that is shorter
    than the layout expects (e.g. Multiome FASTQs whose 24-bp i5 read was
    written as its 16 barcode bases only)."""
    if read_len and layout.bc_start + layout.bc_length > read_len:
        if layout.bc_length > read_len:
            raise ValueError(f"barcode read is {read_len} bp, shorter than the {layout.bc_length}-bp barcode")
        return replace(layout, bc_start=read_len - layout.bc_length)
    return layout


def _resolve_orientation(layout: ReadLayout, bc_files: Sequence[str], wl_set) -> Tuple[str, Dict[str, float]]:
    sample = _sample_barcode_reads(bc_files)
    o, f_fwd, f_rev = detect_barcode_orientation(sample, wl_set, start=layout.bc_start, length=layout.bc_length,
                                                 min_fraction=0.0 if layout.bc_orientation != "auto" else 0.05)
    info = {"frac_whitelist_forward": f_fwd, "frac_whitelist_reverse": f_rev,
            "barcode_read_length": float(len(sample[0])) if sample else 0.0}
    if layout.bc_orientation != "auto":
        o = layout.bc_orientation
    return o, info


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass
class FragmentsResult(os.PathLike):
    """What :func:`fastq_to_fragments` returns.

    Behaves as the path of the fragments file (``os.fspath(res)``,
    ``str(res)``), and carries the QC summary (``res.qc``, also written to
    ``qc_summary.json``) and the per-barcode table (``barcode_metrics.csv``).
    """

    fragments: str
    qc: Dict[str, float]
    barcode_metrics: str
    outdir: str
    bam: Optional[str] = None
    log: Optional[str] = None

    def __fspath__(self) -> str:
        return self.fragments

    def __str__(self) -> str:
        return self.fragments

    def barcodes(self):
        """The per-barcode table as a DataFrame."""
        import pandas as pd

        return pd.read_csv(self.barcode_metrics)


def _qc_from_fragments(frags, barcodes, n_pairs: int, *, min_fragments: int) -> Tuple[Dict[str, float], object]:
    tab = barcode_table(frags, barcodes)
    n_frag = int(frags["count"].size)
    n_frag_pairs = int(frags["count"].sum()) if n_frag else 0
    big = tab["n_fragments"][tab["n_fragments"] >= min_fragments]
    qc = {
        "n_fragments": n_frag,
        "n_read_pairs_in_fragments": n_frag_pairs,
        "frac_duplicate": (1.0 - n_frag / n_frag_pairs) if n_frag_pairs else float("nan"),
        "frac_read_pairs_in_fragments": (n_frag_pairs / n_pairs) if n_pairs else float("nan"),
        "n_barcodes": int(len(tab)),
        f"n_barcodes_ge_{min_fragments}_fragments": int(big.size),
        f"median_fragments_per_barcode_ge_{min_fragments}": float(np.median(big)) if big.size else 0.0,
    }
    return qc, tab


# ---------------------------------------------------------------------------
# chromap
# ---------------------------------------------------------------------------
def _chromap_read_format(layout: ReadLayout, orientation: str, read_len: int) -> str:
    # chromap slices the raw read first (end inclusive) and then
    # reverse-complements when the field ends in ":-".
    if orientation == "forward":
        s = layout.bc_start
        return f"bc:{s}:{s + layout.bc_length - 1}"
    s = read_len - layout.bc_start - layout.bc_length
    if s < 0:
        raise ValueError("barcode window lies outside the barcode read")
    return f"bc:{s}:{s + layout.bc_length - 1}:-"


_CHROMAP_LOG = {
    "Number of reads": "reads",
    "Number of mapped reads": "mapped_reads",
    "Number of uniquely mapped reads": "uniquely_mapped_reads",
    "Number of barcodes in whitelist": "barcodes_in_whitelist",
    "Number of corrected barcodes": "corrected_barcodes",
    "Number of output mappings (passed filters)": "output_mappings",
}


def _parse_chromap_log(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            key, sep, value = line.partition(":")
            name = _CHROMAP_LOG.get(key.strip())
            if sep and name:
                m = re.match(r"\s*(\d+)", value)
                if m:
                    out[name] = int(m.group(1))
    return out


def _run_chromap(
    fq: Dict[str, List[str]], layout: ReadLayout, orientation: str, read_len: int, *,
    index: str, fasta: str, whitelist_txt: str, threads: int, min_mapq: int,
    max_fragment_length: int, out_bed: str, summary: str, log: str, extra_args: Sequence[str],
) -> None:
    chromap = resolve_tool("chromap")
    cmd = [
        chromap, "--preset", "atac", "-x", index, "-r", fasta,
        "-1", ",".join(fq[layout.read1]), "-2", ",".join(fq[layout.read2]),
        "-b", ",".join(fq[layout.barcode]),
        "--barcode-whitelist", whitelist_txt,
        "--read-format", _chromap_read_format(layout, orientation, read_len),
        "-t", str(max(1, threads)), "-q", str(min_mapq), "-l", str(max_fragment_length),
        "-o", out_bed, "--summary", summary,
    ]
    cmd += list(map(str, extra_args))
    with open(log, "w") as lf:
        lf.write(" ".join(cmd) + "\n")
        lf.flush()
        p = subprocess.run(cmd, stdout=lf, stderr=lf, env=build_env())
    if p.returncode != 0:
        raise CommandError(f"chromap failed (exit {p.returncode}); see {log}")


def _load_bed_fragments(bed: str, chroms: Sequence[str]):
    import pandas as pd

    df = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 1, 2, 3, 4],
                     names=["chrom", "start", "end", "barcode", "count"],
                     dtype={"chrom": "category", "barcode": "category", "start": np.int64,
                            "end": np.int64, "count": np.int64})
    pos = {c: i for i, c in enumerate(chroms)}
    cat_to_cid = np.asarray([pos.get(c, -1) for c in df["chrom"].cat.categories], dtype=np.int64)
    cid = cat_to_cid[df["chrom"].cat.codes.to_numpy()]
    keep = cid >= 0
    bcs = list(df["barcode"].cat.categories.astype(str))
    d = dedup_fragments(cid[keep], df["start"].to_numpy()[keep], df["end"].to_numpy()[keep],
                        df["barcode"].cat.codes.to_numpy().astype(np.int64)[keep],
                        count=df["count"].to_numpy()[keep], barcode_names=bcs)
    return d, bcs


# ---------------------------------------------------------------------------
# bwa-mem2 / bowtie2
# ---------------------------------------------------------------------------
class _BarcodeStats:
    __slots__ = ("pairs", "exact", "corrected", "invalid")

    def __init__(self) -> None:
        self.pairs = self.exact = self.corrected = self.invalid = 0


def _count_barcodes(bc_files: Sequence[str], layout: ReadLayout, orientation: str, corrector: BarcodeCorrector) -> int:
    """Pass 1: exact whitelist hits -> the correction prior."""
    from collections import Counter

    s0, ln = layout.bc_start, layout.bc_length
    total = 0
    for f in bc_files:
        st = _FastqStream(f)
        c: Counter = Counter()
        try:
            if orientation == "forward":
                for _, s, _ in st:
                    c[s[s0:s0 + ln]] += 1
            else:
                for _, s, _ in st:
                    c[s] += 1
        finally:
            st.close()
        if orientation == "forward":
            items = ((k.decode(), v) for k, v in c.items())
        else:
            agg: Counter = Counter()
            for k, v in c.items():
                agg[revcomp(k.decode())[s0:s0 + ln]] += v
            items = agg.items()
        idx = corrector.index
        for bc, n in items:
            total += n
            i = idx.get(bc)
            if i is not None:
                corrector.counts[i] += n
    return total


def _write_tagged_pairs(
    fq: Dict[str, List[str]], layout: ReadLayout, orientation: str, corrector: BarcodeCorrector,
    sink, stats: _BarcodeStats, check_every: int = 100_000,
) -> None:
    """Pass 2: interleaved FASTQ with ``CR``/``CY``/``CB`` in the comment."""
    s0, ln = layout.bc_start, layout.bc_length
    index = corrector.index
    cache: Dict[Tuple[str, str], Optional[str]] = {}
    for f1, fb, f2 in zip(fq[layout.read1], fq[layout.barcode], fq[layout.read2]):
        a, b, c = _FastqStream(f1), _FastqStream(fb), _FastqStream(f2)
        buf: List[bytes] = []
        try:
            for i, ((h1, s1, q1), (hb, sb, qb), (h2, s2, q2)) in enumerate(zip(a, b, c)):
                name = _read_name(h1)
                if i % check_every == 0 and not (name == _read_name(hb) == _read_name(h2)):
                    raise ValueError(f"read names out of sync in {f1} / {fb} / {f2} at record {i}")
                seq, qual = orient_barcode(sb.decode(), qb.decode(), orientation, s0, ln)
                stats.pairs += 1
                if seq in index:
                    bc = seq
                    stats.exact += 1
                else:
                    key = (seq, qual)
                    bc = cache.get(key, "")
                    if bc == "":
                        bc = corrector.correct(seq, qual)
                        if len(cache) > 1_000_000:
                            cache.clear()
                        cache[key] = bc
                    if bc is None:
                        stats.invalid += 1
                    else:
                        stats.corrected += 1
                tag = f"CR:Z:{seq}\tCY:Z:{qual}"
                if bc is not None:
                    tag += f"\tCB:Z:{bc}"
                t = tag.encode()
                buf.append(b"@%s %s\n%s\n+\n%s\n@%s %s\n%s\n+\n%s\n" % (name, t, s1, q1, name, t, s2, q2))
                if len(buf) >= 4096:
                    sink.write(b"".join(buf))
                    buf.clear()
            if buf:
                sink.write(b"".join(buf))
        finally:
            a.close(); b.close(); c.close()


def _align_stream(
    fq, layout, orientation, corrector, *, aligner: str, index: str, threads: int,
    trim: bool, out_bam: str, log: str, outdir: Path, extra_args: Sequence[str],
) -> _BarcodeStats:
    samtools = resolve_tool("samtools")
    env = build_env()
    if aligner == "bwa-mem2":
        al = [resolve_tool("bwa-mem2"), "mem", "-p", "-C", "-t", str(threads), *map(str, extra_args), index, "/dev/stdin"]
    elif aligner == "bowtie2":
        al = [resolve_tool("bowtie2"), "-p", str(threads), "-X", "2000", "--very-sensitive",
              "--sam-append-comment", *map(str, extra_args), "-x", index, "--interleaved", "-"]
    else:
        raise ValueError(aligner)
    procs: List[subprocess.Popen] = []
    lf = open(log, "a")
    stats = _BarcodeStats()
    err: List[BaseException] = []
    try:
        lf.write(" ".join(al) + "\n"); lf.flush()
        if trim:
            fastp = [resolve_tool("fastp"), "--stdin", "--interleaved_in", "--stdout", "-Q",
                     "--detect_adapter_for_pe", "-w", str(min(16, max(1, threads))),
                     "-j", str(outdir / "fastp.json"), "-h", str(outdir / "fastp.html")]
            lf.write(" ".join(fastp) + "\n"); lf.flush()
            pf = subprocess.Popen(fastp, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=lf, env=env)
            pa = subprocess.Popen(al, stdin=pf.stdout, stdout=subprocess.PIPE, stderr=lf, env=env)
            pf.stdout.close()
            head, procs = pf, [pf, pa]
        else:
            pa = subprocess.Popen(al, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=lf, env=env)
            head, procs = pa, [pa]
        ps = subprocess.Popen([samtools, "view", "-b", "-@", "2", "-o", out_bam, "-"], stdin=pa.stdout,
                              stderr=lf, env=env)
        pa.stdout.close()
        procs.append(ps)
        try:
            _write_tagged_pairs(fq, layout, orientation, corrector, head.stdin, stats)
        except BrokenPipeError as e:  # aligner died; report its exit code below
            err.append(e)
        finally:
            try:
                head.stdin.close()
            except BrokenPipeError:
                pass
        codes = [p.wait() for p in procs]
    finally:
        lf.close()
    if any(codes) or err:
        raise CommandError(f"{aligner} pipeline failed (exit codes {codes}); see {log}")
    return stats


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def fastq_to_fragments(
    fastqs: Optional[Union[Mapping[str, Union[PathLike, Sequence[PathLike]]], Sequence[PathLike]]] = None,
    *,
    fastq_dir: Optional[PathLike] = None,
    sample: Optional[str] = None,
    reference: Union[PathLike, Mapping[str, str]],
    whitelist: PathLike,
    outdir: PathLike,
    aligner: str = "chromap",
    chemistry: Union[str, ReadLayout, Mapping] = "atac",
    index: Optional[PathLike] = None,
    threads: int = 8,
    min_mapq: int = 30,
    max_fragment_length: int = 2000,
    barcode_translation: Optional[Union[PathLike, Mapping[str, str], Tuple[PathLike, PathLike]]] = None,
    bc_confidence: float = 0.975,
    bc_pseudocount: float = 1.0,
    trim_adapters: Union[str, bool] = "auto",
    chroms: Optional[Sequence[str]] = None,
    keep_bam: bool = False,
    min_fragments_qc: int = 1000,
    aligner_args: Sequence[str] = (),
    overwrite: bool = False,
) -> FragmentsResult:
    """Single-cell ATAC FASTQs -> ``fragments.tsv.gz`` (+ ``.tbi``) and QC.

    Parameters
    ----------
    fastqs
        ``{"R1": [...], "R2": [...], "R3": [...]}`` (lanes in order), or a
        flat list of Illumina-named files. Alternatively give ``fastq_dir``
        (+ ``sample`` when the folder holds several samples).
    reference
        Genome FASTA (``.fa`` or ``.fa.gz``) or the dict returned by
        :func:`~epione.upstream.prepare_reference`. The aligner index is
        built next to the FASTA when missing, unless ``index`` points to an
        existing one (chromap index file, bwa-mem2 / bowtie2 prefix).
    whitelist
        Barcode whitelist (``737K-cratac-v1.txt`` for 10x scATAC,
        ``737K-arc-v1.txt`` ATAC version for Multiome; plain or gzipped).
        10x ships these with Cell Ranger ATAC / ARC under
        ``lib/python/atac/barcodes/``.
    outdir
        Output folder: ``fragments.tsv.gz``, ``fragments.tsv.gz.tbi``,
        ``qc_summary.json``, ``barcode_metrics.csv``, the aligner log and,
        with ``keep_bam=True`` (bwa-mem2 / bowtie2), ``possorted.bam``.
    aligner
        ``"chromap"`` (default), ``"bwa-mem2"`` or ``"bowtie2"``.
    chemistry
        ``"atac"`` (16-bp barcode read), ``"multiome"`` (24-bp barcode read,
        barcode at 8-23), or a :class:`ReadLayout` / dict. When there is no
        ``R3`` but an ``I2`` file, the barcode is taken from ``I2``. The
        barcode-read orientation is detected from the data unless the
        layout fixes it.
    min_mapq, max_fragment_length
        Both mates must reach ``min_mapq``; pairs with an insert longer than
        ``max_fragment_length`` are dropped (chromap ``-q`` / ``-l``).
    barcode_translation
        Rename barcodes in the output: a two-column file, a dict, or a
        ``(atac_whitelist, gex_whitelist)`` pair, paired line by line (how
        Cell Ranger ARC reports Multiome ATAC fragments under GEX barcodes).
    bc_confidence, bc_pseudocount
        Posterior threshold and prior pseudocount of the bwa-mem2 / bowtie2
        barcode correction (see :class:`~epione.upstream.BarcodeCorrector`).
        chromap uses its own correction (``--bc-probability-threshold``,
        0.9 by default; pass it through ``aligner_args``).
    trim_adapters
        Run ``fastp`` adapter trimming (paired-end overlap detection) before
        the aligner. ``"auto"``: on for bowtie2 (end-to-end alignment cannot
        place reads that run into adapter), off for bwa-mem2 (local
        alignment soft-clips it) and chromap (which trims itself).
    chroms
        Keep and order these chromosomes (default: all FASTA sequences, in
        FASTA order).
    keep_bam
        bwa-mem2 / bowtie2 only: keep a coordinate-sorted, indexed BAM with
        ``CR``/``CY``/``CB`` tags.
    min_fragments_qc
        Barcodes with at least this many fragments are summarised in the QC
        (count and median fragments per barcode).
    aligner_args
        Extra command-line arguments for the aligner.
    overwrite
        Re-run even if ``fragments.tsv.gz`` already exists in ``outdir``.

    Returns
    -------
    :class:`FragmentsResult` -- usable as the fragments path; ``.qc`` holds
    ``n_read_pairs``, ``frac_valid_barcode``, ``frac_mapped``,
    ``frac_duplicate``, fragments per barcode, and wall time.
    """
    t0 = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_gz = outdir / "fragments.tsv.gz"
    qc_path = outdir / "qc_summary.json"
    bc_path = outdir / "barcode_metrics.csv"
    log = outdir / f"{aligner}.log"
    bam_path = outdir / "possorted.bam"
    if out_gz.exists() and Path(str(out_gz) + ".tbi").exists() and qc_path.exists() and not overwrite:
        qc = json.loads(qc_path.read_text())
        return FragmentsResult(str(out_gz), qc, str(bc_path), str(outdir),
                               str(bam_path) if bam_path.exists() else None, str(log))
    if aligner not in ("chromap", "bwa-mem2", "bowtie2"):
        raise ValueError("aligner must be 'chromap', 'bwa-mem2' or 'bowtie2'")

    # --- inputs ---------------------------------------------------------
    if fastqs is None:
        if fastq_dir is None:
            raise ValueError("give fastqs or fastq_dir")
        fq = find_fastqs(fastq_dir, sample)
    elif isinstance(fastqs, Mapping):
        fq = {k: ([str(v)] if isinstance(v, (str, Path)) else [str(x) for x in v]) for k, v in fastqs.items()}
    else:
        fq = _fastqs_from_list(fastqs)
    layout = _resolve_layout(chemistry, list(fq))
    for role in (layout.read1, layout.read2, layout.barcode):
        if role not in fq:
            raise ValueError(f"no {role} FASTQ for layout {layout} (have {sorted(fq)})")
    if not (len(fq[layout.read1]) == len(fq[layout.read2]) == len(fq[layout.barcode])):
        raise ValueError("read1 / read2 / barcode FASTQ lists differ in length")

    if isinstance(reference, Mapping):
        fasta = Path(reference["fasta"])
        if index is None and reference.get("ref_index") and Path(str(reference["ref_index"])).exists():
            index = reference["ref_index"]
    else:
        fasta = Path(reference)
    fasta = ensure_fasta_unzipped(fasta)
    fai = ensure_fasta_index(fasta)
    fasta_chroms = [l.split("\t", 1)[0] for l in open(fai)]
    chroms_out = list(chroms) if chroms is not None else fasta_chroms

    wl = load_whitelist(whitelist)
    wl_set = set(wl)
    translation: Optional[Dict[str, str]] = None
    if barcode_translation is not None:
        if isinstance(barcode_translation, Mapping):
            translation = dict(barcode_translation)
        elif isinstance(barcode_translation, (tuple, list)) and len(barcode_translation) == 2:
            translation = barcode_translation_table(*barcode_translation)
        else:
            translation = load_barcode_translation(barcode_translation)

    probe = _sample_barcode_reads(fq[layout.barcode], n=1)
    if not probe:
        raise ValueError(f"barcode FASTQ {fq[layout.barcode][0]} is empty")
    read_len = len(probe[0])
    layout = _fit_layout(layout, read_len)
    orientation, orient_info = _resolve_orientation(layout, fq[layout.barcode], wl_set)

    if index is None:
        index = ensure_aligner_index(aligner, fasta, threads=threads)
    index = str(index)
    if log.exists():
        log.unlink()

    qc: Dict[str, float] = {
        "aligner": aligner,
        "chemistry": chemistry if isinstance(chemistry, str) else "custom",
        "layout": asdict(layout),
        "barcode_orientation": orientation,
        **orient_info,
        "threads": threads,
        "min_mapq": min_mapq,
    }
    bam_out: Optional[str] = None

    if aligner == "chromap":
        tmpdir = outdir / "_tmp"
        tmpdir.mkdir(exist_ok=True)
        wl_txt = tmpdir / "whitelist.txt"
        wl_txt.write_text("\n".join(wl) + "\n")
        bed = tmpdir / "chromap.bed"
        summary = outdir / "chromap_summary.csv"
        _run_chromap(fq, layout, orientation, read_len, index=index, fasta=str(fasta),
                     whitelist_txt=str(wl_txt), threads=threads, min_mapq=min_mapq,
                     max_fragment_length=max_fragment_length, out_bed=str(bed), summary=str(summary),
                     log=str(log), extra_args=aligner_args)
        st = _parse_chromap_log(str(log))
        frags, barcodes = _load_bed_fragments(str(bed), chroms_out)
        n_pairs = st.get("reads", 0) // 2
        qc.update({
            "n_read_pairs": n_pairs,
            "frac_valid_barcode": (st.get("barcodes_in_whitelist", 0) + st.get("corrected_barcodes", 0)) / n_pairs if n_pairs else float("nan"),
            "frac_corrected_barcode": st.get("corrected_barcodes", 0) / n_pairs if n_pairs else float("nan"),
            "frac_mapped": st.get("mapped_reads", 0) / st["reads"] if st.get("reads") else float("nan"),
        })
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        trim = (aligner == "bowtie2") if trim_adapters == "auto" else bool(trim_adapters)
        corrector = BarcodeCorrector(wl, confidence=bc_confidence, pseudocount=bc_pseudocount)
        _count_barcodes(fq[layout.barcode], layout, orientation, corrector)
        raw_bam = outdir / "_aligned.unsorted.bam"
        bstats = _align_stream(fq, layout, orientation, corrector, aligner=aligner, index=index,
                               threads=threads, trim=trim, out_bam=str(raw_bam), log=str(log),
                               outdir=outdir, extra_args=aligner_args)
        frags, _, barcodes, pstats = collect_fragments(
            raw_bam, barcode_tag="CB", min_mapq=min_mapq, max_fragment_length=max_fragment_length,
            chroms=chroms_out, threads=1,
        )
        n_pairs = bstats.pairs
        qc.update({
            "n_read_pairs": n_pairs,
            "frac_valid_barcode": (bstats.exact + bstats.corrected) / n_pairs if n_pairs else float("nan"),
            "frac_corrected_barcode": bstats.corrected / n_pairs if n_pairs else float("nan"),
            "n_read_pairs_aligned": pstats.pairs,
            "frac_mapped": pstats.both_mapped / pstats.pairs if pstats.pairs else float("nan"),
            "frac_confidently_mapped": pstats.passing / pstats.pairs if pstats.pairs else float("nan"),
            "adapter_trimming": trim,
        })
        if keep_bam:
            samtools = resolve_tool("samtools")
            subprocess.run([samtools, "sort", "-@", str(max(1, threads)), "-m", "1G", "-o", str(bam_path),
                            str(raw_bam)], check=True, env=build_env())
            subprocess.run([samtools, "index", "-@", str(max(1, threads)), str(bam_path)], check=True, env=build_env())
            bam_out = str(bam_path)
        raw_bam.unlink()

    fqc, tab = _qc_from_fragments(frags, barcodes, qc["n_read_pairs"], min_fragments=min_fragments_qc)
    if translation:
        tab["barcode"] = [translation.get(b, b) for b in tab["barcode"]]
    write_fragments(frags, chroms_out, barcodes, out_gz, barcode_translation=translation, threads=threads)
    tab.to_csv(bc_path, index=False)
    qc.update(fqc)
    qc["wall_time_s"] = round(time.time() - t0, 1)
    qc_path.write_text(json.dumps(qc, indent=2, default=float))
    return FragmentsResult(str(out_gz), qc, str(bc_path), str(outdir), bam_out, str(log))


__all__ = [
    "ReadLayout",
    "CHEMISTRIES",
    "FragmentsResult",
    "find_fastqs",
    "fastq_to_fragments",
]
