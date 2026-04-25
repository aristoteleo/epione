import h5py
from scipy.sparse import csr_matrix
import pandas as pd
import anndata as ad
import numpy as np
from scipy.io import mmread

import re
import gzip
from collections import defaultdict
from urllib.parse import unquote

from typing import Optional, Union
from tqdm import tqdm

from epione.core import console

def read_ATAC_10x(matrix, cell_names='', var_names='', path_file=''):
    """
    Copy from Episcanpy
    Load sparse matrix (including matrices corresponding to 10x data) as AnnData objects.
    read the mtx file, tsv file coresponding to cell_names and the bed file containing the variable names

    Parameters
    ----------
    matrix: sparse count matrix

    cell_names: optional, tsv file containing cell names

    var_names: optional, bed file containing the feature names

    Return
    ------
    AnnData object

    """

    
    mat = mmread(''.join([path_file, matrix])).tocsr().transpose()
    
    with open(path_file+cell_names) as f:
        barcodes = f.readlines() 
        barcodes = [x[:-1] for x in barcodes]
        
    with open(path_file+var_names) as f:
        var_names = f.readlines()
        var_names = ["_".join(x[:-1].split('\t')) for x in var_names]
        
    adata = ad.AnnData(mat, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=var_names))
    adata.uns['omic'] = 'ATAC'
    
    return(adata)

def get_gene_annotation(
    genome_or_path,
    *,
    feature: str = "gene",
    gene_type: Optional[Union[str, list, tuple, set]] = "protein_coding",
    exclude_chroms: Optional[Union[list, tuple, set]] = ("chrM", "chrMT"),
) -> pd.DataFrame:
    """Return a per-gene DataFrame from a Gencode GTF/GFF3 annotation.

    Columns: ``gene_name / chrom / start / end / strand`` — the format
    :func:`epi.pp.tsse` and :func:`epi.tl.add_gene_score_matrix` expect.
    Coordinates are 1-based inclusive (UCSC/GFF convention).

    Parameters
    ----------
    genome_or_path
        A :class:`~epione.utils.genome.Genome` (``.annotation`` resolves
        the cached GFF3) or a path to a GTF/GFF3 file (``.gz`` ok).
        This is what auto-downloads the Gencode annotation on first
        call via pooch — no hardcoded TSV needed.
    feature
        GFF/GTF feature type to keep (default ``"gene"`` — one row per
        gene; use ``"transcript"`` to keep isoforms).
    gene_type
        Keep only the specified ``gene_type`` / ``gene_biotype``
        (default ``"protein_coding"``). Pass ``None`` to keep all.
    exclude_chroms
        Chromosomes to drop (default mitochondria).

    Returns
    -------
    pandas.DataFrame
        One row per matching gene with columns
        ``gene_name``, ``chrom``, ``start``, ``end``, ``strand``.

    Examples
    --------
    >>> import epione as epi
    >>> genes = epi.utils.get_gene_annotation(epi.utils.genome.hg19)
    >>> epi.pp.tsse(adata, genes)
    >>> epi.tl.add_gene_score_matrix(adata, gene_anno=genes, use_x="auto")
    """
    # Resolve Genome → annotation path.
    from epione.core.genome import Genome
    if isinstance(genome_or_path, Genome):
        path = str(genome_or_path.annotation)
    else:
        path = str(genome_or_path)

    # Auto-detect GFF3 vs GTF by file extension or first non-comment line.
    is_gff3 = ".gff" in path.lower()

    gene_type_set = None
    if gene_type is not None:
        if isinstance(gene_type, str):
            gene_type_set = {gene_type}
        else:
            gene_type_set = set(gene_type)

    exclude_set = set(exclude_chroms) if exclude_chroms else set()

    rows = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t", 8)
            if len(parts) < 8:
                continue
            if parts[2] != feature:
                continue
            chrom = parts[0]
            if chrom in exclude_set:
                continue
            try:
                start = int(parts[3])
                end = int(parts[4])
            except ValueError:
                continue
            strand = parts[6]
            attrs = parts[8] if len(parts) > 8 else ""

            # Parse attributes: GFF3 is ``key=value;``; GTF is ``key "value";``
            name = None
            biotype = None
            if is_gff3:
                for field in attrs.split(";"):
                    field = field.strip()
                    if not field:
                        continue
                    eq = field.find("=")
                    if eq < 0:
                        continue
                    k = field[:eq]; v = field[eq+1:]
                    if k == "gene_name":
                        name = v
                    elif k in ("gene_type", "gene_biotype"):
                        biotype = v
            else:
                for field in attrs.split(";"):
                    field = field.strip()
                    if not field:
                        continue
                    sp = field.split(" ", 1)
                    if len(sp) != 2:
                        continue
                    k, v = sp[0], sp[1].strip().strip('"')
                    if k == "gene_name":
                        name = v
                    elif k in ("gene_type", "gene_biotype"):
                        biotype = v

            if gene_type_set is not None and biotype is not None \
                    and biotype not in gene_type_set:
                continue
            if name is None:
                continue
            rows.append((name, chrom, start, end, strand))

    df = pd.DataFrame(rows, columns=["gene_name", "chrom", "start", "end", "strand"])
    # For duplicate gene names (rare with ``feature='gene'``), keep the
    # longest locus to have one canonical row per gene.
    if df["gene_name"].duplicated().any():
        df["_len"] = df["end"] - df["start"]
        df = df.sort_values("_len", ascending=False).drop_duplicates("gene_name", keep="first")
        df = df.drop(columns="_len").sort_values(["chrom", "start"]).reset_index(drop=True)
    return df


def _parse_gff3_attributes(attr_text: str) -> dict:
    """Parse a GFF3 attribute column into an ordered dict-like mapping."""
    attrs = {}
    if not attr_text:
        return attrs
    for field in attr_text.strip().split(";"):
        field = field.strip()
        if not field or "=" not in field:
            continue
        key, value = field.split("=", 1)
        attrs[key] = unquote(value)
    return attrs


def _format_gtf_attributes(attrs: list[tuple[str, str]]) -> str:
    parts = []
    for key, value in attrs:
        if value is None:
            continue
        value = str(value).replace('"', '\\"')
        parts.append(f'{key} "{value}";')
    return " ".join(parts)


_GFF_GENE_FEATURES = {
    "gene",
    "pseudogene",
    "nc_gene",
}

_GFF_TRANSCRIPT_FEATURES = {
    "transcript",
    "mrna",
    "ncrna",
    "lnc_rna",
    "lncrna",
    "mirna",
    "snrna",
    "snorna",
    "rrna",
    "trna",
    "scrna",
    "srp_rna",
    "antisense_rna",
    "guide_rna",
    "pirna",
    "rnase_mrp_rna",
    "rnase_p_rna",
    "telomerase_rna",
    "vault_rna",
    "y_rna",
    "pseudogenic_transcript",
    "primary_transcript",
}

_GTF3_FEATURE_MAP = {
    "gene": "gene",
    "pseudogene": "gene",
    "nc_gene": "gene",
    "transcript": "transcript",
    "mrna": "transcript",
    "ncrna": "transcript",
    "lnc_rna": "transcript",
    "lncrna": "transcript",
    "mirna": "transcript",
    "snrna": "transcript",
    "snorna": "transcript",
    "rrna": "transcript",
    "trna": "transcript",
    "scrna": "transcript",
    "srp_rna": "transcript",
    "antisense_rna": "transcript",
    "guide_rna": "transcript",
    "pirna": "transcript",
    "rnase_mrp_rna": "transcript",
    "rnase_p_rna": "transcript",
    "telomerase_rna": "transcript",
    "vault_rna": "transcript",
    "y_rna": "transcript",
    "pseudogenic_transcript": "transcript",
    "primary_transcript": "transcript",
    "exon": "exon",
    "cds": "CDS",
    "start_codon": "start_codon",
    "stop_codon": "stop_codon",
    "selenocysteine": "Selenocysteine",
    "five_prime_utr": "five_prime_UTR",
    "5utr": "five_prime_UTR",
    "three_prime_utr": "three_prime_UTR",
    "3utr": "three_prime_UTR",
}


def _normalize_gff_feature(feature: str, gtf_version: str) -> Optional[str]:
    f = str(feature)
    lower = f.lower()
    if gtf_version == "relax":
        return f
    return _GTF3_FEATURE_MAP.get(lower)


def convert_gff_to_gtf(
    gff_path,
    gtf_path=None,
    *,
    gtf_version: str = "3",
    feature_whitelist=None,
    seqname_whitelist=None,
    keep_comments: bool = False,
):
    """
    Convert a GFF/GFF3 annotation into a GTF file.

    This is an AGAT-inspired, pure-Python converter intended for the
    common Gencode / Ensembl / RefSeq-style GFF3 files used in epione
    workflows. It resolves ``ID`` / ``Parent`` relationships, adds
    ``gene_id`` / ``transcript_id`` attributes, standardizes common
    feature names to GTF3, and can restrict output to a subset of
    sequences or features.

    Parameters
    ----------
    gff_path
        Input GFF/GFF3 path (optionally ``.gz``).
    gtf_path
        Output GTF path. If omitted, the suffix is replaced with
        ``.gtf`` (or ``.gtf.gz`` for gzipped input).
    gtf_version
        ``"3"`` for normalized GTF3-style output or ``"relax"`` to keep
        original feature names.
    feature_whitelist
        Optional iterable of input feature names to keep before
        conversion, e.g. ``["transcript"]``.
    seqname_whitelist
        Optional iterable of sequence names to keep, e.g. ``["chr22"]``.
    keep_comments
        When ``True``, copy input comment lines to the output header.

    Returns
    -------
    str
        The output GTF path.
    """
    if gtf_version not in {"3", "relax"}:
        raise ValueError("gtf_version must be '3' or 'relax'")

    gff_path = str(gff_path)
    if gtf_path is None:
        if gff_path.endswith(".gz"):
            base = re.sub(r"(\.gff3?|\.gxf)\.gz$", "", gff_path, flags=re.IGNORECASE)
            gtf_path = base + ".gtf.gz"
        else:
            base = re.sub(r"\.(gff3?|gxf)$", "", gff_path, flags=re.IGNORECASE)
            gtf_path = base + ".gtf"
    gtf_path = str(gtf_path)

    feature_filter = {str(x).lower() for x in feature_whitelist} if feature_whitelist else None
    seqname_filter = {str(x) for x in seqname_whitelist} if seqname_whitelist else None

    comments = []
    records = []
    id_to_record = {}
    parent_to_children = defaultdict(list)

    opener = gzip.open if gff_path.endswith(".gz") else open
    console.level1(f"Converting GFF to GTF: {gff_path} -> {gtf_path}")
    with opener(gff_path, "rt") as fh:
        for idx, line in enumerate(fh):
            if not line:
                continue
            if line.startswith("#"):
                if keep_comments:
                    comments.append(line.rstrip("\n"))
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            seqname, source, feature, start, end, score, strand, frame, attr_text = parts
            if seqname_filter and seqname not in seqname_filter:
                continue
            if feature_filter and feature.lower() not in feature_filter:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except ValueError:
                continue
            attrs = _parse_gff3_attributes(attr_text)
            rec_id = attrs.get("ID") or attrs.get("transcript_id") or attrs.get("gene_id")
            parents = [x for x in attrs.get("Parent", "").split(",") if x]
            rec = {
                "order": idx,
                "seqname": seqname,
                "source": source,
                "feature": feature,
                "start": start_i,
                "end": end_i,
                "score": score,
                "strand": strand,
                "frame": frame,
                "attrs": attrs,
                "id": rec_id,
                "parents": parents,
            }
            records.append(rec)
            if rec_id and rec_id not in id_to_record:
                id_to_record[rec_id] = rec
            for parent in parents:
                parent_to_children[parent].append(rec)

    gene_ids = set()
    transcript_ids = set()
    child_like_features = {
        "exon", "cds", "start_codon", "stop_codon",
        "five_prime_utr", "three_prime_utr", "5utr", "3utr", "utr",
        "selenocysteine",
    }

    for rec in records:
        if not rec["id"]:
            continue
        feature_l = rec["feature"].lower()
        if feature_l in _GFF_GENE_FEATURES:
            gene_ids.add(rec["id"])
        elif feature_l in _GFF_TRANSCRIPT_FEATURES:
            transcript_ids.add(rec["id"])

    for rec in records:
        rec_id = rec["id"]
        if not rec_id or rec_id in gene_ids or rec_id in transcript_ids:
            continue
        children = parent_to_children.get(rec_id, [])
        if any(child["feature"].lower() in child_like_features for child in children):
            transcript_ids.add(rec_id)

    transcript_to_gene = {}
    gene_name_map = {}
    transcript_name_map = {}

    for gene_id in gene_ids:
        rec = id_to_record.get(gene_id)
        if not rec:
            continue
        attrs = rec["attrs"]
        gene_name_map[gene_id] = (
            attrs.get("gene_name")
            or attrs.get("Name")
            or attrs.get("gene")
            or gene_id
        )

    for tx_id in transcript_ids:
        rec = id_to_record.get(tx_id)
        if rec is None:
            continue
        attrs = rec["attrs"]
        gene_id = attrs.get("gene_id")
        if not gene_id:
            for parent in rec["parents"]:
                if parent in gene_ids:
                    gene_id = parent
                    break
        if not gene_id and rec["parents"]:
            gene_id = rec["parents"][0]
        if not gene_id:
            gene_id = tx_id
        transcript_to_gene[tx_id] = gene_id
        transcript_name_map[tx_id] = (
            attrs.get("transcript_name")
            or attrs.get("Name")
            or tx_id
        )
        gene_name_map.setdefault(
            gene_id,
            attrs.get("gene_name") or attrs.get("gene") or attrs.get("Name") or gene_id,
        )

    def iter_contexts(rec):
        attrs = rec["attrs"]
        rec_id = rec["id"]
        normalized_feature = _normalize_gff_feature(rec["feature"], gtf_version)

        if normalized_feature == "gene":
            gene_id = attrs.get("gene_id") or rec_id or attrs.get("Name")
            if gene_id:
                yield gene_id, None
            return

        if normalized_feature == "transcript":
            tx_id = attrs.get("transcript_id") or rec_id or (rec["parents"][0] if rec["parents"] else None)
            if not tx_id:
                return
            gene_id = attrs.get("gene_id") or transcript_to_gene.get(tx_id)
            if not gene_id:
                for parent in rec["parents"]:
                    if parent in gene_ids:
                        gene_id = parent
                        break
            if not gene_id:
                gene_id = tx_id
            yield gene_id, tx_id
            return

        parent_txs = [p for p in rec["parents"] if p in transcript_ids]
        if parent_txs:
            for tx_id in parent_txs:
                gene_id = attrs.get("gene_id") or transcript_to_gene.get(tx_id) or tx_id
                yield gene_id, tx_id
            return

        if attrs.get("transcript_id"):
            tx_id = attrs["transcript_id"]
            gene_id = attrs.get("gene_id") or transcript_to_gene.get(tx_id) or tx_id
            yield gene_id, tx_id
            return

        parent_genes = [p for p in rec["parents"] if p in gene_ids]
        if parent_genes:
            for gene_id in parent_genes:
                tx_id = f"{gene_id}.t1"
                yield gene_id, tx_id
            return

        if rec_id:
            gene_id = attrs.get("gene_id") or rec_id
            tx_id = attrs.get("transcript_id") or rec_id
            yield gene_id, tx_id

    rows = []
    gene_spans = {}
    tx_spans = {}
    emitted_genes = set()
    emitted_transcripts = set()

    feature_rank = {
        "gene": 0,
        "transcript": 1,
        "exon": 2,
        "CDS": 3,
        "five_prime_UTR": 4,
        "three_prime_UTR": 5,
        "start_codon": 6,
        "stop_codon": 7,
        "Selenocysteine": 8,
    }

    def add_span(span_map, key, rec):
        if key not in span_map:
            span_map[key] = {
                "seqname": rec["seqname"],
                "source": rec["source"],
                "start": rec["start"],
                "end": rec["end"],
                "strand": rec["strand"],
            }
        else:
            span_map[key]["start"] = min(span_map[key]["start"], rec["start"])
            span_map[key]["end"] = max(span_map[key]["end"], rec["end"])

    for rec in records:
        feature_out = _normalize_gff_feature(rec["feature"], gtf_version)
        if feature_out is None:
            continue
        contexts = list(iter_contexts(rec))
        if feature_out == "gene" and not contexts:
            gene_id = rec["attrs"].get("gene_id") or rec["id"] or rec["attrs"].get("Name")
            if gene_id:
                contexts = [(gene_id, None)]
        for gene_id, tx_id in contexts:
            add_span(gene_spans, gene_id, rec)
            if tx_id is not None:
                add_span(tx_spans, (gene_id, tx_id), rec)

            ordered_attrs = []
            if gene_id is not None:
                ordered_attrs.append(("gene_id", gene_id))
            if feature_out != "gene" and tx_id is not None:
                ordered_attrs.append(("transcript_id", tx_id))

            gene_name = rec["attrs"].get("gene_name") or gene_name_map.get(gene_id)
            transcript_name = None
            if tx_id is not None:
                transcript_name = rec["attrs"].get("transcript_name") or transcript_name_map.get(tx_id)

            for key, value in [
                ("gene_name", gene_name),
                ("transcript_name", transcript_name),
                ("gene_type", rec["attrs"].get("gene_type") or rec["attrs"].get("gene_biotype")),
                ("transcript_type", rec["attrs"].get("transcript_type") or rec["attrs"].get("transcript_biotype")),
                ("protein_id", rec["attrs"].get("protein_id")),
                ("exon_number", rec["attrs"].get("exon_number")),
            ]:
                if value is not None:
                    ordered_attrs.append((key, value))

            skip_keys = {
                "ID", "Parent", "gene_id", "transcript_id",
                "gene_name", "transcript_name",
                "gene_type", "gene_biotype",
                "transcript_type", "transcript_biotype",
                "protein_id", "exon_number",
            }
            for key, value in rec["attrs"].items():
                if key in skip_keys or value is None:
                    continue
                ordered_attrs.append((key, value))

            rows.append({
                "order": rec["order"],
                "seqname": rec["seqname"],
                "source": rec["source"],
                "feature": feature_out,
                "start": rec["start"],
                "end": rec["end"],
                "score": rec["score"],
                "strand": rec["strand"],
                "frame": rec["frame"],
                "attrs": _format_gtf_attributes(ordered_attrs),
            })
            if feature_out == "gene":
                emitted_genes.add(gene_id)
            elif feature_out == "transcript" and tx_id is not None:
                emitted_transcripts.add((gene_id, tx_id))

    synthetic_order = len(records) + 1
    for (gene_id, tx_id), span in tx_spans.items():
        if (gene_id, tx_id) in emitted_transcripts:
            continue
        attrs = [("gene_id", gene_id), ("transcript_id", tx_id)]
        if gene_id in gene_name_map:
            attrs.append(("gene_name", gene_name_map[gene_id]))
        if tx_id in transcript_name_map:
            attrs.append(("transcript_name", transcript_name_map[tx_id]))
        rows.append({
            "order": synthetic_order,
            "seqname": span["seqname"],
            "source": "epione",
            "feature": "transcript",
            "start": span["start"],
            "end": span["end"],
            "score": ".",
            "strand": span["strand"],
            "frame": ".",
            "attrs": _format_gtf_attributes(attrs),
        })
        synthetic_order += 1

    for gene_id, span in gene_spans.items():
        if gene_id in emitted_genes:
            continue
        attrs = [("gene_id", gene_id)]
        if gene_id in gene_name_map:
            attrs.append(("gene_name", gene_name_map[gene_id]))
        rows.append({
            "order": synthetic_order,
            "seqname": span["seqname"],
            "source": "epione",
            "feature": "gene",
            "start": span["start"],
            "end": span["end"],
            "score": ".",
            "strand": span["strand"],
            "frame": ".",
            "attrs": _format_gtf_attributes(attrs),
        })
        synthetic_order += 1

    rows.sort(
        key=lambda r: (
            r["seqname"],
            r["start"],
            r["end"],
            feature_rank.get(r["feature"], 99),
            r["order"],
        )
    )

    out_opener = gzip.open if gtf_path.endswith(".gz") else open
    with out_opener(gtf_path, "wt") as out:
        if keep_comments:
            for line in comments:
                out.write(line + "\n")
        for row in rows:
            out.write(
                "\t".join([
                    row["seqname"],
                    row["source"],
                    row["feature"],
                    str(row["start"]),
                    str(row["end"]),
                    row["score"],
                    row["strand"],
                    row["frame"],
                    row["attrs"],
                ]) + "\n"
            )

    console.success(f"Wrote {len(rows)} GTF records", level=1)
    return gtf_path


def read_gtf(
    gtf_path,
    required_attrs=("gene_id", "gene_name", "transcript_id"),
    feature_whitelist=None,
    chr_prefix=None,
    keep_attribute=True,
):
    """
    Fast GTF reader with inline attribute parsing (no pandas CSV parser).

    Notes:
    - Streams the file line-by-line (supports .gz) to reduce overhead.
    - Parses only a small set of attributes by default for speed.
    - Keeps the original "attribute" string column for compatibility.

    Parameters
    ----------
    gtf_path : str or path-like
        Path to the GTF file (supports .gz).
    required_attrs : tuple of str
        Attribute keys to extract (e.g., "gene_id", "gene_name").

    Returns
    -------
    pandas.DataFrame
        DataFrame with standard GTF columns and selected attributes.
    """
    console.level2("Reading GTF file from {}...".format(gtf_path))
    req_set = set(required_attrs or ())
    cols = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame"]
    if keep_attribute:
        cols.append("attribute")

    data = {c: [] for c in cols}
    for key in req_set:
        data[key] = []

    def parse_attr_fast(attr_text: str, keys_set: set) -> dict:
        # Expect tokens like: key "value"; key2 "value2";
        out = {}
        if not attr_text:
            return out
        for field in attr_text.split(';'):
            field = field.strip()
            if not field:
                continue
            # split only on the first space to preserve values
            sp = field.split(' ', 1)
            if len(sp) != 2:
                continue
            k, v = sp[0], sp[1]
            if k in keys_set:
                v = v.strip().strip('"')
                out[k] = v
        return out

    feature_set = set(feature_whitelist) if feature_whitelist else None
    prefix = str(chr_prefix) if chr_prefix else None
    opener = gzip.open if str(gtf_path).endswith('.gz') else open
    with opener(gtf_path, 'rt') as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            parts = line.rstrip('\n').split('\t', 8)
            if len(parts) < 8:
                continue
            # Unpack with graceful fallback for optional attribute column
            seqname = parts[0]
            source = parts[1]
            feature = parts[2]
            if prefix and not seqname.startswith(prefix):
                continue
            if feature_set and feature not in feature_set:
                continue
            try:
                start = int(parts[3])
            except Exception:
                # Sometimes GTF can contain malformed entries; skip them
                continue
            try:
                end = int(parts[4])
            except Exception:
                continue
            score = parts[5] if len(parts) > 5 else '.'
            strand = parts[6] if len(parts) > 6 else '.'
            frame = parts[7] if len(parts) > 7 else '.'
            # The raw attribute string is always needed to extract
            # ``required_attrs``. ``keep_attribute`` only controls
            # whether it gets stored as its own column on the output
            # DataFrame. (Previously this guard accidentally blanked
            # ``attribute`` when ``keep_attribute=False``, so every
            # req_attrs lookup returned None — see ``_bigwig.load_gtf``
            # which passes ``keep_attribute=False``.)
            attribute = parts[8] if len(parts) > 8 else ''

            data["seqname"].append(seqname)
            data["source"].append(source)
            data["feature"].append(feature)
            data["start"].append(start)
            data["end"].append(end)
            data["score"].append(score)
            data["strand"].append(strand)
            data["frame"].append(frame)
            if keep_attribute:
                data["attribute"].append(attribute)

            if req_set:
                d = parse_attr_fast(attribute, req_set)
                for k in req_set:
                    data[k].append(d.get(k, None))
    console.success("GTF file read successfully", level=2)
    df = pd.DataFrame(data)
    return df


def fetch_regions_to_df(
    fragment_path: str,
    features: Union[pd.DataFrame, str],
    extend_upstream: int = 0,
    extend_downstream: int = 0,
    relative_coordinates=False,
) -> pd.DataFrame:
    """
    Parse peak annotation file and return it as DataFrame.

    Parameters
    ----------
    fragment_path
        Location of the fragments file (must be tabix indexed).
    features
        A DataFrame with feature annotation, e.g. genes or a string of format `chr1:1-2000000` or`chr1-1-2000000`.
        Annotation has to contain columns: Chromosome, Start, End.
    extend_upsteam
        Number of nucleotides to extend every gene upstream (2000 by default to extend gene coordinates to promoter regions)
    extend_downstream
        Number of nucleotides to extend every gene downstream (0 by default)
    relative_coordinates
        Return the coordinates with their relative position to the middle of the features.
    """

    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. Install pysam from PyPI (`pip install pysam`) or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
        )

    if isinstance(features, str):
        features = parse_region_string(features)

    fragments = pysam.TabixFile(fragment_path, parser=pysam.asBed())
    n_features = features.shape[0]

    dfs = []
    for i in tqdm(
        range(n_features), desc="Fetching Regions..."
    ):  # iterate over features (e.g. genes)
        f = features.iloc[i]
        fr = fragments.fetch(f.Chromosome, f.Start - extend_upstream, f.End + extend_downstream)
        df = pd.DataFrame(
            [(x.contig, x.start, x.end, x.name, x.score) for x in fr],
            columns=["Chromosome", "Start", "End", "Cell", "Score"],
        )
        if df.shape[0] != 0:
            df["Feature"] = f.Chromosome + "_" + str(f.Start) + "_" + str(f.End)

            if relative_coordinates:
                middle = int(f.Start + (f.End - f.Start) / 2)
                df.Start = df.Start - middle
                df.End = df.End - middle

            dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df




def parse_region_string(region: str) -> pd.DataFrame:
    feat_list = re.split("-|:", region)
    feature_df = pd.DataFrame(columns=["Chromosome", "Start", "End"])
    feature_df.loc[0] = feat_list
    feature_df = feature_df.astype({"Start": int, "End": int})

    return feature_df


def read_features(
    features_path,
    feature_type="transcript",
    chr_prefix=None,
    keep_attribute=True,
    annotation="HAVANA",
    
):
    """
    Read features from a file.
    """
    console.level1("Reading features...")
    features = read_gtf(features_path, feature_whitelist=[feature_type], chr_prefix=chr_prefix, keep_attribute=keep_attribute)
    features=features.loc[(features['feature']==feature_type)&(features['source']==annotation)]
    features.loc[:, 'Chromosome'] = features['seqname']
    features.loc[:, 'Start'] = features['start']
    features.loc[:, 'End'] = features['end']    
    console.success("Features read successfully", level=1)
    return features
