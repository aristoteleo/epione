from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


class CommandError(RuntimeError):
    pass


def _run(
    cmd: Sequence[str],
    cwd: Optional[Union[str, Path]] = None,
    stream: bool = True,
) -> None:
    """Run a command and raise a readable error on failure."""
    if stream:
        proc = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None)
        if proc.returncode != 0:
            raise CommandError(f"Command failed: {' '.join(map(str, cmd))}")
        return

    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise CommandError(
            f"Command failed: {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _resolve_tool(name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        if shutil.which(explicit) is None:
            raise FileNotFoundError(f"Requested binary '{explicit}' for '{name}' not found in PATH.")
        return explicit

    try:
        from ._env import tool_path

        return tool_path(name)
    except Exception:
        path = shutil.which(name)
        if path is None:
            raise FileNotFoundError(
                f"Required binary '{name}' not found in PATH. Please install it and try again."
            )
        return path


def _run_pipe(commands: List[List[str]], cwd: Optional[Union[str, Path]] = None) -> None:
    """Run a pipeline and fail if any stage fails."""
    procs = []
    prev_stdout = None
    try:
        for i, cmd in enumerate(commands):
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdin=prev_stdout,
                stdout=subprocess.PIPE if i < len(commands) - 1 else None,
            )
            procs.append(proc)
            if prev_stdout is not None:
                prev_stdout.close()
            prev_stdout = proc.stdout

        for proc in procs:
            proc.wait()

        failed = [cmd for cmd, proc in zip(commands, procs) if proc.returncode != 0]
        if failed:
            raise CommandError(
                "Pipeline failed:\n" + "\n".join("  " + " ".join(cmd) for cmd in failed)
            )
    finally:
        for proc in procs:
            if proc.stdout is not None:
                proc.stdout.close()


def ensure_fasta_unzipped(fa_gz: Union[str, Path]) -> Path:
    fa_gz = Path(fa_gz)
    if fa_gz.suffix != ".gz":
        return fa_gz
    out = fa_gz.with_suffix("")
    if out.exists():
        return out
    with gzip.open(fa_gz, "rb") as fin, open(out, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return out


def ensure_fasta_index(fasta: Union[str, Path]) -> Path:
    fasta = Path(fasta)
    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if fai.exists():
        return fai
    samtools = _resolve_tool("samtools")
    _run([samtools, "faidx", str(fasta)])
    return fai


def ensure_chrom_sizes(
    fasta: Optional[Union[str, Path]] = None,
    chrom_sizes: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Ensure a UCSC-style chrom.sizes file exists.

    If ``chrom_sizes`` already exists it is returned unchanged. Otherwise
    ``fasta`` is indexed with ``samtools faidx`` and the first two columns of
    ``.fai`` are written to ``<fasta>.chrom.sizes``.
    """
    if chrom_sizes is not None:
        chrom_sizes = Path(chrom_sizes)
        if chrom_sizes.exists():
            return chrom_sizes

    if fasta is None:
        raise ValueError("Provide either chrom_sizes or fasta.")

    fasta = Path(fasta)
    fai = ensure_fasta_index(fasta)
    out = Path(chrom_sizes) if chrom_sizes is not None else fasta.with_suffix(".chrom.sizes")
    with open(fai) as fin, open(out, "w") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            fout.write(f"{parts[0]}\t{parts[1]}\n")
    return out


def ensure_aligner_index(
    aligner: str,
    fasta: Union[str, Path],
    index_prefix: Union[str, Path, None] = None,
    overwrite: bool = False,
) -> str:
    """
    Ensure reference index exists for the chosen aligner; build if missing.
    Returns the index prefix usable by the aligner.
    """
    fasta = Path(fasta)
    if aligner == "bwa-mem2":
        bwa_mem2 = _resolve_tool("bwa-mem2")
        if index_prefix is None:
            prefix_path = fasta
        else:
            prefix_path = Path(index_prefix)
            prefix_path.parent.mkdir(parents=True, exist_ok=True)
        idx_candidates = [Path(str(prefix_path) + ext) for ext in [".0123", ".amb", ".ann", ".bwt"]]
        if overwrite or not all(p.exists() for p in idx_candidates):
            cmd = [bwa_mem2, "index"]
            if index_prefix is not None:
                cmd += ["-p", str(prefix_path)]
            cmd += [str(fasta)]
            _run(cmd)
        return str(prefix_path)

    if aligner == "bowtie2":
        bowtie2_build = _resolve_tool("bowtie2-build")
        if index_prefix is None:
            prefix = str(Path(fasta).with_suffix(""))
        else:
            prefix = str(index_prefix)
            Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        bt2_files = [
            Path(prefix + ext)
            for ext in [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]
        ]
        if overwrite or not all(p.exists() for p in bt2_files):
            _run([bowtie2_build, str(fasta), prefix])
        return prefix

    raise ValueError("aligner must be 'bwa-mem2' or 'bowtie2'")


def prepare_reference(
    *,
    genome: Optional[str] = None,
    fasta: Optional[Union[str, Path]] = None,
    chrom_sizes: Optional[Union[str, Path]] = None,
    aligner: str = "bowtie2",
    index_prefix: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    """
    Fetch/build everything the bulk upstream pipeline needs for one reference.
    """
    if fasta is None:
        if genome is None:
            raise ValueError("Provide either fasta or genome.")
        fasta = fetch_genome_fasta(genome)
    fasta = ensure_fasta_unzipped(fasta)
    fai = ensure_fasta_index(fasta)
    chrom_sizes = ensure_chrom_sizes(fasta=fasta, chrom_sizes=chrom_sizes)
    ref_index = ensure_aligner_index(
        aligner=aligner,
        fasta=fasta,
        index_prefix=index_prefix,
        overwrite=overwrite,
    )
    return {
        "fasta": str(fasta),
        "fai": str(fai),
        "chrom_sizes": str(chrom_sizes),
        "ref_index": str(ref_index),
    }


def trim_fastq_pair(
    fq1: Union[str, Path],
    fq2: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    sample_name: Optional[str] = None,
    method: str = "fastp",
    threads: int = 8,
    html_report: Optional[Union[str, Path]] = None,
    json_report: Optional[Union[str, Path]] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    """
    Trim a paired-end FASTQ pair and return the trimmed R1/R2 paths.

    Supports ``fastp`` and ``trim_galore``. ``fastp`` is preferred because it
    produces deterministic output file names that are easier to use in
    notebooks.
    """
    fq1 = str(fq1)
    fq2 = str(fq2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_name = sample_name or Path(fq1).name.split("_R1")[0].split("_1")[0]

    if method == "fastp":
        fastp = _resolve_tool("fastp")
        out1 = out_dir / f"{sample_name}_R1.trim.fastq.gz"
        out2 = out_dir / f"{sample_name}_R2.trim.fastq.gz"
        html_report = Path(html_report) if html_report is not None else out_dir / f"{sample_name}.fastp.html"
        json_report = Path(json_report) if json_report is not None else out_dir / f"{sample_name}.fastp.json"
        cmd = [
            fastp,
            "-i",
            fq1,
            "-I",
            fq2,
            "-o",
            str(out1),
            "-O",
            str(out2),
            "-w",
            str(threads),
            "-h",
            str(html_report),
            "-j",
            str(json_report),
        ]
        if extra_args:
            cmd.extend(map(str, extra_args))
        _run(cmd)
        return str(out1), str(out2)

    if method == "trim_galore":
        trim_galore = _resolve_tool("trim_galore")
        cmd = [
            trim_galore,
            "--paired",
            "--gzip",
            "--cores",
            str(max(1, threads)),
            "-o",
            str(out_dir),
            fq1,
            fq2,
        ]
        if extra_args:
            cmd.extend(map(str, extra_args))
        _run(cmd)
        r1_val = out_dir / (Path(fq1).name.replace(".fastq.gz", "_val_1.fq.gz").replace(".fq.gz", "_val_1.fq.gz"))
        r2_val = out_dir / (Path(fq2).name.replace(".fastq.gz", "_val_2.fq.gz").replace(".fq.gz", "_val_2.fq.gz"))
        if not r1_val.exists() or not r2_val.exists():
            raise CommandError("trim_galore finished but expected output FASTQs were not found.")
        return str(r1_val), str(r2_val)

    raise ValueError("method must be 'fastp' or 'trim_galore'")


def sort_bam(
    in_bam: Union[str, Path],
    out_bam: Union[str, Path],
    *,
    threads: int = 4,
    name_sort: bool = False,
) -> str:
    samtools = _resolve_tool("samtools")
    cmd = [samtools, "sort"]
    if name_sort:
        cmd.append("-n")
    cmd += ["-@", str(threads), "-o", str(out_bam), str(in_bam)]
    _run(cmd)
    return str(out_bam)


def index_bam(bam: Union[str, Path], *, threads: int = 4) -> str:
    samtools = _resolve_tool("samtools")
    _run([samtools, "index", "-@", str(threads), str(bam)])
    return str(bam) + ".bai"


def merge_bams(
    bam_files: Sequence[Union[str, Path]],
    out_bam: Union[str, Path],
    *,
    threads: int = 4,
    index: bool = True,
) -> str:
    if len(bam_files) == 0:
        raise ValueError("bam_files cannot be empty.")
    samtools = _resolve_tool("samtools")
    out_bam = Path(out_bam)
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    _run([samtools, "merge", "-@", str(threads), "-o", str(out_bam), *map(str, bam_files)])
    if index:
        index_bam(out_bam, threads=threads)
    return str(out_bam)


def align_fastq_to_bam(
    fq1: Union[str, Path],
    fq2: Union[str, Path],
    out_bam: Union[str, Path],
    *,
    aligner: str = "bwa-mem2",
    ref_index: Optional[Union[str, Path]] = None,
    threads: int = 8,
    rg: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    remove_duplicates: bool = False,
) -> str:
    """
    Align paired-end FASTQs to reference and produce a coordinate-sorted BAM.
    """
    if ref_index is None:
        raise ValueError("ref_index is required (path prefix to the aligner index)")

    out_bam = Path(out_bam)
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_bam.with_suffix("")).rstrip(".")
    tmp_dir = out_bam.parent

    name_sorted = str(tmp_dir / (Path(out_prefix).name + ".namesort.bam"))
    fixmate_bam = str(tmp_dir / (Path(out_prefix).name + ".fixmate.bam"))
    coord_sorted = str(tmp_dir / (Path(out_prefix).name + ".coordsort.bam"))

    samtools = _resolve_tool("samtools")

    if aligner == "bwa-mem2":
        aligner_bin = _resolve_tool("bwa-mem2")
        cmd = [aligner_bin, "mem", "-t", str(threads), str(ref_index), str(fq1), str(fq2)]
        if rg:
            cmd.extend(["-R", rg])
        if extra_args:
            cmd.extend(map(str, extra_args))
    elif aligner == "bowtie2":
        aligner_bin = _resolve_tool("bowtie2")
        cmd = [
            aligner_bin,
            "-x",
            str(ref_index),
            "-1",
            str(fq1),
            "-2",
            str(fq2),
            "-p",
            str(threads),
        ]
        if extra_args:
            cmd.extend(map(str, extra_args))
    else:
        raise ValueError("aligner must be 'bwa-mem2' or 'bowtie2'")

    _run_pipe(
        [
            cmd,
            [samtools, "view", "-b", "-"],
            [samtools, "sort", "-n", "-@", str(threads), "-o", name_sorted, "-"],
        ]
    )

    _run([samtools, "fixmate", "-m", name_sorted, fixmate_bam])
    _run([samtools, "sort", "-@", str(threads), "-o", coord_sorted, fixmate_bam])

    md_cmd = [samtools, "markdup", "-@", str(threads)]
    if remove_duplicates:
        md_cmd.append("-r")
    md_cmd += [coord_sorted, str(out_bam)]
    _run(md_cmd)

    for f in (name_sorted, fixmate_bam, coord_sorted):
        try:
            os.remove(f)
        except OSError:
            pass
    return str(out_bam)


def filter_bam(
    in_bam: Union[str, Path],
    out_bam: Union[str, Path],
    *,
    mapq: int = 30,
    proper_pair: bool = True,
    drop_secondary_supp: bool = True,
    drop_duplicates: bool = False,
    drop_qcfail: bool = False,
    drop_unmapped: bool = True,
    drop_mate_unmapped: bool = True,
    drop_chroms: Optional[Sequence[str]] = None,
    threads: int = 4,
) -> str:
    """
    Filter a BAM with common ATAC/CUT&RUN defaults and re-sort the result.
    """
    samtools = _resolve_tool("samtools")
    out_bam = str(out_bam)

    include_flag = 0x2 if proper_pair else 0
    exclude_flag = 0
    if drop_secondary_supp:
        exclude_flag |= 0x100 | 0x800
    if drop_duplicates:
        exclude_flag |= 0x400
    if drop_qcfail:
        exclude_flag |= 0x200
    if drop_unmapped:
        exclude_flag |= 0x4
    if drop_mate_unmapped:
        exclude_flag |= 0x8

    if not drop_chroms:
        cmd = [samtools, "view", "-b"]
        if include_flag:
            cmd += ["-f", hex(include_flag)]
        if exclude_flag:
            cmd += ["-F", hex(exclude_flag)]
        cmd += ["-q", str(mapq), str(in_bam)]
        sort_cmd = [samtools, "sort", "-@", str(threads), "-o", out_bam, "-"]
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.run(sort_cmd, stdin=p1.stdout)
        if p1.stdout is not None:
            p1.stdout.close()
        p1.wait()
        if p1.returncode != 0 or p2.returncode != 0:
            raise CommandError("BAM filtering pipeline failed.")
        return out_bam

    drop_chroms = set(drop_chroms)
    cmd = [samtools, "view", "-h"]
    if include_flag:
        cmd += ["-f", hex(include_flag)]
    if exclude_flag:
        cmd += ["-F", hex(exclude_flag)]
    cmd += ["-q", str(mapq), str(in_bam)]

    awk_script = (
        'BEGIN{OFS="\\t"} '
        '/^@/ {print; next} '
        "{if (!($3 in drop)) print}"
    )
    awk_cmd = ["awk", "-v", "drop=" + " ".join(sorted(drop_chroms))]
    awk_cmd.append(
        'BEGIN{n=split(drop, a, " "); for(i=1;i<=n;i++) bad[a[i]]=1} '
        '/^@/ {print; next} '
        '{if (!($3 in bad)) print}'
    )

    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(awk_cmd, stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.run([samtools, "view", "-b", "-"], stdin=p2.stdout, stdout=subprocess.PIPE)
    p4 = subprocess.run([samtools, "sort", "-@", str(threads), "-o", out_bam, "-"], input=p3.stdout)
    if p1.stdout is not None:
        p1.stdout.close()
    if p2.stdout is not None:
        p2.stdout.close()
    p1.wait()
    p2.wait()
    if any(p.returncode != 0 for p in (p1, p2, p3, p4)):
        raise CommandError("BAM filtering pipeline with chromosome exclusion failed.")
    return out_bam


def bam_to_frags(
    bam: Union[str, Path],
    sample_name: str,
    out_frags_gz: Union[str, Path],
    *,
    bedtools_path: Optional[str] = None,
) -> str:
    """BAM -> 4-col fragments (chr, start, end, barcode=sample_name)."""
    bedtools = _resolve_tool("bedtools", explicit=bedtools_path)
    samtools = _resolve_tool("samtools")
    use_bgzip = shutil.which("bgzip") is not None

    bam = str(bam)
    out_frags_gz = str(out_frags_gz)
    stem = str(Path(out_frags_gz).with_suffix("").with_suffix(""))
    tmp_name_bam = stem + ".namesort.bam"
    bedpe = stem + ".bedpe"

    _run([samtools, "sort", "-n", "-@", str(os.cpu_count() or 2), "-o", tmp_name_bam, bam])
    with open(bedpe, "w") as fh:
        proc = subprocess.run(
            [bedtools, "bamtobed", "-bedpe", "-i", tmp_name_bam],
            stdout=fh,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise CommandError(f"bedtools bamtobed failed.\nstderr:\n{proc.stderr}")

    awk_script = (
        f'BEGIN{{OFS="\\t"}} $1==$4 {{s=($2<$5?$2:$5); e=($3>$6?$3:$6); '
        f'if(e>s) print $1,s,e,"{sample_name}"}}'
    )
    compressor = ["bgzip", "-c"] if use_bgzip else [_resolve_tool("gzip"), "-c"]
    with open(out_frags_gz, "wb") as fout:
        pawk = subprocess.Popen(["awk", awk_script, bedpe], stdout=subprocess.PIPE)
        pgz = subprocess.Popen(compressor, stdin=pawk.stdout, stdout=fout)
        if pawk.stdout is not None:
            pawk.stdout.close()
        pgz.communicate()
        pawk.wait()
        if pawk.returncode != 0 or pgz.returncode != 0:
            raise CommandError("Fragment compression pipeline failed.")

    for f in (tmp_name_bam, bedpe):
        try:
            os.remove(f)
        except OSError:
            pass
    return out_frags_gz


def bulk_fastq_to_bam(
    fq1: Union[str, Path],
    fq2: Union[str, Path],
    sample_name: str,
    ref_index: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    aligner: str = "bwa-mem2",
    threads: int = 8,
    mapq: int = 30,
    trim: bool = False,
    trim_method: str = "fastp",
    trim_extra_args: Optional[Sequence[str]] = None,
    keep_trimmed_fastq: bool = False,
    remove_duplicates: bool = True,
    proper_pair: bool = True,
    drop_secondary_supp: bool = True,
    drop_duplicates: bool = False,
    drop_qcfail: bool = False,
    drop_unmapped: bool = True,
    drop_mate_unmapped: bool = True,
    drop_chroms: Optional[Sequence[str]] = None,
    rg: Optional[str] = None,
    extra_align_args: Optional[Sequence[str]] = None,
    keep_intermediates: bool = False,
    index: bool = True,
) -> str:
    """
    FASTQ -> aligned BAM -> filtered BAM.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fq1_use = str(fq1)
    fq2_use = str(fq2)
    if trim:
        trim_dir = out_dir / "trim"
        fq1_use, fq2_use = trim_fastq_pair(
            fq1_use,
            fq2_use,
            trim_dir,
            sample_name=sample_name,
            method=trim_method,
            threads=threads,
            extra_args=trim_extra_args,
        )

    raw_bam = str(out_dir / f"{sample_name}.bam")
    filt_bam = str(out_dir / f"{sample_name}.filtered.bam")

    align_fastq_to_bam(
        fq1_use,
        fq2_use,
        raw_bam,
        aligner=aligner,
        ref_index=str(ref_index),
        threads=threads,
        rg=rg,
        extra_args=extra_align_args,
        remove_duplicates=remove_duplicates,
    )
    filter_bam(
        raw_bam,
        filt_bam,
        mapq=mapq,
        proper_pair=proper_pair,
        drop_secondary_supp=drop_secondary_supp,
        drop_duplicates=drop_duplicates,
        drop_qcfail=drop_qcfail,
        drop_unmapped=drop_unmapped,
        drop_mate_unmapped=drop_mate_unmapped,
        drop_chroms=drop_chroms,
        threads=threads,
    )
    if index:
        index_bam(filt_bam, threads=threads)

    if not keep_intermediates:
        for f in (raw_bam,):
            try:
                os.remove(f)
            except OSError:
                pass
    if trim and not keep_trimmed_fastq:
        for f in (fq1_use, fq2_use):
            try:
                os.remove(f)
            except OSError:
                pass
    return filt_bam


def bulk_fastq_to_frag(
    fq1: Union[str, Path],
    fq2: Union[str, Path],
    sample_name: str,
    ref_index: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    aligner: str = "bwa-mem2",
    threads: int = 8,
    mapq: int = 30,
    trim: bool = False,
    trim_method: str = "fastp",
    trim_extra_args: Optional[Sequence[str]] = None,
    keep_trimmed_fastq: bool = False,
    keep_intermediates: bool = False,
    rg: Optional[str] = None,
    extra_align_args: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    """
    FASTQ -> filtered BAM -> fragments.tsv.gz
    """
    out_dir = Path(out_dir)
    filt_bam = bulk_fastq_to_bam(
        fq1,
        fq2,
        sample_name,
        ref_index,
        out_dir,
        aligner=aligner,
        threads=threads,
        mapq=mapq,
        trim=trim,
        trim_method=trim_method,
        trim_extra_args=trim_extra_args,
        keep_trimmed_fastq=keep_trimmed_fastq,
        remove_duplicates=True,
        keep_intermediates=keep_intermediates,
        rg=rg,
        extra_align_args=extra_align_args,
    )
    frags_gz = str(Path(out_dir) / f"{sample_name}.frags.tsv.gz")
    bam_to_frags(filt_bam, sample_name, frags_gz)
    return filt_bam, frags_gz


def shift_atac_bam(
    in_bam: Union[str, Path],
    out_bam: Union[str, Path],
    *,
    threads: int = 8,
    index: bool = True,
) -> str:
    """
    Apply the standard ATAC Tn5 shift (+4 on forward reads, -5 on reverse reads).
    """
    out_bam = Path(out_bam)
    out_bam.parent.mkdir(parents=True, exist_ok=True)

    # Process pairs in query-name order so mate coordinates and template
    # lengths stay consistent after shifting.
    import pysam

    name_sorted = out_bam.with_suffix(".qname.tmp.bam")
    tmp_bam = out_bam.with_suffix(".shift.unsorted.bam")
    pysam.sort("-n", "-@", str(threads), "-o", str(name_sorted), str(in_bam))

    def _shift_one(read):
        if read.is_unmapped:
            return read
        shifted = read.__copy__()
        offset = 4 if not shifted.is_reverse else -5
        shifted.reference_start = max(0, shifted.reference_start + offset)
        return shifted

    def _sync_pair(read1, read2):
        if (
            read1.is_unmapped or read2.is_unmapped
            or read1.reference_id != read2.reference_id
        ):
            return
        read1.next_reference_id = read2.reference_id
        read2.next_reference_id = read1.reference_id
        read1.next_reference_start = read2.reference_start
        read2.next_reference_start = read1.reference_start

        left, right = (read1, read2) if read1.reference_start <= read2.reference_start else (read2, read1)
        left_end = left.reference_end or left.reference_start
        right_end = right.reference_end or right.reference_start
        template_len = max(left_end, right_end) - min(left.reference_start, right.reference_start)
        if template_len < 0:
            template_len = 0

        if left is read1:
            read1.template_length = template_len
            read2.template_length = -template_len
        else:
            read2.template_length = template_len
            read1.template_length = -template_len

    with pysam.AlignmentFile(str(name_sorted), "rb") as src, pysam.AlignmentFile(str(tmp_bam), "wb", template=src) as dst:
        pending = {}
        for read in src.fetch(until_eof=True):
            shifted = _shift_one(read)
            qname = shifted.query_name
            mate = pending.pop(qname, None)
            if mate is None:
                pending[qname] = shifted
                continue
            _sync_pair(mate, shifted)
            dst.write(mate)
            dst.write(shifted)

        for leftover in pending.values():
            dst.write(leftover)

    pysam.sort("-@", str(threads), "-o", str(out_bam), str(tmp_bam))
    try:
        os.remove(tmp_bam)
    except OSError:
        pass
    try:
        os.remove(name_sorted)
    except OSError:
        pass

    if index:
        index_bam(out_bam, threads=threads)
    return str(out_bam)


def bam_to_bigwig(
    bam: Union[str, Path],
    out_bw: Union[str, Path],
    *,
    bin_size: int = 10,
    normalize_using: Optional[str] = None,
    effective_genome_size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    extend_reads: Optional[int] = None,
    center_reads: bool = False,
    min_mapping_quality: Optional[int] = None,
    ignore_for_normalization: Optional[Sequence[str]] = None,
    threads: int = 8,
) -> str:
    out_bw = Path(out_bw)
    out_bw.parent.mkdir(parents=True, exist_ok=True)

    # Write a fixed-bin bigWig from per-base alignment coverage.
    import numpy as np
    import pyBigWig
    import pysam

    ignore_for_normalization = set(ignore_for_normalization or [])
    mapped_reads = 0
    chrom_sizes: Dict[str, int] = {}
    with pysam.AlignmentFile(str(bam), "rb") as bf:
        for sq in bf.header.get("SQ", []):
            chrom_sizes[sq["SN"]] = int(sq["LN"])
        for read in bf.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.is_qcfail or read.is_duplicate:
                continue
            if min_mapping_quality is not None and read.mapping_quality < min_mapping_quality:
                continue
            if read.reference_name in ignore_for_normalization:
                continue
            mapped_reads += 1

    if mapped_reads == 0:
        raise CommandError("No mapped reads remained after filtering; cannot write bigWig.")

    bw = pyBigWig.open(str(out_bw), "w")
    bw.addHeader(list(chrom_sizes.items()))

    with pysam.AlignmentFile(str(bam), "rb") as bf:
        for chrom, length in chrom_sizes.items():
            cov = np.zeros(length, dtype=np.float32)
            for read in bf.fetch(chrom):
                if read.is_unmapped:
                    continue
                if read.is_secondary or read.is_supplementary:
                    continue
                if read.is_qcfail or read.is_duplicate:
                    continue
                if min_mapping_quality is not None and read.mapping_quality < min_mapping_quality:
                    continue

                blocks = read.get_blocks()
                if not blocks:
                    continue

                if center_reads:
                    mid = (blocks[0][0] + blocks[-1][1]) // 2
                    s = max(0, mid)
                    e = min(length, mid + 1)
                    cov[s:e] += 1.0
                    continue

                if extend_reads is not None and len(blocks) == 1:
                    s, e = blocks[0]
                    if read.is_reverse:
                        s = max(0, e - int(extend_reads))
                    else:
                        e = min(length, s + int(extend_reads))
                    cov[s:e] += 1.0
                    continue

                for s, e in blocks:
                    s = max(0, s)
                    e = min(length, e)
                    if e > s:
                        cov[s:e] += 1.0

            starts = []
            ends = []
            values = []
            for s in range(0, length, int(bin_size)):
                e = min(length, s + int(bin_size))
                val = float(cov[s:e].mean())
                if normalize_using:
                    norm = str(normalize_using).upper()
                    if norm == "CPM":
                        val = val * 1e6 / mapped_reads
                    elif norm == "RPKM":
                        val = val * 1e9 / (mapped_reads * max(1, (e - s)))
                    elif norm == "BPM":
                        val = val * 1e6 / mapped_reads
                if scale_factor is not None:
                    val *= float(scale_factor)
                starts.append(s)
                ends.append(e)
                values.append(val)
            bw.addEntries([chrom] * len(starts), starts, ends=ends, values=values)

    bw.close()
    return str(out_bw)


def call_peaks_macs2(
    bam: Union[str, Path],
    out_dir: Union[str, Path],
    name: str,
    *,
    control_bam: Optional[Union[str, Path]] = None,
    genome_size: str = "hs",
    format: str = "BAMPE",
    qvalue: float = 0.01,
    keep_dup: str = "all",
    call_summits: bool = True,
    nomodel: bool = True,
    shift: Optional[int] = None,
    extsize: Optional[int] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    macs2 = _resolve_tool("macs2")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        macs2,
        "callpeak",
        "-t",
        str(bam),
        "-f",
        format,
        "-g",
        genome_size,
        "--keep-dup",
        keep_dup,
        "-q",
        str(qvalue),
        "-n",
        name,
        "--outdir",
        str(out_dir),
    ]
    if control_bam is not None:
        cmd += ["-c", str(control_bam)]
    if nomodel:
        cmd.append("--nomodel")
    if call_summits:
        cmd.append("--call-summits")
    if shift is not None:
        cmd += ["--shift", str(shift)]
    if extsize is not None:
        cmd += ["--extsize", str(extsize)]
    if extra_args:
        cmd += list(map(str, extra_args))
    _run(cmd)

    prefix = out_dir / name
    return {
        "narrowPeak": str(prefix.with_name(prefix.name + "_peaks.narrowPeak")),
        "summits": str(prefix.with_name(prefix.name + "_summits.bed")),
        "xls": str(prefix.with_name(prefix.name + "_peaks.xls")),
    }


def _epione_datasets():
    try:
        from epione.utils._genome import register_datasets
    except Exception as e:
        raise RuntimeError("epione.utils._genome.register_datasets not available") from e
    return register_datasets()


def fetch_genome_fasta(genome: str) -> Path:
    """
    Download reference FASTA for a genome key using epione datasets.
    genome: one of {'GRCh37','GRCh38','GRCm38','GRCm39'}
    Returns uncompressed FASTA Path.
    """
    ds = _epione_datasets()
    key_map = {
        "GRCh37": "gencode_v41_GRCh37.fa.gz",
        "GRCh38": "gencode_v41_GRCh38.fa.gz",
        "GRCm38": "gencode_vM25_GRCm38.fa.gz",
        "GRCm39": "gencode_vM30_GRCm39.fa.gz",
    }
    if genome not in key_map:
        raise ValueError(f"Unsupported genome: {genome}")
    fa_gz = Path(ds.fetch(key_map[genome]))
    return ensure_fasta_unzipped(fa_gz)


def list_dataset_fastqs_tar() -> List[str]:
    """List FASTQ members inside the demo tar without extracting."""
    ds = _epione_datasets()
    tar_path = Path(ds.fetch("atac_pbmc_500_fastqs.tar"))
    with tarfile.open(tar_path, "r") as tf:
        return [
            m.name
            for m in tf.getmembers()
            if m.isfile() and m.name.endswith((".fastq.gz", ".fq.gz"))
        ]


def extract_fastqs_from_tar(members: Sequence[str], dest: Union[str, Path]) -> List[Path]:
    """Extract only selected FASTQ members from the demo tar to dest."""
    ds = _epione_datasets()
    tar_path = Path(ds.fetch("atac_pbmc_500_fastqs.tar"))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    out_paths: List[Path] = []
    with tarfile.open(tar_path, "r") as tf:
        member_map = {m.name: m for m in tf.getmembers()}
        for name in members:
            if name not in member_map:
                raise FileNotFoundError(f"{name} not found in atac_pbmc_500_fastqs.tar")
            member = member_map[name]
            extracted = tf.extractfile(member)
            if extracted is None:
                raise CommandError(f"Could not extract {name} from demo tar.")
            out = dest / Path(name).name
            with extracted, open(out, "wb") as fout:
                shutil.copyfileobj(extracted, fout)
            out_paths.append(out)
    return out_paths


def fetch_dataset_fastq_pairs(
    out_dir: Union[str, Path],
    *,
    limit_pairs: int = 1,
) -> List[Tuple[Path, Path]]:
    """
    Download small demo FASTQs (10x PBMC 500) via epione datasets.
    """
    out_dir = Path(out_dir)
    fq_dir = out_dir / "fastqs"
    fq_dir.mkdir(parents=True, exist_ok=True)

    members = list_dataset_fastqs_tar()
    r1_names = sorted([m for m in members if "R1" in Path(m).name and m.endswith((".fastq.gz", ".fq.gz"))])
    r2_names = sorted([m for m in members if "R2" in Path(m).name and m.endswith((".fastq.gz", ".fq.gz"))])

    def _key(name: str) -> str:
        return Path(name).name.replace("R1", "").replace("R2", "")

    pairs_names: List[Tuple[str, str]] = []
    used_r2 = set()
    for r1 in r1_names:
        key = _key(r1)
        match = next((r2 for r2 in r2_names if _key(r2) == key and r2 not in used_r2), None)
        if match is not None:
            pairs_names.append((r1, match))
            used_r2.add(match)
    if not pairs_names:
        raise RuntimeError("No FASTQ pairs detected in demo tar.")

    pairs_names = pairs_names[:limit_pairs]
    flat_members = [m for pair in pairs_names for m in pair]
    extracted = extract_fastqs_from_tar(flat_members, fq_dir)
    extracted_map = {p.name: p for p in extracted}
    return [(extracted_map[Path(r1).name], extracted_map[Path(r2).name]) for r1, r2 in pairs_names]


def bulk_from_genome_demo(
    genome: str,
    out_dir: Union[str, Path],
    *,
    aligner: str = "bwa-mem2",
    threads: int = 8,
    mapq: int = 30,
    keep_intermediates: bool = False,
) -> Tuple[Path, Path]:
    """
    Download demo FASTQs and genome FASTA via epione datasets, then make BAM+frags.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fa = fetch_genome_fasta(genome)
    ref_index = ensure_aligner_index(aligner, fa)
    pairs = fetch_dataset_fastq_pairs(out_dir, limit_pairs=1)
    fq1, fq2 = pairs[0]
    sample_name = fq1.name.split("_R1")[0].split("_1")[0]
    bam, frags = bulk_fastq_to_frag(
        str(fq1),
        str(fq2),
        sample_name,
        ref_index,
        str(out_dir),
        aligner=aligner,
        threads=threads,
        mapq=mapq,
        keep_intermediates=keep_intermediates,
    )
    return Path(bam), Path(frags)


__all__ = [
    "CommandError",
    "ensure_fasta_unzipped",
    "ensure_fasta_index",
    "ensure_chrom_sizes",
    "ensure_aligner_index",
    "prepare_reference",
    "trim_fastq_pair",
    "sort_bam",
    "index_bam",
    "merge_bams",
    "align_fastq_to_bam",
    "filter_bam",
    "bam_to_frags",
    "bulk_fastq_to_bam",
    "bulk_fastq_to_frag",
    "shift_atac_bam",
    "bam_to_bigwig",
    "call_peaks_macs2",
    "fetch_genome_fasta",
    "list_dataset_fastqs_tar",
    "extract_fastqs_from_tar",
    "fetch_dataset_fastq_pairs",
    "bulk_from_genome_demo",
]
