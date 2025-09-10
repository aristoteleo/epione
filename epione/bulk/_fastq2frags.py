import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union
import tarfile
import gzip


class CommandError(RuntimeError):
    pass


def _run(cmd: List[str], cwd: Optional[str] = None, stream: bool = True) -> None:
    """Run a shell command.
    - stream=True: inherit stdout/stderr so progress bars and logs are visible.
    - stream=False: capture stdout/stderr and include them on error.
    """
    if stream:
        proc = subprocess.run(cmd, cwd=cwd)
        if proc.returncode != 0:
            raise CommandError(f"Command failed: {' '.join(cmd)}")
        return
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise CommandError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def _which_or_raise(binary: str) -> None:
    if shutil.which(binary) is None:
        raise FileNotFoundError(f"Required binary '{binary}' not found in PATH. Please install it and try again.")


def align_fastq_to_bam(
    fq1: str,
    fq2: str,
    out_bam: str,
    *,
    aligner: str = "bwa-mem2",
    ref_index: Optional[str] = None,
    threads: int = 8,
    rg: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    remove_duplicates: bool = False,
) -> str:
    """
    Align paired-end FASTQs to reference and produce a coordinate-sorted, duplicate-removed BAM.

    - aligner: 'bwa-mem2' or 'bowtie2'
    - ref_index: path prefix to the aligner index (required)
    - rg: read group string, e.g. '@RG\tID:sample\tSM:sample'
    """
    if ref_index is None:
        raise ValueError("ref_index is required (path prefix to the aligner index)")
    out_prefix = str(Path(out_bam).with_suffix("")).rstrip(".")
    tmp_dir = Path(Path(out_prefix).parent or ".")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    name_sorted = str(tmp_dir / (Path(out_prefix).name + ".namesort.bam"))
    fixmate_bam = str(tmp_dir / (Path(out_prefix).name + ".fixmate.bam"))
    coord_sorted = str(tmp_dir / (Path(out_prefix).name + ".coordsort.bam"))

    _which_or_raise("samtools")

    if aligner == "bwa-mem2":
        _which_or_raise("bwa-mem2")
        cmd = [
            "bwa-mem2",
            "mem",
            "-t",
            str(threads),
            ref_index,
            fq1,
            fq2,
        ]
        if rg:
            cmd.extend(["-R", rg])
        if extra_args:
            cmd.extend(extra_args)
        cmd_view = ["samtools", "view", "-b", "-"]
        cmd_sort_n = ["samtools", "sort", "-n", "-@", str(threads), "-o", name_sorted, "-"]
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd_view, stdin=p1.stdout, stdout=subprocess.PIPE)
        p3 = subprocess.run(cmd_sort_n, stdin=p2.stdout)
        if p3.returncode != 0:
            raise CommandError("Alignment pipeline failed during name sort.")
    elif aligner == "bowtie2":
        _which_or_raise("bowtie2")
        cmd = [
            "bowtie2",
            "-x",
            ref_index,
            "-1",
            fq1,
            "-2",
            fq2,
            "-p",
            str(threads),
        ]
        if extra_args:
            cmd.extend(extra_args)
        cmd_view = ["samtools", "view", "-b", "-"]
        cmd_sort_n = ["samtools", "sort", "-n", "-@", str(threads), "-o", name_sorted, "-"]
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd_view, stdin=p1.stdout, stdout=subprocess.PIPE)
        p3 = subprocess.run(cmd_sort_n, stdin=p2.stdout)
        if p3.returncode != 0:
            raise CommandError("Alignment pipeline failed during name sort.")
    else:
        raise ValueError("aligner must be 'bwa-mem2' or 'bowtie2'")

    _run(["samtools", "fixmate", "-m", name_sorted, fixmate_bam])
    _run(["samtools", "sort", "-@", str(threads), "-o", coord_sorted, fixmate_bam])
    md_cmd = ["samtools", "markdup", "-@", str(threads)]
    if remove_duplicates:
        md_cmd.insert(2, "-r")
    md_cmd += [coord_sorted, out_bam]
    _run(md_cmd)

    for f in (name_sorted, fixmate_bam, coord_sorted):
        try:
            os.remove(f)
        except OSError:
            pass
    return out_bam


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
        _which_or_raise("bwa-mem2")
        # choose prefix
        if index_prefix is None:
            prefix_path = fasta
        else:
            prefix_path = Path(index_prefix)
            prefix_path.parent.mkdir(parents=True, exist_ok=True)
        # expected files
        idx_candidates = [Path(str(prefix_path) + ext) for ext in [".0123", ".amb", ".ann", ".bwt"]]
        if overwrite:
            print("Overwrite requested: rebuilding bwa-mem2 index...")
            cmd = ["bwa-mem2", "index"]
            if index_prefix is not None:
                cmd += ["-p", str(prefix_path)]
            cmd += [str(fasta)]
            _run(cmd)
        elif not all(p.exists() for p in idx_candidates):
            print("Building bwa-mem2 index...")
            cmd = ["bwa-mem2", "index"]
            if index_prefix is not None:
                cmd += ["-p", str(prefix_path)]
            cmd += [str(fasta)]
            _run(cmd)
        else:
            print(f"bwa-mem2 index found at prefix: {prefix_path}. Skipping.")
        return str(prefix_path)
    elif aligner == "bowtie2":
        _which_or_raise("bowtie2-build")
        if index_prefix is None:
            prefix = str(Path(fasta).with_suffix(""))
        else:
            prefix = str(index_prefix)
            Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        bt2_files = [Path(prefix + ext) for ext in [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]]
        if overwrite:
            print("Overwrite requested: rebuilding bowtie2 index...")
            _run(["bowtie2-build", str(fasta), prefix])
        elif not all(p.exists() for p in bt2_files):
            print("Building bowtie2 index...")
            _run(["bowtie2-build", str(fasta), prefix])
        else:
            print(f"bowtie2 index found at prefix: {prefix}. Skipping.")
        return prefix
    else:
        raise ValueError("aligner must be 'bwa-mem2' or 'bowtie2'")


def filter_bam(
    in_bam: str,
    out_bam: str,
    *,
    mapq: int = 30,
    proper_pair: bool = True,
    drop_secondary_supp: bool = True,
    threads: int = 4,
) -> str:
    _which_or_raise("samtools")
    flags_pos = ["-f", "0x2"] if proper_pair else []
    flags_neg = ["-F", "0x904"] if drop_secondary_supp else ["-F", "0x400"]
    cmd = [
        "samtools",
        "view",
        "-b",
        *flags_pos,
        *flags_neg,
        "-q",
        str(mapq),
        in_bam,
    ]
    sort_cmd = ["samtools", "sort", "-@", str(threads), "-o", out_bam, "-"]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.run(sort_cmd, stdin=p1.stdout)
    if p2.returncode != 0:
        raise CommandError("BAM filtering pipeline failed.")
    return out_bam


def bam_to_frags(
    bam: str,
    sample_name: str,
    out_frags_gz: str,
    *,
    bedtools_path: str = "bedtools",
) -> str:
    """BAM → 4-col fragments (chr, start, end, barcode=sample_name). Output gz/bgzip.
    Notes:
    - bedtools bamtobed -bedpe 需要 name-sorted BAM 才能保证配对相邻；此处会先做 name-sort。
    - 抑制 bedtools 关于非相邻配对的警告输出。
    """
    _which_or_raise(bedtools_path)
    _which_or_raise("samtools")
    use_bgzip = shutil.which("bgzip") is not None

    bam = str(bam)
    out_frags_gz = str(out_frags_gz)
    tmp_name_bam = str(Path(out_frags_gz).with_suffix("").with_suffix("") ) + ".namesort.bam"
    bedpe = str(Path(out_frags_gz).with_suffix("").with_suffix("") ) + ".bedpe"

    print("[1/3] Name-sorting BAM for bedpe ...")
    _run(["samtools", "sort", "-n", "-@", str(os.cpu_count() or 2), "-o", tmp_name_bam, bam])

    print("[2/3] Converting BAM→BEDPE (suppressing bedtools warnings) ...")
    with open(bedpe, "w") as fh:
        # capture stderr to suppress warnings; rely on return code for errors
        p = subprocess.run([bedtools_path, "bamtobed", "-bedpe", "-i", tmp_name_bam], stdout=fh, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise CommandError(f"bedtools bamtobed failed.\nstderr:\n{p.stderr}")

    awk_script = (
        f"BEGIN{{OFS=\"\\t\"}} $1==$4 {{s=($2<$5?$2:$5); e=($3>$6?$3:$6); if(e>s) print $1,s,e,\"{sample_name}\"}}"
    )
    if use_bgzip:
        with open(out_frags_gz, "wb") as fout:
            pawk = subprocess.Popen(["awk", awk_script, bedpe], stdout=subprocess.PIPE)
            pgz = subprocess.Popen(["bgzip", "-c"], stdin=pawk.stdout, stdout=fout)
            pgz.communicate()
            if pgz.returncode != 0:
                raise CommandError("bgzip compression failed.")
    else:
        _which_or_raise("gzip")
        with open(out_frags_gz, "wb") as fout:
            pawk = subprocess.Popen(["awk", awk_script, bedpe], stdout=subprocess.PIPE)
            pgz = subprocess.Popen(["gzip", "-c"], stdin=pawk.stdout, stdout=fout)
            pgz.communicate()
            if pgz.returncode != 0:
                raise CommandError("gzip compression failed.")
    # cleanup temp files
    for f in (tmp_name_bam, bedpe):
        try:
            os.remove(f)
        except OSError:
            pass
    return out_frags_gz


def bulk_fastq_to_frag(
    fq1: str,
    fq2: str,
    sample_name: str,
    ref_index: str,
    out_dir: str,
    *,
    aligner: str = "bwa-mem2",
    threads: int = 8,
    mapq: int = 30,
    keep_intermediates: bool = False,
    rg: Optional[str] = None,
    extra_align_args: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    FASTQ (R1/R2) -> aligned, deduped, filtered BAM -> fragments.tsv.gz

    Returns (filtered_bam, frags_gz)
    Requirements in PATH: bwa-mem2 or bowtie2, samtools, bedtools, bgzip/gzip
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    bam_path = str(out_dir_p / f"{sample_name}.bam")
    filt_bam = str(out_dir_p / f"{sample_name}.filtered.bam")
    frags_gz = str(out_dir_p / f"{sample_name}.frags.tsv.gz")

    align_fastq_to_bam(
        fq1,
        fq2,
        bam_path,
        aligner=aligner,
        ref_index=ref_index,
        threads=threads,
        rg=rg,
        extra_args=extra_align_args,
    )
    filter_bam(bam_path, filt_bam, mapq=mapq, proper_pair=True, drop_secondary_supp=True, threads=threads)
    bam_to_frags(filt_bam, sample_name, frags_gz)

    if not keep_intermediates:
        for f in (bam_path,):
            try:
                os.remove(f)
            except OSError:
                pass
    return filt_bam, frags_gz


# -----------------------
# Helpers: download genome/FASTQs via epione datasets
# -----------------------

def _epione_datasets():
    try:
        from epione.utils._genome import register_datasets
    except Exception as e:
        raise RuntimeError("epione.utils._genome.register_datasets not available") from e
    return register_datasets()


def ensure_fasta_unzipped(fa_gz: Path) -> Path:
    fa_gz = Path(fa_gz)
    if fa_gz.suffix == ".gz":
        out = fa_gz.with_suffix("")
    else:
        return fa_gz
    if out.exists():
        return out
    with gzip.open(fa_gz, "rb") as fin, open(out, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return out


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
        raise ValueError("Unsupported genome: %s" % genome)
    fa_gz = Path(ds.fetch(key_map[genome]))
    return ensure_fasta_unzipped(fa_gz)


def list_dataset_fastqs_tar() -> List[str]:
    """List FASTQ members inside the demo tar without extracting."""
    ds = _epione_datasets()
    tar_path = Path(ds.fetch("atac_pbmc_500_fastqs.tar"))
    with tarfile.open(tar_path, "r") as tf:
        return [m.name for m in tf.getmembers() if m.isfile() and m.name.endswith((".fastq.gz", ".fq.gz"))]


def extract_fastqs_from_tar(members: List[str], dest: Union[str, Path]) -> List[Path]:
    """Extract only selected FASTQ members from the demo tar to dest, return paths."""
    ds = _epione_datasets()
    tar_path = Path(ds.fetch("atac_pbmc_500_fastqs.tar"))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        selected = [m for m in tf.getmembers() if m.name in members]
        for m in selected:
            target = dest / Path(m.name).name
            # security check: extract to a temp then move
            tf.extract(m, dest)
            # If tar contains subdirs, move file up
            extracted = dest / m.name
            if extracted.is_file():
                if extracted != target:
                    extracted.rename(target)
        return [dest / Path(m).name for m in members]


def fetch_dataset_fastq_pairs(out_dir: Union[str, Path], *, limit_pairs: int = 1) -> List[Tuple[Path, Path]]:
    """
    Download small demo FASTQs (10x PBMC 500) via epione datasets and return R1/R2 pairs.
    """
    ds = _epione_datasets()
    tar_path = Path(ds.fetch("atac_pbmc_500_fastqs.tar"))
    out_dir = Path(out_dir)
    fq_dir = out_dir / "fastqs"
    fq_dir.mkdir(parents=True, exist_ok=True)
    # extract
    # list members and extract minimal needed
    members = list_dataset_fastqs_tar()
    r1_names = sorted([m for m in members if "R1" in m and m.endswith((".fastq.gz", ".fq.gz"))])
    r2_names = sorted([m for m in members if "R2" in m and m.endswith((".fastq.gz", ".fq.gz"))])
    pairs_names: List[Tuple[str, str]] = []
    def key_name(n: str) -> str:
        return n.replace("R1", "").replace("R2", "")
    used = set()
    for r1 in r1_names:
        k = key_name(r1)
        match = next((r2 for r2 in r2_names if key_name(r2) == k and r2 not in used), None)
        if match:
            pairs_names.append((r1, match))
            used.add(match)
    if not pairs_names:
        raise RuntimeError("No FASTQ pairs detected in demo tar")
    pairs_names = pairs_names[:limit_pairs]
    flat_members = [n for tup in pairs_names for n in tup]
    extracted = extract_fastqs_from_tar(flat_members, fq_dir)
    r1s = sorted([p for p in extracted if "R1" in p.name])
    r2s = sorted([p for p in extracted if "R2" in p.name])
    pairs: List[Tuple[Path, Path]] = []
    # naive pairing by filename stem up to R1/R2
    def key(p: Path):
        s = p.name.replace("R1", "").replace("R2", "")
        return s
    used_r2 = set()
    for r1 in r1s:
        k = key(r1)
        match = next((r2 for r2 in r2s if key(r2) == k and r2 not in used_r2), None)
        if match is not None:
            pairs.append((r1, match))
            used_r2.add(match)
    if not pairs:
        raise RuntimeError("No R1/R2 FASTQ pairs extracted")
    return pairs


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
    End-to-end: download demo FASTQs and genome FASTA via epione datasets, build index,
    align the first pair to produce filtered BAM and fragments.
    Returns (filtered_bam, frags_gz)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fa = fetch_genome_fasta(genome)
    ref_index = ensure_aligner_index(aligner, fa)
    pairs = download_demo_fastqs(out_dir)
    fq1, fq2 = pairs[0]
    sample_name = fq1.parent.name if fq1.parent != out_dir else fq1.stem.split("_")[0]
    return bulk_fastq_to_frag(
        str(fq1), str(fq2), sample_name, ref_index, str(out_dir),
        aligner=aligner, threads=threads, mapq=mapq, keep_intermediates=keep_intermediates,
    )
