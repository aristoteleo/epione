"""
Utilities for splitting BAM files into cluster-specific BAMs.

This module ports the `split_bam_clusters` implementation from
SC-Framework's `sctoolbox.tools.bam` with minimal adjustments so it can
be used inside epione without the original dependency tree.
"""

from __future__ import annotations

import logging
import os
import re
import multiprocessing as mp
from functools import partial
from multiprocessing.managers import BaseProxy
from multiprocessing.pool import ApplyResult
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

try:
    import scanpy as sc  # type: ignore
except ImportError as exc:  # pragma: no cover - handled by calling code
    raise ImportError(
        "scanpy is required for BAM splitting. Please install scanpy in your environment."
    ) from exc

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# -------------------- lightweight helper utils -------------------- #
# ------------------------------------------------------------------ #

def _check_module(name: str) -> None:
    """Ensure a module can be imported."""
    __import__(name)


def _is_notebook() -> bool:
    """Best-effort detection of a Jupyter environment."""
    try:
        from IPython import get_ipython  # type: ignore
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return "IPKernelApp" in shell.config


def _create_dir(path: str | None) -> None:
    """Create a directory (including parents) if it does not exist."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _split_list(lst: Sequence[Any], n: int) -> List[Sequence[Any]]:
    """Split list into n evenly interleaved chunks."""
    if n <= 0:
        return [lst]
    chunks: List[Sequence[Any]] = []
    for i in range(n):
        chunks.append(lst[i::n])
    return [chunk for chunk in chunks if len(chunk) > 0]


def _check_type(value: Any, name: str, expected: Any) -> None:
    """Runtime type assertion similar to sctoolbox.utils.checker.check_type."""
    if expected is None:
        return
    if isinstance(expected, tuple):
        valid = any(isinstance(value, exp) for exp in expected)
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise TypeError(f"Parameter '{name}' expected type {expected}, got {type(value)}.")


# ------------------------------------------------------------------ #
# ---------------------- bam helper functions ---------------------- #
# ------------------------------------------------------------------ #

def open_bam(
    file: str,
    mode: str,
    verbosity: Literal[0, 1, 2, 3] = 3,
    **kwargs: Any,
) -> "pysam.AlignmentFile":
    """
    Open BAM file with pysam.AlignmentFile while controlling verbosity.
    """

    _check_module("pysam")
    import pysam

    former_verbosity = pysam.get_verbosity()
    pysam.set_verbosity(verbosity)
    handle = pysam.AlignmentFile(file, mode, **kwargs)
    pysam.set_verbosity(former_verbosity)
    return handle


def get_bam_reads(bam_obj: "pysam.AlignmentFile") -> int:
    """
    Get the number of reads from an open pysam.AlignmentFile.
    """

    _check_module("pysam")
    import pysam

    try:
        total = bam_obj.mapped + bam_obj.unmapped
    except ValueError:
        path = bam_obj.filename  # type: ignore[attr-defined]
        total = int(pysam.view("-c", path))
    return total


# ------------------------------------------------------------------ #
# -------------------- multiprocessing functions ------------------- #
# ------------------------------------------------------------------ #

def _monitor_progress(
    progress_queue: Any,
    cluster_queues: Dict[str, Any],
    reader_jobs: List[ApplyResult],
    writer_jobs: List[ApplyResult],
    total_reads: Dict[str, int],
    individual_pbars: bool = False,
) -> int:
    """Monitor read/write progress of split_bam_clusters workers."""

    for value in cluster_queues.values():
        _check_type(value, "cluster_queues value", BaseProxy)
    for jobs in reader_jobs:
        _check_type(jobs, "reader_jobs", ApplyResult)
    for jobs in writer_jobs:
        _check_type(jobs, "writer_jobs", ApplyResult)

    if _is_notebook():
        from tqdm import tqdm_notebook as tqdm  # type: ignore
    else:
        from tqdm import tqdm  # type: ignore

    cluster_names = list(cluster_queues.keys())
    print(" ", end="", flush=True)  # tqdm notebook quirk

    pbars: Dict[str, Any] = {}
    if individual_pbars:
        for i, name in enumerate(total_reads):
            pbars[name] = tqdm(total=total_reads[name], position=i, desc=f"Reading ({name})", unit="reads")
        offset = len(total_reads)
        for j, cluster in enumerate(cluster_names):
            bar = tqdm(total=1, position=offset + j, desc=f"Writing queued reads ({cluster})")
            bar.total = 0
            bar.refresh()
            pbars[cluster] = bar
    else:
        sum_reads = sum(total_reads.values())
        read_bar = tqdm(total=sum_reads, position=0, desc="Reading from bams", unit="reads")
        for name in total_reads:
            pbars[name] = read_bar
        write_bar = tqdm(total=1, position=1, desc="Writing queued reads", unit="reads")
        write_bar.total = 0
        write_bar.refresh()
        for cluster in cluster_names:
            pbars[cluster] = write_bar

    writers_running = len(writer_jobs)
    reading_done = False
    while True:
        task, name, value = progress_queue.get()
        pbar = pbars[name]

        if task == "read":
            pbar.update(value)
            pbar.refresh()
        elif task == "sent":
            pbar.total += value
            pbar.refresh()
        elif task == "written":
            pbar.update(value)
            pbar.refresh()
        elif task == "done":
            writers_running -= 1

        reader_pbars = [pbars[n] for n in total_reads]
        if not reading_done and (
            sum(p.total for p in reader_pbars) >= sum(p.n for p in reader_pbars)
        ):
            reading_done = True
            _ = [job.get() for job in reader_jobs]
            for queue in cluster_queues.values():
                queue.put((None, None))

        if writers_running == 0:
            _ = [job.get() for job in writer_jobs]
            break

    return 0


def _buffered_reader(
    path: str,
    out_queues: Dict[str, Any],
    bc2cluster: Dict[str, str],
    tag: str,
    progress_queue: Any,
    buffer_size: int = 10000,
) -> int:
    """Read BAM and forward reads to the appropriate cluster queue."""

    for value in out_queues.values():
        _check_type(value, "out_queues value", BaseProxy)
    _check_type(progress_queue, "progress_queue", BaseProxy)

    try:
        bam = open_bam(path, "rb", verbosity=0)
        read_buffer: Dict[str, List[str]] = {cluster: [] for cluster in set(bc2cluster.values())}
        step = 100000
        n_reads_step = 0

        for read in bam:
            bc = read.get_tag(tag) if read.has_tag(tag) else None

            if bc in bc2cluster:
                cluster = bc2cluster[bc]  # type: ignore[index]
                read_buffer[cluster].append(read.to_string())
                if len(read_buffer[cluster]) == buffer_size:
                    out_queues[cluster].put((cluster, read_buffer[cluster]))
                    progress_queue.put(("sent", cluster, len(read_buffer[cluster])))
                    read_buffer[cluster] = []

            n_reads_step += 1
            if n_reads_step == step:
                progress_queue.put(("read", path, n_reads_step))
                n_reads_step = 0

        progress_queue.put(("read", path, n_reads_step))

        for cluster, buffered_reads in read_buffer.items():
            if buffered_reads:
                out_queues[cluster].put((cluster, buffered_reads))
                progress_queue.put(("sent", cluster, len(buffered_reads)))

        return 0

    except Exception as exc:  # pragma: no cover - passthrough
        logger.error("Buffered reader failed for %s: %s", path, exc)
        raise


def _writer(
    read_queue: Any,
    out_paths: Dict[str, str],
    bam_header: str,
    progress_queue: Any,
    pysam_threads: int = 4,
) -> int:
    """Write reads pulled from queue to the corresponding BAM file."""

    _check_type(read_queue, "read_queue", BaseProxy)
    _check_type(progress_queue, "progress_queue", BaseProxy)

    try:
        import pysam

        handles: Dict[str, "pysam.AlignmentFile"] = {}
        for cluster, path in out_paths.items():
            handles[cluster] = pysam.AlignmentFile(path, "wb", text=bam_header, threads=pysam_threads)

        while True:
            cluster, read_list = read_queue.get()
            if cluster is None:
                break

            handle = handles[cluster]
            for read in read_list:
                record = pysam.AlignedSegment.fromstring(read, handle.header)
                handle.write(record)

            progress_queue.put(("written", cluster, len(read_list)))

        for handle in handles.values():
            handle.close()

        first_cluster = next(iter(out_paths.keys()))
        progress_queue.put(("done", first_cluster, None))
        return 0

    except Exception as exc:  # pragma: no cover - passthrough
        logger.error("Writer failed: %s", exc)
        raise


# ------------------------------------------------------------------ #
# --------------------------- main API ----------------------------- #
# ------------------------------------------------------------------ #

def split_bam_clusters(
    adata: sc.AnnData,
    bams: str | Iterable[str],
    groupby: str,
    barcode_col: Optional[str] = None,
    read_tag: str = "CB",
    output_prefix: str = "split_",
    reader_threads: Optional[int] = 1,
    writer_threads: Optional[int] = 1,
    parallel: bool = False,
    pysam_threads: Optional[int] = 4,
    buffer_size: int = 10000,
    max_queue_size: int = 1000,
    individual_pbars: bool = False,
    sort_bams: bool = False,
    index_bams: bool = False,
) -> None:
    """
    Split BAM files into clusters based on `groupby` from the AnnData `.obs` table.
    """

    _check_module("tqdm")
    if _is_notebook():
        from tqdm import tqdm_notebook as tqdm  # type: ignore
    else:
        from tqdm import tqdm  # type: ignore

    _check_module("pysam")
    import pysam

    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs!")
    if barcode_col is not None and barcode_col not in adata.obs.columns:
        raise ValueError(f"Column '{barcode_col}' not found in adata.obs!")
    if index_bams and not sort_bams:
        raise ValueError("`sort_bams=True` must be set for indexing to be possible.")

    if isinstance(bams, str):
        bams = [bams]

    _create_dir(os.path.dirname(output_prefix))

    clusters = list(set(adata.obs[groupby]))
    logger.info("Found %d groups in .obs.%s: %s", len(clusters), groupby, clusters)

    if writer_threads and writer_threads > len(clusters):
        logger.info(
            "Limiting writer_threads from %d to number of clusters %d",
            writer_threads,
            len(clusters),
        )
        writer_threads = len(clusters)

    if barcode_col is None:
        barcode2cluster = dict(zip(adata.obs.index.tolist(), adata.obs[groupby]))
    else:
        barcode2cluster = dict(zip(adata.obs[barcode_col], adata.obs[groupby]))

    template = open_bam(list(bams)[0], "rb", verbosity=0)

    logger.info("Reading total number of reads from bams...")
    n_reads: Dict[str, int] = {}
    for path in bams:
        handle = open_bam(path, "rb", verbosity=0)
        n_reads[path] = get_bam_reads(handle)
        handle.close()

    logger.info("Starting splitting of bams...")
    if parallel:
        out_paths: Dict[str, str] = {}
        output_files: List[str] = []
        for cluster in clusters:
            save_cluster_name = re.sub(r'[\\/*?:"<>| ]', "_", str(cluster))
            out_paths[cluster] = f"{output_prefix}{save_cluster_name}.bam"
            output_files.append(out_paths[cluster])

        reader_pool = mp.Pool(reader_threads)
        writer_pool = mp.Pool(writer_threads or 1)
        manager = mp.Manager()

        cluster_chunks = _split_list(clusters, writer_threads or 1)
        cluster_queues: Dict[str, Any] = {}
        for chunk in cluster_chunks:
            queue = manager.Queue(maxsize=max_queue_size)
            for cluster in chunk:
                cluster_queues[cluster] = queue

        progress_queue = manager.Queue()

        reader_jobs: List[ApplyResult] = []
        for bam_path in bams:
            reader_jobs.append(
                reader_pool.apply_async(
                    _buffered_reader,
                    (bam_path, cluster_queues, barcode2cluster, read_tag, progress_queue, buffer_size),
                )
            )
        reader_pool.close()

        writer_jobs: List[ApplyResult] = []
        for chunk in cluster_chunks:
            queue = cluster_queues[chunk[0]]
            path_subset = {cluster: out_paths[cluster] for cluster in chunk}
            writer_jobs.append(
                writer_pool.apply_async(
                    _writer,
                    (queue, path_subset, str(template.header), progress_queue, pysam_threads or 4),
                )
            )
        writer_pool.close()

        _monitor_progress(progress_queue, cluster_queues, reader_jobs, writer_jobs, n_reads, individual_pbars)

    else:
        for bam_path in bams:
            bam_handle = open_bam(bam_path, "rb", verbosity=0)

            buffers = {cluster: [] for cluster in clusters}
            writers: Dict[str, "pysam.AlignmentFile"] = {}
            for cluster in clusters:
                save_cluster_name = re.sub(r'[\\/*?:"<>| ]', "_", str(cluster))
                out_path = f"{output_prefix}{save_cluster_name}.bam"
                writers[cluster] = pysam.AlignmentFile(out_path, "wb", header=template.header)  # type: ignore[arg-type]

            step = 100000
            n_reads_step = 0
            for read in bam_handle:
                bc = read.get_tag(read_tag) if read.has_tag(read_tag) else None
                if bc in barcode2cluster:
                    cluster = barcode2cluster[bc]
                    buffers[cluster].append(read)
                    if len(buffers[cluster]) >= buffer_size:
                        for buffered_read in buffers[cluster]:
                            writers[cluster].write(buffered_read)
                        buffers[cluster] = []

                n_reads_step += 1
                if n_reads_step == step:
                    n_reads_step = 0

            for cluster, buffer_reads in buffers.items():
                if buffer_reads:
                    for buffered_read in buffer_reads:
                        writers[cluster].write(buffered_read)

            for handle in writers.values():
                handle.close()

            bam_handle.close()

    if sort_bams:
        logger.info("Sorting output bams...")
        output_files = [
            f"{output_prefix}{re.sub(r'[\\/*?:\"<>| ]', '_', str(cluster))}.bam" for cluster in clusters
        ]
        for file in tqdm(output_files, desc="Sorting reads", unit="files"):
            temp_file = file + ".tmp"
            pysam.sort("-o", temp_file, file)
            os.replace(temp_file, file)

    if index_bams:
        logger.info("Indexing output bams...")
        output_files = [
            f"{output_prefix}{re.sub(r'[\\/*?:\"<>| ]', '_', str(cluster))}.bam" for cluster in clusters
        ]
        for file in tqdm(output_files, desc="Indexing", unit="files"):
            pysam.index(file, "-@", str(pysam_threads or 4))

    logger.info("Finished splitting bams!")
