# Installation

EpiOne requires Python 3.10+.

- From PyPI (recommended once published): `pip install epione`
- From source (this repository):

```bash
git clone https://github.com/aristoteleo/epione.git
cd epione
pip install -U pip
pip install -e .
```

Optional extras for full functionality:

```bash
pip install .[full]
```


GPU acceleration (PyTorch backend of the `device=` arguments):

```bash
pip install .[gpu]
```

See {doc}`gpu` for CUDA builds of torch and the optional rapids-singlecell backend.

## Command-line tools for upstream processing

The FASTQ-level functions in `epione.upstream` call command-line programs that are
not Python packages, so pip cannot install them and they are not listed in
`pyproject.toml`. Install them from bioconda, into the same environment as Epione or
a separate one on `PATH`:

```bash
# single-cell ATAC FASTQ -> fragments (epi.upstream.fastq_to_fragments)
mamba install -c conda-forge -c bioconda chromap bwa-mem2 bowtie2 samtools fastp htslib

# bulk ATAC / ChIP / CUT&RUN tutorials additionally use
mamba install -c conda-forge -c bioconda bedtools macs2 subread
```

`epi.upstream.check_tools(epi.upstream.SC_ATAC_TOOLS)` reports which ones Epione can
find (on `PATH` or next to the running Python). Only the aligner you choose is needed:
chromap alone for the default `aligner="chromap"`, bwa-mem2 + samtools, or bowtie2 +
samtools + fastp.
