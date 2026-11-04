"""Generate t_fastq_to_fragments.ipynb (scATAC FASTQ -> fragments).

Execute with ``jupyter nbconvert --to notebook --execute --inplace``. Needs
chromap, bwa-mem2 and samtools on PATH (bioconda), the 10x
``atac_pbmc_500_nextgem`` FASTQs and outputs, a GRCh38 FASTA and the 10x
barcode whitelist; see the "Data" cell for the environment variables.
With prebuilt indexes it takes about 4 min on 16 cores (chromap on the full
run, bwa-mem2 on a 2-million-read-pair subset).
"""
import pathlib

import nbformat as nbf

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# From FASTQ to fragments

Every single-cell ATAC analysis in Epione starts from a **fragment file**: one line per
Tn5-cut DNA fragment, `chrom  start  end  barcode  count`, bgzip-compressed and
tabix-indexed. Cell Ranger ATAC writes one, but you can make it yourself from the
FASTQs with `epi.upstream.fastq_to_fragments`. That helps when Cell Ranger is not an
option (licence, non-10x chemistry, other genomes, cluster limits), or when you want
to control the aligner and the filters.

The steps are the same whatever the aligner:

1. **Barcode extraction.** 10x scATAC has three reads per fragment: `R1` and `R3` are
   genomic, `R2` (the i5 index read) holds the 16-bp cell barcode. Multiome ATAC has a
   24-bp `R2`: an 8-bp spacer, then the barcode. Depending on the sequencer the barcode
   read comes out forward or reverse-complemented, so Epione checks the first 200 000
   reads against the whitelist in both orientations.
2. **Barcode correction.** A barcode on the whitelist is kept. Otherwise every whitelist
   barcode one substitution away is a candidate. Each candidate is weighted by how often
   it was seen exactly in the run (the prior) times the error probability of the
   differing base, from its quality score. The best candidate is accepted when its
   posterior is at least 0.975. This is the scheme Cell Ranger documents.
3. **Alignment** of the read pairs, keeping pairs where both mates have MAPQ ≥ 30 and
   form a proper pair with an insert of at most 2 kb.
4. **Tn5 shift.** Tn5 inserts as a dimer with a 9-bp stagger, so the cut site is +4 bp
   from the 5′ end of the forward read and −5 bp from the 5′ end of the reverse read.
5. **Duplicate removal per cell.** Read pairs with the same start, end and barcode are
   one fragment; the 5th column counts them.

Three aligners are available through `aligner=`:

| `aligner` | What runs | Barcode correction | Notes |
|---|---|---|---|
| `"chromap"` (default) | `chromap --preset atac`: alignment, adapter trimming, correction, dedup and shift in one program | chromap's own (posterior ≥ 0.9) | fastest; index ~4 min / 26 GB RAM for GRCh38 |
| `"bwa-mem2"` | Epione corrects barcodes and streams tagged read pairs into `bwa-mem2 mem` | Epione (above) | same alignments as Cell Ranger's bwa; index needs ~ 75 GB RAM |
| `"bowtie2"` | as bwa-mem2, through `fastp` adapter trimming, then `bowtie2 --very-sensitive -X 2000` | Epione | the classic bulk-ATAC aligner |

The aligners are command-line programs, not Python packages. Install them from bioconda:

```bash
mamba install -c conda-forge -c bioconda chromap bwa-mem2 bowtie2 samtools fastp htslib
```
""")

md("## Tools")
code(r"""
import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import epione as epi

tools = epi.upstream.check_tools(epi.upstream.SC_ATAC_TOOLS, verbose=False)
{name: ("found" if path else "missing") for name, path in tools.items()}
""")

md(r"""
## Data

The 10x Genomics example dataset **atac_pbmc_500_nextgem**: about 500 peripheral blood
mononuclear cells, Next GEM Single Cell ATAC v1.1, sequenced on a NovaSeq (2 lanes).
10x also provides its Cell Ranger ATAC 2.0.0 output, which we use as the comparison:

* FASTQs: `atac_pbmc_500_nextgem_fastqs.tar` (3 GB),
* `atac_pbmc_500_nextgem_fragments.tsv.gz` (+ `.tbi`) and `atac_pbmc_500_nextgem_singlecell.csv`,

all at `https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_500_nextgem/`.
The genome is the FASTA of the 10x reference `refdata-cellranger-arc-GRCh38-2020-A-2.0.0`
(`fasta/genome.fa`), the one 10x used. The whitelist `737K-cratac-v1.txt.gz` ships with
Cell Ranger ATAC (`lib/python/atac/barcodes/`).

Paths come from environment variables so the notebook runs anywhere:

| Variable | Content |
|---|---|
| `EPIONE_SCATAC_FASTQ_DIR` | folder with `atac_pbmc_500_nextgem_S1_L00*_{I1,R1,R2,R3}_001.fastq.gz` |
| `EPIONE_GENOME_FASTA` | GRCh38 `genome.fa` (indexes are built next to it if missing) |
| `EPIONE_BARCODE_WHITELIST` | `737K-cratac-v1.txt.gz` |
| `EPIONE_CR_FRAGMENTS`, `EPIONE_CR_SINGLECELL` | the 10x fragments file and `singlecell.csv` |
| `EPIONE_TSS_BED` | `regions/tss.bed` of the same 10x reference |
| `EPIONE_CHROMAP_INDEX`, `EPIONE_BWA_MEM2_INDEX` | optional prebuilt indexes |
| `EPIONE_FRAGMENTS_OUT` | output folder (default `./fastq_to_fragments`) |
""")
code(r"""
FASTQ_DIR = os.environ["EPIONE_SCATAC_FASTQ_DIR"]
FASTA = os.environ["EPIONE_GENOME_FASTA"]
WHITELIST = os.environ["EPIONE_BARCODE_WHITELIST"]
CR_FRAGMENTS = os.environ["EPIONE_CR_FRAGMENTS"]
CR_SINGLECELL = os.environ["EPIONE_CR_SINGLECELL"]
TSS_BED = os.environ["EPIONE_TSS_BED"]
OUT = os.environ.get("EPIONE_FRAGMENTS_OUT", "fastq_to_fragments")
THREADS = int(os.environ.get("EPIONE_THREADS", "16"))

fq = epi.upstream.find_fastqs(FASTQ_DIR)
{role: [os.path.basename(f) for f in files] for role, files in fq.items()}
""")

md(r"""
`find_fastqs` groups the Illumina-named files by read role and lane. `I1` (the sample
index) is not needed. Let's look at one read of each file: `R2` is the 16-bp barcode.
""")
code(r"""
import gzip
for role in ("R1", "R2", "R3"):
    with gzip.open(fq[role][0], "rt") as fh:
        head = [next(fh).rstrip() for _ in range(2)]
    print(f"{role}: {head[1]}  ({len(head[1])} bp)")
""")

md(r"""
## chromap (default)

One call does everything. The index is built next to the FASTA the first time (or pass
`index=`). `fastq_to_fragments` returns a `FragmentsResult`: it can be used wherever a
path is expected, and carries the QC summary.
""")
code(r"""
t0 = time.time()
res_chromap = epi.upstream.fastq_to_fragments(
    fastq_dir=FASTQ_DIR,
    reference=FASTA,
    index=os.environ.get("EPIONE_CHROMAP_INDEX"),
    whitelist=WHITELIST,
    aligner="chromap",
    chemistry="atac",
    outdir=f"{OUT}/chromap",
    threads=THREADS,
    overwrite=True,
)
print(f"{time.time() - t0:.0f} s")
print(os.path.relpath(res_chromap))      # the result is the path of the fragments file
sorted(os.listdir(res_chromap.outdir))
""")
code(r"""
pd.Series(res_chromap.qc, dtype=object).to_frame("chromap")
""")
md(r"""
The barcode read matches the whitelist in forward orientation for this NovaSeq run
(`frac_whitelist_forward`: 94% of the first 200 000 barcode reads are exact whitelist
hits, none when reverse-complemented), so no reverse complement is needed. About 4% of
the read pairs carry a barcode that chromap corrected.

The fragment file is ordinary text:
""")
code(r"""
epi.upstream.read_fragments(res_chromap, nrows=5)
""")

md(r"""
## bwa-mem2

Here Epione itself extracts and corrects the barcodes, writes them as `CR` (raw), `CY`
(quality) and `CB` (corrected) tags into the read comments so that bwa-mem2 copies them
into the BAM (`-C`), and streams the pairs into `bwa-mem2 mem`. The BAM is then sorted
and indexed, and the fragments are built from it one chromosome per worker by
`epi.upstream.bam_to_fragments`, which you can also call on any paired-end BAM with
`CB` tags. `keep_bam=True` keeps the sorted, indexed BAM.

bwa-mem2 is a full read aligner and is several times slower than chromap (about 15 min
on 16 cores for this run, 35 min for a 77-million-pair Multiome library). To keep the
tutorial short we run it on the first 2 million read pairs of lane 1, and run chromap
on the same subset to compare.
""")
code(r"""
N_SUB = 2_000_000
SUB = f"{OUT}/subset_fastqs"
os.makedirs(SUB, exist_ok=True)
for role in ("R1", "R2", "R3"):
    src = fq[role][0]                                  # lane 1
    dst = os.path.join(SUB, os.path.basename(src))
    if not os.path.exists(dst):
        with gzip.open(src, "rt") as fi, gzip.open(dst, "wt", compresslevel=1) as fo:
            for _ in range(4 * N_SUB):
                fo.write(fi.readline())
sorted(os.listdir(SUB))
""")
code(r"""
sub = {}
for aligner, index in (("bwa-mem2", os.environ.get("EPIONE_BWA_MEM2_INDEX")),
                       ("chromap", os.environ.get("EPIONE_CHROMAP_INDEX"))):
    t0 = time.time()
    sub[aligner] = epi.upstream.fastq_to_fragments(
        fastq_dir=SUB, reference=FASTA, index=index, whitelist=WHITELIST,
        aligner=aligner, outdir=f"{OUT}/subset_{aligner}", threads=THREADS, overwrite=True,
    )
    print(f"{aligner}: {time.time() - t0:.0f} s")
pd.DataFrame({k: v.qc for k, v in sub.items()}).loc[
    ["n_read_pairs", "frac_valid_barcode", "frac_mapped", "frac_duplicate", "n_fragments"]]
""")
md(r"""
Most fragments are identical in the two files; the rest differ at the margin (repeats
near the MAPQ cut-off, soft-clipped ends):
""")
code(r"""
def frag_keys(path):
    df = epi.upstream.read_fragments(path)
    return set(zip(df["chrom"], df["start"], df["end"], df["barcode"]))

a, b = frag_keys(sub["bwa-mem2"]), frag_keys(sub["chromap"])
print(f"bwa-mem2 {len(a):,}  chromap {len(b):,}  shared {len(a & b):,}  "
      f"Jaccard {len(a & b) / len(a | b):.3f}")
""")

md(r"""
## QC summary

`frac_valid_barcode` is the share of read pairs with an exact or corrected whitelist
barcode, `frac_mapped` the share aligned, `frac_duplicate` the share of read pairs in
fragments that were PCR / optical duplicates. 10x's own Cell Ranger ATAC 2.0.0 run of
the same library is in the right-hand column.
""")
code(r"""
cr_sc = pd.read_csv(CR_SINGLECELL)
is_cell = cr_sc["is__cell_barcode"] == 1
cr_cells = cr_sc.loc[is_cell, "barcode"].str.replace("-1", "", regex=False).tolist()
keys = ["n_read_pairs", "frac_valid_barcode", "frac_mapped", "frac_duplicate", "n_fragments",
        "n_barcodes_ge_1000_fragments", "median_fragments_per_barcode_ge_1000", "wall_time_s"]
qc = pd.DataFrame({"chromap": res_chromap.qc}).loc[keys]
tot = cr_sc["total"].sum()
qc["Cell Ranger 2.0.0"] = [
    tot,
    1 - cr_sc.loc[cr_sc["barcode"] == "NO_BARCODE", "total"].sum() / tot,
    1 - cr_sc["unmapped"].sum() / tot,
    cr_sc["duplicate"].sum() / (cr_sc["duplicate"].sum() + cr_sc["passed_filters"].sum()),
    cr_sc["passed_filters"].sum(),
    int(is_cell.sum()),                                  # Cell Ranger's called cells
    cr_sc.loc[is_cell, "passed_filters"].median(),       # median fragments per cell
    np.nan,
]
def fmt(v):
    if pd.isna(v):
        return ""
    return f"{v:.4f}" if abs(v) < 1 else f"{v:,.0f}"
qc.astype(float).map(fmt)
""")

md(r"""
The Cell Ranger column comes from its `singlecell.csv`; its counts are read pairs
(including chrM) rather than unique fragments, and its cell count is its cell caller, so
those rows are close but not identical in definition. The comparison below uses the same definition for both files.
""")

md(r"""
## Comparison with Cell Ranger

We load the two fragment files, keep the nuclear chromosomes (Cell Ranger leaves out
chrM) and compare them on Cell Ranger's ~500 called cells. These are the definitions of
the parity gates in `tests/parity/fastq_to_fragments/MANIFEST.md`, which also has the
results with Cell Ranger ATAC 2.2.0 re-run on the same FASTQs, for all three
aligners on the full data.
""")
code(r"""
NUCLEAR = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

def load(path):
    df = epi.upstream.read_fragments(path)
    df["barcode"] = df["barcode"].str.replace("-1", "", regex=False)
    return df[df["chrom"].isin(NUCLEAR)].reset_index(drop=True)

frags = {"Cell Ranger": load(CR_FRAGMENTS), "chromap": load(res_chromap)}
{k: len(v) for k, v in frags.items()}
""")

md(r"""
**Which barcodes are cells?** Epione does not call cells; the QC in
`epi.pp` does that downstream. A crude check is to take, for each file, the K barcodes
with most fragments (K = number of Cell Ranger cells). Cell Ranger's own caller is not
a top-K rule (it fits a mixture model and removes gel-bead multiplets), so even its own
fragment file gives the same overlap with its cell calls. What matters is that the
top-K sets of the two files agree with each other:
""")
code(r"""
counts = {k: v["barcode"].value_counts() for k, v in frags.items()}
K = len(cr_cells)
jac_set = lambda a, b: len(a & b) / len(a | b)
top = {k: set(c.index[:K]) for k, c in counts.items()}
for k in top:
    print(f"{k:11s} top-{K}: Jaccard with Cell Ranger cells {jac_set(top[k], set(cr_cells)):.3f}, "
          f"with Cell Ranger's top-{K} {jac_set(top[k], top['Cell Ranger']):.3f}")
""")
code(r"""
fig, ax = plt.subplots(figsize=(4.5, 4))
x = counts["Cell Ranger"].reindex(cr_cells).fillna(0)
for k in ("chromap",):
    y = counts[k].reindex(cr_cells).fillna(0)
    ax.scatter(x, y, s=6, alpha=0.6)
    lim = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lim, lim, color="grey", lw=0.8, ls="--")
    ax.set(xlabel="Cell Ranger unique fragments", ylabel=f"{k} unique fragments",
           title=f"{k}: r = {np.corrcoef(x, y)[0, 1]:.4f}")
fig.tight_layout()
""")

md(r"""
**The same fragments?** For each cell, the Jaccard index of the exact
(chrom, start, end) sets. chromap is a different aligner from Cell Ranger's bwa, with its
own mapping-quality model, so some fragments differ at the margin. On the full data
bwa-mem2 reproduces Cell Ranger's fragments more closely (median per-cell Jaccard 0.991
vs 0.974 for chromap; see the parity manifest).
""")
code(r"""
def keys(df):
    cid = df["chrom"].map({c: i for i, c in enumerate(NUCLEAR)}).to_numpy(np.int64)
    return (cid << 50) | (df["start"].to_numpy() << 21) | (df["end"] - df["start"]).to_numpy()

def by_cell(df):
    k = keys(df)
    return {c: np.unique(k[idx]) for c, idx in pd.Series(np.arange(len(df))).groupby(df["barcode"].to_numpy()).indices.items()}

cells_ref = by_cell(frags["Cell Ranger"])
jac = {}
for k in ("chromap",):
    other = by_cell(frags[k])
    jac[k] = [np.intersect1d(cells_ref[c], other.get(c, []), assume_unique=True).size
              / np.union1d(cells_ref[c], other.get(c, [])).size for c in cr_cells]
fig, ax = plt.subplots(figsize=(5, 3.5))
for k, v in jac.items():
    ax.hist(v, bins=40, range=(0.5, 1), alpha=0.6, label=f"{k} (median {np.median(v):.3f})")
ax.set(xlabel="per-cell fragment Jaccard vs Cell Ranger", ylabel="cells")
ax.legend()
""")

md(r"""
**Fragment length** shows the nucleosome ladder in both files:
""")
code(r"""
fig, ax = plt.subplots(figsize=(6, 3.5))
for k, df in frags.items():
    sub = df[df["barcode"].isin(cr_cells)]
    h = np.bincount((sub["end"] - sub["start"]).clip(upper=1000), minlength=1001)[:1000]
    ax.plot(h / h.sum(), label=k, lw=1)
ax.set(xlabel="fragment length (bp)", ylabel="fraction", yscale="log")
ax.legend()
""")

md(r"""
**TSS enrichment** per cell: Tn5 insertions within ±50 bp of a TSS per bp, over the
insertions per bp in the flanks 1.9–2 kb away.
""")
code(r"""
tss = pd.read_csv(TSS_BED, sep="\t", header=None, usecols=[0, 1], names=["chrom", "pos"])
tss = {c: np.unique(g["pos"].to_numpy()) for c, g in tss[tss["chrom"].isin(NUCLEAR)].groupby("chrom")}

def tsse(df, cells):
    idx = {c: i for i, c in enumerate(cells)}
    df = df[df["barcode"].isin(idx)]
    center, flank = np.zeros(len(cells)), np.zeros(len(cells))
    for chrom, g in df.groupby("chrom"):
        T, b = tss[chrom], g["barcode"].map(idx).to_numpy()
        for pos in (g["start"].to_numpy(), g["end"].to_numpy() - 1):
            j = np.searchsorted(T, pos)
            d = np.minimum(np.abs(pos - T[np.clip(j - 1, 0, T.size - 1)]), np.abs(T[np.clip(j, 0, T.size - 1)] - pos))
            np.add.at(center, b[d <= 50], 1)
            np.add.at(flank, b[(d >= 1901) & (d <= 2000)], 1)
    return (center / 101) / ((flank + 1) / 200)

scores = pd.DataFrame({k: tsse(df, cr_cells) for k, df in frags.items()}, index=cr_cells)
print(scores.corr().round(4))
scores.median().round(2)
""")

md(r"""
## Downstream

The fragment files plug straight into the rest of Epione. For instance, the per-cell QC
of `epi.pp.import_fragments`:
""")
code(r"""
sizes = dict(pd.read_csv(FASTA + ".fai", sep="\t", header=None, usecols=[0, 1]).values)
sizes = {c: int(sizes[c]) for c in NUCLEAR + ["chrM"]}
adata = epi.pp.import_fragments(str(res_chromap), chrom_sizes=sizes, min_num_fragments=1000)
adata
""")
code(r"""
adata.obs.describe().round(3)
""")

md(r"""
## Other layouts

* **Multiome ATAC**: `chemistry="multiome"` with the ATAC version of the `737K-arc-v1`
  whitelist. Cell Ranger ARC reports ATAC fragments under the *gene-expression* barcode
  of the same gel bead; to get the same names pass both whitelists,
  `barcode_translation=(atac_whitelist, gex_whitelist)`: line *i* of one is line *i* of
  the other.
* **Barcode in an index file** (`bcl2fastq` without the 10x base mask: genomic `R1`/`R2`,
  barcode in `I2`): detected automatically when there is no `R3`.
* **Anything else**: `chemistry=epi.upstream.ReadLayout(read1="R1", read2="R2",
  barcode="I1", bc_start=0, bc_length=16, bc_orientation="auto")`.
* **A BAM you already have** (e.g. Cell Ranger's `possorted_bam.bam`):
  `epi.upstream.bam_to_fragments(bam, "fragments.tsv.gz", barcode_tag="CB")`.
""")


nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = pathlib.Path(__file__).with_suffix("").name.replace("_build", "") + ".ipynb"
nbf.write(nb, pathlib.Path(__file__).parent / out)
print("wrote", out)
