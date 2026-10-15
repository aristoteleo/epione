# Parity manifest — `epi.upstream.fastq_to_fragments` vs Cell Ranger ATAC 2.2.0 and SnapATAC2

Pre-registered on 2026-10-15, before the first comparison. The thresholds below are
read-only: if a gate fails, the cause is diagnosed and reported, and the threshold is
not changed. Any amendment is dated, written here before the re-run, and reported.

## References (outside this repository)

* **Cell Ranger ATAC 2.2.0** (10x Genomics, binary distribution; licence forbids
  redistribution, so only its outputs are used, and only outside the repo).
  `cellranger-atac count --reference refdata-cellranger-arc-GRCh38-2020-A-2.0.0
  --localcores 16`, default parameters (D2 adds `--chemistry ARC-v1`). Used: per-read
  `CR` / `CB` tags of `possorted_bam.bam`, `fragments.tsv.gz`, cell calls in
  `singlecell.csv` (`is__cell_barcode == 1`).
* **10x-published outputs** of the same datasets (Cell Ranger ATAC 2.0.0 for D1,
  Cell Ranger ARC 2.0.0 for D2): informational for D1, and the reference of gate G8
  (Multiome barcode translation) for D2.
* **SnapATAC2** `pp.make_fragment_file(bam, out, is_paired=True, barcode_tag="CB",
  shift_left=4, shift_right=-5, min_mapq=30)` (latest PyPI release, its own venv) on the
  coordinate-sorted BAM written by Epione's bwa-mem2 and bowtie2 runs.

## Datasets

* **D1** `atac_pbmc_500_nextgem` — 10x Single Cell ATAC (Next GEM v1.1), PBMC, FASTQs
  from the 10x website (2 lanes; R1 / R2 = 16-bp barcode / R3).
* **D2** `pbmc_granulocyte_sorted_3k` — 10x Multiome, ATAC library FASTQs only (4 lanes).
  Whitelist `737K-arc-v1` (ATAC version). The Cell Ranger ATAC 2.2.0 run reports
  ATAC-space barcodes; the published ARC output reports GEX-space barcodes.

Both use the FASTA of `refdata-cellranger-arc-GRCh38-2020-A-2.0.0` for every tool, the
whitelists shipped with Cell Ranger ATAC (`737K-cratac-v1`, `737K-arc-v1`), and 16
threads. Epione runs use the defaults of `fastq_to_fragments` (MAPQ ≥ 30 on both mates,
insert ≤ 2000 bp, posterior ≥ 0.975, pseudocount 1, adapter trimming only for bowtie2 — see amendments A1, A3)
with `aligner` in `chromap`, `bwa-mem2`, `bowtie2`.

## Fixed definitions

* All fragment comparisons are restricted to the nuclear primary contigs chr1–22, X, Y
  (Cell Ranger leaves chrM out of its fragment file).
* **Cells** = Cell Ranger ATAC 2.2.0 called cells of the dataset (D2: in ATAC space).
  Per-cell metrics are computed on those barcodes for every fragment file.
* **Unique fragments of a cell** = number of lines of the fragment file for that barcode
  (duplicates collapsed).
* **Epione called cells** (G2): the top-K barcodes by unique nuclear fragments in the
  Epione file, K = number of Cell Ranger cells.
* **TSS enrichment** of a cell: Tn5 insertions (both fragment ends) are counted around
  the TSSs of `regions/tss.bed` (strand-aware); score =
  (insertions within ±50 bp / 101) / ((insertions in the flanks 1901–2000 bp up- and
  downstream + 1) / 200). Same code for every file.
* **Clustering** (G6): 5-kb genome tiles, counts per cell from each file; the 50 000
  tiles with the most Cell Ranger insertions over the cells are used for every file;
  log-TF-IDF, TruncatedSVD with 30 components (`random_state=0`), drop component 1,
  scanpy neighbours (k = 15, euclidean on the 29 L2-normalised components),
  Leiden resolution 1.0, `random_state=0`. ARI against the same pipeline on the Cell
  Ranger fragments.

## Gates

| # | Output | Data | Aligner(s) | Metric | Threshold |
|---|---|---|---|---|---|
| G1a | raw barcode extraction + orientation | D1, D2 | (aligner-free) | fraction of Cell Ranger reads whose `CR` tag equals Epione's oriented raw barcode for the same read name | ≥ 0.9999 |
| G1b | corrected barcode per read | D1, D2 | (aligner-free, = bwa-mem2 / bowtie2 path) | fraction of reads where Epione's `CB` equals Cell Ranger's `CB` (both absent counts as agreement) | ≥ 0.999 |
| G1c | corrected barcode, reads whose raw barcode is off the whitelist | D1, D2 | same | same agreement on that subset | ≥ 0.98 |
| G2 | called-cell barcodes | D1, D2 | each | Jaccard(Epione top-K, Cell Ranger cells) | ≥ 0.95 |
| G3 | unique fragments per cell | D1, D2 | each | Pearson r over cells | ≥ 0.99 |
| G4 | fragments per cell | D1, D2 | bwa-mem2 / chromap / bowtie2 | median over cells of the Jaccard of exact (chrom, start, end) sets | ≥ 0.90 / ≥ 0.80 / ≥ 0.75 |
| G5 | TSS enrichment per cell | D1, D2 | each | Pearson r over cells | ≥ 0.95 |
| G6 | Leiden clusters on LSI | D1, D2 | each | ARI vs Cell Ranger fragments | ≥ 0.75 |
| G7 | BAM → fragments step | D1 | bwa-mem2, bowtie2 BAMs | Jaccard of unique (chrom, start, end, barcode) records, Epione `bam_to_fragments(max_fragment_length=None)` vs SnapATAC2 `make_fragment_file`; and fraction of shared records with equal count column | ≥ 0.999 ; ≥ 0.999 |
| G8 | Multiome ATAC → GEX barcode translation | D2 | chromap with `barcode_translation=(atac_wl, gex_wl)` | (a) Jaccard(Epione top-K, ARC cells), K = number of ARC cells; (b) Pearson r of unique fragments per ARC cell | ≥ 0.90 ; ≥ 0.99 |

Notes fixed in advance:

* G1 compares per read name over every primary read-1 record of the Cell Ranger BAM;
  Epione's correction is run on the FASTQs with the prior from the same run, exactly as
  inside `fastq_to_fragments`. Cell Ranger may drop reads before writing the BAM; reads
  missing from either side are counted and reported, not scored.
* G4 thresholds differ by aligner because only bwa-mem2 reproduces Cell Ranger's
  aligner (bwa mem) exactly; chromap and bowtie2 are different aligners with their own
  MAPQ calibration.
* G8 cells come from ARC's joint RNA + ATAC calling, which Epione cannot reproduce from
  ATAC alone, hence the lower overlap threshold.
* Speed: wall time and peak RSS of every run (Cell Ranger and each aligner, 16 threads,
  same node; index building reported separately) are reported next to the gates; they
  are not gates.

## Amendments (dated, registered before the re-run they apply to)

* **A1 — 2026-10-15, implementation default, not a threshold.** The first D1 bwa-mem2
  run (no adapter trimming, as pre-registered above) failed G4: median per-cell
  fragment Jaccard 0.859 < 0.90, while chromap (0.974) and bowtie2 with fastp
  trimming (0.966) passed. Diagnosis on chr1: the fragments that differ are short
  (median 36 bp vs 125 bp overall, reads are 50 / 49 bp), and of the 26 222
  Cell Ranger-only fragments with a same-start, same-cell bwa-mem2 counterpart, 22 002
  (84 %) end 1–3 bp further out in bwa-mem2 (the mirror image holds for same-end
  pairs), i.e.
  local alignment extending into untrimmed adapter whose first bases match the genome
  by chance. Cell Ranger trims adapters before bwa. The default of
  `fastq_to_fragments` is changed to trim with fastp for bwa-mem2 too
  (`trim_adapters="auto"` now means trimming for bwa-mem2 and bowtie2). bwa-mem2 is
  re-run on D1 and run on D2 with the new default; the failing first run is reported
  next to it. All thresholds unchanged. *(Superseded by A3 before that re-run was
  made.)*
* **A2 — 2026-10-15, reference version.** SnapATAC2 2.9.0 (latest) has no installable
  build on this cluster (its `macs3` dependency fails to build); 2.8.0, the newest
  wheel that installs, is used for G7. `make_fragment_file` arguments as above.

* **A3 — 2026-10-15, implementation fix; supersedes A1's diagnosis and default
  change.** Before re-running with A1, G7 was run on the BAM of the failing first
  bwa-mem2 run (no trimming): Epione `bam_to_fragments` vs SnapATAC2 gave record
  Jaccard 0.879, and SnapATAC2's fragments from that same BAM scored a per-cell Jaccard
  of 0.975 against Cell Ranger. So the bwa-mem2 G4 failure came from how Epione turned a
  read pair into a fragment, not from missing trimming. Two errors, both fixed:
  1. Epione took the leftmost aligned base and the rightmost aligned end over *both*
     mates. On short fragments each mate runs into adapter at its 3' end, and bwa-mem2
     extends a few bases into it, past the other mate's 5' end. The fragment is now
     bounded by the forward mate's 5' end and the reverse mate's 5' end.
  2. Epione used aligned 5' ends. For mates soft-clipped at the 5' end, Cell Ranger
     and SnapATAC2 use the unclipped 5' end: among Epione-only records on chr1, 7.6 %
     are in Cell Ranger's file, against 91 % of SnapATAC2's variant of the same
     fragment. The 5' ends are now unclipped.
  The `trim_adapters="auto"` default is back to the pre-registered one (trim for
  bowtie2 only). All D1 runs (chromap, bwa-mem2, bowtie2) and all D2 runs are made
  again with the fixed code, and only those results are scored. The failing first
  bwa-mem2 run is reported. Thresholds unchanged. Caveat: this fix was found by
  diagnosing the D1 gates. D2 had not been run by any aligner when it was made, so D2
  is the held-out check.

## Results (appended after the run; thresholds above unchanged)

_pending_
