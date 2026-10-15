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

Final code (after A3), 16 threads. D1 = atac_pbmc_500_nextgem (493 Cell Ranger cells,
33.5 M read pairs), D2 = pbmc_granulocyte_sorted_3k ATAC (2 638 Cell Ranger cells, 77.2 M
read pairs; 2 711 ARC cells for G8).

| Gate | Output | Threshold | D1 chromap | D1 bwa-mem2 | D1 bowtie2 | D2 chromap | D2 bwa-mem2 | D2 bowtie2 | Pass |
|---|---|---|---|---|---|---|---|---|---|
| G1a | raw barcode per read | ≥ 0.9999 | 1.0 (33 535 381 reads) | | | 1.0 (77 215 568 reads) | | | yes |
| G1b | corrected barcode per read | ≥ 0.999 | 0.99999997 (1 read differs) | | | 0.99999992 (6 reads differ) | | | yes |
| G1c | corrected barcode, raw off whitelist | ≥ 0.98 | 0.9999995 (2.10 M reads) | | | 0.9999986 (4.23 M reads) | | | yes |
| G2 | top-K barcodes vs Cell Ranger cells | ≥ 0.95 | 0.8925 | 0.8925 | 0.8925 | 0.9418 | 0.9418 | 0.9418 | **no** |
| G3 | unique fragments per cell, Pearson | ≥ 0.99 | 0.999997 | 0.999997 | 0.999987 | 0.999996 | 0.999997 | 0.999996 | yes |
| G4 | median per-cell fragment Jaccard | ≥ 0.80 / 0.90 / 0.75 | 0.9743 | 0.9910 | 0.9660 | 0.9680 | 0.9853 | 0.9617 | yes |
| G5 | TSS enrichment per cell, Pearson | ≥ 0.95 | 0.9899 | 0.9991 | 0.9893 | 0.9931 | 0.9973 | 0.9937 | yes |
| G6 | Leiden ARI on LSI | ≥ 0.75 | 0.9594 | 0.9102 | 0.8683 | 0.7950 | 0.9128 | 0.9387 | yes |
| G7a | records vs SnapATAC2 2.8.0, same BAM | ≥ 0.999 | | 0.9823 | 0.99999 | | | | **no** (bwa-mem2) / yes (bowtie2) |
| G7b | equal count column among shared records | ≥ 0.999 | | 0.99903 | 1.0 | | | | yes |
| G8a | top-K (GEX-translated) vs ARC cells | ≥ 0.90 | | | | 0.9350 | | | yes |
| G8b | fragments per ARC cell, Pearson | ≥ 0.99 | | | | 0.999997 | | | yes |

First (pre-A3) bwa-mem2 run on D1, reported as required: G4 **0.859** (fail), G3 0.99991,
G5 0.9984, G6 0.958.

**Failures, diagnosed (thresholds unchanged)**

* **G2, every aligner, both datasets: the gate definition, not Epione.** The same top-K
  rule applied to *Cell Ranger's own* fragment file gives the same overlap with Cell
  Ranger's cell calls (D1 0.8925, D2 0.9418, identical to Epione's numbers). The top-K
  sets of Epione and Cell Ranger agree almost completely: Jaccard 1.0 for all D1 runs
  and for D2 chromap / bowtie2, 0.9992 for D2 bwa-mem2. Cell Ranger's caller is not a
  top-K rule (it fits a mixture model and removes gel-bead multiplets), and a 0.95
  threshold on top-K against it was not attainable even for Cell Ranger itself. G8a,
  which compares against ARC's calls, has the same control (ARC's own top-K: 0.9357 vs
  Epione 0.9350).
* **G7a on the bwa-mem2 BAM: a soft-clip convention difference, where Epione is the one
  closer to Cell Ranger.** On the bowtie2 BAM (end-to-end alignments, no soft clips)
  Epione and SnapATAC2 agree on 99.999 % of records, with identical counts. On
  bwa-mem2's BAM (local alignment, soft clips) they differ in how a soft-clipped 5' end
  is placed. Epione always uses the unclipped 5' end. SnapATAC2 2.8.0 does this for only
  some reads (its differing records sit at ±2–5 bp and ±14–15 bp). On chr1 Epione's file
  holds 99.4 % of Cell Ranger's records, SnapATAC2's 98.5 %. Of the records only Epione
  has, 99.1 % are in Cell Ranger's file; of SnapATAC2's alternatives, 31.8 %. The counts
  of shared records agree (G7b passes).
* Minor: 133 of 10.7 M records (mostly chrM, high duplicate counts) are in Epione's
  bowtie2-BAM output but not in SnapATAC2's; not investigated further.

**Speed** (wall time, `/usr/bin/time -v`, 16 threads each, same node; peak RSS is that of
the largest single process: the aligner for Epione, one Martian stage for Cell Ranger)

| Data | Tool | Wall time | Peak RSS |
|---|---|---|---|
| D1 | Cell Ranger ATAC 2.2.0 (`count`, all outputs incl. peaks / matrices) | 28 m 57 s | 7.0 GB |
| D1 | Epione chromap | 2 m 17 s | 21.9 GB |
| D1 | Epione bwa-mem2 (+ sorted BAM) | 14 m 50 s | 26.3 GB |
| D1 | Epione bowtie2 (+ fastp, + sorted BAM) | 33 m 47 s | 19.0 GB |
| D2 | Cell Ranger ATAC 2.2.0 | 1 h 04 m 58 s | 7.8 GB |
| D2 | Epione chromap | 6 m 32 s | 22.4 GB |
| D2 | Epione chromap + ATAC→GEX translation | 6 m 46 s | 22.6 GB |
| D2 | Epione bwa-mem2 (+ sorted BAM) | 34 m 42 s | 27.8 GB |
| D2 | Epione bowtie2 (+ fastp, + sorted BAM) | 1 h 15 m 16 s | 9.2 GB |

D2 bwa-mem2 / bowtie2 shared the node with a 6-thread tutorial run for part of their
time. Index building (once per genome, GRCh38 2020-A): chromap 3 m 43 s (4 threads,
26.6 GB), bwa-mem2 21 m 14 s (1 thread, 72.7 GB), bowtie2-build 31 m 41 s (4 threads,
5.9 GB). Cell Ranger ships its bwa index in the reference.

Scripts: `parity_barcodes.py` (G1), `parity_fragments.py` (G2–G6, G8),
`parity_snapatac2.py` (G7). Data, references and Cell Ranger outputs live outside the
repository.

### Follow-up: parallel BAM → fragments (no change to outputs)

The bwa-mem2 / bowtie2 path now coordinate-sorts and indexes the aligner BAM first and collects
fragments one reference per worker, instead of one single-threaded pass over the unsorted BAM.
The per-reference workers also return the reads whose mate they did not see (pairs split
across references, unmapped mates without a coordinate), so the pair counters match the single
pass. Checked on the D1 bwa-mem2 BAM (16 threads): the 10,892,576 nuclear + chrM fragment
records, with their counts, are identical to the file of the run reported above, and
`pairs` / `both_mapped` / `passing` / `with_barcode` are identical to its `qc_summary.json`.
Fragment collection took 266 s single-threaded and 65 s in parallel. The gate results above
therefore stand without a re-run.
