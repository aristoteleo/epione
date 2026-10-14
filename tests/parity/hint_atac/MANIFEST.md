# HINT-ATAC port: pre-registered parity gates

Written on 2026-10-13, before any Epione HINT code existed and before any comparison was run.
The thresholds below are read-only. Results are appended in a separate section after the run;
a failed gate is reported as failed and is not re-tuned.

## Scope

| Epione | Upstream (RGT 1.0.2, `rgt-hint` / `rgt-motifanalysis`, GPL-3) | Gates |
|---|---|---|
| `epi.tl.HintBias` (and `epi.tl.tn5_bias(method="hint")`) | `BiasTable.load_table` + the k-mer lookup inside `GenomicSignal.bias_correction_atac` / `get_bc_signal_by_fragment_length` | H1 |
| `epi.tl.hint_signal` | `GenomicSignal.bias_correction_atac`, `get_signal_atac` (single-end model), `get_bc_signal_by_fragment_length` + Boyle/Hon normalisation + Savitzky-Golay slope (paired-end model) | H2-H4 |
| `epi.tl.hint_footprints` | `rgt-hint footprinting --atac-seq [--paired-end]` (HMM Viterbi decoding, footprint post-processing, tag count) | H5-H7 |
| `epi.tl.hint_match_motifs` | `rgt-motifanalysis matching` (MOODS, `--fpr 1e-4 --pseudocounts 1`) | H8 |
| `epi.tl.hint_differential` | `rgt-hint differential [--bc]` (TF activity, protection score, TC, z-score, p-value) | H9 |
| full Epione pipeline | full official pipeline | H10 |

## Reference

- **Code.** `CostaLab/reg-gen` at commit `66f5fbb3` (`__version__ = "1.0.2"`), installed from that checkout
  into a conda env outside the repository: Python 3.7, hmmlearn 0.2.2 (the version `setup.py` pins),
  scikit-learn 0.24.2, numpy 1.21.6, scipy 1.7.3, pysam 0.16.0, MOODS 1.9.4.1. `RGTDATA` points at a
  directory outside the repository; a custom organism `tobias_chr4` is registered in `data.config.user`
  for the fixture genome.
- **Fixture.** The TOBIAS tutorial bundle already used by `epione_guide/tutorials/bulk/t_footprint.ipynb`
  (Bentsen *et al.* 2020): human B-cell and T-cell ATAC-seq, paired-end, chr4 only, hg38;
  `merged_peaks.bed` (6,007 peaks); `motifs.jaspar` (83 JASPAR 2020 motifs).
- **Model files.** The default HMMs (`atac_single.pkl`, `atac_paired.pkl`) and ATAC bias tables
  (`atac_bias_table_{F,R}.txt`) from `data/fp_hmms/` at that commit. Epione stores the HMM parameters
  as plain arrays; their equality with the pickles is part of gate H1.
- **Runs.** `rgt-hint footprinting --atac-seq` with and without `--paired-end`, for each of Bcell and
  Tcell, on `merged_peaks.bed`. `rgt-motifanalysis matching` on the paired-end footprints, with the 83
  motifs written as RGT `.pwm` files. `rgt-hint differential --nc 1 --output-profiles`, with and without
  `--bc`, on those MPBS files. Per-peak intermediate tracks (H1-H5) are dumped by a driver that calls the
  upstream `GenomicSignal` methods directly, with exactly the arguments `atac_seq()` passes.

Reference outputs are written outside the repository and read from `$EPIONE_HINT_PARITY_DIR`.
`parity_check.py` and `tests/test_hint.py` skip the comparisons when the variable is unset.

## Gates

Classes follow the rebuildr parity taxonomy. "Every peak" means all 6,007 peaks of both samples
(and both models where the track exists); a gate passes only if it passes on every sample and model.

| # | Output | Class | Metric | Threshold |
|---|---|---|---|---|
| H1 | HMM parameters (startprob, transmat, means, covars) vs the pickles; per-base forward / reverse k-mer bias of every peak window | exact | equality | identical |
| H2 | raw strand-specific Tn5 cut counts of every peak window (all reads; fragment length ≤145 and >145 for the paired model) | exact | equality | identical |
| H3 | bias-corrected strand-specific signal of every peak (single model: `bias_correction_atac` f/r; paired model: f/r for ≤145 and >145) | deterministic-strict | max abs err | ≤ 1e-13 (also reported: fraction of bases bit-identical) |
| H4 | HMM input tracks of every peak (single: 4 normalised/slope tracks; paired: 8) | deterministic-standard | max abs err | ≤ 1e-8 |
| H5 | Viterbi state sequence of every peak | classification | per-base agreement over all peaks | ≥ 0.999 |
| H6 | footprint BED (4 runs: Bcell/Tcell × single/paired) | overlap | bp-level Jaccard **and** fraction of reference footprints with an identical (chrom, start, end) in Epione, and vice versa | each ≥ 0.99 |
| H7 | footprint tag count (BED column 5) on footprints with identical coordinates | exact | equality | identical on all of them |
| H8 | MPBS from motif matching on the *reference* footprints (same input both sides) | exact set / deterministic | Jaccard of (chrom, start, end, motif, strand); max abs err of the bit-score on shared sites | Jaccard ≥ 0.999; score ≤ 1e-6 |
| H9a | differential on the *reference* MPBS (same input both sides), `--bc` and raw: `Num` per motif | exact | equality | identical |
| H9b | normalisation factors per condition | deterministic-bounded | relative err | ≤ 1e-6 |
| H9c | `Protection_Score_*`, `TC_*` per motif and condition | ordinal + sign | Pearson r; sign agreement of protection score | r ≥ 0.9999; 100 % |
| H9d | `TF_Activity` per motif | ordinal + sign | Pearson r; **exact** sign agreement | r ≥ 0.9999; 100 % |
| H9e | `Z_score`, `P_values` | inference | Spearman ρ of −log10 p; set of motifs with p < 0.05 | ρ ≥ 0.999; identical |
| H9f | normalised per-motif footprint profiles (`--output-profiles`) | ordinal | Pearson r per motif and condition, minimum over all | ≥ 0.9999 |
| H10 | end to end: Epione footprints → Epione MPBS → Epione differential (`--bc`) vs the fully official chain | ordinal + sign | Pearson r of `TF_Activity`; sign agreement | r ≥ 0.99; ≥ 95 % |

Notes fixed before the run:

- Motif names: upstream builds its motif list with `list(set(...))`, so the output order depends on the
  string hash seed. All per-motif comparisons join on the motif name.
- Upstream accumulates differential signals in a float32 array and normalises with numpy 1.21 scalar
  casting. Epione reproduces those dtypes explicitly; gates H9b-H9f are written to tolerate float32
  rounding but not a different algorithm.
- Known upstream quirks are reproduced, not fixed, in the default code path (each is documented in the
  Epione docstrings): the reverse-strand slope of the single-end model is scaled with the *forward* slope's
  standard deviation; footprinting uses a reverse shift of −5 but `differential` uses −4; the paired-end
  model skips a peak whose signal extraction raises.

## Results (appended after the run; thresholds above unchanged)

Run on 2026-10-14 with `parity_check.py --gates H1,H6,H8,H9,H10` in the Epione environment (Python
3.11, numpy 2.4, scipy 1.17, pysam 0.24, numba 0.68, MOODS 1.9.4.2). All 6,007 peaks × 2 samples ×
2 models; no peak was skipped by either side.

| # | Value | Pass |
|---|---|---|
| H1 | HMM parameters identical to the pickles (both models); k-mer bias identical at every base | ✅ |
| H2 | raw cut counts identical (all reads; ≤145 / >145 bp) | ✅ |
| H3 | max abs err **0**; 100 % of bases bit-identical | ✅ |
| H4 | max abs err 4.4e-16 | ✅ |
| H5 | state agreement 1.000000 (every base of every peak) | ✅ |
| H6 | Bcell single 13,855 / 13,855, Tcell single 12,006 / 12,006, Bcell paired 30,568 / 30,568, Tcell paired 33,673 / 33,673 footprints: bp Jaccard 1.0, identical coordinates both ways 100 % | ✅ |
| H7 | tag counts identical on all footprints | ✅ |
| H8 | Bcell 12,720 / 12,720, Tcell 12,404 / 12,404 MPBS: Jaccard 1.0, score max abs err 0 | ✅ |
| H9a | `Num` identical, 83 / 83 motifs (`--bc` and raw) | ✅ |
| H9b | factor rel. err 1.6e-8 (`--bc`), 2.0e-8 (raw), i.e. the 8-digit text rounding; identical as float32 | ✅ |
| H9c | min Pearson 0.999999999999999; protection-score signs 100 %; identical as float32 | ✅ |
| H9d | Pearson 1.0, sign agreement 100 %, max abs err 0 | ✅ |
| H9e | Spearman of −log10 p 1.0; p < 0.05 sets identical (5 motifs `--bc`, 4 raw); z and p bit-identical | ✅ |
| H9f | min profile Pearson 0.9999999999999931; profiles identical as float32 | ✅ |
| H10 | 83 motifs: Pearson 1.0, sign agreement 100 % (the full Epione chain reproduces the official one) | ✅ |

The float64 errors in H9b/H9c/H9f come from upstream writing float32 values with `str()` (8 significant
digits). Compared after casting to float32, every value is the same number.

**Timing** (same node, one core each; upstream from `/usr/bin/time`, Epione with a warm numba cache):

| Step | rgt-hint 1.0.2 | Epione, 1 core |
|---|---|---|
| footprinting, single-end model (Bcell / Tcell) | 57 s / 59 s | 11.5 s / 12.4 s |
| footprinting, paired-end model (Bcell / Tcell) | 70 s / 76 s | 16.6 s / 18.1 s |
| motif matching, both samples | 60 s | 2.3 s |
| differential `--bc` / raw | 308 s / 214 s | 15.3 s / 10.2 s |

`hint_footprints` and `hint_differential` also take `n_jobs` (peaks and motifs are independent; the
result does not depend on it, which the unit tests check).

**Reproducing.** Reference: `rgt-hint` and `rgt-motifanalysis` from the env described above, with
`PYTHONHASHSEED=0` and one thread, on the commands listed under "Reference", plus a dump driver for the
per-peak tracks. Epione: set `EPIONE_HINT_PARITY_DIR` and `EPIONE_HINT_DATA` and run `parity_check.py`
or `pytest tests/test_hint.py` (the four parity tests take about 90 s on 4 cores).
