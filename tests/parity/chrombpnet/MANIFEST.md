# ChromBPNet parity, pre-registered

Written before any Epione ChromBPNet code ran against a reference. The gates and
tolerances below are fixed; results are appended under **Results** without
changing them. Amendments, if any, are added as dated sections above the
results and say why.

## What is compared

- **Reference:** official ChromBPNet (kundajelab/chrombpnet, MIT), commit `09938fd`
  (v1.0.1 code). It runs in a TensorFlow env *outside the repository*:
  - TF 2.15.1 and tensorflow-probability 0.23.0. Upstream pins TF 2.8; that pin has
    no wheel for this cluster's glibc, while 2.15 runs the unmodified code.
  - Python 3.10, CUDA via the `tensorflow[and-cuda]` wheels.
  - The env is not referenced by any file in the repository.
- **Epione:** `epione.models.chrombpnet`, pure PyTorch. The repository never imports
  TensorFlow or Keras. Official `.h5` files are read with `h5py`.
- **Discovery:**
  - gReLU (Genentech, MIT) has no BPNet/ChromBPNet model; its trunks are Borzoi, Enformer
    and ExplaiNN.
  - bpnet-lite (jmschrei, MIT) has a PyTorch ChromBPNet and an h5 loader, but trains with
    its own recipe; its README warns its models "may underperform those trained using
    the official repositories". It also pulls in macs3, bam2bw, tangermeme and
    modisco-lite as hard dependencies.
  - So Epione writes the small network itself, following the official Keras definitions
    (`bpnet_model.py`, `chrombpnet_with_bias_model.py`), and reproduces the official
    training recipe (`train.py`, `batchgen_generator.py`, `augment.py`,
    `find_*_hyperparams.py`), not bpnet-lite's.
- **Folds:** ChromBPNet's ENCODE folds (`cv_params.fold_k.json` in ENCODE ENCFF633GMB) are
  identical, for all five folds, to `epione.models.seq2print.FOLDS["hg38"]` (scPrinter
  copied them). Epione reuses that table.

## Gates

### G1: same weights, class 1 (numerical identity up to float32 rounding)

- **Models:** ENCODE GM12878 ChromBPNet (annotation ENCSR081YFO, file ENCFF772UBB,
  md5 `47a4549c40d25bb2440f4fc9b07eec21`), fold 0. Three files: `bias_scaled`,
  `chrombpnet_nobias` and `chrombpnet` (full).
- **Inputs:** 512 fixed sequences of 2,114 bp, float32 on CPU on both sides:
  - 256 centred on GM12878-relevant hg38 positions, drawn with seed 0 from test chromosomes
    chr1/chr3/chr6, uniformly;
  - 256 uniformly random ACGT sequences, seed 1.
- **Pass if, for every model:**
  - profile logits: max |Δ| ≤ 1e-3;
  - log-counts: max |Δ| ≤ 1e-3;
  - profile probabilities: mean Jensen–Shannon distance ≤ 1e-5.

### G2: data recipe, class 1 (exact)

On the training dataset below, with identical peaks, non-peaks, bigWig and fold, Epione's
`find_bias_hyperparams` / `find_chrombpnet_hyperparams` equivalents must reproduce the
official values exactly:

- the counts thresholds, `counts_loss_weight` (rounded as upstream writes it), and the
  filtered peak and non-peak sets (same rows, same order);
- the scaled bias model's count-head bias after `adjust_bias_model_logcounts`, given the
  same bias weights. Tolerance 1e-5, because it is a float32 mean.

The insertion track must also match. Epione reads insertions from the fragments file with
`InsertionReader(shift=(0, 0))`. Its per-base counts must equal the official
`reads_to_bigwig` bigWig for 10x fragments, exactly, on 2,000 random 1-kb windows. That
bigWig is built by the official pipeline with its auto-detected shift.
- Precondition: `bedtools genomecov` and `bedGraphToBigWig` must be available in the
  reference env.
- If `bedGraphToBigWig` is unavailable, Epione writes the bigWig the official pipeline
  would: the same awk shift and genomecov `-5` bedGraph, converted with pyBigWig. That is
  then recorded as a deviation, not a pass.

### G3: training from scratch, metric level

- **Data:** 10x Multiome PBMC granulocyte-sorted 3k (public 10x URL), all cells pooled.
  - Peaks: MACS3 narrowPeak on the pooled fragments (`-f BED --nomodel --shift -75
    --extsize 150 --keep-dup all -q 0.01`), then the official top-300k/500-bp flow is not
    needed at this size.
  - Non-peaks: GC-matched with the official `prep nonpeaks` helper, ratio 2. One set,
    generated once and shared.
  - Fold 0 restricted for compute: train on chr2, chr4, chr5, chr7, chr9, chr10;
    validation chr8, chr20; test chr1, chr3, chr6, as in fold 0. The fold JSON is shared.
- **Recipe:** official defaults throughout.
  - Bias model: 128 filters, 4 dilated layers, `bias_threshold_factor` 0.5, no jitter.
  - ChromBPNet: 512 filters, 8 dilated layers, jitter 500, negative sampling 0.1.
  - Batch 64, Adam with learning rate 1e-3, 50 epochs maximum, early stopping with
    patience 5 on `val_loss`, restoring the best weights.
- **Runs:**
  - Reference: seeds 1234, 1235 and 1236, each training a bias model and then ChromBPNet
    on it.
  - Epione: the same three seeds. TF and torch RNGs differ, so seeds pair by name only.
- **Metrics** on test-chromosome peaks (summit-centred, no jitter):
  - (a) count Pearson of the full model, log(1 + total) predicted vs observed;
  - (b) median profile Jensen–Shannon distance of the full model (observed counts
    normalised vs predicted probabilities);
  - (c) bias model count Pearson on test non-peaks;
  - (d) bias-corrected footprint depth: CTCF motif (JASPAR MA0139.1) marginal footprint
    from the `nobias` model, i.e. the mean predicted profile probability ±5 bp around the
    motif centre, divided by the mean in the ±150–250 bp flanks, over 500 dinucleotide-
    shuffled backgrounds with the motif inserted at the centre.
- **Pass if** for each metric |mean_Epione − mean_ref| ≤ max(2 × SD_ref, floor). SD_ref is
  the sample SD of the three reference seeds. Floors:
  - (a) 0.02;
  - (b) 0.01;
  - (c) 0.03;
  - (d) 0.05.
- G3 passes only if all four do.

### G4: contribution scores (reported, not gated)

DeepLIFT-SHAP counts-head attributions of the `nobias` model, with 20 dinucleotide
shuffles, compared with the official interpretation on the same 200 test peaks:

- per-region Pearson of the projected (×one-hot) scores, median over regions;
- the official tool's own two-seed agreement shown alongside.

This is reported only, because the official path needs the kundajelab `shap` fork and
`deeplift`. If those cannot be installed against TF 2.15, the comparison is skipped and
that is stated.

### G5: Tn5 bias backend (internal, exact)

- `epi.tl.tn5_bias(method="chrombpnet", ...)` returns `ChromBPNetBias`. Its
  `track(chrom, start, end)` must equal the bias model's direct prediction stitched over
  the same windows, bit for bit, on CPU.
- `shift` is `(0, 0)`, the convention under which the official pipeline builds the
  training bigWig from 10x fragments.

## Speed (reported)

Wall time per training epoch and for prediction over 10,000 regions, on the same H100,
both sides. Any speed-up is reported as measured. No acceleration path is added unless it
passes G1 to G3 unchanged.

## Results

(Appended after the runs.)
