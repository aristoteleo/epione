#!/bin/bash
# Reference run for MANIFEST.md, in the RGT 1.0.2 environment (not Epione's).
#
#   REF_ENV   conda env with RGT installed (bin/rgt-hint, bin/python)
#   RGTDATA   RGT data dir; data.config.user must define the organism
#             `tobias_chr4` (genome = the fixture genome.fa, chromosome_sizes
#             from its .fai)
#   DATA      the TOBIAS fixture (Bcell.bam, Tcell.bam, merged_peaks.bed, motifs.jaspar)
#   OUT       output dir (becomes EPIONE_HINT_PARITY_DIR)
set -euo pipefail
: "${REF_ENV:?}" "${RGTDATA:?}" "${DATA:?}" "${OUT:?}"
export RGTDATA PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HERE=$(cd "$(dirname "$0")" && pwd)
B=$REF_ENV/bin
mkdir -p "$OUT"/{fp_paired,fp_single,match,diff_bc,diff_raw,tracks,hmm_ref,pwm}
"$B/python" "$HERE/ref_jaspar_to_pwm.py" "$DATA/motifs.jaspar" "$OUT/pwm"
"$B/python" - "$RGTDATA/fp_hmms" "$OUT/hmm_ref" <<'PY'
import pickle, sys, os, numpy as np
src, out = sys.argv[1:3]
for n in ("atac_single", "atac_paired"):
    h = pickle.load(open(os.path.join(src, n + ".pkl"), "rb"), encoding="latin1")
    np.savez(os.path.join(out, n + ".npz"), startprob=h.startprob_, transmat=h.transmat_,
             means=h.means_, covars=h._covars_)
PY
for c in Bcell Tcell; do
  "$B/rgt-hint" footprinting --atac-seq --paired-end --organism tobias_chr4 \
      --output-location "$OUT/fp_paired" --output-prefix $c "$DATA/$c.bam" "$DATA/merged_peaks.bed" &
  "$B/rgt-hint" footprinting --atac-seq --organism tobias_chr4 \
      --output-location "$OUT/fp_single" --output-prefix $c "$DATA/$c.bam" "$DATA/merged_peaks.bed" &
  for m in single paired; do
    "$B/python" "$HERE/ref_dump_tracks.py" "$DATA/$c.bam" "$DATA/merged_peaks.bed" $m "$OUT/tracks/${c}_$m.npz" &
  done
done
wait
"$B/rgt-motifanalysis" matching --organism tobias_chr4 --motif-dbs "$OUT/pwm" \
    --output-location "$OUT/match" --input-files "$OUT/fp_paired/Bcell.bed" "$OUT/fp_paired/Tcell.bed"
for v in bc raw; do
  flag=""; [ $v = bc ] && flag="--bc"
  ( cd "$OUT/diff_$v" && "$B/rgt-hint" differential --organism tobias_chr4 $flag --nc 1 --output-profiles \
      --mpbs-files "$OUT/match/Bcell_mpbs.bed,$OUT/match/Tcell_mpbs.bed" \
      --reads-files "$DATA/Bcell.bam,$DATA/Tcell.bam" --conditions Bcell,Tcell \
      --output-location "$OUT/diff_$v" ) &
done
wait
