"""Dump HINT-ATAC per-peak intermediate tracks from upstream RGT code.

Usage: dump_tracks.py <bam> <peaks.bed> <single|paired> <out.npz>

H3-H5 tracks come from calling upstream GenomicSignal methods with exactly
the arguments rgt.HINT.Footprinting.atac_seq passes. The k-mer bias (H1) and
raw cut counts (H2) are internal to those methods; they are recomputed here
with the lines copied verbatim from rgt/HINT/signalProcessing.py
(bias_correction_atac / get_bc_signal_by_fragment_length).
"""
import sys, types, pickle
from math import floor, ceil
import numpy as np
from pysam import Samfile, Fastafile
from scipy.stats import scoreatpercentile
from rgt.Util import GenomeData, HmmData, AuxiliaryFunctions
from rgt.GenomicRegionSet import GenomicRegionSet
from rgt.HINT.signalProcessing import GenomicSignal
from rgt.HINT.hmm import _compute_log_likelihood
from rgt.HINT.biasTable import BiasTable

bam_path, peaks, model, out = sys.argv[1:5]
genome_data = GenomeData("tobias_chr4")
hmm_data = HmmData()
hmm_file = hmm_data.get_default_hmm_atac_paired() if model == "paired" else hmm_data.get_default_hmm_atac_single()
with open(hmm_file, "rb") as f:
    hmm = pickle.load(f, encoding="latin1")
hmm._compute_log_likelihood = types.MethodType(_compute_log_likelihood, hmm)
bias_table = BiasTable().load_table(table_file_name_F=hmm_data.get_default_bias_table_F_ATAC(),
                                    table_file_name_R=hmm_data.get_default_bias_table_R_ATAC())
fwd, rev = 5, -5
reads_file = GenomicSignal(bam_path)
reads_file.load_sg_coefs(9)
regions = GenomicRegionSet("regions"); regions.read(peaks)
bam = Samfile(bam_path, "rb")
fasta = Fastafile(genome_data.get_genome())
k_nb = len(list(bias_table[0].keys())[0])
window = 50

def kmer_bias(chrom, p1, p2, paired):
    # verbatim from bias_correction_atac (single) / get_bc_signal_by_fragment_length (paired)
    fBiasDict, rBiasDict = bias_table
    p1_w = p1 - (window // 2); p2_w = p2 + (window // 2)
    if paired:
        p1_wk = p1_w - int(k_nb / 2.); p2_wk = p2_w + int(k_nb / 2.)
    else:
        p1_wk = p1_w - int(floor(k_nb / 2.)); p2_wk = p2_w + int(ceil(k_nb / 2.))
    currStr = str(fasta.fetch(chrom, p1_wk, p2_wk - 1)).upper()
    currRevComp = AuxiliaryFunctions.revcomp(str(fasta.fetch(chrom, p1_wk + 1, p2_wk)).upper())
    af, ar = [], []
    for i in range(int(ceil(k_nb / 2.)), len(currStr) - int(floor(k_nb / 2)) + 1):
        fseq = currStr[i - int(floor(k_nb / 2.)):i + int(ceil(k_nb / 2.))]
        rseq = currRevComp[len(currStr) - int(ceil(k_nb / 2.)) - i:len(currStr) + int(floor(k_nb / 2.)) - i]
        af.append(fBiasDict.get(fseq, 1.0)); ar.append(rBiasDict.get(rseq, 1.0))
    return af, ar

def raw_counts(chrom, p1, p2, lo=None, hi=None):
    p1_w = p1 - (window // 2); p2_w = p2 + (window // 2)
    nf = [0.0] * (p2_w - p1_w); nr = [0.0] * (p2_w - p1_w)
    for read in bam.fetch(chrom, p1_w, p2_w):
        if lo is None and read.is_unmapped:
            continue
        tl = abs(read.template_length)
        if lo is not None and not tl > lo:
            continue
        if hi is not None and not tl <= hi:
            continue
        if not read.is_reverse:
            c = read.pos + fwd
            if p1_w <= c < p2_w: nf[c - p1_w] += 1.0
        else:
            c = read.aend + rev - 1
            if p1_w <= c < p2_w: nr[c - p1_w] += 1.0
    return nf, nr

store = {}
def put(key, arr):
    store.setdefault(key, []).append(np.asarray(arr, dtype=np.float64))

for r in regions:
    ch, s, e = r.chrom, r.initial, r.final
    af, ar = kmer_bias(ch, s, e, model == "paired")
    put("bias_f", af); put("bias_r", ar)
    if model == "single":
        nf, nr = raw_counts(ch, s, e)
        put("raw_f", nf); put("raw_r", nr)
        bf, br = reads_file.bias_correction_atac(bias_table, genome_data.get_genome(), ch, s, e, fwd, rev)
        put("bc_f", bf); put("bc_r", br)
        x = reads_file.get_signal_atac(ch, s, e, 1, 0, fwd, rev, 50, 98, 98, bias_table, genome_data.get_genome())
        X = np.array(list(x)).T
    else:
        for tag, lo, hi in (("max145", None, 145), ("min145", 145, None)):
            nf, nr = raw_counts(ch, s, e, lo, hi)
            put(f"raw_f_{tag}", nf); put(f"raw_r_{tag}", nr)
        try:
            f1, r1 = reads_file.get_bc_signal_by_fragment_length(ref=ch, start=s, end=e, bam=bam, fasta=fasta,
                bias_table=bias_table, forward_shift=fwd, reverse_shift=rev, min_length=None, max_length=145, strand=True)
            f2, r2 = reads_file.get_bc_signal_by_fragment_length(ref=ch, start=s, end=e, bam=bam, fasta=fasta,
                bias_table=bias_table, forward_shift=fwd, reverse_shift=rev, min_length=145, max_length=None, strand=True)
            ok = True
        except Exception:
            ok = False
        if ok:
            put("bc_f_max145", f1); put("bc_r_max145", r1); put("bc_f_min145", f2); put("bc_r_min145", r2)
            seq = []
            for sig in (f1, r1, f2, r2):
                sig = reads_file.boyle_norm(sig)
                perc = scoreatpercentile(sig, 98); std = np.array(sig).std()
                sig = reads_file.hon_norm_atac(sig, perc, std)
                seq += [sig, reads_file.slope(sig, reads_file.sg_coefs)]
            X = np.array(seq).T
        else:
            for k in ("bc_f_max145", "bc_r_max145", "bc_f_min145", "bc_r_min145"):
                put(k, [])
            X = np.zeros((0, 8))
    put("X", X.ravel())
    try:
        st = hmm.predict(X)
    except Exception:
        st = np.array([], dtype=int)
    put("states", st)

res = {}
for k, v in store.items():
    res[k] = np.concatenate(v) if v else np.zeros(0)
    res[k + "__len"] = np.array([len(a) for a in v])
np.savez_compressed(out, **res)
print("wrote", out, len(regions))
