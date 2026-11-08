"""Inputs for t_tf_binding.ipynb: ENCODE K562 snATAC (chr19) + six ENCODE K562 ChIP-seq peak sets.

Usage::

    python t_tf_binding_data.py OUT_DIR [--fragments A.tsv.gz B.tsv.gz] [--fasta GRCh38.fa]
                                        [--chip-dir DIR] [--jaspar-dir DIR]

Without the optional arguments everything is downloaded (ENCODE: no use restrictions;
JASPAR 2024: CC BY 4.0): the two K562 snATAC fragment archives (2.4 + 2.2 GB, only chr19 is
kept), the six ChIP files and JASPAR matrices. ``--fasta`` must be a GRCh38 FASTA with
``chr`` names (e.g. GENCODE v41 primary assembly); only chr19 is written out.

Writes OUT_DIR/{fragments.chr19.tsv.gz(.tbi), cells.txt, peaks.chr19.bed, chr19.fa(.fai),
motifs.jaspar, chip/<TF>.bed}.
"""
import argparse, bisect, gzip, os, subprocess, tarfile, urllib.request
from collections import Counter

import numpy as np
import pandas as pd

ENC = "https://www.encodeproject.org/files/{0}/@@download/{0}.{1}"
RUNS = {"ENCSR332SXG": "ENCFF504ASC", "ENCSR308ZGJ": "ENCFF794UXO"}      # K562 snATAC (Snyder lab)
TFS = {  # TF: (JASPAR 2024 matrix, ENCODE conservative IDR peaks, GRCh38)
    "CTCF": ("MA0139.2", "ENCFF901CBP"), "GABPA": ("MA0062.4", "ENCFF889XXJ"),
    "MAFK": ("MA0496.4", "ENCFF188OEK"), "NRF1": ("MA0506.3", "ENCFF777PKJ"),
    "REST": ("MA0138.3", "ENCFF649HPP"), "USF1": ("MA0093.4", "ENCFF913XQO"),
}
CHROM, MIN_FRAGS, HALF = "chr19", 40, 500   # 1-kb peaks, as scPrinter uses


def fetch(url, path):
    if not os.path.exists(path):
        print("download", url, flush=True)
        urllib.request.urlretrieve(url, path + ".part"); os.replace(path + ".part", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--fragments", nargs=2); ap.add_argument("--fasta", required=True)
    ap.add_argument("--chip-dir"); ap.add_argument("--jaspar-dir")
    a = ap.parse_args()
    import pysam
    os.makedirs(f"{a.out}/chip", exist_ok=True)
    # 1. chr19 fragments of both runs, barcodes prefixed with the run, cells >= MIN_FRAGS chr19 fragments
    frag_files = a.fragments
    if frag_files is None:
        frag_files = []
        for run, acc in RUNS.items():
            tgz = fetch(ENC.format(acc, "tar.gz"), f"{a.out}/{acc}.tar.gz")
            with tarfile.open(tgz) as t:
                for m in t.getmembers():
                    if m.name.endswith("fragments.tsv.gz") or m.name.endswith(".tbi"):
                        m.name = f"{run}." + os.path.basename(m.name); t.extract(m, a.out)
            frag_files.append(f"{a.out}/{run}.fragments.tsv.gz")
    rows = []
    for run, f in zip(RUNS, frag_files):
        for line in pysam.TabixFile(f).fetch(CHROM):
            p = line.split("\t")
            rows.append((int(p[1]), int(p[2]), f"{run}#{p[3]}"))
    df = pd.DataFrame(rows, columns=["start", "end", "bc"])
    n = df.bc.value_counts(); cells = sorted(n.index[n >= MIN_FRAGS])
    df = df[df.bc.isin(set(cells))].sort_values(["start", "end", "bc"])
    raw = f"{a.out}/fragments.chr19.tsv"
    df.assign(chrom=CHROM, n=1)[["chrom", "start", "end", "bc", "n"]].to_csv(raw, sep="\t", header=False, index=False)
    pysam.tabix_compress(raw, raw + ".gz", force=True); os.remove(raw)
    pysam.tabix_index(raw + ".gz", preset="bed", force=True)
    open(f"{a.out}/cells.txt", "w").write("\n".join(cells) + "\n")
    print(f"{len(cells)} cells, {len(df):,} chr19 fragments", flush=True)
    # 2. peaks: MACS3 on Tn5 insertions, summit +- 500 bp, overlaps resolved by the stronger summit
    ins = f"{a.out}/insertions.bed"
    with open(ins, "w") as fh:
        for s, e in zip(df.start.values, df.end.values):
            fh.write(f"{CHROM}\t{s}\t{s + 1}\n{CHROM}\t{e - 1}\t{e}\n")
    subprocess.run(["macs3", "callpeak", "-t", ins, "-f", "BED", "-g", "hs", "-n", "k562_chr19", "--nomodel",
                    "--shift", "-75", "--extsize", "150", "--keep-dup", "all", "--outdir", f"{a.out}/macs3"], check=True)
    os.remove(ins)
    sm = pd.read_csv(f"{a.out}/macs3/k562_chr19_summits.bed", sep="\t", header=None, names=["chrom", "s", "e", "name", "q"])
    sm = sm.sort_values(["q", "s"], ascending=[False, True]); kept = []
    for s in sm.s.values:                       # strongest summit first; keep if no kept summit within 2*HALF
        j = bisect.bisect_left(kept, s)
        if s - HALF > 0 and (j == len(kept) or kept[j] - s >= 2 * HALF) and (j == 0 or s - kept[j - 1] >= 2 * HALF):
            kept.insert(j, s)
    pk = pd.DataFrame({"chrom": CHROM, "start": np.sort(kept) - HALF, "end": np.sort(kept) + HALF})
    pk.to_csv(f"{a.out}/peaks.chr19.bed", sep="\t", header=False, index=False)
    print(f"{len(pk):,} peaks", flush=True)
    # 3. chr19 sequence
    fa = pysam.FastaFile(a.fasta); seq = fa.fetch(CHROM)
    with open(f"{a.out}/chr19.fa", "w") as fh:
        fh.write(f">{CHROM}\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    pysam.faidx(f"{a.out}/chr19.fa")
    # 4. motifs and ChIP peaks
    with open(f"{a.out}/motifs.jaspar", "w") as out:
        for tf, (mid, acc) in TFS.items():
            jf = (f"{a.jaspar_dir}/{mid}.jaspar" if a.jaspar_dir else
                  fetch(f"https://jaspar.elixir.no/api/v1/matrix/{mid}.jaspar", f"{a.out}/{mid}.jaspar"))
            lines = open(jf).read().strip().splitlines()
            out.write(f">{mid}\t{tf}\n" + "\n".join(lines[1:]) + "\n\n")
            cf = (f"{a.chip_dir}/{acc}.bed.gz" if a.chip_dir else fetch(ENC.format(acc, "bed.gz"), f"{a.out}/{acc}.bed.gz"))
            with gzip.open(cf, "rt") as fh, open(f"{a.out}/chip/{tf}.bed", "w") as o:
                o.writelines(l for l in fh if l.split("\t", 1)[0] == CHROM)
    print("done", a.out)


if __name__ == "__main__":
    main()
