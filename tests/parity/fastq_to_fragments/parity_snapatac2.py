"""Gate G7 of MANIFEST.md: Epione bam_to_fragments vs SnapATAC2 make_fragment_file.

Two steps, because SnapATAC2 lives in its own environment:

    # in the SnapATAC2 env
    python parity_snapatac2.py snap BAM OUT_SNAP.tsv.gz
    # in the Epione env
    python parity_snapatac2.py compare BAM OUT_SNAP.tsv.gz OUT_JSON THREADS
"""
import json
import sys
import time


def snap(bam, out):
    import snapatac2 as snap

    t0 = time.time()
    snap.pp.make_fragment_file(bam, out, is_paired=True, barcode_tag="CB",
                               shift_left=4, shift_right=-5, min_mapq=30)
    print(json.dumps({"snapatac2": snap.__version__, "seconds": round(time.time() - t0, 1)}))


def compare(bam, snap_out, out_json, threads):
    import numpy as np
    import pandas as pd
    from epione.upstream import bam_to_fragments, read_fragments

    t0 = time.time()
    ep = bam_to_fragments(bam, out_json.replace(".json", ".epione.tsv.gz"), max_fragment_length=None,
                          threads=int(threads))
    t_ep = time.time() - t0
    a = read_fragments(ep)
    b = pd.read_csv(snap_out, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3, 4],
                    names=["chrom", "start", "end", "barcode", "count"])
    key = ["chrom", "start", "end", "barcode"]
    m = a.merge(b, on=key, how="outer", suffixes=("_ep", "_snap"), indicator=True)
    both = m["_merge"] == "both"
    res = {
        "n_epione": int(len(a)), "n_snapatac2": int(len(b)),
        "n_snapatac2_duplicate_keys": int(b.duplicated(key).sum()),
        "G7a_record_jaccard": float(both.sum() / len(m)),
        "G7b_equal_count_among_shared": float((m.loc[both, "count_ep"] == m.loc[both, "count_snap"]).mean()),
        "only_epione": int((m["_merge"] == "left_only").sum()),
        "only_snapatac2": int((m["_merge"] == "right_only").sum()),
        "seconds_epione": round(t_ep, 1),
    }
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"snap": snap, "compare": compare}[sys.argv[1]](*sys.argv[2:])
