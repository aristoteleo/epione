"""Gate G1 of MANIFEST.md: per-read barcode extraction and correction.

usage: python parity_barcodes.py CELLRANGER_BAM FASTQ_DIR SAMPLE WHITELIST CHEMISTRY OUT_JSON

Runs Epione's barcode pass exactly as inside ``fastq_to_fragments`` for the
bwa-mem2 / bowtie2 aligners (orientation detection, prior from exact
whitelist hits of the whole run, Hamming-1 posterior correction) and
compares the oriented raw barcode and the corrected barcode with the ``CR``
and ``CB`` tags Cell Ranger wrote for the same read name (primary read-1
records of its BAM).
"""
from __future__ import annotations

import json
import sys
import time
from array import array

import numpy as np

from epione.upstream._barcodes import BarcodeCorrector, load_whitelist, orient_barcode
from epione.upstream.scatac import (
    _count_barcodes, _FastqStream, _fit_layout, _read_name, _resolve_layout,
    _resolve_orientation, _sample_barcode_reads, find_fastqs,
)

_ENC = np.full(256, 4, dtype=np.int64)
for i, b in enumerate(b"ACGT"):
    _ENC[b] = i


def enc(seq: str) -> int:
    """Base-5 code of a barcode (N and others -> 4); -1 for none."""
    if not seq:
        return -1
    v = 0
    for ch in seq.encode():
        v = v * 5 + int(_ENC[ch])
    return v


def main():
    bam, fq_dir, sample, wl_path, chemistry, out = sys.argv[1:7]
    import pysam

    t0 = time.time()
    # --- Cell Ranger side -------------------------------------------------
    names, cr_raw, cr_cb = array('q'), array('q'), array('q')
    with pysam.AlignmentFile(bam, threads=4) as fh:
        for r in fh.fetch(until_eof=True):
            if r.is_secondary or r.is_supplementary or not r.is_read1:
                continue
            names.append(hash(r.query_name.encode()))
            cr_raw.append(enc(r.get_tag("CR")) if r.has_tag("CR") else -1)
            cb = r.get_tag("CB").split("-")[0] if r.has_tag("CB") else ""
            cr_cb.append(enc(cb))
    names = np.frombuffer(names, dtype=np.int64)
    order = np.argsort(names, kind="stable")
    names, cr_raw, cr_cb = names[order], np.frombuffer(cr_raw, dtype=np.int64)[order], np.frombuffer(cr_cb, dtype=np.int64)[order]
    t_bam = time.time() - t0

    # --- Epione side --------------------------------------------------------
    fq = find_fastqs(fq_dir, sample)
    layout = _resolve_layout(chemistry, list(fq))
    layout = _fit_layout(layout, len(_sample_barcode_reads(fq[layout.barcode], n=1)[0]))
    wl = load_whitelist(wl_path)
    orientation, info = _resolve_orientation(layout, fq[layout.barcode], set(wl))
    corr = BarcodeCorrector(wl)
    _count_barcodes(fq[layout.barcode], layout, orientation, corr)
    e_names, e_raw, e_cb, e_exact = array('q'), array('q'), array('q'), array('b')
    for f in fq[layout.barcode]:
        st = _FastqStream(f)
        try:
            for h, s, q in st:
                seq, qual = orient_barcode(s.decode(), q.decode(), orientation, layout.bc_start, layout.bc_length)
                bc = corr.correct(seq, qual)
                e_names.append(hash(_read_name(h)))
                e_raw.append(enc(seq))
                e_cb.append(enc(bc) if bc else -1)
                e_exact.append(seq in corr.index)
        finally:
            st.close()
    e_names = np.frombuffer(e_names, dtype=np.int64)
    e_raw, e_cb = np.frombuffer(e_raw, dtype=np.int64), np.frombuffer(e_cb, dtype=np.int64)
    e_exact = np.frombuffer(e_exact, dtype=np.int8).astype(bool)

    # --- join ---------------------------------------------------------------
    j = np.searchsorted(names, e_names)
    j = np.clip(j, 0, names.size - 1)
    hit = names[j] == e_names
    jj = j[hit]
    raw_eq = cr_raw[jj] == e_raw[hit]
    cb_eq = cr_cb[jj] == e_cb[hit]
    off_wl = ~e_exact[hit]
    res = {
        "orientation": orientation, **info, "layout": vars(layout),
        "n_reads_fastq": int(e_names.size), "n_reads_cellranger_bam": int(names.size),
        "n_joined": int(hit.sum()), "n_fastq_missing_in_bam": int((~hit).sum()),
        "G1a_raw_barcode_agreement": float(raw_eq.mean()),
        "G1b_corrected_barcode_agreement": float(cb_eq.mean()),
        "G1c_agreement_off_whitelist": float(cb_eq[off_wl].mean()) if off_wl.any() else None,
        "n_off_whitelist": int(off_wl.sum()),
        "frac_valid_epione": float((e_cb[hit] >= 0).mean()),
        "frac_valid_cellranger": float((cr_cb[jj] >= 0).mean()),
        "disagree_epione_only": int(((e_cb[hit] >= 0) & (cr_cb[jj] < 0)).sum()),
        "disagree_cellranger_only": int(((e_cb[hit] < 0) & (cr_cb[jj] >= 0)).sum()),
        "disagree_both_assigned": int(((e_cb[hit] >= 0) & (cr_cb[jj] >= 0) & ~cb_eq).sum()),
        "seconds_bam": round(t_bam, 1), "seconds_total": round(time.time() - t0, 1),
    }
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
