"""Tests for epi.upstream.fastq_to_fragments and its pure-Python parts.

A tiny random genome, a barcode whitelist and simulated 10x read triplets
(R1 genomic, R2 barcode, R3 genomic) with known fragments. Barcode
correction, orientation detection, tagging, pairing, Tn5 shift, duplicate
removal and fragment writing run everywhere; the end-to-end aligner tests
skip when chromap / bwa-mem2 / bowtie2 / samtools are not installed.
"""
from __future__ import annotations

import gzip
import io
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pysam = pytest.importorskip("pysam")

from epione.upstream import (
    BarcodeCorrector,
    ReadLayout,
    barcode_translation_table,
    bam_to_fragments,
    detect_barcode_orientation,
    fastq_to_fragments,
    find_fastqs,
    load_whitelist,
    read_fragments,
)
from epione.upstream._barcodes import orient_barcode, revcomp
from epione.upstream._fragments import dedup_fragments, pair_to_fragment, write_fragments
from epione.upstream.scatac import (
    _BarcodeStats,
    _chromap_read_format,
    _count_barcodes,
    _resolve_layout,
    _write_tagged_pairs,
)

READ_LEN = 50


# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------
def _rand_seq(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def simulate(
    root: Path,
    *,
    n_fragments: int = 3000,
    n_cells: int = 20,
    chemistry: str = "atac",
    reverse_barcode: bool = False,
    lanes: int = 2,
    dup_rate: float = 0.15,
    bc_error_rate: float = 0.05,
    seed: int = 0,
):
    """Write genome.fa, whitelist.txt and Illumina-named FASTQs under ``root``.

    Returns ``(fastq_dir, fasta, whitelist, truth)``; ``truth`` maps
    ``(chrom, start, end, barcode) -> number of read pairs``, counting only
    pairs whose barcode is either exact or a single substitution at a
    low-quality base (the ones correction must recover).
    """
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    genome = {f"chr{i}": _rand_seq(rng, 40_000) for i in (1, 2, 3)}
    fasta = root / "genome.fa"
    with open(fasta, "w") as fh:
        for k, v in genome.items():
            fh.write(f">{k}\n")
            for i in range(0, len(v), 60):
                fh.write(v[i:i + 60] + "\n")
    whitelist = [_rand_seq(rng, 16) for _ in range(200)]
    wl_path = root / "whitelist.txt"
    wl_path.write_text("\n".join(whitelist) + "\n")
    cells = whitelist[:n_cells]

    fq_dir = root / "fastq"
    fq_dir.mkdir(exist_ok=True)
    spacer = "ACGTACGT"
    handles = {}
    for lane in range(1, lanes + 1):
        for role in ("R1", "R2", "R3"):
            handles[(lane, role)] = gzip.open(fq_dir / f"sim_S1_L00{lane}_{role}_001.fastq.gz", "wt")
    truth: Counter = Counter()
    k = 0
    for _ in range(n_fragments):
        chrom = rng.choice(sorted(genome))
        start = rng.randrange(200, 39_000 - 700)
        end = start + rng.randrange(60, 600)
        bc = rng.choice(cells)
        copies = 1 + (rng.random() < dup_rate) + (rng.random() < dup_rate / 3)
        for _c in range(copies):
            k += 1
            g = genome[chrom]
            fwd = g[start - 4:start - 4 + READ_LEN]
            rev = revcomp(g[end + 5 - READ_LEN:end + 5])
            r1, r3 = (fwd, rev) if rng.random() < 0.5 else (rev, fwd)
            obs = list(bc)
            qual = ["F"] * 16
            if rng.random() < bc_error_rate:
                p = rng.randrange(16)
                obs[p] = rng.choice([b for b in "ACGT" if b != obs[p]])
                qual[p] = "#"  # low quality where the error is
            obs, qual = "".join(obs), "".join(qual)
            # exact or correctable to bc (the neighbour must not be another
            # whitelist barcode, which with 200 random 16-mers it is not)
            truth[(chrom, start, end, bc)] += 1
            bc_read, bc_qual = obs, qual
            if chemistry == "multiome":
                bc_read, bc_qual = spacer + obs, "F" * 8 + qual
            if reverse_barcode:
                bc_read, bc_qual = revcomp(bc_read), bc_qual[::-1]
            lane = 1 + (k % lanes)
            name = f"@SIM:1:FC:{lane}:1:{k}:1"
            handles[(lane, "R1")].write(f"{name} 1:N:0:AAAA\n{r1}\n+\n{'F' * READ_LEN}\n")
            handles[(lane, "R2")].write(f"{name} 2:N:0:AAAA\n{bc_read}\n+\n{bc_qual}\n")
            handles[(lane, "R3")].write(f"{name} 3:N:0:AAAA\n{r3}\n+\n{'F' * READ_LEN}\n")
    # a few reads with junk barcodes (not correctable)
    for lane in range(1, lanes + 1):
        for j in range(20):
            k += 1
            name = f"@SIM:1:FC:{lane}:1:{k}:1"
            junk = _rand_seq(rng, 24 if chemistry == "multiome" else 16)
            g = genome["chr1"]
            handles[(lane, "R1")].write(f"{name} 1:N:0:AAAA\n{g[1000:1050]}\n+\n{'F' * READ_LEN}\n")
            handles[(lane, "R2")].write(f"{name} 2:N:0:AAAA\n{junk}\n+\n{'F' * len(junk)}\n")
            handles[(lane, "R3")].write(f"{name} 3:N:0:AAAA\n{revcomp(g[1200:1250])}\n+\n{'F' * READ_LEN}\n")
    for h in handles.values():
        h.close()
    return fq_dir, fasta, wl_path, truth


# ---------------------------------------------------------------------------
# barcodes
# ---------------------------------------------------------------------------
def test_revcomp_and_orient_barcode():
    assert revcomp("AACGTN") == "NACGTT"
    read = revcomp("ACGTACGT" + "TTTTCCCCGGGGAAAA")
    seq, qual = orient_barcode(read, "ABCDEFGHIJKLMNOPQRSTUVWX", "reverse", 8, 16)
    assert seq == "TTTTCCCCGGGGAAAA"
    assert qual == "ABCDEFGHIJKLMNOPQRSTUVWX"[::-1][8:24]
    assert orient_barcode("NNNNNNNNACGTACGTACGTACGT", "F" * 24, "forward", 8, 16)[0] == "ACGTACGTACGTACGT"


def test_load_whitelist_strips_suffix_and_columns(tmp_path):
    p = tmp_path / "wl.txt.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("# comment\nAAAACCCCGGGGTTTT-1\nACGTACGTACGTACGT,extra\n\nAAAACCCCGGGGTTTT\n")
    assert load_whitelist(p) == ["AAAACCCCGGGGTTTT", "ACGTACGTACGTACGT"]


def test_detect_orientation():
    rng = random.Random(1)
    wl = [_rand_seq(rng, 16) for _ in range(50)]
    fwd = [rng.choice(wl) for _ in range(500)]
    assert detect_barcode_orientation(fwd, wl)[0] == "forward"
    rev = [revcomp("ACGTACGT" + b) for b in fwd]
    o, f, r = detect_barcode_orientation(rev, wl, start=8, length=16)
    assert o == "reverse" and r == 1.0 and f < 0.05
    with pytest.raises(ValueError):
        detect_barcode_orientation([_rand_seq(rng, 16) for _ in range(200)], wl)


def test_corrector_exact_and_hamming1():
    wl = ["AAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCC", "AAAAAAAAAAAAAAAC"]
    c = BarcodeCorrector(wl)
    c.set_counts({"AAAAAAAAAAAAAAAA": 1000, "AAAAAAAAAAAAAAAC": 10})
    hi = "F" * 16
    assert c.correct("CCCCCCCCCCCCCCCC", hi) == "CCCCCCCCCCCCCCCC"  # exact, whatever the prior
    # one mismatch from C...C only
    assert c.correct("CCCCCCCCCCCCCCCA", hi) == "CCCCCCCCCCCCCCCC"
    # two whitelist neighbours: AAAA..AG is 1 away from both A..A and A..AC.
    # prior 1001 vs 11 at the same base -> posterior 0.989 >= 0.975
    bc, p = c.posterior("AAAAAAAAAAAAAAAG", hi)
    assert bc == "AAAAAAAAAAAAAAAA" and p == pytest.approx(1001 / 1012)
    assert c.correct("AAAAAAAAAAAAAAAG", hi) == "AAAAAAAAAAAAAAAA"
    # balanced prior -> ambiguous -> not corrected
    c.set_counts({"AAAAAAAAAAAAAAAA": 10, "AAAAAAAAAAAAAAAC": 10})
    assert c.correct("AAAAAAAAAAAAAAAG", hi) is None
    # two mismatches -> no candidate
    assert c.correct("CCCCCCCCCCCCCCAA", hi) is None
    # wrong length -> None
    assert c.correct("CCCC", "FFFF") is None


def test_corrector_quality_decides():
    # GTTT..T is one substitution from both ATTT..T (pos 0) and GTTT..A
    # (pos 15). Equal priors: the low-quality base wins.
    wl = ["ATTTTTTTTTTTTTTT", "GTTTTTTTTTTTTTTA"]
    c = BarcodeCorrector(wl)
    c.set_counts(np.array([100, 100]))
    seq = "GTTTTTTTTTTTTTTT"
    q_low0 = "#" + "F" * 15  # Q2 at position 0
    q_low15 = "F" * 15 + "#"
    assert c.correct(seq, q_low0) == "ATTTTTTTTTTTTTTT"
    assert c.correct(seq, q_low15) == "GTTTTTTTTTTTTTTA"
    assert c.correct(seq, "F" * 16) is None  # equal evidence -> 0.5


def test_corrector_n_base_and_count():
    wl = ["ACGTACGTACGTACGT", "TTTTTTTTTTTTTTTT"]
    c = BarcodeCorrector(wl)
    hits = c.count(["ACGTACGTACGTACGT"] * 5 + ["NOTABARCODE"] * 3)
    assert hits == 5 and c.counts.tolist() == [5, 0]
    assert c.correct("ACGTACGTNCGTACGT", "FFFFFFFF#FFFFFFF") == "ACGTACGTACGTACGT"
    assert c.correct("ACGTACGTNCGTNCGT", "F" * 16) is None


def test_translation_table(tmp_path):
    a = tmp_path / "a.txt"; a.write_text("AAAA\nCCCC\n")
    g = tmp_path / "g.txt"; g.write_text("GGGG\nTTTT\n")
    assert barcode_translation_table(a, g) == {"AAAA": "GGGG", "CCCC": "TTTT"}
    (tmp_path / "short.txt").write_text("GGGG\n")
    with pytest.raises(ValueError):
        barcode_translation_table(a, tmp_path / "short.txt")


# ---------------------------------------------------------------------------
# FASTQ discovery / layouts / tagging
# ---------------------------------------------------------------------------
def test_short_multiome_barcode_read_uses_last_16():
    from epione.upstream.scatac import _fit_layout

    assert _fit_layout(ReadLayout(bc_start=8), 16).bc_start == 0
    assert _fit_layout(ReadLayout(bc_start=8), 24).bc_start == 8
    with pytest.raises(ValueError):
        _fit_layout(ReadLayout(), 12)


def test_find_fastqs_and_layouts(tmp_path):
    for lane in ("L002", "L001"):
        for role in ("R1", "R2", "R3", "I1"):
            (tmp_path / f"pbmc_S1_{lane}_{role}_001.fastq.gz").write_bytes(b"")
    (tmp_path / "other_S2_L001_R1_001.fastq.gz").write_bytes(b"")
    with pytest.raises(ValueError):
        find_fastqs(tmp_path)
    fq = find_fastqs(tmp_path, "pbmc")
    assert sorted(fq) == ["I1", "R1", "R2", "R3"]
    assert [Path(p).name for p in fq["R1"]] == ["pbmc_S1_L001_R1_001.fastq.gz", "pbmc_S1_L002_R1_001.fastq.gz"]
    assert _resolve_layout("atac", list(fq)) == ReadLayout("R1", "R3", "R2", 0, 16)
    assert _resolve_layout("multiome", ["R1", "R2", "I2"]) == ReadLayout("R1", "R2", "I2", 8, 16)
    assert _resolve_layout({"barcode": "I1", "bc_length": 8}, []).bc_length == 8
    with pytest.raises(ValueError):
        _resolve_layout("nonsense", [])


def test_chromap_read_format():
    atac, mult = ReadLayout(), ReadLayout(bc_start=8)
    assert _chromap_read_format(atac, "forward", 16) == "bc:0:15"
    assert _chromap_read_format(atac, "reverse", 16) == "bc:0:15:-"
    assert _chromap_read_format(mult, "forward", 24) == "bc:8:23"
    # reverse 24-mer: whitelist-oriented [8,24) is raw [0,16) reverse-complemented
    assert _chromap_read_format(mult, "reverse", 24) == "bc:0:15:-"


def test_parse_chromap_log(tmp_path):
    from epione.upstream.scatac import _parse_chromap_log

    log = tmp_path / "chromap.log"
    log.write_text(
        "Number of reads: 200.\nNumber of mapped reads: 180.\n"
        "Number of reads have multi-mappings: 7.\nNumber of barcodes in whitelist: 90.\n"
        "Number of corrected barcodes: 4.\nNumber of output mappings (passed filters): 60\n"
    )
    assert _parse_chromap_log(str(log)) == {
        "reads": 200, "mapped_reads": 180, "barcodes_in_whitelist": 90,
        "corrected_barcodes": 4, "output_mappings": 60,
    }


@pytest.mark.parametrize("chemistry,reverse", [("atac", False), ("multiome", True)])
def test_tagged_pairs_correct_barcodes(tmp_path, chemistry, reverse):
    fq_dir, _, wl_path, truth = simulate(tmp_path, n_fragments=400, chemistry=chemistry,
                                         reverse_barcode=reverse, bc_error_rate=0.2, seed=3)
    fq = find_fastqs(fq_dir)
    layout = _resolve_layout(chemistry, list(fq))
    wl = load_whitelist(wl_path)
    from epione.upstream.scatac import _resolve_orientation

    orientation, _ = _resolve_orientation(layout, fq[layout.barcode], set(wl))
    assert orientation == ("reverse" if reverse else "forward")
    corr = BarcodeCorrector(wl)
    _count_barcodes(fq[layout.barcode], layout, orientation, corr)
    sink, stats = io.BytesIO(), _BarcodeStats()
    _write_tagged_pairs(fq, layout, orientation, corr, sink, stats)
    n_true = sum(truth.values())
    assert stats.pairs == n_true + 40
    assert stats.exact + stats.corrected == n_true
    assert stats.corrected > 0 and stats.invalid == 40
    lines = sink.getvalue().decode().splitlines()
    assert len(lines) == 8 * stats.pairs
    head = lines[0].split(" ", 1)[1].split("\t")
    assert head[0].startswith("CR:Z:") and head[1].startswith("CY:Z:")
    cbs = Counter(l.split("CB:Z:")[1] for l in lines[0::8] if "CB:Z:" in l)
    true_bc = Counter()
    for (_, _, _, bc), n in truth.items():
        true_bc[bc] += n
    assert cbs == true_bc


# ---------------------------------------------------------------------------
# pairs -> fragments
# ---------------------------------------------------------------------------
def _write_pair_bam(path, pairs, chroms=(("chr1", 10_000), ("chr2", 10_000))):
    """pairs: list of (name, chrom_id, left, right, cb, mapq1, mapq2, proper)."""
    header = {"HD": {"VN": "1.6", "SO": "unsorted"}, "SQ": [{"SN": c, "LN": n} for c, n in chroms]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, tid, left, right, cb, mq1, mq2, proper in pairs:
            for i, (pos, rev, mq) in enumerate(((left, False, mq1), (right - 30, True, mq2))):
                a = pysam.AlignedSegment(out.header)
                a.query_name = name
                a.query_sequence = "A" * 30
                a.query_qualities = pysam.qualitystring_to_array("F" * 30)
                a.flag = 0x1 | (0x2 if proper else 0) | (0x10 if rev else 0x20) | (0x40 if i == 0 else 0x80)
                a.reference_id = tid
                a.reference_start = pos
                a.mapping_quality = mq
                a.cigarstring = "30M"
                a.next_reference_id = tid
                a.next_reference_start = right - 30 if i == 0 else left
                a.template_length = (right - left) * (1 if i == 0 else -1)
                if cb is not None:
                    a.set_tag("CB", cb)
                out.write(a)


def test_bam_to_fragments_filters_shift_dedup(tmp_path):
    pairs = [
        ("p1", 0, 100, 400, "AAAA", 60, 60, True),
        ("p2", 0, 100, 400, "AAAA", 60, 60, True),   # duplicate of p1
        ("p3", 0, 100, 400, "CCCC", 60, 60, True),   # same position, other cell
        ("p4", 0, 500, 800, "AAAA", 60, 10, True),   # mate MAPQ < 30
        ("p5", 0, 900, 1200, "AAAA", 60, 60, False), # not a proper pair
        ("p6", 1, 50, 300, None, 60, 60, True),      # no CB
        ("p7", 1, 50, 3000, "AAAA", 60, 60, True),   # insert > 2000
        ("p8", 1, 20, 120, "GGGG", 30, 30, True),    # MAPQ exactly 30 passes
    ]
    bam = tmp_path / "x.bam"
    _write_pair_bam(bam, pairs)
    out = bam_to_fragments(bam, tmp_path / "fragments.tsv.gz")
    assert Path(out + ".tbi").exists()
    df = read_fragments(out)
    got = [tuple(r) for r in df.itertuples(index=False)]
    assert got == [
        ("chr1", 104, 395, "AAAA", 2),
        ("chr1", 104, 395, "CCCC", 1),
        ("chr2", 24, 115, "GGGG", 1),
    ]
    # bulk mode: one constant barcode, no MAPQ filter
    out2 = bam_to_fragments(bam, tmp_path / "bulk.tsv.gz", constant_barcode="bulk", min_mapq=0)
    df2 = read_fragments(out2)
    assert set(df2["barcode"]) == {"bulk"} and df2["count"].sum() == 6


def test_coordinate_sorted_bam_parallel_matches(tmp_path):
    rng = random.Random(5)
    pairs = []
    for i in range(300):
        left = rng.randrange(0, 9000)
        pairs.append((f"r{i}", rng.randrange(2), left, left + rng.randrange(60, 900),
                      rng.choice(["AAAA", "CCCC", "GGGG"]), 60, 60, True))
    pairs += [(f"d{i}", *p[1:]) for i, p in enumerate(pairs[:40])]
    bam = tmp_path / "x.bam"
    _write_pair_bam(bam, pairs)
    a = read_fragments(bam_to_fragments(bam, tmp_path / "a.tsv.gz"))
    pysam.sort("-o", str(tmp_path / "s.bam"), str(bam))
    pysam.index(str(tmp_path / "s.bam"))
    b = read_fragments(bam_to_fragments(tmp_path / "s.bam", tmp_path / "b.tsv.gz", threads=2))
    assert a.equals(b)
    assert a["count"].sum() == 340 and (a["count"] == 2).sum() >= 38


def test_parallel_pair_stats_match_single_pass(tmp_path):
    """Per-reference workers must give the single-pass counters, including
    pairs split across references, half-mapped and unplaced unmapped pairs."""
    from epione.upstream._fragments import collect_fragments

    chroms = (("chr1", 10_000), ("chr2", 10_000), ("chr3", 10_000))
    header = {"HD": {"VN": "1.6", "SO": "unsorted"}, "SQ": [{"SN": c, "LN": n} for c, n in chroms]}
    bam = tmp_path / "x.bam"

    def seg(out, name, first, tid, pos, rev, mtid, mpos, unmapped=False, mate_unmapped=False, proper=True):
        a = pysam.AlignedSegment(out.header)
        a.query_name = name
        a.query_sequence = "A" * 30
        a.query_qualities = pysam.qualitystring_to_array("F" * 30)
        a.flag = (0x1 | (0x2 if proper else 0) | (0x4 if unmapped else 0) | (0x8 if mate_unmapped else 0)
                  | (0x10 if rev else 0) | (0x40 if first else 0x80))
        a.reference_id, a.reference_start = tid, pos
        a.next_reference_id, a.next_reference_start = mtid, mpos
        if not unmapped:
            a.mapping_quality, a.cigarstring = 60, "30M"
        a.set_tag("CB", "AAAA")
        out.write(a)

    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        for i in range(50):                      # ordinary pairs on chr1..chr3
            tid, left = i % 3, 100 + 37 * i
            seg(out, f"p{i}", True, tid, left, False, tid, left + 200)
            seg(out, f"p{i}", False, tid, left + 200, True, tid, left)
        for i in range(4):                       # mates on different references
            seg(out, f"x{i}", True, 0, 3000 + i, False, 1, 4000, proper=False)
            seg(out, f"x{i}", False, 1, 4000, True, 0, 3000 + i, proper=False)
        for i in range(3):                       # one mate unmapped, placed at its mate
            seg(out, f"h{i}", True, 1, 5000 + i, False, 1, 5000 + i, mate_unmapped=True, proper=False)
            seg(out, f"h{i}", False, 1, 5000 + i, False, 1, 5000 + i, unmapped=True, proper=False)
        for i in range(5):                       # both unmapped, no coordinate
            seg(out, f"u{i}", True, -1, -1, False, -1, -1, unmapped=True, mate_unmapped=True, proper=False)
            seg(out, f"u{i}", False, -1, -1, False, -1, -1, unmapped=True, mate_unmapped=True, proper=False)
    pysam.sort("-o", str(tmp_path / "s.bam"), str(bam))
    pysam.index(str(tmp_path / "s.bam"))
    out_chroms = ["chr1", "chr2"]                 # chr3 counted but not output
    f1, _, b1, s1 = collect_fragments(tmp_path / "s.bam", chroms=out_chroms, threads=1)
    f2, _, b2, s2 = collect_fragments(tmp_path / "s.bam", chroms=out_chroms, threads=3)
    assert s1.as_dict() == s2.as_dict()
    assert s1.pairs == 62 and s1.both_mapped == 54
    for k in ("chrom", "start", "end", "count"):
        np.testing.assert_array_equal(f1[k], f2[k])
    np.testing.assert_array_equal(np.asarray(b1)[f1["barcode"]], np.asarray(b2)[f2["barcode"]])


def test_fragment_ends_are_mate_5prime_ends(tmp_path):
    """A 50-bp fragment read with 60-bp reads: each mate runs into adapter and
    the aligner extends a few bases past the other mate's 5' end. The
    fragment is still bounded by the two 5' ends."""
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]}
    bam = tmp_path / "rt.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        # forward mate: 5' end at 1000, aligned 1000-1053 (3 bp past 1050)
        # reverse mate: 5' end at 1050, aligned 998-1050 (2 bp before 1000)
        for i, (pos, n, rev) in enumerate(((1000, 53, False), (998, 52, True))):
            a = pysam.AlignedSegment(out.header)
            a.query_name = "rt"
            a.query_sequence = "A" * 60
            a.query_qualities = pysam.qualitystring_to_array("F" * 60)
            a.flag = 0x1 | 0x2 | (0x10 if rev else 0x20) | (0x40 if i == 0 else 0x80)
            a.reference_id = 0
            a.reference_start = pos
            a.mapping_quality = 60
            a.cigarstring = f"{n}M{60 - n}S" if not rev else f"{60 - n}S{n}M"
            a.next_reference_id = 0
            a.next_reference_start = 998 if i == 0 else 1000
            a.set_tag("CB", "AAAA")
            out.write(a)
    r1, r2 = list(pysam.AlignmentFile(str(bam)))
    assert pair_to_fragment(r1, r2) == (0, 1004, 1045)


def test_fragment_ends_use_unclipped_5prime(tmp_path):
    """Soft clips at a read's 5' end still count (Cell Ranger, SnapATAC2);
    soft clips at the 3' end do not."""
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10_000}]}
    bam = tmp_path / "clip.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        # forward: 3S47M at 1003 -> 5' end 1000; reverse: 45M5S at 1350 -> 5' end 1400
        for i, (pos, cig, rev) in enumerate(((1003, "3S47M", False), (1350, "45M5S", True))):
            a = pysam.AlignedSegment(out.header)
            a.query_name = "c"
            a.query_sequence = "A" * 50
            a.query_qualities = pysam.qualitystring_to_array("F" * 50)
            a.flag = 0x1 | 0x2 | (0x10 if rev else 0x20) | (0x40 if i == 0 else 0x80)
            a.reference_id = 0
            a.reference_start = pos
            a.mapping_quality = 60
            a.cigarstring = cig
            a.next_reference_id = 0
            a.next_reference_start = 1350 if i == 0 else 1003
            out.write(a)
    r1, r2 = list(pysam.AlignmentFile(str(bam)))
    assert pair_to_fragment(r1, r2) == (0, 1004, 1395)


def test_dedup_sorts_by_barcode_string():
    c = np.array([0, 0, 0, 0]); s = np.array([5, 5, 5, 1]); e = np.array([9, 9, 9, 3])
    b = np.array([0, 1, 0, 0])
    d = dedup_fragments(c, s, e, b, barcode_names=["ZZ", "AA"])
    assert d["start"].tolist() == [1, 5, 5]
    assert d["barcode"].tolist() == [0, 1, 0]  # "AA" (id 1) before "ZZ" (id 0) at pos 5
    assert d["count"].tolist() == [1, 1, 2]


def test_write_fragments_translation_and_tabix(tmp_path):
    frags = dedup_fragments(np.array([0, 0, 1]), np.array([10, 10, 5]), np.array([50, 50, 40]),
                            np.array([0, 1, 0]), barcode_names=["AAAA", "CCCC"])
    out = write_fragments(frags, ["chr1", "chr2"], ["AAAA", "CCCC"], tmp_path / "f.tsv.gz",
                          barcode_translation={"AAAA": "GGGG-1", "CCCC": "GGGG-1"})
    df = read_fragments(out)
    assert [tuple(r) for r in df.itertuples(index=False)] == [("chr1", 10, 50, "GGGG-1", 2), ("chr2", 5, 40, "GGGG-1", 1)]
    tbx = pysam.TabixFile(out)
    assert len(list(tbx.fetch("chr2", 0, 100))) == 1


def test_pair_to_fragment_needs_opposite_strands(tmp_path):
    bam = tmp_path / "x.bam"
    _write_pair_bam(bam, [("p", 0, 100, 400, "AAAA", 60, 60, True)])
    r1, r2 = list(pysam.AlignmentFile(str(bam)))
    assert pair_to_fragment(r1, r2) == (0, 104, 395)
    r2.is_reverse = False
    assert pair_to_fragment(r1, r2) is None


# ---------------------------------------------------------------------------
# end to end (aligners)
# ---------------------------------------------------------------------------
def _have(*tools):
    from epione.upstream._env import check_tools

    return all(check_tools(tools, verbose=False).values())


def _truth_frame(truth):
    import pandas as pd

    rows = [(c, s, e, b, n) for (c, s, e, b), n in truth.items()]
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "barcode", "count"])


@pytest.mark.parametrize(
    "aligner,tools",
    [("chromap", ("chromap",)), ("bwa-mem2", ("bwa-mem2", "samtools")),
     ("bowtie2", ("bowtie2", "bowtie2-build", "samtools", "fastp"))],
)
@pytest.mark.parametrize("chemistry,reverse", [("atac", False), ("multiome", True)])
def test_fastq_to_fragments_end_to_end(tmp_path, aligner, tools, chemistry, reverse):
    if not _have(*tools):
        pytest.skip(f"{aligner} not installed")
    fq_dir, fasta, wl, truth = simulate(tmp_path / "sim", chemistry=chemistry, reverse_barcode=reverse, seed=11)
    res = fastq_to_fragments(fastq_dir=fq_dir, reference=fasta, whitelist=wl, aligner=aligner,
                             chemistry=chemistry, outdir=tmp_path / aligner, threads=2, keep_bam=True)
    got = read_fragments(res)
    exp = _truth_frame(truth)
    key = ["chrom", "start", "end", "barcode"]
    m = exp.merge(got, on=key, how="left", suffixes=("", "_got"))
    recovered = m["count_got"].notna().mean()
    assert recovered >= 0.98, recovered
    hit = m[m["count_got"].notna()]
    assert (hit["count"] == hit["count_got"]).mean() >= 0.98
    assert len(got.merge(exp, on=key)) / len(got) >= 0.98  # few spurious fragments
    qc = res.qc
    assert qc["barcode_orientation"] == ("reverse" if reverse else "forward")
    n_pairs = sum(truth.values()) + 40
    assert qc["n_read_pairs"] == n_pairs
    assert qc["frac_valid_barcode"] == pytest.approx(sum(truth.values()) / n_pairs, abs=0.01)
    assert 0.05 < qc["frac_duplicate"] < 0.3
    assert Path(res.barcode_metrics).exists() and len(res.barcodes()) == 20
    if aligner != "chromap":
        assert res.bam and Path(res.bam + ".bai").exists()
        r = next(iter(pysam.AlignmentFile(res.bam)))
        assert r.has_tag("CR")
        # the shared BAM -> fragments step reproduces the pipeline output
        again = read_fragments(bam_to_fragments(res.bam, tmp_path / "again.tsv.gz", threads=2))
        assert again.equals(got)


def test_multiome_translation_end_to_end(tmp_path):
    if not _have("chromap"):
        pytest.skip("chromap not installed")
    fq_dir, fasta, wl, truth = simulate(tmp_path / "sim", chemistry="multiome", n_fragments=500, seed=2)
    atac = load_whitelist(wl)
    gex = tmp_path / "gex.txt"
    gex.write_text("\n".join(f"G{b[1:]}" for b in atac) + "\n")
    res = fastq_to_fragments(fastq_dir=fq_dir, reference=fasta, whitelist=wl, chemistry="multiome",
                             outdir=tmp_path / "out", threads=1, barcode_translation=(wl, gex))
    got = read_fragments(res)
    assert got["barcode"].str.startswith("G").all()
    assert set(res.barcodes()["barcode"]) <= {f"G{b[1:]}" for b in atac}
