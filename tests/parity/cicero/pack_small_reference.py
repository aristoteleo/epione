"""Pack the small committed reference (fixture F1 + a subset of the R 4.3.3 / cicero 1.3.9
outputs, seed 42) used by tests/test_cicero.py, from r_reference_driver.R dumps.
Usage: python pack_small_reference.py <fixture_F1_dir> <ref_F1_seed42_dir> <out.npz>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

fx, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sh = [int(x) for x in (fx / "shape.txt").read_text().split()]
d = {}
d["x_i"] = np.fromfile(fx / "x.i.bin", "<i4")
d["x_p"] = np.fromfile(fx / "x.p.bin", "<f8").astype(np.int64)
d["x_x"] = np.fromfile(fx / "x.x.bin", "<f8")
d["shape"] = np.array(sh[:2])
d["peaks"] = np.array((fx / "peaks.txt").read_text().split())
d["cells"] = np.array((fx / "cells.txt").read_text().split())
co = pd.read_csv(fx / "coords.csv", index_col=0)
d["coords"] = co.loc[d["cells"]].to_numpy()
g = pd.read_csv(fx / "genome.tsv", sep="\t", header=None)
d["genome_chr"] = g[0].to_numpy().astype(str)
d["genome_len"] = g[1].to_numpy()
pr = pd.read_csv(fx / "promoters.tsv", sep="\t")
for c in pr.columns:
    d["prom_" + c] = pr[c].to_numpy() if c != "gene" else pr[c].astype(str).to_numpy()
d["r_agg_cells"] = np.array((ref / "agg_cells.txt").read_text().split())
sw = pd.read_csv(ref / "sampled_windows.tsv", sep="\t")
d["r_sampled_win"] = sw["win"].to_numpy()
d["r_dps"] = np.fromfile(ref / "distance_parameters.f64", "<f8")
d["r_mean_dp"] = np.fromfile(ref / "mean_distance_parameter.f64", "<f8")
p1 = (ref / "conns_peak1.txt").read_text().split("\n")
p2 = (ref / "conns_peak2.txt").read_text().split("\n")
cv = np.fromfile(ref / "conns_coaccess.f64", "<f8")
d["r_n_conns"] = np.array([len(cv)])
sel = np.arange(0, len(cv), 97)
d["r_conn_rows"] = sel
d["r_conn_p1"] = np.array(p1)[sel]
d["r_conn_p2"] = np.array(p2)[sel]
d["r_conn_co"] = cv[sel]
for tag in ("syn",):
    t = pd.read_csv(ref / "synthetic.tsv", sep="\t")
    d["syn_n"] = t["n"].to_numpy()
    d["syn_off"] = t["offset"].to_numpy()
    for m in ("S", "rho", "w", "wi"):
        d[f"syn_{m}"] = np.fromfile(ref / f"syn_{m}.f64", "<f8")
ann = pd.read_csv(ref / "annotated_sites.tsv", sep="\t", keep_default_na=False, na_values=["NA"])
d["r_ann_gene"] = ann["gene"].fillna("").astype(str).to_numpy()
for nm in ("unnorm_ga", "ga"):
    genes = [x for x in (ref / f"{nm}_genes.txt").read_text().split("\n") if x]
    cells = [x for x in (ref / f"{nm}_cells.txt").read_text().split("\n") if x]
    M = np.fromfile(ref / f"{nm}.f64", "<f8").reshape(len(genes), len(cells), order="F")
    gi = np.arange(0, len(genes), 12)
    d[f"r_{nm}_genes"] = np.array(genes)[gi]
    d[f"r_{nm}_cells"] = np.array(cells)
    d[f"r_{nm}"] = M[gi]
cc = pd.read_csv(ref / "ccan_cutoffs.tsv", sep="\t")
d["r_ccan_cutoff"] = cc["cutoff"].to_numpy()[:1]
c42 = pd.read_csv(ref / "ccan_seed42.tsv", sep="\t")
d["r_ccan42_peak"] = c42["Peak"].astype(str).to_numpy()
d["r_ccan42_ccan"] = c42["CCAN"].to_numpy()
d = {k: (v.astype(str) if getattr(v, 'dtype', None) == object else v) for k, v in d.items()}
np.savez_compressed(out, **d)
print(out, {k: v.shape for k, v in d.items() if hasattr(v, "shape")})
