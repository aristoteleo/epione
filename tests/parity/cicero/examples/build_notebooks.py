"""Build the omicverse-rebuildr notebooks of the Cicero port (NOTEBOOKS.md):

  compare_R_vs_Python.ipynb             Notebook 1 - pipeline parity
  function_by_function_R_parity.ipynb   Notebook 3 - R <-> Python dictionary
  evolution.ipynb                       Notebook 4 - one block per iteration

Notebook 2 (tutorial) is epione_guide/tutorials/single/t_cicero.ipynb.
Execute (on a node with the dumps):
  export EPIONE_CICERO_PARITY_DIR=<dir with fixtures/, ref/, results/>  R_TEST_ENV=<R env prefix>
  jupyter nbconvert --to notebook --execute --inplace <notebook>
"""
import pathlib

import nbformat as nbf

HERE = pathlib.Path(__file__).parent


def new():
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

    def code(s):
        cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    return cells, md, code


def save(cells, name):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    (HERE / name).write_text(nbf.writes(nb))
    print("wrote", name)


SETUP = r"""
import os, sys, json, time, subprocess, tempfile
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "8")
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml
import epione as epi
from epione.tl import _cicero as C
from epione.tl._rrng import RRandom
HERE = Path(".").resolve()                       # tests/parity/cicero/examples
PARITY = HERE.parent                             # tests/parity/cicero
sys.path.insert(0, str(PARITY))
import parity_check as P, parity_utils as U
D = Path(os.environ["EPIONE_CICERO_PARITY_DIR"])  # fixtures/, ref/, results/
manifest = yaml.safe_load((PARITY / "manifest.yaml").read_text())
"""

# =============================================================================== Notebook 1
cells, md, code = new()
md(r"""
# Cicero → `epione.tl` — R parity validation

Pipeline-level comparison of the epione port with R **cicero 1.3.9** (Monocle 3 branch,
cole-trapnell-lab/cicero-release@495ef0d), monocle3 1.3.1, glasso 1.11, R 4.3.3, following the
6-section schema of omicverse-rebuildr `NOTEBOOKS.md`. Gates and thresholds are those
pre-registered in `../manifest.yaml` / `../MANIFEST.md` before the port was written.

Fixtures: **F1** Cicero's `cicero_data` (200 cells, hg19 chr18); **F2** E8.5 heart multiome ATAC,
chr19 (16,787 cells × 8,927 peaks); **F3** the same data, whole genome (331,372 peaks), runtime.
""")
md("## 1. Setup")
code(SETUP + r"""
gates = pd.DataFrame(manifest["outputs"] + manifest["addendum_1_outputs"])[["id", "name", "fixtures", "mode", "class", "threshold"]]
print("upstream:", manifest["upstream"]["name"], manifest["upstream"]["version"], manifest["upstream"]["commit"][:7], "| seed", manifest["seed"])
gates
""")
md(r"""
## 2. R reference run

The full reference dumps (all intermediates, glasso probes, per-window distance parameters,
CCANs for five seeds) were produced by `../r_reference_driver.R` in `full` mode and are read from
`$EPIONE_CICERO_PARITY_DIR/ref`. Here R is re-run on F1 in `light` mode to show the call and
check that the stored dump is reproducible.
""")
code(r"""
rs = Path(os.environ["R_TEST_ENV"]) / "bin" / "Rscript"
tmp = Path(tempfile.mkdtemp())
t0 = time.perf_counter()
r = subprocess.run([str(rs), str(PARITY / "r_reference_driver.R"), str(D / "fixtures/F1"), str(tmp), "42", "light"],
                   capture_output=True, text=True)
r_wall = time.perf_counter() - t0
print((tmp / "versions.txt").read_text())
print(f"R wall-clock (read + make_cicero_cds + run_cicero steps): {r_wall:.1f} s")
same = all((tmp / f).read_bytes() == (D / "ref/F1_seed42" / f).read_bytes()
           for f in ("conns_peak1.txt", "conns_peak2.txt", "conns_coaccess.f64", "agg_cells.txt"))
print("re-run identical to the stored F1 dump:", same)
""")
md(r"""
## 3. Python candidate run

`parity_check.main` runs the port on a fixture (seed 42, R's RNG emulated) and evaluates every
gate against the R dump; `parity_utils.main` covers the exported utilities (gates 19–24).
""")
code(r"""
t0 = time.perf_counter(); res1 = P.main([str(D / "fixtures/F1"), str(D / "ref/F1_seed42"), str(tmp / "F1.json"), "--threads", "8"]); w1 = time.perf_counter() - t0
t0 = time.perf_counter(); res2 = P.main([str(D / "fixtures/F2"), str(D / "ref/F2_seed42"), str(tmp / "F2.json"), "--threads", "8"]); w2 = time.perf_counter() - t0
resu = U.main([str(D / "ref/F1_utils"), str(tmp / "utils.json")])
print(f"parity harness wall-clock: F1 {w1:.0f} s, F2 {w2:.0f} s (includes all gate computations)")
""")
md("## 4. Per-output parity")
md(r"""
### 4.1 Aggregation, RNG and kNN (gates 1–3, deterministic)

The bins are chosen by R's `sample()`; with `RRandom` the port draws the same numbers, so the
same seed cells are accepted in the same order. Gate 3 feeds R's bins to the port ("injected")
and compares the size-factor-normalised matrix.
""")
code(r"""
rows = []
for fx, res in (("F1", res1), ("F2", res2)):
    g = res["gates"]
    rows.append({"fixture": fx, "bins R": g["1a_bins"]["n_bins_r"], "bins Py": g["1a_bins"]["n_bins_py"],
                 "bins identical": g["1a_bins"]["identical"], "sampled windows identical": g["1b_windows"]["identical"],
                 "kNN sets differing": g["2"]["cells_with_different_neighbour_set"],
                 "matrix max abs err": g["3"]["max_abs_err_matrix"], "size factor max abs err": g["3"]["max_abs_err_size_factor_final"]})
pd.DataFrame(rows)
""")
md("### 4.2 Graphical lasso on identical inputs (gate 4, deterministic)")
code(r"""
pd.DataFrame([{"fixture": fx, "set": k, **v} for fx, res in (("F1", res1), ("F2", res2))
              for k, v in res["gates"]["4"].items() if isinstance(v, dict)])
""")
md(r"""
### 4.3 Distance parameters (gates 5–7)

Per sampled window (R's window sequence injected) and the mean that `run_cicero` uses.
""")
code(r"""
ad2, co2, g2 = P.read_fixture(D / "fixtures/F2")
sw = pd.read_csv(D / "ref/F2_seed42/sampled_windows.tsv", sep="\t")
r_dp = sw.loc[sw.status == "ok", "dp"].to_numpy(float)
cds2 = C.make_cicero_cds(ad2, co2, random_state=42, silent=True)
py_dp = C.estimate_distance_parameter(cds2, genomic_coords=g2, sample_windows=sw["win"].tolist(), n_jobs=8)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].scatter(r_dp, py_dp, s=10); ax[0].plot([r_dp.min(), r_dp.max()], [r_dp.min(), r_dp.max()], "r--", lw=0.8)
ax[0].set(xlabel="R distance_parameter", ylabel="Python", title="F2, 100 sampled windows")
ax[1].hist(np.abs(py_dp - r_dp) / r_dp, bins=30); ax[1].set(xlabel="relative error", title="threshold 1e-6")
plt.tight_layout(); plt.show()
for fx, res in (("F1", res1), ("F2", res2)):
    print(fx, "gate 5 max rel err %.1e | gate 6 mean R %r Py %r | gate 7 window status identical: %s"
          % (res["gates"]["5"]["max_rel_err"], res["gates"]["6"]["mean_r"], res["gates"]["6"]["mean_py"], res["gates"]["7"]["status_identical"]))
""")
md(r"""
### 4.4 Co-accessibility (gates 8–12)

Same set of `(Peak1, Peak2)` rows (also the same row order), same `NA` pattern, and values within
rounding of R (deterministic class, threshold 1e-8; ordinal Pearson/Spearman ≥ 0.9999; Jaccard
of pairs > 0.25 ≥ 0.999).
""")
code(r"""
conns2 = C.run_cicero(cds2, g2, random_state=42, n_jobs=8, silent=True)
rc2 = P.read_ref_conns(D / "ref/F2_seed42")
ok = ~np.isnan(rc2["coaccess"].to_numpy())
x, y = rc2["coaccess"].to_numpy()[ok], conns2["coaccess"].to_numpy()[ok]     # same row order as R
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].hexbin(x, y, gridsize=60, bins="log", cmap="viridis"); ax[0].set(xlabel="R coaccess", ylabel="Python", title="F2, 1.2 M rows")
d = np.abs(x - y); ax[1].hist(np.log10(d[d > 0]), bins=40)
ax[1].set(xlabel="log10 |Py - R| (non-zero differences)", title=f"{(d == 0).mean():.0%} bit-identical; max {d.max():.1e}")
plt.tight_layout(); plt.show()
pd.DataFrame([{"fixture": fx, **{f"g{k}": res["gates"][k]["pass"] for k in ("8", "9", "10", "11", "12", "13")},
               "max abs err": res["gates"]["9"]["max_abs_err"], "Pearson": res["gates"]["10"]["pearson"],
               "Jaccard>0.25": res["gates"]["11"]["jaccard"], "same row order": res["gates"]["8"]["same_row_order"]}
              for fx, res in (("F1", res1), ("F2", res2))])
""")
md("### 4.5 Gene activity (gates 14–15, F1)")
code(r"""
pd.DataFrame([{"gate": k, **res1["gates"][k]} for k in ("14", "15")])
""")
md(r"""
### 4.6 CCANs (gates 16–17, clustering)

Louvain is randomised. The port runs igraph's Louvain (C core 0.10, as in R igraph 2.1.4) on R's
random stream, so with the same seed the CCAN table is identical to R's (checked for seeds
{42, 2017, 7, 11, 23}); the gate (ARI against R at R's cutoff, seed 42) is then 1. The R-vs-R
agreement across seeds is shown as the noise floor of the pre-registered rule.
""")
code(r"""
from sklearn.metrics import adjusted_rand_score, confusion_matrix
for fx, res in (("F1", res1), ("F2", res2)):
    g = res["gates"]["16"]
    print(fx, "identical CCAN table, same seed:", g["identical_tables_same_seed"])
    print(fx, "ARI Py(seed 42) vs R per R seed:", {k: round(v, 3) for k, v in g["ari_py_vs_r_seed"].items()},
          "| R vs R median %.3f min %.3f | threshold %.3f -> %s" % (g["r_vs_r_median"], g["r_vs_r_min"], g["threshold_applied"], "PASS" if g["pass"] else "FAIL"))
    print("   cutoff Py %.2f, R %.2f (R range over seeds %s) -> %s" % (res["gates"]["17"]["cutoff_py"], res["gates"]["17"]["cutoff_r"], res["gates"]["17"]["r_range"], "PASS" if res["gates"]["17"]["pass"] else "FAIL"))
cut = float(pd.read_csv(D / "ref/F2_seed42/ccan_cutoffs.tsv", sep="\t")["cutoff"].iloc[0])
py = C.generate_ccans(rc2, coaccess_cutoff_override=cut, random_state=42, silent=True)
rr = pd.read_csv(D / "ref/F2_seed42/ccan_fixedcut_seed42.tsv", sep="\t")
uni = sorted(set(py.Peak) | set(rr.Peak))
lp = pd.Series(py.CCAN.values, index=py.Peak).reindex(uni).fillna(0).astype(int)
lr = pd.Series(rr.CCAN.values, index=rr.Peak).reindex(uni).fillna(0).astype(int)
ct = pd.crosstab(lr, lp)
order_r = ct.max(1).sort_values(ascending=False).index[:40]; order_p = ct.loc[order_r].idxmax(1).unique()
plt.figure(figsize=(5, 4.5)); plt.imshow(np.log1p(ct.loc[order_r, order_p].to_numpy()), cmap="magma", aspect="auto")
plt.xlabel("Python CCAN"); plt.ylabel("R CCAN (seed 42)"); plt.title("F2: 40 largest R CCANs, log(1 + peaks)"); plt.colorbar(); plt.show()
""")
md("### 4.7 Exported utilities (gates 19–24, F1)")
code(r"""
pd.DataFrame([{"gate": k, "pass": v["pass"]} for k, v in resu["gates"].items()] + [{"gate": "find_overlapping_ccans (info)", "pass": resu["informational"].get("find_overlapping_ccans_identical")}])
""")
md(r"""
## 5. Wall-clock comparison

R runs single-threaded (Cicero calls `mclapply(mc.cores = 1)`). Python: 1 thread and 8 threads
(`n_jobs`, identical output). Same node; the allocation was shared with other jobs, so expect
±25 % noise. "run_cicero" = `estimate_distance_parameter` + `generate_cicero_models` +
`assemble_connections`.
""")
code(r"""
def rt(fx):
    t = pd.read_csv(D / f"ref/{fx}_seed42/timings.tsv", sep="\t").set_index("step")["seconds"]
    return t.get("make_cicero_cds"), t.get("run_cicero_steps_total")
def pt(name):
    f = D / f"results/{name}.timings.json"
    if not f.exists(): return (np.nan, np.nan)
    t = json.loads(f.read_text()); return t["make_cicero_cds"], t["run_cicero_steps_total"]
rows = []
for fx, n1, n8 in (("F2", "py_F2_nj1", "py_F2_nj8"), ("F3", "py_F3_nj1", "py_F3_nj8")):
    if not (D / f"ref/{fx}_seed42/timings.tsv").exists(): continue
    for who, (m, r) in (("R", rt(fx)), ("Python, 1 thread", pt(n1)), ("Python, 8 threads", pt(n8))):
        rows.append({"fixture": fx, "implementation": who, "make_cicero_cds_s": m, "run_cicero_s": r})
tim = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for a, fx in zip(ax, ("F2", "F3")):
    sub = tim[tim.fixture == fx]
    if sub.empty: continue
    tot = sub.make_cicero_cds_s + sub.run_cicero_s
    a.barh(sub.implementation, tot, color=["#a4262c", "#0078d4", "#004e8c"][:len(sub)])
    a.set_xscale("log"); a.set_xlabel("make_cicero_cds + run_cicero (s)")
    a.set_title(f"{fx}: R / Py-1 = {tot.iloc[0] / tot.iloc[1]:.0f}x, R / Py-8 = {tot.iloc[0] / tot.iloc[2]:.0f}x" if len(sub) == 3 else fx)
plt.tight_layout(); plt.show()
tim
""")
md("## 6. Verdict")
code(r"""
rows = []
def thr(g):
    for o in manifest["outputs"] + manifest["addendum_1_outputs"]:
        if str(o["id"]) == g: return o["threshold"]
for fx, res in (("F1", res1), ("F2", res2)):
    for g, v in res["gates"].items():
        if "pass" in v and g[0].isdigit() and g.isdigit():
            rows.append({"fixture": fx, "gate": int(g), "threshold": thr(g), "pass": v["pass"]})
for g, v in resu["gates"].items():
    rows.append({"fixture": "F1", "gate": int(g), "threshold": thr(g), "pass": v["pass"]})
ver = pd.DataFrame(rows).sort_values(["gate", "fixture"]).reset_index(drop=True)
display(ver.pivot_table(index="gate", columns="fixture", values="pass", aggfunc="first"))
failed = ver[~ver["pass"]]
print("PASS — all outputs cleared the pre-registered gate" if failed.empty else
      "FAIL — gates %s; see RECONSTRUCTION_REPORT.md §3 (stochastic Louvain gate)" % sorted(set(failed.gate)))
""")
save(cells, "compare_R_vs_Python.ipynb")

# =============================================================================== Notebook 3
cells, md, code = new()
md(r"""
# Cicero R ⇄ `epione.tl` — function-by-function dictionary

For R users porting Cicero code to Python: every exported R function of cicero 1.3.9 (Monocle 3
branch), its parameters, the R call, the Python call on the **same input**, and the comparison
with R's output (R outputs dumped by `../r_reference_driver.R` and `../r_utils_driver.R`, which
together play the role of the kit's `r_per_function_dump.R`). Fixture: Cicero's `cicero_data`
(F1), seed 42. Orientation: R's CDS is peaks × cells, the AnnData here is cells × peaks.
""")
md("## 1. Setup")
code(SETUP + r"""
ad, coords, genome = P.read_fixture(D / "fixtures/F1")
R1 = D / "ref/F1_seed42"; RU = D / "ref/F1_utils"
verdict = []
def add(fn, output, cls, metric, value, ok):
    shown = f"{value:.2e}" if isinstance(value, float) else str(value)
    verdict.append({"Function": fn, "Output": output, "Class": cls, "Metric": metric, "Value": shown, "Pass": "✅" if ok else "❌"})
    print(f"{fn} / {output}: {metric} = {value} -> {'✅' if ok else '❌'}")
""")
md("## 2. Function by function")

FUNCS = []


def fn(title, desc, table, rcall, pycode, cmpcode):
    md(f"### {title}\n\n{desc}\n\n**Parameters**\n\n| R name | Python name | Type | Default | Range / values | Description |\n|---|---|---|---|---|---|\n{table.strip()}\n\n**R**\n```r\n{rcall.strip()}\n```")
    code(pycode)
    md("**Comparison with R**")
    code(cmpcode)


fn("2.1 `make_atac_cds(input, binarize = FALSE)`",
   "Builds the accessibility object from sparse `(peak, cell, count)` triplets; peaks sorted by chromosome and start.",
   r"""
| `input` | `input` | data.frame / path | — | 3 columns: peak, cell, count | sparse triplets; a path is read as tab-separated without header |
| `binarize` | `binarize` | logical | `FALSE` | TRUE/FALSE | counts > 0 become 1 (before duplicates are summed, as R) |
""",
   "input_cds <- make_atac_cds(cicero_data, binarize = TRUE)",
   r"""
cicero_data = pd.read_csv(RU / "cicero_data.tsv", sep="\t")
input_cds = epi.tl.make_atac_cds(cicero_data, binarize=True)
input_cds
""",
   r"""
rows, cols, m = U.rmat(RU, "atac")
ok = list(input_cds.var_names) == rows and list(input_cds.obs_names) == cols and U.same_matrix(input_cds.X, m)
add("make_atac_cds", "matrix + order", "deterministic", "identical", ok, ok)
""")

fn("2.2 `make_cicero_cds(cds, reduced_coordinates, k = 50, ...)`",
   "kNN aggregation of cells into overlapping bins (≤ 5000 candidates, ≥ 90 % overlap rejected), detect_genes, size factors, normalisation.",
   r"""
| `cds` | `adata` | cell_data_set | — | peaks × cells (R) / cells × peaks (Py) | accessibility |
| `reduced_coordinates` | `reduced_coordinates` | matrix / data.frame | — (Py: `"X_umap"`) | cells × d | embedding used for the kNN graph; Py also accepts an `obsm` key |
| `k` | `k` | integer | `50` | > 1 | cells per bin |
| `summary_stats` | `summary_stats` | character | `NULL` | numeric `pData`/`obs` columns | averaged per bin as `mean_<col>` |
| `size_factor_normalize` | `size_factor_normalize` | logical | `TRUE` | | divide bins by size factor |
| `silent` | `silent` | logical | `FALSE` | | overlap QC message |
| `return_agg_info` | `return_agg_info` | logical | `FALSE` | | also return the cell → bin table |
| — | `random_state` | int / `RRandom` / Generator | `0` | | **new in Python**: `s` ≡ `set.seed(s)` before the call; an `RRandom` continues a stream |
| — | `rng` | str | `"R"` | `"R"`, `"numpy"` | **new**: R's generator (exact) or numpy's (distributional) |
| — | `chosen` | int sequence | `None` | | **new**: inject the bins (0-based seed cells), e.g. from R |
| — | `layer` | str | `None` | | **new**: use `adata.layers[layer]` |
""",
   "set.seed(42)\ncicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = umap_coords)",
   r"""
cicero_cds = epi.tl.make_cicero_cds(ad, reduced_coordinates=coords, k=50, random_state=42, silent=True)
cicero_cds
""",
   r"""
r_bins = (R1 / "agg_cells.txt").read_text().split()
ok1 = list(cicero_cds.obs_names) == r_bins
err = float(abs(cicero_cds.X - P.read_ref_cicero_x(R1)).max())
add("make_cicero_cds", "bins (RNG)", "deterministic", "identical", ok1, ok1)
add("make_cicero_cds", "normalised matrix", "deterministic", "max abs err", err, err <= 1e-8)
""")

fn("2.3 `estimate_distance_parameter(cds, window = 5e5, maxit = 100, s = 0.75, sample_num = 100, ...)`",
   "Bisection per random window for the smallest penalty scale that leaves ≤ 5 % of > 250 kb pairs connected.",
   r"""
| `cds` | `cds` | cicero CDS | — | output of make_cicero_cds | |
| `window` | `window` | numeric | `500000` | > distance_constraint | window size (bp) |
| `maxit` | `maxit` | integer | `100` | | bisection iterations |
| `s` | `s` | numeric | `0.75` | (0, 1) | power-law exponent |
| `sample_num` | `sample_num` | integer | `100` | | windows to calibrate on |
| `distance_constraint` | `distance_constraint` | integer | `250000` | < window | "long" pair distance |
| `distance_parameter_convergence` | `distance_parameter_convergence` | numeric | `1e-22` | | bisection step tolerance |
| `max_elements` | `max_elements` | integer | `200` | | skip denser windows |
| `genomic_coords` | `genomic_coords` | data.frame / path | `human.hg19.genome` (Py: required) | chr, length | chromosome lengths |
| `max_sample_windows` | `max_sample_windows` | integer | `500` | | windows tried at most |
| — | `random_state`, `rng` | | `0`, `"R"` | | **new**: as in make_cicero_cds |
| — | `sample_windows` | int sequence | `None` | 1-based | **new**: inject the sampled windows |
| — | `return_details` | bool | `False` | | **new**: per-window table |
| — | `n_jobs` | int | `1` | | **new**: threads (identical result) |
""",
   "set.seed(42)\ndistance_parameters <- estimate_distance_parameter(cicero_cds, genomic_coords = sample_genome)",
   r"""
dps = epi.tl.estimate_distance_parameter(cicero_cds, genomic_coords=genome, random_state=42, n_jobs=8)
dps[:5]
""",
   r"""
r_dps = np.fromfile(R1 / "distance_parameters.f64")
rel = float(np.max(np.abs(dps - r_dps) / r_dps))
add("estimate_distance_parameter", "per-window values", "deterministic-bounded", "max rel err", rel, rel <= 1e-6)
""")

fn("2.4 `generate_cicero_models(cds, distance_parameter, s = 0.75, window = 5e5, max_elements = 200, genomic_coords)`",
   "Graphical lasso (glasso 1.11, ported line by line) in every overlapping window with the distance-scaled penalty.",
   r"""
| `cds` | `cds` | cicero CDS | — | | |
| `distance_parameter` | `distance_parameter` | numeric | — | > 0 | usually mean of estimate_distance_parameter |
| `s` | `s` | numeric | `0.75` | (0, 1) | |
| `window` | `window` | numeric | `500000` | | |
| `max_elements` | `max_elements` | integer | `200` | | |
| `genomic_coords` | `genomic_coords` | data.frame / path | `human.hg19.genome` (Py: required) | | |
| — | `n_jobs` | int | `1` | | **new**: threads (identical result) |
| — | `keep_wi` | bool | `True` | | **new**: keep the inverse (memory) |
""",
   "cicero_out <- generate_cicero_models(cicero_cds, distance_parameter = mean(distance_parameters), genomic_coords = sample_genome)",
   r"""
mean_dp = C._r_mean(dps)          # R's mean(): long-double sum + correction pass
models = epi.tl.generate_cicero_models(cicero_cds, mean_dp, genomic_coords=genome, n_jobs=8)
len(models)
""",
   r"""
mt = pd.read_csv(R1 / "models.tsv", sep="\t")
st = ["glasso" if isinstance(m, C.CiceroModel) else m for m in models.values()]
ok = list(models) == mt.window.tolist() and st == mt.status.tolist()
add("generate_cicero_models", "window status", "deterministic", "identical", ok, ok)
""")

fn("2.5 `assemble_connections(cicero_model_list, silent = FALSE)`",
   "cov2cor of every window model; pairs from overlapping windows reconciled (mean if same sign, else NA).",
   r"""
| `cicero_model_list` | `cicero_model_list` | list | — | output of generate_cicero_models | |
| `silent` | `silent` | logical | `FALSE` | | |
""",
   "conns <- assemble_connections(cicero_out)",
   r"""
conns = epi.tl.assemble_connections(models, silent=True)
conns.head()
""",
   r"""
cmp = P.compare_conns(conns, P.read_ref_conns(R1))
add("assemble_connections", "coaccess", "deterministic", "max abs err", cmp["max_abs_err"], cmp["max_abs_err"] <= 1e-8 and cmp["only_py"] == cmp["only_r"] == 0)
add("assemble_connections", "row order / NA pattern", "deterministic", "identical", cmp["same_row_order"] and cmp["na_pattern_mismatch"] == 0, cmp["same_row_order"] and cmp["na_pattern_mismatch"] == 0)
""")

fn("2.6 `run_cicero(cds, genomic_coords, window = 5e5, silent = FALSE, sample_num = 100)`",
   "Wrapper: estimate_distance_parameter (maxit 100, distance_constraint 250000, convergence 1e-22) → mean → generate_cicero_models → assemble_connections.",
   r"""
| `cds` | `cds` | cicero CDS | — | | |
| `genomic_coords` | `genomic_coords` | data.frame / path | — | | |
| `window` | `window` | numeric | `500000` | | |
| `silent` | `silent` | logical | `FALSE` | | |
| `sample_num` | `sample_num` | integer | `100` | | |
| — | `random_state`, `rng`, `n_jobs` | | `0`, `"R"`, `1` | | **new** |
""",
   "set.seed(42)\nconns <- run_cicero(cicero_cds, sample_genome)",
   r"""
conns_rc = epi.tl.run_cicero(cicero_cds, genome, random_state=42, n_jobs=8, silent=True)
conns_rc.attrs["mean_distance_parameter"]
""",
   r"""
r_mean = float(np.fromfile(R1 / "mean_distance_parameter.f64")[0])
cmp = P.compare_conns(conns_rc, P.read_ref_conns(R1))
add("run_cicero", "mean distance parameter", "deterministic", "rel err", abs(conns_rc.attrs["mean_distance_parameter"] - r_mean) / r_mean, conns_rc.attrs["mean_distance_parameter"] == r_mean)
add("run_cicero", "coaccess", "deterministic", "max abs err", cmp["max_abs_err"], cmp["max_abs_err"] <= 1e-8)
""")

fn("2.7 `generate_ccans(connections_df, coaccess_cutoff_override = NULL, tolerance_digits = 2)`",
   "Louvain modules (> 2 peaks) of the graph of pairs above a cutoff found by bisection on the CCAN count.",
   r"""
| `connections_df` | `connections_df` | data.frame | — | Peak1, Peak2, coaccess | |
| `coaccess_cutoff_override` | `coaccess_cutoff_override` | numeric | `NULL` | [0, 1] | skip the cutoff search |
| `tolerance_digits` | `tolerance_digits` | numeric | `2` | | cutoff precision |
| — | `random_state` | int / `RRandom` | `0` | | **new**: `s` ≡ `set.seed(s)` before the R call |
| — | `rng` | str | `"R"` | `"R"`, `"igraph"` | **new**: igraph Louvain on R's stream (exact) or python-igraph (distributional) |
| — | `silent` | bool | `False` | | **new** |
""",
   "set.seed(42)\nccan_assigns <- generate_ccans(conns)",
   r"""
rc = P.read_ref_conns(R1)
ccans = epi.tl.generate_ccans(rc, random_state=42, silent=True)
ccans.attrs["coaccess_cutoff"], ccans.CCAN.nunique()
""",
   r"""
from sklearn.metrics import adjusted_rand_score
cc = pd.read_csv(R1 / "ccan_cutoffs.tsv", sep="\t")
add("generate_ccans", "cutoff", "deterministic-bounded", "|Δ|", abs(ccans.attrs["coaccess_cutoff"] - cc.cutoff.iloc[0]), abs(ccans.attrs["coaccess_cutoff"] - cc.cutoff.iloc[0]) <= 0.02)
rr = {s: pd.read_csv(R1 / f"ccan_fixedcut_seed{s}.tsv", sep="\t") for s in cc.seed}
py = epi.tl.generate_ccans(rc, coaccess_cutoff_override=cc.cutoff.iloc[0], random_state=42, silent=True)
uni = sorted(set(py.Peak).union(*[set(v.Peak) for v in rr.values()]))
lab = lambda d: pd.Series(d.CCAN.values, index=d.Peak).reindex(uni).fillna(0).astype(int)
aris = {s: adjusted_rand_score(lab(v), lab(py)) for s, v in rr.items()}
print("ARI Py vs R seeds:", {k: round(v, 3) for k, v in aris.items()})
add("generate_ccans", "membership vs R seed 42", "clustering", "ARI", float(aris[42]), aris[42] >= 0.95)
r42 = pd.read_csv(R1 / "ccan_seed42.tsv", sep="\t")
same = len(ccans) == len(r42) and (ccans.Peak.values == r42.Peak.values).all() and (ccans.CCAN.values == r42.CCAN.values).all()
add("generate_ccans", "table, same seed (42)", "clustering", "identical", bool(same), bool(same))
""")

fn("2.8 `find_overlapping_ccans(ccan_assignments, min_overlap = 1)`",
   "Pairs of CCANs whose genomic spans overlap (reciprocal).",
   r"""
| `ccan_assignments` | `ccan_assignments` | data.frame | — | Peak, CCAN | |
| `min_overlap` | `min_overlap` | integer | `1` | | bp |
""",
   "olap <- find_overlapping_ccans(ccan_assigns)",
   r"""
r_cc = pd.read_csv(R1 / "ccan_seed42.tsv", sep="\t")
ov = epi.tl.find_overlapping_ccans(r_cc)
ov.head()
""",
   r"""
r_ov = pd.read_csv(RU / "foccans.tsv", sep="\t")
ok = len(ov) == len(r_ov) and (ov.values == r_ov.values).all()
add("find_overlapping_ccans", "pairs (R's CCANs)", "deterministic", "identical", ok, ok)
""")

fn("2.9 `annotate_cds_by_site(cds, feature_data, verbose = FALSE, maxgap = 0, all = FALSE, header = FALSE)`",
   "Adds overlapping features (e.g. promoter gene names) to the peak table; largest overlap wins.",
   r"""
| `cds` | `adata` | CDS | — | | |
| `feature_data` | `feature_data` | data.frame / path | — | chr, start, end, extra | 1-based closed |
| `verbose` | `verbose` | logical | `FALSE` | | |
| `maxgap` | `maxgap` | numeric / "nearest" | `0` | ≥ 0 | "nearest" not ported |
| `all` | `all` | logical | `FALSE` | | collapse all hits with "," |
| `header` | `header` | logical | `FALSE` | | for files |
| — | `inplace` | bool | `True` | | **new** |
""",
   "input_cds <- annotate_cds_by_site(input_cds, gene_annotation_sub)",
   r"""
prom = pd.read_csv(D / "fixtures/F1/promoters.tsv", sep="\t")
ad_ann = ad.copy(); epi.tl.annotate_cds_by_site(ad_ann, prom)
ad_ann.var.dropna().head(3)
""",
   r"""
ra = pd.read_csv(R1 / "annotated_sites.tsv", sep="\t", keep_default_na=False, na_values=["NA"]).set_index("peak").loc[ad_ann.var_names]
ok = (ad_ann.var.gene.fillna("").astype(str).values == ra.gene.fillna("").astype(str).values).all()
add("annotate_cds_by_site", "gene column", "deterministic", "identical", bool(ok), bool(ok))
""")

fn("2.10 `build_gene_activity_matrix(input_cds, cicero_cons_info, site_weights = NULL, dist_thresh = 250000, coaccess_cutoff = 0.25)`",
   "Promoter accessibility plus co-accessibility-weighted distal accessibility.",
   r"""
| `input_cds` | `adata` | CDS | — | `gene` column | |
| `cicero_cons_info` | `cicero_cons_info` | data.frame | — | Peak1, Peak2, coaccess | |
| `site_weights` | `site_weights` | numeric | `NULL` (all 1) | per peak | |
| `dist_thresh` | `dist_thresh` | numeric | `250000` | | |
| `coaccess_cutoff` | `coaccess_cutoff` | numeric | `0.25` | | |
| — | `gene_col`, `layer` | str | `"gene"`, `None` | | **new** |
""",
   "unnorm_ga <- build_gene_activity_matrix(input_cds, conns)",
   r"""
ga = epi.tl.build_gene_activity_matrix(ad_ann, rc)
ga
""",
   r"""
genes = (R1 / "unnorm_ga_genes.txt").read_text().split(); cells = (R1 / "unnorm_ga_cells.txt").read_text().split()
Rm = np.fromfile(R1 / "unnorm_ga.f64").reshape(len(genes), len(cells), order="F")
Pm = pd.DataFrame(ga.X.T.toarray(), index=ga.var_names, columns=ga.obs_names).loc[genes, cells].to_numpy()
err = float(np.abs(Pm - Rm).max()); add("build_gene_activity_matrix", "activity", "deterministic", "max abs err", err, err <= 1e-8)
""")

fn("2.11 `normalize_gene_activities(activity_matrices, cell_num_genes)`",
   "Size-factor model log(total activity) ~ log(accessible sites), exp transform, column scaling.",
   r"""
| `activity_matrices` | `activity_matrices` | matrix / list | — | no all-zero genes or cells | |
| `cell_num_genes` | `cell_num_genes` | named numeric | — | per cell | accessible sites per cell |
""",
   "cicero_gene_activities <- normalize_gene_activities(unnorm_ga, num_genes)",
   r"""
kg = np.asarray(ga.X.sum(0)).ravel() != 0; kc = np.asarray(ga.X.sum(1)).ravel() != 0
gn = epi.tl.normalize_gene_activities(ga[kc][:, kg].copy(), pd.Series(np.asarray((ad.X > 0).sum(1)).ravel(), index=ad.obs_names))
gn
""",
   r"""
genes = (R1 / "ga_genes.txt").read_text().split(); cells = (R1 / "ga_cells.txt").read_text().split()
Rm = np.fromfile(R1 / "ga.f64").reshape(len(genes), len(cells), order="F")
Pm = pd.DataFrame(gn.X.T.toarray(), index=gn.var_names, columns=gn.obs_names).loc[genes, cells].to_numpy()
err = float(np.abs(Pm - Rm).max() / np.abs(Rm).max()); add("normalize_gene_activities", "activity", "deterministic", "max err / max|R|", err, err <= 1e-8)
""")

UTIL_PY = {
    "aggregate_nearby_peaks": "agg = epi.tl.aggregate_nearby_peaks(input_cds, distance=10000)\nagg",
    "aggregate_by_cell_bin": "cell_data = pd.read_csv(RU / 'cell_data.tsv', sep='\\t').set_index('cell')\nx = input_cds.copy(); x.obs['timepoint'] = cell_data.loc[x.obs_names, 'timepoint'].to_numpy()\nbinned = epi.tl.aggregate_by_cell_bin(x, 'timepoint')\nbinned.obs",
    "compare_connections": "c1 = pd.read_csv(RU / 'cc_conns1.tsv', sep='\\t'); alt = pd.read_csv(RU / 'cc_alt.tsv', sep='\\t')\nin_alt = epi.tl.compare_connections(c1, alt, maxgap=1000)\nin_alt.sum(), len(in_alt)",
    "find_overlapping_coordinates": "epi.tl.find_overlapping_coordinates(list(input_cds.var_names), 'chr18:10,100-1246234')[:5]",
    "df_for_coords": "epi.tl.df_for_coords(['chr1:2,039-30,239', 'chrX:28884:101293'])",
}
for name, title, desc, table, rcall, gate in (
    ("aggregate_nearby_peaks", "2.12 `aggregate_nearby_peaks(cds, distance = 1000)`", "Sum peaks closer than `distance` (GenomicRanges::reduce).",
     "| `cds` | `adata` | CDS | — | | |\n| `distance` | `distance` | numeric | `1000` | | min.gapwidth |\n| — | `layer` | str | `None` | | **new** |",
     "agg_cds <- aggregate_nearby_peaks(input_cds, distance = 10000)", "20"),
    ("aggregate_by_cell_bin", "2.13 `aggregate_by_cell_bin(cds, group_col)`", "Sum cells per group of a `pData` column.",
     "| `cds` | `adata` | CDS | — | | |\n| `group_col` | `group_col` | character | — | `obs` column | |\n| — | `layer` | str | `None` | | **new** |",
     "binned <- aggregate_by_cell_bin(input_cds, \"timepoint\")", "21"),
    ("compare_connections", "2.14 `compare_connections(conns1, conns2, maxgap = 0)`", "Is each pair of `conns1` present (either orientation, within maxgap) in `conns2`?",
     "| `conns1` | `conns1` | data.frame | — | | |\n| `conns2` | `conns2` | data.frame | — | | |\n| `maxgap` | `maxgap` | numeric | `0` | | bp |",
     "in_alt <- compare_connections(conns, alt, maxgap = 1000)", "22"),
    ("find_overlapping_coordinates", "2.15 `find_overlapping_coordinates(coord_list, coord, maxgap = 0)`", "Peaks overlapping one or several coordinates (NA → `None`).",
     "| `coord_list` | `coord_list` | character | — | | |\n| `coord` | `coord` | character | — | one or many | commas allowed |\n| `maxgap` | `maxgap` | numeric | `0` | | |",
     "find_overlapping_coordinates(peaks, \"chr18:10,100-1246234\")", "23"),
    ("df_for_coords", "2.16 `df_for_coords(coord_strings)`", "Coordinate strings to chr / bp1 / bp2 / Peak.",
     "| `coord_strings` | `coord_strings` | character | — | `:`, `-`, `_` separators | |",
     "df_for_coords(c(\"chr1:2,039-30,239\", \"chrX:28884:101293\"))", "24"),
):
    fn(title, desc, table, rcall, UTIL_PY[name],
       f"import contextlib, io\nif 'ures' not in dir():\n    with contextlib.redirect_stdout(io.StringIO()):\n        ures = U.main([str(RU), str(Path(tempfile.mkdtemp()) / 'u.json')])\n"
       f"g = ures['gates']['{gate}']; add('{name}', 'gate {gate} (all cases of r_utils_driver.R)', 'deterministic', 'identical', g['pass'], g['pass'])")

fn("2.17 `glasso::glasso(s, rho, nobs, zero, thr = 1e-4, maxit = 1e4, approx = FALSE, penalize.diagonal = TRUE, start, w.init, wi.init, trace)`",
   "The solver Cicero calls (CRAN glasso 1.11, Fortran), ported subroutine by subroutine; cold start, exact path only.",
   r"""
| `s` | `s` | matrix | — | square | covariance |
| `rho` | `rho` | scalar / vector / matrix | — | ≥ 0 | penalty |
| `nobs` | — | integer | `NULL` | | not ported (log-likelihood only) |
| `zero` | — | matrix | `NULL` | | not ported (Cicero does not use it) |
| `thr` | `thr` | numeric | `1e-4` | | |
| `maxit` | `maxit` | integer | `1e4` | | |
| `approx` | — | logical | `FALSE` | | not ported (Meinshausen–Bühlmann) |
| `penalize.diagonal` | `penalize_diagonal` | logical | `TRUE` | | |
| `start`, `w.init`, `wi.init` | — | | `"cold"` | | warm start not ported |
| `trace` | — | logical | `FALSE` | | not ported |
""",
   "GL <- glasso::glasso(cov_mat, rho_mat)",
   r"""
tab = pd.read_csv(R1 / "probes.tsv", sep="\t")
S, Rh, W, WI = (np.fromfile(R1 / f"probe_{m}.f64") for m in ("S", "rho", "w", "wi"))
errs = []
for _, r in tab.iterrows():
    n, o = int(r.n), int(r.offset); sl = slice(o, o + n * n)
    g = epi.tl.glasso(S[sl].reshape(n, n, order="F"), Rh[sl].reshape(n, n, order="F"))
    errs.append((np.abs(g.w - W[sl].reshape(n, n, order="F")).max(), np.abs(g.wi - WI[sl].reshape(n, n, order="F")).max(), g.niter == r.niter))
errs = np.array(errs, dtype=float)
""",
   r"""
add("glasso", "w (40 Cicero windows)", "deterministic", "max abs err", float(errs[:, 0].max()), errs[:, 0].max() <= 1e-8)
add("glasso", "wi (40 Cicero windows)", "deterministic", "max abs err", float(errs[:, 1].max()), errs[:, 1].max() <= 1e-8)
add("glasso", "iterations", "deterministic", "identical", bool(errs[:, 2].all()), bool(errs[:, 2].all()))
""")

md(r"""
### Not ported

| R function | Reason |
|---|---|
| `plot_connections`, `plot_accessibility_in_pseudotime` | Gviz / ggplot2 plotting; epione has its own track plots (`epi.pl.plot_browser_track`, `epi.pl.igv_browser`) |
| `ranges_for_coords` | returns a Bioconductor `GRanges`; `df_for_coords` gives the same coordinates as a DataFrame |
""")
md("## 3. Aggregate verdict")
code(r"""
pd.DataFrame(verdict)
""")
save(cells, "function_by_function_R_parity.ipynb")

# =============================================================================== Notebook 4
cells, md, code = new()
md(r"""
# Evolution log — Cicero → `epione.tl`

One `## Iteration N — <title>` block per iteration of the port, each with the narrative and a
subplot of the trajectory so far (wall-clock top, parity bottom), as required by
omicverse-rebuildr `NOTEBOOKS.md`. Numbers come from `../ITERATION_LOG.md` (benchmark:
`make_cicero_cds` + `run_cicero` on F1, 1 warm-up + 3 runs; parity = gate 10, Pearson of
coaccess vs R, threshold 0.9999; gate 9's max abs error is shown as a second marker).
""")
md("## Setup")
code(r"""
import re, yaml
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
LOG = Path("..") / "ITERATION_LOG.md"
blocks = re.findall(r"##\s+(iter\s+\d+|Baseline)\s*—[^\n]*\n+```yaml\n(.*?)\n```", LOG.read_text(), re.S)
it = {b["iter"]: b for b in (yaml.safe_load(t) for _, t in blocks)}
MAXERR = {0: 8.0e-16, 1: 8.0e-16, 2: 8.0e-16, 3: 8.0e-16, 4: 8.0e-16, 5: 8.9e-16, 6: 8.0e-16, 7: 8.0e-16}   # gate 9 on F1 (bench JSONs)
def upto(n):
    xs = [i for i in sorted(it) if i <= n]
    fig, (a, b) = plt.subplots(2, 1, figsize=(6, 4.2), sharex=True)
    acc = [i for i in xs if it[i]["status"] in ("baseline", "ACCEPT")]
    a.errorbar(acc, [it[i]["wall_clock_mean_s"] for i in acc], yerr=[it[i]["wall_clock_stddev_s"] for i in acc], marker="o", color="#0078d4")
    rej = [i for i in xs if i not in acc]
    a.scatter(rej, [it[i]["wall_clock_mean_s"] for i in rej], marker="x", color="grey", zorder=3, label="rejected")
    a.set_yscale("log"); a.set_ylabel("wall-clock F1 (s)"); a.grid(alpha=.3)
    if rej: a.legend(frameon=False, fontsize=8)
    b.plot(acc, [1 - it[i]["parity_metric"] for i in acc], marker="o", color="#107c10", label="1 − Pearson (gate 10)")
    b.plot(xs, [MAXERR[i] for i in xs], marker="s", color="#8a8886", ls=":", label="max abs err (gate 9)")
    b.axhline(1e-4, color="red", ls="--", lw=1, label="gate 10 threshold (1 − 0.9999)")
    b.axhline(1e-8, color="orange", ls="--", lw=1, label="gate 9 threshold")
    b.set_yscale("symlog", linthresh=1e-17); b.set_ylabel("distance to R"); b.set_xlabel("iteration"); b.grid(alpha=.3)
    b.legend(fontsize=7, frameon=False, loc="upper left"); b.set_xticks(range(0, max(it) + 1))
    a.scatter([n], [it[n]["wall_clock_mean_s"]], s=150, facecolors="none", edgecolors="crimson")
    plt.tight_layout(); plt.show()
""")
TITLES = {
    0: ("Baseline translation", "baseline"),
    1: ("Covariance: upper triangle, row-wise long-double sums (exact)", "ACCEPT"),
    2: ("Distance parameter: replay sampling, bisect each window once (exact)", "ACCEPT"),
    3: ("Threads over windows (exact)", "ACCEPT"),
    4: ("Equivalence: remaining exported utilities and R's locale collation", "ACCEPT"),
    5: ("BLAS covariance (bounded) — rejected", "REJECT_INADMISSIBLE"),
    6: ("Compiled x87 long-double covariance (exact)", "ACCEPT"),
    7: ("Equivalence: Louvain on R's random stream, R's round()", "ACCEPT"),
}
NARR = {
    0: """Every exported R function on the pipeline path was translated in dependency order: `glasso.f` (connect / row / lasinv1 / setup / lasso / fatmul / inv) to numba keeping the Fortran accumulation order; `make_cicero_cds` with R's `sample()` stream reproduced by `RRandom`; `get_genomic_range`, `get_rho_mat`, `find_distance_parameter`, `cov2cor`, `reconcile` and data.table's key order mirrored; R's long-double `mean()` and `cov()` emulated with numpy `longdouble`. The first full run already cleared every deterministic gate on F1 (bins, windows, bit-identical mean distance parameter, coaccess max abs error 8e-16 in R's row order). The one finding of this pass came from `annotate_cds_by_site`: `findOverlaps` reports a query's hits in NCList order (feature start, then end), not subject order, which decides ties between equally overlapping promoters; fixed before the first commit.""",
    1: """R's `cov_na_1` forms only `j <= i`. The covariance was rewritten to build the upper triangle and add one row of double products at a time into a long-double accumulator — the same additions in the same order, verified bit for bit against a naive loop. On F1 (25 bins) covariance is negligible, so the gain only shows on F2-sized windows (3.4 s → 1.3 s per 200-site window).""",
    2: """`find_distance_parameter` returns a non-number only for "No long edges", which depends on peak positions alone, so R's sampling loop can be replayed with cheap checks first; then each distinct window (windows are drawn with replacement) is bisected once. Exact by construction; small gain on F1 (≈ 1.04×, within noise).""",
    3: """Windows are independent (R uses `mclapply(mc.cores = 1)`). Bisections and window models run in a thread pool; the numba solver is `nogil` and numpy long-double ufuncs release the GIL, results are collected in window order. 10× on F1 with 8 threads, output identical for any `n_jobs`.""",
    4: """Not a speed step. The six remaining exported utilities were ported after an addendum pre-registration (gates 19–24). Gate 21 failed on row order: R sorts factor levels and `dcast` output with the session's collation (en_US.UTF-8: punctuation < digits < letters, case-insensitive, lowercase first), not by bytes. `_r_sorted()` reproduces R's `sort()`; data.table output stays in byte order. All addendum gates then passed exactly.""",
    5: """Candidate (B): a double-precision BLAS covariance. Every gate still passed on F2 (coaccess error 6.9e-15, mean distance parameter off by 5.5e-16 relative) and `run_cicero` went from 32.6 s to 8.9 s with 8 threads. Rejected: the brief admits only provably equivalent rewrites, and this one is bounded, not exact.""",
    7: """The last output that differed from R was the CCAN membership: R's `cluster_louvain` is igraph's multilevel algorithm drawing from R's `unif_rand()` through a 30-bit bridge, while python-igraph has its own generator. Ported: the algorithm (`louvain.c`, C core 0.10), igraph's bounded integers and shuffle, the undirected edge store and `igraph_incident` order, `igraph_qsort` (its order of equal keys decides the order of floating-point sums) and the graph Cicero builds, zero-weight self loops included. The cutoff search then still diverged at its fifth step: R 4.x `round(x, 3)` is not the correctly rounded decimal (`round(0.10549999999999999656, 3)` is 0.106), so `fround.c` was ported too. Now `generate_ccans` returns R's exact table for five seeds on F1 and F2; F1's gate 16 goes from 0.935 (fail) to 1.""",
    6: """R's `LDOUBLE` is the x87 80-bit format on x86-64. A numba intrinsic emits LLVM `x86_fp80` code — `fpext`, one x87 `fadd`, `fptrunc` — exactly what GCC compiles `sum += prod` to; the running sum is held exactly as hi + lo doubles. Bit-identical to the numpy long-double path (tests) and to R on F2 (mean distance parameter equal, coaccess error 4.4e-16); per-window covariance 6× faster and F2 `run_cicero` 32.6 s → 11.6 s with 8 threads. F1 is glasso-bound, hence flat here.""",
}
for n in range(8):
    t, st = TITLES[n]
    md(f"## Iteration {n} — {t}\n\n{NARR[n]}\n\n**Status**: {st}. **ITERATION_LOG cross-ref**: [iter {n}](../ITERATION_LOG.md)")
    code(f"upto({n})")
md("## Aggregate evolution figure")
code(r"""
import os, sys, subprocess
kit = os.environ.get("REBUILDR_KIT")
if kit:
    subprocess.run([sys.executable, "-m", "engine.plot_evolution", "--port-dir", str(Path("..").resolve()),
                    "--output", str(Path("evolution.png").resolve()), "--threshold", "0.9999"], cwd=kit, check=True)
else:
    upto(max(it)); plt.gcf().savefig("evolution.png", dpi=150)
from IPython.display import Image
Image("evolution.png")
""")
save(cells, "evolution.ipynb")
