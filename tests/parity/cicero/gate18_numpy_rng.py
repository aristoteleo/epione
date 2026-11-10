"""Gate 18 (pre-registered): coaccess with the free-running numpy RNG vs R, against R's own
seed-to-seed agreement. Pass iff Pearson and Jaccard(>0.25) of Python(rng="numpy", seed 42) vs
R(seed 42) are >= median over extra seeds of R(seed 42) vs R(seed s) minus 0.02.
Usage: python gate18_numpy_rng.py <fixture_F2> <ref_root> <out.json> [threads]"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import parity_check as P
from epione.tl import _cicero as C

fx, root, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
nj = int(sys.argv[4]) if len(sys.argv) > 4 else 8
seeds = [2017, 7, 11, 23]


def agree(a, b):
    """Pearson on rows non-NA in both + Jaccard of pairs > 0.25; tables in R's row order."""
    assert (a["Peak1"].astype(str).to_numpy() == b["Peak1"].astype(str).to_numpy()).all()
    x, y = a["coaccess"].to_numpy(), b["coaccess"].to_numpy()
    ok = ~np.isnan(x) & ~np.isnan(y)
    sx, sy = x > 0.25, y > 0.25
    return float(pearsonr(x[ok], y[ok])[0]), float((sx & sy).sum() / max(1, (sx | sy).sum()))


ad, co, g = P.read_fixture(fx)
cds = C.make_cicero_cds(ad, co, rng="numpy", random_state=42, silent=True)
py = C.run_cicero(cds, g, rng="numpy", random_state=42, n_jobs=nj, silent=True)
r42 = P.read_ref_conns(root / "F2_seed42")
rr = {s: agree(r42, P.read_ref_conns(root / f"F2_seed{s}")) for s in seeds if (root / f"F2_seed{s}" / "conns_coaccess.f64").exists()}
pr, pj = agree(py, r42)
med_r = float(np.median([v[0] for v in rr.values()])); med_j = float(np.median([v[1] for v in rr.values()]))
res = {"py_numpy_vs_r42": {"pearson": pr, "jaccard": pj}, "r42_vs_r_seed": {str(k): {"pearson": v[0], "jaccard": v[1]} for k, v in rr.items()},
       "threshold_pearson": med_r - 0.02, "threshold_jaccard": med_j - 0.02, "n_bins_py": int(cds.n_obs),
       "pass": bool(pr >= med_r - 0.02 and pj >= med_j - 0.02)}
Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
