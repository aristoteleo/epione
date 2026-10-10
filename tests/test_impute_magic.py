import json
import os
import pathlib
import subprocess
import sys

import anndata as ad
import numpy as np
import pytest

import epione as epi
from epione.tl._impute import _RRandom, _bin_size

PARITY_DIR = pathlib.Path(__file__).parent / "parity" / "impute_magic"
_PARITY_ROOT = os.environ.get("EPIONE_ARCHR_PARITY_DIR")
REF_DIR = os.path.join(_PARITY_ROOT, "impute_magic") if _PARITY_ROOT else None


def _toy(n=300, d=8, g=5, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, d)) * 4
    lab = rng.integers(0, 3, n)
    a = ad.AnnData(X=rng.poisson(2.0, size=(n, g)).astype(float))
    a.obsm["X_lsi"] = centers[lab] + rng.normal(size=(n, d))
    return a


def test_r_rng_matches_r_sample():
    # R: set.seed(1); lapply(1:2, function(i) sample(10, 10))
    r = _RRandom(1)
    assert (r.permutation(10) + 1).tolist() == [9, 4, 7, 1, 2, 5, 3, 10, 6, 8]
    assert (r.permutation(10) + 1).tolist() == [3, 1, 5, 8, 2, 6, 10, 9, 4, 7]


def test_bin_size_matches_archr():
    assert _bin_size(10660, 5000) == 5331
    assert _bin_size(300, 100) == 101


def test_weights_are_markov_with_self_loops():
    a = _toy()
    epi.tl.add_impute_weights(a, use_rep="X_lsi", sample_cells=None, k=10, verbose=False)
    w = a.uns["impute_weights"]["weights"]
    assert len(w) == 1 and len(w[0]) == 1
    W = w[0][0]["weights"]
    np.testing.assert_allclose(W.sum(1), 1.0, atol=1e-12)
    assert (W >= 0).all() and (np.diag(W) > 0).all()


def test_blocks_partition_cells_and_constant_is_preserved():
    a = _toy(n=301)
    epi.tl.add_impute_weights(a, use_rep="X_lsi", sample_cells=100, n_rep=3, k=10, verbose=False)
    for rep in a.uns["impute_weights"]["weights"]:
        cells = np.concatenate([b["cells"] for b in rep])
        assert sorted(cells.tolist()) == list(range(a.n_obs))
    out = epi.tl.impute_matrix(a, X=np.full((a.n_obs, 2), 7.0))
    np.testing.assert_allclose(out, 7.0, atol=1e-12)


def test_injected_blocks_are_deterministic_and_layer_added():
    a = _toy()
    perm = np.random.default_rng(5).permutation(a.n_obs)
    blocks = [[perm[:150], perm[150:]]]
    epi.tl.add_impute_weights(a, use_rep="X_lsi", blocks=blocks, k=10, verbose=False)
    o1 = epi.tl.impute_matrix(a, layer_added="magic")
    epi.tl.add_impute_weights(a, use_rep="X_lsi", blocks=blocks, k=10, verbose=False)
    np.testing.assert_array_equal(o1, epi.tl.impute_matrix(a))
    assert a.layers["magic"].shape == a.shape


@pytest.mark.skipif(REF_DIR is None or not pathlib.Path(REF_DIR, "seed1_imputed.tsv.gz").exists(),
                    reason="EPIONE_ARCHR_PARITY_DIR not set or ArchR reference outputs missing")
def test_parity_vs_archr(tmp_path):
    out = tmp_path / "parity.json"
    subprocess.run([sys.executable, str(PARITY_DIR / "parity_check.py"), REF_DIR, str(out)],
                   check=True, env={**os.environ})
    res = json.loads(out.read_text())
    if "reduced_dims_max_abs_err" in res:
        assert res["reduced_dims_max_abs_err"] <= 1e-6
    for tag in ("injected", "rngR"):
        assert res[f"{tag}_blocks_identical"]
        assert res[f"{tag}_weights_max_abs_err"] <= 1e-6
    assert res["injected_imputed_max_rel_err"] <= 1e-6
    assert res["rngR_imputed_max_rel_err"] <= 1e-6
