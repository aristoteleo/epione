"""Tests for epione.tl.coaccessibility."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _make_coacc_adata(seed=0, n_cells=800, n_groups=4, n_peaks=1500,
                      n_pairs=30, tile_w=500, spacing=1000, chrom="chr1"):
    """Planted peak-peak pairs that co-open together within 50 kb; random
    background peaks remain uncorrelated."""
    rng = np.random.default_rng(seed)
    n_cells = (n_cells // n_groups) * n_groups
    labels = np.repeat(np.arange(n_groups), n_cells // n_groups)
    peak_start = 100_000 + np.arange(n_peaks) * spacing
    peak_end = peak_start + tile_w
    peak_names = [f"{chrom}:{s}-{e}" for s, e in zip(peak_start, peak_end)]

    X = (rng.random((n_cells, n_peaks)) < 0.03).astype(np.float32)

    planted = []
    for g in range(n_groups):
        # In each group, pick n_pairs/n_groups pairs within 50 kb
        for _ in range(n_pairs // n_groups):
            i = int(rng.integers(n_peaks - 60))
            j = int(i + rng.integers(1, 60))   # within ~60*1000 = 60kb
            # Open both in group cells
            mask = labels == g
            X[mask, i] = (rng.random(mask.sum()) < 0.9).astype(np.float32)
            X[mask, j] = (rng.random(mask.sum()) < 0.9).astype(np.float32)
            # Closed in other groups
            X[~mask, i] = (rng.random((~mask).sum()) < 0.05).astype(np.float32)
            X[~mask, j] = (rng.random((~mask).sum()) < 0.05).astype(np.float32)
            planted.append((peak_names[i], peak_names[j]))

    ad = AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"group": labels.astype(str),
                          "n_fragment": np.asarray(X.sum(axis=1)).ravel()},
                         index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=peak_names),
    )
    return ad, planted


def test_coaccessibility_recovers_planted_pairs():
    import epione as ep
    ad, planted = _make_coacc_adata()
    # Need an embedding for metacells
    ep.tl.iterative_lsi(
        ad, n_components=15, iterations=2, var_features=1000,
        total_features=1400, resolution=0.5, sample_cells_pre=None,
        seed=0, verbose=False,
    )
    co = ep.tl.coaccessibility(
        ad, use_rep="X_iterative_lsi",
        n_metacells=200, k_neighbors=30,
        max_distance=80_000,
        seed=0, verbose=True,
    )
    assert "coaccessibility" in ad.uns
    # Planted pair retrieval: make keys order-insensitive
    planted_keys = {frozenset(p) for p in planted}
    co["key"] = co.apply(lambda r: frozenset([r["peak1"], r["peak2"]]), axis=1)
    hits = co[co["key"].isin(planted_keys)]
    print(f"planted {len(planted)} pairs, tested {len(hits)}, "
          f"median |r|={hits['correlation'].abs().median():.3f}")
    assert len(hits) >= 0.8 * len(planted), "most planted pairs should fit in window"
    assert hits["correlation"].abs().median() > 0.6, (
        f"planted pairs should have median |r|>0.6, got "
        f"{hits['correlation'].abs().median():.3f}"
    )
    # False-positive rate
    non_hits = co[~co["key"].isin(planted_keys)]
    fp_rate = (non_hits["correlation"].abs() > 0.7).mean()
    print(f"non-planted |r|>0.7 rate: {fp_rate:.2%}")
    assert fp_rate < 0.15, f"too many distal false positives: {fp_rate:.2%}"


if __name__ == "__main__":
    test_coaccessibility_recovers_planted_pairs()
    print("ok")
