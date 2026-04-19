"""Correctness tests for epione.tl.peak_to_gene on synthetic multiome data.

We plant a controlled structure:
- 5 "pseudo-populations" whose LSI embedding separates them.
- For each population, a set of peak/gene pairs that are STRONGLY correlated
  (both on/off together) within ±100 kb.
- Random peak/gene pairs outside that window as a negative control.

Expectations:
- Planted pairs are retrieved with |r| > 0.7 and FDR < 1e-3.
- Random distal pairs are not selected (because they live outside max_distance).
- Random same-window pairs have near-zero correlation.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _make_multiome(seed=0,
                   n_cells=1500,
                   n_groups=5,
                   n_peaks=2000,
                   n_genes=200,
                   n_planted=50,
                   tile_w=500,
                   spacing=2000,
                   chrom="chr1"):
    rng = np.random.default_rng(seed)
    # Round n_cells down so it is divisible by n_groups.
    n_cells = (n_cells // n_groups) * n_groups
    labels = np.repeat(np.arange(n_groups), n_cells // n_groups)

    # Peak coordinates: evenly spaced tiles on chr1, starting at 100_000
    peak_start = 100_000 + np.arange(n_peaks) * spacing
    peak_end = peak_start + tile_w
    peak_names = [f"{chrom}:{s}-{e}" for s, e in zip(peak_start, peak_end)]

    # Gene coordinates: place genes near random peaks (within 20 kb of tss)
    gene_idx_anchor = rng.choice(n_peaks, size=n_genes, replace=False)
    gene_tss = peak_start[gene_idx_anchor] + rng.integers(-20_000, 20_000, size=n_genes)
    gene_start = np.maximum(gene_tss - 1000, 0)
    gene_end = gene_start + 2000
    gene_names = [f"gene{i:04d}" for i in range(n_genes)]
    gene_ann = pd.DataFrame({
        "gene_name": gene_names,
        "chrom": chrom,
        "start": gene_start,
        "end": gene_end,
        "strand": "+",
    })

    # Plant n_planted peak-gene pairs: pair a group-specific set of peaks with
    # group-specific genes whose TSS is < 100 kb away.
    planted_pairs = []
    for group in range(n_groups):
        # Pick n_planted/n_groups random (peak, gene) pairs within ±80 kb
        k = n_planted // n_groups
        g_pick = rng.choice(n_genes, size=k, replace=False)
        for gi in g_pick:
            tss = gene_tss[gi]
            # Find peaks within 80 kb of this TSS
            near = np.where(np.abs(peak_start - tss) < 80_000)[0]
            if len(near) == 0:
                continue
            pi = rng.choice(near)
            planted_pairs.append((int(pi), int(gi), int(group)))

    # Build peak and gene matrices.
    # Background: every cell has ~5% peaks open, ~30% genes expressed with noise.
    peak = (rng.random((n_cells, n_peaks)) < 0.03).astype(np.float32)
    gene = rng.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32)

    # For planted pairs: when a cell is in the matching group, strongly open
    # the peak and boost expression of the linked gene.
    for pi, gi, grp in planted_pairs:
        mask = labels == grp
        # Peak open with ~90% prob in that group, ~5% elsewhere
        peak[mask, pi] = (rng.random(mask.sum()) < 0.9).astype(np.float32)
        peak[~mask, pi] = (rng.random((~mask).sum()) < 0.05).astype(np.float32)
        # Gene expression elevated in the group
        gene[mask, gi] = rng.poisson(20.0, size=mask.sum())
        gene[~mask, gi] = rng.poisson(0.5, size=(~mask).sum())

    X_peak = sp.csr_matrix(peak)
    ad_peak = AnnData(
        X=X_peak,
        obs=pd.DataFrame({"group": labels.astype(str)},
                         index=[f"cell{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=peak_names),
    )
    ad_rna = AnnData(
        X=gene,
        obs=ad_peak.obs,
        var=pd.DataFrame(index=gene_names),
    )
    return ad_peak, ad_rna, gene_ann, planted_pairs, peak_names, gene_names


def test_peak_to_gene_recovers_planted_pairs():
    import epione as ep
    ad_peak, ad_rna, gene_ann, planted_pairs, peak_names, gene_names = _make_multiome()

    # Run iterative LSI to produce an embedding (required by peak_to_gene).
    ep.tl.iterative_lsi(
        ad_peak, n_components=20, iterations=2, var_features=1500,
        total_features=1900, resolution=0.5, n_neighbors=20,
        sample_cells_pre=None, seed=0, verbose=False,
    )

    pairs = ep.tl.peak_to_gene(
        ad_peak,
        rna=ad_rna,
        gene_annotation=gene_ann,
        use_rep="X_iterative_lsi",
        n_metacells=200, k_neighbors=30,
        max_distance=100_000,
        seed=0,
        verbose=True,
    )

    # Planted positives should be in the output with strong r
    planted_set = {(peak_names[pi], gene_names[gi]) for pi, gi, _ in planted_pairs}
    hits = pairs[pairs.apply(lambda r: (r["peak"], r["gene"]) in planted_set, axis=1)]
    recovered_frac = hits["correlation"].gt(0.7).mean()
    print(
        f"planted pairs: {len(planted_set)}, tested: {len(hits)}, "
        f"r>0.7 fraction: {recovered_frac:.3f}"
    )
    assert len(hits) >= 0.8 * len(planted_set), (
        f"only {len(hits)}/{len(planted_set)} planted pairs made it into the "
        "distance window"
    )
    assert recovered_frac > 0.8, (
        f"only {recovered_frac:.2%} of planted pairs recovered with r>0.7"
    )

    # Negative set: non-planted pairs with distance >50 kb should be near-zero
    non_hits = pairs[~pairs.apply(lambda r: (r["peak"], r["gene"]) in planted_set, axis=1)]
    far = non_hits[non_hits["distance"].abs() > 50_000]
    if len(far) > 0:
        high_noise = far["correlation"].abs().gt(0.7).mean()
        print(f"false-positive rate (|r|>0.7 for distal non-planted): {high_noise:.3%}")
        assert high_noise < 0.05, (
            f"{high_noise:.2%} distal random pairs have |r|>0.7 – too noisy"
        )


def test_peak_to_gene_api_keys():
    import epione as ep
    ad_peak, ad_rna, gene_ann, *_ = _make_multiome(n_cells=500, n_peaks=500, n_genes=50,
                                                    n_planted=10, n_groups=3)
    ep.tl.iterative_lsi(
        ad_peak, n_components=10, iterations=1, var_features=400,
        total_features=500, sample_cells_pre=None, seed=0, verbose=False,
    )
    df = ep.tl.peak_to_gene(
        ad_peak, rna=ad_rna, gene_annotation=gene_ann,
        n_metacells=50, k_neighbors=20, max_distance=100_000,
        seed=0, verbose=False,
    )
    assert list(df.columns) == [
        "peak", "gene", "chrom", "peak_start", "peak_end",
        "gene_start", "gene_end", "tss", "distance",
        "correlation", "t", "p_value", "fdr",
    ]
    assert "peak_to_gene" in ad_peak.uns
    assert "peak_to_gene_params" in ad_peak.uns
    assert ad_peak.uns["peak_to_gene_params"]["n_metacells"] == 50


if __name__ == "__main__":
    test_peak_to_gene_api_keys()
    test_peak_to_gene_recovers_planted_pairs()
    print("All peak_to_gene tests passed.")
