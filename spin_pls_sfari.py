"""
spin_pls_sfari.py — gene-resolved spatial concordance (pre-specified, run once).

Two complementary, spin-test-gated analyses of whether SFARI risk-gene cortical
expression tracks the ASD functional-disruption map:

  (A) Per-gene concordance: Spearman(disruption, gene_expr) for each SFARI gene,
      spin-test null (1000), BH-FDR across genes. Counts genes surviving.
  (B) Spin-refit PLS1: best weighted SFARI combination vs disruption; null is
      built by REFITTING PLS on each spun map (controls both spatial
      autocorrelation AND the >p>n overfitting). Bootstrap gene weights -> Z.

This is the standard Whitaker/Vertes/Morgan procedure adapted to be fully
offline. We report whatever it yields — significant or null.
"""
import os, json, numpy as np
from scipy import stats
from sklearn.cross_decomposition import PLSRegression

import imaging_transcriptomics as it   # reuse loaders + spin machinery

RESDIR = it.RESDIR
N_SPIN = 1000
rng = np.random.default_rng(7)


def main():
    dmap = np.load(os.path.join(RESDIR, "disruption_map.npy"))           # (100,)
    import torch
    g = torch.load(it.CONT_GRAPH, map_location="cpu")
    X = g.x.numpy().astype(np.float64)                                   # (668,100)
    gmap = g.gene_mapping
    present = X.std(1) > 1e-9
    genes = [gmap[i] for i in range(X.shape[0]) if present[i]]
    G = X[present]                                                       # (n_genes,100)
    print(f"[setup] {G.shape[0]} SFARI genes with AHBA probes; disruption map 100 ROIs")

    coords, hemi = it.schaefer_centroids()
    perms = it.spin_permutations(coords, hemi, N_SPIN)                   # (1000,100)

    # ── (A) vectorised per-gene spin concordance ────────────────────────────
    def zrank(v):
        r = stats.rankdata(v); return (r - r.mean()) / (r.std() + 1e-12)
    Gz = np.vstack([zrank(G[i]) for i in range(G.shape[0])])            # (n,100)
    yz = zrank(dmap)
    obs = (Gz @ yz) / len(yz)                                           # (n,) Spearman r
    null = np.empty((N_SPIN, G.shape[0]))
    for k, p in enumerate(perms):
        null[k] = (Gz @ yz[p]) / len(yz)
    pvals = (np.sum(np.abs(null) >= np.abs(obs), axis=0) + 1) / (N_SPIN + 1)
    # BH-FDR
    order = np.argsort(pvals); m = len(pvals)
    bh = np.empty(m); cummin = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = m - rank
        cummin = min(cummin, pvals[idx] * m / i); bh[idx] = cummin
    n_sig = int((bh < 0.05).sum())
    top = np.argsort(np.abs(obs))[::-1][:15]
    print(f"\n(A) Per-gene spin concordance: {n_sig}/{len(genes)} genes FDR<0.05")
    print("    top |r| genes:  " + ", ".join(
        f"{genes[i]}({obs[i]:+.2f},q={bh[i]:.3f})" for i in top[:10]))

    # ── (B) spin-refit PLS1 ─────────────────────────────────────────────────
    Xpls = G.T                                                          # (100 regions, n genes)
    def pls_r(y):
        pls = PLSRegression(n_components=1, scale=True)
        s = pls.fit_transform(Xpls, y)[0].ravel()
        return abs(stats.pearsonr(s, y)[0]), pls
    obs_r, pls = pls_r(dmap)
    null_r = np.array([pls_r(dmap[p])[0] for p in perms])
    p_pls = (np.sum(null_r >= obs_r) + 1) / (N_SPIN + 1)
    print(f"\n(B) Spin-refit PLS1: r={obs_r:.3f}  p_spin={p_pls:.4f}  "
          f"(null mean {null_r.mean():.3f})")

    # bootstrap gene weights (region resampling)
    W = []
    for _ in range(500):
        idx = rng.integers(0, 100, 100)
        try:
            p = PLSRegression(n_components=1, scale=True).fit(Xpls[idx], dmap[idx])
            W.append(p.x_weights_.ravel())
        except Exception:
            pass
    W = np.vstack(W); Z = W.mean(0) / (W.std(0) + 1e-12)
    topZ = np.argsort(np.abs(Z))[::-1][:15]
    print("    top PLS driver genes (|Z|): " + ", ".join(
        f"{genes[i]}({Z[i]:+.1f})" for i in topZ[:10]))

    out = dict(per_gene=dict(n_sig_fdr05=n_sig, n_genes=len(genes),
                             top=[dict(gene=genes[i], r=float(obs[i]), q=float(bh[i]))
                                  for i in top]),
               pls=dict(r=float(obs_r), p_spin=float(p_pls),
                        drivers=[dict(gene=genes[i], Z=float(Z[i])) for i in topZ]))
    with open(os.path.join(RESDIR, "sfari_generesolved_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    verdict = ("SIGNIFICANT spatial concordance" if (p_pls < 0.05 or n_sig > 0)
               else "NULL (no SFARI spatial concordance beyond autocorrelation)")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
