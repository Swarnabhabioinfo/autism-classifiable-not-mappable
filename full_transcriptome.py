"""
full_transcriptome.py — transcriptome-wide imaging-transcriptomics (offline AHBA).

Canonical test (Morgan 2019 PNAS / Romero-Garcia / Vertes): correlate EVERY AHBA
gene's cortical expression with the ASD functional-disruption map, then ask
whether high-confidence autism-risk (SFARI) genes are ENRICHED among the
spatially-coupled genes — with a spin-test null on the enrichment statistic so
spatial autocorrelation cannot create false positives.

Left-hemisphere Schaefer-100 only (AHBA right-hemisphere coverage is sparse;
this is the field-standard restriction). Uses the locally cached 6-donor AHBA.

Run once; report whatever it yields.
"""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

import imaging_transcriptomics as it

RESDIR  = it.RESDIR
AHBA_CSV = os.path.join(RESDIR, "ahba_region_gene_LH.csv")
N_SPIN  = 1000


def get_ahba():
    if os.path.exists(AHBA_CSV):
        print("[ahba] loading cached region x gene matrix...")
        return pd.read_csv(AHBA_CSV, index_col=0)
    print("[ahba] running abagen on cached 6-donor AHBA (slow, ~10-15 min)...")
    import abagen
    from nilearn import datasets
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=it.N_ROI, yeo_networks=7)
    expr = abagen.get_expression_data(atlas.maps, lr_mirror="bidirectional",
                                      missing="interpolate", verbose=1)
    expr.to_csv(AHBA_CSV)
    return expr


def lh_indices():
    coords, hemi = it.schaefer_centroids()
    return np.where(hemi == 0)[0]          # left-hemisphere Schaefer indices


def main():
    print("=" * 70)
    print("  Transcriptome-wide imaging-transcriptomics (spin-gated enrichment)")
    print("=" * 70)

    dmap = np.load(os.path.join(RESDIR, "disruption_map.npy"))   # (100,)
    expr = get_ahba()                                            # regions x genes
    expr = expr.dropna(axis=0, how="all").dropna(axis=1, how="any")
    # align region order: abagen rows are region labels 1..100 in atlas order
    region_ids = np.array([int(r) for r in expr.index]) - 1
    lh = set(lh_indices().tolist())
    keep = np.array([rid in lh for rid in region_ids])
    expr = expr.iloc[keep]
    d_lh = dmap[region_ids[keep]]
    genes = np.array(expr.columns)
    E = expr.values.astype(np.float64)                          # (n_reg, n_gene)
    print(f"[data] {E.shape[0]} LH regions x {E.shape[1]} genes")

    # SFARI membership among AHBA genes
    import torch
    g = torch.load(it.CONT_GRAPH, map_location="cpu")
    sfari = {str(g.gene_mapping[i]).upper() for i in range(len(g.gene_mapping))}
    is_sfari = np.array([sym.upper() in sfari for sym in genes])
    print(f"[data] SFARI genes present in AHBA: {is_sfari.sum()} / {len(sfari)}")

    # per-gene Spearman across LH regions (vectorised via ranks)
    def colrank_z(M):
        R = np.apply_along_axis(stats.rankdata, 0, M)
        return (R - R.mean(0)) / (R.std(0) + 1e-12)
    Ez = colrank_z(E)                                           # (n_reg, n_gene)
    yz = stats.rankdata(d_lh); yz = (yz - yz.mean()) / (yz.std() + 1e-12)
    r_gene = (Ez.T @ yz) / len(yz)                              # (n_gene,)

    # enrichment statistic: AUC (Mann-Whitney) of |r| SFARI vs non-SFARI
    def auc_stat(rvec):
        a = np.abs(rvec[is_sfari]); b = np.abs(rvec[~is_sfari])
        u = stats.mannwhitneyu(a, b, alternative="greater").statistic
        return u / (len(a) * len(b))                            # AUC in [0,1]
    obs_auc = auc_stat(r_gene)
    obs_p_mwu = stats.mannwhitneyu(np.abs(r_gene[is_sfari]),
                                   np.abs(r_gene[~is_sfari]),
                                   alternative="greater").pvalue

    # spin null on the enrichment statistic (recompute r_gene for each spun map)
    coords, hemi = it.schaefer_centroids()
    # spin restricted to LH regions only
    lh_idx = lh_indices()
    cL = coords[lh_idx]
    null_auc = np.empty(N_SPIN)
    rng = np.random.default_rng(11)
    for k in range(N_SPIN):
        Rot = it._rand_rotation()
        rc = cL @ Rot.T
        D = np.linalg.norm(rc[:, None, :] - cL[None, :, :], axis=2)
        perm = D.argmin(1)
        # map perm (over lh_idx order) to expr region order
        # expr regions are region_ids[keep]; build position map
        pos = {rid: j for j, rid in enumerate(region_ids[keep])}
        # spun disruption over expr regions
        spun = np.empty(len(d_lh))
        for j, rid in enumerate(region_ids[keep]):
            # rid is a LH region; find its index within lh_idx, apply perm, map back
            li = np.where(lh_idx == rid)[0][0]
            target_rid = lh_idx[perm[li]]
            spun[j] = d_lh[pos[target_rid]] if target_rid in pos else d_lh[j]
        sz = stats.rankdata(spun); sz = (sz - sz.mean()) / (sz.std() + 1e-12)
        r_spun = (Ez.T @ sz) / len(sz)
        null_auc[k] = auc_stat(r_spun)
    p_spin = (np.sum(null_auc >= obs_auc) + 1) / (N_SPIN + 1)

    print("\n[result] SFARI enrichment among spatially-coupled genes")
    print(f"    AUC(|r| SFARI > rest) = {obs_auc:.3f}  (0.5=no enrichment)")
    print(f"    naive Mann-Whitney p  = {obs_p_mwu:.4f}")
    print(f"    SPIN-test p           = {p_spin:.4f}   (null AUC mean {null_auc.mean():.3f})")

    top = np.argsort(np.abs(r_gene))[::-1][:20]
    print("\n    Top transcriptome-wide spatially-coupled genes (|r|):")
    print("    " + ", ".join(f"{genes[i]}{'*' if is_sfari[i] else ''}({r_gene[i]:+.2f})"
                             for i in top[:12]) + "    (*=SFARI)")

    out = dict(n_lh_regions=int(E.shape[0]), n_genes=int(E.shape[1]),
               n_sfari_in_ahba=int(is_sfari.sum()),
               enrichment_auc=float(obs_auc), p_naive=float(obs_p_mwu),
               p_spin=float(p_spin), null_auc_mean=float(null_auc.mean()),
               top_genes=[dict(gene=str(genes[i]), r=float(r_gene[i]),
                               sfari=bool(is_sfari[i])) for i in top])
    np.save(os.path.join(RESDIR, "transcriptome_r.npy"), r_gene)
    with open(os.path.join(RESDIR, "transcriptome_enrichment.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    verdict = ("SIGNIFICANT SFARI spatial enrichment" if p_spin < 0.05
               else "NULL (no SFARI enrichment beyond spatial autocorrelation)")
    print(f"\nVERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
