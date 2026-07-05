"""
aggregate_lh_concordance.py — unify the aggregate SFARI concordance (and the
cell-type controls) onto the SAME abagen left-hemisphere (Schaefer-100 LH, 50
regions) pipeline already used for the per-gene, PLS and transcriptome-wide tests.

This removes the last dependency on the pre-rigorous 100-region expression graph
for the headline brain-gene null (Figs 4, 7b). The disruption maps are unchanged
(the ABIDE-I map is loaded from disruption_map.npy; the ABIDE-II map is rebuilt
exactly as in replicate_abide2.py); only the SFARI/cell-type expression maps and
the spin null are moved to the documented abagen LH pipeline.

Outputs (results/):
    concordance_lh.json        ABIDE-I & ABIDE-II aggregate r/p_spin + cell types
    disruption_map_lh.npy      (50,) ABIDE-I disruption t-stat, LH regions
    sfari_expr_map_lh.npy      (50,) aggregate SFARI (cat 1-2) expression, LH
    concordance_lh_null.npy    (1000,) ABIDE-I spin-null distribution (LH)
"""
import os, sys, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from scipy import stats
import imaging_transcriptomics as it
import replicate_abide2 as ra

R = it.RESDIR
N_SPIN = 1000

coords, hemi = it.schaefer_centroids()
lh_idx = np.where(hemi == 0)[0]                       # 50 LH Schaefer indices

# ---- abagen LH expression (cached, differential-stability, bidirectional mirror) ----
expr = pd.read_csv(os.path.join(R, "ahba_region_gene_LH.csv"), index_col=0)
expr = expr.dropna(axis=0, how="all").dropna(axis=1, how="any")
region_ids = np.array([int(r) for r in expr.index]) - 1
keep = np.array([rid in set(lh_idx.tolist()) for rid in region_ids])
expr = expr.iloc[keep]
lh_regions = region_ids[keep]                        # 0-based Schaefer idx, len 50
genes_ab = np.array([str(c).upper() for c in expr.columns])
E = expr.values.astype(np.float64)                   # (50, n_gene)
pos = {int(rid): j for j, rid in enumerate(lh_regions)}
print(f"[abagen] {E.shape[0]} LH regions x {E.shape[1]} genes")

# ---- SFARI (categories 1-2) gene set + Velmeshev cell-type weights ----
g = torch.load(it.CONT_GRAPH, map_location="cpu")
sfari_syms = [str(g.gene_mapping[i]).upper() for i in range(len(g.gene_mapping))]
sfari_set = set(sfari_syms)
is_sfari = np.array([s in sfari_set for s in genes_ab])
sfari_lh = E[:, is_sfari].mean(1)                    # (50,) aggregate SFARI expr
print(f"[sfari] {is_sfari.sum()} SFARI(cat1-2) genes present in abagen LH")

W = np.log1p(np.load(it.CELL_CACHE).astype(np.float64))  # (668,4) Velmeshev weights
sym2w = {sfari_syms[i]: W[i] for i in range(len(sfari_syms))}
cells = ["Excitatory", "Inhibitory", "Astrocyte", "Microglia"]
sfari_ab = genes_ab[is_sfari]
w_mat = np.array([sym2w.get(s, np.zeros(4)) for s in sfari_ab])   # (n_sfari,4)
E_sfari = E[:, is_sfari]
ct_lh = np.zeros((4, len(lh_regions)))
for c in range(4):
    w = w_mat[:, c]
    if w.sum() > 0:
        ct_lh[c] = (E_sfari * w[None, :]).sum(1) / w.sum()


def lh_spin(dmap_lh, gene_lh, seed=11, nperm=N_SPIN):
    """Spin gene_lh within LH; hold disruption ranks fixed (matches it.spin_test)."""
    cL = coords[lh_idx]
    obs = stats.spearmanr(dmap_lh, gene_lh).correlation
    rank_d = stats.rankdata(dmap_lh)
    rng = np.random.default_rng(seed); null = np.empty(nperm)
    li_of = {int(rid): int(np.where(lh_idx == rid)[0][0]) for rid in lh_regions}
    for k in range(nperm):
        Rot = it._rand_rotation(); rc = cL @ Rot.T
        perm = np.linalg.norm(rc[:, None, :] - cL[None, :, :], axis=2).argmin(1)
        spun = np.empty(len(gene_lh))
        for j, rid in enumerate(lh_regions):
            target = int(lh_idx[perm[li_of[int(rid)]]])
            spun[j] = gene_lh[pos[target]] if target in pos else gene_lh[j]
        null[k] = stats.spearmanr(rank_d, spun).correlation
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)
    return float(obs), float(p), null


# ---- ABIDE-I aggregate + cell types (LH) ----
dmap1 = np.load(os.path.join(R, "disruption_map.npy"))       # (100,)
dmap1_lh = dmap1[lh_regions]
r1, p1, null1 = lh_spin(dmap1_lh, sfari_lh)
print(f"\n[ABIDE-I  LH-50] aggregate SFARI  r={r1:+.4f}  p_spin={p1:.4f}")
ct_res = {}
for c, nm in enumerate(cells):
    if ct_lh[c].std() < 1e-9:
        continue
    rc, pc, _ = lh_spin(dmap1_lh, ct_lh[c])
    ct_res[nm] = dict(r=rc, p_spin=pc)
    print(f"    {nm:12s}  r={rc:+.4f}  p_spin={pc:.4f}")

# ---- ABIDE-II aggregate (LH); rebuild disruption exactly as replicate_abide2 ----
print("\n[ABIDE-II] rebuilding disruption map (LH) ...")
X2, y2, s2, age2, sex2, S2 = ra.load_abide2()
site_clean = np.array([re.sub(r"[^A-Za-z0-9]", "_", x) for x in s2])
tab = pd.DataFrame(dict(ASD=y2, age=age2, sex=sex2, site=site_clean,
                        mean_fd=np.zeros(len(y2))))
tab = tab.dropna(subset=["age"])
S2v = S2[tab.index.values]; tab = tab.reset_index(drop=True)
dmap2 = it.disruption_map(tab, S2v)
dmap2_lh = dmap2[lh_regions]
r2, p2, null2 = lh_spin(dmap2_lh, sfari_lh)
print(f"[ABIDE-II LH-50] aggregate SFARI  r={r2:+.4f}  p_spin={p2:.4f}")

# ---- biotype transcriptomic concordance (LH) — re-based off the .pt map ----
print("\n[biotypes LH-50] per-biotype deviation-map concordance")
Z = np.load(os.path.join(R, "asd_deviation_maps.npy"))     # (n_asd, 100)
lab = np.load(os.path.join(R, "biotype_labels.npy"))       # (n_asd,)
bt_res = {}
for c in sorted(set(int(x) for x in lab)):
    dev_lh = Z[lab == c][:, lh_regions].mean(0)
    rb, pb, _ = lh_spin(dev_lh, sfari_lh)
    bt_res[f"biotype{c}"] = dict(r=float(rb), p_spin=float(pb), n=int((lab == c).sum()))
    print(f"    biotype{c} (n={int((lab==c).sum())})  r={rb:+.4f}  p_spin={pb:.4f}")

# 95% spin-null intervals (per cohort) + pooled band for Fig 7b
ci1 = [float(np.percentile(null1, 2.5)), float(np.percentile(null1, 97.5))]
ci2 = [float(np.percentile(null2, 2.5)), float(np.percentile(null2, 97.5))]
pooled = np.concatenate([null1, null2])
band = [float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))]
print(f"[null 95%] ABIDE-I {ci1}  ABIDE-II {ci2}  pooled band {band}")

# ---- save ----
np.save(os.path.join(R, "disruption_map_lh.npy"), dmap1_lh)
np.save(os.path.join(R, "sfari_expr_map_lh.npy"), sfari_lh)
np.save(os.path.join(R, "concordance_lh_null.npy"), null1)
out = dict(n_lh_regions=int(len(lh_regions)), n_sfari=int(is_sfari.sum()),
           abide1=dict(r=r1, p_spin=p1, null_ci=ci1),
           abide2=dict(r=r2, p_spin=p2, null_ci=ci2),
           null_band_95=band, celltypes=ct_res, biotypes=bt_res)
json.dump(out, open(os.path.join(R, "concordance_lh.json"), "w"), indent=2)
print("\n[saved] concordance_lh.json + LH arrays")
