"""
imaging_transcriptomics.py  —  NeuroGen-4D (rigorous rebuild)

THE legitimate, novel core that replaces the (invalid) per-subject "brain-genome
CLIP bridge". ABIDE has no per-subject genomes, so per-subject genomic fusion is
impossible. Instead we test a real, spatially-resolved hypothesis:

    Does the cortical topography of ASD functional-connectivity disruption
    spatially align with the AHBA expression gradient of high-confidence
    autism-risk genes, BEYOND spatial-autocorrelation null models — and which
    cell types drive that alignment?

This is the imaging-transcriptomics paradigm (Fornito/Vertes/Seidlitz/Morgan/
Arnatkeviciute). Significance is assessed with a from-scratch spin test
(Alexander-Bloch 2018 / Vazquez-Rodriguez 2019), the correct null for cortical
maps with strong spatial autocorrelation. No network access required.

Outputs (rigorous/results/):
    disruption_map.npy            (100,)  per-region ASD-vs-CTRL GLM t-stat
    sfari_expr_map.npy            (100,)  mean AHBA expr of SFARI risk genes
    celltype_expr_maps.npy        (4,100) cell-type-weighted expression maps
    schaefer_centroids.npy        (100,3) MNI centroids (for spin test)
    concordance_results.json      real Spearman r + spin-test p-values
"""

import os, json, re, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
RESDIR = os.path.join(HERE, "results")
os.makedirs(RESDIR, exist_ok=True)

DATA_DIR = os.path.join(ROOT, "processed_matrices")
PHENO    = os.path.join(ROOT, "Phenotypic_V1_0b_preprocessed1.csv")
CONT_GRAPH = os.path.join(ROOT, "sfari_ppi_graph_continuous.pt")
CELL_CACHE = os.path.join(ROOT, "cell_expr_matrix_cache.npy")

N_ROI    = 100
FD_THRESH = 0.5      # mm; standard motion QC (subjects above are excluded)
N_SPIN   = 1000
SEED     = 42
rng = np.random.default_rng(SEED)


# ───────────────────────── 1. Phenotype + motion ────────────────────────────
def load_phenotype():
    df = pd.read_csv(PHENO)
    df["SUB_ID"] = df["SUB_ID"].astype(int).astype(str)
    keep = ["SUB_ID","SITE_ID","DX_GROUP","AGE_AT_SCAN","SEX","FIQ",
            "func_mean_fd","func_perc_fd"]
    df = df[keep].copy()
    # ASD=1, CTRL=0  (ABIDE encodes 1=autism, 2=control)
    df["ASD"] = (df["DX_GROUP"].astype(int) == 1).astype(int)
    return df.set_index("SUB_ID")


# ───────────────────────── 2. Signed Fisher-z FC ─────────────────────────────
def signed_fc_strength(ts):
    """Signed Fisher-z FC -> per-region weighted degree (keeps anti-correlation)."""
    ts = np.asarray(ts, dtype=np.float64)
    # standardize each region's timeseries
    ts = (ts - ts.mean(0, keepdims=True)) / (ts.std(0, keepdims=True) + 1e-8)
    r = np.corrcoef(ts, rowvar=False)
    r = np.nan_to_num(r, nan=0.0)
    np.clip(r, -0.999, 0.999, out=r)
    z = np.arctanh(r)
    np.fill_diagonal(z, 0.0)
    strength = z.sum(1) / (N_ROI - 1)          # signed mean connectivity per ROI
    return strength


def build_subject_table(pheno):
    rows, strengths = [], []
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npy"))
    for f in files:
        nums = re.findall(r"\d+", f)
        if not nums:
            continue
        sid = str(int("".join(nums)))            # e.g. Pitt_50042_... -> 50042
        if sid not in pheno.index:
            continue
        row = pheno.loc[sid]
        if isinstance(row, pd.DataFrame):        # guard against dup ids
            row = row.iloc[0]
        fd = row["func_mean_fd"]
        if pd.isna(fd) or fd > FD_THRESH:        # motion QC
            continue
        ts = np.load(os.path.join(DATA_DIR, f))
        if ts.ndim != 2 or ts.shape[1] != N_ROI:
            continue
        strengths.append(signed_fc_strength(ts))
        rows.append(dict(SUB_ID=sid, site=row["SITE_ID"], ASD=int(row["ASD"]),
                         age=float(row["AGE_AT_SCAN"]), sex=int(row["SEX"]),
                         mean_fd=float(fd)))
    tab = pd.DataFrame(rows)
    S = np.vstack(strengths)                      # (N, 100)
    return tab, S


# ─────────────────── 3. Case-control disruption map (GLM) ────────────────────
def disruption_map(tab, S):
    """Per-region ASD-vs-CTRL t-stat, covarying age, sex, mean-FD, and site."""
    import statsmodels.formula.api as smf
    site_d = pd.get_dummies(tab["site"], prefix="site", drop_first=True).astype(float)
    base = pd.concat([tab[["ASD","age","sex","mean_fd"]].reset_index(drop=True),
                      site_d.reset_index(drop=True)], axis=1)
    covars = "+".join(["age","sex","mean_fd"] + list(site_d.columns))
    tvals = np.zeros(N_ROI)
    for r in range(N_ROI):
        d = base.copy()
        d["y"] = S[:, r]
        m = smf.ols(f"y ~ ASD + {covars}", data=d).fit()
        tvals[r] = m.tvalues["ASD"]
    return tvals


# ─────────────────── 4. Gene-expression cortical maps ────────────────────────
def load_expression_maps():
    import torch
    g = torch.load(CONT_GRAPH, map_location="cpu")     # torch 1.12: no weights_only
    X = g.x.numpy().astype(np.float64)                  # (668 genes, 100 regions)
    gene_map = g.gene_mapping                           # idx -> symbol
    present = X.std(1) > 1e-9                            # genes with real AHBA probes
    Xp = X[present]
    sfari_map = Xp.mean(0)                              # aggregate SFARI expr topography
    # Cell-type-weighted maps (Velmeshev): w (668x4) raw -> log1p
    cells = ["Excitatory","Inhibitory","Astrocyte","Microglia"]
    ct_maps = np.zeros((4, N_ROI))
    if os.path.exists(CELL_CACHE):
        W = np.log1p(np.load(CELL_CACHE).astype(np.float64))   # (668,4)
        Wp = W[present]
        for c in range(4):
            w = Wp[:, c]
            if w.sum() > 0:
                ct_maps[c] = (Xp * w[:, None]).sum(0) / w.sum()
    return sfari_map, ct_maps, cells, int(present.sum()), X.shape[0]


# ─────────────────────── 5. Spin-test spatial null ───────────────────────────
def schaefer_centroids():
    from nilearn import datasets
    from nilearn.plotting import find_parcellation_cut_coords
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=N_ROI, yeo_networks=7)
    coords = find_parcellation_cut_coords(atlas.maps)          # (100,3) MNI
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]
    hemi = np.array([0 if "_LH_" in l else 1 for l in labels])  # 0=L,1=R
    return np.asarray(coords), hemi


def _rand_rotation():
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def spin_permutations(coords, hemi, n=N_SPIN):
    """Vazquez-Rodriguez nearest-neighbour spin; returns (n, 100) index perms."""
    perms = np.zeros((n, N_ROI), dtype=int)
    idxL, idxR = np.where(hemi == 0)[0], np.where(hemi == 1)[0]
    cL, cR = coords[idxL], coords[idxR]
    for k in range(n):
        Rot = _rand_rotation()
        refl = np.diag([-1, 1, 1])                  # mirror L<->R convention
        permk = np.arange(N_ROI)
        for idx, c, rot in [(idxL, cL, Rot), (idxR, cR, refl @ Rot @ refl)]:
            rc = c @ rot.T
            D = np.linalg.norm(rc[:, None, :] - c[None, :, :], axis=2)
            nn = D.argmin(1)                         # nearest original for each rotated
            permk[idx] = idx[nn]
        perms[k] = permk
    return perms


def spin_test(map_a, map_b, perms):
    """Spearman concordance of map_a vs map_b with spin null on map_b."""
    obs = stats.spearmanr(map_a, map_b).correlation
    rank_a = stats.rankdata(map_a)
    null = np.array([stats.spearmanr(rank_a, map_b[p]).correlation for p in perms])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return float(obs), float(p), null


# ─────────────────────────────── main ───────────────────────────────────────
def main():
    print("=" * 70)
    print("  NeuroGen-4D | Imaging-Transcriptomics of ASD (spin-test gated)")
    print("=" * 70)

    pheno = load_phenotype()
    print("\n[1] Building signed-FC subject table (motion QC: mean FD < %.1f mm)..." % FD_THRESH)
    tab, S = build_subject_table(pheno)
    print(f"    Retained {len(tab)} subjects  | ASD {tab.ASD.sum()} / CTRL {(tab.ASD==0).sum()}"
          f"  | sites {tab.site.nunique()}")
    print(f"    Mean FD: ASD {tab[tab.ASD==1].mean_fd.mean():.3f} vs "
          f"CTRL {tab[tab.ASD==0].mean_fd.mean():.3f} "
          f"(p={stats.ttest_ind(tab[tab.ASD==1].mean_fd, tab[tab.ASD==0].mean_fd).pvalue:.2e})")

    print("\n[2] Case-control disruption map (per-region GLM, covarying age/sex/FD/site)...")
    dmap = disruption_map(tab, S)
    print(f"    |t| range {np.abs(dmap).min():.2f}-{np.abs(dmap).max():.2f}; "
          f"regions |t|>2: {(np.abs(dmap)>2).sum()}/100")

    print("\n[3] AHBA expression maps (SFARI risk genes + cell-type weighted)...")
    sfari_map, ct_maps, cells, n_present, n_total = load_expression_maps()
    print(f"    SFARI genes with AHBA probes: {n_present}/{n_total}")

    print("\n[4] Schaefer-100 centroids + %d spin permutations..." % N_SPIN)
    coords, hemi = schaefer_centroids()
    perms = spin_permutations(coords, hemi, N_SPIN)

    print("\n[5] Spatial concordance (Spearman) with SPIN-TEST nulls")
    print("    " + "-" * 60)
    r_sfari, p_sfari, _ = spin_test(dmap, sfari_map, perms)
    p_naive = stats.spearmanr(dmap, sfari_map).pvalue
    print(f"    ASD disruption  vs  SFARI expression : r={r_sfari:+.3f}  "
          f"p_spin={p_sfari:.4f}  (naive p={p_naive:.4f})")

    ct_results = {}
    for c, name in enumerate(cells):
        if ct_maps[c].std() < 1e-9:
            continue
        r_c, p_c, _ = spin_test(dmap, ct_maps[c], perms)
        ct_results[name] = dict(r=r_c, p_spin=p_c)
        print(f"      {name:12s} expression       : r={r_c:+.3f}  p_spin={p_c:.4f}")

    # Save real arrays for honest figures
    np.save(os.path.join(RESDIR, "disruption_map.npy"), dmap)
    np.save(os.path.join(RESDIR, "sfari_expr_map.npy"), sfari_map)
    np.save(os.path.join(RESDIR, "celltype_expr_maps.npy"), ct_maps)
    np.save(os.path.join(RESDIR, "schaefer_centroids.npy"), coords)
    tab.to_csv(os.path.join(RESDIR, "qc_subject_table.csv"), index=False)
    out = dict(n_subjects=int(len(tab)), n_asd=int(tab.ASD.sum()),
               n_ctrl=int((tab.ASD == 0).sum()), fd_thresh=FD_THRESH, n_spin=N_SPIN,
               sfari=dict(r=r_sfari, p_spin=p_sfari, p_naive=float(p_naive)),
               celltypes=ct_results, sfari_genes_with_probes=n_present)
    with open(os.path.join(RESDIR, "concordance_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[6] Saved real arrays + concordance_results.json to rigorous/results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
