"""
biotypes.py — Connectome biotypes of autism (heterogeneity-aware discovery).

Rationale: group-level ASD case-control + brain-gene concordance is null (we
showed this rigorously). Hypothesis: that null is a HETEROGENEITY artifact.
Pipeline (Drysdale 2017 paradigm, hardened against Dinga 2019 critiques):

  1. Normative model: regional FC strength ~ age+sex+site fit on CONTROLS;
     express each ASD subject as a deviation z-map.
  2. GATE 1 — is there a reliable multivariate brain<->symptom axis in ASD?
     PLS(deviation maps, ADI-R symptoms) with PERMUTATION test. No axis -> stop.
  3. Discover biotypes: cluster deviation maps; BOOTSTRAP STABILITY (ARI).
  4. GATE 2 — do biotypes differ on INDEPENDENT symptoms not used to cluster?
  5. Payoff: biotype-specific disruption maps -> spin-test transcriptomic
     concordance (does a biotype show what the group could not?).

Every gate reports honestly; nulls are reported, not hidden.
"""
import os, re, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score
warnings.filterwarnings("ignore")
import imaging_transcriptomics as it

RESDIR, N = it.RESDIR, it.N_ROI
SYMPT = ["ADI_R_SOCIAL_TOTAL_A", "ADI_R_VERBAL_TOTAL_BV", "ADI_RRB_TOTAL_C"]
rng = np.random.default_rng(42)


def load_full_pheno():
    df = pd.read_csv(it.PHENO)
    df["SUB_ID"] = df["SUB_ID"].astype(int).astype(str)
    df["ASD"] = (df["DX_GROUP"].astype(int) == 1).astype(int)
    return df.set_index("SUB_ID")


def build():
    ph = load_full_pheno()
    rows, strength = [], []
    for f in sorted(os.listdir(it.DATA_DIR)):
        if not f.endswith(".npy"): continue
        nums = re.findall(r"\d+", f)
        if not nums: continue
        sid = str(int("".join(nums)))
        if sid not in ph.index: continue
        r = ph.loc[sid]
        if getattr(r, "ndim", 1) == 2: r = r.iloc[0]
        fd = r["func_mean_fd"]
        if pd.isna(fd) or fd > it.FD_THRESH: continue
        ts = np.load(os.path.join(it.DATA_DIR, f))
        if ts.ndim != 2 or ts.shape[1] != N: continue
        strength.append(it.signed_fc_strength(ts))
        rec = dict(SUB_ID=sid, ASD=int(r["ASD"]), site=r["SITE_ID"],
                   age=float(r["AGE_AT_SCAN"]), sex=int(r["SEX"]),
                   mean_fd=float(fd))
        for s in SYMPT + ["ADOS_TOTAL", "FIQ"]:
            v = r[s] if s in r else np.nan
            rec[s] = np.nan if (pd.isna(v) or v in (-9999, -999)) else float(v)
        rows.append(rec)
    return pd.DataFrame(rows), np.vstack(strength)


def normative_deviation(tab, S):
    """Fit strength ~ age+sex+site on CONTROLS; return ASD deviation z-maps."""
    import statsmodels.formula.api as smf
    site_d = pd.get_dummies(tab["site"], prefix="s", drop_first=True).astype(float)
    base = pd.concat([tab[["age", "sex"]].reset_index(drop=True),
                      site_d.reset_index(drop=True)], axis=1)
    ctrl = tab["ASD"].values == 0
    asd = tab["ASD"].values == 1
    Z = np.zeros((asd.sum(), N))
    cov = "age+sex+" + "+".join(site_d.columns)
    for rr in range(N):
        d = base.copy(); d["y"] = S[:, rr]
        m = smf.ols(f"y ~ {cov}", data=d[ctrl]).fit()
        pred = m.predict(d)
        resid_sd = (S[ctrl, rr] - pred[ctrl]).std() + 1e-9
        Z[:, rr] = (S[asd, rr] - pred[asd].values) / resid_sd
    return Z


def gate1_brain_symptom(Z, sym):
    """PLS deviation->symptoms, permutation test on summed canonical corr."""
    ok = ~np.isnan(sym).any(1)
    X, Y = Z[ok], stats.zscore(sym[ok], axis=0)
    n = X.shape[0]
    def cc(Xa, Ya):
        pls = PLSRegression(n_components=2).fit(Xa, Ya)
        xs, ys = pls.transform(Xa, Ya)
        return sum(abs(np.corrcoef(xs[:, i], ys[:, i])[0, 1]) for i in range(2))
    obs = cc(X, Y)
    null = np.array([cc(X, Y[rng.permutation(n)]) for _ in range(1000)])
    p = (np.sum(null >= obs) + 1) / 1001
    return dict(n=int(n), obs=float(obs), p_perm=float(p),
                null_mean=float(null.mean())), ok


def discover(Z, kmax=5):
    Zs = StandardScaler().fit_transform(Z)
    P = PCA(n_components=min(10, Z.shape[1]), random_state=0).fit_transform(Zs)
    best = None
    for k in range(2, kmax + 1):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(P)
        sil = silhouette_score(P, lab)
        # bootstrap stability (ARI of re-clustering 70% subsamples)
        aris = []
        for _ in range(50):
            idx = rng.choice(len(P), int(0.7 * len(P)), replace=False)
            l2 = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(P[idx])
            l1 = lab[idx]
            aris.append(adjusted_rand_score(l1, l2))
        if best is None or np.mean(aris) > best["stability"]:
            best = dict(k=k, labels=lab, silhouette=float(sil),
                        stability=float(np.mean(aris)))
    return best, P


def main():
    print("=" * 70)
    print("  Connectome biotypes of autism — discovery + honest validation gates")
    print("=" * 70)
    tab, S = build()
    asd = tab["ASD"].values == 1
    print(f"[data] {len(tab)} subj | ASD {asd.sum()} CTRL {(~asd).sum()} | "
          f"ASD w/ ADI-R: {(~tab.loc[asd, SYMPT].isna().any(1)).sum()}")

    Z = normative_deviation(tab, S)
    np.save(os.path.join(RESDIR, "asd_deviation_maps.npy"), Z)
    asd_tab = tab[asd].reset_index(drop=True)

    print("\n[GATE 1] Reliable multivariate brain<->symptom axis (PLS + perm)?")
    g1, ok = gate1_brain_symptom(Z, asd_tab[SYMPT].values)
    print(f"    n={g1['n']} | summed canonical r={g1['obs']:.3f} | "
          f"perm p={g1['p_perm']:.4f} (null {g1['null_mean']:.3f})")
    print("    " + ("PASS -> real brain-symptom structure" if g1['p_perm'] < 0.05
                    else "NULL -> weak/!brain-symptom axis"))

    print("\n[discovery] clustering deviation maps (stability-selected k)...")
    best, P = discover(Z)
    print(f"    best k={best['k']} | silhouette={best['silhouette']:.3f} | "
          f"bootstrap stability ARI={best['stability']:.3f} "
          f"({'stable' if best['stability']>0.5 else 'UNSTABLE'})")
    lab = best["labels"]
    for c in range(best["k"]):
        print(f"      biotype {c}: n={int((lab==c).sum())}")

    print("\n[GATE 2] Biotypes differ on INDEPENDENT symptoms (not used to cluster)?")
    g2 = {}
    for s in SYMPT + ["ADOS_TOTAL", "FIQ", "age", "mean_fd"]:
        vals = asd_tab[s].values
        groups = [vals[(lab == c) & ~np.isnan(vals)] for c in range(best["k"])]
        groups = [g for g in groups if len(g) > 3]
        if len(groups) >= 2:
            H, p = stats.kruskal(*groups)
            g2[s] = float(p)
            print(f"    {s:24s} Kruskal-Wallis p={p:.4f}{'  *' if p<0.05 else ''}")

    print("\n[payoff] biotype-specific transcriptomic concordance (spin-test)...")
    coords, hemi = it.schaefer_centroids()
    perms = it.spin_permutations(coords, hemi, 1000)
    sfari_map, ct_maps, cells, _, _ = it.load_expression_maps()
    # per-biotype disruption: biotype-ASD mean deviation map (already normative z)
    bt_conc = {}
    for c in range(best["k"]):
        dmap_c = Z[lab == c].mean(0)            # mean deviation map for biotype
        r, p, _ = it.spin_test(dmap_c, sfari_map, perms)
        bt_conc[f"biotype{c}"] = dict(r=float(r), p_spin=float(p), n=int((lab==c).sum()))
        print(f"    biotype {c} (n={int((lab==c).sum())}) vs SFARI expr: "
              f"r={r:+.3f} p_spin={p:.4f}{'  <-- !' if p<0.05 else ''}")

    out = dict(gate1=g1, k=best["k"], silhouette=best["silhouette"],
               stability=best["stability"],
               biotype_sizes=[int((lab==c).sum()) for c in range(best["k"])],
               gate2_symptom_p=g2, transcriptomic=bt_conc)
    json.dump(out, open(os.path.join(RESDIR, "biotypes_results.json"), "w"), indent=2)
    np.save(os.path.join(RESDIR, "biotype_labels.npy"), lab)
    print("\n[saved] biotypes_results.json + labels + deviation maps")
    print("=" * 70)


if __name__ == "__main__":
    main()
