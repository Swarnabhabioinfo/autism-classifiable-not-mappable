"""
replicate_abide2.py — independent replication in ABIDE-II (locally preprocessed).

Tests whether the two ABIDE-I findings reproduce in an INDEPENDENT cohort:
  1. Classification: within-ABIDE-II LOSO AUC, and the harder cross-cohort
     transfer (train ABIDE-I -> test ABIDE-II), raw and ComBat-harmonized.
  2. Imaging-transcriptomics null: ABIDE-II disruption map -> spin-test
     concordance with SFARI risk-gene expression.

Caveat (reported): ABIDE-II here uses a lighter local FSL pipeline than ABIDE-I's
CPAC; cross-cohort transfer is therefore also a pipeline-robustness test, and we
harmonize with ComBat.
"""
import os, re, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import imaging_transcriptomics as it
from combat import ComBat
import classify_eval as ce

A2_DIR = os.path.join(it.HERE, "abide2_matrices")
A2_PHENO = os.path.join(it.ROOT, "ABIDEII_Composite_Phenotypic.csv")
RESDIR = it.RESDIR; N = it.N_ROI
iu, ju = np.triu_indices(N, 1)


def fc_vec(ts):
    ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-8)
    r = np.nan_to_num(np.corrcoef(ts, rowvar=False), nan=0.0)
    np.clip(r, -0.999, 0.999, out=r); z = np.arctanh(r); np.fill_diagonal(z, 0)
    return z


def load_abide2():
    ph = pd.read_csv(A2_PHENO, encoding="latin1")
    ph.columns = ph.columns.str.strip()      # ABIDE-II has 'AGE_AT_SCAN ' (trailing space)
    ph["SUB_ID"] = ph["SUB_ID"].astype(str).str.replace(".0", "", regex=False)
    ph = ph.set_index("SUB_ID")
    fdcol = "func_mean_fd" if "func_mean_fd" in ph.columns else None
    X, y, site, age, sex, strength = [], [], [], [], [], []
    for f in sorted(os.listdir(A2_DIR)):
        if not f.endswith(".npy"): continue
        sub = re.findall(r"\d+", f)[0]
        if sub not in ph.index: continue
        r = ph.loc[sub]
        if getattr(r, "ndim", 1) == 2: r = r.iloc[0]
        ts = np.load(os.path.join(A2_DIR, f))
        if ts.ndim != 2 or ts.shape[1] != N: continue
        z = fc_vec(ts)
        X.append(z[iu, ju]); strength.append(z.sum(1) / (N - 1))
        y.append(1 if int(r["DX_GROUP"]) == 1 else 0)
        site.append(str(r.get("SITE_ID", "NA")))
        age.append(float(r.get("AGE_AT_SCAN", np.nan)))
        try: sex.append(int(r.get("SEX", 1)))
        except Exception: sex.append(1)
    return (np.array(X), np.array(y), np.array(site), np.array(age),
            np.array(sex), np.vstack(strength))


def loso_auc(X, y, site, model):
    if len(set(site)) < 2:
        return np.nan
    logo = LeaveOneGroupOut(); oof = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups=site):
        sc = StandardScaler().fit(X[tr])
        oof[te] = model().fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


def main():
    print("=" * 70)
    print("  ABIDE-II INDEPENDENT REPLICATION")
    print("=" * 70)
    X2, y2, s2, age2, sex2, S2 = load_abide2()
    print(f"[abide2] {len(y2)} subjects | ASD {y2.sum()} CTRL {(y2==0).sum()} | sites {len(set(s2))}")
    logit = lambda: LogisticRegression(C=0.1, max_iter=2000)
    svm = lambda: SVC(C=1.0, gamma="scale", probability=True, random_state=42)

    # 1. within-ABIDE-II LOSO
    print("\n[1] Within-ABIDE-II leave-one-site-out:")
    a2_logit = loso_auc(X2, y2, s2, logit); a2_svm = loso_auc(X2, y2, s2, svm)
    print(f"    Logistic LOSO AUC = {a2_logit:.3f}   |   SVM LOSO AUC = {a2_svm:.3f}")
    print(f"    (ABIDE-I reference: Logistic 0.718, SVM 0.720)")

    # 2. cross-cohort transfer (train ABIDE-I -> test ABIDE-II)
    print("\n[2] Cross-cohort transfer (train ABIDE-I -> test ABIDE-II):")
    X1, y1, s1, fd1 = ce.build_features()
    sc = StandardScaler().fit(X1)
    p_raw = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X1), y1)\
        .predict_proba(sc.transform(X2))[:, 1]
    auc_raw = roc_auc_score(y2, p_raw)
    # ComBat harmonize cohorts (batch=cohort; preserve nothing biological beyond intercept)
    Xc = np.vstack([X1, X2]); batch = np.array([0]*len(X1) + [1]*len(X2))
    cb = ComBat().fit(Xc, batch)
    Xh = cb.transform(Xc, batch); X1h, X2h = Xh[:len(X1)], Xh[len(X1):]
    sc2 = StandardScaler().fit(X1h)
    p_h = LogisticRegression(C=0.1, max_iter=2000).fit(sc2.transform(X1h), y1)\
        .predict_proba(sc2.transform(X2h))[:, 1]
    auc_h = roc_auc_score(y2, p_h)
    print(f"    transfer AUC raw = {auc_raw:.3f}  |  ComBat-harmonized = {auc_h:.3f}")

    # 3. imaging-transcriptomics null replication
    print("\n[3] ABIDE-II disruption map -> SFARI spatial concordance (spin):")
    site_clean = np.array([re.sub(r"[^A-Za-z0-9]", "_", x) for x in s2])
    tab = pd.DataFrame(dict(ASD=y2, age=age2, sex=sex2, site=site_clean,
                            mean_fd=np.zeros(len(y2))))
    tab = tab.dropna(subset=["age"])
    S2v = S2[tab.index.values]; tab = tab.reset_index(drop=True)
    dmap2 = it.disruption_map(tab, S2v)
    sfari_map, ct_maps, cells, _, _ = it.load_expression_maps()
    coords, hemi = it.schaefer_centroids(); perms = it.spin_permutations(coords, hemi, 1000)
    r, p, _ = it.spin_test(dmap2, sfari_map, perms)
    print(f"    ABIDE-II concordance r={r:+.3f}  p_spin={p:.3f}  "
          f"{'(null replicates)' if p>0.05 else '(!)'}")
    print(f"    (ABIDE-I reference: r=-0.002, p_spin=0.99)")

    out = dict(n=int(len(y2)), n_asd=int(y2.sum()), n_sites=len(set(s2)),
               within_loso=dict(logistic=float(a2_logit), svm=float(a2_svm)),
               cross_cohort=dict(raw=float(auc_raw), combat=float(auc_h)),
               concordance=dict(r=float(r), p_spin=float(p)))
    json.dump(out, open(os.path.join(RESDIR, "abide2_replication.json"), "w"), indent=2)
    print("\n[saved] abide2_replication.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
