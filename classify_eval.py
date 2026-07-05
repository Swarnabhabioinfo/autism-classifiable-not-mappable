"""
classify_eval.py — honest, leakage-free ASD classification benchmark.

Fixes the original pipeline's fatal evaluation flaws:
  * NO model selection on the test set (the original reported the val-AUC it
    early-stopped on). Here: leave-one-site-out (LOSO) — train on N-1 sites,
    test on a completely unseen site. This is the real cross-site number.
  * Signed Fisher-z FC (keeps anti-correlation, unlike the original abs()).
  * Motion QC (mean FD < 0.5) + a motion-CONTROLLED variant that regresses
    mean-FD out of every feature (fit on train) to quantify how much apparent
    "ASD signal" is head motion.
  * Saves REAL pooled ROC arrays -> honest figures (no np.random.beta()).

Baselines: L2 logistic regression and RBF-SVM on the upper-triangle FC vector
(the field-standard representation; cf. Heinsfeld 2018, Abraham 2017).
"""
import os, json, numpy as np
from scipy import stats
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix

import imaging_transcriptomics as it
RESDIR = it.RESDIR
N_ROI  = it.N_ROI
SEED   = 42


def build_features():
    import re
    pheno = it.load_phenotype()
    iu, ju = np.triu_indices(N_ROI, k=1)
    X, y, site, fd = [], [], [], []
    for f in sorted(os.listdir(it.DATA_DIR)):
        if not f.endswith(".npy"):
            continue
        nums = re.findall(r"\d+", f)
        if not nums:
            continue
        sid = str(int("".join(nums)))
        if sid not in pheno.index:
            continue
        row = pheno.loc[sid]
        if hasattr(row, "iloc") and getattr(row, "ndim", 1) == 2:
            row = row.iloc[0]
        m = row["func_mean_fd"]
        if np.isnan(m) or m > it.FD_THRESH:
            continue
        ts = np.load(os.path.join(it.DATA_DIR, f))
        if ts.ndim != 2 or ts.shape[1] != N_ROI:
            continue
        ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-8)
        r = np.nan_to_num(np.corrcoef(ts, rowvar=False), nan=0.0)
        np.clip(r, -0.999, 0.999, out=r)
        z = np.arctanh(r)
        X.append(z[iu, ju]); y.append(int(row["ASD"]))
        site.append(str(row["SITE_ID"])); fd.append(float(m))
    return (np.array(X), np.array(y), np.array(site), np.array(fd))


def regress_out(train_X, test_X, train_c, test_c):
    """Remove linear effect of confound c (e.g., mean FD) from each feature."""
    lr = LinearRegression().fit(train_c.reshape(-1, 1), train_X)
    return train_X - lr.predict(train_c.reshape(-1, 1)), \
           test_X - lr.predict(test_c.reshape(-1, 1))


def eval_loso(X, y, site, fd, model_fn, control_motion=False):
    logo = LeaveOneGroupOut()
    oof_p = np.full(len(y), np.nan); per_site = {}
    for tr, te in logo.split(X, y, groups=site):
        Xtr, Xte = X[tr], X[te]
        if control_motion:
            Xtr, Xte = regress_out(Xtr, Xte, fd[tr], fd[te])
        sc = StandardScaler().fit(Xtr)
        clf = model_fn().fit(sc.transform(Xtr), y[tr])
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        oof_p[te] = p
        s = site[te][0]
        if len(np.unique(y[te])) == 2:
            per_site[s] = roc_auc_score(y[te], p)
    auc = roc_auc_score(y, oof_p)
    preds = (oof_p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    return dict(auc=float(auc), f1=float(f1_score(y, preds)),
                sens=float(tp/(tp+fn)), spec=float(tn/(tn+fp)),
                per_site=per_site), oof_p


def boot_ci(y, p, n=2000, seed=123):
    rng = np.random.default_rng(seed); a = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            a.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    print("=" * 70)
    print("  Honest ASD classification — leave-one-site-out, motion-controlled")
    print("=" * 70)
    X, y, site, fd = build_features()
    print(f"[data] {len(y)} subjects | ASD {y.sum()} CTRL {(y==0).sum()} | "
          f"{len(set(site))} sites | features {X.shape[1]} (signed Fisher-z FC)")

    models = {
        "Logistic-L2": lambda: LogisticRegression(C=0.1, max_iter=2000),
        "SVM-RBF":     lambda: SVC(C=1.0, gamma="scale", probability=True,
                                   random_state=SEED),
    }
    summary = {}
    roc_arrays = {}
    for name, fn in models.items():
        raw, p_raw = eval_loso(X, y, site, fd, fn, control_motion=False)
        mot, p_mot = eval_loso(X, y, site, fd, fn, control_motion=True)
        lo, hi = boot_ci(y, p_raw)
        sites_auc = np.array(list(raw["per_site"].values()))
        print(f"\n  {name}")
        print(f"    LOSO AUC (raw)            : {raw['auc']:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
        print(f"    LOSO AUC (motion-removed) : {mot['auc']:.3f}   "
              f"(Δ from motion control = {raw['auc']-mot['auc']:+.3f})")
        print(f"    per-site AUC range        : {sites_auc.min():.2f}-{sites_auc.max():.2f} "
              f"(mean {sites_auc.mean():.3f})")
        print(f"    sens {raw['sens']:.2f} | spec {raw['spec']:.2f} | F1 {raw['f1']:.2f}")
        summary[name] = dict(raw=raw, motion_controlled=mot, ci=[lo, hi])
        if name == "SVM-RBF":
            fpr, tpr, _ = roc_curve(y, p_raw)
            roc_arrays = dict(fpr=fpr, tpr=tpr, auc=raw["auc"], ci_lo=lo, ci_hi=hi,
                              y=y, p=p_raw)

    # majority-class reference
    maj = max(y.mean(), 1 - y.mean())
    print(f"\n  Reference: majority-class accuracy = {maj:.3f}; chance AUC = 0.50")

    np.savez(os.path.join(RESDIR, "val_roc_data.npz"),
             fpr=roc_arrays["fpr"], tpr=roc_arrays["tpr"],
             auc=roc_arrays["auc"], ci_lo=roc_arrays["ci_lo"], ci_hi=roc_arrays["ci_hi"])
    with open(os.path.join(RESDIR, "classification_results.json"), "w") as fh:
        json.dump({k: {kk: (vv if kk != "per_site" else vv)
                       for kk, vv in v["raw"].items()} | {"ci": v["ci"],
                       "motion_controlled_auc": v["motion_controlled"]["auc"]}
                   for k, v in summary.items()} | {"majority_acc": float(maj)},
                  fh, indent=2)
    print("\n[saved] real ROC arrays -> results/val_roc_data.npz "
          "(honest figures read this; no simulation)")
    print("=" * 70)


if __name__ == "__main__":
    main()
