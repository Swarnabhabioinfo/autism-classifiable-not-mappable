"""boot_abide2_ci.py — add bootstrap 95% CIs for the ABIDE-II replication AUCs
(within-LOSO logistic/SVM and cross-cohort ComBat transfer) for Figure 5.

The point AUCs are deterministic and reproduce abide2_replication.json; this script
ONLY adds uncertainty bands and does NOT touch the spin-test concordance result.
Outputs results/abide2_ci.json.
"""
import os, json, numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import imaging_transcriptomics as it
from combat import ComBat
import classify_eval as ce
import replicate_abide2 as rep

RESDIR = it.RESDIR


def loso_oof(X, y, site, model):
    logo = LeaveOneGroupOut(); oof = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups=site):
        sc = StandardScaler().fit(X[tr])
        oof[te] = model().fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); a = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    X2, y2, s2, age2, sex2, S2 = rep.load_abide2()
    logit = lambda: LogisticRegression(C=0.1, max_iter=2000)
    svm = lambda: SVC(C=1.0, gamma="scale", probability=True, random_state=42)

    oof_l = loso_oof(X2, y2, s2, logit); oof_s = loso_oof(X2, y2, s2, svm)
    auc_l = roc_auc_score(y2, oof_l); auc_s = roc_auc_score(y2, oof_s)

    X1, y1, s1, fd1 = ce.build_features()
    Xc = np.vstack([X1, X2]); batch = np.array([0] * len(X1) + [1] * len(X2))
    cb = ComBat().fit(Xc, batch); Xh = cb.transform(Xc, batch)
    X1h, X2h = Xh[:len(X1)], Xh[len(X1):]
    sc2 = StandardScaler().fit(X1h)
    p_h = LogisticRegression(C=0.1, max_iter=2000).fit(sc2.transform(X1h), y1)\
        .predict_proba(sc2.transform(X2h))[:, 1]
    auc_h = roc_auc_score(y2, p_h)

    out = dict(
        within_logistic=dict(auc=float(auc_l), ci=boot_ci(y2, oof_l)),
        within_svm=dict(auc=float(auc_s), ci=boot_ci(y2, oof_s)),
        cross_combat=dict(auc=float(auc_h), ci=boot_ci(y2, p_h)),
    )
    json.dump(out, open(os.path.join(RESDIR, "abide2_ci.json"), "w"), indent=2)
    print("saved abide2_ci.json:", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
