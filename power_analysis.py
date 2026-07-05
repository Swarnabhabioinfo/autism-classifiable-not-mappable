"""
power_analysis.py — why n~1000 cannot find autism brain-association effects
(Marek 2022, Nature, adapted). The scientific core of the honest paper.

Two curves that together make the dissociation argument:
  (A) Split-half RELIABILITY of the ASD-vs-CTRL disruption map vs sample size.
      Brain-wide *association* maps need very large N to be reproducible.
  (B) Classification AUC vs training size — a strong *group* effect that
      saturates early. n~1000 is plenty for (B), nowhere near enough for (A).

Confounds (age, sex, mean FD) residualized out first. Saves arrays + figure.
"""
import os, json, numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imaging_transcriptomics as it
RESDIR = it.RESDIR
rng = np.random.default_rng(42)


def residualize(S, tab):
    C = np.column_stack([np.ones(len(tab)), tab["age"].values,
                         tab["sex"].values, tab["mean_fd"].values])
    beta = np.linalg.lstsq(C, S, rcond=None)[0]
    return S - C @ beta


def dmap(S, asd):
    a, c = S[asd], S[~asd]
    sd = np.sqrt((a.var(0) + c.var(0)) / 2) + 1e-9
    return (a.mean(0) - c.mean(0)) / sd            # per-region Cohen's d


def reliability_curve(S, asd, Ns, B=100):
    ia, ic = np.where(asd)[0], np.where(~asd)[0]
    out = {}
    for Nq in Ns:
        rs = []
        for _ in range(B):
            sa = rng.choice(ia, 2 * Nq, replace=False)
            sc = rng.choice(ic, 2 * Nq, replace=False)
            a1, a2 = sa[:Nq], sa[Nq:]; c1, c2 = sc[:Nq], sc[Nq:]
            m1 = dmap(S, None) if False else (
                (S[a1].mean(0) - S[c1].mean(0)) /
                (np.sqrt((S[a1].var(0) + S[c1].var(0)) / 2) + 1e-9))
            m2 = ((S[a2].mean(0) - S[c2].mean(0)) /
                  (np.sqrt((S[a2].var(0) + S[c2].var(0)) / 2) + 1e-9))
            rs.append(stats.spearmanr(m1, m2).correlation)
        out[Nq] = (float(np.mean(rs)), float(np.std(rs)))
    return out


def auc_curve(X, y, Ns, B=20):
    n = len(y); out = {}
    for Nq in Ns:
        if Nq >= n - 100:
            continue
        aucs = []
        for _ in range(B):
            idx = rng.permutation(n)
            tr, te = idx[:Nq], idx[Nq:Nq + 200]
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.1, max_iter=1500).fit(sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
        out[Nq] = (float(np.mean(aucs)), float(np.std(aucs)))
    return out


def main():
    print("=" * 68)
    print("  Power analysis: association reliability vs classification (Marek'22)")
    print("=" * 68)
    tab, S = it.build_subject_table(it.load_phenotype())
    asd = tab["ASD"].values == 1
    Sr = residualize(S, tab)
    print(f"[data] {len(tab)} subj | ASD {asd.sum()} CTRL {(~asd).sum()}")

    print("\n[A] Split-half reliability of disruption map vs N/group...")
    Ns_rel = [40, 60, 80, 120, 160, 200, 250]
    rel = reliability_curve(Sr, asd, Ns_rel)
    for nq, (m, s) in rel.items():
        print(f"    N/group={nq:4d}  reliability r={m:+.3f} ± {s:.3f}")
    full = rel[max(rel)][0]

    # FC features for classification curve
    print("\n[B] Classification AUC vs training N...")
    import re
    iu, ju = np.triu_indices(it.N_ROI, 1)
    X, y = [], []
    ph = it.load_phenotype()
    for f in sorted(os.listdir(it.DATA_DIR)):
        if not f.endswith(".npy"):
            continue
        nums = re.findall(r"\d+", f)
        if not nums:
            continue
        sid = str(int("".join(nums)))
        if sid not in ph.index:
            continue
        r = ph.loc[sid]
        if getattr(r, "ndim", 1) == 2:
            r = r.iloc[0]
        if np.isnan(r["func_mean_fd"]) or r["func_mean_fd"] > it.FD_THRESH:
            continue
        ts = np.load(os.path.join(it.DATA_DIR, f))
        if ts.ndim != 2 or ts.shape[1] != it.N_ROI:
            continue
        ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-8)
        cc = np.nan_to_num(np.corrcoef(ts, rowvar=False), nan=0.0)
        np.clip(cc, -0.999, 0.999, out=cc)
        X.append(np.arctanh(cc)[iu, ju]); y.append(int(r["ASD"]))
    X, y = np.array(X), np.array(y)
    Ns_auc = [100, 200, 400, 600, 800]
    auc = auc_curve(X, y, Ns_auc)
    for nq, (m, s) in auc.items():
        print(f"    train N={nq:4d}  AUC={m:.3f} ± {s:.3f}")

    # crude extrapolation: reliability ~ N/(N+N0) -> solve for r=0.8
    Ns_arr = np.array(list(rel)); rvals = np.array([rel[n][0] for n in Ns_arr])
    rvals_c = np.clip(rvals, 1e-3, 0.999)
    N0 = np.median(Ns_arr * (1 - rvals_c) / rvals_c)
    N_for_08 = N0 * 0.8 / 0.2
    print(f"\n[extrapolation] Spearman-Brown-style: reliability r=0.8 needs "
          f"~{int(N_for_08)}/group (~{int(2*N_for_08)} total)")
    print(f"    (we have {asd.sum()} ASD; full-sample split-half r={full:.2f})")

    json.dump(dict(reliability={int(k): v for k, v in rel.items()},
                   auc_curve={int(k): v for k, v in auc.items()},
                   N_for_reliability_0p8=int(N_for_08)),
              open(os.path.join(RESDIR, "power_results.json"), "w"), indent=2)

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ns = list(rel); ms = [rel[n][0] for n in ns]; ss = [rel[n][1] for n in ns]
    ax[0].errorbar([2*n for n in ns], ms, yerr=ss, marker="o", color="#C0392B", capsize=3)
    ax[0].axhline(0.8, ls="--", c="k", lw=0.8, label="reliable (0.8)")
    ax[0].set_xlabel("sample size (total)"); ax[0].set_ylabel("disruption-map split-half reliability")
    ax[0].set_title("Brain-association map\nneeds large N (Marek'22)"); ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=7)
    na = list(auc); ma = [auc[n][0] for n in na]; sa = [auc[n][1] for n in na]
    ax[1].errorbar(na, ma, yerr=sa, marker="s", color="#2980B9", capsize=3)
    ax[1].axhline(0.5, ls="--", c="k", lw=0.8)
    ax[1].set_xlabel("training sample size"); ax[1].set_ylabel("classification AUC")
    ax[1].set_title("Group classification\nsaturates early"); ax[1].set_ylim(0.45, 0.85)
    fig.suptitle("Why n~1000 classifies ASD but cannot map its brain-association effects",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, "Fig_power.png")); fig.savefig(os.path.join(RESDIR, "Fig_power.pdf"))
    print("  ✓ Fig_power saved")
    print("=" * 68)


if __name__ == "__main__":
    main()
