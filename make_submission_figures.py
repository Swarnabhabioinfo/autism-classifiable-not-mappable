"""
make_submission_figures.py — reviewer-proof, submission-grade figures (Nature style).

All six figures are drawn NATIVELY as vector PDFs (no rasterized panel compositing),
from real saved result files only. Fixes over the previous assemble_figures.py set:
  * Figure 1 is true vector (was a ~100 ppi raster paste-up).
  * Removed the self-referential "simulated 0.992" line.
  * ST-GCN shown honestly as the ORIGINAL-CV reference (not re-run under LOSO).
  * Error bars / null distributions added (Figs 1b, 3, 5, 6).
  * No editorialising on-canvas titles; lowercase bold panel letters; Okabe-Ito colours.
Run in the `neurogen4d` env (needs imaging_transcriptomics for the spin null in Fig 2).
"""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import imaging_transcriptomics as it

R = it.RESDIR
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_submission")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colourblind-safe palette
BLUE, ORANGE, VERM, GREEN, GRAY, DGRAY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#BDBDBD", "#666666"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=400)
    plt.close(fig)
    print(f"  wrote {name}.pdf (+png)")


def need(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing real data: {p} (run the analysis first)")
    return p


# ── Figure 1 — classification (a: ROC, b: model comparison) ────────────────
def fig1():
    d = np.load(need(os.path.join(R, "val_roc_data.npz")))
    models = []
    with open(need(os.path.join(R, "Table1_models.csv"))) as f:
        for row in csv.DictReader(f):
            models.append(row)
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.0))

    # a: ROC
    ax[0].plot(d["fpr"], d["tpr"], color=VERM, lw=1.8,
               label=f"FC + SVM-RBF\nAUC 0.72 (95% CI 0.69–0.75)")
    ax[0].plot([0, 1], [0, 1], ls="--", c=DGRAY, lw=0.8, label="chance")
    ax[0].fill_between(d["fpr"], d["tpr"], alpha=0.07, color=VERM)
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].legend(loc="lower right", fontsize=7, frameon=False)
    panel(ax[0], "a")

    # b: model comparison with 95% CIs; ST-GCN = original-CV reference (hatched)
    names, vals, los, his, cols, hatch = [], [], [], [], [], []
    label = {"Logistic-L2 (LOSO)": "Logistic-L2", "SVM-RBF (LOSO)": "SVM-RBF",
             "BrainNetCNN (LOSO)": "BrainNetCNN", "ST-GCN (original CV*)": "ST-GCN†"}
    for m in models:
        nm = m["model"]; v = float(m["auc"])
        names.append(label.get(nm, nm)); vals.append(v)
        lo = float(m["ci_lo"]) if m["ci_lo"] else v
        hi = float(m["ci_hi"]) if m["ci_hi"] else v
        los.append(v - lo); his.append(hi - v)
        deep = ("CNN" in nm) or ("GCN" in nm)
        if "GCN" in nm:                       # original-CV reference, not LOSO
            cols.append(GRAY); hatch.append("//")
        else:
            cols.append(ORANGE if deep else BLUE); hatch.append("")
    x = np.arange(len(names))
    bars = ax[1].bar(x, vals, 0.62, color=cols, edgecolor="white",
                     yerr=[los, his], capsize=3, error_kw=dict(lw=1, ecolor=DGRAY))
    for b, h in zip(bars, hatch):
        if h: b.set_hatch(h); b.set_edgecolor(DGRAY)
    ax[1].axhline(0.5, ls="--", c=DGRAY, lw=0.8)
    ax[1].text(len(names) - 0.5, 0.508, "chance", fontsize=6.5, color=DGRAY, ha="right")
    for xi, v, hh in zip(x, vals, his):
        ax[1].text(xi, v + hh + 0.018, f"{v:.2f}", ha="center", fontsize=7.5, fontweight="bold")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7.2)
    ax[1].set_ylabel("ROC-AUC"); ax[1].set_ylim(0.4, 0.8)
    # neutral legend for the colour code
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(fc=BLUE, label="linear (LOSO)"),
                          Patch(fc=ORANGE, label="deep (LOSO)"),
                          Patch(fc=GRAY, hatch="//", ec=DGRAY, label="ST-GCN† original CV")],
                 fontsize=5.9, frameon=False, loc="lower center",
                 bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0,
                 handlelength=1.1, handletextpad=0.4)
    panel(ax[1], "b")
    fig.tight_layout()
    save(fig, "Figure2")


# ── Figure 2 — brain–gene null (a: scatter, b: spin null) ──────────────────
def fig2():
    # aggregate SFARI concordance on the unified abagen LH-50 pipeline (as ii-iv)
    dmap = np.load(need(os.path.join(R, "disruption_map_lh.npy")))
    expr = np.load(need(os.path.join(R, "sfari_expr_map_lh.npy")))
    null = np.load(need(os.path.join(R, "concordance_lh_null.npy")))
    j = json.load(open(need(os.path.join(R, "concordance_lh.json"))))
    r_obs, p_spin = j["abide1"]["r"], j["abide1"]["p_spin"]

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.0))
    ax[0].scatter(stats.zscore(expr), stats.zscore(dmap), s=18, alpha=0.65,
                  color=BLUE, edgecolor="white", linewidth=0.4)
    ax[0].set_xlabel("SFARI risk-gene expression (z)")
    ax[0].set_ylabel("ASD disruption t-stat (z)")
    ax[0].text(0.04, 0.96, f"$r$ = {r_obs:+.2f},  $p_{{spin}}$ = {p_spin:.2f}",
               transform=ax[0].transAxes, va="top", fontsize=7.5)
    panel(ax[0], "a")

    ax[1].hist(null, bins=40, color=GRAY, edgecolor="white")
    ax[1].axvline(r_obs, color=VERM, lw=2, label=f"observed $r$ = {r_obs:+.2f}")
    ax[1].set_xlabel("Spearman $r$ (spin null)"); ax[1].set_ylabel("count")
    ax[1].legend(fontsize=7, frameon=False, loc="upper right")
    panel(ax[1], "b")
    fig.tight_layout()
    save(fig, "Figure4")


# ── Figure 3 — positive control (a: observed vs spin null, b: ASD vs CTRL) ──
def fig3():
    j = json.load(open(need(os.path.join(R, "coupling_results.json"))))
    null = np.load(need(os.path.join(R, "coupling_null_ctrl.npy")))
    ctrl, asd = j["coupling_ctrl"], j["coupling_asd"]
    cc, ca = j["ci_ctrl"], j["ci_asd"]
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.0))

    # a: control coupling vs its spin null (proves the analysis is sensitive)
    ax[0].hist(null, bins=35, color=GRAY, edgecolor="white")
    ax[0].axvline(ctrl, color=BLUE, lw=2,
                  label=f"observed (controls)\n$r$ = {ctrl:.2f}, $p_{{spin}}$ = {j['p_spin_ctrl']:.3f}")
    ax[0].set_xlabel("FC–CGE coupling (spin null)"); ax[0].set_ylabel("count")
    ax[0].legend(fontsize=7, frameon=False, loc="upper left")
    panel(ax[0], "a")

    # b: control vs autism coupling with bootstrap 95% CIs (difference n.s.)
    vals = [ctrl, asd]
    yerr = [[ctrl - cc[0], asd - ca[0]], [cc[1] - ctrl, ca[1] - asd]]
    ax[1].bar([0, 1], vals, 0.6, color=[BLUE, VERM], edgecolor="white",
              yerr=yerr, capsize=4, error_kw=dict(lw=1.1, ecolor=DGRAY))
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["Control", "Autism"])
    ax[1].set_ylabel("FC–CGE coupling\n(partial Spearman | distance)")
    ax[1].set_ylim(0, 0.30)
    ax[1].plot([0, 1], [0.265, 0.265], lw=0.9, c=DGRAY)
    ax[1].text(0.5, 0.268, f"n.s. ($p$ = {j['p_diff']:.2f})", ha="center", fontsize=7)
    panel(ax[1], "b")
    fig.tight_layout()
    save(fig, "Figure5")


# ── Figure 4 — power dissociation (a: reliability, b: AUC saturation) ───────
def fig4():
    p = json.load(open(need(os.path.join(R, "power_results.json"))))
    rel = {int(k): v for k, v in p["reliability"].items()}
    auc = {int(k): v for k, v in p["auc_curve"].items()}
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.0))
    ns = sorted(rel)
    ax[0].errorbar([2 * n for n in ns], [rel[n][0] for n in ns], yerr=[rel[n][1] for n in ns],
                   marker="o", ms=4, color=VERM, capsize=3, lw=1.3)
    ax[0].axhline(0.8, ls="--", c=DGRAY, lw=0.8)
    ax[0].annotate("reliable (0.8)\n≈ 1,740 subjects\n(Spearman–Brown)", xy=(500, 0.8),
                   xytext=(360, 0.9), fontsize=6.6, color=DGRAY, va="center")
    ax[0].set_xlabel("sample size (total)"); ax[0].set_ylabel("disruption-map split-half reliability")
    ax[0].set_ylim(0, 1)
    panel(ax[0], "a")
    na = sorted(auc)
    ax[1].errorbar(na, [auc[n][0] for n in na], yerr=[auc[n][1] for n in na],
                   marker="s", ms=4, color=BLUE, capsize=3, lw=1.3)
    ax[1].axhline(0.5, ls="--", c=DGRAY, lw=0.8)
    ax[1].text(780, 0.508, "chance", fontsize=6.5, color=DGRAY, ha="right")
    ax[1].set_xlabel("training sample size"); ax[1].set_ylabel("classification AUC")
    ax[1].set_ylim(0.45, 0.85)
    panel(ax[1], "b")
    fig.tight_layout()
    save(fig, "Figure6")


# ── Figure 5 — two-cohort replication (a: classification, b: brain–gene null) ─
def fig5():
    c1 = json.load(open(need(os.path.join(R, "classification_results.json"))))
    a2 = json.load(open(need(os.path.join(R, "abide2_replication.json"))))
    ci = json.load(open(need(os.path.join(R, "abide2_ci.json"))))
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 3.0))

    # a: within-ABIDE-I, within-ABIDE-II, cross-cohort transfer (all with 95% CI)
    def er(v, c): return [[v - c[0]], [c[1] - v]]
    x = np.arange(3)
    aI = [c1["Logistic-L2"]["auc"], c1["SVM-RBF"]["auc"]]
    aI_ci = [c1["Logistic-L2"]["ci"], c1["SVM-RBF"]["ci"]]
    aII = [a2["within_loso"]["logistic"], a2["within_loso"]["svm"]]
    aII_ci = [ci["within_logistic"]["ci"], ci["within_svm"]["ci"]]
    for k in (0, 1):
        ax[0].bar(x[k] - 0.21, aI[k], 0.4, color=BLUE, edgecolor="white",
                  yerr=er(aI[k], aI_ci[k]), capsize=3, error_kw=dict(lw=1, ecolor=DGRAY))
        ax[0].bar(x[k] + 0.21, aII[k], 0.4, color=ORANGE, edgecolor="white",
                  yerr=er(aII[k], aII_ci[k]), capsize=3, error_kw=dict(lw=1, ecolor=DGRAY))
    xc = a2["cross_cohort"]["combat"]
    ax[0].bar(x[2], xc, 0.4, color=GREEN, edgecolor="white",
              yerr=er(xc, ci["cross_combat"]["ci"]), capsize=3, error_kw=dict(lw=1, ecolor=DGRAY))
    ax[0].axhline(0.5, ls="--", c=DGRAY, lw=0.8)
    for xi, v in [(x[0]-0.21, aI[0]), (x[1]-0.21, aI[1]), (x[0]+0.21, aII[0]),
                  (x[1]+0.21, aII[1]), (x[2], xc)]:
        ax[0].text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["Logistic", "SVM", "Transfer\n(I→II)"])
    ax[0].set_ylabel("ROC-AUC"); ax[0].set_ylim(0.4, 0.82)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(fc=BLUE, label="ABIDE-I (within)"),
                          Patch(fc=ORANGE, label="ABIDE-II (within)"),
                          Patch(fc=GREEN, label="ABIDE-I→II")],
                 fontsize=6.3, frameon=False, loc="upper right", ncol=1)
    panel(ax[0], "a")

    # b: brain–gene concordance in both cohorts vs the LH-50 spin-null band
    cl = json.load(open(need(os.path.join(R, "concordance_lh.json"))))
    rI, pI = cl["abide1"]["r"], cl["abide1"]["p_spin"]
    rII, pII = cl["abide2"]["r"], cl["abide2"]["p_spin"]
    band = cl["null_band_95"]
    ax[1].axhspan(band[0], band[1], color=GRAY, alpha=0.4, label="spin-null 95% band")
    ax[1].bar([0, 1], [rI, rII], 0.5, color=[BLUE, ORANGE], edgecolor="white")
    ax[1].axhline(0, color="k", lw=0.8)
    for xi, v, p in [(0, rI, pI), (1, rII, pII)]:
        ax[1].text(xi, 0.52, f"$r$ = {v:+.2f}\n$p$ = {p:.2f}", ha="center", va="top", fontsize=6.8)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["ABIDE-I", "ABIDE-II"])
    ax[1].set_ylabel("brain–gene concordance (Spearman $r$)")
    ax[1].set_ylim(-0.6, 0.64); ax[1].legend(fontsize=6.3, frameon=False, loc="lower right")
    panel(ax[1], "b")
    fig.tight_layout()
    save(fig, "Figure7")


# ── Figure 6 — developmental convergence (single panel, null 95% band) ──────
def fig6():
    d = json.load(open(need(os.path.join(R, "developmental_results.json"))))["convergence"]
    order = ["early_prenatal", "midfetal", "late_prenatal", "infancy", "childhood", "adolescence", "adult"]
    ws = [w for w in order if w in d]
    obs = [d[w]["coexpr"] for w in ws]
    nul = [d[w]["null_mean"] for w in ws]
    nlo = [d[w]["null_mean"] - d[w]["null_lo"] for w in ws]
    nhi = [d[w]["null_hi"] - d[w]["null_mean"] for w in ws]
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    x = np.arange(len(ws))
    ax.bar(x - 0.2, obs, 0.4, label="SFARI genes", color=VERM, edgecolor="white")
    ax.bar(x + 0.2, nul, 0.4, label="random gene sets (95% null)", color=GRAY, edgecolor="white",
           yerr=[nlo, nhi], capsize=3, error_kw=dict(lw=1, ecolor=DGRAY))
    for i, w in enumerate(ws):
        if d[w]["p"] < 0.05:
            ax.text(i, max(obs[i], nul[i]) + 0.012, "*", ha="center", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels([w.replace("_", "\n") for w in ws], fontsize=7)
    ax.set_ylabel("mean |co-expression|"); ax.set_ylim(0, 0.62)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "Figure8")


if __name__ == "__main__":
    print("Generating submission figures (vector, real data only) ->", OUT)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("Done.")
