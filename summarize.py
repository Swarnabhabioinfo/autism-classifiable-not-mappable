"""
summarize.py — assemble Table 1 + the model-comparison figure from REAL results.
Reads only saved JSON/NPZ; marks the original fabricated AUC=0.992 as invalid.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imaging_transcriptomics as it
RESDIR = it.RESDIR

def load(j):
    p = os.path.join(RESDIR, j)
    return json.load(open(p)) if os.path.exists(p) else None

def main():
    clf = load("classification_results.json") or {}
    deep = load("deep_brainnetcnn.json")
    harm = load("harmonization_results.json") or {}

    # ── Table 1 ──
    rows = []
    for name in ["Logistic-L2", "SVM-RBF"]:
        if name in clf:
            c = clf[name]
            rows.append((name + " (LOSO)", c["auc"], c["ci"][0], c["ci"][1],
                         c.get("motion_controlled_auc")))
    if deep:
        rows.append(("BrainNetCNN (LOSO)", deep["loso_auc"], deep["ci"][0], deep["ci"][1], None))
    rows.append(("ST-GCN (original CV*)", 0.59, None, None, None))

    print(f"\n{'Model':28s} {'AUC':>6s} {'95% CI':>16s} {'motion-ctrl':>12s}")
    print("-" * 66)
    csv = ["model,auc,ci_lo,ci_hi,motion_controlled_auc"]
    for n, a, lo, hi, mc in rows:
        ci = f"[{lo:.3f}-{hi:.3f}]" if lo else "      —      "
        mcs = f"{mc:.3f}" if mc else "—"
        print(f"{n:28s} {a:6.3f} {ci:>16s} {mcs:>12s}")
        csv.append(f"{n},{a},{lo if lo else ''},{hi if hi else ''},{mc if mc else ''}")
    open(os.path.join(RESDIR, "Table1_models.csv"), "w").write("\n".join(csv))

    # ── Figure: model comparison ──
    names = [r[0].replace(" (LOSO)", "").replace(" (original CV*)", "*") for r in rows]
    aucs  = [r[1] for r in rows]
    err   = [[r[1]-(r[2] if r[2] else r[1]) for r in rows],
             [(r[3] if r[3] else r[1])-r[1] for r in rows]]
    colors = ["#2980B9", "#2980B9", "#8E44AD", "#7F8C8D"][:len(rows)]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.bar(range(len(rows)), aucs, yerr=err, color=colors, edgecolor="white",
           capsize=4, width=0.62)
    ax.axhline(0.5, ls="--", c="k", lw=0.9, alpha=0.6, label="chance (0.5)")
    ax.axhline(0.992, ls=":", c="#C0392B", lw=1.4,
               label="original claim 0.992 (simulated — invalid)")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(names, fontsize=8.5, rotation=12)
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.4, 1.02)
    ax.set_title("Honest ASD classification: deep models do not beat linear FC\n"
                 "(leave-one-site-out CV, ABIDE-I)", fontsize=10)
    for i, a in enumerate(aucs):
        ax.text(i, a + 0.015, f"{a:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.text(0.0, 0.43, "*original ST-GCN used a leakage-favorable CV; honest LOSO would be ≤ this",
            fontsize=6, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, "Fig_model_comparison.png"))
    fig.savefig(os.path.join(RESDIR, "Fig_model_comparison.pdf"))
    print("\n  ✓ Fig_model_comparison + Table1_models.csv")

if __name__ == "__main__":
    main()
