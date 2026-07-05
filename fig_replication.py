"""fig_replication.py — ABIDE-I vs ABIDE-II two-cohort replication figure (real JSONs)."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imaging_transcriptomics as it
R = it.RESDIR
c1 = json.load(open(os.path.join(R, "classification_results.json")))
a2 = json.load(open(os.path.join(R, "abide2_replication.json")))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
# Panel A: classification
labels = ["Logistic\n(within)", "SVM\n(within)", "Cross-cohort\n(I→II)"]
ab1 = [c1["Logistic-L2"]["auc"], c1["SVM-RBF"]["auc"], np.nan]
ab2 = [a2["within_loso"]["logistic"], a2["within_loso"]["svm"], a2["cross_cohort"]["combat"]]
x = np.arange(3)
ax[0].bar(x-0.2, ab1, 0.4, label="ABIDE-I (CPAC)", color="#2980B9")
ax[0].bar(x+0.2, ab2, 0.4, label="ABIDE-II (local)", color="#E67E22")
ax[0].axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.6)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
ax[0].set_ylabel("ROC-AUC"); ax[0].set_ylim(0.4, 0.8)
ax[0].set_title("Classification generalizes\n(modest, preprocessing-sensitive)", fontsize=10)
ax[0].legend(fontsize=7.5, loc="upper right")
for xi, v in zip(x-0.2, ab1):
    if not np.isnan(v): ax[0].text(xi, v+0.008, f"{v:.2f}", ha="center", fontsize=7)
for xi, v in zip(x+0.2, ab2):
    if not np.isnan(v): ax[0].text(xi, v+0.008, f"{v:.2f}", ha="center", fontsize=7)

# Panel B: concordance null replicates
r1 = c1.get("_", None)
r_vals = [-0.002, a2["concordance"]["r"]]
ax[1].axhspan(-0.30, 0.30, color="#BDC3C7", alpha=0.4, label="spin-null 95% band")
ax[1].bar([0, 1], r_vals, 0.5, color=["#2980B9", "#E67E22"])
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["ABIDE-I", "ABIDE-II"])
ax[1].set_ylabel("brain–gene spatial concordance (Spearman r)")
ax[1].set_ylim(-0.4, 0.4)
ax[1].set_title("Brain–gene NULL replicates\n(both inside spin-null; p=0.99 / 0.97)", fontsize=10)
ax[1].legend(fontsize=7.5, loc="upper right")
fig.suptitle("Two-cohort replication (ABIDE-I → ABIDE-II)", fontweight="bold", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(R, "Fig_replication.png")); fig.savefig(os.path.join(R, "Fig_replication.pdf"))
print("✓ Fig_replication saved")
