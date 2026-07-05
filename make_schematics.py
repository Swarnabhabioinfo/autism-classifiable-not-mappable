"""
make_schematics.py — two schematic figures (vector):
  Figure1.pdf      study pipeline: acquisition -> preprocessing ->
                           connectivity -> analyses -> validation (icons)
  Figure3.pdf  BrainNetCNN deep-baseline architecture, every
                           mathematical layer + equation (honest: the deep model
                           that does NOT exceed linear FC).
Vector output, Okabe-Ito palette, matches the other submission figures.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_submission")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, VERM, GREEN, GRAY, DGRAY, PURP = \
    "#0072B2", "#E69F00", "#D55E00", "#009E73", "#E9E9E9", "#555555", "#7B5EA7"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "savefig.dpi": 400, "savefig.bbox": "tight", "pdf.fonttype": 42})


def box(ax, x, y, w, h, text, fc="white", ec=DGRAY, fs=8.5, bold=False, tc="black", lw=1.2, rad=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={rad}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2, c=DGRAY, lw=1.6, style="-|>", ms=10):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=c, zorder=1, shrinkA=0, shrinkB=0))


# ── icons ────────────────────────────────────────────────────────────────
def icon_brain(ax, cx, cy, s=1.0, color=VERM):
    ax.add_patch(Ellipse((cx, cy), 3.2 * s, 2.3 * s, fc="white", ec=color, lw=1.6, zorder=4))
    for dx in np.linspace(-1.1, 1.1, 4):
        xs = np.linspace(cx + dx * s - 0.35 * s, cx + dx * s + 0.35 * s, 20)
        ys = cy + 0.9 * s * np.sin((xs - cx) * 6) * np.exp(-((xs - cx) ** 2) / (2.5 * s ** 2))
        ax.plot(xs, ys, color=color, lw=0.8, zorder=5)
    ax.plot([cx, cx], [cy - 1.05 * s, cy + 1.05 * s], color=color, lw=0.8, zorder=5)


def icon_matrix(ax, cx, cy, s=1.0, n=6, color=BLUE):
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, n)); M = (M + M.T) / 2
    x0, y0 = cx - 1.5 * s, cy - 1.5 * s; c = 3.0 * s / n
    for i in range(n):
        for j in range(n):
            v = np.tanh(M[i, j]); col = plt.cm.RdBu_r((v + 1) / 2)
            ax.add_patch(plt.Rectangle((x0 + j * c, y0 + i * c), c, c, fc=col, ec="white", lw=0.3, zorder=4))
    ax.add_patch(plt.Rectangle((x0, y0), 3.0 * s, 3.0 * s, fc="none", ec=color, lw=1.3, zorder=5))


def icon_dna(ax, cx, cy, s=1.0, color=GREEN):
    t = np.linspace(0, 2 * np.pi, 60)
    x1 = cx + 0.6 * s * np.sin(t); x2 = cx + 0.6 * s * np.sin(t + np.pi)
    y = cy + np.linspace(-1.6 * s, 1.6 * s, 60)
    ax.plot(x1, y, color=color, lw=1.6, zorder=4); ax.plot(x2, y, color=DGRAY, lw=1.6, zorder=4)
    for k in range(0, 60, 6):
        ax.plot([x1[k], x2[k]], [y[k], y[k]], color=color, lw=0.7, zorder=3)


# ═══════════════════════ Figure 1 — workflow ═══════════════════════════════
def workflow():
    fig, ax = plt.subplots(figsize=(7.2, 9.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def band(y, label, col):
        ax.add_patch(FancyBboxPatch((1.5, y), 97, 0.5, boxstyle="round,pad=0.2", fc=col, ec="none", alpha=0.16, zorder=0))
        ax.text(3, y + 0.25, label, fontsize=9, fontweight="bold", color=col, va="center", zorder=3)

    # Stage 1 — data acquisition
    band(90, "1  Data acquisition", BLUE)
    box(ax, 8, 78, 40, 9, "", fc="#F4F9FD", ec=BLUE)
    icon_brain(ax, 14, 82.5, 1.15, VERM)
    ax.text(20, 84.7, "Resting-state fMRI", fontsize=8.5, fontweight="bold", va="center")
    ax.text(20, 82.0, "ABIDE-I  n=1,064 (20 sites)", fontsize=7.6, va="center")
    ax.text(20, 79.9, "ABIDE-II  n=500 (15 sites)", fontsize=7.6, va="center")
    box(ax, 52, 78, 40, 9, "", fc="#F1FBF6", ec=GREEN)
    icon_dna(ax, 57, 82.5, 1.0, GREEN)
    ax.text(62, 84.7, "Reference genomics", fontsize=8.5, fontweight="bold", va="center")
    ax.text(62, 82.3, "AHBA (6 donors) · SFARI genes", fontsize=7.6, va="center")
    ax.text(62, 80.1, "BrainSpan · single-cell atlas", fontsize=7.6, va="center")

    arrow(ax, 50, 77.5, 50, 73)
    # Stage 2 — preprocessing
    band(70.5, "2  Preprocessing & quality control", ORANGE)
    box(ax, 14, 58, 72, 10.5, "", fc="#FEF7EC", ec=ORANGE)
    icon_brain(ax, 21, 63.3, 1.15, ORANGE)
    ax.text(28, 66.0, "ABIDE-I: CPAC   |   ABIDE-II: FSL (mcflirt · bet · flirt)", fontsize=7.8, va="center")
    ax.text(28, 63.4, "head-motion QC (mean FD < 0.5 mm); nuisance regression", fontsize=7.8, va="center")
    ax.text(28, 60.8, "Schaefer-100 / Yeo-7 parcellation", fontsize=7.8, va="center")

    arrow(ax, 50, 57.5, 50, 53)
    # Stage 3 — connectivity
    band(50.5, "3  Functional connectivity", PURP)
    box(ax, 22, 40, 56, 9.5, "", fc="#F5F1FA", ec=PURP)
    icon_matrix(ax, 29, 44.7, 1.15, color=PURP)
    ax.text(36, 46.6, "signed Fisher-$z$ FC", fontsize=8.4, fontweight="bold", va="center")
    ax.text(36, 44.2, "4,950 edges per participant", fontsize=7.6, va="center")
    ax.text(36, 42.0, "nodal strength = signed mean edge weight", fontsize=7.6, va="center")

    # fan-out to analyses
    for xc in (16, 33, 50, 67, 84):
        arrow(ax, 50, 39.5, xc, 34.5, c=DGRAY, lw=1.2)
    band(35.5, "4  Analyses (leakage-free)", VERM)
    arms = [("Classification\nLOSO: linear vs deep", BLUE),
            ("Disruption map\n$\\rightarrow$ spin-null vs\nSFARI expression", VERM),
            ("Positive control\nFC$-$gene coupling", GREEN),
            ("Connectome\nbiotypes", ORANGE),
            ("Power analysis\nclassify vs map", PURP)]
    xs = [7.5, 26, 44.5, 63, 81.5]
    for (txt, col), x in zip(arms, xs):
        box(ax, x, 23, 15.5, 10, txt, fc="white", ec=col, fs=7.1)
        arrow(ax, x + 7.75, 22.5, 50, 18.5, c=DGRAY, lw=1.0)

    band(19.5, "5  Validation & replication", GREEN)
    box(ax, 12, 9.5, 76, 8.5, "", fc="#F1FBF6", ec=GREEN)
    ax.text(50, 15.3, "independent ABIDE-II replication  ·  cross-cohort transfer (ComBat)",
            fontsize=7.8, ha="center", va="center")
    ax.text(50, 12.4, "BrainSpan developmental convergence  ·  spatial-null & bootstrap inference",
            fontsize=7.8, ha="center", va="center")

    arrow(ax, 50, 9, 50, 5.5)
    box(ax, 5, 0.3, 90, 4.6, "Outcome:  autism is classifiable but not spatially mappable",
        fc=DGRAY, ec=DGRAY, tc="white", fs=8.7, bold=True)

    fig.savefig(os.path.join(OUT, "Figure1.pdf"))
    fig.savefig(os.path.join(OUT, "Figure1.png"), dpi=300)
    plt.close(fig); print("wrote Figure1")


# ═══════════════════ Figure 2 — BrainNetCNN architecture ═══════════════════
def architecture():
    fig, ax = plt.subplots(figsize=(7.4, 10.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    LX, LW = 6, 52          # layer column
    EX = 60                 # equation column
    layers = [
        ("Input — connectivity tensor", "$\\mathbf{A}\\in\\mathbb{R}^{100\\times100\\times1}$",
         "$A_{ij}=\\mathrm{arctanh}\\,\\rho(x_i,x_j)$, sign retained", BLUE),
        ("Edge-to-Edge  E2E$_1$  (1$\\rightarrow$16)", "$\\mathbf{H}^{(1)}\\in\\mathbb{R}^{16\\times100\\times100}$",
         "$H^{(1),m}_{ij}=\\varphi\\!\\left(\\sum_{n} r^{m}_{n}A_{in}+\\sum_{n}s^{m}_{n}A_{nj}\\right)$", VERM),
        ("Edge-to-Edge  E2E$_2$  (16$\\rightarrow$16)", "$\\mathbf{H}^{(2)}\\in\\mathbb{R}^{16\\times100\\times100}$",
         "$H^{(2),m}_{ij}=\\varphi\\!\\left(\\sum_{c,n} r^{m}_{cn}H^{(1),c}_{in}+s^{m}_{cn}H^{(1),c}_{nj}\\right)$", VERM),
        ("Edge-to-Node  E2N  (16$\\rightarrow$32)", "$\\mathbf{N}\\in\\mathbb{R}^{32\\times100}$",
         "$N^{m}_{i}=\\varphi\\!\\left(\\sum_{c}\\sum_{n} w^{m}_{cn}\\,H^{(2),c}_{in}\\right)$", ORANGE),
        ("Node-to-Graph  N2G  (32$\\rightarrow$64)", "$\\mathbf{g}\\in\\mathbb{R}^{64}$",
         "$g^{m}=\\varphi\\!\\left(\\sum_{c}\\sum_{i} v^{m}_{ci}\\,N^{c}_{i}\\right)$", GREEN),
        ("Dense  FC$_1$  (64$\\rightarrow$64) + dropout", "$\\mathbf{z}\\in\\mathbb{R}^{64}$",
         "$\\mathbf{z}=\\mathrm{Drop}_{0.5}\\left(\\varphi(\\mathbf{W}_1\\mathbf{g}+\\mathbf{b}_1)\\right)$", PURP),
        ("Dense  FC$_2$  (64$\\rightarrow$2)", "$\\mathbf{o}\\in\\mathbb{R}^{2}$",
         "$\\mathbf{o}=\\mathbf{W}_2\\mathbf{z}+\\mathbf{b}_2$", PURP),
        ("Soft-max  output", "$\\hat{p}\\in[0,1]^{2}$",
         "$\\hat{p}_k=e^{o_k}/\\sum_{k'}e^{o_{k'}}$", BLUE),
    ]
    n = len(layers); top, bot, h = 95, 12, 8.4
    ys = np.linspace(top - h, bot, n)
    for (name, shape, eq, col), y in zip(layers, ys):
        box(ax, LX, y, LW, h, "", fc=col, ec=DGRAY, lw=1.1)
        # tint: fill light then header bar
        ax.add_patch(FancyBboxPatch((LX, y + h - 2.4), LW, 2.4, boxstyle="round,pad=0.006,rounding_size=0.02",
                                    fc=col, ec="none", alpha=0.9, zorder=2.5))
        ax.add_patch(FancyBboxPatch((LX, y), LW, h - 2.4, boxstyle="square,pad=0",
                                    fc="white", ec="none", alpha=0.82, zorder=2.4))
        ax.text(LX + LW / 2, y + h - 1.2, name, ha="center", va="center", fontsize=8.0,
                fontweight="bold", color="white", zorder=4)
        ax.text(LX + LW / 2, y + (h - 2.4) / 2, shape, ha="center", va="center", fontsize=8.6, zorder=4)
        ax.text(EX + 1, y + h / 2, eq, ha="left", va="center", fontsize=9.2, zorder=4)
        if y != ys[-1]:
            arrow(ax, LX + LW / 2, y - 0.2, LX + LW / 2, y - (ys[0] - ys[1]) + h + 0.2, c=DGRAY, lw=1.6)

    # side / footer notes
    ax.text(EX + 1, 99, "layer operations", fontsize=8, style="italic", color=DGRAY, ha="left")
    box(ax, LX, 0.2, 92, 9, "", fc="#F7F7F7", ec=DGRAY, lw=1)
    ax.text(8, 7.2, "Nonlinearity:  $\\varphi(x)=\\max(0.1x,\\,x)$  (leaky-ReLU)", fontsize=8.2, va="center")
    ax.text(8, 4.7, "Objective:  $\\mathcal{L}=-\\sum_k w_k\\,y_k\\log\\hat{p}_k$  (class-weighted cross-entropy)",
            fontsize=8.2, va="center")
    ax.text(8, 2.2, "Optimiser:  Adam, lr $10^{-3}$, weight decay $10^{-3}$, batch 32, 40 epochs · leave-one-site-out",
            fontsize=8.2, va="center")
    ax.text(50, 10.6, "Deep-learning baseline (BrainNetCNN); LOSO AUC 0.68 $<$ 0.72 (linear FC)",
            fontsize=7.8, ha="center", color=VERM, fontweight="bold")

    fig.savefig(os.path.join(OUT, "Figure3.pdf"))
    fig.savefig(os.path.join(OUT, "Figure3.png"), dpi=300)
    plt.close(fig); print("wrote Figure3")


if __name__ == "__main__":
    workflow(); architecture(); print("done ->", OUT)
