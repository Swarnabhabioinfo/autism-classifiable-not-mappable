"""
make_surface_figure.py — Supplementary Fig. 1: cortical-surface renderings of the
two left-hemisphere maps the spin test compared (ASD disruption vs aggregate SFARI
expression). Visualises WHY the concordance is null: the two spatial patterns are
unrelated (r = -0.04, p_spin = 0.85). Real arrays only; abagen LH-50 pipeline.

Projection: Schaefer-100 (FSLMNI152) -> fsaverage5 surface (nearest-neighbour), the
standard nilearn illustration path. Left hemisphere, lateral + medial views.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import datasets, surface
from nilearn.plotting import plot_surf_stat_map
import imaging_transcriptomics as it

R = it.RESDIR
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_submission")
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "savefig.dpi": 400})

fs = datasets.fetch_surf_fsaverage("fsaverage5")
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)

# Schaefer parcel label per left-hemisphere vertex (nearest-neighbour projection)
vlab = surface.vol_to_surf(atlas.maps, fs["pial_left"],
                           interpolation="nearest").astype(int)

# ---- per-parcel values (real arrays) ----
disr = np.load(os.path.join(R, "disruption_map.npy"))            # (100,) t-stat
sfari_lh = np.load(os.path.join(R, "sfari_expr_map_lh.npy"))     # (50,) LH aggregate
# recover the LH Schaefer indices that sfari_lh is aligned to (abagen order)
coords, hemi = it.schaefer_centroids()
lh_idx = set(np.where(hemi == 0)[0].tolist())
expr = pd.read_csv(os.path.join(R, "ahba_region_gene_LH.csv"), index_col=0)
expr = expr.dropna(axis=0, how="all").dropna(axis=1, how="any")
region_ids = np.array([int(r) for r in expr.index]) - 1
lh_regions = region_ids[np.array([rid in lh_idx for rid in region_ids])]

val_disr = np.full(101, np.nan); val_disr[1:] = disr             # index by parcel label
val_sfari = np.full(101, np.nan)
for j, rid in enumerate(lh_regions):
    val_sfari[rid + 1] = sfari_lh[j]

def vert(table):
    out = np.full(len(vlab), np.nan)
    m = vlab > 0
    out[m] = table[vlab[m]]
    return out
vd, vs = vert(val_disr), vert(val_sfari)
# The Spearman/spin test compares topography (region ranks), so display the
# standardized spatial pattern each map contributes. Disruption t is negative
# throughout (a global connectivity-strength reduction); z-scoring reveals its
# topography and puts both maps on one comparable diverging scale.
def zmap(v): return (v - np.nanmean(v)) / np.nanstd(v)
vd_z, vs_z = zmap(vd), zmap(vs)

# ---- 2 rows (disruption, SFARI) x 2 views (lateral, medial) ----
fig = plt.figure(figsize=(7.4, 6.6))
mesh, bg = fs["infl_left"], fs["sulc_left"]
rows = [(vd_z, "ASD disruption ($t$, $z$-scored)"),
        (vs_z, "SFARI expression ($z$-scored)")]
for ri, (data, rlab) in enumerate(rows):
    for ci, view in enumerate(("lateral", "medial")):
        ax = fig.add_subplot(2, 2, ri * 2 + ci + 1, projection="3d")
        plot_surf_stat_map(mesh, data, hemi="left", view=view, bg_map=bg,
                           cmap="RdBu_r", symmetric_cbar=True, vmax=2.5,
                           colorbar=(ci == 1), axes=ax, figure=fig, bg_on_data=True)
        if ci == 0:
            ax.set_title(rlab, fontsize=10, fontweight="bold", y=0.96)
        ax.text2D(0.5, 0.02, view, transform=ax.transAxes, ha="center", fontsize=8)

fig.suptitle("Left-hemisphere cortical topography (Schaefer-100 → fsaverage5)",
             fontsize=10.5, fontweight="bold", y=0.99)
fig.text(0.5, 0.005,
         "Maps are $z$-scored to compare topography; the two are spatially "
         "uncorrelated ($r$ = −0.04, $p_{spin}$ = 0.85).",
         ha="center", fontsize=8.5)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(os.path.join(OUT, "FigureS1.pdf"))
fig.savefig(os.path.join(OUT, "FigureS1.png"), dpi=400)
print("wrote FigureS1.pdf (+png)")
