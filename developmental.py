"""
developmental.py — BrainSpan developmental test of the autism brain-gene link.

Adult AHBA showed no spatial concordance (our well-powered null). ASD is
neurodevelopmental, so the signal may be TEMPORAL. BrainSpan is temporally rich
(8 pcw -> adult) but spatially coarse (~11 neocortical areas), so we ask the
well-powered temporal questions:

  A. CONVERGENCE (canonical positive control): do the 668 SFARI risk genes show
     elevated neocortical co-expression in a specific developmental window vs
     size-matched random gene sets (permutation)? Expect a midfetal peak
     (Willsey 2013 / Parikshak 2013) -> validates the developmental data is sound.

  B. NOVEL LINK: are genes whose ADULT expression spatially aligns with the ASD
     functional-disruption map (our transcriptome_r) developmentally PRENATAL-
     biased? Correlate adult disruption-alignment with prenatal/postnatal
     expression ratio across all overlapping genes. Tests whether the (adult-
     spatially-null) imaging-transcriptomic relationship is actually a
     developmental-timing phenomenon.

Uses the canonical BrainSpan RNA-Seq zip already on disk. Offline.
"""
import os, json, zipfile, warnings
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import imaging_transcriptomics as it

RESDIR = it.RESDIR
BS_ZIP = "/home/swarnabha/Downloads/Brainspan/rnaseq_genes_csv.zip"
CACHE  = os.path.join(RESDIR, "brainspan_cache")
rng = np.random.default_rng(42)

NEOCORTEX = {"DFC","VFC","MFC","OFC","M1C","S1C","M1C-S1C","IPC","A1C","STC",
             "ITC","V1C","Ocx","PCx","TCx","FCx"}   # neocortical acronyms


def age_window(s):
    n = float(s.split()[0]); u = s.split()[1]
    if u == "pcw":
        return "early_prenatal" if n < 13 else ("midfetal" if n < 25 else "late_prenatal")
    if u == "mos": return "infancy"
    if u == "yrs":
        return "childhood" if n < 12 else ("adolescence" if n < 20 else "adult")
    return "other"

WIN_ORDER = ["early_prenatal","midfetal","late_prenatal","infancy","childhood","adolescence","adult"]


def load_brainspan():
    os.makedirs(CACHE, exist_ok=True)
    with zipfile.ZipFile(BS_ZIP) as z:
        for f in ["columns_metadata.csv","rows_metadata.csv","expression_matrix.csv"]:
            if not os.path.exists(os.path.join(CACHE, f)):
                z.extract(f, CACHE)
    cols = pd.read_csv(os.path.join(CACHE, "columns_metadata.csv"))
    rows = pd.read_csv(os.path.join(CACHE, "rows_metadata.csv"))
    expr = pd.read_csv(os.path.join(CACHE, "expression_matrix.csv"), header=None, index_col=0)
    expr.columns = range(expr.shape[1])
    expr.index = rows["gene_symbol"].values
    cols["window"] = cols["age"].map(age_window)
    cols["neocortex"] = cols["structure_acronym"].isin(NEOCORTEX)
    return expr, cols, rows


def main():
    print("=" * 70)
    print("  BrainSpan developmental test of the autism brain-gene link")
    print("=" * 70)
    expr, cols, rows = load_brainspan()
    print(f"[data] {expr.shape[0]} genes x {expr.shape[1]} samples | "
          f"neocortical samples {cols.neocortex.sum()}")
    print("       window counts (neocortex): " +
          ", ".join(f"{w}:{((cols.window==w)&cols.neocortex).sum()}" for w in WIN_ORDER))

    # SFARI gene set (from our PPI graph)
    import torch
    g = torch.load(it.CONT_GRAPH, map_location="cpu")
    sfari = {str(g.gene_mapping[i]).upper() for i in range(len(g.gene_mapping))}
    expr.index = expr.index.astype(str).str.upper()
    expr = expr[~expr.index.duplicated()]
    expressed = expr.index[(expr.values > 1).mean(1) > 0.1]      # filter low genes
    E = expr.loc[expressed]
    sfari_in = [gsym for gsym in [list(sfari)] for gsym in gsym if gsym in E.index]
    sfari_in = [s for s in sfari if s in set(E.index)]
    print(f"[genes] expressed {E.shape[0]} | SFARI present {len(sfari_in)}")

    # ── A. developmental convergence (co-expression) by window ──────────────
    print("\n[A] SFARI neocortical co-expression vs random gene sets (per window):")
    def mean_coexpr(genes, samp):
        if len(samp) < 6 or len(genes) < 10: return np.nan
        M = E.loc[genes, samp].values
        M = np.log2(M + 1)
        C = np.corrcoef(M)
        iu = np.triu_indices(len(genes), 1)
        return np.nanmean(np.abs(C[iu]))
    allg = list(E.index)
    convergence = {}
    for w in WIN_ORDER:
        samp = cols.index[(cols.window == w) & cols.neocortex].tolist()
        obs = mean_coexpr(sfari_in, samp)
        if np.isnan(obs):
            continue
        null = np.array([mean_coexpr(list(rng.choice(allg, len(sfari_in), replace=False)), samp)
                         for _ in range(500)])
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)
        convergence[w] = dict(coexpr=float(obs), null_mean=float(np.nanmean(null)),
                              null_std=float(np.nanstd(null)),
                              null_lo=float(np.nanpercentile(null, 2.5)),
                              null_hi=float(np.nanpercentile(null, 97.5)),
                              p=float(p), n_samp=len(samp))
        flag = "  <-- CONVERGENCE" if p < 0.05 else ""
        print(f"    {w:16s} n={len(samp):3d}  SFARI |r|={obs:.3f}  null={np.nanmean(null):.3f}  p={p:.3f}{flag}")

    # ── B. adult disruption-alignment vs prenatal bias (novel link) ─────────
    print("\n[B] Do adult disruption-aligned genes show prenatal bias? (novel link)")
    r_path = os.path.join(RESDIR, "transcriptome_r.npy")
    ahba_csv = os.path.join(RESDIR, "ahba_region_gene_LH.csv")
    res_b = None
    if os.path.exists(r_path) and os.path.exists(ahba_csv):
        r_adult = np.load(r_path)
        ahba_genes = pd.read_csv(ahba_csv, index_col=0, nrows=1).columns.str.upper()
        align = pd.Series(np.abs(r_adult), index=ahba_genes)
        pre = cols.index[cols.window.isin(["early_prenatal","midfetal","late_prenatal"]) & cols.neocortex]
        post = cols.index[cols.window.isin(["childhood","adolescence","adult"]) & cols.neocortex]
        pre_m = np.log2(E[pre].mean(1) + 1); post_m = np.log2(E[post].mean(1) + 1)
        prenatal_bias = (pre_m - post_m)                       # >0 = prenatal-biased
        common = align.index.intersection(prenatal_bias.index)
        a = align.loc[common].groupby(level=0).mean()
        b = prenatal_bias.loc[common].groupby(level=0).mean()
        common2 = a.index.intersection(b.index)
        rho, p = stats.spearmanr(a.loc[common2], b.loc[common2])
        print(f"    genes overlapped: {len(common2)}")
        print(f"    Spearman(adult disruption-alignment, prenatal bias) = {rho:+.3f}, p={p:.2e}")
        # SFARI subset
        sf_mask = np.array([gsym in set(sfari_in) for gsym in common2])
        if sf_mask.sum() > 20:
            rho_sf, p_sf = stats.spearmanr(a.loc[common2][sf_mask], b.loc[common2][sf_mask])
            print(f"    within SFARI ({sf_mask.sum()} genes): rho={rho_sf:+.3f}, p={p_sf:.3f}")
        res_b = dict(n=int(len(common2)), rho=float(rho), p=float(p))
    else:
        print("    (run full_transcriptome.py first for transcriptome_r.npy)")

    # ── figure: convergence by window ───────────────────────────────────────
    if convergence:
        ws = [w for w in WIN_ORDER if w in convergence]
        obs = [convergence[w]["coexpr"] for w in ws]
        nul = [convergence[w]["null_mean"] for w in ws]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(ws))
        ax.bar(x - 0.2, obs, 0.4, label="SFARI genes", color="#C0392B")
        ax.bar(x + 0.2, nul, 0.4, label="random genes (null)", color="#BDC3C7")
        for i, w in enumerate(ws):
            if convergence[w]["p"] < 0.05:
                ax.text(i, max(obs[i], nul[i]) + 0.005, "*", ha="center", fontsize=14)
        ax.set_xticks(x); ax.set_xticklabels(ws, rotation=20, fontsize=8)
        ax.set_ylabel("mean |co-expression|"); ax.legend(fontsize=8)
        ax.set_title("Autism-risk-gene developmental convergence in neocortex\n"
                     "(* p<0.05 vs size-matched random gene sets)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(RESDIR, "Fig_developmental.png"))
        fig.savefig(os.path.join(RESDIR, "Fig_developmental.pdf"))
        print("\n  ✓ Fig_developmental saved")

    json.dump(dict(convergence=convergence, prenatal_link=res_b),
              open(os.path.join(RESDIR, "developmental_results.json"), "w"), indent=2)
    print("=" * 70)


if __name__ == "__main__":
    main()
