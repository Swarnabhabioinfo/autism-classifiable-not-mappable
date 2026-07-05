"""
coupling.py — Transcriptional-functional coupling and its disruption in autism.

Different object from the (null) nodal maps: do region-pairs that are functionally
connected also have similar gene-expression profiles (correlated gene expression,
CGE)? And is this FC<->CGE coupling WEAKER in ASD than controls?

  CGE[i,j]  = corr of AHBA expression profiles between regions i,j  (50x50 LH)
  FC[i,j]   = group mean signed Fisher-z connectivity                (50x50 LH)
  coupling  = partial Spearman(FC_edges, CGE_edges | distance_edges)
  null      = region spin-permutation of FC (1000)
  ASD vs CTRL coupling difference: subject bootstrap CI
"""
import os, re, json, numpy as np, pandas as pd
from scipy import stats
import imaging_transcriptomics as it
RESDIR, N = it.RESDIR, it.N_ROI


def partial_spearman(a, b, c):
    ra, rb, rc = (stats.rankdata(x) for x in (a, b, c))
    def resid(y, x):
        x = np.c_[np.ones_like(x), x]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        return y - x @ beta
    return stats.pearsonr(resid(ra, rc), resid(rb, rc))[0]


def group_fc(tab, mats, mask):
    return np.mean(mats[mask], 0)


def subject_fc_matrices():
    ph = it.load_phenotype(); rows, mats = [], []
    for f in sorted(os.listdir(it.DATA_DIR)):
        if not f.endswith(".npy"): continue
        nums = re.findall(r"\d+", f)
        if not nums: continue
        sid = str(int("".join(nums)))
        if sid not in ph.index: continue
        r = ph.loc[sid]
        if getattr(r, "ndim", 1) == 2: r = r.iloc[0]
        fd = r["func_mean_fd"]
        if pd.isna(fd) or fd > it.FD_THRESH: continue
        ts = np.load(os.path.join(it.DATA_DIR, f))
        if ts.ndim != 2 or ts.shape[1] != N: continue
        ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-8)
        c = np.nan_to_num(np.corrcoef(ts, rowvar=False), nan=0.0)
        np.clip(c, -0.999, 0.999, out=c); z = np.arctanh(c); np.fill_diagonal(z, 0)
        mats.append(z.astype(np.float32)); rows.append(int(r["ASD"]))
    return np.array(rows), np.stack(mats)


def main():
    print("=" * 70)
    print("  Transcriptional-functional coupling & autism disruption")
    print("=" * 70)
    expr = pd.read_csv(os.path.join(RESDIR, "ahba_region_gene_LH.csv"), index_col=0)
    expr = expr.dropna(axis=1, how="any")
    reg_ids = np.array([int(r) for r in expr.index]) - 1
    coords, hemi = it.schaefer_centroids()
    lh = np.where(hemi == 0)[0]
    keep = np.array([rid in set(lh.tolist()) for rid in reg_ids])
    expr = expr.iloc[keep]; reg_ids = reg_ids[keep]
    E = stats.zscore(expr.values, axis=0)
    CGE = np.corrcoef(E)                                   # (R,R) gene co-expression
    R = CGE.shape[0]
    print(f"[data] {R} LH regions x {expr.shape[1]} genes -> CGE {CGE.shape}")

    # distance between these regions
    c = coords[reg_ids]
    D = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)

    asd, mats = subject_fc_matrices()
    mats_lh = mats[:, reg_ids][:, :, reg_ids]            # (n,R,R)
    iu, ju = np.triu_indices(R, 1)
    cge_e, d_e = CGE[iu, ju], D[iu, ju]

    def coupling(FC):
        return partial_spearman(FC[iu, ju], cge_e, d_e)

    fc_ctrl = mats_lh[asd == 0].mean(0)
    fc_asd  = mats_lh[asd == 1].mean(0)
    cpl_ctrl, cpl_asd = coupling(fc_ctrl), coupling(fc_asd)

    # spin null on regions for the control coupling (is coupling real at all?)
    perms = []
    rng = np.random.default_rng(0)
    cL = c
    for _ in range(1000):
        Rot = it._rand_rotation(); rc = cL @ Rot.T
        Dm = np.linalg.norm(rc[:, None, :] - cL[None, :, :], axis=2)
        perms.append(Dm.argmin(1))
    null = []
    for p in perms:
        FCp = fc_ctrl[p][:, p]
        null.append(coupling(FCp))
    null = np.array(null)
    p_spin = (np.sum(np.abs(null) >= abs(cpl_ctrl)) + 1) / 1001

    # ASD vs CTRL coupling difference + per-arm 95% CIs: subject bootstrap
    diffs, bs_ctrl, bs_asd = [], [], []
    idx_a = np.where(asd == 1)[0]; idx_c = np.where(asd == 0)[0]
    for _ in range(1000):
        ba = rng.choice(idx_a, len(idx_a), replace=True)
        bc = rng.choice(idx_c, len(idx_c), replace=True)
        cc = coupling(mats_lh[bc].mean(0)); ca = coupling(mats_lh[ba].mean(0))
        bs_ctrl.append(cc); bs_asd.append(ca); diffs.append(cc - ca)
    diffs = np.array(diffs)
    ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
    ci_ctrl = (np.percentile(bs_ctrl, 2.5), np.percentile(bs_ctrl, 97.5))
    ci_asd  = (np.percentile(bs_asd, 2.5), np.percentile(bs_asd, 97.5))
    p_diff = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    # save the control spin-null distribution so the positive control can be
    # shown the same way as the brain-gene null (observed vs spin null)
    np.save(os.path.join(RESDIR, "coupling_null_ctrl.npy"), null)

    print(f"\n[coupling] FC<->CGE partial Spearman (| distance):")
    print(f"    Controls : r={cpl_ctrl:+.3f}  spin p={p_spin:.4f}"
          f"{'  REAL coupling' if p_spin<0.05 else '  n.s.'}")
    print(f"    Autism   : r={cpl_asd:+.3f}")
    print(f"    CTRL-ASD : Δ={cpl_ctrl-cpl_asd:+.3f}  95% CI [{ci[0]:+.3f},{ci[1]:+.3f}] "
          f" p={p_diff:.4f}{'  DISRUPTED in ASD' if p_diff<0.05 else '  n.s.'}")
    json.dump(dict(coupling_ctrl=float(cpl_ctrl), coupling_asd=float(cpl_asd),
                   p_spin_ctrl=float(p_spin), diff=float(cpl_ctrl-cpl_asd),
                   diff_ci=[float(ci[0]), float(ci[1])], p_diff=float(p_diff),
                   ci_ctrl=[float(ci_ctrl[0]), float(ci_ctrl[1])],
                   ci_asd=[float(ci_asd[0]), float(ci_asd[1])],
                   n_ctrl=int((asd == 0).sum()), n_asd=int((asd == 1).sum())),
              open(os.path.join(RESDIR, "coupling_results.json"), "w"), indent=2)
    print("=" * 70)


if __name__ == "__main__":
    main()
