"""
deep_baselines_loso.py — BrainNetCNN under the SAME leave-one-site-out protocol
as the linear baselines, to test 'does deep learning beat linear FC for ASD?'.

BrainNetCNN (Kawahara 2017) is the canonical FC deep model: Edge-to-Edge ->
Edge-to-Node -> Node-to-Graph. Trained on signed Fisher-z FC, motion-QC'd,
class-weighted. Fixed epoch budget (no test peeking). CPU.
"""
import os, re, json, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, roc_curve
import imaging_transcriptomics as it

torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(os.cpu_count() or 4)
RESDIR, N = it.RESDIR, it.N_ROI


def build_fc_matrices():
    pheno = it.load_phenotype(); Xs, ys, sites = [], [], []
    for f in sorted(os.listdir(it.DATA_DIR)):
        if not f.endswith(".npy"): continue
        nums = re.findall(r"\d+", f)
        if not nums: continue
        sid = str(int("".join(nums)))
        if sid not in pheno.index: continue
        row = pheno.loc[sid]
        if getattr(row, "ndim", 1) == 2: row = row.iloc[0]
        m = row["func_mean_fd"]
        if np.isnan(m) or m > it.FD_THRESH: continue
        ts = np.load(os.path.join(it.DATA_DIR, f))
        if ts.ndim != 2 or ts.shape[1] != N: continue
        ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-8)
        r = np.nan_to_num(np.corrcoef(ts, rowvar=False), nan=0.0)
        np.clip(r, -0.999, 0.999, out=r); z = np.arctanh(r); np.fill_diagonal(z, 0)
        Xs.append(z.astype(np.float32)); ys.append(int(row["ASD"])); sites.append(str(row["SITE_ID"]))
    return np.stack(Xs), np.array(ys), np.array(sites)


class E2E(nn.Module):
    def __init__(s, ci, co, n):
        super().__init__(); s.row = nn.Conv2d(ci, co, (1, n)); s.col = nn.Conv2d(ci, co, (n, 1)); s.n = n
    def forward(s, x):
        r = s.row(x); c = s.col(x)
        return r.expand(-1, -1, s.n, -1) + c.expand(-1, -1, -1, s.n)


class BrainNetCNN(nn.Module):
    def __init__(s, n=100):
        super().__init__()
        s.e2e1 = E2E(1, 16, n); s.e2e2 = E2E(16, 16, n)
        s.e2n = nn.Conv2d(16, 32, (1, n)); s.n2g = nn.Conv2d(32, 64, (n, 1))
        s.fc1 = nn.Linear(64, 64); s.fc2 = nn.Linear(64, 2)
        s.do = nn.Dropout(0.5); s.act = nn.LeakyReLU(0.1)
    def forward(s, x):
        x = s.act(s.e2e1(x)); x = s.act(s.e2e2(x))
        x = s.act(s.e2n(x)); x = s.act(s.n2g(x))
        x = x.view(x.size(0), -1); x = s.do(s.act(s.fc1(x)))
        return s.fc2(x)


def train_eval():
    X, y, site = build_fc_matrices()
    print(f"[data] {len(y)} subj | ASD {y.sum()} CTRL {(y==0).sum()} | {len(set(site))} sites", flush=True)
    logo = LeaveOneGroupOut(); oof = np.full(len(y), np.nan); per_site = {}
    for fold, (tr, te) in enumerate(logo.split(X, y, groups=site)):
        Xtr = torch.tensor(X[tr]).unsqueeze(1); ytr = torch.tensor(y[tr])
        Xte = torch.tensor(X[te]).unsqueeze(1)
        w = torch.tensor([len(ytr)/(2*(ytr==0).sum()), len(ytr)/(2*(ytr==1).sum())], dtype=torch.float)
        net = BrainNetCNN(N); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
        lossf = nn.CrossEntropyLoss(weight=w)
        net.train()
        for ep in range(40):
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 32):
                b = perm[i:i+32]
                opt.zero_grad(); out = net(Xtr[b]); loss = lossf(out, ytr[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.softmax(net(Xte), 1)[:, 1].numpy()
        oof[te] = p
        if len(np.unique(y[te])) == 2: per_site[site[te][0]] = float(roc_auc_score(y[te], p))
        print(f"  fold {fold+1}/{len(set(site))} site={site[te][0]:8s} "
              f"AUC={per_site.get(site[te][0], float('nan')):.3f}", flush=True)
    auc = float(roc_auc_score(y, oof))
    rng = np.random.default_rng(123)
    ci = [float(np.percentile([roc_auc_score(y[idx], oof[idx]) for idx in
          (rng.integers(0, len(y), len(y)) for _ in range(2000))
          if len(np.unique(y[idx])) == 2], q)) for q in (2.5, 97.5)]
    fpr, tpr, _ = roc_curve(y, oof)
    np.savez(os.path.join(RESDIR, "roc_brainnetcnn.npz"), fpr=fpr, tpr=tpr, auc=auc)
    json.dump(dict(model="BrainNetCNN", loso_auc=auc, ci=ci,
                   per_site_auc=per_site,
                   per_site_mean=float(np.mean(list(per_site.values())))),
              open(os.path.join(RESDIR, "deep_brainnetcnn.json"), "w"), indent=2)
    print(f"\n[BrainNetCNN] LOSO AUC = {auc:.3f}  CI {ci}", flush=True)
    print(f"  per-site mean {np.mean(list(per_site.values())):.3f}", flush=True)


if __name__ == "__main__":
    train_eval()
