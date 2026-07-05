"""
combat.py — parametric ComBat harmonization (Johnson 2007), from scratch.

neuroCombat is not installed; this is a clean, dependency-free fit/transform
implementation with empirical-Bayes batch adjustment and biological-covariate
preservation. For ASD classification we preserve age + sex (NOT diagnosis — that
would leak labels) and harmonize across scanner site.

API:
    cb = ComBat().fit(X_train, batch_train, mod_train)
    Xh_train = cb.transform(X_train, batch_train, mod_train)
    Xh_test  = cb.transform(X_test,  batch_test,  mod_test)   # known batches only
"""
import numpy as np


def _aprior(g): m, s2 = g.mean(), g.var(); return (2*s2 + m**2) / s2
def _bprior(g): m, s2 = g.mean(), g.var(); return (m*s2 + m**3) / s2


def _it_sol(s, g_hat, d_hat, g_bar, t2, a, b, tol=1e-4):
    g_old, d_old = g_hat.copy(), d_hat.copy()
    n = (~np.isnan(s)).sum(1)
    change = 1.0
    while change > tol:
        g_new = (t2*n*g_hat + d_old*g_bar) / (t2*n + d_old)
        sum2 = ((s - g_new[:, None])**2).sum(1)
        d_new = (0.5*sum2 + b) / (0.5*n + a - 1)
        change = max(np.max(np.abs(g_new-g_old)/ (np.abs(g_old)+1e-12)),
                     np.max(np.abs(d_new-d_old)/ (np.abs(d_old)+1e-12)))
        g_old, d_old = g_new, d_new
    return g_old, d_old


class ComBat:
    def fit(self, X, batch, mod=None):
        Y = np.asarray(X, float).T                      # p x n
        p, n = Y.shape
        batch = np.asarray(batch)
        self.batches_ = list(np.unique(batch))
        bdm = np.array([[1.0 if b == bb else 0.0 for bb in self.batches_] for b in batch])  # n x nb
        nb = len(self.batches_)
        mod = np.zeros((n, 0)) if mod is None else np.asarray(mod, float)
        design = np.hstack([bdm, mod])                  # n x (nb+k)
        B_hat = np.linalg.lstsq(design, Y.T, rcond=None)[0]   # (nb+k) x p
        sizes = bdm.sum(0)
        grand = (sizes / n) @ B_hat[:nb]                # p,
        resid = Y - (design @ B_hat).T
        var_pooled = (resid**2).sum(1) / n              # p,
        var_pooled[var_pooled < 1e-12] = 1e-12
        self.grand_, self.var_, self.Bcov_ = grand, var_pooled, B_hat[nb:]
        stand_mean = grand[:, None] + (mod @ self.Bcov_).T
        s = (Y - stand_mean) / np.sqrt(var_pooled)[:, None]
        gamma_hat = np.linalg.lstsq(bdm, s.T, rcond=None)[0]   # nb x p
        delta_hat = np.array([s[:, batch == self.batches_[i]].var(1) for i in range(nb)])
        self.gamma_star_, self.delta_star_ = {}, {}
        for i, bb in enumerate(self.batches_):
            si = s[:, batch == bb]
            g_bar, t2 = gamma_hat[i].mean(), gamma_hat[i].var()
            a, b = _aprior(delta_hat[i]), _bprior(delta_hat[i])
            gs, ds = _it_sol(si, gamma_hat[i], delta_hat[i], g_bar, t2, a, b)
            self.gamma_star_[bb], self.delta_star_[bb] = gs, ds
        return self

    def transform(self, X, batch, mod=None):
        Y = np.asarray(X, float).T
        n = Y.shape[1]
        batch = np.asarray(batch)
        mod = np.zeros((n, 0)) if mod is None else np.asarray(mod, float)
        stand_mean = self.grand_[:, None] + (mod @ self.Bcov_).T
        s = (Y - stand_mean) / np.sqrt(self.var_)[:, None]
        out = s.copy()
        for j in range(n):
            bb = batch[j]
            if bb in self.gamma_star_:
                out[:, j] = (s[:, j] - self.gamma_star_[bb]) / np.sqrt(self.delta_star_[bb])
        adj = out * np.sqrt(self.var_)[:, None] + stand_mean
        return adj.T


if __name__ == "__main__":
    # self-test: ComBat should shrink between-batch variance in feature means
    rng = np.random.default_rng(0)
    n, p = 300, 50
    batch = rng.integers(0, 4, n)
    X = rng.normal(size=(n, p)) + (batch[:, None] * 0.8)     # strong batch offset
    cb = ComBat().fit(X, batch)
    Xh = cb.transform(X, batch)
    def betw(D): return np.mean([D[batch == b].mean(0) for b in range(4)], 0).std()
    print(f"between-site spread of feature means: raw {betw(X):.3f} -> ComBat {betw(Xh):.3f}")
