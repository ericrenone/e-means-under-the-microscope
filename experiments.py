"""
Reproducible stress tests: e-means (Bessa et al., arXiv:2611.09501) vs Lloyd,
FIM / Mahalanobis-preconditioned k-means, and Sherman-Morrison online precision.

Everything here is numpy only. Run:  python3 -I experiments.py
"""
import numpy as np, json, time, sys

rng_global = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# e-means, implemented from the paper's description (Sec. 3, Eq. 5-7)
# ----------------------------------------------------------------------------
def emeans_laws(n, k, T):
    B = 0.035 * k**0.13 * n**0.88 * T**0.81
    B = int(min(2 ** round(np.log2(max(B, 1))), n))
    a0 = min(0.824 * k**-0.46 * B**0.90 * n**-0.45 * T**-0.11, 1.0)
    return B, a0

def assign(X, C):
    d2 = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
    return d2.argmin(1), d2

def mse(X, C):
    _, d2 = assign(X, C); return d2.min(1).mean()

def emeans(X, k, T=25, rng=None, revive=True, alpha_override=None, B_override=None,
           cosine=True, eps=0.1, revival_period=5, tau=0.1, finalize=True, log=None):
    rng = rng or np.random.default_rng()
    n, d = X.shape
    B, a0 = emeans_laws(n, k, T)
    if B_override: B = B_override
    if alpha_override is not None: a0 = alpha_override
    C0 = X[rng.choice(n, k, replace=False)]
    N = np.full(k, 1e-10); S = N[:, None] * C0
    M = T * (n // B); t = 0
    delta = 2.0**-8  # BF16 machine epsilon (paper: delta = BF16 eps)
    pattern = np.where(np.arange(d) % 2 == 0, 1 + delta, 1 - delta)
    n_revivals = 0; identical_children = 0
    for ep in range(T):
        perm = rng.permutation(n)
        for b in range(n // B):
            idx = perm[b*B:(b+1)*B]; xb = X[idx]
            C = S / N[:, None]
            lab, _ = assign(xb, C)
            nj = np.bincount(lab, minlength=k).astype(float)
            sj = np.zeros((k, d)); np.add.at(sj, lab, xb)
            a = a0 * (eps + (1-eps) * 0.5 * (1 + np.cos(np.pi * t / M))) if cosine else a0
            g = 1 - a
            N = g*N + (1-g)*nj; S = g*S + (1-g)*sj
            t += 1
            if revive and (t % revival_period == 0):
                dead = np.where(N < tau * N.mean())[0]
                if len(dead):
                    C = S / N[:, None]
                    donors = rng.choice(k, size=len(dead), replace=False, p=N/N.sum())
                    for j, jp in zip(dead, donors):
                        if j == jp: continue
                        cj = C[jp] * pattern; cjp = C[jp] * (2 - pattern)
                        if np.allclose(cj, cjp): identical_children += 1
                        N[j] = N[jp] = N[jp] / 2
                        S[j] = cj * N[j]; S[jp] = cjp * N[jp]
                        n_revivals += 1
        if log is not None: log.append(mse(X, S / N[:, None]))
    C = S / N[:, None]
    if finalize:  # last-epoch mean finalization (paper default)
        lab, _ = assign(X, C)
        for j in range(k):
            m = lab == j
            if m.any(): C[j] = X[m].mean(0)
    return C, dict(B=B, alpha0=a0, revivals=n_revivals, identical_children=identical_children,
                   dead_final=int((np.bincount(assign(X, C)[0], minlength=k) < 10).sum()))

# ----------------------------------------------------------------------------
# Lloyd (k-means++ init) and FIM / Mahalanobis-preconditioned Lloyd
# ----------------------------------------------------------------------------
def kpp_init(X, k, rng):
    n = X.shape[0]; C = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = ((X[:, None, :] - np.array(C)[None])**2).sum(2).min(1)
        C.append(X[rng.choice(n, p=d2/d2.sum())])
    return np.array(C)

def lloyd(X, k, iters=100, rng=None, init=None):
    rng = rng or np.random.default_rng()
    C = kpp_init(X, k, rng) if init is None else init.copy()
    for _ in range(iters):
        lab, _ = assign(X, C); Cn = C.copy()
        for j in range(k):
            m = lab == j
            if m.any(): Cn[j] = X[m].mean(0)
        if np.allclose(Cn, C): break
        C = Cn
    return C

def mahalanobis_kmeans(X, k, iters=100, rng=None, shrink=1e-3):
    """Hard-assignment EM with per-cluster full covariance = FIM-preconditioned
    k-means (Fisher metric of N(mu, Sigma) w.r.t. mu is Sigma^{-1})."""
    rng = rng or np.random.default_rng(); n, d = X.shape
    C = lloyd(X, k, 10, rng)  # cheap warm start
    P = np.array([np.eye(d)] * k); logdet = np.zeros(k)
    for _ in range(iters):
        D = np.empty((n, k))
        for j in range(k):
            r = X - C[j]; D[:, j] = np.einsum('ij,jk,ik->i', r, P[j], r) + logdet[j]
        lab = D.argmin(1); Cn = C.copy()
        for j in range(k):
            m = lab == j
            if m.sum() > d:
                Cn[j] = X[m].mean(0)
                Sg = np.cov(X[m].T) + shrink * np.eye(d)
                P[j] = np.linalg.inv(Sg); logdet[j] = np.linalg.slogdet(Sg)[1]
        if np.allclose(Cn, C): break
        C = Cn
    return C, P

# ----------------------------------------------------------------------------
# Sherman-Morrison online precision tracker with forgetting (one cluster)
# ----------------------------------------------------------------------------
class SMPrecision:
    def __init__(self, d, lam=0.98, ridge=1e-2):
        self.mu = np.zeros(d); self.P = np.eye(d) / ridge; self.lam = lam; self.w = 0.0
    def update(self, x):
        self.w = self.lam * self.w + 1
        e = x - self.mu; self.mu += e / self.w
        u = e / np.sqrt(self.lam)                           # exact rank-1 on (lam*Sigma + e e^T)
        Pu = self.P @ u
        self.P = (self.P - np.outer(Pu, Pu) / (1 + u @ Pu)) / self.lam
    def maha(self, x):
        r = x - self.mu; return r @ self.P @ r

# ----------------------------------------------------------------------------
def ari(a, b):
    from itertools import product
    a = np.asarray(a); b = np.asarray(b)
    n = len(a); ua, ub = np.unique(a), np.unique(b)
    M = np.array([[np.sum((a == i) & (b == j)) for j in ub] for i in ua])
    comb = lambda x: x * (x - 1) / 2
    sij = comb(M).sum(); sa = comb(M.sum(1)).sum(); sb = comb(M.sum(0)).sum(); sn = comb(n)
    exp = sa * sb / sn; mx = (sa + sb) / 2
    return (sij - exp) / (mx - exp + 1e-12)

def blobs(n, k, d, rng, sep=6.0, cov=None):
    mus = rng.normal(0, sep, (k, d)); lab = rng.integers(k, size=n)
    cov = np.eye(d) if cov is None else cov
    L = np.linalg.cholesky(cov)
    X = mus[lab] + rng.normal(size=(n, d)) @ L.T
    return X, lab

results = {}

# ============ Exp 1: low-d / small-n regime where the scaling laws were not fit
print("Exp1: low-d small-n")
r1 = []
for d in [2, 4, 8]:
    n, k = 2000, 10
    B, a0 = emeans_laws(n, k, 25)
    e_m, l_m, rev, dead = [], [], [], []
    for s in range(10):
        rng = np.random.default_rng(s); X, _ = blobs(n, k, d, rng, sep=4.0)
        C, info = emeans(X, k, rng=np.random.default_rng(100+s)); e_m.append(mse(X, C))
        l_m.append(mse(X, lloyd(X, k, rng=np.random.default_rng(200+s))))
        rev.append(info['revivals']); dead.append(info['dead_final'])
    r1.append(dict(d=d, B=B, alpha0=round(a0, 3), emeans_mse=float(np.mean(e_m)), lloyd_pp_mse=float(np.mean(l_m)),
                   ratio=float(np.mean(e_m) / np.mean(l_m)), revivals=float(np.mean(rev)), dead_final=float(np.mean(dead))))
    print(r1[-1])
results['exp1_lowd'] = r1

# ============ Exp 2: anisotropic / correlated clusters
print("Exp2: anisotropic")
r2 = []
for kappa in [1, 10, 100]:
    n, k, d = 3000, 3, 2
    a_e, a_l, a_m = [], [], []
    for s in range(8):
        rng = np.random.default_rng(s)
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        cov = Q @ np.diag([kappa, 1.0]) @ Q.T
        X, y = blobs(n, k, d, rng, sep=5.0, cov=cov)
        Ce, _ = emeans(X, k, rng=np.random.default_rng(100+s)); a_e.append(ari(y, assign(X, Ce)[0]))
        a_l.append(ari(y, assign(X, lloyd(X, k, rng=np.random.default_rng(200+s)))[0]))
        Cm, P = mahalanobis_kmeans(X, k, rng=np.random.default_rng(300+s))
        D = np.stack([np.einsum('ij,jk,ik->i', X-Cm[j], P[j], X-Cm[j]) - np.linalg.slogdet(P[j])[1] for j in range(k)], 1)
        a_m.append(ari(y, D.argmin(1)))
    r2.append(dict(condition_number=kappa, ari_emeans=float(np.mean(a_e)), ari_lloyd=float(np.mean(a_l)), ari_fim_mahalanobis=float(np.mean(a_m))))
    print(r2[-1])
results['exp2_anisotropic'] = r2

# ============ Exp 3: ill-conditioned feature scales (no standardization)
print("Exp3: feature scale")
r3 = []
for scale in [1, 100, 10000]:
    n, k, d = 3000, 4, 5
    a_e, a_l, a_m = [], [], []
    for s in range(8):
        rng = np.random.default_rng(s); X, y = blobs(n, k, d, rng, sep=4.0)
        X = X * np.array([scale, 1, 1, 1, 1])  # one nuisance feature dominates
        Ce, _ = emeans(X, k, rng=np.random.default_rng(100+s)); a_e.append(ari(y, assign(X, Ce)[0]))
        a_l.append(ari(y, assign(X, lloyd(X, k, rng=np.random.default_rng(200+s)))[0]))
        Cm, P = mahalanobis_kmeans(X, k, rng=np.random.default_rng(300+s), shrink=1e-6 * scale**2)
        D = np.stack([np.einsum('ij,jk,ik->i', X-Cm[j], P[j], X-Cm[j]) - np.linalg.slogdet(P[j])[1] for j in range(k)], 1)
        a_m.append(ari(y, D.argmin(1)))
    r3.append(dict(scale=scale, ari_emeans=float(np.mean(a_e)), ari_lloyd=float(np.mean(a_l)), ari_fim_mahalanobis=float(np.mean(a_m))))
    print(r3[-1])
results['exp3_scale'] = r3

# ============ Exp 4: concept drift -- cosine-annealed EMA vs SM tracker
print("Exp4: drift")
d = 4; steps = 20000; rng = np.random.default_rng(7)
drift = np.zeros(d); drift[0] = 1.0
err_cos, err_const, err_sm = [], [], []
N = 1e-10; S = np.zeros(d); mu_c = np.zeros(d); sm = SMPrecision(d, lam=0.995, ridge=1.0)
a0 = 0.5; eps = 0.1; M = steps
for t in range(steps):
    true_mu = drift * (t / steps) * 20.0          # linear drift, 20 units over the stream
    x = true_mu + rng.normal(size=d)
    a = a0 * (eps + (1-eps) * 0.5 * (1 + np.cos(np.pi * t / M)))   # e-means schedule, single cluster
    N = (1-a)*N + a*1; S = (1-a)*S + a*x
    mu_c = (1-a0*eps)*mu_c + a0*eps*x              # constant small EMA for reference
    sm.update(x)
    if t > steps // 2:                             # measure in the second half (annealed regime)
        err_cos.append(np.linalg.norm(S/N - true_mu)); err_const.append(np.linalg.norm(mu_c - true_mu)); err_sm.append(np.linalg.norm(sm.mu - true_mu))
results['exp4_drift'] = dict(tracking_err_emeans_cosine=float(np.mean(err_cos)), tracking_err_constant_alpha=float(np.mean(err_const)),
                             tracking_err_sm_forgetting=float(np.mean(err_sm)), note="mean ||mu_hat - mu_true|| over second half of stream")
print(results['exp4_drift'])

# ============ Exp 5: determinism / order sensitivity
print("Exp5: determinism")
n, k, d = 4000, 8, 16; rng = np.random.default_rng(3); X, _ = blobs(n, k, d, rng)
init = X[np.random.default_rng(11).choice(n, k, replace=False)]
Cl1 = lloyd(X, k, init=init); Cl2 = lloyd(X, k, init=init)
spread_e = []
for s in range(6):
    C, _ = emeans(X, k, rng=np.random.default_rng(500 + s)); spread_e.append(mse(X, C))
results['exp5_determinism'] = dict(lloyd_same_init_max_abs_diff=float(np.abs(Cl1-Cl2).max()),
                                   emeans_mse_across_batch_orders=[round(float(v), 4) for v in spread_e],
                                   emeans_mse_relative_spread=float((max(spread_e)-min(spread_e))/np.mean(spread_e)))
print(results['exp5_determinism'])

# ============ Exp 6: revival artifact -- multiplicative BF16-eps split on zero coordinates
print("Exp6: revival split")
delta = 2.0**-8
for name, c in [("centered_data_centroid_near_zero", np.array([0.0, 0.0, 0.0, 0.0])), ("generic", np.array([3.0, -2.0, 1.5, 0.7]))]:
    pattern = np.where(np.arange(4) % 2 == 0, 1 + delta, 1 - delta)
    c1, c2 = c * pattern, c * (2 - pattern)
    results.setdefault('exp6_revival', {})[name] = dict(child_separation=float(np.linalg.norm(c1 - c2)),
                                                      relative_to_norm=float(np.linalg.norm(c1-c2) / (np.linalg.norm(c) + 1e-300)))
print(results['exp6_revival'])

# ============ Exp 7: Sherman-Morrison exactness vs direct inverse, and the denominator guard
print("Exp7: SM exactness")
d = 32; rng = np.random.default_rng(1); A = rng.normal(size=(d, d)); A = A @ A.T + d*np.eye(d); Ai = np.linalg.inv(A)
u = rng.normal(size=d); Ai_sm = Ai - np.outer(Ai@u, Ai@u) / (1 + u@Ai@u); Ai_direct = np.linalg.inv(A + np.outer(u, u))
# drift of repeated SM (10k updates) without refactorization
P = Ai.copy(); Acc = A.copy()
for _ in range(10000):
    u = rng.normal(size=d) * 0.1; Pu = P @ u; P = P - np.outer(Pu, Pu) / (1 + u @ Pu); Acc += np.outer(u, u)
results['exp7_sm'] = dict(single_update_rel_err=float(np.linalg.norm(Ai_sm - Ai_direct) / np.linalg.norm(Ai_direct)),
                          after_10k_updates_rel_err=float(np.linalg.norm(P - np.linalg.inv(Acc)) / np.linalg.norm(np.linalg.inv(Acc))),
                          symmetric_after_10k=float(np.abs(P - P.T).max()))
print(results['exp7_sm'])

json.dump(results, open('/home/claude/emeans_analysis/results.json', 'w'), indent=2)
print("saved")
