# e-means under the microscope

### Why an exponential-moving-average k-means is still a zeroth-order method, and where Lloyd, Fisher preconditioning, and Sherman–Morrison rank-1 updates beat it

> **Scope.** This is a formal dissection of *e-means* (Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou — Meta FAIR / ENPC, arXiv:2611.09501, 6 Nov 2026) against three reference methods: Lloyd's algorithm, Fisher-Information-Matrix (FIM) preconditioned clustering, and Sherman–Morrison (SM) online precision tracking. It is not a takedown. e-means does what it claims inside its regime (n ≥ 10⁶, d ≈ 100–3000, embedding data, GPU). The point of this document is to make the *boundary* of that regime precise, derive *why* the boundary is where it is, and show what sits on the other side of it.
>
> Every numerical claim is either (a) from the paper, cited by table/figure, or (b) from `experiments.py` in this folder, which reimplements e-means from the paper's equations and is reproducible with `python3 -I experiments.py` (numpy only, ~3 min on CPU).

---

## 0. Verdict in one paragraph

e-means is Lloyd's algorithm with a *decayed count* standing in for curvature. Written as an optimizer, its per-cluster step is $\eta_j = \alpha\, n_j / N_j'$ applied to the Euclidean gradient of the batch loss — a diagonal, isotropic, stale Newton step. That single sentence predicts all five failure modes below: it has no notion of cluster *shape* (anisotropy, correlation, feature scale), its curvature estimate goes stale under drift and is then *deliberately frozen* by cosine annealing, its scaling laws were fit on high-d data and extrapolate to $\alpha_0=1$ (full replacement) on small/low-d problems, its revival heuristic is count-blind to the most common small-n failure (two centroids on one blob), and it trades Lloyd's monotone, order-independent descent for a batch-order-dependent trajectory. FIM preconditioning supplies the missing metric ($\Sigma_j^{-1}$); Sherman–Morrison supplies that metric *online and exactly* at $O(d^2)$ per update with machine-precision agreement to a direct inverse after 10⁴ updates. Lloyd remains the right tool when $n$ is small enough to afford it, because then its guarantees cost nothing.

---

## 1. What e-means is, formally

### 1.1 The update (paper §3)

Per cluster $j$, e-means stores a count $N_j$ and a sum $S_j$, never a centroid. For a mini-batch $B$ with $n_j$ points assigned to $j$ and their sum $s_j$:

$$
N_j \leftarrow \gamma N_j + (1-\gamma)\, n_j,\qquad
S_j \leftarrow \gamma S_j + (1-\gamma)\, s_j,\qquad
c_j := S_j / N_j,\qquad
a(x) = \arg\min_j \|x - S_j/N_j\|^2 .
$$

Step $\alpha = 1-\gamma$ follows a cosine schedule (paper Eq. 5):

$$
\alpha(t) = \alpha_0\Big[\varepsilon + (1-\varepsilon)\tfrac{1}{2}\big(1+\cos(\pi t/M)\big)\Big],\quad \varepsilon = 0.1,\quad M = T\lfloor n/B\rfloor .
$$

Initialization: $N_j = 10^{-10}$, $S_j = N_j c_j^{(0)}$. Finalization (default): replace each centroid by the plain mean of its last-epoch assignments — i.e. one Lloyd M-step.

**Scaling laws** (Eq. 6–7), fit by OLS in log-space on DEEP ($n=10^6$, $k=1024$), ~75k runs:

$$
\alpha_0 = \min\!\big(0.824\; k^{-0.46}\, B^{0.90}\, n^{-0.45}\, T^{-0.11},\; 1\big),\qquad
B = 0.035\; k^{0.13}\, n^{0.88}\, T^{0.81}\ \text{(rounded to a power of two, capped at } n).
$$

The paper states that an exponent in $d$ was tried and dropped as unstable. **Dimension does not appear in either law.** Keep that in mind for §3.1.

**Revival** (every 5 batches): any cluster with $N_j < 0.1\cdot\overline{N}$ is reseeded by splitting a donor drawn with probability $\propto N_{j'}$:

$$
c_j = c_{j'} \odot [1+\delta, 1-\delta, 1+\delta, \dots],\qquad c_{j'} \leftarrow c_{j'} \odot [1-\delta, 1+\delta, \dots],\qquad \delta = \varepsilon_{\mathrm{BF16}} = 2^{-8},
$$

with the donor's count halved. Ablation (Table 1, DEEP 1M): removing revival costs +2.24% test MSE, the largest single component; without it 26% of clusters end with <10 vectors (Table 14).

### 1.2 The same update as an optimizer (the key identity)

Let $m_j = s_j/n_j$ be the batch mean of cluster $j$ and $N_j' = \gamma N_j + (1-\gamma) n_j$ the post-update count. Then

$$
c_j' \;=\; \frac{\gamma S_j + (1-\gamma) s_j}{N_j'}
\;=\; c_j \;+\; \underbrace{\frac{(1-\gamma)\, n_j}{N_j'}}_{\eta_j}\,\big(m_j - c_j\big).
$$

The batch k-means loss is $L_B(c) = \tfrac12\sum_{x\in B}\|x - c_{a(x)}\|^2$, whose gradient is $\nabla_{c_j} L_B = n_j (c_j - m_j)$. Hence

$$
\boxed{\;c_j' = c_j - \big(N_j'\big)^{-1} I_d\;\nabla_{c_j} L_B\;}
$$

**e-means is preconditioned gradient descent with preconditioner $P_j = (N_j')^{-1} I_d$.** Three immediate corollaries:

1. **Lloyd is the $\gamma = 0$, $B = $ full-data special case**: $N_j' = n_j$, $\eta_j = 1$, which is the *exact* Newton step for the quadratic loss restricted to a fixed assignment. e-means replaces the true Hessian block $n_j I_d$ by a *decayed, stale* count. The Faiss README's remark that full-data with forgetting rate 1 "matches Lloyd" is exactly this.
2. **The preconditioner is isotropic.** Nothing in $N_j'$ knows the shape of cluster $j$. This is the root of failure modes 2 and 3.
3. **The step-size schedule is a curvature-staleness schedule.** Cosine annealing drives $\alpha \to 0.1\alpha_0$, which means $N_j'$ becomes a long-memory average, which means the implicit Hessian stops responding to the data. Under stationarity that is variance reduction. Under drift it is lag. This is failure mode 4.

The paper itself notes the EMA–SGD equivalence with learning rate $\eta = (1-\gamma)/2$ from Łańcucki et al. (2020) and the view of EMA as hard-EM with decayed sufficient statistics (Nowlan 1991; Neal & Hinton 1998). The paper gives no convergence theorem of its own; its theoretical anchors are Lloyd's finite convergence, the $O(1/t)$ rate for stochastic k-means (Tang & Monteleoni 2017), and the SGD-with-stochastic-learning-rate interpretation of online k-means (Dasgupta, Mahajan, So, AISTATS 2022). Those results cover MacQueen/Sculley running means, not a constant-decay EMA with cosine annealing and revival; they do not transfer without additional work.

### 1.3 What is new in e-means (credit where due)

- Constant-decay EMA on *both* count and sum (vs. Sculley's running mean, which has $\eta_j = 1/N_j$ and cannot forget).
- Cosine annealing of the forgetting rate, which the paper says had not been used in this context.
- Empirically fitted, hyperparameter-free batch/step laws.
- Count-based revival with donor splitting, which is the single largest contributor to its quality.
- Observation (Appendix E) that Adam/SGD degrade at large batch while EMA does not; the paper's hypothesis is that EMA keeps centroids in the convex hull of assigned points. (Stated as a hypothesis, not proven.)

Results: on DEEP*, DINO10B, SPACEV, SimSearchNet++ at 1M–1B it reaches Lloyd-25 loss in 1.4–2.6 epochs, 9.5–45.5× faster than Flash-KMeans (Table 4). Those are real and large.

---

## 2. The four methods in one frame

All four minimize (or hard-EM-approximate) a sum-of-squares or Gaussian-likelihood objective. They differ only in the **metric** they descend in and in how **exactly** they maintain it.

| | **Lloyd** | **e-means** | **FIM-preconditioned (Mahalanobis / natural-gradient hard-EM)** | **Sherman–Morrison online precision** |
|---|---|---|---|---|
| Objective | $\sum_i \min_j \|x_i-c_j\|^2$ | same | $\sum_i \min_j \big[(x_i-c_j)^\top\Sigma_j^{-1}(x_i-c_j) + \log\det\Sigma_j\big]$ | same as FIM, maintained incrementally |
| Update order | exact block-Newton (M-step is closed form) | 1st-order with isotropic stale-count preconditioner $(N_j')^{-1}I$ | 2nd-order in the Fisher metric: $P_j = \Sigma_j^{-1}$ (= FIM of $\mathcal N(\mu,\Sigma)$ w.r.t. $\mu$, up to $n_j$) | same metric as FIM, with $\Sigma_j^{-1}$ updated by exact rank-1 identity per sample |
| Geometric assumption | isotropic, equal-variance Voronoi cells | isotropic, equal-variance Voronoi cells | elliptical cells, per-cluster shape | elliptical cells, per-cluster shape, *time-varying* |
| State per cluster | $d$ | $d+1$ (+ optional slow EMAs) | $d + d^2$ (or $2d$ diagonal, $d + dr$ low-rank) | $d + d^2$ (precision) |
| Cost per point | $O(kd)$ | $O(kd)$ | $O(kd^2)$ assign; $O(d^3)$ per re-estimate | $O(kd^2)$ assign; $O(d^2)$ per update, **no inversion** |
| Determinism | deterministic given init | depends on batch order, revival RNG, BF16 ties | deterministic given init | deterministic given stream order (it *is* the stream) |
| Monotone descent | yes, every iteration | no (only the final Lloyd step) | yes (hard-EM) | n/a (tracker, not optimizer) |
| Handles drift | no (batch) | poorly: cosine anneals memory to floor | no (batch) | yes, with forgetting factor $\lambda$ |
| Handles anisotropy / feature scale | no | no | yes | yes |
| Fails when | $n$ too big to afford $O(nkdT)$ passes; also high-d fixed-point pathology (§3.6) | $n$ small, $d$ small, $\kappa(\Sigma)\gg1$, drift, exactness required | $d$ large without structure ($d^2$ memory, $n_j < d$ singular) | $1+u^\top P u \to 0$ (numerically); $d^2$ memory |
| Sweet spot | $n \lesssim 10^6$, needs guarantees | $n \gtrsim 10^6$, $d \gtrsim 100$, isotropic-ish embeddings, GPU | $d \lesssim 10^3$, elliptical clusters, moderate $n$ | streaming, drift, anomaly scoring, per-cluster Mahalanobis online |

**The one-line summary of the table:** e-means and Lloyd share the same (wrong, for shaped data) metric; FIM and SM share the same (right) metric; Lloyd and FIM maintain their metric exactly in batch; SM maintains it exactly online; e-means maintains a stale scalar approximation of it online.

---

## 3. Five failure modes, with mechanism, evidence, and fix

Experimental setup for all numbers from `experiments.py`: isotropic Gaussian blobs unless stated, e-means run exactly per §1.1 with the published laws, $T=25$, random init (the paper's default), revival on, last-epoch-mean finalization. Lloyd uses k-means++ unless stated. "FIM" is hard-EM with per-cluster full covariance (shrinkage $10^{-3}$), warm-started from 10 Lloyd iterations. Means over 8–10 seeds.

### 3.1 Low dimension and small $n$: the laws extrapolate to $\alpha_0 = 1$

**Mechanism.** The laws were fit at $n=10^6$, $k=1024$, $d=96$. Evaluate them at $n=2000$, $k=10$, $T=25$:

$$
B = 0.035\cdot 10^{0.13}\cdot 2000^{0.88}\cdot 25^{0.81} \approx 516 \to 512,\qquad
\alpha_0 = \min(0.824\cdot 10^{-0.46}\cdot 512^{0.90}\cdot 2000^{-0.45}\cdot 25^{-0.11},\,1) = \min(1.80,\,1) = 1 .
$$

$\alpha_0 = 1$ means $\gamma = 0$: every batch **fully replaces** the sufficient statistics. The algorithm starts as mini-batch Lloyd on a quarter of the data with no memory, and only acquires memory as the cosine decays. The law has left its fitted range and clipped. The paper acknowledges this directly: the laws were tuned on relatively high-dimensional data and are not optimal for very low-dimensional or tiny datasets (Figure 11), and it names refining the law as its key open problem.

**Evidence** (Exp 1, $n=2000$, $k=10$, 10 seeds):

| $d$ | $B$ | $\alpha_0$ | e-means MSE | Lloyd++ MSE | ratio |
|---|---|---|---|---|---|
| 2 | 512 | 1.0 | 1.509 | 1.521 | 0.99 |
| 4 | 512 | 1.0 | 6.14 | 4.48 | **1.37** |
| 8 | 512 | 1.0 | 15.57 | 11.33 | **1.37** |

At $d=2$ it ties; at $d=4,8$ e-means converges 37% above Lloyd++. Random init is the paper's default and it argues k-means++ is too expensive at its scale — but at small $n$ k-means++ costs nothing, and Lloyd gets it for free.

**The revival artifact.** Two separate problems:

1. *Count-blind criterion.* Exp 5 addendum, $n=4000$, $k=8$, $d=16$, well-separated blobs: Lloyd++ reaches the optimum (MSE 15.90) in 6/6 seeds. e-means random-init reaches it in 1/6; the other five end at MSE 64–179 with **zero revivals fired**. Diagnosis: the failed runs have two centroids sharing one blob and one blob orphaned; the smallest cluster count is 141–218 against a mean of 500, i.e. 3–4× *above* the $0.1\overline N = 50$ threshold. The criterion detects *dead* clusters, not *redundant* ones, and the redundant case is the generic small-$n$ failure. The fix is a loss-aware criterion (e.g., revive the cluster whose removal increases loss least, seed from the point with largest residual — the "k-means‖ / global k-means" style move), not a count threshold.

2. *Multiplicative split with a fixed sign pattern.* The children are $c\odot(1\pm\delta)$ with $\delta = 2^{-8}$. Their separation is $\|c_1 - c_2\| = 2\delta\|c\| \approx 0.0078\,\|c\|$ — **proportional to the centroid's norm, not to the cluster's spread**, and oriented along the fixed direction $\mathrm{sign}$-pattern $\odot c$. Exp 6: for a centroid at the origin (centered/standardized data, or any feature that is exactly zero) the children are **identical** (separation 0.000). The split then relies on BF16 rounding noise in the *next* assignment to break symmetry. The sensible split is additive and shape-aware: $c \pm \delta\, \sigma_j \odot r$ with $r$ a random unit vector and $\sigma_j$ the per-coordinate cluster RMS — which is one diagonal second moment, i.e. the cheapest possible FIM.

**When e-means also degrades a good init.** Given k-means++ seeds (Exp 5 addendum), Lloyd holds the optimum 6/6; e-means holds it **3/6**. With $\alpha_0$ clipped at 1, the first few batches of 1024 points re-estimate all 8 centroids from a quarter of the data with no memory, which is enough to knock two centroids onto one blob. e-means at small $n$ is not merely "no better than Lloyd" — it can actively destroy information Lloyd would keep.

### 3.2 Anisotropic and correlated clusters: the metric is wrong, and e-means cannot learn it

**Mechanism.** The hard-EM / Mahalanobis objective has Fisher information w.r.t. $\mu_j$ equal to $n_j\Sigma_j^{-1}$. Natural-gradient descent in that metric is the Mahalanobis k-means M-step. Lloyd and e-means both use $\Sigma_j = I$. For a two-cluster mixture with shared covariance $\Sigma$ and separation $\Delta = \mu_1-\mu_2$, the Euclidean assignment error rate is governed by $\Delta^\top\Delta / \Delta^\top\Sigma\Delta$-type ratios while the Mahalanobis rate is governed by $\Delta^\top\Sigma^{-1}\Delta$; the gap grows with $\kappa(\Sigma)$ when $\Delta$ is not aligned with the smallest-variance eigenvector. No amount of EMA on first moments recovers this — the information is in the second moment, which e-means never stores. (The paper's hypothesis that EMA stays in the convex hull of assigned points is a statement about *location*, not *shape*; it does not help here.)

**Evidence** (Exp 2, $k=3$, $d=2$, $n=3000$, covariance eigenvalues $(\kappa, 1)$ in a random rotation, ARI vs. ground truth):

| $\kappa(\Sigma)$ | e-means | Lloyd++ | FIM / Mahalanobis |
|---|---|---|---|
| 1 | 0.68 | 0.84 | 0.86 |
| 10 | 0.56 | 0.61 | 0.63 |
| 100 | 0.12 | 0.10 | **0.47** |

At $\kappa = 100$ both Euclidean methods are near chance; the Fisher-metric method recovers 4–5× the structure. (The FIM method's own ceiling is initialization: it is warm-started from Euclidean Lloyd, so at $\kappa = 100$ it inherits a partially wrong partition; a Mahalanobis-aware init would do better still.)

The literature agrees: Melnykov & Michael and Armah & Melnykov (arXiv:2411.00870) report Mahalanobis k-means consistently beating Euclidean k-means on elliptical clusters; sparse $K$-spatial-median clustering (arXiv:2605.00598, May 2026) specifically adds a Mahalanobis-type metric from the spatial-sign covariance to handle heterogeneous scales and feature dependence; natural-gradient EM for mixtures (arXiv:2602.10602, Feb 2026) exploits the block-diagonal complete-data FIM to turn an NGD step on a Gaussian mixture into $K+1$ independent preconditioned steps. That last paper is the closest thing to "e-means with the right metric": the EMA statistics $(N_j, S_j)$ are exactly the first two of the three complete-data sufficient statistics; the third, $\sum x x^\top$, is what makes the FIM available.

### 3.3 Ill-conditioned feature scales: same mechanism, different cause

**Mechanism.** Feature rescaling by $D = \mathrm{diag}(s_1,\dots,s_d)$ maps $\Sigma \to D\Sigma D$. Euclidean k-means is not invariant to $D$; Mahalanobis k-means is exactly invariant (the metric transforms contravariantly). A single nuisance feature at scale $s \gg 1$ makes $\kappa(\Sigma) \sim s^2$, and the Voronoi cells collapse onto that axis. e-means has no standardization step, and its laws contain no scale term, because on unit-norm embeddings none is needed.

**Evidence** (Exp 3, $k=4$, $d=5$, one feature multiplied by $s$, ARI):

| $s$ | e-means | Lloyd++ | FIM / Mahalanobis |
|---|---|---|---|
| 1 | 0.86 | 0.99 | 0.82 |
| 100 | 0.40 | 0.41 | **0.91** |
| 10 000 | 0.40 | 0.41 | 0.44 |

At $s=100$ Euclidean methods lose 60% of the structure and the Fisher-metric method keeps 91%. At $s = 10^4$ even the FIM method degrades — not because the metric is wrong, but because the Euclidean warm start is wrong and the shrinkage ($10^{-6}s^2$) is no longer negligible. This is the honest limit: FIM preconditioning fixes the *metric*, not the *initialization*; standardization is still the cheap first line of defense, and both Lloyd and e-means should get it. The point for this document is that e-means offers **nothing** here that Lloyd does not, and the method that *does* help is one e-means cannot express.

### 3.4 Non-stationary streams: cosine annealing is an anti-tracking schedule

**Mechanism.** An EMA with step $\alpha$ tracking a mean drifting linearly at $v$ per step has steady-state lag $\approx v(1-\alpha)/\alpha$ and noise variance $\approx \alpha\sigma^2/(2-\alpha)$; the optimal $\alpha$ for a given $(v,\sigma)$ is fixed and positive. e-means instead drives $\alpha \to \varepsilon\alpha_0 = 0.1\alpha_0$ along a cosine, so the lag grows 10× over the run **by design**. And $\alpha_0$ itself scales like $n^{-0.45}k^{-0.46}B^{0.9}$: at $n = 10^9$ the floor is tiny. That is correct for a *stationary* dataset (it is simply variance reduction toward a fixed point) and exactly wrong for a *stream*. The paper never claims streaming capability — it processes epochs over a fixed dataset — but the Faiss contrib packaging and the "replace Lloyd" framing invite that misuse.

Sherman–Morrison with a forgetting factor $\lambda$ solves the same problem at the second-moment level with a *constant* effective window $1/(1-\lambda)$:

$$
\Sigma_t = \lambda\Sigma_{t-1} + e_t e_t^\top \;\Longrightarrow\;
P_t = \frac1\lambda\Big(P_{t-1} - \frac{P_{t-1}u u^\top P_{t-1}}{1 + u^\top P_{t-1} u}\Big),\quad u = e_t/\sqrt\lambda,
$$

which is $O(d^2)$ per sample, exact, and never calls an inverse. This is the recursive-least-squares / Kalman gain, and it is what online Mahalanobis anomaly detectors and the River library's online precision matrix do.

**Evidence** (Exp 4, $d=4$, mean drifting 20 units over 20 000 samples, unit noise, tracking error measured over the second half):

| tracker | mean $\|\hat\mu - \mu_t\|$ |
|---|---|
| e-means cosine schedule ($\alpha_0=0.5 \to 0.05$) | 0.496 |
| constant EMA $\alpha = 0.05$ | 0.310 |
| SM precision tracker, $\lambda = 0.995$ | **0.222** |

Note the constant-$\alpha$ EMA already beats the cosine schedule by 37% — the schedule, not the EMA, is the problem — and the SM tracker beats both because its forgetting is applied to the full sufficient statistic, which also gives it a Mahalanobis radius for free. The 2025–26 stream-clustering literature (SDOstreamclust, IJCNN 2025; ICD3 drift detection, arXiv:2603.06757; the streaming continual learning guide, arXiv:2603.01677) treats drift adaptation as a first-class design requirement; an annealed-to-floor EMA is the opposite design.

### 3.5 Exactness and determinism: what e-means trades away

**Lloyd** given an initialization is a deterministic map: Exp 5, two runs from identical seeds, max centroid difference **0.0**. Each iteration is monotone non-increasing in the loss, and the loss is a function of the partition alone, so convergence in finitely many steps is a theorem.

**e-means** given the same data and the same initial centroids depends on (i) the permutation of batches, (ii) the revival RNG, (iii) BF16 rounding in assignment. Exp 5: six batch orders on one dataset give final MSE $\{121.7, 64.3, 179.5, 15.9, 121.5, 92.8\}$, a relative spread of 165%. The paper's own precision tables (Table 17) show BF16 flips 0.29–3.98% of final assignments relative to double precision, and it notes that rounding can reorder nearly-tied distances. Its only monotone step is the final last-epoch-mean, which *is* a Lloyd M-step.

**Sherman–Morrison** is an algebraic identity, not an approximation. Exp 7, $d=32$: single-update relative error vs. direct inverse $2.9\times10^{-16}$; after 10 000 consecutive rank-1 updates with no refactorization, $6.1\times10^{-15}$, symmetry preserved to $4\times10^{-18}$. The only failure condition is $1 + u^\top P u \to 0$, which for a positive-definite $P$ and an *additive* update cannot happen (the denominator is $\ge 1$); it is a concern only for *downdates* (removing a sample), where the denominator is $1 - u^\top P u$ and can cross zero if the sample being removed carried the entire information in some direction. A recent technical note (arXiv:2603.16697, Mar 2026) gives explicit cost crossover rules between direct inversion, iterated SM, and Woodbury as a function of $d$ and update rank $k$; Fast IGMN (Pinto & Engel, PLOS ONE 2015 / arXiv:1701.03940) shows the same rank-1 precision update reduces incremental full-covariance GMM learning from $O(NKD^3)$ to $O(NKD^2)$ with results identical to the covariance form.

When a workflow needs *bitwise reproducibility* (regulatory, audit, A/B-test attribution, codebook pinning for a deployed quantizer), e-means has to be run with pinned seeds, pinned batch order, pinned precision, and pinned kernel reduction order, and even then the result is a function of all of those rather than of the data. Lloyd's result is a function of the data and the initialization.

### 3.6 An inherited failure the paper does not discuss: the high-dimensional fixed-point pathology

Lederman, Silva-Sánchez, Chen, Mordant, Balanov, Bendory (ICML 2026, arXiv:2602.09936) prove that for a two-component isotropic GMM with $\sigma^2 > n$ and $d \gtrsim n^3$, essentially *every* balanced partition is a fixed point of Lloyd's update map, so Lloyd returns its initialization even when the clusters are trivially recoverable. The cause is **self-influence**: a sample's own presence pulls its centroid toward it by $\sim 1/|C_j|$, which dominates the weak class signal. Hartigan's update removes the bias with the weights $n_j/(n_j-1)$ (own cluster) and $n_j/(n_j+1)$ (other cluster), and provably has no incorrect fixed points when $d \gtrsim n\sigma^4$.

The theorem is stated for Lloyd and does not analyze mini-batch variants. But the mechanism transfers by inspection: in e-means the point $x$ contributes to $S_{a(x)}$ with weight $\alpha$ and is then compared against $S_{a(x)}/N_{a(x)}$, so the self-influence term is $\alpha/N_j' \cdot$ (its own residual) — the same bias, scaled by the current effective step. Nothing in the EMA, the cosine schedule, or the revival cancels it. Any Lloyd fixed point is an e-means fixed point in the $\alpha\to$ floor limit. The Hartigan correction, by contrast, is a one-line change to the assignment rule, costs nothing, and is *compatible* with EMA statistics: compare $\frac{N_j}{N_j - w}\|x - c_j\|^2$ against $\frac{N_{j'}}{N_{j'}+w}\|x-c_{j'}\|^2$ with $w$ the sample's EMA weight. e-means does not do this.

---

## 4. Decision matrix

Read top to bottom; take the first row that applies.

| If… | then use… | because… |
|---|---|---|
| you need bitwise-reproducible or auditable output, or a monotone-descent guarantee | **Lloyd** (k-means++ init, or Hartigan if $d \gg n$) | deterministic given init; loss is a function of the partition; e-means is a function of batch order and precision |
| $n \lesssim 10^6$ and a pass over the data is cheap | **Lloyd++** | its guarantees are free at this scale; e-means' laws extrapolate to $\alpha_0 = 1$ and it reaches the optimum in 1/6 seeds where Lloyd++ reaches it in 6/6 (Exp 5) |
| $d \lesssim 16$ | **Lloyd++** | the laws have no $d$ term and were fit at $d \ge 96$; e-means is 37% worse at $d=4,8$ (Exp 1) |
| clusters are elliptical / correlated, or features are on different scales and you cannot or will not standardize | **FIM-preconditioned hard-EM** (full $\Sigma_j$ if $d \lesssim 10^3$, diagonal or low-rank otherwise) | the information is in the second moment, which neither Lloyd nor e-means stores; ARI 0.47 vs 0.12 at $\kappa = 100$ (Exp 2), 0.91 vs 0.40 at scale 100 (Exp 3) |
| data arrive as a stream, the distribution drifts, or you need per-cluster Mahalanobis scores online (anomaly, novelty) | **Sherman–Morrison precision per cluster** with forgetting $\lambda$ | constant effective window; exact to $10^{-15}$ over $10^4$ updates; no inversion; cosine-annealed EMA lags by 2.2× (Exp 4) |
| $d \gtrsim n^3$-ish, high-noise (single-particle imaging, some genomics) | **Hartigan**, not Lloyd and not e-means | both inherit the self-influence fixed-point pathology (§3.6) |
| $n \gtrsim 10^7$, $d \sim 10^2$–$10^3$, roughly isotropic embedding vectors, GPU available, IVF/PQ codebook is the goal, $k \approx 4\sqrt n$–$32\sqrt n$ | **e-means** | this is the regime the laws were fit in and the speedups are 10–45× at matched loss |
| the above, but you also want shape awareness | **e-means + diagonal second-moment EMA** (see §5) | $2\times$ memory, same cost order, restores scale invariance and makes revival shape-aware |

---

## 5. A holistic reframing, and what a better algorithm looks like

Strip away the branding and the four methods are points on a two-axis grid: *which sufficient statistics do you carry* (count; count+sum; count+sum+scatter) and *how do you forget* (never; by epoch; by constant rate; by adaptive rate).

```
                  forget: never      by epoch (batch)     constant rate       adaptive rate
count + sum       MacQueen/Sculley   Lloyd                e-means             (gap)
count+sum+scatter —                  Mahalanobis k-means  SM precision w/ λ   SM w/ adaptive λ
                                     / natural-grad EM                        (gap)
```

e-means filled the "count+sum, constant rate" cell well and then annealed itself toward the "by epoch" cell for variance reduction. The cells it cannot reach are the entire second row — and that row is where anisotropy, scale, drift, and shape-aware revival live. The cheapest move down a row is the diagonal: carry $Q_j \leftarrow \gamma Q_j + (1-\gamma)\sum_{x\in B_j} x\odot x$, which costs one extra $d$-vector per cluster, gives $\hat\sigma_j^2 = Q_j/N_j - c_j\odot c_j$, and immediately enables:

1. **Scale-invariant assignment** $\arg\min_j \sum_\ell (x_\ell - c_{j\ell})^2/\hat\sigma^2_{j\ell} + \sum_\ell\log\hat\sigma_{j\ell}^2$ — a diagonal Fisher metric. Fixes §3.3 outright and most of §3.2.
2. **Shape-aware revival** $c \pm \delta\,\hat\sigma_j\odot r$ — fixes the zero-coordinate and fixed-direction artifacts of §3.1.
3. **A drift signal**: if the batch residual energy $\sum_{x\in B_j}\|x-c_j\|^2 / n_j$ runs persistently above $\mathrm{tr}\,\hat\sigma_j^2$, the cluster is being dragged; raise $\alpha_j$ locally instead of following the global cosine. This is the "adaptive rate" column, per cluster, for free.

The full move down a row — per-cluster $\Sigma_j^{-1}$ via Sherman–Morrison on each batch — costs $O(k d^2)$ memory and $O(B d^2)$ per batch. At $d = 96$, $k = 10^6$ that is $9\times10^9$ floats, which is why e-means does not do it and should not. At $d = 96$, $k = 10^3$ it is $9\times10^6$ floats — trivial — and that is the regime where most non-embedding clustering actually happens. A low-rank variant ($\Sigma_j^{-1} \approx \sigma^{-2}(I - U_jU_j^\top)$ with $U_j \in \mathbb R^{d\times r}$ updated by rank-1 eigen-updates à la ROIPCA) interpolates at $O(kdr)$.

Two further corrections cost nothing and should be in any successor:

- **Hartigan weights in assignment** (§3.6), to remove self-influence bias at high $d/n$.
- **A loss-aware revival criterion** (§3.1), to catch redundant centroids rather than only empty ones.

None of this is exotic; natural-gradient EM for mixtures (Feb 2026) already factorizes the complete-data FIM block-diagonally across components, which is precisely the structure an e-means-style per-cluster EMA would exploit. The research gap is not the metric — it is fitting *scaling laws* for a method that carries second moments, since the current laws are tied to the count+sum state and to $d \ge 96$.

---

## 6. Reproduction

```
python3 -I experiments.py     # writes results.json; ~3 min CPU, numpy only
python3 -I addendum.py        # Exp 5 fairness controls (Lloyd++ / e-means with k-means++ seeds)
```

`experiments.py` contains: `emeans()` (EMA on $N,S$; cosine schedule Eq. 5; laws Eq. 6–7; BF16-$\delta$ donor split revival every 5 batches at $\tau = 0.1$; last-epoch-mean finalization), `lloyd()` with k-means++, `mahalanobis_kmeans()` (hard-EM, full per-cluster covariance, shrinkage), `SMPrecision` (forgetting-factor Sherman–Morrison tracker). All experiments are small by design: the goal is to probe the boundary of e-means' regime, not to replicate its headline results, which require 1M–1B vectors and a GPU and which this document does not dispute.

What these experiments do **not** show, and should not be read as showing: e-means' behavior at $n \ge 10^6$ (not tested here; the paper's Tables 4–5 and 11 stand); whether a re-fitted law with a $d$ term would close the low-$d$ gap (plausible; not tested); FIM/Mahalanobis performance with a Mahalanobis-aware initialization (would be higher than reported here; the Euclidean warm start is the binding constraint at $\kappa=100$ and $s=10^4$).

---

## 7. Three independent review passes

Each pass was asked to find the weakest point of the analysis above and say whether it survives.

**Pass 1 — "You are testing e-means outside its stated regime; of course it loses."**
Correct, and intended. The user-facing claim being examined is "a replacement of Lloyd's algorithm" (Faiss README, project page). A replacement must be safe across the domain of the thing it replaces. The paper is careful (Figure 11, the dropped $d$ exponent, the epoch cap); the packaging is not. §4 marks the regime where e-means is the right choice unambiguously. Survives.

**Pass 2 — "The FIM method is doing more work; compare at matched compute."**
Partially fair. At $d=2$–$5$ the FIM method's extra cost is negligible, so the comparison is at essentially matched compute. At $d = 10^3$ it is not, and the document says so (§5: $O(kd^2)$ memory, diagonal/low-rank alternatives). The stronger objection is initialization: the FIM method is warm-started from Euclidean Lloyd and so inherits Euclidean mistakes at extreme $\kappa$ and $s$; the numbers in §3.2–3.3 are therefore a *lower bound* on the FIM advantage. Survives, with the lower-bound caveat now stated.

**Pass 3 — "§3.6 extends a Lloyd theorem to e-means without proof."**
Correct. The Lederman et al. result is for full-batch Lloyd, $K=2$, and the authors call their bounds conservative. The self-influence argument for e-means is by inspection of the update, not a theorem, and the document labels it as such. What can be said rigorously: at the cosine floor with $\alpha \to 0$ the e-means fixed points coincide with Lloyd's (the EMA converges to the plain mean of the assigned points over a long window), so any partition that is a Lloyd fixed point is asymptotically an e-means fixed point. Whether the stochasticity of mini-batching *escapes* those fixed points at finite $\alpha$ is an open question and would be a good follow-up paper. Survives as a flagged inference, not as a result.

One correction the passes forced: an earlier draft stated that the SM denominator "can vanish." For additive updates with positive-definite $P$ it cannot ($1 + u^\top P u \ge 1$); the risk is confined to downdates. §3.5 now says this precisely.

---

## 8. Sources

Primary subject
- Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou. *e-means: outperforming k-means (in one-tenth the epochs).* arXiv:2611.09501, 6 Nov 2026. https://arxiv.org/abs/2611.09501 — project page https://e-means.github.io — code `faiss/contrib/e_means`.

Theory of stochastic / online k-means
- Tang, Monteleoni. *Convergence rate of stochastic k-means.* arXiv:1611.05132 (AISTATS 2017).
- Dasgupta, Mahajan, So. *Convergence of online k-means.* arXiv:2202.10640 (AISTATS 2022).
- Lederman, Silva-Sánchez, Chen, Mordant, Balanov, Bendory. *The Catastrophic Failure of the k-Means Algorithm in High Dimensions, and How Hartigan's Algorithm Avoids It.* arXiv:2602.09936 (ICML 2026).
- Łańcucki et al. *Robust Training of Vector Quantized Bottleneck Models.* arXiv:2005.08520 (IJCNN 2020) — EMA/SGD equivalence, codeword re-initialization.

Fisher metric / natural gradient for mixtures
- Chen, Bayrooti, Morad. *Learning Mixture Density via Natural Gradient Expectation Maximization.* arXiv:2602.10602 (Feb 2026).
- Tan. *Natural gradient updates for Cholesky factor in Gaussian and structured variational inference.* arXiv:2109.00375 (v10, Feb 2025).
- *Convergence properties of natural gradient descent for minimizing KL divergence.* TMLR 2025 (condition number of the FIM-Hessian at the optimum).
- Armah, Melnykov. *K-Means Clustering With Incomplete Data with the Use of Mahalanobis Distances.* arXiv:2411.00870 (v2, Apr 2025).
- Zhao, Zhuang, Feng. *Sparse K-spatial-median clustering for high-dimensional data.* arXiv:2605.00598 (May 2026).

Sherman–Morrison / online precision
- Pinto, Engel. *A Fast Incremental Gaussian Mixture Model.* PLOS ONE 2015, arXiv:1506.04422; *Scalable and Incremental Learning of GMMs.* arXiv:1701.03940.
- *Rank-k inverse update strategies for streaming Christoffel-function outlier detection (DI vs iterated SM vs Woodbury).* arXiv:2603.16697 (Mar 2026).
- Giovannucci, Minden, Pehlevan, Chklovskii. *Efficient Principal Subspace Projection of Streaming Data Through Fast Similarity Matching.* arXiv:1808.02083.
- Mitz, Shkolnisky. *ROIPCA: an online PCA algorithm based on rank-one updates.* arXiv:1911.11049.
- *New Robust Streaming DMD with Forecasting.* arXiv:2604.02928 (SM updates of Gram inverses).

Streaming / drift
- Iglesias, Konzett, Zseby, Bifet. *Stream Clustering Robust to Concept Drift (SDOstreamclust).* IJCNN 2025.
- Zhang et al. *Learning Unbiased Cluster Descriptors for Interpretable Imbalanced Concept Drift Detection.* arXiv:2603.06757 (Mar 2026).
- Cossu et al. *A Practical Guide to Streaming Continual Learning.* arXiv:2603.01677, Neurocomputing 2026.
- Bhanderi, Bhatnagar. *Incremental Gaussian Mixture Clustering for Data Streams.* arXiv:2412.07217 (ICDMW 2024).
