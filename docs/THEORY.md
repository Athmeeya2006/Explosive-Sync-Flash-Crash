# Leading 1/k theory of the fold: derivation

Status: derived on 2026-10-13 in response to the audit. On directed graphs the prediction is confirmed
by the branch-shift test of section 6 and the Rician closure is excluded. On undirected graphs the
reaction term has the predicted sign and roughly the predicted size, but the prediction is **not
confirmed**: too few seeds, and disagreement at small and at very large k. Measured values are in
`results/` and on `research/index.html`; none are quoted here. Code: `src/esfc/cavity.py`. Units:
`gamma = 1`.

## 1. Setting

Random k-regular graph, all nodes adaptive with exponent p, no noise, Lorentzian frequencies:

```
theta_i' = omega_i + lam r_i^(p+1) sin(phi_i - theta_i) = omega_i + Im[ H_i e^{-i theta_i} ]
F_i = (1/k) sum_j e^{i theta_j} = r_i e^{i phi_i},        H_i = lam |F_i|^p F_i
```

All-to-all (`F_i = R`): `R = z0(h)`, `z0(h) = (sqrt(1 + h^2) - 1)/h`, `h = lam R^(p+1)`, with the fold
at `R* = sqrt(p/(p+2))`, `lam_b_inf = (p+2)((p+2)/p)^(p/2)` (Zou and Wang 2020).

Write `F_i = R + xi_i`. On a k-regular graph `xi_i` has per-component variance `(1 - R^2)/(2k)`. The
goal is the coefficient `a(p)` in `lam_b = lam_b_inf (1 + a(p)/k + O(k^-2))`.

## 2. What the Rician closure assumes

1. **Isotropy** of the scatter of a neighbour phasor about `(R, 0)`.
2. **Frozen scatter:** `xi_i` is a time-independent random offset.
3. **Independence:** `xi_i` is independent of node i's own frequency and phase.

Assumption 1 is exact for Lorentzian frequencies. On the Ott-Antonsen manifold `<z^2> = <z>^2`, so
`<cos 2 theta> = R^2`, hence `<cos^2> - R^2 = <sin^2>`.

Assumptions 2 and 3 are both wrong at order 1/k, in opposite directions.

## 3. Independent neighbours with the correct dynamics (directed graphs)

Keep assumption 3, drop assumption 2. Because `omega_i` is Lorentzian and independent of `xi_i(t)`,
the average over `omega_i` at fixed `xi_i(t)` obeys the Ott-Antonsen equation **exactly**, for any time
dependence of the drive:

```
z' = -z + (1/2) (H - H* z^2),        H(t) = lam |R + xi(t)|^p (R + xi(t)),       R = E_xi < Re z >_t
```

So the many-body problem reduces to one complex ODE driven by a stationary random field. Expand to
second order in `xi = u + i v` about the fixed point `z0 = R`. With `L = lam R^p`, `mu = sqrt(1 + h^2)`,
`alpha = (1 - R^2)/2`, `beta = (1 + R^2)/2`:

```
H = h + A + iB,   A = L (p+1) u + lam R^(p-1) [ p(p+1)/2 u^2 + p/2 v^2 ],   B = L v + lam R^(p-1) p u v
first order:   a1' = -mu a1 + alpha A1,     b1' = -mu b1 + beta B1          (both relax at rate mu)
second order:  <a2> = (1/(2 mu)) [ (1 - R^2) <A2> - 2R <A1 a1> - 2R <B1 b1> - h (<a1^2> - <b1^2>) ]
```

`<A2>` involves the **instantaneous** variance of `xi`. Every other term involves `xi` through the
linear response, i.e. through `int_0^inf e^{-mu s} C(s) ds` with `C` the autocorrelation of `xi`. A
frozen component contributes `C/mu`; a component rotating at frequency W contributes
`C mu/(mu^2 + W^2)`. So a spectral line at W enters the response terms with weight

```
mu^2 / (mu^2 + W^2)
```

**Spectrum of `xi`.** A locked neighbour (`|x| <= 1`, `x = omega/h`) is frozen at
`sqrt(1 - x^2) + i x`. A drifting neighbour (`x > 1`) has, with `q = x - sqrt(x^2 - 1)` and
`W = h sqrt(x^2 - 1)`,

```
e^{i theta(t)} = i q + (1 - q^2) sum_{n >= 1} (i q)^(n-1) e^{i n W t}
```

so its frozen part is `i q` and its line at `n W` has power `(1 - q^2)^2 q^(2(n-1))`. Both components
`u` and `v` receive half of each line, and their cross-correlation cancels between `+x` and `-x`.

Setting `R = z0(lam R^(p+1)) + <a2>` and using that `d lam_0/dR = 0` at the fold,

```
a(p) = - k <a2> / (lam_b_inf  d z0/d lam)       evaluated at (lam_b_inf, R*)
```

- If every line is given weight 1 (all scatter frozen) this returns `(p+2)/2` **exactly**, which
  reproduces the Rician closure and checks the algebra (`tests/test_cavity.py`).
- With the correct weights it gives `a_dyn(p)`, smaller than `(p+2)/2`.

Only second moments of `xi` enter at this order, so no Gaussian assumption is needed.

**Where it is exact.** On a directed graph in which each node listens to k random others, a node's
neighbours do not listen to it, so assumption 3 holds as k and N grow. `a_dyn(p)` is therefore the
prediction for directed graphs (`graphs.directed_regular`).

## 4. The reaction term (undirected graphs)

On an undirected graph node i appears in each neighbour's field. Relative to a typical member of the
population, neighbour j's field is shifted by `(z_i - R)/k` (the mean `R/k` is already in the mean
field; using `z_i/k` would double count it). Neighbours respond with the linearised dynamics of
section 3: transverse response `beta L/mu = 1` exactly (rotating the field rotates the response), and
longitudinal response `L (p+1) dz0/dh`. The k responses, each of order 1/k, sum to a field on i of
order 1/k that is correlated with `theta_i`:

```
delta F_i = (1/k) [ L (p+1) (dz0/dh) (cos theta_i - R) + i sin theta_i ]     (locked i; filtered at rate mu for drifting i)
```

Inserting this into node i's own equation and averaging `cos theta_i` over the Lorentzian gives the
shift `Delta/k` of the self-consistency, and `a_react(p) = - Delta / (lam_b_inf dz0/dlam)`. For locked
nodes this is a closed-form integral; for drifting nodes the reaction oscillates and is integrated
numerically. The transverse part is a self-coupling `(L/k) sin theta cos theta` that weakens the
restoring force, so `a_react > 0`: reaction makes the synchronised state harder to hold.

Prediction: `a = a_dyn(p) + a_react(p)` on undirected tree-like graphs.

## 5. What this does and does not establish

- The three treatments differ most on **directed** graphs, where the closure predicts `(p+2)/2` and
  this theory predicts the much smaller `a_dyn(p)`. That is the cleanest test.
- On undirected graphs the two corrections partly cancel, so the total can land close to the
  closure's value for a different reason.
- The expansion parameter is not small at low connectivity: the self-coupling is `L/(h k)` relative
  to the restoring force, which is of order `1.7/k` at p = 1. Second-order terms are therefore
  expected to matter for k below a few tens. Agreement must be tested by the trend in k, not at one k.
- Not included: loops (random regular graphs are tree-like only locally), degree heterogeneity,
  clustering, and the quenched sample-to-sample fluctuation of the fold at finite N.
- Whether bistability disappears below a critical connectivity is **not** addressed by a leading-order
  expansion.

## 6. The same expansion on the synchronised branch

Sections 3 and 4 evaluate the second-order term at the all-to-all fold. Nothing in the expansion
requires that. At any coupling `lam > lam_b_inf` let `R0` be the stable all-to-all solution,
`R0 = z0(lam R0^(p+1))`, and `h = lam R0^(p+1)`. At finite k

```
R = z0(lam R^(p+1)) + <a2>/k + O(k^-3/2)
```

with `<a2>` the same second-order average as in section 3, evaluated with the branch constants
`(R0, h, mu = sqrt(1 + h^2))` in place of the fold constants. Linearising in `R - R0`:

```
k (R - R0)  ->  c(lam) = <a2> / (1 - z0'(h) lam (p+1) R0^p),      z0'(h) = (mu - 1)/(mu h^2)
```

The denominator vanishes at the fold, where `c` diverges and the shift of `R` turns into a shift of
`lam_b`; that is the limit sections 3 and 4 treat. Away from the fold `c` is finite and can be measured
as a difference of two stationary order parameters.

The three treatments enter through the variance used for the response term of `<a2>`:

| treatment | variance of the neighbour scatter acting on node i | graphs |
|---|---|---|
| Rician closure | instantaneous, `(1 - R0^2)/2` | none exactly |
| dynamic | frozen part plus rotating part filtered by `mu^2/(mu^2 + (n W)^2)` per harmonic | directed |
| dynamic plus reaction | as dynamic, plus the reaction integral of section 4 at the branch point | undirected |

Code: `experiments/branch_theory.py` (closure and dynamic variants; its closure variant is checked
against `closure.self_consistency` at large k) and `experiments/react_branch.py` (reaction integral
with branch constants substituted into `cavity.reaction_parts`). Measurement:
`experiments/branch_sim.py`.

Two conditions for comparing with simulation:

1. **Neighbourhood overlap.** Independence of the neighbours requires `k^2/N` well below 1.
2. **Reference.** `R0` must be measured on the same clipped, finite frequency sample, because an
   offset in `R0` is multiplied by k.


## 7. Degree heterogeneity: the degree-based mean field

Sections 3 and 4 treat graphs in which every node has the same degree. With a degree distribution P(k), the
simplest treatment that keeps only the heterogeneity is the degree-based (annealed) mean field: neighbours are
drawn in proportion to their degree, so every node sees the same local order parameter r, and a node of degree k
feels the field `lam (k/<k>) r^(p+1)` (gain r^p, coupling scaled by 1/<k>, k neighbours). With x = k/<k>:

```
r = E[ x c(lam x r^(p+1)) ],     c(h) = (sqrt(h^2 + gamma^2) - gamma) / h      (exact Lorentzian response)
```

For a single degree this is the all-to-all self-consistency, with the fold lam_b(inf). Expanding to second order
in x - 1 (E[x - 1] = 0, E[(x - 1)^2] = CV^2, the squared coefficient of variation of the degree) and shifting
the fold at fixed r (the fold is the minimum of lam(r), so its first-order shift is -F_v / F_lam there):

```
lam_b / lam_b(inf) = 1 - (p + 2) / (2 (p + 1)^2) * CV^2 + O(CV^3)
```

Derivation. Write u = lam r^(p+1) (the field at x = 1) and g(x) = c(u x), so the right-hand side is
E[(1 + eps) g(1 + eps)] = c(u) + CV^2 (u c'(u) + u^2 c''(u) / 2) + O(eps^3), eps = x - 1. The all-to-all fold
has c(u) = r and lam (p+1) r^p c'(u) = 1, i.e. u c'(u) = r / (p+1), with r = sqrt(p/(p+2)) and
u = 2 gamma r / (1 - r^2). The derivative of the right-hand side with respect to lam at fixed r is
(u/lam) c'(u), so

```
(lam_b - lam_b(inf)) / lam_b(inf) = - CV^2 (u c' + u^2 c''/2) / (u c') = - CV^2 (1 + u c''(u) / (2 c'(u)))
```

and with sqrt(u^2 + gamma^2) = gamma + r u (from c(u) = r) this simplifies exactly to -(p+2)/(2(p+1)^2),
independent of gamma (symbolic check with sympy). The numerical solution of the full self-consistency
(`esfc.hetero`, `results/paperA_theory_hetero/`) agrees to 1e-7 for p = 0.25 to 6, and the next order shows up
as expected for broader distributions (uniform degrees with CV^2 = 0.085: -0.42 instead of -0.375 at p = 1).
For Poisson degrees CV^2 = 1/<k>, so heterogeneity
enters at the same order as the finite-connectivity terms, with coefficient a_het = -3/8 at p = 1 and -2/9 at
p = 2. Heterogeneity LOWERS the fold: high-degree nodes are coupled more strongly (k_i/<k>) and hold the
synchronised state.

Test (`results/paperA_structure_gpu/`, N = 10000): random-regular coefficient plus a_het(k) against the measured
Erdos-Renyi coefficient: k = 24, p = 1: 0.82 against 0.79 [0.71, 0.87]; p = 2: 1.64 against 1.63 [1.46, 1.80];
at k = 12 the prediction is too high (0.66 against 0.34; 1.40 against 1.10), as expected if terms of order 1/k^2
(low-degree nodes, and the interaction of heterogeneity with the finite-k terms) matter there.
`configs/paperA_hetero_gpu.yaml` measures k = 24, 48, 96 to test whether the two contributions add at leading order.
