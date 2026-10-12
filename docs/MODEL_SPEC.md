# Model specification

This is a description of what `research/model.py` computes at commit `fc98f02`, read from the code.
Where the code and its docstrings disagree, the code is recorded here and the disagreement is flagged.

## Symbols

| symbol | code name | meaning |
|---|---|---|
| `theta_i` | `theta` | phase of node i |
| `omega_i` | `Params.omega` | natural frequency |
| `lam` | `Params.lam` | herding coupling (float or function of time) |
| `K` | n/a | plain Kuramoto coupling (legacy layer and reproductions only) |
| `p` | `Params.herd_power` | adaptation exponent |
| `R` | `Trajectory.R` | global order parameter `abs(mean(exp(i theta)))` |
| `r_i` | `r_loc` | local order parameter `abs(sum_j A_ij exp(i theta_j)) / k_i` |
| `k_i`, `<k>` | `Network.deg`, `Network.mean_deg` | degree, realised mean degree |
| `gamma` | `GAMMA` in S11, S13, S14 | Lorentzian half-width |
| `sigma` | `Params.sigma` | idiosyncratic phase noise amplitude |
| `sigma_c` | `Params.sigma_common` | common phase noise amplitude |
| `f` | fraction of `Params.adaptive` | fraction of adaptive nodes |

## Equation of motion

```
d theta_i = [ omega_i
              + lam(t) * scale_i * alpha_i * sum_j A_ij sin(theta_j - theta_i - alpha_lag)
              + eps(t) * s_i * sin(theta_target - theta_i) ] dt
            + sigma dW_i + sigma_c dW_c

alpha_i = r_i^p   for adaptive nodes,   1 otherwise
scale_i = 1/<k>   (norm="mean", default)
        = 1       (norm="none", Gomez-Gardenes 2011)
        = 1/k_i   (norm="degree", the legacy layer's form)
```

`r_i` is the local order parameter, not the global `R`. With `norm="mean"` the locking field on node i
is `(lam/<k>) k_i r_i^(p+1)`.

Optional terms, all off by default: `mass` (second-order dynamics), `csign` (contrarians), `alpha_lag`
(Sakaguchi lag), `delay`, `net2`/`lam2` (second layer), `rewire_rate` (co-evolving graph), `asym`
(gain multiplied by `1 - asym sin theta_i`), `active` (halt mask), `shock_eps`/`shock_mask` (common
sell signal toward `theta_target = -pi/2`).

## Integrator

Stochastic Heun with fixed `dt`. One noise draw per step is used in both predictor and corrector.
Deterministic order 2 when `sigma = 0`. `simulate` returns phases reduced mod `2 pi`.

## Networks (`make_network`)

`er`, `ba`, `sf` (configuration model, structural cutoff `sqrt(<k> N)`, multi-edges collapsed),
`ws`, `ring`, `rr`, `complete`. Only the giant component is kept, so the realised `n` and `<k>` can
differ from nominal. For `rr` the degree is rounded to an integer and bumped by one if `k n` is odd.

## All-to-all limit (published, not new)

With all-to-all coupling and Lorentzian `g(omega)` of half-width `gamma`:

```
lam R^(p+2) - lam R^p + 2 gamma = 0
R*(p)       = sqrt(p / (p+2))
lam_b_inf(p) = gamma (p+2) ((p+2)/p)^(p/2)
```

This is Zou and Wang, Phys. Rev. E 102, 012219 (2020) with `alpha = p + 1`. Incoherence is linearly
stable for every `lam` in this limit, so the forward threshold is fluctuation-triggered.

## S14 Rician closure (an approximation)

`S_i/k_i` is treated as complex Gaussian with mean `(R, 0)` and per-component variance
`(1 - R^2)/(2k)`; node i locks to the direction of its own neighbour sum, giving a projection factor
`X/r`:

```
R = < c((lam k/<k>) r^(p+1)) X/r >,   c(h) = (sqrt(h^2 + gamma^2) - gamma)/h
lam_b(p,k) = lam_b_inf(p) [1 + (p+2)/(2k) + O(k^-2)]      (expansion of the closure)
```

The closure assumes neighbour phases are independent of node i and of each other given `R`. It has no
back-reaction (cavity) term. `(p+2)/2` is exact for the closure, not for the network model.

## Known discrepancies between code and documentation

1. Module docstring writes `alpha_i = r_i`; the code uses `r_i^herd_power`.
2. `adiabatic_sweep` overwrites `p.lam` on the caller's `Params`.
3. Second-order branch (`mass > 0`) adds the noise increment to the velocity without `1/mass`, so
   `sigma` is not the same physical quantity as in the first-order model.
4. The `net2` layer always uses exponent 1 and `1/<k_2>` scaling, ignoring `herd_power` and `norm`.
5. `rewire_toward_agreement` sets `deg = 1` on isolated nodes and lets `<k>` drift; `scale` follows.
6. Three different `lorentz_omegas`: S11 clips at 60 and centres; S13 and S14 clip at 60 and do not
   centre. Quantile sampling puts the outermost values at the clip.
7. Gaussian frequencies in S2 to S10, S12, A3, E5; Lorentzian in S11, S13, S14. The closed form above
   covers only the Lorentzian scripts.
8. `measure_lam_b` is described as level-free but uses `max(4 * 0.886/sqrt(N), 0.55 R*)` as the
   survival threshold.
