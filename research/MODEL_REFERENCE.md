# Model reference: equations, assumptions, and where to go next

Complete specification of the herding-oscillator market model in this repository, every assumption it
rests on, what should be removed from it, and what should be added.

Status of the numbers quoted here:

| source | state | reliability |
|---|---|---|
| S1-S7, E1-E2, A1-A2 | complete, in `research/index.html` | as published there |
| S8 factorial (144 runs) | complete | `dR_jump` usable; **`d_lam` unreliable** (see §7.1) |
| S8c corrected correlation sweep | **complete** | see §7.3 |
| S9 observable mapping | **complete** | see §7.4 |
| S10 extension testing | **complete** | see §7.5 |
| S8b theory check | ran, **inconclusive** | see §7.6 |
| E3 Lee et al. reproduction | **complete** | see §7.7 |
| E4 intraday | **complete**, data-limited | see §7.8 |
| A3 lambda estimation | **complete, negative** | see §7.9 |
| A2 signal propagation | **does not finish** | see §7.10 |
| E5 real market network | **complete, and decisive** | see §7.11 |
| S11 exact thresholds + finite size | **complete** | see §7.12 |
| E6 2010 flash crash, 1-minute | **complete** | see §7.13 |
| E7 minute event study 2002-2026 | **complete, and decisive** | see §7.14 |
| S12 structural extensions (A11-A13) | **complete** | see §7.15 |
| E8 out-of-sample prediction | **complete, and final** | see §7.16 |
| **S13 two new hypotheses** | **complete, both confirmed** | see §7.17 |

---

## 1. What the model is

### 1.1 State

Node `i` is **one trading desk / strategy**, not one stock. Its entire state is a phase
`θ_i ∈ [0, 2π)`, its position in a buy-sell cycle. Net order flow is the projection

```
q_i = sin θ_i                    +1 = max buying, −1 = max selling
Q(t) = (1/N) Σ_i sin θ_i         aggregate imbalance, moves the price
```

`θ = +π/2` is peak buying, `θ = −π/2` is peak selling (`SELL = -np.pi/2` in `model.py`).

### 1.2 The stochastic differential equation

Exactly as integrated in `research/model.py`:

```
dθ_i = [ ω_i
         + (λ(t) / norm_i) · α_i · Σ_j A_ij sin(θ_j − θ_i)
         + ε(t) · s_i · sin(θ_tgt − θ_i) ] dt
       + σ dW_i
       + σ_c dW_c
```

| symbol | meaning | values used |
|---|---|---|
| `ω_i` | natural frequency = inverse trading horizon | standardised, `⟨ω⟩ = 0` |
| `λ(t)` | herding / coupling strength, the control parameter | ramped 0 → 8 |
| `norm_i` | coupling normalisation | `⟨k⟩` (default) \| `1` (GG2011) \| `k_i` (repo's old form, broken) |
| `α_i` | herding switch | `r_i` for a fraction `f` of desks, else `1` |
| `r_i` | local agreement `\|Σ_j A_ij e^{iθ_j}\| / k_i` | 0 = neighbours split, 1 = unanimous |
| `ε(t), s_i` | common news shock and its target mask | `θ_tgt = −π/2` (sell) |
| `σ` | idiosyncratic news, independent per desk | 0 - 0.8 |
| `σ_c` | **common market mode (added for S9)**, one shared increment for every node | 0 - 0.8 |

`f = 0` with `ω_i = k_i` and `norm_i = 1` is Gómez-Gardeñes et al., PRL 106, 128701 (2011).
`f > 0` with random `ω` is Zhang, Boccaletti, Guan, Liu, PRL 114, 038701 (2015).

### 1.3 Coupling as actually computed

```
c_i = Σ_j A_ij cos θ_j          s_i = Σ_j A_ij sin θ_j
Σ_j A_ij sin(θ_j − θ_i) = s_i cos θ_i − c_i sin θ_i
r_i = √(c_i² + s_i²) / k_i
```

Both sums are `np.bincount` over the directed edge list: O(E) per evaluation, not O(N²).

### 1.4 Order parameters

```
R e^{iΨ} = (1/N) Σ_j e^{iθ_j}       R = global synchrony, Ψ = herd's common phase
R₂       = |(1/N) Σ_j e^{2iθ_j}|    second harmonic
Q        = (1/N) Σ_j sin θ_j = R sin Ψ
⟨cos θ⟩  = R cos Ψ
```

`R ≈ 0` incoherent (orders cancel, efficient market). `R ≈ 1` with `Ψ ≈ −π/2` is a synchronised crash.

### 1.5 Integrator, stochastic Heun

```
f₁ = drift(θ, t)
θ̃  = θ + dt·f₁ + ξ
f₂ = drift(θ̃, t + dt)
θ  ← θ + (dt/2)(f₁ + f₂) + ξ
ξ  = σ√dt·η_i + σ_c√dt·η_c          (same ξ in predictor and corrector)
```

Strong order 1 for additive noise; 2nd-order accurate when `σ = 0`. `dt = 0.01-0.05`.

### 1.6 Price mapping

```
index:      dx   = (β Q   − κ x  ) dt + η  dB        P = 100 e^x
per-asset:  dx_i = (β q_i − κ x_i) dt + 3η dB_i      dB_i INDEPENDENT   ← the defect
β = 0.01 (price impact), κ = 0.05 (anchoring), η = 0.004
```

### 1.7 Observables, identical estimators on model output and real data, 60-observation windows

```
var_index   Var(r_index)
ac1_index   lag-1 autocorrelation of r_index
mean_corr   ρ̄ = 2/(N(N−1)) Σ_{i<j} ρ_ij
absorption  λ₁(C)/N
cssd        ⟨ std_i r_{i,t} ⟩_t
flicker     Pr_t[ |⟨sign r_{i,t}⟩_i| ≥ 0.6 ]
```

### 1.8 Network

Nodes are desks, edges are *who observes whose order flow*. Undirected, unweighted,
`A_ij ∈ {0,1}`, `k_i = Σ_j A_ij`. Giant component only.

| kind | construction | degree law | `κ = ⟨k²⟩/⟨k⟩²` | clustering `C` |
|---|---|---|---|---|
| `er` | `G(N,p)`, `p = ⟨k⟩/(N−1)` | Poisson | `1 + 1/⟨k⟩ ≈ 1.17` | `≈ p ≈ 0.01` |
| `ba` | preferential attachment, `m = ⟨k⟩/2` | power law γ=3 | ≈ 2.3 | `~(ln N)²/N` |
| `sf` | configuration model, cutoff `√(⟨k⟩N)` | power law γ=2.5 | larger | ≈ 0 |
| `ws` | ring rewired w.p. 0.1 | ≈ delta | ≈ 1.01 | `≈ 0.6(1−p)³` |
| `ring` | 1-D lattice, no rewiring | exactly `δ(k−6)` | exactly 1 | `3(k−2)/(4(k−1)) = 0.6` |
| `rr` | random regular | exactly `δ(k−6)` | exactly 1 | `≈ (k−1)/N` |

Structural quantities:

```
local clustering   C_i = 2T_i / (k_i(k_i − 1))        T_i = triangles touching i
average clustering C   = (1/N) Σ_i C_i
transitivity       T   = 3·(#triangles) / (#connected triples)
heterogeneity      κ   = ⟨k²⟩/⟨k⟩² = 1 + cv_k²
assortativity      r   = Pearson corr of (k_i, k_j) over edges
spectral radius    λ_max(A) ≈ max(√k_max, ⟨k²⟩/⟨k⟩)
Laplacian L = D − A, eigenratio μ_N/μ_2
```

### 1.9 Frequencies, the distribution that does the work

`ω_i` is the desk's intrinsic cycling rate, i.e. its inverse trading horizon: an HFT market maker has
large `|ω|`, a pension fund small `|ω|`. The **spread** `Δω` is strategy diversity, and it is what
resists synchronisation. The transition is the fight between `λ` (herding) and `Δω` (diversity).

**Scaling invariance.** The equation is invariant under `(ω, λ, t) → (aω, aλ, t/a)`. Only `λ/Δω` is
physical. Hence (i) frequency laws must be standardised before comparison, or you are only relabelling
the λ axis; (ii) S5's halt result comes out as `τ* = f(λ/λ_b)/Δω`.

Sampling uses deterministic quantiles, not i.i.d. draws, to suppress finite-N sampling noise:

```
ω_i = Φ⁻¹((i + 0.5)/N), randomly permuted, mean exactly 0
```

| law | `ω_i` | `g(0)` | consequence |
|---|---|---|---|
| Gaussian | `Φ⁻¹(u)` | `1/√(2π) = 0.3989` | continuous (2nd-order) |
| uniform | `√3(2u−1)` on `[−√3,√3]` | `1/(2√3) = 0.2887` | **first-order**, Pazó 2005: jump exactly `π/4` |
| Lorentzian | `tan(π(u−½))`, HWHM 1 | `1/π = 0.3183` | Ott-Antonsen exact: `R = √(1 − λ_c/λ)` |
| bimodal | `±1 + jitter` | **0** | threshold formula inapplicable; bistable near `λ = 2ω₀` |

**Locking condition.**

```
node i locks  ⟺  |ω_i − Ω| ≤ h_i,      h_i = (λ/⟨k⟩) k_i R_link
locked:   sin(θ_i − Ψ) = (ω_i − Ω)/h_i
drifting: contributes ≈ 0 to R
```

**Frequency-degree correlation `c`.** Because `h_i ∝ k_i`, hubs have the strongest local fields. If
hubs also carry the most extreme frequencies (`ω_i ∝ k_i`, i.e. `c = +1`), the hardest nodes to entrain
are exactly the ones with the most pull, so nothing locks until everything locks at once, the
explosive, hysteretic transition. Imposed by a Gaussian copula:

```
z_i = c · Φ⁻¹(rank(k_i)/N) + √(1−c²) · ε_i,   then assign the rank-matched ω
```

On a regular graph all degrees tie, so every `c` collapses to a random assignment, the built-in control.

### 1.10 Theory

**Annealed mean-field threshold** (Ichinomiya 2004; Restrepo, Ott & Hunt 2005), in this code's
`λ/⟨k⟩` normalisation:

```
λ_c = 2⟨k⟩² / (π g(0) ⟨k²⟩) = 2 / (π g(0) κ)
```

Structure enters through `κ` alone. Clustering, assortativity and path length do **not** appear, a
falsifiable claim that S8b tests.

**Herding mean field** (`theory.py`, `f = 1`), with `u = λR²`:

```
R = H(u) = u ∫_{−π/2}^{π/2} cos²φ · g(u sin φ) dφ
λ(u) = u / H(u)²          λ_b = min_u λ(u)     (saddle-node = recovery threshold)
```

**Trading halt** (S5): coupling off, oscillators rotate freely from the locked state,

```
R_halt(t) = | ∫_{|ω|<u} g(ω) e^{i[arcsin(ω/u) + ωt]} dω | · e^{−σ²t/2}
τ*(λ) = min{ τ : R_halt(τ) < R_u(λ) }        τ* = f(λ/λ_b) / Δω
```

**Observable mapping (S9, new).** With `a = βσ_c` and `v` the idiosyncratic return variance,

```
u_i = a cos θ_i / √(a² cos²θ_i + v)
ρ̄  = ⟨u_i u_j⟩_{i≠j} = ⟨u⟩² + O(1/N)
weak-mode limit:  ρ̄ ≈ (a²/v) · R² cos²Ψ          since ⟨cos θ⟩ = R cos Ψ
σ_c = 0  ⟹  ρ̄ ≡ 0                                an identity, not a prediction
```

The null alternative, a common shock added to the **prices** instead (`+ η_c dB_c`), gives
`ρ̄ = η_c²/(η_c² + v)`, a constant independent of `R`: it can match the average level of real
co-movement but produces no crash spike and no hysteresis. Distinguishing the two is the point of S9.

### 1.11 Explosiveness metrics

```
ΔR_jump = max Δ(R_fwd)                       first-order discontinuity
Δλ      = λ_f − λ_b                          bistable window
area    = ∫ max(R_bwd − R_fwd, 0) dλ         hysteresis loop area
```

### 1.12 Dataset

`research/empirical/`, 23 markets, ≈2,900 constituent stocks, daily, 2000-2026: US large/mid/small
(S&P 500 / 400 / 600), UK 100 & 250, Germany, France, Spain, Switzerland, Netherlands, Italy, Sweden,
Euro Stoxx 50, Japan, Hong Kong, India, Australia, Canada, Brazil, Korea, Singapore, 33 world indices
as nodes, 40 cryptocurrencies.

Crash = ≥10% fall within 30 trading days (30% for crypto); onset dated at the true peak *before* the
fall. 324 scored onsets across 69 calendar quarters (E1); 335-347 (E2). Indicators are causal
(past-data only) on 60-day trailing windows; CIs by cluster bootstrap over calendar quarters.

---

## 2. Every assumption

**[LB]** marks load-bearing: a result changes if the assumption is wrong.

### 2.1 Agent reduction

1. A trader is **one angle**, no wealth, inventory, leverage, balance sheet.
2. Behaviour is **strictly periodic**; a desk cannot simply hold.
3. **Order flow is exactly `sin θ`**, so every desk has identical maximum size; a pension fund and an
   HFT shop carry equal weight in `Q`. **[LB]**, real order flow is Pareto-distributed.
4. **First-order dynamics, no inertia**, instant reversal, no unwind cost. **[LB]**: `olmi2014` shows
   inertia alone produces hysteresis, so this is what licenses attributing hysteresis to herding.
5. **Fixed population**, no entry, exit, bankruptcy, capital constraint.
6. Agents differ **only** in `ω` and network position.

### 2.2 Interaction

7. **Pairwise and additive**, except the `r_i` herding term, which A1 proved is a 3-body interaction.
8. **Sinusoidal, first harmonic only**; depends only on the phase difference, odd and 2π-periodic.
9. **Reciprocal `A_ij = A_ji`.** **[LB]**, false in markets: everyone watches the large players, who
   do not watch back.
10. **Unweighted**, no trust or size weighting on edges.
11. **Instantaneous**, zero propagation delay. **[LB]** for a millisecond-scale phenomenon.
12. **Imitation only, no contrarians.** **[LB]**, market makers lean *against* flow by profession.
13. **One global `λ`** for the entire market.

### 2.3 Network

14. **Static**, no rewiring during a crisis.
15. **Exogenous and synthetic.** **[LB]**, no real market network is used anywhere in this project.
16. **Single layer**, information = trading = contagion network.
17. **Connected**, isolated traders discarded.

### 2.4 Frequencies

18. **Quenched**, a desk's horizon never changes, even mid-crash.
19. **Zero mean and symmetric.** **[LB]**, this is what freezes `Ψ`, which is what makes `ρ̄ ≡ 0`,
    which is what broke E2.
20. **A scalar rate per desk**, real desks trade many horizons at once.
21. **`ω ∝ k` for the GG2011 route.** **[LB]** and *fragile*: §7.1 shows a rank correlation is not
    sufficient, the **tail** of `g(ω)` must match the tail of `P(k)`.

### 2.5 Noise

22. Additive, Gaussian, white, identical `σ` for all desks.
23. Enters the **phase**, not the price.
24. **No common component** in the original model.
25. **State-independent**, no volatility clustering, no crisis amplification.

### 2.6 Price mapping (weakest layer)

26. **Linear price impact.** **[LB]**, real impact is concave (square-root law).
27. **Constant liquidity**, `β` fixed. **[LB]**, crashes *are* liquidity events, so this assumes away
    the main mechanism.
28. **Mean reversion to a constant fundamental** (= 100). No fundamentals process, no news about value.
29. **Node = trader AND node = asset simultaneously.** **[LB]**, should be bipartite.
30. **Independent per-asset price noise**, the `ρ̄ ≡ 0` identity.
31. **No microstructure**, no order book, spread, tick size, discrete time.
32. **`β, κ, η` chosen, not calibrated** to any market.

### 2.7 Analysis

33. **Annealed mean field** `A_ij ≈ k_i k_j/(N⟨k⟩)`, assumes no degree correlation, no clustering,
    large N, independent neighbours. Fails structurally on `ws` and `ring`.
34. **Adiabatic equilibration** at each `λ`. **[LB], this failed empirically**, see §7.1.
35. Thermodynamic-limit reasoning applied at N = 600.
36. **`λ` is exogenous** and slowly varying, not endogenously determined by the market.
37. **`R ≥ 0.5` defines "crashed"**, arbitrary threshold.

### 2.8 Empirical test

38. **Current constituents only** → survivorship bias (stated in the repo).
39. Crash = ≥10% fall in 30 trading days (30% crypto); onset = ex-post peak.
40. Cluster bootstrap over calendar quarters captures all the dependence.
41. **Daily closing prices can see this phenomenon.** **[LB], the deepest problem in the project.**
    The 2010 flash crash lasted 36 minutes; the model describes intraday synchronisation but the data
    is daily closes on 60-day windows, roughly three orders of magnitude apart. E1's and E2's null
    results may be measuring nothing but that mismatch.

---

## 3. What should be removed

| # | remove | why |
|---|---|---|
| R1 | the `κ` anchoring term | pins price to a constant fundamental of 100, so permanent repricing is impossible, but crashes *do* permanently reprice. Ad-hoc, no microfoundation. `κ = 0` gives the standard martingale price-impact model |
| R2 | the per-asset price system, **or** the claims built on it | node = trader = asset is a conflation. Either go bipartite (A13) or model only the index, then E2's correlation test simply does not apply and the over-claim disappears |
| R3 | the `norm="degree"` (`K/k_i`) code path | S1 proved it destroys explosive synchronisation (gap 0.06 vs 0.79), yet it is still in `model.py`, still used by `python/analysis/hysteresis_sweep.py`, and the README **still claims explosive sync for it**. A live contradiction |
| R4 | `NEWS_EPS = 0.02` | a constant bearish drift added only to stop the locked herd drifting sell→buy: a band-aid for assumption 19. Fix `⟨ω⟩` or add `σ_c` and it is unnecessary |
| R5 | the Stuart-Landau subsystem | `run_stuart_landau.py`, `stuart_landau.cpp`, its tests and CI, used by **none** of S1-S9 or E1-E2 |
| R6 | `flicker` | S6 retracted the flickering claim; E1 gives AUC 0.547-0.552 with CI touching 0.5 |
| R7 | `absorption` as a *separate* finding | `λ₁(C)/N` is nearly a monotone function of `mean_corr` under one-factor structure; reporting both inflates the apparent number of independent tests |
| R8 | `ring` from production sweeps | measured `λ_f = NaN` in **all 8 cells**, never reaches `R = 0.5`. Keep as a one-line negative control, not 1/6 of the compute |
| R9 | the bimodal jitter `0.05·randn` | unmotivated third parameter that makes "bimodal" not bimodal. Use exact `±ω₀` deltas or bimodal Lorentzians (Martens 2009, now in `references/`) |
| R10 | six topologies → three | ER (homogeneous), BA (heterogeneous), one high-clustering control. `ws`/`ring`/`rr` duplicate each other's role |

---

## 4. What should be added

### Tier 1, fixes a known defect

| # | add | parameters | effect |
|---|---|---|---|
| A1 | **common market mode `σ_c`** *(in flight)* | 1 | makes `ρ̄ ∝ R²`, turning R-hysteresis into observable co-movement hysteresis. Prior art: `pikovsky2016_common_noise`, `peter2019_microscopic_crosscorr` |
| A2 | **non-zero mean frequency `Ω₀ = ⟨ω⟩`** | 1 | currently *exactly* zero, which freezes `Ψ`. Letting it rotate produces genuine co-movement and removes R4 |
| A3 | **state-dependent liquidity** `β(Q) = β₀/(1−a\|Q\|)` | 1 | fat tails, volatility clustering, self-amplifying crashes. `bouchaud2010_endogenous` |
| A4 | **square-root impact** `dx = β·sign(Q)\|Q\|^½ dt` | **0** (replaces linear) | best-established law in market microstructure |

### Tier 2, new physics with literature support

| # | add | note |
|---|---|---|
| A5 | **inertia** `m θ̈_i + θ̇_i = ω_i + coupling` | inventory / risk limits. `olmi2014`, `multiplex_inertia2017`. Gives hysteresis **independently of `ω ∝ k`**, a second, more defensible route |
| A6 | **Sakaguchi phase lag** `sin(θ_j − θ_i − α)` | reaction delay / systematic lead-lag; A1 already derived its Jacobian |
| A7 | **explicit delay** `sin(θ_j(t−τ) − θ_i(t))` | more honest than `α` for millisecond markets |
| A8 | **triadic herding made explicit** | A1 *proved* the `r_i²` coupling equals `k_i⁻² Σ_{l,m,j} cos(θ_l−θ_m) sin(θ_j−θ_i)`, an exact 3-body hypergraph interaction. Promote it from footnote to model |
| A9 | **contrarians / repulsive coupling** (`λ < 0` for a fraction) | the most obviously missing economic actor: market making *is* mean reversion. Produces π-states and chimeras |
| A10 | **asymmetric herding** `λ(1 + b sin θ_i)` | breaks `θ → θ+π` symmetry so selling herds harder → leverage effect. `synchronization_market_asymmetry2006` |
| A11 | **co-evolving network** `A_ij(t)` | rewire toward successful neighbours |
| A12 | **multilayer** | separate information and trading layers. `danziger2019` |
| A13 | **bipartite trader × asset structure** | the principled fix for assumption 29 |

### Tier 3, analysis, not dynamics

| # | add | note |
|---|---|---|
| A14 | **estimate `λ` from data** by inverting `ρ̄ → R → λ` | **highest scientific value**, every result is phrased relative to `λ_b`, yet `λ` is never measured in a real market |
| A15 | **intraday data** | the biggest empirical fix: minute bars for 2010-05-06, 2015-08-24, 2018-02-05 would test the model on its own timescale |
| A16 | **estimate the real network** by lead-lag / Granger causality | instead of imposing BA |
| A17 | **Ott-Antonsen exact reduction** for Lorentzian `g` | closed-form `λ_f`, `λ_b`, replacing S2's ~10%-off numerics |
| A18 | **finite-size scaling of `ΔR`** | prove the jump is a true discontinuity, not a finite-N artifact |

### Priority

| rank | action | cost | payoff |
|---|---|---|---|
| 1 | A14 estimate λ from data | high | makes the whole framework operational |
| 2 | A15 intraday data | medium | fixes the timescale mismatch |
| 3 | A1 + A2 | running | fixes `ρ̄ ≡ 0`, resolves E2's caveat |
| 4 | R3 delete broken code path | trivial | removes a false README claim |
| 5 | A3 / A4 | low | fat tails, volatility clustering |
| 6 | A5 inertia | low | hysteresis without needing `ω ∝ k` |
| 7 | A9 contrarians | low | the missing economic actor |
| 8 | R1, R4, R5, R6 | trivial | less code, fewer fudges |

---

## 5. Novelty position

| claim | status |
|---|---|
| Markets near a first-order sync transition | **Prior art, peer-reviewed.** Lee et al., *PNAS* 2025, `doi:10.1073/pnas.2505434122`, ES-proximity predicts collapse/recovery, validated on EEG and **39 country indices, 2008**. This repo differs by working at **constituent level** (≈2,900 stocks, 23 markets, 324 onsets, 2000-2026) rather than 39 indices in one crisis |
| Routes to explosive synchronisation | **Not novel.** Degree-frequency correlation, bimodal frequencies, inertia, adaptive coupling, repulsive/multilayer interactions are all catalogued in `boccaletti2016_review`, `dsouza2019_review`. S8's value is a scale-matched benchmark, not a discovery |
| Common noise → cross-correlations | **Not novel as physics.** `pikovsky2016_common_noise`, `peter2019_microscopic_crosscorr`. What is open is the *financial observable mapping* (§1.10) and using it to test whether E2 had any power |
| Circuit breakers as desynchronisation control, `τ*(λ)` | **No prior work found.** Most novel item in the repo |
| E1/E2 negative results | rigorous, larger scale than the closest prior work |

---

## 6. Reference library

64 PDFs in `references/`. Added for this work:

| key | folder | why |
|---|---|---|
| `pikovsky2016_common_noise` | `09_common_noise` | prior art for `σ_c` |
| `peter2019_microscopic_crosscorr` | `09_common_noise` | common noise → cross-correlations |
| `nagai2010_common_noise_sync` | `09_common_noise` | noise-induced synchronisation |
| `martens2009_bimodal` | `01_kuramoto_foundations` | exact results for bimodal `g(ω)` |
| `restrepo2005_onset` | `01_kuramoto_foundations` | annealed threshold |
| `ichinomiya2004` | `01_kuramoto_foundations` | annealed threshold (`cond-mat/0406580`) |

**Still missing:** Sakaguchi 1988 (Prog. Theor. Phys., predates arXiv), the noise sweep has no
published anchor. Lee et al. PNAS 2025 PDF is bot-blocked; the bioRxiv version is in
`02_explosive_sync/` and only the citation metadata needs updating.

**Known breakage:** arXiv's **API** now returns HTTP 406 from this environment for every query, so
`references/fetch_references.py` can no longer resolve new papers by title. Direct PDF download by
known ID still works (`https://arxiv.org/pdf/<id>`), and every such download should be verified against
the paper's first page, a guessed ID for Ichinomiya returned a superconductivity paper.

---

## 7. Measured results

### 7.1 S8 factorial, and the design error it exposed

N = 600, `⟨k⟩ = 6`, 3 seeds, `λ ∈ [0,8]` in 81 steps, `t_relax = t_measure = 10`.

| cell | `ΔR_jump` | `λ_f` |
|---|---|---|
| `rr` / bimodal / c=1 | **0.581** | 1.80 |
| `rr` / bimodal / c=0 | 0.545 | 1.77 |
| `ba` / lorentz / c=1 | 0.377 | 5.27 |
| `ws` / bimodal / c=0 | 0.365 | 4.87 |
| `sf` / lorentz / c=1 | 0.321 | 7.60 |
| **`ba` / gauss / c=1** | **0.086** | 1.40 |
| `ring` / all 8 cells |, | **NaN** (never synchronises) |

**Two problems, both real.**

1. **`d_lam` is unreliable.** Several cells returned *negative* hysteresis width (`λ_b > λ_f`), which is
   unphysical for a true loop and means assumption 34 (adiabatic equilibration) failed at
   `t_relax = 10`. S8c doubles it to 20. Only `dR_jump` from this run should be quoted.

2. **The correlation was imposed wrongly.** The Gaussian copula fixes the *rank* correlation but forces
   the marginal of `ω`. With a Gaussian marginal the largest hub (`k = 79`) receives
   `ω = Φ⁻¹(1 − 1/2N) ≈ 3.1`, not `ω ∝ 79`. That is not the GG2011 mechanism, which needs
   *proportionality*, not merely ordering. The signature is in the table: `ba/gauss/c=1` gives
   `ΔR = 0.086` although S1 finds a 0.79 gap with literal `ω_i = k_i`, while heavy-tailed marginals
   (`lorentz`) restore the jump.

   **Conclusion: explosiveness requires the tail of `g(ω)` to match the tail of `P(k)`; a rank
   correlation is not sufficient.** S8c tests this with a `degree` law whose marginal *is* the
   standardised degree sequence, affine in `k`, so `c = 1` is literally GG2011 with a rescaled λ axis,
   and it carries a no-free-parameter check: `λ_f(here) = 1.56 · ⟨k⟩/std(k)`.

### 7.2 Real-data co-movement, the calibration target

Median over 23 markets; calm = ≥365 days from any onset, crash = within 90 days after an onset.

| measure | calm | crash | rise | p95 |
|---|---|---|---|---|
| `mean_corr` | 0.229 | 0.389 | **+0.127** | 0.532 |
| `absorption` | 0.271 | 0.424 | +0.114 | 0.556 |
| `flicker` | 0.200 | 0.367 | +0.117 | 0.483 |

The rise is positive in **23 markets out of 23**, from +0.029 (Brazil) to +0.265 (Japan).
**The current model produces 0.000**, it fails the most basic co-movement fact before hysteresis is
even at issue. That is the gap `σ_c` is meant to close.

### 7.3 S8c, the corrected correlation sweep

BA network, N=600, `t_relax = 20`, λ grid 0.125. Forward jump `ΔR` against the frequency-degree rank
correlation `c`, for three marginals of `g(ω)`:

| `c` | `degree` (tail matched to P(k)) | `lorentz` (heavy, unmatched) | `gauss` (thin) |
|---|---|---|---|
| −1.00 | 0.323 | 0.297 | 0.090 |
| −0.50 | 0.279 | 0.061 | 0.088 |
| 0.00 | 0.280 | 0.099 | 0.079 |
| 0.25 | 0.246 | 0.068 | 0.082 |
| 0.50 | 0.265 | 0.062 | 0.083 |
| 0.75 | 0.277 | 0.092 | 0.087 |
| **1.00** | **0.578**  (Δλ = 0.438) | **0.498**  (Δλ = 0.312) | **0.088**  (Δλ = 0) |

Three conclusions, all new relative to §7.1:

1. **A Gaussian marginal never produces an explosive transition, at any correlation**, `ΔR ≈ 0.08`
   flat across the whole range, `Δλ = 0` everywhere. This confirms the diagnosis in §7.1: S8's part B
   was measuring nothing.
2. **The jump appears only at |c| = 1, and only with a heavy-tailed marginal.** Even `c = 0.75` with
   the matched marginal gives 0.277, barely above the uncorrelated value. The mechanism needs near
   *proportionality* between `ω_i` and `k_i`, not a strong rank correlation. That is a sharper
   statement than the literature's usual "frequency-degree correlation causes explosive
   synchronisation".
3. **Falsifiable check passes.** The no-free-parameter prediction `λ_f = 1.56·⟨k⟩/std(k)` from S1 gives
   1.357; measured 1.562, ratio **1.15**. Within 15% with nothing fitted (residual is explainable by
   the different N, network realisation and the 0.125 λ grid).

Noise erodes the transition monotonically but does not destroy it (most explosive cell, `ba/degree/c=1`):

| σ | 0.0 | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|---|---|
| `ΔR_jump` | 0.578 | 0.645 | 0.639 | 0.621 | 0.452 | 0.480 |
| `Δλ` | 0.438 | 0.500 | 0.500 | 0.438 | 0.312 | 0.250 |

### 7.4 S9, the observable mapping, and the resolution of E2's caveat

Herding market (ER, f=1, σ=0.3), λ ramped up through λ_f and back down; co-movement measured with E1's
own estimators. Real-data targets: calm 0.229, crash 0.389.

| `σ_c` | `mc_calm` | `mc_crash` | theory `⟨u⟩²` at crash |
|---|---|---|---|
| 0.00 | 0.0006 | **0.0001** | 0.0000 |
| 0.10 | 0.0005 | 0.0026 | 0.0014 |
| 0.20 | 0.0005 | 0.0147 | 0.0093 |
| 0.30 | 0.0006 | 0.0322 | 0.0258 |
| 0.50 | 0.0007 | 0.0709 | 0.0621 |
| 0.80 | 0.0006 | 0.0955 | 0.1374 |

1. **The `σ_c = 0` row confirms the defect is an identity, not a parameter choice**: crash-time
   co-movement is 0.0001 against a real 0.389.
2. **The mechanism works and the closed form is right.** `mc_crash` rises monotonically with `σ_c` and
   the analytic `ρ̄ = ⟨u⟩²` tracks the measurement within about a factor of two over three decades.
3. **But one channel is not enough.** `mc_calm` stays at ~0.0006 for *every* `σ_c`: the phase-side
   common mode cannot produce a calm-period baseline at all. With lower idiosyncratic price noise the
   crash level is reachable (`σ_c = 0.5` → 0.347, `σ_c = 0.8` → 0.432, bracketing the real 0.389), but
   the calm level stays near zero.
4. **The null repair has the opposite failure.** A common shock in the *prices* reproduces the level
   (0.182 calm) but produces essentially no crash rise (0.182 → 0.190, i.e. +0.008 against a real
   +0.127), exactly as `ρ̄ = η_c²/(η_c² + v)` predicts.

   So a calibrated model needs **both** channels: `η_c` to set the calm baseline ≈0.23, `σ_c` to supply
   the crash-time rise. Neither alone fits.

**The decisive result, E2 had no power.** E2's hysteresis-gap statistic computed on the model:

| `σ_c` | 0.00 | 0.10 | 0.20 | 0.30 | 0.50 | 0.80 |
|---|---|---|---|---|---|---|
| gap (`mean_corr`) | −0.0001 | 0.0002 | 0.0005 | 0.0003 | 0.0026 | 0.0028 |

Every value lies far inside E2's real-data bound of `|gap| < 0.013`. **Even with the observable mapping
repaired, and even at implausibly large `σ_c`, the model predicts a co-movement hysteresis gap at most
~0.003, roughly four times smaller than the smallest effect E2 could have detected.**

E2's null result therefore does **not** falsify the model's R-hysteresis. It was a test without the
resolution to see the prediction. The caveat flagged in E2 is resolved, in the model's favour, and the
honest conclusion is a negative result about the *test*, not about the model. Detecting this would need
either a far larger sample or an observable that responds to `R` more directly than `ρ̄ ∝ R²` does.


### 7.5 S10 - the extensions, tested rather than asserted

Every extension in section 4 was implemented and scored against the 23 real markets on four stylised
facts the base model misses. Lambda cycles slowly across the bistable window so each run contains real
crash-and-recover episodes (the first version held lambda fixed and measured `frac_crashed = 0.0`, which
made the whole test vacuous; that bug is the reason the design changed).

Targets, measured from the 23 markets: excess kurtosis **10.20**, volatility clustering ACF1(|r|)
**0.246**, leverage effect **-0.113**, crash co-movement **0.389**.

| | baseline (base+linear) | best achieved | by | verdict |
|---|---|---|---|---|
| excess kurtosis | 0.035 | **2.06** | lag + liquidity | 59x better, still 5x short of 10.2 |
| volatility clustering | 0.008 | **0.34** | base + liquidity | **solved** (slightly overshoots 0.246) |
| leverage effect | -0.019 | **-0.029** | common + liquidity | **not solved**, 4x short |
| crash co-movement | 0.001 | **0.43** | lag + liquidity | **solved** (target 0.389) |

**State-dependent liquidity (A3) is the single most valuable addition.** It appears in every top-scoring
combination and is the only change that produces fat tails or volatility clustering at all. Without it
the model's returns are essentially Gaussian white noise.

**Asymmetric herding (A10) failed.** It was designed specifically to produce the leverage effect and
instead produces the **wrong sign** (+0.007 to +0.100 against a real -0.113), while halving the bistable
window. Reject it. The leverage effect remains unexplained by this model.

Does the bistable window survive each extension?

| variant | lam_f | lam_b | width | verdict |
|---|---|---|---|---|
| inertia | 5.700 | 3.300 | **2.400** | widens the window 3.6x; Olmi et al. confirmed |
| common (sigma_c) | 3.975 | 3.075 | 0.900 | preserves and slightly widens |
| meanfreq | 3.975 | 3.150 | 0.825 | preserves |
| **base** | 3.825 | 3.150 | 0.675 | reference |
| lag | 3.900 | 3.375 | 0.525 | narrows |
| asym | 3.675 | 3.375 | 0.300 | halves it |
| liq_common | 3.675 | 4.050 | **-0.375** | broken (anti-hysteresis) |
| contrarian | - | - | **none** | **destroys the transition entirely** |

**Contrarians (A9) must be rejected as a model improvement.** Just 15% of desks leaning against
observed flow suppresses explosive synchronisation completely (max hysteresis gap 0.108 against 0.767
for the baseline, and R never reaches 0.5). **This is known physics, not a new finding**: the
contrarian-oscillator literature reports that a fraction as small as 5% of contrarians significantly
suppresses synchronisation (Nature Sci. Rep. 2, 658; Sci. Rep. 5, 18235; arXiv:1702.08620). The
economic reading, that market makers are structurally stabilising, is the only part that is not already
in the physics literature.

**Keep:** A3 liquidity, A1 common mode, A5 inertia. **Reject:** A9 contrarians, A10 asymmetric herding.
**Mixed:** A2 mean frequency (scores well but overshoots volatility clustering when combined with
liquidity), A6 phase lag (best combined score but narrows the window).

### 7.6 S8b - inconclusive, and why

The theory check ran but **its results should not be quoted as findings**, for two reasons.

1. It compares the *measured* `lambda_f` (where R crosses 0.5) against a theory that predicts the
   *onset* of non-zero R. Those are different quantities and the measured one must systematically
   exceed the predicted one, so the reported ratio of 2.40 is not evidence the theory is wrong.
2. It is computed on the S8 factorial, whose adiabatic-equilibration assumption failed (section 7.1).

The Pazo check fails for the same reason plus a third: Pazo's `pi/4` jump is an all-to-all result, and
these are sparse networks at `<k> = 6`. Testing it properly needs an all-to-all run and a fitted onset,
not a 0.5 crossing. The structural correlations (n = 6 topologies) are far too small a sample to
separate kappa from clustering or assortativity.


### 7.7 E3 - Lee et al. (PNAS 2025) reproduced and tested out of sample

Their estimator, implemented exactly: Hilbert transform each asset's returns, build the instantaneous
Kuramoto order parameter across the market, take the ACF of `R(t)` in overlapping moving windows, and
use the **kurtosis of those ACF values** as ES proximity. 315 events, 23 markets, cluster bootstrap.

| sample | target | n | Spearman | 95% CI | holds |
|---|---|---|---|---|---|
| 2008 only (their setup) | collapse | 53 | -0.207 | [-0.435, +0.015] | **no** |
| 2008 only | recovery | 53 | -0.017 | [-0.313, +0.268] | **no** |
| out of sample | collapse | 262 | +0.080 | [-0.049, +0.184] | no |
| **out of sample** | **recovery** | 262 | **+0.133** | [**+0.038**, +0.215] | **yes** |

**The 2008 result does not reproduce at constituent level.** Out of sample across 262 independent
events the *recovery* half survives, with the sign hysteresis predicts (closer to explosive means a more
prolonged recovery); the *collapse* half does not. The surviving effect explains under 2% of variance.

Not an exact replication: they used 39 country indices from Compustat Global, this uses
constituent-level Yahoo data inside 23 markets, so the networks differ and survivorship bias applies.

### 7.8 E4 - the timescale objection, tested as far as free data allows

**Hard data limit.** Yahoo serves 1-minute bars for 7 days, 5-minute for 60 days, hourly for 730 days.
The 2010, 2015 and 2018 events are unreachable without a paid feed. Hourly is roughly a 7x improvement
over daily, not the full fix.

On hourly bars (119 US names, 730 days), pre-crash co-movement is **elevated before crashes but not
before ordinary peaks**: 0.206 against 0.123 sixty bars out, AUC **0.717** against E1's 0.542 on daily.

**But there are only 6 events.** Observations inside one event are heavily autocorrelated, so the
effective sample is about 6, not 72. Event-level sign test: 5 of 6. **Suggestive, not significant.**
Its value is identifying a paid intraday feed, rather than more daily data, as what would settle E1/E2.

### 7.9 A3 - lambda cannot be measured from co-movement, and the reason is structural

This was the headline recommendation of §4 (A14). It fails, for two independent reasons.

1. **The inversion is ill-posed.** `R_eq(lambda)` jumps from **R = 0.137 to R = 0.899 across a lambda
   window only 0.167 wide**. That is not numerical, it is the definition of a first-order transition.
   **97.2% of real market-days land inside that blind region**, leaving an inter-quartile range of 0.04.
   *The property that makes explosive synchronization interesting is exactly the property that makes its
   control parameter unidentifiable from its order parameter.*
2. **Even a perfect inversion could not help.** `lambda_hat` is a monotone transform of `rho_bar`, and
   AUC is invariant under monotone transforms, so it must score identically to the raw indicator. No
   reparametrisation of an indicator can beat that indicator.

An earlier version of the script reported AUC 0.949; that was an artifact of a thinning bug leaving only
8 comparison points. **A14 is withdrawn.** The replacement top recommendation is A15, intraday data.

### 7.10 A2 - does not finish

A2 was relaunched and ran for **12 h 22 min without writing a single new log line** past the
steady-state stage, at about 7% CPU per worker. It was killed. The propagation stage (160 source nodes
per model/network across 14 combinations, each needing integration to steady state plus a perturbation
response) is simply too large as configured.

**Consequence: the A1+A2 theory paper does not currently exist.** A1 completed and its separability
results stand, but A2's exponent table, on which the entire "generalised propagation theory" claim
rests, has never been produced. Any statement that the theory is validated is unsupported. Getting it
would need the source count cut by roughly an order of magnitude and the stage instrumented with
progress logging.


### 7.11 E5 (A16) - the model run on a REAL market network, and why this is the most serious result here

Assumption 15 said no real market network is used anywhere, and marked it load-bearing. It is now
tested. Two networks were estimated per market and thresholded to the same mean degree (12) as the
synthetic benchmark, so degree is held fixed and only structure differs:

* `corr` : threshold the pairwise return-correlation matrix.
* `leadlag` : keep i-j when i's returns lead j's at +1 day more strongly than the reverse, then
  symmetrise.

**Structure (mean over US, UK, Japan, Germany):**

| network | `<k>` | kappa | clustering | assortativity |
|---|---|---|---|---|
| real, corr | 17.3 | 1.80 | **0.731** | -0.065 |
| real, leadlag | 12.7 | 1.98 | **0.138** | -0.247 |
| synthetic BA | 11.8 | 1.68 | 0.081 | -0.049 |
| synthetic ER | 12.0 | 1.09 | **0.030** | -0.009 |

Real market networks are **an order of magnitude more clustered** than any synthetic graph used
anywhere else in this repository, and more degree-heterogeneous than ER.

**Dynamics, same herding model, same f = 1, same sigma = 0.3:**

| network | lambda_f | lambda_b | window | max hysteresis gap | jump |
|---|---|---|---|---|---|
| **synthetic ER** (what S2-S5 use) | 4.10 | 3.30 | **0.800** | **0.793** | 0.768 |
| synthetic BA | 3.40 | 2.90 | 0.500 | 0.502 | 0.546 |
| **real, corr** | 2.45 | 2.55 | **-0.100** | **0.172** | 0.291 |
| **real, leadlag** | 2.60 | 2.60 | **0.000** | **0.190** | 0.325 |

**The bistable window essentially vanishes on networks estimated from real market data.** The
hysteresis gap falls from 0.793 to 0.17-0.19, a factor of four, and the window width goes to zero.

This matters more than any other result in the repository, because **S4 (crash persistence is set by
the hysteresis branch) and S5 (minimum safe trading-halt duration) both depend entirely on that window
existing.** Both were measured on synthetic ER. On a real market network there is no window for a shock
to leave the market in, so:

* S4's central claim, that whether a crash heals or sticks is a property of where lambda sits rather
  than of the shock, has no bistable region to operate in.
* S5's tau*(lambda), the most novel item in the repository, is a recovery-threshold calculation for a
  threshold that does not appear.

The most likely mechanism is the clustering gap. High clustering means neighbours share neighbours, so
local agreement r_i saturates early and locally instead of propagating globally, which is exactly the
positive feedback the herding route depends on.

**Novelty correction.** That mechanism is **already published**: "Topologically-induced suppression of
explosive synchronization", arXiv:2301.08000, *Chaos* 33, 053103 (2023), shows that converting a
star-like topology into one containing cycles turns explosive synchronization into the ordinary
continuous kind, and the broader literature states the rule directly ("any suppressing factor which
hampers the merging of small synchronous clusters leads to abrupt formation of a single giant
component", so clustering, which merges clusters smoothly, suppresses it). So E5 is **not** a new
physical mechanism. What is new here is the **empirical** half: that networks estimated from real
market returns sit squarely in the clustered, suppression regime (C = 0.73 against 0.030 for the
synthetic ER graph used throughout), and that this removes the bistable window S4 and S5 depend on.
The contribution is "the market's own network is the wrong network for this model", not "clustering
suppresses ES". That also means assumption 33 (annealed mean field,
which assumes no clustering) fails on the real networks, so the theory in section 1.10 does not apply
to them either.

**Caveats, stated plainly.** The real networks are smaller (n = 125 to 335 against 400) and the
thresholding is a modelling choice, not an observable; a different construction could give a different
graph. Neither caveat explains a fourfold collapse of the gap, but both should be checked before this
is treated as settled. The honest next step is to rebuild the synthetic comparison at matched n and to
sweep the correlation threshold.

### 7.12 S11 (A17, A18) - exact thresholds, and whether the jump is real

**A17.** For Lorentzian g with half-width gamma the herding model is exactly solvable. The classic
Kuramoto result R = sqrt(1 - K_c/K), K_c = 2 gamma, enters the mean field as K R; in the herding model
the gain is alpha_i = r_i so it enters as lambda R^2, i.e. K_eff = lambda R. Substituting gives a cubic

    lambda R^3 - lambda R + 2 gamma = 0

whose fold (double root) is at dR: 3 lambda R^2 - lambda = 0, so R* = 1/sqrt(3), and back-substituting

    **lambda_b = 3 sqrt(3) gamma ~= 5.196 gamma,    R at the fold = 1/sqrt(3) ~= 0.5774.**

A closed form, replacing the numerical saddle-node search in `theory.py`. Verified two ways: at
lambda = 5.2, just above the fold, the two roots are 0.5901 and 0.5645, converging on 0.5774 as they
must; and a direct simulation at N = 2000 gives lambda_b = 5.800 against the exact 5.196, a ratio of
**1.116**. That 12% gap is not an error in the algebra, it is the finite-connectivity correction: the
closed form is exact for all-to-all coupling, while the simulation runs on a sparse ER graph with
<k> = 12. It is the same ~10% offset S2 reported for its numerical mean field, now explained rather
than merely measured.

**A18.** Finite-size sweep, 3 seeds per size, Lorentzian frequencies, sigma = 0:

| N | 125 | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|
| jump `dR` | 0.183 | 0.254 | 0.300 | 0.330 | 0.277 | **0.486** |
| max hysteresis gap | 0.119 | 0.181 | 0.283 | 0.284 | 0.318 | **0.473** |
| window `lam_f - lam_b` | 0.000 | 0.067 | 0.200 | 0.067 | 0.267 | **0.400** |

**The jump grows with N rather than decaying** (0.183 -> 0.486 over a 32x range of system size), and so
do the hysteresis gap and the window. A finite-size artifact would shrink toward zero. **So the
first-order discontinuity is genuine**, and the values quoted elsewhere at N = 500 to 600 are if
anything conservative.

Caveats: only 3 seeds per size, the trend is not perfectly monotone (N = 2000 dips to 0.277), and the
window is resolved only to the 0.2-wide lambda grid. The direction is unambiguous but the exponent is
not estimated here.

**Read this together with E5 (section 7.11).** A18 says the transition is real *on an ER graph*. E5 says
it very nearly disappears on a network estimated from real market data. Those are not in conflict: the
discontinuity is a genuine property of the model, and the model's network is not the market's.


### 7.13 E6 - the 2010 flash crash at 1-minute resolution

The test E1, E2 and E4 could not do. HF Data Library publishes 1-minute bars back to 2002 under
CC BY 4.0, so 2010-05-06 is reachable. 32 large caps, 1,950 one-minute bars over 3-7 May 2010.

| window | R | mean correlation |
|---|---|---|
| baseline, 3-5 May | 0.5142 | 0.3361 |
| 13:00-14:32, before the crash | 0.6681 | **0.4734** |
| 14:32-15:08, the crash itself | 0.6487 | 0.4322 |

Co-movement rose **+0.1373 BEFORE 14:32** and then fell slightly during the crash itself. On daily data
E1 found the opposite: flat until the peak, rising only afterwards.

**But the null matters, and it cuts the claim down.** Running the same 13:00-14:32 window on every day
in the sample:

| day | 13:00-14:32 level | afternoon rise over the morning |
|---|---|---|
| 2010-05-03 | 0.2160 | +0.0133 |
| 2010-05-04 | 0.3891 | +0.0468 |
| 2010-05-05 | 0.4137 | **+0.1151** |
| **2010-05-06 (crash)** | 0.4734 | **+0.1414** |
| 2010-05-07 | **0.6230** | +0.0199 |

The crash day has the largest afternoon rise of the five, but **2010-05-05 reaches +0.115**, and the
crash day's absolute level is **not** the highest in the week (2010-05-07, the aftermath, is). With
**one crash and four control days** this cannot separate a genuine precursor from ordinary intraday
variation, and the whole week sits inside the European debt crisis, so every day is elevated.

**Verdict: suggestive, not established.** It motivates E7 rather than settling anything.

### 7.14 E7 - E1's event study at 1-minute resolution, 2002-2026. The timescale objection, settled.

23 tickers, **2,319,682 one-minute bars, 2002-12-30 to 2026-09-24**, 31 intraday crash events (a 3%
fall of the equal-weight index within 60 minutes), against E1's own matched null of local maxima not
followed by a fall. One value per event, so events rather than minutes are the unit of analysis.

**The unmatched result looks spectacular:**

| lag (min) | -90 | -60 | -30 | 0 | +30 | +90 |
|---|---|---|---|---|---|---|
| crash onsets | 0.429 | 0.434 | 0.538 | 0.480 | 0.551 | 0.499 |
| ordinary peaks | 0.241 | 0.223 | 0.204 | 0.227 | 0.237 | 0.238 |

Pre-event co-movement: **0.458 before crashes against 0.228 before ordinary peaks, AUC 0.911
[0.843, 0.969]**, versus E1's 0.542 on daily data. Taken at face value this would say E1's null was a
resolution artifact and the signal lives intraday.

**It is not. The volatility-matched control removes it entirely.**

Crashes happen in high-volatility regimes and co-movement rises with volatility, so an unmatched
comparison can score well purely as a volatility proxy. E1 handled this on daily data with a
volatility-matched null; applying the same control here, pairing each crash with the nearest-volatility
ordinary peak without replacement:

| | value |
|---|---|
| pre-event volatility, crashes vs peaks | 0.00151 vs 0.00031, **4.94x higher** |
| AUC, unmatched | 0.897 |
| **AUC, volatility-matched** | **0.526, CI [0.340, 0.712]** |
| beats chance | **no** |
| median co-movement, crashes vs matched peaks | 0.459 vs 0.409 |

**Once volatility is held fixed, co-movement carries no skill at minute resolution either.**

**This settles the single biggest threat to the empirical chapters.** Assumption 41 said the daily grid
might be three orders of magnitude too coarse and that E1's and E2's nulls could be measuring nothing
but that mismatch. Tested at the model's own timescale, across 24 years, the same confound structure
appears and the same conclusion holds. **E1's daily null was not a resolution artifact, and E1's
conclusion survives the test it was most vulnerable to.**

It also retires recommendation A15: intraday data was the top open item, it has now been obtained and
used, and it did not change the answer.

Caveats: 21 crashes survive the pre-window data requirement, the matched CI is correspondingly wide
[0.340, 0.712], and the cross-section is 23 large caps rather than a full index. A larger ticker set
would tighten the interval but the point estimate of 0.526 is not close to meaningful skill.


### 7.15 S12 - the last three structural extensions (A11, A12, A13)

| variant | kurtosis | vol clustering | crash co-movement | score | bistable window |
|---|---|---|---|---|---|
| **base** | 0.167 | 0.076 | 0.026 | 0.936 | **0.975** (gap 0.789) |
| A11 co-evolving network | 0.162 | 0.055 | 0.021 | 0.923 | 0.825 (gap 0.764) |
| A12 multilayer | 0.045 | **0.127** | 0.035 | **0.793** | **none** (gap 0.465) |
| A13 bipartite mapping | slightly worse on every fact | | | | unchanged (price-side only) |

* **A11 (co-evolving network)** barely moves the stylised facts and mildly narrows the window. Its one
  interesting effect is counter-intuitive: rewiring toward desks that already agree with you *reduces*
  global synchrony (R 0.926 -> 0.712 at a 20% rewire rate), because homophily fragments the network into
  like-minded clusters. Local order, global disorder. An echo-chamber effect, not an amplifier.
* **A12 (multilayer)** scores best on the stylised facts but **destroys the bistable window**
  (`lambda_b` undefined, gap 0.465 against 0.789). Same verdict as contrarians in S10: it improves the
  fit by removing the phenomenon.
* **A13 (bipartite trader x asset)** makes every stylised fact slightly worse and leaves co-movement
  unchanged. It is still the structurally honest choice (assumption 29), but it buys nothing measurable.

**None of the three is worth adopting.** Combined with S10, that is five of eleven tested extensions
rejected on evidence.

### 7.16 E8 - could you actually have called a crash? The confusion matrix.

E7 reported AUC, which is a ranking measure and hides the two things that decide usability: the base
rate, and the confound. E8 fixes thresholds on **2002-2015** and applies them unchanged to
**2016-2026**: 10 crashes across 2,697 test trading days, a base rate of **0.37%**.

Three detectors, the third being co-movement after regressing out log-volatility (the part of
synchronization that is *not* volatility; the regression has R^2 = 0.414):

| detector | budget | TP | FP | TN | FN | precision | recall | false alarms per catch |
|---|---|---|---|---|---|---|---|---|
| mean_corr | 1% | 5 | 388 | 2299 | 5 | 0.0127 | 0.50 | **77.6** |
| mean_corr | 5% | 6 | 856 | 1835 | 4 | 0.0070 | 0.60 | 142.7 |
| volatility | 1% | 4 | 261 | 2432 | 6 | **0.0151** | 0.40 | **65.2** |
| **corr \| vol** | 1% | 2 | 549 | 2140 | 8 | **0.0036** | 0.20 | **274.5** |
| corr \| vol | 5% | 1 | 1173 | 1516 | 9 | 0.0009 | 0.10 | 1173.0 |

**Three conclusions, and they close the empirical question.**

1. **Nothing is usable.** The best detector catches half the crashes at **77 false alarms per catch**,
   precision 1.3% against a 0.37% base rate. Operationally worthless at any threshold tested.
2. **Volatility alone is the best of the three.** It beats the model's own observable.
3. **Stripping the confound makes it worse, not better.** `corr | vol`, the part of co-movement that is
   genuinely not volatility, is the *worst* detector: precision 0.0036, recall 0.20, 274 false alarms
   per catch. So the synchronization-specific component carries no predictive information at all.

E7 showed the ranking signal was volatility. E8 shows that what remains after removing volatility is
not merely weak but actively unhelpful, out of sample, at the model's own timescale.


### 7.17 S13 - two NEW results, derived here and tested here

Everything else in this document either reproduces known work or reports a negative result. These two
are positive, quantitative and, as far as the searches reached, unpublished.

#### H1. A closed-form family of recovery thresholds that unifies three models

Generalise the herding gain to `alpha_i = r_i^p`. In mean field `r_i -> R`, so the coupling term is
`lambda R^(p+1)` and the effective Kuramoto coupling is `K_eff = lambda R^p`. Substituting into the
exact Lorentzian result `R = sqrt(1 - 2 gamma / K)`:

    lambda R^(p+2) - lambda R^p + 2 gamma = 0

The fold is where the derivative vanishes, `(p+2) R^(p+1) = p R^(p-1)`, giving a closed form for
**every** p:

    R*(p)       = sqrt( p / (p+2) )
    lambda_b(p) = gamma (p+2) ( (p+2)/p )^(p/2)

**One expression spans three separately-studied models:**

| p | R*(p) | lambda_b(p) | which model |
|---|---|---|---|
| -> 0 | 0 | **2 gamma** | plain Kuramoto: no fold, continuous transition, the classic `K_c = 2 gamma` |
| 1 | 1/sqrt(3) = 0.5774 | **3 sqrt(3) gamma = 5.196** | the pairwise herding model (Zhang et al. 2015 form) |
| 2 | 1/sqrt(2) = 0.7071 | **8 gamma** | the exact 3-body hypergraph interaction A1 proved `r_i^2` to be |

All three limits check analytically. Tested against simulation (N = 1500, Lorentzian, sigma = 0):

| p | predicted | simulated | ratio |
|---|---|---|---|
| 0.5 | 3.738 | 4.231 | 1.132 |
| 1.0 | 5.196 | 5.598 | 1.077 |
| 1.5 | 6.608 | 7.208 | 1.091 |
| 2.0 | 8.000 | 8.946 | 1.118 |

**Median ratio 1.105, total spread across p just 0.055.** That is the decisive pattern: the predicted
threshold moves by a factor of 2.1 across this range and the ratio stays flat to within +-2.5%. A
formula with the wrong *shape* would show a ratio that drifts with p. The constant ~10% offset is the
finite-connectivity correction already identified in S11 (which measured 1.116 independently at p = 1,
N = 2000): the closed form is exact for all-to-all coupling, the simulation is sparse at `<k> = 12`.

**Failure to report:** p = 3 returned no transition (jump 0.0137) inside the scanned coupling range, so
the family is verified over p in [0.5, 2] only. Whether it holds at large p is untested.

#### H2. A critical clustering, with a number

E5 found the window nearly vanishes on real market networks, and the literature already knows
clustering suppresses explosive synchronization *qualitatively* (Chaos 33, 053103, 2023). Nobody has
given a threshold. Sweeping Watts-Strogatz rewiring at fixed mean degree (which holds `kappa` at
1.00-1.04 throughout, so this isolates clustering from degree heterogeneity):

| clustering C | 0.014 | 0.092 | **0.354** | 0.504 | 0.586 | 0.642 | 0.682 |
|---|---|---|---|---|---|---|---|
| hysteresis gap | 0.327 | 0.312 | **0.099** | 0.089 | 0.116 | 0.065 | 0.075 |
| jump dR | 0.371 | 0.387 | 0.163 | 0.118 | 0.135 | 0.067 | 0.082 |
| window exists | yes | yes | marginal | marginal | **no** | **no** | **no** |

    **C* ~= 0.35**: below it the window is open (gap ~ 0.31), above it closed (gap < 0.10).

And the number is what makes it useful:

| network | clustering | relative to C* |
|---|---|---|
| synthetic ER used throughout this repository | **0.030** | far below: window open |
| real market network, lead-lag | 0.138 | below, but only by 2.5x |
| **real market network, correlation** | **0.731** | **2.1x above: window closed** |

This turns E5 from an observation into an explanation. The synthetic graph every crash result was
measured on sits an order of magnitude below the critical clustering; the real correlation network sits
twice above it. **The model's bistable window is an artifact of using an unclustered graph**, and the
crossing point is now quantified rather than asserted.
