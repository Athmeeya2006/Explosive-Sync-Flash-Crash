# Research ideas, keyed to the five papers

Written for: Athmeeya, as a working plan to take to Prof. Hens.

Every idea below is tied to a specific paper in `references/`, states what that paper leaves open, says
what the new claim would be, and says what has to be computed. The five papers are:

| # | paper | local copy |
|---|---|---|
| P1 | Dutta, Pal & Hens, *Generalized Adaptation-Induced Non-universal Synchronization Transitions in Random Hypergraphs*, arXiv:2507.18421 (2025) | `references/10_prior_art_adaptive/adaptation_hypergraphs2025__2507.18421.pdf` |
| P2 | Kumpeerakij & Restrepo, *Self-consistent analysis of the Kuramoto model with higher-order interactions*, arXiv:2605.24701 (2026) | `references/10_prior_art_adaptive/selfconsistent_hoi2026__2605.24701.pdf` |
| P3 | Miranda, Frasca & Estrada, *Topologically-induced suppression of explosive synchronization*, Chaos 33, 053103 (2023) | `references/10_prior_art_adaptive/miranda2023_topological_suppression__2301.08000.pdf` |
| P4 | *Early warning signals for stock market crashes*, EPJ Data Science 13, 25 (2024) | `references/06_finance/ews_market_crashes_epjds2024.pdf` |
| P5 | Pang & Lin, *Market state transitions and crash early warning in the Chinese stock market*, Front. Phys. (2025) | `references/06_finance/pang_lin2025_market_states.pdf` |

Two further papers matter for what is and is not new, and both are now in the library:
Zou & Wang, PRE 102, 012219 (2020), and Dutta et al., PRR 7, L022049 (2025).

---

## The result these ideas are built on

`research/s14_finite_connectivity.py`. Every analytic threshold in this literature, including P1 and
the Zou-Wang family that S13 wrongly claimed, is **all-to-all**. The annealed mean field replaces the
local order parameter `r_i` by the global `R`. But the exact locking field is

```
h_i = (lam / <k>) * k_i * r_i^(p+1),     r_i = |sum_j A_ij e^{i theta_j}| / k_i
```

and `r_i` is the modulus of a sum of `k_i` unit vectors, hence a **Rician** random variable with scale
`s^2 = (1 - R^2) / (2 k_i)` per component. Node `i` also does not lock to the global mean phase `Psi`;
it locks to the direction `phi_i` of its own neighbour sum, which contributes a projection factor
`cos(phi_i - Psi) = X/r`. That factor is what keeps `R = 0` an exact solution on a sparse graph. The
closure is

```
R = < c( (lam k / <k>) r^(p+1) ) * X / r >,        c(h) = (sqrt(h^2 + gamma^2) - gamma) / h
X ~ N(R, s^2),  Y ~ N(0, s^2),  r = sqrt(X^2 + Y^2)
```

and expanding to `O(s^2)` shifts the fold by an amount whose coefficient is exact:

```
lam_b(p, <k>) = gamma (p+2) ((p+2)/p)^(p/2) * [ 1 + (p+2)/(2<k>) + O(<k>^-2) ]
```

Three further consequences, all in S14: a **second threshold** `lam_c(p, <k>)` from the linear
instability of `R = 0`, which does not exist all-to-all; a **critical connectivity** `k_c(p)` below
which the window `[lam_b, lam_c]` is closed at every coupling; and clustering entering only through an
**effective connectivity** `k_eff = k / (1 + (k-1) rho)`, where `rho` is the mean pairwise phase
correlation inside a neighbourhood.

That last point is the lever for most of what follows: **sparsity, clustering and small system size are
the same effect measured in different units.**

---

## P1. Dutta, Pal & Hens: generalized adaptation on random hypergraphs

**What the paper does.** Partial adaptation, a fraction of nodes whose coupling is multiplied by a
general function `g(r1)` of the global order parameter, on random hypergraphs with pairwise and triadic
coupling. Ott-Antonsen throughout. Polynomial and Gaussian adaptation functions. Findings: a
double-jump transition under power-law adaptation, an intermediate synchronization state for
polynomial adaptation with one negative coefficient, and transitions that switch between continuous
and explosive as the pairwise coupling varies.

**What it leaves open.** The paper's own stated closure is that the joint density factorises "in the
limit of complete incoherent and synchronized states or in the case of dense hypergraphs". Everything
is therefore a dense-limit result. Real hypergraphs, and every network in this repository, are sparse.

### Idea 1: the finite-connectivity correction for an arbitrary adaptation function

The S14 derivation never used `g(r) = r^p`. Redo the `O(s^2)` expansion for a general `g`. The
correction coefficient becomes a functional of `g` evaluated at the fold:

```
a[g] = - s^2 * [ phi'(R)/(2R) + phi''(R)/2 - phi(R)/(2R^2) ] / (d phi / d lam)    at the fold
       with phi(r) = c(lam * g(r) * r)
```

For `g(r) = r^p` this collapses to `(p+2)/2`, which is the check. For P1's Gaussian and polynomial
adaptation functions it gives a different number, computable in closed form or by quadrature.

**The claim.** A single formula gives the leading sparsity correction to the fold for *any* adaptation
function, with the power-law case as a special value. This is a direct extension of P1 using P1's own
machinery, and it is the correction their dense-limit closure necessarily misses.

**What to compute.** Reuse `rician_fold` with `g` passed as a callable. Validate by bisection on
random regular graphs at `<k>` from 6 to 96 for three adaptation functions: power law, the Gaussian of
P1, and one polynomial with a sign change. Two days of work; the harness already exists.

### Idea 2: sparsity should destroy the double jump, at a predictable connectivity

This is the sharper version and I think it is the better paper.

P1's double-jump transition requires **two separate folds** in the same `R(lam)` curve. S14 shows that
folds are squeezed shut as connectivity falls: the relative window width at `p = 1` goes from 1.93 at
`<k>` = 96 to 0.0027 at `<k>` = 6. If there are two folds, each has its own critical connectivity.

**The prediction.** As `<k>` decreases, the two hysteresis loops of P1's double-jump transition close
at **different** `k_c`, so the phase diagram passes through an ordered sequence:
double explosive, then single explosive, then continuous. The order in which the two loops die, and the
two `k_c` values, follow from the closure with no fitted parameter.

This is a new phenomenon in their own system, it is a falsifiable ordering rather than a fitted curve,
and it matters physically because real hypergraphs, brain networks especially, are sparse. If it holds,
the double-jump transition is a dense-hypergraph phenomenon and the paper's phase diagram needs a
connectivity axis.

**What to compute.** Implement their triadic hypergraph in `model.py` (the repo's `herd_power = 2`
already *is* the exact triadic interaction, proved in A1, so most of it exists), then sweep `<k>` and
the adaptation exponent and locate both folds by bisection. The expensive part is that both folds need
resolving, so budget maybe 5,000 bisection runs. That is hours on 16 cores, not days.

**Why take this to Hens first.** It extends his own 2025 PRR and his 2025 arXiv paper, it uses his
group's model, and the collaboration is natural rather than competitive.

---

## P2. Kumpeerakij & Restrepo: self-consistent analysis on finite hypergraphs

**What the paper does.** Exactly the gap S14 addresses, but for a different closure. They build a
self-consistent framework for dyadic plus triadic Kuramoto on **finite** hypergraphs, introduce
generalized local order parameters, and derive a hierarchy of approximations. Their headline is that
the critical triadic coupling for the onset of bistability depends on correlations between the
eigenvectors of the dyadic adjacency matrix and the triadic interaction structure.

**What it leaves open, precisely.** Their stated assumptions include that the local order parameters
depend **only on the weighted nodal degrees**, and that the natural frequencies `omega_n` are
uncorrelated with the local field. So their closure keeps the *mean* local order parameter as a
function of degree and discards its *distribution*. There are no triangles in that closure.

### Idea 3: the fluctuation closure, combining their eigenvectors with the Rician distribution

Their framework is a mean closure; S14's is a distributional one. They are complementary and they
compose: their eigenvector machinery gives the *mean* local field per node, and the Rician scale gives
the *spread* about it.

**The claim.** The leading correction to any bistability threshold on a sparse structure is set by the
variance their closure discards, so a degree-based local-order-parameter closure systematically
**under-predicts** thresholds on sparse hypergraphs by a computable amount. Write that amount down,
and the combined closure handles sparse, clustered and degree-heterogeneous hypergraphs at once.

**What to compute.** Implement their hierarchy (their paper gives it explicitly), then add the
fluctuation term. Test on the same homogeneous and heterogeneous hypergraphs they use, plus clustered
ones, which is where the two closures should visibly separate. This is the most technically demanding
idea here and the most likely to become a solid Phys. Rev. E paper.

### Idea 4: their uncorrelated-frequency assumption fails exactly where explosive synchronization lives

In the locked state, high-degree nodes feel a stronger field, so they stay locked at larger `|omega|`.
Conditional on being locked, `omega_i` and `h_i` are therefore **correlated**, and that correlation is
not a nuisance: it *is* the frequency-degree mechanism of Gomez-Gardenes et al. So their assumption
breaks precisely in the regime that produces explosive synchronization.

This connects to a result this repository already has and has not exploited. S8c found that a
frequency-degree **rank** correlation is not sufficient: with a Gaussian frequency law there is no jump
at any correlation from -1 to +1, and the jump appears only when the **tail** of `g(omega)` matches the
tail of `P(k)`. In the language of P2, tail-matching is the condition under which the
omega-to-local-field correlation becomes strong enough to matter.

**The claim.** Quantify the error P2's assumption introduces as a function of the frequency-degree
coupling, and show that the tail-matching criterion of S8c is exactly the boundary at which their
closure stops being valid. That reframes an existing negative result in this repo as a statement about
when a published analytic framework applies.

**What to compute.** Cheap. Reuse `s8c_correlation_corrected.py`, add a measurement of
`corr(omega_i, h_i | locked)` along the same sweep, and compare against P2's predicted threshold.

---

## P3. Miranda, Frasca & Estrada: cycles suppress explosive synchronization

**What the paper does.** Oscillators coupled through a **degree-biased Laplacian** on star-like, tree
and Cayley graphs. Explosive synchronization appears on the star-like structure and reverts to a
classical transition once cycles are introduced. They distinguish short cycles (triangles) from long
cycles and find that **long cycles suppress the transition strongly, while the triangle case is much
less clear-cut and the hubs-repelling operator "behaves more erratic" when a short cycle appears**.

**What this means for S13 and S14.** S13 cited this paper as prior art for "clustering suppresses
explosive synchronization" and conceded the mechanism. That concession was too generous: the coupling
operator is different, the route is frequency-degree rather than adaptive, and their controlling
variable is cycle length, not the clustering coefficient.

### Idea 5: two orthogonal topological mechanisms, and an experiment that separates them

This is the idea I would push hardest after Idea 2, because it reconciles two papers instead of
duplicating either.

In the S14 picture, clustering suppresses the adaptive fold through exactly one channel: triangles put
two neighbours of `i` in contact with each other, correlating their phases, raising `rho`, and lowering
`k_eff`. **Long cycles do not enter `r_i` at all.** They have no first-order effect on the local
order parameter, so in the adaptive route they should be nearly inert. In P3's degree-biased route the
suppression comes from the spectrum of the coupling operator, which long cycles change substantially
and triangles change only weakly. So the two routes have **orthogonal topological sensitivities**:

| | adaptive route (S14) | degree-biased Laplacian route (P3) |
|---|---|---|
| triangle density at fixed spectrum | strong suppression | weak, erratic |
| long-cycle structure at fixed clustering | little effect | strong suppression |

**The experiment.** Build graph ensembles that decouple the two. Newman's clustered configuration model
sets the triangle density independently of the rest of the structure, so you can hold the clustering
coefficient fixed while varying the long-cycle spectrum, and vice versa. Watts-Strogatz cannot do this
because it entangles clustering with girth, which is exactly why S13's sweep could not have separated
the mechanisms even if the coupling range had been right. Then run both coupling schemes on both
ensembles: a two-by-two design.

**The claim.** Explosive synchronization has at least two distinct topological kill switches, they are
sensitive to different features of the cycle structure, and `k_eff` predicts one of them quantitatively
while the adjacency spectrum predicts the other. That is a clean dichotomy, it is falsifiable in four
cells, and it tells anyone modelling a real network which topological statistic they actually need to
measure.

**What to compute.** The clustered configuration model is about thirty lines with `networkx`. The
degree-biased Laplacian is a one-line change to `drift()` in `model.py`. Then the same bisection
harness. Genuinely a week of work for a complete paper.

---

## P4. EPJ Data Science: recurrence-network early warning signals

**What the paper does.** Builds multiplex recurrence networks from multidimensional returns and uses
two indicators, the average mutual information and the average edge overlap, as early warning signals
for extreme volatility on S&P 500 and Chinese constituents. It benchmarks against Guttal et al., whose
signal is rising volatility.

**The methodological gap, stated exactly.** Its validation criterion is, in the paper's own words, "a
one-to-one correspondence between the local peaks of the indicators and the periods of financial
extreme volatility". That is in-sample peak matching. There is no volatility-matched null, no
chronological out-of-sample split, and no confusion matrix against the base rate. And its own
benchmark is volatility, which is the confound.

This repository has precisely the two tests that this design omits, already built and already run:
E1's matched-peak null and E8's chronological out-of-sample confusion matrix, across 23 markets,
2,900 constituents and 2.3 million minute bars.

### Idea 6: a volatility-matched null for recurrence-network indicators

**What to do.** Reimplement their two MRN indicators exactly, then push them through the existing
harness: (i) E1's null of ordinary market peaks not followed by a crash, (ii) E7's volatility-matched
pairing without replacement, (iii) E8's thresholds fixed on the early period and applied unchanged to
the late period, scored as a confusion matrix against a 0.37% base rate.

**The prediction.** Unmatched AUC near 0.9, collapsing toward 0.5 once volatility is held fixed,
because recurrence density is a monotone function of volatility. Same structure as E7's 0.911 to 0.526.

**Why this is a positive paper and not a complaint.** The deliverable is a **reusable protocol** with a
demonstrated failure mode, tested on 23 markets rather than 2. The field currently has no agreed null
for this class of indicator, and the paper supplies one. The title writes itself: *volatility-matched
nulls eliminate the apparent predictive power of co-movement-based crash indicators.*

### Idea 7: the variance decomposition, which is a genuinely positive empirical claim

E8 already found the strongest fact in the whole empirical half of this project and buried it as a
caveat: regressing co-movement on log volatility gives `R^2 = 0.414`, plain volatility **beats** the
model's own observable out of sample, and the volatility-orthogonal residual is the **worst** of three
detectors.

**The claim, stated positively.** Across 23 markets and every co-movement indicator in the
early-warning literature, a measurable fraction of the indicator is volatility, and the remainder
carries no out-of-sample skill. Report that fraction, indicator by indicator and market by market.
That is a positive, quantitative, citable result about what "collective panic" indicators actually
measure, and it is a much stronger statement than "we found no signal".

**What to compute.** Nearly free. The indicators, the markets and the regression already exist in
`e1_ews_all_markets.py` and `e8_prediction.py`. It needs a decomposition table and a figure.

---

## P5. Pang & Lin: market state transitions

**What the paper does.** Classical multidimensional scaling plus K-means on rolling constituent
return correlations gives five market states. Temporal features of the transitions between those
states feed a decision tree, reported at recall 96.77%, precision 96.36% and a false positive rate of
3.59%.

**The problem.** The evaluation is five-fold cross-validation in which "the dataset is randomly
partitioned into five". Their windows are 35, 40 and 6 days and they overlap, so random partitioning
puts adjacent and nearly identical windows in both the training and the test fold. That is textbook
temporal leakage. And 96% precision against a crash base rate well under one percent is not a
plausible number on its face: compare E8, which on a chronological split gets precision 0.0127.

### Idea 8: how much of a rolling-window financial ML result is leakage

**What to do.** Reproduce their pipeline exactly, then re-evaluate it three ways: their random
five-fold; blocked and purged cross-validation with an embargo of at least `max(L1, L2, L3)` days; and
a strict chronological split of the kind E8 uses. Report the precision at each.

**The claim.** A number for the leakage penalty in rolling-window financial machine learning. The
field asserts that this matters; nobody has measured it on a published result with the original
authors' own pipeline. Do it on all 23 markets, not one.

### Idea 9: the state-transition current. This is the best remaining empirical idea in the project

The market-state idea in P5 is good, and the repo can make it rigorous in a way that also rescues E2.

E2 tested for hysteresis by comparing a **scalar** co-movement mean on the way into a crash against the
way out, and found nothing. S9 then showed E2 never had the power: the model's own predicted gap is
0.0002 to 0.0028 against E2's detection bound of 0.013, roughly a factor of four too small to see.
A difference of two means is simply a weak test.

A bistable system with hysteresis does something much more distinctive than shifting a mean. It
traverses a **loop**, so its state-transition matrix carries a **net probability current** around that
loop. A non-hysteretic system, however volatile, satisfies detailed balance. This is Kolmogorov's
criterion for reversibility of a Markov chain, and testing it is standard non-equilibrium statistical
mechanics: estimate the transition matrix `P` between the five states, and test

```
P_ij P_jk P_ki  ==  P_ik P_kj P_ji     for every cycle (i, j, k)
```

against the alternative of a directed cycle through the crash state. The cycle affinity
`log( prod P_forward / prod P_backward )` is the entropy production per cycle, it is zero exactly for a
reversible chain, and its sign says which way the market goes round.

**Why this is powerful where E2 was not.** A transition-matrix asymmetry test uses the full sequence of
discrete states, not a difference of two conditional means, so it has far more power at the same sample
size. And it tests the *topological* signature of hysteresis, a loop, rather than a magnitude that S9
proved is below the noise floor.

**The claim.** Either markets show a statistically significant probability current around the crash
state, which is direct evidence of hysteresis in the observable that E2 could not reach, or they do
not, which is a much stronger negative result than E2's because the test finally has power. Either way
it is publishable, and either way it settles a question this project left open.

**What to compute.** Build the five-state classification on all 23 markets using the existing daily
panels in `research/empirical/data/markets/`, estimate `P` with a block bootstrap over calendar
quarters for the confidence intervals, and compute the cycle affinities. Then repeat on model output
with the coupling cycled across the bistable window, which is the positive control, and on
volatility-matched surrogates, which is the null. The empirical half is days of work because the data
is already fetched and cleaned.

**This is the idea I would bet on for a finance-facing paper**, because it takes a published method
(P5), fixes its evaluation (Idea 8), and turns it into a test with real power for the physical claim
this whole project is about.

---

## Cross-field connections worth making explicit

### Idea 10: Rician fading as the framing for the whole closure

`r_i` is literally a Rician random variable, coherent component plus scattered component, and the ratio
`R^2 / (1 - R^2)` is the Rician K-factor of wireless propagation. That is not decoration. The
engineering literature on **K-factor estimation from observed data** transfers directly: it gives
estimators for how far a real network sits from the mean-field limit, from the observed distribution of
local order parameters, with known small-sample behaviour. It also makes the result legible to a much
wider audience than the synchronization community, which matters for where it can be published.

### Idea 11: the identifiability result as a Fisher-information theorem

A3 is stated as an empirical failure: `R_eq(lam)` runs from 0.137 to 0.899 across a `lam` window only
0.167 wide, 97.2% of 27,723 market-days land inside that window, and any inversion is a monotone
transform of the indicator so it cannot beat it on a ranking test.

That is a theorem waiting to be written. The Fisher information for `lam` from an observation of `R` is

```
I(lam) = (dR_eq/dlam)^2 / Var(R)
```

which is enormous inside the window and essentially zero outside it. So a first-order transition is
identifiable from its order parameter **only** inside a parameter window whose width shrinks as the
transition sharpens: the sharper the transition, the less you can learn about what caused it. State it
as a proposition, give the width of the identifiable window as a function of the fold parameters, and
it applies to every bistable-model calibration, not just markets: ecological tipping points,
seizure-onset models, power-grid stability. It is a short, clean, general paper and this repository has
the worked example.

### Idea 12: k_eff as a systemic-risk measure that sidesteps A3's impossibility result

This is the idea that makes the theory operational, and I think it is the most underrated one here.

A3 proved you cannot measure `lam` from `R`. Fine. But S14's threshold does not depend only on `lam`:
it depends on `k_eff`, and `k_eff` is a **structural** quantity, estimated from the network and from
the distribution of local order parameters, not from the global order parameter. So it is not subject
to A3's argument at all.

Concretely: take the market network, measure the phase correlation `rho` inside neighbourhoods,
compute `k_eff = <k> / (1 + (<k>-1) rho)`, and compare it against `k_c(p)`. That single number says how
far the market is from the regime where a shock can leave it permanently repriced. The quantity you can
actually measure is not the coupling strength but the **effective connectivity**, and S14 says that is
the one that decides whether bistability exists at all.

Note also the subtlety E5's rerun exposed, which strengthens the case: for an Erdos-Renyi graph the
neighbourhood overlap is `<k>/n`, so a small random graph already has a reduced `k_eff`. This is why
72% of E5's reported collapse turned out to be finite size rather than clustering. `k_eff` is the
single number that absorbs both.

---

## Suggested order

1. **Finish S14** and write it as a Phys. Rev. E paper: the finite-connectivity law, the second
   threshold, the critical connectivity, the effective connectivity, validated on random regular,
   Erdos-Renyi and Watts-Strogatz graphs. This is the positive result and it is close to done.
2. **Idea 2** (sparsity kills the double jump) as the follow-up with Hens, since it extends P1 directly.
3. **Idea 5** (the two orthogonal topological mechanisms) as the companion, since it settles what S13
   got wrong about P3 and produces a dichotomy rather than a single number.
4. **Idea 9** (the state-transition current) as the finance paper, with Ideas 6, 7 and 8 as its methods
   section.
5. **Ideas 11 and 12** as a short note, or as the discussion of paper 1.

Ideas 3 and 4 are the most technically ambitious and the most likely to need a collaborator who already
works in the P2 framework.
