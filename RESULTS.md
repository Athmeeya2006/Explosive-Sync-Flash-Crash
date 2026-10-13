# Paper A: first positive test of the finite-connectivity theory

Date: 2026-10-13. Repository state tested: commit `d49a0da` ("added some work"), unmodified.
Everything below was run with the repository's own `src/esfc` kernels (`FastSim`, `measure_fold`,
`directed_regular`, `make_network`) plus four short scripts in `scripts/`. Raw outputs are in `raw/`,
the summary table in `branch_summary.csv`. Units: gamma = 1, Lorentzian quantile frequencies clipped at
+/-60 (as in S14), dt = 0.02, Heun, no noise, every node adaptive with exponent p.

## 1. What was missing in the pushed work

`docs/THEORY.md` and `src/esfc/cavity.py` derive two corrections the Rician closure misses: a dynamic
term (drifting neighbours' scatter is filtered by the response rate mu) and a reaction term
(back-reaction on undirected graphs). The configs for testing them (A1 pilot, A7 directed, convergence,
A0) were written but none had results. `docs/PROGRESS.md` still says Phases 1 to 5 are not started,
although the package, tests and Paper B runs exist. That file needs updating.

## 2. Fold measurements: directed graphs separate the theories, but the fold is noisy

Coefficient a = k (lam_b / lam_b_inf - 1), p = 1, `measure_fold` with t_relax = 240, six seeds per
cell unless stated, +/- one standard error.

| k | N | directed | undirected |
|---|---|---|---|
| 12 | 4000 | 0.10 +/- 0.06 | 0.95 +/- 0.09 |
| 24 | 1000 | 0.24 +/- 0.23 | 0.90 +/- 0.19 |
| 24 | 4000 | 0.47 +/- 0.11 | 1.13 +/- 0.10 |
| 24 | 16000 (4 seeds) | 0.30 +/- 0.08 | 1.26 +/- 0.06 |
| 48 | 4000 | 0.34 +/- 0.15 | 1.30 +/- 0.16 |
| theory (k, N to infinity) | | a_dyn = 0.54 | a_dyn + a_react = 1.65 |
| Rician closure | | 1.50 | 1.50 |

Readings: on directed graphs the closure's 1.5 is excluded by roughly ten standard errors at every
k. The fold is a poor observable for a precision test: the per-seed scatter of a grows like k, the
undirected value still drifts upward with N, and the expansion is not analytic near the fold
(`reaction_coefficient_at_k` docstring: next term of order k^-1/2 relative). So the decisive test was
moved off the fold.

## 3. The decisive test: the synchronised branch above the fold

The second-order expansion of `docs/THEORY.md` section 3 holds at every point of the stable branch,
not only at the fold. Away from the fold the self-consistency gives a finite shift

```
k (R - R0)  ->  c(lam) = <a2> / (1 - z0'(h) lam (p+1) R0^p)
```

`scripts/branch_theory.py` evaluates this for the closure (all scatter frozen) and the dynamic theory;
`scripts/react_branch.py` evaluates the repository's reaction integral at the same branch point by
substituting branch constants into `cavity.reaction_parts`. Check: the generalised "closure" variant
reproduces the full Rician closure (`closure.self_consistency`) on the branch, -0.4245 against -0.4254
at k = 1600 (p = 1, lam = 1.25 lam_b_inf), and likewise at p = 2 and at 1.5 and 2.0 lam_b_inf.

Simulation (`scripts/branch_sim.py`): start fully synchronised at lam = 1.25 lam_b_inf, relax 150,
average R over the next 250. The reference R0 is the all-to-all kernel with the SAME frequencies and
N, so frequency clipping and finite-N effects of the frequency sample cancel (measured R0 = 0.77693
at N = 20000 against the analytic 0.77700).

| p | graph | k | N | seeds | c = k (R - R0) | dynamic theory | dynamic + reaction | closure |
|---|---|---|---|---|---|---|---|---|
| 1 | directed | 12 | 20000 | 4 | -0.279 +/- 0.018 | **-0.266** | | -0.425 |
| 1 | directed | 24 | 20000 | 4 | -0.267 +/- 0.019 | **-0.266** | | -0.425 |
| 1 | directed | 48 | 20000 | 4 | -0.266 +/- 0.013 | **-0.266** | | -0.425 |
| 1 | directed | 96 | 80000 | 2 | -0.246 +/- 0.053 | **-0.266** | | -0.425 |
| 2 | directed | 12 | 20000 | 2 | -0.221 +/- 0.009 | **-0.210** | | -0.340 |
| 2 | directed | 24 | 20000 | 2 | -0.190 +/- 0.042 | **-0.210** | | -0.340 |
| 1 | undirected | 12 | 20000 | 2 | -0.998 +/- 0.011 | | **-0.670** | -0.425 |
| 1 | undirected | 24 | 20000 | 2 | -0.787 +/- 0.028 | | **-0.670** | -0.425 |
| 1 | undirected | 24 | 80000 | 1 | -0.777 | | **-0.670** | -0.425 |
| 1 | undirected | 48 | 20000 | 2 | -0.678 +/- 0.030 | | **-0.670** | -0.425 |
| 1 | undirected | 96 | 20000 | 2 | -0.659 +/- 0.051 | | **-0.670** | -0.425 |

Findings:

1. **Directed graphs: the dynamic theory is confirmed with no free parameter**, at p = 1 for
   k = 12 to 96 and at p = 2 for k = 12 and 24. The Rician closure is excluded at every point.
2. **Undirected graphs: the data converge to the reaction-corrected prediction** as k grows
   (-1.00, -0.79, -0.68, -0.66 against -0.67), with large corrections at small k, as the theory's own
   non-analyticity argument anticipates. The reaction has the predicted sign and size.
3. **Finite-size control.** At k = 96, N = 20000 a single directed run gave -0.13; at N = 80000 the
   same k gives -0.25. Directed neighbourhoods overlap by about k^2/N nodes (0.46 at k = 96,
   N = 20000), which correlates neighbour fields. Keep k^2/N well below 1. At k = 24 the undirected
   value does not move between N = 20000 and 80000.
4. **Not yet resolved:** undirected k = 192 at N = 20000 gives -0.78 +/- 0.04 (k^2/N = 1.8, so the
   same overlap effect is suspected, but its sign differs from the directed case). Rerun at N >= 200000
   before quoting.

## 4. What this means for the paper

The Rician closure gets the right undirected number at the fold for the wrong reason: it freezes all
scatter (overestimating the dynamic effect) and ignores reaction (underestimating it), and on
undirected graphs the two errors partly cancel. Directed graphs remove the reaction and expose the
first error. That is a clean, falsifiable, physically transparent story, and section 3 is the first
direct numerical confirmation of it.

Working title: *Dynamic filtering and Onsager reaction set the finite-connectivity corrections to
explosive synchronization with adaptive coupling.*

## 5. Next runs, in order (16-core workstation estimates from timings here: one directed run at
N = 20000, k = 48 takes about 5 minutes on one core)

1. **Branch curve.** lam / lam_b_inf in {1.1, 1.25, 1.5, 2.0, 3.0} x p in {0.5, 1, 2, 3} x
   {directed, undirected} x k in {12, 24, 48, 96}, N = 20000 for k <= 48 and 80000 for k = 96,
   6 seeds. Theory curves c(lam) for all three treatments overlaid. This is the main figure.
2. **Finite size.** k = 48 and 96 at N in {20000, 80000, 320000}, 4 seeds, both graph types: fit
   the k^2/N correction so it can be stated, not just avoided.
3. **Fold from the theory.** With the branch theory validated, predict lam_b(k) at finite k by solving
   the second-order self-consistency without expanding about the all-to-all fold, and compare with the
   fold data of section 2 extended to N = 64000, 12 seeds. If the fold still disagrees, the paper
   says the 1/k series is asymptotic near the fold and quantifies where it breaks.
4. **Critical connectivity.** The closure's k_c(p) values (12.30, 7.23, 4.86, 4.25) were derived from
   the wrong closure. Recompute the window with the dynamic and reaction terms and measure bistability
   directly at k = 3 to 16 (hysteresis area and basin stability, 200 initial conditions).
5. **Clustering.** Repeat section 3 on Watts-Strogatz graphs: clustering should add correlated
   scatter AND change reaction loops, so k_eff = k/(1 + (k-1) rho) is now a hypothesis to test, not a
   result.

## 6. Paper B, from the pushed results

- Daily (E1c, 23 markets, 324 onsets): after quarter-block bootstrap on both sides and
  Benjamini-Hochberg over 42 tests, nothing survives, index variance included. Clean negative result.
- Intraday primary (3% in 60 minutes, first 30 minutes excluded): co-movement vs volatility-matched
  peaks AUC 0.602 [0.445, 0.757], 25 pairs. The minimum detectable AUC at this sample size is 0.698,
  so the primary test cannot detect the effect size it observes: it is underpowered by construction.
- Robustness variant (2% drop, 79 pairs): AUC 0.643 [0.552, 0.726], survives BH. Point estimates sit
  near 0.60 in every variant. This is a lead, not a finding: it is one of five robustness variants of
  a pre-registered primary that did not reject.
- Confirmation without forking paths: pre-register (before computing anything) the 2% definition as
  the confirmatory test on a DISJOINT cross-section, the tickers of the HF Data Library not among the
  23 used so far. Same events, independent stocks. If co-movement still beats volatility at matched
  volatility, that is a real result; if not, the 0.643 was noise.
- Volatility dominates operational alarms: 8.4 false alarms per catch at a 0.5% budget against 153
  for mean correlation. Keep this as the practical conclusion either way.

## Files

- `scripts/branch_theory.py`: branch shift, closure and dynamic variants, with self-check.
- `scripts/react_branch.py`: reaction term on the branch via `esfc.cavity.reaction_parts`.
- `scripts/branch_sim.py`: branch measurement (directed, undirected, all-to-all reference).
- `scripts/fold.py`: fold measurement used in section 2.
- `raw/*.jsonl`: every run, one JSON line each. `branch_summary.csv`: section 3 table.
