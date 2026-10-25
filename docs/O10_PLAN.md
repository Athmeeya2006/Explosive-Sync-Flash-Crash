# O10: the main project's sparse-graph checks for every oscillator family

Script: `oscillators/o10_family_pipeline.py` (stages `validate`, `fold`, `branch`, `hetero`, `window`, `all`).
Jobs: `oscillators/ada/o10_*.sbatch` (1 GPU, 10 CPUs, 24G each). Outputs: `oscillators/data/o10_*.csv`.

## Question

The main project measured four finite-connectivity effects for the adaptive (herding) Kuramoto model,
coupling (lambda/k) r_i^p sum_j A_ij sin(theta_j - theta_i), p = 1:

1. branch shift c = k (R - R0) at 1.25, 2, 3 times the all-to-all fold (undirected about -0.21 at 2x, directed about -0.09);
2. fold coefficient k (lambda_b(k)/lambda_b(inf) - 1) (about 1.2 to 1.4 undirected, about 0.4 directed, k = 24 to 48);
3. Erdos-Renyi against random regular at the same mean degree (0.34 against 1.07 at k = 12; degree-based mean field: lower by 3/8 CV^2 k = 3/8 for Poisson degrees);
4. the bistable window: forward threshold from a random start against the fold.

Phase reduction maps every limit-cycle family onto a Kuramoto model whose coupling is multiplied by the
family's reduction factor a1. Every quantity above is a ratio of couplings or is taken at a fixed multiple of the
family's own all-to-all fold, so a1 cancels: every family should reproduce the Kuramoto numbers. Deviations locate
where amplitude or relaxation dynamics break phase reduction.

## Design

- Families: Stuart-Landau (mu = 1), van der Pol (eps = 1), Roessler, FitzHugh-Nagumo, and Kuramoto as the control,
  all through the same GPU kernel `zoo_net_gpu.run` (same integrator, same noise). Frequencies, time steps and noise
  as in O6 (`FREQ`, `DT`, SIG = 0.02). Local herding r_i^p with p = 1, scale 1/k (nominal degree) on graphs and
  1/(N - 1) on the complete graph.
- Fold by a coupling grid instead of bisection. Constant coupling, synchronised start (r0 = 1), relaxation
  T_relax = 500 time units, survival if the mean of R over the last quarter exceeds a threshold. Grid in three passes,
  every realisation refined in parallel: 13 points over [0.85, 1.45] x reference (step 0.05), then 9 points inside the
  bracket (step 0.005), then 9 more (step 0.0005); the bracket is extended if the fold lies outside the first grid.
  Fold = lowest surviving coupling. Survival monotonicity is checked on every realisation: a violation is a failing
  coupling above the lowest surviving one; violations are counted and reported.
- Survival threshold per family: 0.55 x R_b, where R_b is the complete-graph R at its own fold (the main project
  uses 0.55 x r*(p), the mean-field R at the fold). Incoherent states have R of order N^(-1/2) = 0.01, so the
  threshold sits in the gap of a bimodal distribution; the reference pass uses a provisional 0.25 to locate R_b.
- Reference: the complete graph with the same N, the same frequencies and the same grid (4 noise realisations),
  computed inside every stage at every N it uses. ratio = fold(graph)/fold(complete), coefficient k (ratio - 1).
  With noise 0.02 the Kuramoto control differs slightly from the noiseless main project; the ratio uses the reference
  under the same noise.
- Branch: constant coupling 1.25, 2, 3 x the complete-graph fold, synchronised start, burn-in 200, R averaged over
  the next 300 time units; c = k (R - R0) with R0 from the complete graph at the same coupling (4 realisations).
  Directed and undirected random regular, k = 12, 24 (N = 20000), 48 (N = 50000), 8 realisations.
- Fold stage: directed and undirected, k = 12, 24 (N = 10000), 48 (N = 50000), 4 realisations.
- Heterogeneity: Erdos-Renyi (giant component, realisations with fewer than N nodes are redrawn) against random
  regular, k = 12, 24, N = 10000, 4 realisations.
- Window: fold (sync start) and forward threshold (random start r0 = 0, grid 14 points log spaced over
  [0.9, 5] x reference, then two refining passes) at k = 6, 12, 24, directed and undirected, N = 10000, 4 realisations.
- Batches are cut at 2e6 nodes (`O10_MAXNODES`). Graphs are built on the CPU pool (SLURM_CPUS_PER_TASK workers).
- Validation before production (`validate` stage): (a) Kuramoto control, k = 24, undirected, coefficient near
  1.2 to 1.4; van der Pol on the same cell; (b) survival monotone in lambda; (c) Kuramoto cell rerun with
  T_relax = 1000.

## Validation results

(filled in below after the validation job)
