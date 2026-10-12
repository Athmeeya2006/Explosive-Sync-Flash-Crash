# Progress

## Phase 0: reproduce the audit, inventory, audit of claims. Complete (2026-10-13)

**What changed.** New: `docs/INVENTORY.md`, `docs/MODEL_SPEC.md`, `docs/CLAIMS_AUDIT.md` (generated),
`docs/DECISIONS.md`, `research/p0_spot_checks.py`, `research/data/p0_spot_checks.json`. Updated pages:
`research/index.html` (section P0), `research/reference.html` (step 26), `simulation/index.html`
(block 0). No simulation or analysis code was modified.

**What was measured.** The audit's three spot checks, with the repository's unmodified functions.
All measured values are in `docs/CLAIMS_AUDIT.md`. In words: the closure's expansion is correct; the
bisection harness is unbiased in the all-to-all limit; on random regular graphs the simulated fold is
below the closure in every run. The stopping condition for Phase 0 (numbers differing from the audit
by more than the stated seed scatter) was not triggered.

**What is still open.**
1. One run (long relaxation time) differs from the audit by more than the bisection tolerance, though
   far less than the seed scatter. Probably library-version sensitivity of a long chaotic run. To be
   checked when the regression tests exist.
2. Near the fold at k = 12 the surviving state has synchrony close to the estimator's survival
   threshold. Threshold and relaxation-time sensitivity must be mapped before quoting a coefficient.
3. The audit's file-by-file findings on the empirical layer (E1, E3, E7, E8, A3) were confirmed by
   reading the code and the committed tables, not by re-running: the market data is not in the
   repository.
4. Compute: one bisection at N = 19200 took about 15 minutes with 13 jobs sharing this 16-core machine.
   The full Paper A grid is not feasible here without a faster backend or a cluster.

## Not started

Phase 1 (package, tests, pipeline), Phase 2 (defect fixes, legacy move), Phase 3 (prior-art search),
Phase 4 (Paper A pilot, cavity solver), Phase 5 (Paper B pre-registration and runs). Phase 6
(manuscripts) is on hold by the author's decision.
