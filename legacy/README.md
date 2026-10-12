# Legacy layer (retired)

This directory holds the original C++17 and Python command-line layer of the project. It is kept so the
history stays auditable. **No scientific claim of the project rests on anything here.**

## Why it is retired

1. **It cannot show explosive synchronization.** Both engines normalise the coupling by node degree
   (`K/d_i`). `research/s1_reproduce_gg2011.py` measured a hysteresis gap of 0.06 for this form against
   0.79 for the unnormalised Gomez-Gardenes form. `--freq-mode degree-weighted` does not change that.
2. **Its "flash crash" has the wrong sign.** `flash_crash_sim.py` and `cpp/src/flash_crash.cpp` cut the
   coupling from K = 2.5 to 0.3. That desynchronises any Kuramoto network. A herding crash is the
   opposite: synchrony appearing, not disappearing.
3. **The Lyapunov analysis does not measure what it claims.** `python/analysis/lyapunov_analysis.py`
   draws the tangent vector from an unseeded generator and never projects out the neutral phase-shift
   direction `(1, ..., 1)`, so the "largest Lyapunov exponent" is pinned near zero in synchronised
   states. The earlier "onset of chaos" statement is withdrawn.

## Statements from the old README that are withdrawn

| statement | status |
|---|---|
| "~0.15s on the C++ backend" and the 30x speedup | Not reproducible; withdrawn |
| "Both layers normalize coupling ... guaranteeing deterministic cross-layer reproducibility" | False: the two engines use different integrators and different random number generators |
| "525 simulations" | The documented command and the Makefile pass 5 couplings, which is 125 runs |
| "95% coverage with 27 tests" | True only of this legacy layer. It never covered the research code |
| Convergence "within 2% of K_c" | Withdrawn |

## Stale outputs removed

`data/finite_size.csv` (old `n,k,order` header that broke two plot scripts) and `data/flash_crash.csv`
(showed no crash, unlike a fresh run of the documented command) were deleted.

## Running it anyway

From inside this directory: `make test`, `make data`, `make figures`. Paths in the Makefile are relative
to `legacy/`.
