# Progress

Last updated 2026-10-13. Measured values are on `research/index.html` (sections P0, P1, B1) and in
`results/`; this file describes status only.

## Done

| phase | what | where |
|---|---|---|
| 0 | Audit spot checks reproduced; inventory, model spec, claims audit | `docs/INVENTORY.md`, `docs/MODEL_SPEC.md`, `docs/CLAIMS_AUDIT.md` |
| 1 | Package, compiled kernel, regression and analytic tests, run directories, CLI | `src/esfc/`, `tests/`, `configs/` |
| 2 (part) | Legacy layer retired; README rewritten with generated tables; CI on Python 3.12; dependencies; one Lorentzian sampler; sweep no longer mutates its input; S6 comment, theory naming, dangling references fixed | `legacy/`, `README.md`, `.github/workflows/ci.yml` |
| 3 | Prior-art search (web search, not full text) | `docs/PRIOR_ART.md` |
| 4 (part) | Leading-order theory with dynamic and reaction terms; directed-graph experiment (complete); undirected pilot (partial) | `docs/THEORY.md`, `src/esfc/cavity.py`, `results/paperA_*` |
| 5 (part) | Design fixed in advance; corrected minute-data study; corrected daily study | `docs/PREREGISTRATION_B.md`, `results/paperB_*` |
| pages | Results page sections P0, P1, B1; reference page steps 26 and 27; simulation with one-way and two-way networks, new readouts and block 10 | `research/index.html`, `research/reference.html`, `simulation/` |

## State of the main physics question

1. On directed graphs the Rician closure is contradicted and the new theory is supported, at pilot
   precision.
2. On undirected graphs the measured coefficient is below both predictions, nearer the closure's.
   The echo measured as undirected minus directed is clearly non-zero but smaller than predicted.
3. Open: whether the shortfall closes at larger k, which needs many more seeds than a workstation can
   run.

## Not done

1. **Undirected pilot**: interrupted; only the largest size ran. No finite-size extrapolation.
2. **Estimator convergence checks and all-to-all validation at scale**: configs exist
   (`configs/paperA_convergence.yaml`, `configs/paperA_A0_meanfield.yaml`), not run.
3. **Full grid** (30 seeds, four sizes, four exponents), low-connectivity bistability, clustering,
   heterogeneous graphs: need the cluster.
4. **Lee et al. replication** with band-pass filtering and survival analysis: needs their code and data.
5. **A3** against the same nulls as E1; Lorentzian reruns of S2 to S10; remaining pickle caches;
   hand-typed numbers in the older S and E sections of the results page.
6. **Timestamp for the pre-registration**: the file is not version controlled as a pre-registration.
7. **Manuscripts**: on hold by the author's decision.

## Known weaknesses to state in any write-up

- Minute-data study: 23 surviving large caps, 12 test crashes of which most fall in one month.
- The theory is leading order in 1/k and neglects loops, heterogeneity and clustering.
- The simulation page's one-way against two-way contrast is at different conditions from the theory.
