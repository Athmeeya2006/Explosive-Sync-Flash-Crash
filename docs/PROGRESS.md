# Progress

Last updated 2026-10-14. Measured values are on `research/index.html` (status card, sections P0, P1, P2, B1) and in
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
| 4 (part) | Cloud runs: fold on directed and undirected graphs at several sizes, and the branch-shift test; checked against the local runs | `results/paperA_cloud/`, `experiments/`, `docs/CLOUD_RUN_2026-10-13.md` |
| pages | Results page status card and sections P0, P1, P2, B1; reference page steps 26 to 28; simulation with one-way and two-way networks, new readouts and block 10 | `research/index.html`, `research/reference.html`, `simulation/` |

## State of the main physics question

1. **Directed graphs: settled at p = 1.** The shift of the synchronised branch matches the dynamic
   theory with no free parameter and excludes the Rician closure. Both sets of fold runs also exclude
   the closure.
2. **Directed graphs, fold coefficient:** measured values are consistently below the leading-order
   prediction. The leading term overestimates the fold shift at the connectivities reached.
3. **Undirected graphs: open.** The branch shift approaches the reaction-corrected prediction at
   intermediate k and departs at small k and at the largest k. Two seeds per cell. At the fold the
   closure's number is still the closer one.
4. **Cloud and local fold runs agree** within their combined uncertainty in every comparable cell.

## Next, in order

1. Branch curve on directed graphs: several couplings and exponents, 8 or more seeds, through a config.
2. Size scan at k = 48, 96, 192 on both graph types to measure the neighbourhood-overlap effect.
3. Finite-k reaction calculation on the branch.
4. Fold predicted from the finite-k self-consistency instead of the expansion.
5. Critical connectivity and clustering, re-derived with the corrected theory.
6. Theory scripts in `experiments/` to write a `summary.json`, so the branch predictions are stored
   results and not values copied from the run note.

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
- Cloud runs were launched by hand: no environment file or seed file, branch predictions copied from
  the run note, most undirected and all p = 2 branch cells have two seeds, one coupling only.
