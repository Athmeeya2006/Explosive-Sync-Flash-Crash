# Progress

Last updated 2026-10-25. Measured values are on `research/index.html` (status card, sections P0, P1, P2, B1) and in
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

| osc | Early-warning benchmark across 14 transitions in 11 oscillator models; exact lag-1 autocorrelation result for forced mechanical oscillators; warning before explosive desynchronisation against Ott-Antonsen theory; market data audit and cleaning; registered market test of three oscillator indicators (negative); exploratory market correlations | `oscillators/`, results on `oscillators/index.html` |
| osc | The main project's pipeline repeated for every oscillator family (O0 to O8): single units against exact bifurcations; transition atlas with and without herding against exact thresholds; phase reduction (a1 per family) and its predicted onsets and folds, checked after removing the sweep-speed delay; parameter-free prediction of the collapse delay; shocks, circuit breakers and rate; frequency-degree correlation and sparse networks; separability; reference page, interactive demo validated against the kernels, references added to `references/11_oscillator_zoo/` | `oscillators/reference.html`, `oscillators/demo.html`, `oscillators/README.md` |
| cluster | GPU path for the cluster (fold, branch and window experiments; oscillator zoo); validated against the CPU; reciprocity, branch curve, clustering and heterogeneity runs; oscillator items A8 to A12 | `docs/ADA_RUN_2026-10-24.md`, `docs/DECISIONS.md` items 16 to 18 |
| 4 (part) | All-to-all validation (`configs/paperA_A0_meanfield.yaml`) and estimator convergence (`configs/paperA_convergence.yaml`) run on the workstation | `results/paperA_A0/`, `results/paperA_convergence/` |

## State of the main physics question

1. **Directed graphs: agreement to within about 5 to 10%.** The cloud run (one coupling, p = 1) matched
   the dynamic theory; the GPU branch curve (2026-10-24: five couplings, p = 1 and 2, 10 seeds, k = 12 to
   48) finds the raw shift about 10% beyond it at every k and 6% beyond it on average after
   extrapolating in 1/k (8 of 10 cells within about 2 standard errors), one cell clearly off (p = 1 at 3 times the fold). The Rician closure is excluded
   everywhere (`docs/ADA_RUN_2026-10-24.md`).
2. **Directed graphs, fold coefficient:** measured values are consistently below the leading-order
   prediction. The leading term overestimates the fold shift at the connectivities reached.
3. **Undirected graphs: converging on the reaction-corrected theory, not yet settled near the fold.**
   With 10 seeds (GPU branch curve and reciprocity runs, 2026-10-24) the shift approaches the prediction
   as k grows at every coupling; extrapolated to large k it agrees to 1 to 3% at 2 and 3 times the fold
   and falls 5 to 9% short at 1.25 and 1.5 times it. The A4 size scan (k = 96, 192) decides the region
   near the fold. The fold itself is being measured to k = 96 (`paperA_fold_kscan_gpu`).
4. **Cloud and local fold runs agree** within their combined uncertainty in every comparable cell.

## Next, in order

1. ~~Branch curve on directed graphs~~: done on the GPU for both graph types (`paperA_branch_curve_gpu`).
   New question from it: the residual 5 to 10% excess on directed graphs (a missing leading-order term?).
2. Size scan at k = 48, 96, 192 on both graph types to measure the neighbourhood-overlap effect: running
   (`paperA_branch_ada_gpu`), with the fold to k = 96 (`paperA_fold_kscan_gpu`) and the bistable window at
   k = 3 to 24 (`paperA_window_gpu`).
3. Finite-k reaction calculation on the branch.
4. Fold predicted from the finite-k self-consistency instead of the expansion.
5. Critical connectivity and clustering, re-derived with the corrected theory. Measured clustering and
   degree-heterogeneity effects now exist to compare with (`paperA_structure_gpu`); heterogeneity lowers the
   fold strongly, which a heterogeneous mean field should capture.
6. Theory scripts in `experiments/` to write a `summary.json`, so the branch predictions are stored
   results and not values copied from the run note.

## Not done

1. **Undirected pilot**: interrupted; only the largest size ran. No finite-size extrapolation.
2. **All-to-all validation at scale** (N up to 1e5, 10 seeds): the workstation-sized versions of the
   convergence and all-to-all configs have now run (see Done); the larger grid is in `oscillators/ADA_COMPUTE.md`.
3. **Full grid** (30 seeds, four sizes, four exponents), low-connectivity bistability, clustering,
   heterogeneous graphs: need the cluster (item A5 of `oscillators/ADA_COMPUTE.md`).
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
