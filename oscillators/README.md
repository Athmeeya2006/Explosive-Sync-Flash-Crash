# oscillators/: every oscillator family, early warnings, and real markets

The main project builds and tests one model, the adaptive (herding) Kuramoto model. This folder repeats
that pipeline for every oscillator family used here, and adds early-warning and market studies:

1. **Per-oscillator pipeline (O0 to O9).** For the simple harmonic oscillator, Stuart-Landau, van der Pol,
   FitzHugh-Nagumo, Kuramoto with and without inertia, Winfree, chaotic Roessler and theta neurons: what one
   unit does, where its oscillation is born, how a population synchronises or stops, what herding does to
   it, what exact theory and phase reduction predict, what shocks and circuit breakers do, what happens on
   networks, whether its coupling is separable, and how a local push spreads.
2. **Early warnings (Z1 to Z3).** Which indicator warns before which transition, and why.
3. **Real data (M0 to M9).** Do the indicators the oscillator picture favours warn before real crashes in
   22 stock markets and across 33 national indices, once the data are cleaned? Then a set of further
   predictions from oscillator theory (M5 to M9), each written down in the script's docstring before it was run.

Pages: [reference.html](reference.html) (every family built up from one unit, the counterpart of
`research/reference.html`), [index.html](index.html) (every result, explained), [demo.html](demo.html)
(interactive simulation of every family). Runs that need more than this workstation (16 cores, 15 GB RAM)
are in [ADA_COMPUTE.md](ADA_COMPUTE.md).

## Studies

| id | question | main-project counterpart |
|---|---|---|
| O0 | One unit of each family: time series, phase plane, and the bifurcation that switches it on, against exact formulas | reference steps 1 to 3 |
| O1 | Transition atlas: every family swept up and down with plain and herding coupling (p = 0, 1, 2); special transitions (amplitude death, Winfree death, theta bistability, single-unit Hopf points) against exact values | S1, S2, S11 |
| O3 | Phase reduction: each family's interaction function and its first Fourier coefficient a1, and the onset and herding fold they predict | S11, S13 |
| O3b | Stuart-Landau fold against amplitude relaxation rate | |
| O3c | Herding folds at three sweep speeds, extrapolated to an infinitely slow sweep | Z2 (slow passage) |
| O3d | Delay of the collapse after a slow ramp passes the fold, from the exact Ott-Antonsen equation, against Z2 and O3c | Z2 |
| O3e | Why the Stuart-Landau fold lies below phase reduction: exact mean-field self-consistency with amplitudes | |
| O4 | Shocks, circuit breakers (with a predicted minimum halt) and rate of change, every family with herding | S3, S4, S5, S7 |
| O5 | Frequency-degree correlation on Barabasi-Albert networks, every family | S1, S8 |
| O6 | Herding fold on sparse random regular networks against 1/k, every family | S14, Part D |
| O8 | Separability of every coupling (Hens et al. framework) | A1 |
| O9 | Propagation of a local push through Winfree and theta-neuron networks (the a2 test, Hens et al. 2019) | A2 |
| Z1 | Which indicator warns before which of 18 transitions (14, plus the herding collapse in four more families), and is the warning in the collective signal or in single units? | S6 |
| Z2 | Size of the warning before explosive desynchronisation against Ott-Antonsen theory; late collapse in slow ramps | S6, S7 |
| Z3 | When lag-1 autocorrelation warns before an oscillatory loss of stability (exact) | |
| M0 | Market data audit and cleaning | E1 data |
| M1 | Oscillator-motivated indicators before real crashes, design written before the run | E1c |
| M2 | Indicators against crash depth, speed, recovery and future drawdowns (exploratory) | |
| M3 | Phase synchrony across national indices before global crashes (exploratory) | |
| M4 | Index against single-stock variance (exploratory) | |
| M5 | Four hypotheses: synchrony in the 2 to 10 day band, frequency-degree correlation, contagion order, eigenvector localisation (exploratory) | S1, S8 |
| M5b | Contagion timing with scale-free measures (half-loss date, partial correlations, time zones) | |
| M5c | Is the negative frequency-degree sign mechanical? Volatility split into systematic and idiosyncratic parts | S8 |
| M5d | Contagion order with leave-one-out coupling, thresholds 10%, 7%, 5%, and dose-response | |
| M6 | Does synchrony lead volatility or follow it? (Granger-style, daily); M6b controls and placebo; M6c frequency bands | |
| M6d | Phase-synchrony and mean-correlation leads in one regression (check made after M6b) | |
| M11 | Flickering: bimodality and variance of daily breadth before crashes | |
| M10 | Does phase synchrony improve an out-of-sample HAR volatility forecast? | |
| M7 | The same lead inside the trading day (1-minute data) | |
| M8 | Frequency locking, lead-lag reciprocity, sectors locking first, phase offsets of locked units | |
| M8b | Is the sector-ratio fall before US crashes just the rise of mean correlation? (check made after M8) | |
| M9 | Systematic frequency-degree correlation as a crash indicator and against crash speed and depth | S8 |

## Claim to evidence

| claim | code | result file |
|---|---|---|
| Single-unit bifurcations reproduce sqrt(mu), 2 sqrt(mu), sqrt(eta)/pi and the FitzHugh-Nagumo Hopf point | `o0_anatomy.py` | `data/o0_bifurcations.csv` |
| Herding makes synchronisation explosive and bistable in every family; exact thresholds reproduced (amplitude death window, Winfree death, theta folds, Kuramoto onset and folds) | `o1_atlas.py`, `theory_zoo.py` | `data/o1_thresholds.csv`, `data/o1_special.csv`, `data/o3_exact.csv` |
| Atlas thresholds are converged in the time step (folds within 0.5%) | `o1_dt_check.py` | `data/o1_dt_check.csv` |
| Phase reduction: a1 = 1.000 (Stuart-Landau, exact value 1), 0.499 (van der Pol), 0.400 (FitzHugh-Nagumo); predicted onsets match; slow-sweep folds within a few percent for Kuramoto, inertia, van der Pol, Roessler; Stuart-Landau -10%, FitzHugh-Nagumo -19% | `o3_phase_reduction.py`, `o3c_rate_extrapolation.py` | `data/o3_reduction.csv`, `data/o3_predictions.csv`, `data/o3c_extrapolated.csv` |
| Faster amplitude relaxation moves the Stuart-Landau fold toward phase reduction | `o3b_amplitude.py` | `data/o3b_amplitude.csv` |
| Winfree and theta-neuron couplings are product-separable (rank 1); diffusive couplings rank 2; herding never pairwise additive | `o8_separability.py` | `data/o8_separability.csv` |
| The browser demo simulates the same models as zoo.py | `validate_demo.py`, `validate_demo.js` | `data/demo_validation.csv` |
| Collective variance warned before the first 14 transitions but fell before the FitzHugh-Nagumo herding collapse (AUC 0.03); every other indicator fails or reverses somewhere; single units often fall | `run_zoo.py`, `figures_zoo.py` | `data/z1_auc.csv` |
| Position autocorrelation of a forced mechanical oscillator falls at every lag below Nyquist as damping vanishes; velocity rises up to w dt = 2.03 | `z3_theory.py`, `z3_sampling.py` | `data/z3_theory.csv`, `data/z3_sampling.csv` |
| Ott-Antonsen branch exact to 0.4%; slow mode in low-frequency power; variance diluted; collapse after the fold with delay ~ speed^0.65 | `z2_fold_scaling.py`, `figures_zoo.py` | `data/z2_stationary.csv`, `data/z2_ramp.csv`, `data/z2_fits.json` |
| Cleaning rules and their effect on agreement with the official index | `marketdata.py`, `data_audit.py` | `data/m0_audit.csv` |
| No registered market test survives BH (0 of 18) | `m1_market_ews.py` | `data/m1_pooled_auc.csv`, `data/m1_summary.json` |
| No indicator anticipates drawdowns beyond variance; nothing relates to crash speed or recovery | `m2_correlations.py` | `data/m2_*.csv` |
| World synchrony does not anticipate global drawdowns | `m3_world_sync.py` | `data/m3_forward_corr.csv`, `data/m3_events.csv` |
| Index variance is not significantly better than single-stock variance | `m4_collective_vs_unit.py` | `data/m4_difference.csv` |
| Slow-passage delay after the fold follows the exact Ott-Antonsen equation with no fitted parameter | `o3d_slow_passage.py` | `data/o3d_slow_passage.csv` |
| Contagion order: national indices more correlated with the world index lose half of their fall sooner (10 of 11 global crashes, sign test p = 0.006); weaker within one time-zone band (8 of 11); only at the 10% threshold, fades for smaller drawdowns (M5d) | `m5b_contagion.py`, `m5d_contagion_robust.py` | `data/m5b_contagion.csv`, `data/m5d_contagion_robust.csv` |
| Frequency-degree sign is mechanical: positive with systematic volatility, negative with idiosyncratic volatility; the systematic version gives no crash warning and is unrelated to crash speed or depth (M9) | `m5c_fd_confound.py`, `m9_systematic_fd.py` | `data/m5c_fd_confound.csv`, `data/m9_*.csv` |
| Flickering of daily breadth: weak rise before crashes, at chance against market tops | `m11_flickering.py` | `data/m11_auc.csv` |
| Changes of Hilbert phase synchrony lead changes of volatility in 20 to 21 of 22 markets (partial r about 0.06, mostly after 2013, any band, unchanged with mean correlation controlled), but it does not improve an out-of-sample HAR volatility forecast (M10); inside the day only after 2013, and smaller | `m6_leadlag.py`, `m6b_granger_robust.py`, `m6c_band_sensitivity.py`, `m6d_joint.py`, `m7_intraday_leadlag.py`, `m10_har_forecast.py` | `data/m6*.csv`, `data/m7_intraday_leadlag.csv`, `data/m10_*.csv` |
| Frequency spread, reciprocity, sector ratio and phase offsets give no crash warning; the sector ratio falls before US crashes only because mean correlation rises (M8b) | `m8_more_hypotheses.py`, `m8b_cluster_check.py` | `data/m8_*.csv`, `data/m8b_cluster_check.csv` |
| Main-project fold harness exact all-to-all; estimator converged except for the frequency-sampling variant | `esfc run` | `results/paperA_A0/`, `results/paperA_convergence/` |

## Files

| file | what it does | writes |
|---|---|---|
| `zoo.py` | One numba kernel per model (mean-field populations and single units). Same slow ramp for every kernel; optional herding exponent, initial concentration and initial phases (defaults reproduce Z1 exactly). Frequency samplers | nothing (library) |
| `zoo_net.py` | Network versions of the population kernels (neighbour sums, local herding r_i^p, CSR graphs from `esfc.graphs`); `allconn` gives the exact complete graph at O(N) cost | nothing (library) |
| `theory_zoo.py` | Exact thresholds: Kuramoto branch and folds, amplitude-death window, Winfree death, FitzHugh-Nagumo Hopf point, theta-neuron Ott-Antonsen equation and its folds | nothing (library) |
| `ewslib.py` | Early-warning indicators, Kendall trend, ROC AUC with bootstrap CI | nothing (library) |
| `o0_anatomy.py` | **O0** | `figures/o0_*.png`, `data/o0_bifurcations.csv` |
| `o1_atlas.py` | **O1**; `special` as argument reruns only the special transitions | `data/o1_thresholds.csv`, `data/o1_curves.parquet`, `data/o1_special.csv` |
| `o1_dt_check.py` | **O1** check: every sweep threshold with the time step halved | `data/o1_dt_check.csv` |
| `o3_phase_reduction.py` | **O3**: adjoint phase sensitivity, interaction functions, predicted thresholds, table of exact thresholds | `data/o3_reduction.csv`, `data/o3_predictions.csv`, `data/o3_interaction.json`, `data/o3_exact.csv` |
| `o3b_amplitude.py` | **O3b** | `data/o3b_amplitude.csv` |
| `o3c_rate_extrapolation.py` | **O3c**; `analyse` refits from saved runs | `data/o3c_rate.csv`, `data/o3c_extrapolated.csv` |
| `o3d_slow_passage.py` | **O3d** | `data/o3d_slow_passage.csv` |
| `o3e_sl_selfconsistency.py` | **O3e** | `data/o3e_sl_selfconsistency.csv` |
| `o4_crash_protocols.py` | **O4**; `agreement` recomputes the breaker agreement table | `data/o4_*.csv` |
| `o5_frequency_degree.py` | **O5** | `data/o5_curves.parquet`, `data/o5_summary.csv` |
| `o6_sparse_fold.py` | **O6** | `data/o6_sparse_fold.csv`, `data/o6_coefficients.csv` |
| `o8_separability.py` | **O8** (uses `research/analytics/a1_separability.py`) | `data/o8_separability.csv` |
| `o9_signal_propagation.py` | **O9** (uses the a2 machinery of `research/analytics/`) | `data/o9_tau.csv`, `data/o9_fits.csv` |
| `figures_atlas.py` | Figures for O1, O3, O3b, O3c, O4, O5, O6 | `figures/o1_*.png` to `figures/o6_*.png` |
| `run_zoo.py` | **Z1**. Grid from `ZOO_N`, `ZOO_SIGMA`, `ZOO_TSCALE`, `ZOO_TAG`; `--only name` reruns one scenario | `data/z1_scores*.parquet`, `data/z1_runs*.parquet`, `data/z1_examples*.npz` |
| `z2_fold_scaling.py` | **Z2**; `stationary` or `ramp` runs one stage. Grid from `Z2_SIZES`, `Z2_T_SIZE_SCAN`, `Z2_RATE_N`, `Z2_TRAMPS`, `Z2_RUNS`, `Z2_SIGMA`, `Z2_TAG` | `data/z2_stationary*.csv`, `data/z2_ramp*.csv` |
| `z3_theory.py`, `z3_sampling.py` | **Z3** theory (SymPy, Lyapunov check) and simulation | `data/z3_*.csv` |
| `figures_zoo.py` | Tables and figures for Z1 to Z3; `z1tag=_s0.05` writes the AUC table of a grid run | `data/z1_auc*.csv`, `data/z2_fits.json`, `figures/z*.png` |
| `marketdata.py` | Daily market panels with cleaning rules R1 to R6 | nothing (library) |
| `data_audit.py` | **M0** | `data/m0_audit.csv` |
| `DESIGN_MARKETS.md` | The M1 design, written before M1 was run, with one recorded deviation | |
| `m1_market_ews.py` | **M1**; `M1_WORKERS` sets the number of processes (default 2) | `data/m1_*` |
| `m2_correlations.py`, `m3_world_sync.py`, `m4_collective_vs_unit.py` | **M2**, **M3**, **M4** | `data/m2_*`, `data/m3_*`, `data/m4_*` |
| `m5_hypotheses.py` | **M5** | `data/m5_auc.csv`, `data/m5_fd_summary.csv`, `data/m5_contagion.csv` |
| `m5b_contagion.py` | **M5b** | `data/m5b_contagion.csv`, `data/m5b_contagion_events.csv` |
| `m5d_contagion_robust.py` | **M5d** | `data/m5d_contagion_robust.csv`, `data/m5d_events.csv` |
| `m5c_fd_confound.py` | **M5c** | `data/m5c_fd_confound.csv` |
| `m6_leadlag.py` | **M6** | `data/m6_leadlag.csv`, `data/m6_granger.csv` |
| `m6b_granger_robust.py` | **M6b**; its `per_market` and `pool` functions are reused by M6c and M8 | `data/m6b_granger_robust.csv` |
| `m6c_band_sensitivity.py` | **M6c** | `data/m6c_band_sensitivity.csv` |
| `m6d_joint.py` | **M6d** | `data/m6d_joint.csv` |
| `m11_flickering.py` | **M11** | `data/m11_auc.csv`, `data/m11_indicators/<market>.parquet` |
| `m10_har_forecast.py` | **M10** | `data/m10_har_oos.csv`, `data/m10_har_pooled.csv`, `data/m10_har_coefs.csv` |
| `m7_intraday_leadlag.py` | **M7** (minute data of the main project) | `data/m7_blocks.parquet`, `data/m7_intraday_leadlag.csv` |
| `m8_more_hypotheses.py` | **M8** | `data/m8_auc.csv`, `data/m8_granger.csv`, `data/m8_offsets.csv` |
| `m8b_cluster_check.py` | **M8b** (needs the M1 and M8 caches) | `data/m8b_cluster_check.csv` |
| `m9_systematic_fd.py` | **M9** | `data/m9_auc.csv`, `data/m9_severity.csv` |
| `figures_markets.py` | Figures for M0 to M3 | `figures/m*.png` |
| `demo_models.js` | The models of the browser demo (same equations as zoo.py) | |
| `build_demo.py` | Writes `demo.html`, with each family's thresholds inlined | `demo.html` |
| `validate_demo.js`, `validate_demo.py`, `validate_demo_cases.json` | Checks the demo against zoo.py (statistical comparison of time-averaged order parameters) | `data/demo_validation.csv` |
| `build_reference.py` | Writes `reference.html` from the result files | `reference.html` |
| `build_index.py` | Writes `index.html` from the result files; fails on a missing figure or a dash character | `index.html` |
| `ADA_COMPUTE.md` | Runs that need the cluster, with measured costs; what is limited by data rather than compute | |

Tests: `tests/test_oscillators.py` in the repository root (estimators, kernels against exact variances and
the Ott-Antonsen branch, exact reproduction of default kernels, network kernels against mean field, the
complete-graph mode, cleaning rules).

## Folders

| folder | contents |
|---|---|
| `data/` | every result table (CSV, parquet, JSON) |
| `data/logs/` | the console output of every run, including the two main-project configs run from here |
| `data/m1_indicators/`, `data/m4_indicators/`, `data/m5_indicators/`, `data/m8_indicators/`, `data/m9_indicators/`, `data/m11_indicators/` | per-market indicator caches; delete them to recompute |
| `data/downloads/` | `world_extra.pkl`: a re-download attempt of world indices missing on Yahoo, kept as the record that they are not available there; `us_gics_sectors.csv`: GICS sector of 1506 US tickers (S&P 500, 400, 600 lists on Wikipedia), used by M8 |
| `figures/` | every figure shown on the pages |
| `ada/` | SLURM scripts for the cluster runs in `ADA_COMPUTE.md` (account and partition to fill in) |

## Run order

```
cd oscillators
../research/.venv/bin/python o0_anatomy.py
../research/.venv/bin/python o1_atlas.py               # about 20 min on 8 cores
../research/.venv/bin/python o1_dt_check.py
../research/.venv/bin/python o3_phase_reduction.py     # needs o1_atlas (frequencies)
../research/.venv/bin/python o3b_amplitude.py
../research/.venv/bin/python o3c_rate_extrapolation.py # about 1 h on 8 cores
../research/.venv/bin/python o3d_slow_passage.py       # needs o3c and z2
../research/.venv/bin/python o3e_sl_selfconsistency.py # needs o3c
../research/.venv/bin/python o4_crash_protocols.py     # needs o3c
../research/.venv/bin/python o5_frequency_degree.py
../research/.venv/bin/python o6_sparse_fold.py         # needs o1_atlas
../research/.venv/bin/python o8_separability.py
../research/.venv/bin/python o9_signal_propagation.py
../research/.venv/bin/python figures_atlas.py
../research/.venv/bin/python run_zoo.py                # Z1, 27 min on 14 cores
../research/.venv/bin/python z3_theory.py
../research/.venv/bin/python z3_sampling.py
../research/.venv/bin/python z2_fold_scaling.py
../research/.venv/bin/python figures_zoo.py
../research/.venv/bin/python data_audit.py             # M0
../research/.venv/bin/python m1_market_ews.py          # M1
../research/.venv/bin/python m2_correlations.py
../research/.venv/bin/python m3_world_sync.py
../research/.venv/bin/python m4_collective_vs_unit.py
../research/.venv/bin/python m5_hypotheses.py          # M5 to M9 need m1
../research/.venv/bin/python m5b_contagion.py
../research/.venv/bin/python m5c_fd_confound.py
../research/.venv/bin/python m5d_contagion_robust.py
../research/.venv/bin/python m6_leadlag.py
../research/.venv/bin/python m6b_granger_robust.py
../research/.venv/bin/python m6c_band_sensitivity.py
../research/.venv/bin/python m6d_joint.py
../research/.venv/bin/python m7_intraday_leadlag.py
../research/.venv/bin/python m8_more_hypotheses.py
../research/.venv/bin/python m8b_cluster_check.py
../research/.venv/bin/python m9_systematic_fd.py
../research/.venv/bin/python m10_har_forecast.py
../research/.venv/bin/python m11_flickering.py
../research/.venv/bin/python figures_markets.py
../research/.venv/bin/python validate_demo.py          # needs node
../research/.venv/bin/python build_demo.py
../research/.venv/bin/python build_reference.py
../research/.venv/bin/python build_index.py
```

Process pools use all cores but two on the workstation, or `SLURM_CPUS_PER_TASK` on the cluster. Market
prices are not stored in the repository ([../docs/DATA_LICENSES.md](../docs/DATA_LICENSES.md)); they come
from `research/empirical/fetch_markets.py`.
