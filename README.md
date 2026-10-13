# Explosive synchronization with adaptive coupling on sparse networks

Research code for one question in nonlinear dynamics and one in empirical finance.

1. **Physics.** Kuramoto oscillators whose coupling is scaled by a power p of the *local* order
   parameter (the adaptive model of Zhang, Boccaletti, Guan and Liu, PRL 114, 038701 (2015)) have a
   bistable window between a backward threshold (the fold) and a forward threshold. In the all-to-all
   limit the fold is known exactly (Zou and Wang, PRE 102, 012219 (2020)). This repository studies how
   the fold moves when each oscillator has only k neighbours.
2. **Finance.** If a market crash is modelled as such a synchronization transition, does synchrony
   measured from prices warn of crashes? This is tested on real data with a design fixed before the corrected run.

**Status: one result established, the rest a pilot. No manuscript exists.** An external audit at
commit `fc98f02` found that several headline claims of the earlier README were not backed by committed
data. Those claims are withdrawn or marked ([docs/CORRECTIONS.md](docs/CORRECTIONS.md),
[docs/CLAIMS_AUDIT.md](docs/CLAIMS_AUDIT.md)). Since then one prediction has been confirmed: on directed
graphs the shift of the synchronised branch matches the dynamic theory and excludes the earlier Rician
closure. The undirected case is open. See the status card at the top of
[research/index.html](research/index.html).

## Start here

| what | where |
|---|---|
| All results, with every table explained | [research/index.html](research/index.html) (status card, then sections P0, P1, P2, B1) |
| The model built up from one oscillator, and what the audit changed | [research/reference.html](research/reference.html) (steps 26 to 28) |
| Interactive demonstration (not evidence) | [simulation/index.html](simulation/index.html) |
| Derivation of the finite-connectivity theory | [docs/THEORY.md](docs/THEORY.md) |
| Cloud runs and how they were checked | [docs/CLOUD_RUN_2026-10-13.md](docs/CLOUD_RUN_2026-10-13.md), results page section P2 |
| Which claim rests on which file | [docs/CLAIM_EVIDENCE.md](docs/CLAIM_EVIDENCE.md) |
| Prior work and how this differs | [docs/PRIOR_ART.md](docs/PRIOR_ART.md) |
| What is done and what is open | [docs/PROGRESS.md](docs/PROGRESS.md) |

## Current results

Everything between the markers is written by `esfc audit` from `results/**/summary.json`. Do not edit
it by hand.

<!-- BEGIN GENERATED -->
### Predicted 1/k coefficient of the fold

`fold = all-to-all fold x (1 + a/k)`. Rician closure: all neighbour scatter frozen, no back-reaction. Directed: rotating scatter filtered, no back-reaction. Undirected: directed plus the reaction term. Derivation in `docs/THEORY.md`.

| p | Rician closure | theory, directed | reaction term | theory, undirected |
|---|---|---|---|---|
| 0.5 | 1.250 | 0.331 | 0.958 | 1.289 |
| 1 | 1.500 | 0.537 | 1.117 | 1.654 |
| 1.5 | 1.750 | 0.715 | 1.293 | 2.008 |
| 2 | 2.000 | 0.881 | 1.484 | 2.366 |
| 3 | 2.500 | 1.198 | 1.876 | 3.073 |

### Measured fold at the largest size run (pilot)

Bisection from a synchronised start, relaxation time 480, tolerance 0.2%, Lorentzian quantile frequencies. Coefficient is `k (ratio - 1)`. Both predictions are leading-order and are expected to fail at small k.

| graph | p | k | N | seeds | fold ratio [95% CI] | closure ratio | coefficient [95% CI] | theory | Rician |
|---|---|---|---|---|---|---|---|---|---|
| directed | 1 | 24 | 10000 | 6 | 1.0152 [1.0064, 1.0241] | 1.0612 | 0.37 [0.15, 0.58] | 0.54 | 1.50 |
| directed | 1 | 48 | 10000 | 6 | 1.0090 [1.0066, 1.0114] | 1.0310 | 0.43 [0.32, 0.54] | 0.54 | 1.50 |
| directed | 2 | 24 | 10000 | 6 | 1.0299 [1.0236, 1.0363] | 1.0843 | 0.72 [0.57, 0.87] | 0.88 | 2.00 |
| directed | 2 | 48 | 10000 | 6 | 1.0164 [1.0121, 1.0206] | 1.0419 | 0.78 [0.58, 0.99] | 0.88 | 2.00 |
| undirected | 1 | 16 | 10000 | 5 | 1.0750 [1.0661, 1.0839] | 1.0904 | 1.20 [1.06, 1.34] | 1.61 | 1.50 |
| undirected | 1 | 24 | 10000 | 5 | 1.0504 [1.0415, 1.0594] | 1.0612 | 1.21 [1.00, 1.42] | 1.62 | 1.50 |
| undirected | 1 | 32 | 10000 | 5 | 1.0399 [1.0324, 1.0474] | 1.0462 | 1.28 [1.04, 1.52] | 1.63 | 1.50 |
| undirected | 1 | 48 | 10000 | 5 | 1.0292 [1.0254, 1.0330] | 1.0310 | 1.40 [1.22, 1.59] | 1.64 | 1.50 |
| undirected | 2 | 16 | 10000 | 3 | 1.1156 [1.0952, 1.1360] | 1.1271 | 1.85 [1.52, 2.18] | 2.25 | 2.00 |
| undirected | 2 | 24 | 10000 | 5 | 1.0789 [1.0716, 1.0862] | 1.0843 | 1.89 [1.72, 2.07] | 2.27 | 2.00 |
| undirected | 2 | 32 | 10000 | 5 | 1.0527 [1.0423, 1.0632] | 1.0630 | 1.69 [1.35, 2.02] | 2.29 | 2.00 |
| undirected | 2 | 48 | 10000 | 5 | 1.0400 [1.0318, 1.0482] | 1.0419 | 1.92 [1.53, 2.31] | 2.30 | 2.00 |

### Shift of the synchronised branch (cloud runs)

`c = k (R - R0)` at 1.25 times the all-to-all fold, `R0` from the all-to-all model on the same frequencies. Theory is the dynamic theory on directed graphs and dynamic plus reaction on undirected graphs; the theory columns were printed by the cloud run and have not been regenerated locally. Directed graphs, p = 1, k = 12 to 48 pooled: -0.270 +/- 0.009. With two seeds the 95% interval is very wide; those cells are not confirmations. Full discussion: `research/index.html` section P2.

| graph | p | k | N | seeds | c [95% CI] | overlap k^2/N | theory | Rician |
|---|---|---|---|---|---|---|---|---|
| directed | 1 | 12 | 20000 | 4 | -0.279 [-0.335, -0.223] | 0.01 | -0.266 | -0.425 |
| directed | 1 | 24 | 20000 | 4 | -0.267 [-0.328, -0.207] | 0.03 | -0.266 | -0.425 |
| directed | 1 | 48 | 20000 | 4 | -0.266 [-0.306, -0.226] | 0.12 | -0.266 | -0.425 |
| directed | 1 | 96 | 20000 | 1 | -0.126 | 0.46 | -0.266 | -0.425 |
| directed | 1 | 96 | 80000 | 2 | -0.246 [-0.917, 0.426] | 0.12 | -0.266 | -0.425 |
| directed | 2 | 12 | 20000 | 2 | -0.221 [-0.338, -0.105] | 0.01 | -0.210 | -0.340 |
| directed | 2 | 24 | 20000 | 2 | -0.190 [-0.724, 0.344] | 0.03 | -0.210 | -0.340 |
| undirected | 1 | 12 | 20000 | 2 | -0.998 [-1.138, -0.857] | 0.01 | -0.670 | -0.425 |
| undirected | 1 | 24 | 20000 | 2 | -0.787 [-1.143, -0.430] | 0.03 | -0.670 | -0.425 |
| undirected | 1 | 24 | 80000 | 1 | -0.777 | 0.01 | -0.670 | -0.425 |
| undirected | 1 | 48 | 20000 | 2 | -0.678 [-1.061, -0.295] | 0.12 | -0.670 | -0.425 |
| undirected | 1 | 96 | 20000 | 2 | -0.659 [-1.309, -0.008] | 0.46 | -0.670 | -0.425 |
| undirected | 1 | 192 | 20000 | 2 | -0.782 [-1.321, -0.243] | 1.84 | -0.670 | -0.425 |

### Crash warnings on minute data (corrected; design written before the run, not timestamped)

23 stocks, 5972 sessions, 35 intraday crash onsets (12 in the test period after 2016). Primary hypothesis, synchrony beyond volatility (`corr_given_vol` against volatility-matched peaks): AUC 0.587 [0.432, 0.746], corrected p = 0.143; smallest detectable AUC at this sample size 0.70. Not detected, and underpowered.

| indicator | budget | realised rate | caught | false episodes | false per catch |
|---|---|---|---|---|---|
| volatility | 1% | 0.41% | 9 of 12 | 105 | 12 |
| volatility | 5% | 2.15% | 11 of 12 | 575 | 52 |
| mean_corr | 1% | 1.32% | 4 of 12 | 749 | 187 |
| mean_corr | 5% | 4.69% | 9 of 12 | 2127 | 236 |
| lead_eig_share | 1% | 1.33% | 4 of 12 | 769 | 192 |
| lead_eig_share | 5% | 4.65% | 8 of 12 | 2154 | 269 |
| corr_given_vol | 1% | 1.33% | 2 of 12 | 931 | 466 |
| corr_given_vol | 5% | 5.86% | 2 of 12 | 2946 | 1473 |
<!-- END GENERATED -->

## What is reproduction and what might be new

| item | status |
|---|---|
| All-to-all fold `lam_b = gamma (p+2) ((p+2)/p)^(p/2)`, amplitude `R* = sqrt(p/(p+2))` | Reproduction of Zou and Wang (2020), `alpha = p + 1` |
| `p = 2` corner as a three-body interaction | Reproduction of Skardal and Arenas (2020), Xu et al. (2021) |
| Explosive synchronization needs matching tails of g(omega) and P(k) | Reproduction and sharpening of Xu et al., Sci. Rep. 5, 12039 (2015) |
| Rician closure for the local order parameter, coefficient `(p+2)/2` | Ours, but an **approximation**: exact for the closure, not for the model |
| Leading 1/k theory separating frozen from rotating neighbour fluctuations, plus the reaction term; different predictions for directed and undirected graphs | Ours. **Confirmed on directed graphs** by the branch-shift test at p = 1; **not confirmed on undirected graphs**; no equivalent found in a web search of the literature, which is not a full-text search ([docs/PRIOR_ART.md](docs/PRIOR_ART.md)) |
| Critical connectivity below which bistability vanishes; clustering as lost connectivity | **Unverified** closure predictions, not yet measured |
| Crash-warning tests | Corrected, with the design written before the run but not timestamped ([docs/PREREGISTRATION_B.md](docs/PREREGISTRATION_B.md)); a negative, underpowered result so far |

## Layout

```
src/esfc/        package: graphs, frequencies, models, integrate (verbatim move of research/model.py),
                 fast (compiled kernel), continuation (fold estimators), closure (Rician), cavity (1/k theory),
                 stats, ews, intraday, alarms, experiments, report, cli
tests/           regression to research/model.py at 1e-12, analytic limits, invariants, synthetic empirical tests
configs/         one YAML per experiment (grid, seeds, dt, relaxation time, thresholds)
results/         <experiment>/<timestamp>_<git hash>/ : config.yaml, seeds.json, env.txt, rows.parquet, summary.json
figures_src/     one script per figure, reads only results/
research/        the original research scripts (S1 to S14, A1 to A3, E1 to E8), kept unchanged as the record,
                 plus build_index.py, which generates index.html
simulation/      browser demonstration
docs/            inventory, model spec, theory, audit, corrections, decisions, prior art, licences, progress
legacy/          the retired C++ and Python command-line layer (cannot show explosive synchronization)
```

## Install and run

Python 3.12 or newer.

```
python -m venv research/.venv && research/.venv/bin/pip install -e .[dev]
make test-fast          # about a minute
make test               # includes slow tests
esfc run configs/paperA_A1_pilot.yaml      # one experiment; writes a new directory under results/
make audit              # rebuild research/index.html, docs/CLAIMS_AUDIT.md and the tables in this file
```

Exact package versions used for the committed results are in `requirements.lock`. Each run directory
records its own environment in `env.txt` and its seeds in `seeds.json`
(`numpy.random.SeedSequence(root).spawn(n)`, root seed in the config).

The empirical scripts need market data that cannot be redistributed
([docs/DATA_LICENSES.md](docs/DATA_LICENSES.md)): `research/empirical/fetch_markets.py` downloads daily
closes from Yahoo, and the minute bars need a free HF Data Library key in
`~/.config/hfdata/api_key`. No key or price file is stored in this repository.

## Model

```
d theta_i = [ omega_i + lam * scale_i * alpha_i * sum_j A_ij sin(theta_j - theta_i) ] dt + sigma dW_i
alpha_i = r_i^p,      r_i = |sum_j A_ij e^{i theta_j}| / k_i      (local order parameter)
scale_i = 1/<k>
```

Full specification, optional terms and known caveats: [docs/MODEL_SPEC.md](docs/MODEL_SPEC.md).
Symbols used everywhere: `lam` herding coupling, `K` plain Kuramoto coupling, `p` adaptation exponent,
`R` global and `r_i` local order parameter, `<k>` mean degree, `gamma` Lorentzian half-width, `sigma`
idiosyncratic noise, `sigma_c` common noise, `f` adaptive fraction.

A limitation worth stating early: synchrony in this model has no direction. Whether a synchronised
market is a crash or a melt-up is decided only by a sell shock or bias added by hand.

## References

Zhang, Boccaletti, Guan, Liu, PRL 114, 038701 (2015). Zou and Wang, PRE 102, 012219 (2020).
Skardal and Arenas, Commun. Phys. 3, 218 (2020). Xu et al., PRR 3, 043004 (2021).
Khanra, Kundu, Pal, Ji, Hens, Chaos (2021), arXiv:2104.06862. Dutta, Pal, Hens, arXiv:2507.18421 (2025).
Kumpeerakij and Restrepo, arXiv:2605.24701 (2026). Restrepo, Ott, Hunt, PRE 71, 036151 (2005).
Metz and Perez Castillo, arXiv:2606.08689 (2026). Gomez-Gardenes et al., PRL 106, 128701 (2011).
Ott and Antonsen, Chaos 18, 037113 (2008). Lee et al., PNAS 122(44), e2505434122 (2025).
Kritzman et al., J. Portfolio Mgmt 37(4), 112 (2011). Scheffer et al., Nature 461, 53 (2009).
Full list with notes: [docs/PRIOR_ART.md](docs/PRIOR_ART.md) and `references/README.md`.
