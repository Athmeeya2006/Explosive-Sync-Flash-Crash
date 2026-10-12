# Repository inventory

Audited at commit `fc98f02` (184 tracked files). This file describes what exists; it changes nothing.
Counts are tracked files from `git ls-files`.

## Layers

| layer | paths | tracked files | role | tested / in CI |
|---|---|---|---|---|
| Legacy | `cpp/` (13), `python/` (24), `/data` (7), `/figures` (6), `Makefile` | 51 | C++17 RK4 and SciPy RK45 Kuramoto and Stuart-Landau with `K/d_i` coupling; coupling-drop "flash crash" | Yes. CI (Python 3.10, 3.11, 3.12) and the "95% coverage" figure refer only to this layer |
| Research | `research/*.py` (22 scripts), `research/data` (26), `research/figures` (16), `index.html`, `reference.html`, `RESEARCH_IDEAS.md` | 68 | Every scientific claim in the README | No tests, not in CI |
| Analytics | `research/analytics/` (3) | 3 | `a1_separability.py` (symbolic proof, output `a1_separability.json` committed), `a2_signal_propagation.py` (outputs not committed) | No |
| Empirical | `research/empirical/` (9 scripts, 19 result CSVs, figures) | 41 | Market data tests E1 to E8 | No |
| Demo | `simulation/` (7) | 7 | Browser port of `model.py`; outreach, not evidence | Own validators, not in CI |
| References | `references/` (7) | 7 | `README.md`, `index.csv`, fetcher, four text extracts in `08_signal_propagation/` | n/a |

## Research scripts

| script | lines | frequency law | what it produces | committed outputs |
|---|---|---|---|---|
| `model.py` | 374 | n/a | Shared simulator: `make_network`, `Params`, `drift`, `simulate`, `adiabatic_sweep`, `hysteresis`, `transition_points`, `gaussian_omegas` | n/a |
| `theory.py` | 108 | Gaussian | Annealed mean-field self-consistency and halt theory | n/a |
| `s1_reproduce_gg2011.py` | 74 | degree | Gomez-Gardenes 2011 reproduction, normalised vs unnormalised | log only (`.npz` ignored) |
| `s2_herding_explosive.py` | 96 | Gaussian | Herding hysteresis; writes `s2_herding.npz` | log only |
| `s3` to `s7` | 74 to 223 | Gaussian | Crash anatomy, phase diagram, circuit breakers, early warning, rate tipping | logs only |
| `s8`, `s8b`, `s8c` | 123 to 328 | mixed | Topology and frequency sweeps, correlation law | CSVs |
| `s9_observable_mapping.py` | 338 | Gaussian | Common-noise mapping to return co-movement | CSVs |
| `s10`, `s12` | 248, 191 | Gaussian | Model extensions scored against stylised facts | CSVs |
| `s11_analytics.py` | 152 | Lorentzian (clipped, centred) | Ott-Antonsen cubic, finite-size table | `s11_finite_size.csv`, `s11_oa_branches.csv` |
| `s13_new_hypotheses.py` | 243 | Lorentzian (clipped, not centred) | `lam_b(p)` family, clustering | `s13_*.csv` |
| `s14_finite_connectivity.py` | 606 | Lorentzian (clipped, not centred) | Rician closure, bisection harness | `s14_theory.csv`, `s14_rician_validation.csv` only |
| `a3_lambda_estimation.py` | 239 | Gaussian | Identifiability of `lam` | `a3_*.csv` |
| `build_index.py` | 867 | n/a | Generates `index.html`; contains hand-typed numbers | `index.html` |

## Things a fresh clone cannot do

1. Run `s3`, `s4`, `s5`, `s7`, `s9`: each loads `research/data/s2_herding.npz`, and `*.npz` is gitignored.
   `s4`, `s5`, `s7` load it at import time with no fallback; `s3` and `s9` fall back to different
   hardcoded windows (`(3.75, 3.0)` and `(3.96, 3.21)`).
2. Run any of E1 to E8: they need gitignored market pickles, and E6/E7 need an HF Data Library key.
3. Reproduce the S14 simulation half: `s14_measured.csv`, `s14_measured_summary.csv`,
   `s14_clustering.csv`, `s14_clustering_summary.csv` and `figures/s14_finite_connectivity.png` are
   not tracked.
4. Install from `pyproject.toml` or `requirements.txt` and run the research layer: both omit
   networkx, pandas, sympy, yfinance, pyarrow and lxml.

## Present on disk but untracked

`references/10_prior_art_adaptive/` exists locally (7 PDFs, gitignored as third-party copyright).
`RESEARCH_IDEAS.md` cites it, so a clone sees a dangling reference. `MODEL_REFERENCE.md` is cited by
`s10_extensions.py` and `e5_real_network.py` and does not exist anywhere.

## Environment used for the Phase 0 reproduction

Python 3.12.3, NumPy 2.4.6, SciPy 1.17.1, NetworkX 3.6.1, Linux, 16 cores, one process per job with
BLAS threads pinned to 1.
