# Explosive Synchronization and Flash Crash in Networked Oscillators

High-performance numerical framework for simulating synchronization transitions in coupled oscillator networks. Built from scratch in Python and C++17 (~2,300 lines across both languages), this codebase integrates Kuramoto and Stuart-Landau oscillators on Erdos-Renyi (ER) and Barabasi-Albert (BA) topologies, capturing critical phenomena like explosive (first-order) phase transitions, hysteresis loops, and flash-crash desynchronization events.

## Start here

| what | where |
|---|---|
| **Interactive simulation** of a crash spreading through the network, with every control explained | [`simulation/index.html`](simulation/index.html) |
| **The model built up from one oscillator**, each term motivated, every assumption listed | [`research/reference.html`](research/reference.html) |
| **All results**: reproductions, negative results, real-data tests | [`research/index.html`](research/index.html) |
| **What is still open**, keyed paper by paper to the prior art, with the compute each idea needs | [`research/RESEARCH_IDEAS.md`](research/RESEARCH_IDEAS.md) |

The `research/` directory supersedes the original `python/` and `cpp/` layers for every scientific
claim. See the corrections below.

### Headline findings

| | result |
|---|---|
| **NEW: finite-connectivity law for the fold** | `lambda_b(p, <k>) = gamma(p+2)((p+2)/p)^(p/2) * [1 + (p+2)/(2<k>)]`. The bracket is new and has no free parameter: it comes from treating the local order parameter `r_i` as a Rician variable (the modulus of a sum of `k_i` unit vectors) instead of replacing it by `R`. The coefficient `(p+2)/2` is exact. The same closure predicts a **critical connectivity** `k_c(p)` below which bistability does not exist at all (S14) |
| *Prior art, previously misattributed here* | The all-to-all family `R*(p) = sqrt(p/(p+2))`, `lambda_b(p) = gamma(p+2)((p+2)/p)^(p/2)` is **not new**. It is Zou & Wang, Phys. Rev. E 102, 012219 (2020) with `alpha = p+1`; their stability condition `r^2 >= (alpha-1)/(alpha+1)` is identical. The `p=2` corner is Skardal & Arenas, Commun. Phys. 3 (2020). S13's novelty claim is withdrawn (S14) |
| **NEW: clustering is lost connectivity** | Clustering correlates the phases inside a neighbourhood, which inflates the Rician scale and is equivalent to an **effective degree** `k_eff = <k>/(1 + (<k>-1) rho)`. The bistable window closes when `k_eff` falls below `k_c(p)`, so the critical clustering `C*` is predicted rather than measured. S13's measured `C* ~= 0.35` rested on a fixed coupling grid that no longer contained the transition above `C ~ 0.5`; S14 re-measures it without a grid (S14, S13) |
| **Exact threshold** (known result, correctly reproduced) | For Lorentzian frequencies the herding model solves in closed form: `lambda_b = 3*sqrt(3)*gamma`, fold at `R* = 1/sqrt(3)`. This is the `p=1` member of the Zou & Wang (2020) family; S11 derived it independently but it is not new. The 12% residual against simulation is what S14 now predicts from first principles (S11, S14) |
| **The jump is real** | It grows with system size, 0.183 at N=125 to 0.486 at N=4000, so it is not a finite-size artifact. Caveat: 3 seeds per size and the trend is not monotone (the N=2000 mean dips to 0.277), so read it as a trend, not a curve (S11) |
| **What causes it** | Not a frequency-degree rank correlation: the *tail* of g(omega) must match the tail of P(k) (S8c) |
| **On real networks the window collapses, but mostly for a boring reason** | The bistable window does collapse on networks estimated from real returns (gap 0.793 to 0.24), so the direction holds. But an **n-matched control**, which E5 previously listed as an untested caveat, shows a *random* graph at the real networks' own size already drops it to 0.396: **72% of the collapse is finite size, not clustering**, and the structural effect is a further 40% on top. Against a size-matched baseline `C_rand = <k>/(n-1)` the real excess clustering is **3.8x**, not 24x. S14 explains why the two are the same effect (E5, S14) |
| **No usable early warning** | Across 23 markets daily, 730 days hourly and 2.3M minute bars: volatility-matched AUC 0.526, and out of sample co-movement needs **77.6** false alarms per crash caught. **Plain volatility is the better detector** (65.2 false alarms per catch at the same 1% alarm budget), and co-movement with volatility regressed out is the worst of the three (274.5) (E1, E7, E8) |
| **lambda cannot be measured** | R_eq(lambda) is a step function, so 97% of market-days carry no information about it (A3) |

### Data sources

| data | source | access |
|---|---|---|
| daily closes, 23 markets, ~2,900 constituents, 2000-2026 | Yahoo via `research/empirical/fetch_markets.py` | free, no key |
| hourly bars, 730 days | Yahoo via `research/empirical/e4_intraday.py` | free, no key |
| **1-minute bars, 1,391 US tickers, 2002-present** | [HF Data Library](https://hfdatalibrary.com), CC BY 4.0 | **free key required** |

The minute data is what makes `e6` (the 2010 flash crash) and `e7` (the full minute-resolution event
study) possible; Yahoo reaches back only 730 days at hourly and 7 days at 1-minute.

```
# one-time setup for the minute data
mkdir -p ~/.config/hfdata && echo "YOUR_KEY" > ~/.config/hfdata/api_key && chmod 600 ~/.config/hfdata/api_key
# or: export HFDATA_API_KEY=YOUR_KEY
```

Get a free key at <https://hfdatalibrary.com/pages/account> and complete the profile (institution,
country, role), or the API returns 403. **The key is never stored in this repository**: `.gitignore`
blocks `.env`, `*.key` and `**/api_key`.

Note: the `/v1/bars/{TICKER}` endpoint ignores date parameters and always serves the full-history
Parquet (~50 MB per ticker, 2.25M rows). The fetchers stream it to a temp file, slice the window they
need with pyarrow, and delete the download.

## Key Technical Achievements

- **C++ and Python engines, measured honestly**: the core ODE loop exists in both a hand-tuned C++17 RK4 engine over sparse adjacency lists and a vectorised NumPy/SciPy path. For $N=400$ on an ER network ($p=0.05$, $K=2.0$, $T=50$, $dt=0.05$) the measured wall-clock times are **2.15s (C++) against 2.98s (Python)**, of which 1.52s of the Python figure is interpreter and library import. Per right-hand-side evaluation C++ is about **1.6x faster** (0.54ms against 0.87ms), but SciPy's adaptive RK45 needs only 1,568 evaluations against the fixed-step RK4's 4,000, so on pure compute the Python path is currently the faster of the two. An earlier version of this README claimed a 30x speedup (0.15s against 4.50s); that number is not reproducible and has been withdrawn.
- **16x System-Size Scaling**: Extended simulation capacity from $N=50$ to $N=800$ oscillators per ensemble, running 5 independent replica seeds at each parameter point. This 16x increase in system size made finite-size scaling analysis tractable and allowed direct comparison against analytical mean-field predictions.
- **Finite-size scaling pipeline**: sweeps across 5 system sizes and up to 21 coupling values per sweep, 5 replica seeds per point. Note the correction: this layer normalises the coupling by $k_i$, and on a sparse ER graph at $\langle k\rangle \approx 10$ the measured onset sits near $K \approx 2$, not at the all-to-all mean-field value $K_c = \sqrt{8/\pi} \approx 1.5957$ (measured steady-state order: 0.126 at $K=1.0$, 0.209 at $K=1.596$, 0.852 at $K=2.5$). An earlier version of this README claimed convergence to within 2% of $K_c$; that claim is withdrawn.
- **95% Test Coverage with 27 Unit Tests**: Built a test suite of 27 tests across 3 modules covering network generation, ODE integration correctness, order parameter bounds, CSV round-trip fidelity, and end-to-end script execution. Achieved 95% line coverage as measured by `pytest-cov`.
- **Automated CI/CD across 3 Python Versions**: Deployed GitHub Actions pipelines running the full test suite, static type checking (mypy), and linting (ruff) against Python 3.10, 3.11, and 3.12. A separate job builds the C++ CLI from source and runs a deterministic smoke test validating output row count.
- **Lyapunov Stability Analysis Pipeline**: Implemented the variational Benettin algorithm to compute the largest Lyapunov exponent (LLE) across coupling sweeps, providing a quantitative dynamical-systems diagnostic for the onset of chaos and synchronization transitions that goes beyond order-parameter heuristics.

## Explosive Synchronization

Explosive synchronization is a first-order phase transition in networked oscillators where the order parameter $R$ jumps discontinuously from near-zero to near-one as coupling $K$ crosses a critical value. Unlike the continuous (second-order) Kuramoto transition on homogeneous networks, the explosive transition exhibits hysteresis: the forward and backward critical couplings differ. This framework enforces frequency-degree correlation on heterogeneous networks to trigger the explosive regime (Gomez-Gardenes et al., PRL 2011). The hysteresis sweep module quantifies the width of this bistable region by running 40-point forward and backward coupling ramps and measuring the gap between the two transition thresholds.

> **Correction, measured in `research/s1_reproduce_gg2011.py`.** The degree-normalised coupling used by
> the `python/analysis` layer (`K/d_i`, the form written in the Kuramoto equation below) **does not**
> produce explosive synchronization: its hysteresis gap is 0.06 against 0.79 for the unnormalised
> Gomez-Gardenes form. Degree normalisation destroys the mechanism. Use the unnormalised or adaptive
> forms in `research/model.py` for any explosive-synchronization claim.
>
> **Second correction, measured in `research/s8c_correlation_corrected.py`.** A frequency-degree *rank*
> correlation is not sufficient. With a Gaussian frequency distribution the transition stays continuous
> at every correlation value from -1 to +1. The jump appears only when the tail of g(omega) matches the
> tail of P(k). See `research/reference.html` step 6.

## Model overview

Networks:

- ER (Erdos-Renyi): each pair of nodes connects with probability $p$. For $N=400, p=0.05$, each node has a mean degree of about 20, yielding roughly 4,000 undirected edges.
- BA (Barabasi-Albert): preferential attachment with $m$ edges per new node, producing a power-law degree distribution with minimum degree $m$.

Kuramoto dynamics (phases $\theta_i$):

$$
\dot{\theta}_i = \omega_i + \frac{K}{d_i} \sum_{j \in \mathcal{N}_i} \sin(\theta_j - \theta_i)
$$

Stuart-Landau dynamics (complex states $z_i$):

$$
\dot{z}_i = (\alpha + i \omega_i - |z_i|^2) z_i + \frac{K}{d_i} \sum_{j \in \mathcal{N}_i} (z_j - z_i)
$$

Order parameter (coherence):

$$
R(t) = \left| \frac{1}{N} \sum_{j=1}^N e^{i\theta_j} \right|
$$

The C++ engine uses an explicit 4th-order Runge-Kutta (RK4) integrator with fixed step size, while the Python layer uses SciPy's adaptive RK45 (`solve_ivp`) with tolerances `rtol=1e-6`, `atol=1e-8`. Both layers normalize coupling by node degree ($K/d_i$), guaranteeing deterministic cross-layer reproducibility at matching integration parameters.

## Repository layout

```
explosive-sync-flash-crash/
├── README.md
├── cpp/                          # 665 lines of C++17
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── network_gen.cpp/hpp   # ER + BA graph generators
│   │   ├── kuramoto.cpp/hpp      # Kuramoto RK4 integrator
│   │   ├── stuart_landau.cpp/hpp # Stuart-Landau RK4 integrator
│   │   ├── flash_crash.cpp/hpp   # Coupling-drop dynamics
│   │   ├── order_param.cpp/hpp   # R(t) computation
│   │   └── main.cpp              # CLI argument parser (210 lines)
├── python/                       # 1,350+ lines of analysis + plotting
│   ├── analysis/
│   │   ├── generate_networks.py
│   │   ├── run_kuramoto.py
│   │   ├── run_stuart_landau.py
│   │   ├── finite_size_scaling.py
│   │   ├── early_warning.py
│   │   ├── flash_crash_sim.py
│   │   ├── hysteresis_sweep.py
│   │   └── lyapunov_analysis.py
│   ├── plots/                    # 7 publication-quality plot scripts
│   │   ├── plot_order_param.py
│   │   ├── plot_finite_size.py
│   │   ├── plot_early_warning.py
│   │   ├── plot_phase_diagram.py
│   │   ├── plot_flash_crash.py
│   │   ├── plot_degree_distribution.py
│   │   └── plot_percolation.py
│   ├── tests/                    # 27 tests, 95% coverage
│   │   ├── test_analysis.py
│   │   ├── test_helpers.py
│   │   └── test_scripts.py
│   └── utils/
│       └── helpers.py
├── data/
└── figures/
```

## Setup

Requirements:

- Python 3.10+
- CMake 3.15+ and a C++17 compiler

Install Python dependencies:

```
pip install -r requirements.txt
```

## End-to-end workflow

Generate an order-parameter time series (this runs in ~0.15s on the C++ backend for $N=400$):

```
python python/analysis/run_kuramoto.py --model er --n 200 --p 0.05 --k 2.0 --tmax 50 --dt 0.05
```

For explosive synchronization on BA networks, apply degree-weighted frequencies (`--freq-mode degree-weighted --omega-mean 1.0`). Note the correction above: with this layer's `K/d_i` normalisation the transition stays continuous regardless.

Compute early warning indicators (rolling variance and lag-1 autocorrelation over a sliding window):

```
python python/analysis/early_warning.py --input data/kuramoto_order.csv --window 50
```

Generate diagnostic plots:

```
python python/plots/plot_order_param.py --input data/kuramoto_order.csv --out figures/order_param.png
python python/plots/plot_early_warning.py --input data/early_warning.csv --out figures/early_warning.png
```

Execute finite-size scaling sweeps covering 5 system sizes x 21 coupling values x 5 replica seeds (525 total simulations):

```
python python/analysis/finite_size_scaling.py --n-list 50,100,200,400,800 --k-list 0.0,0.5,1.0,1.5,2.0 --n-replicas 5
python python/plots/plot_finite_size.py --input data/finite_size.csv --out figures/finite_size.png
python python/plots/plot_phase_diagram.py --input data/finite_size.csv --out figures/phase_diagram.png
```

Run hysteresis analysis with 40-point forward and backward coupling sweeps:

```
python python/analysis/hysteresis_sweep.py --model ba --freq-mode degree-weighted --k-steps 40 --out data/hysteresis.csv
```

Compute Lyapunov exponents across a 16-point coupling sweep (Benettin variational method with 20s transient burn-in):

```
python python/analysis/lyapunov_analysis.py --n 100 --k-min 0.0 --k-max 3.0 --k-steps 16 --tmax 50 --out data/lyapunov.csv
```

Simulate flash-crash dynamics (abrupt 88% coupling attenuation from $K=2.5$ to $K=0.3$):

```
python python/analysis/flash_crash_sim.py --model er --n 200 --p 0.05 --k-high 2.5 --k-low 0.3 --t-drop 20
python python/plots/plot_flash_crash.py --input data/flash_crash.csv --out figures/flash_crash.png
```

## Plot catalog and interpretation

- **Order parameter time series**: shows growth or decay of coherence. Sharp jumps signal explosive synchronization; gradual growth indicates a second-order transition.
- **Flash-crash response**: captures how coherence collapses after an 88% coupling drop ($K: 2.5 \to 0.3$) and how quickly (or whether) it recovers.
- **Early warning indicators**: rising variance and lag-1 autocorrelation (computed over a rolling window) can signal approaching transitions, consistent with critical slowing down theory.
- **Finite-size scaling**: compares steady-state order across 5 system sizes ($N=50$ to $N=800$) and 21 coupling values to locate the critical coupling and measure the sharpness of the transition.
- **Phase diagram**: heatmap of steady-state order parameter across system size and coupling, generated from 525 simulation runs.
- **Degree distribution**: highlights the structural difference between ER (Poisson) and BA (power-law) networks.
- **Percolation**: identifies the emergence of the giant connected component in ER graphs as edge probability $p$ increases.

## Benchmarking and performance

Runtime depends on $N$, integration step size, and network density. Typical timings for a standard scenario ($N=400$, ER network $p=0.05$, $K=2.0$, $T=50$, $dt=0.05$, i.e. 1,000 integration steps):

| Implementation | Integrator | RHS evaluations | Wall clock | Compute only |
| --- | --- | --- | --- | --- |
| C++ (O2, adjacency list) | RK4, fixed-step | 4,000 | 2.15s | 2.15s |
| Python (SciPy, csr_matrix) | RK45, adaptive | 1,568 | 2.98s | 1.46s (1.52s is imports) |

Per right-hand-side evaluation the C++ engine is about 1.6x faster (0.54ms against 0.87ms). The
adaptive integrator needs 2.6x fewer evaluations, so the two roughly cancel: C++ wins by ~1.4x on
wall clock and loses on pure compute. Re-measure on your own hardware before quoting either number.

The C++ backend stores the network as a vector of adjacency lists (`vector<vector<int>>`), avoiding the overhead of dense $N \times N$ matrix storage. The Python layer uses SciPy's `csr_matrix` for the same reason. Both approaches reduce memory from $O(N^2)$ to $O(N + E)$, where $E$ is the edge count.

A full parameter sweep at the script defaults is 5 system sizes x 21 coupling points x 5 replicas = 525 runs. The committed `data/finite_size.csv` is **not** that sweep: it is a 9-row smoke output (3 sizes, 3 couplings) written by an older version of the script, and its `n,k,order` header no longer matches the `n,k,order_mean,order_std` the script now emits, so the two plot scripts that consume it fail. Regenerate it with `make data` before plotting.

To benchmark locally, time the analysis and plotting steps separately:

```
/usr/bin/time -v python python/analysis/run_kuramoto.py --model er --n 400 --p 0.05 --k 2.0 --tmax 50 --dt 0.05
/usr/bin/time -v python python/plots/plot_finite_size.py --input data/finite_size.csv --out figures/finite_size.png
```

For deeper profiling:

```
python -m cProfile -o profile.stats python/analysis/run_kuramoto.py --model er --n 400 --p 0.05 --k 2.0 --tmax 50 --dt 0.05
```

The C++ CLI is designed for faster parameter sweeps; use the same timing approach:

```
/usr/bin/time -v ./cpp/build/esfc --system kuramoto --network er --n 400 --p 0.05 --k 2.0 --tmax 50 --dt 0.05 --out data/order_param.csv
```

## C++ CLI

Build (compiles 6 translation units with `-O2 -Wall -Wextra -Wpedantic` under C++17):

```
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

Run:

```
./cpp/build/esfc --system kuramoto --network er --n 200 --p 0.05 --k 2.0 --tmax 50 --dt 0.05 --out data/order_param.csv
```

Flash-crash scenario (coupling drop):

```
./cpp/build/esfc --system flash-crash --network er --n 200 --p 0.05 --k-high 2.5 --k-low 0.3 --t-drop 20 --tmax 50 --dt 0.05 --out data/flash_crash.csv
```

Options:

- `--system {kuramoto|stuart-landau|flash-crash}`
- `--network {er|ba}`
- `--n`, `--p`, `--m`, `--k`
- `--freq-mode {gaussian|degree-weighted}`
- `--k-high`, `--k-low`, `--t-drop` (Flash-crash)
- `--tmax`, `--dt`, `--seed`
- `--omega-mean`, `--omega-std`
- `--alpha` (Stuart-Landau)
- `--out` for the output CSV
- `--version` to print the CLI version

## Data formats

- CSV time series: `t,order`
- Early warning CSV: `t,order,variance,autocorr`
- Finite-size CSV: `n,k,order_mean,order_std`
- Hysteresis CSV: `k,direction,order`
- Networks saved as `*.npy` adjacency matrices and `*_degrees.csv`

## Tests and CI

The test suite contains 27 tests organized across 3 modules:

| Module | Tests | What it covers |
| --- | --- | --- |
| `test_helpers.py` | 10 | Network generation (ER/BA symmetry, edge cases), order parameter bounds, CSV I/O round-trip |
| `test_analysis.py` | 8 | ODE integration correctness, flash-crash output bounds, Kuramoto/Stuart-Landau fixed-point convergence, sub-critical incoherence |
| `test_scripts.py` | 9 | End-to-end execution of all 8 analysis scripts and 7 plot scripts via CLI argument injection |

Run tests locally:

```
make test
```

This runs `pytest` with `--cov` to measure line coverage (currently 95%).

CI runs on every push and pull request via GitHub Actions:
- Python jobs test against 3.10, 3.11, and 3.12 with ruff linting and mypy type checking
- C++ job builds the CLI from source and runs a deterministic smoke test (validates that a known input produces exactly 52 output rows)

## Reproducibility

All scripts accept `--seed` to reproduce networks and initial conditions. The finite-size scaling module uses NumPy's `SeedSequence.spawn()` to generate independent, non-overlapping RNG streams for each replica, ensuring statistically valid ensemble averaging without seed collisions.

**Note on RNG Incompatibility**: Python uses NumPy's PCG64 RNG while the C++ CLI uses `std::mt19937`. Identical seeds will **not** generate identical networks across languages. For cross-language comparisons, prefer exporting a shared network representation (e.g., Python writes an adjacency `.npy` or an edge-list CSV) and add a C++ reader for that format.

## Data and figures

`data/` and `figures/` hold generated outputs. `make data` regenerates the standard analysis data files and `make figures` produces the diagnostic plots. Generated CSVs and figures should not be committed; the ones currently tracked are legacy and several are stale (see the finite-size note above, and note that the tracked `data/flash_crash.csv` peaks at R = 0.30, i.e. shows no crash, while a fresh run of the documented command gives R = 0.851 before the drop and 0.068 after).

## Citations

1. Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer Berlin Heidelberg.
2. Gomez-Gardenes, J., Gomez, S., Arenas, A., & Moreno, Y. (2011). *Explosive Synchronization Transitions in Scale-Free Networks*. Physical Review Letters, 106(12), 128701.
3. Barabasi, A.-L., & Albert, R. (1999). *Emergence of Scaling in Random Networks*. Science, 286(5439), 509-512.
4. Scheffer, M., et al. (2009). *Early-warning signals for critical transitions*. Nature, 461, 53-59.
