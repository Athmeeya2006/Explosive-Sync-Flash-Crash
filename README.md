# Explosive Synchronization and Flash Crash in Networked Oscillators

This repository models synchronization transitions in networks of coupled oscillators. It covers two dynamical systems (Kuramoto and Stuart-Landau), two network models (ER and BA), and the analysis needed to detect abrupt changes in collective behavior such as flash-crash style drops in coherence.

The workflow is split into a fast C++ simulator and a Python analysis/plotting layer. Data is exported as simple CSV/NPY files so the steps are easy to inspect and reproduce.

## What this project adds

- A clean separation between simulation (C++) and analysis (Python)
- Network generation for ER and BA graphs with consistent parameters
- Order-parameter tracking and early-warning indicators
- Flash-crash style coupling drops to study abrupt desynchronization
- Finite-size scaling data and phase-diagram visualization
- A minimal test suite and CI build for reliability

## Model overview

Networks:

- ER (Erdos-Renyi): each pair of nodes connects with probability $p$
- BA (Barabasi-Albert): preferential attachment with $m$ edges per new node

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

The Python implementations match the C++ coupling normalization ($K/d_i$), so results are consistent across both layers.

## Repository layout

```
explosive-sync-flash-crash/
├── README.md
├── cpp/
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── network_gen.cpp/hpp
│   │   ├── kuramoto.cpp/hpp
│   │   ├── stuart_landau.cpp/hpp
│   │   ├── flash_crash.cpp/hpp
│   │   ├── order_param.cpp/hpp
│   │   └── main.cpp
│   └── include/
├── python/
│   ├── analysis/
│   │   ├── generate_networks.py
│   │   ├── run_kuramoto.py
│   │   ├── run_stuart_landau.py
│   │   ├── finite_size_scaling.py
│   │   ├── early_warning.py
│   │   └── flash_crash_sim.py
│   ├── plots/
│   │   ├── plot_order_param.py
│   │   ├── plot_finite_size.py
│   │   ├── plot_early_warning.py
│   │   ├── plot_phase_diagram.py
│   │   └── plot_flash_crash.py
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

1) Run a simulation to generate an order-parameter time series.

```
python python/analysis/run_kuramoto.py --model er --n 200 --p 0.05 --k 1.2 --tmax 50 --dt 0.05
```

2) Compute early warning indicators from the saved series.

```
python python/analysis/early_warning.py --input data/kuramoto_order.csv --window 50
```

3) Plot both series and indicators.

```
python python/plots/plot_order_param.py --input data/kuramoto_order.csv --out figures/order_param.png
python python/plots/plot_early_warning.py --input data/early_warning.csv --out figures/early_warning.png
```

4) For phase-diagram style summaries, sweep sizes and couplings.

```
python python/analysis/finite_size_scaling.py --n-list 50,100,200 --k-list 0.4,0.8,1.2
python python/plots/plot_finite_size.py --input data/finite_size.csv --out figures/finite_size.png
python python/plots/plot_phase_diagram.py --input data/finite_size.csv --out figures/phase_diagram.png
```

5) For a flash-crash scenario (abrupt coupling drop):

```
python python/analysis/flash_crash_sim.py --model er --n 200 --p 0.05 --k-high 1.5 --k-low 0.3 --t-drop 20
python python/plots/plot_flash_crash.py --input data/flash_crash.csv --out figures/flash_crash.png
```

## Plot catalog and interpretation

- Order parameter time series: shows growth or decay of coherence; sharp jumps signal explosive synchronization.
- Flash-crash response: shows how coherence collapses after a coupling drop and how quickly it recovers.
- Early warning indicators: rising variance and lag-1 autocorrelation can signal approaching transitions.
- Finite-size scaling: compares steady-state order across $N$ and $K$ to locate sharp transitions.
- Phase diagram: heatmap of steady order parameter across system size and coupling.

## Benchmarking and performance

Runtime depends on $N$, integration step size, and network density. To benchmark locally, time the analysis and plotting steps separately:

```
/usr/bin/time -v python python/analysis/run_kuramoto.py --model er --n 400 --p 0.05 --k 1.2 --tmax 50 --dt 0.05
/usr/bin/time -v python python/plots/plot_finite_size.py --input data/finite_size.csv --out figures/finite_size.png
```

For deeper profiling:

```
python -m cProfile -o profile.stats python/analysis/run_kuramoto.py --model er --n 400 --p 0.05 --k 1.2 --tmax 50 --dt 0.05
```

The C++ CLI is designed for faster parameter sweeps; use the same timing approach:

```
/usr/bin/time -v ./cpp/build/esfc --system kuramoto --network er --n 400 --p 0.05 --k 1.2 --tmax 50 --dt 0.05 --out data/order_param.csv
```

## C++ CLI

Build:

```
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

Run:

```
./cpp/build/esfc --system kuramoto --network er --n 200 --p 0.05 --k 1.2 --tmax 50 --dt 0.05 --out data/order_param.csv
```

Flash-crash scenario (coupling drop):

```
./cpp/build/esfc --system flash-crash --network er --n 200 --p 0.05 --k-high 1.5 --k-low 0.3 --t-drop 20 --tmax 50 --dt 0.05 --out data/flash_crash.csv
```

Options:

- `--system {kuramoto|stuart-landau|flash-crash}`
- `--network {er|ba}`
- `--n`, `--p`, `--m`, `--k`
- `--k-high`, `--k-low`, `--t-drop` (Flash-crash)
- `--tmax`, `--dt`, `--seed`
- `--omega-mean`, `--omega-std`
- `--alpha` (Stuart-Landau)
- `--out` for the output CSV

## Data formats

- CSV time series: `t,order`
- Early warning CSV: `t,order,variance,autocorr`
- Finite-size CSV: `n,k,order`
- Networks saved as `*.npy` adjacency matrices and `*_degrees.csv`

## Tests and CI

Run tests locally:

```
pip install -r requirements.txt pytest
python -m pytest python/tests
```

CI runs the test suite and builds the C++ CLI on GitHub Actions.

## Reproducibility

All scripts accept `--seed` to reproduce networks and initial conditions. For sweeps, vary the seed or add replicate loops to estimate variability.

## Data and figures

`data/` and `figures/` are tracked but ignored by default in `.gitignore`, so local outputs will not pollute commits while the directories remain available.
