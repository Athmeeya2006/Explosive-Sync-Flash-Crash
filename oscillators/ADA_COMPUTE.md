# Runs that need the Ada cluster

Everything else in this folder ran on the workstation: 16 cores, 15 GB RAM, with two cores always left free.
A run is listed here when it would take more than about a day on that machine or needs more memory.

## How the costs were estimated

Measured on the workstation, then scaled:

- **Z1** (`run_zoo.py`): 14 scenarios x 48 runs x 3 conditions at N = 400 took 1595 s on 14 cores,
  6.2 core-hours. Population models scale linearly with N and with the ramp time; the four single-unit
  models are a small fixed part.
- **Z2** (`z2_fold_scaling.py ramp`): 1.25e-5 core-seconds per oscillator per unit of simulated time
  (dt = 0.02), from 1287 s on 14 cores for 1.45e9 oscillator-time units.

Memory per process is below 1 GB for every run below except A4 (2.5 GB per run at k = 192).

## The list

| item | what | why it matters | cost | script |
|---|---|---|---|---|
| A1 | Z1 at N = 1600, 6400 and 25600, 100 runs per condition (all 18 scenarios, including the four herding scenarios added later) | Shows whether the indicator table (collective variance works everywhere, single units fail) holds as populations grow, and whether AUC rises with N as finite-size noise falls. Needed before the zoo benchmark can be submitted | about 1400 core-hours (1100 for the first 14 scenarios, scaled by 18/14) | `ada/zoo_size_scan.sbatch` |
| A2 | Z1 at noise 0.05, 0.1, 0.2 x ramp time 0.5, 2, 4 times the default, N = 400, 100 runs | Shows the table does not depend on one noise level and one ramp speed. A reduced version (two noise levels, 48 runs) ran locally, section R of index.html | about 250 core-hours (18 h on the workstation) | `ada/zoo_robustness.sbatch` |
| A3 | Z2 ramps at N = 25600 to 819200, 16 runs each | Whether the delay past the fold keeps shrinking with N and whether the variance ratio before collapse changes: decides whether the weak warning is a finite-size effect | about 270 core-hours; the single run at N = 819200 takes about 8.5 h on one core | `ada/z2_large_n.sbatch` |
| A4 | Branch shift on sparse graphs at k = 48, 96, 192 with N up to 800000 (main project, item 2 of docs/PROGRESS.md) | Settles whether the reaction term of the finite-connectivity theory converges on undirected graphs | about 10 GPU-hours (estimate from the A7 rate) | `configs/paperA_branch_ada_gpu.yaml` (`esfc run`, GPU) |
| A5 | Full fold grid of the main project: 30 seeds, four sizes, four exponents; low-connectivity bistability; clustered and heterogeneous graphs (docs/PROGRESS.md, "Not done" item 3) | The fold coefficient with real error bars, the predicted critical connectivity, and the clustering prediction | not yet measured; configure a pilot of one cell first and scale from its time | to write, from `configs/paperA_A1_pilot.yaml` |
| A6 | Zoo on sparse networks (k = 12 to 96) instead of mean field, for the Kuramoto and Stuart-Landau scenarios | Real markets and power grids are not all-to-all; neighbour sums cost N k per step, so this is 10 to 100 times Z1 | rough estimate, not measured: several hundred to about 1000 core-hours at N = 1600 for four values of k | to write (needs a sparse version of the population kernels in `zoo.py`) |

| A7 | O6: herding fold on sparse random regular graphs for every family; the design run (N = 2000, k = 8 to 64) and N = 20000 with k up to 192 | Tests whether the finite-connectivity coefficient a of the main project is the same for every family, as phase reduction predicts (a1 cancels in the ratio) | design run: not finished after 21 min on 8 cores on the shared workstation, estimated 10 to 20 core-hours; N = 20000 with k up to 192 is about 30 to 60 times that, so roughly 500 to 1000 core-hours (estimate: cost scales with N k per step) | `ada/o6_sparse.sbatch` (environment overrides `O6_N`, `O6_KS`, `O6_SEEDS`, `O6_TAG`) |
| A8 | Z2 ramps at noise 0.1, 0.3, 0.5 (N = 100 to 1600, 16 runs) | Whether the slow-passage delay and the weak variance warning depend on the noise level | about 12 core-hours per noise level from the Z2 rate above (the 0.1 run had used about 10 core-hours when stopped), 36 in all | `ada/z2_noise.sbatch` |
| A9 | Z1 for the four herding scenarios added after Z1 (van der Pol, Roessler, FitzHugh-Nagumo, Kuramoto with inertia), N = 400, 48 runs | Completes the indicator table for herding in every family | about 2 core-hours (4/14 of the Z1 cost) | `ada/remaining_studies.sbatch` |
| A10 | O5: frequency-degree correlation on a Barabasi-Albert network (N = 1000), every family | The main project's S1 and S8 for every family | not measured; estimated 10 to 30 core-hours (Kuramoto needs dt = 0.002) | `ada/remaining_studies.sbatch` |
| A11 | O9: propagation of a local push on Winfree and theta-neuron networks | The a2 test of Hens et al. 2019 for the two separable couplings | not finished after 29 min on 2 cores; estimated under 10 core-hours | `ada/remaining_studies.sbatch` |
| A12 | O3e: exact mean-field self-consistency of the Stuart-Landau herding branch | Whether amplitudes explain the 10% gap between the measured fold and phase reduction | serial; estimated a few hours on one core | `ada/remaining_studies.sbatch` |

The workstation runs of A7 to A12 were stopped on 2026-10-22 so that the machine stays free for analysis; their
sections are left out of index.html and reference.html until the result files are copied back and the pages rebuilt (the last line of `ada/remaining_studies.sbatch` does this).

## Not limited by compute

These need data or code, not core-hours:

- **Point-in-time constituents** for the 22 markets (removes the survivorship bias stated in M0). Needs a data
  vendor (for example CRSP or Refinitiv index membership histories).
- **Intraday data for more than 40 US stocks.** The minute-data study (paperB_intraday) has 12 test-period crashes,
  so its smallest detectable AUC is 0.70. More stocks and markets would fix the power, not more compute.
- **Lee et al. (PNAS 2025) code and data**, for an exact replication (docs/PROGRESS.md).
- **Missing world indices** (Egypt, Chile, Thailand, Philippines, Poland): not on Yahoo; need another source.

## How to run on Ada

What the cluster actually offers (checked 2026-10-24; the public documentation is older): Rocky Linux 9,
partition `u22` (GPU nodes, 40 CPUs, 128 GB and four GTX 1080 Ti or RTX 2080 Ti each; `u22-cpu` is reserved
for another group), account `research`, QoS `medium`: at most 40 CPUs, 4 GPUs and 125 GB per user at once,
20 jobs, 4 days per job. Python 3.12 is not a module; it comes from `uv`.

1. Copy the repository without `.git`, `research/.venv`, `research/empirical` and `research/tools`, and write
   the commit stamp: `git rev-parse --short HEAD` (plus `-dirty`) into `.esfc_git` at the top level.
2. Environment: `~/.local/bin/uv venv -p 3.12 research/.venv`, then install `-e ".[dev]"`, `numba-cuda[cu12]`
   and the workstation's exact versions of numpy, scipy, pandas, networkx, numba, llvmlite, pyarrow,
   matplotlib, PyYAML and sympy (`requirements.lock` is a whole-system freeze and does not install).
3. GPU path. Every simulation listed above now has one, checked against the CPU code before use:
   - main project (`esfc`): a config with `backend: gpu` runs `fold_grid` and `branch_grid` on
     `esfc.gpu`; submit with `sbatch [--gres=gpu:N -c 10N] experiments/ada/esfc_gpu.sbatch configs/<name>.yaml`
     from the repository root. Check: `configs/paperA_A7_gpu.yaml` (A7 folds, same seeds: 22 of 24 identical,
     2 one bisection step apart) and `configs/paperA_branch_check*.yaml`. Speed: one GTX 1080 Ti did the 12
     core-hours of A7 in 6 minutes.
   - oscillator zoo: `ZOO_GPU=1` (`run_zoo.py`) and `Z2_GPU=1` (`z2_fold_scaling.py`) use `zoo_gpu.py`;
     `ada/zoo_gpu_check.sbatch` runs `zoo_gpu_check.py` (deterministic agreement to 1e-14, transition points
     and mean signals agree statistically). The scripts for A1, A2, A3 and A8 request one GPU each.
4. From `oscillators/ada/`: `sbatch zoo_size_scan.sbatch` (and the others). Every job writes to
   `oscillators/data/` with its own suffix (for example `z1_scores_N6400.parquet`), so jobs never overwrite each
   other or the local results. Worker processes follow `SLURM_CPUS_PER_TASK`.
5. Copy the result files back; `figures_zoo.py` and `build_index.py` pick them up.
