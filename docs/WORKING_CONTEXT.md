# Working context (living document)

Read this first in any new session. Updated after every step so that nothing is lost when a session ends.
Last update: 2026-10-25 ~20:50 (IST), new session (workstation rebooted at 20:24; local runs restarted).

## 1. Goal

Find and finish publishable results. Do every analysis properly, check results against the literature, find
mistakes and shortcomings in others' work, plan multiple papers, keep pages and the dashboard current, use all 4
Ada GPUs. Never commit to git unless asked. No em or en dashes anywhere. Dark pages: white text, graphite or black
background. Explain every figure ("What you are seeing", "Why it matters"). Long commands run in the background.

## 2. Setup

- Workstation: 16 cores, 15 GB, no GPU. Python env `research/.venv`.
- Ada: `ssh ada` (key login; only on the IIIT network or VPN; if the hostname does not resolve, the user is off
  VPN). Account research, QoS medium: 40 CPUs, 4 GPUs, 125 GB, 20 jobs, 4 days; partition u22; memory buys CPUs at
  3 GB each. Repo copy `~/Explosive-Sync-Flash-Crash`; sync code with rsync; commit stamp `.esfc_git`.
  Job scripts (gitignored): `experiments/ada/esfc_gpu.sbatch <config>` and `oscillators/ada/*.sbatch`.
  Rule: keep all 4 GPUs busy; 1-GPU jobs use 10 CPUs; big runs `--gres=gpu:4 -c 40`.
- GPU path (validated, docs/DECISIONS.md 16 to 18): `backend: gpu` configs (fold_grid, branch_grid, window_grid);
  oscillators `ZOO_GPU=1`, `Z2_GPU=1` (zoo_gpu.py), `O6_GPU=1` (zoo_net_gpu.py; validated incl. the complete-graph
  reference, 32-seed diagnosis z = 0.43 and -0.44).
- Pull: `rsync -azu ada:Explosive-Sync-Flash-Crash/results/ results/` and `.../oscillators/data/ oscillators/data/`.
  Rebuild: `research/build_index.py` (slow, background), `oscillators/build_index.py`, `figures_src/*.py`,
  `oscillators/figures_zoo.py z1tag=<tag>` for each new Z1 run, `dashboard/build.py`.
- Dashboard: `dashboard/build.py` + `dashboard/template.html` write `dashboard.html`; published at
  https://claude.ai/artifact/AMkFkYGk89NnsVsYZ68h9K (republish with file_path dashboard.html; the files map lists
  every page and figure, about 64 files, 21 MB, no reference PDFs). Local version is NEWER than the published one
  (heterogeneity diamonds, window regularities): rebuild and republish next session.

## 3. Jobs on Ada (checked 2026-10-25 ~21:00)

| job id | name | what | state |
|---|---|---|---|
| 38473 | fold_kscan | folds k = 16 to 96 dir and undirected, tol 0.05%, 3 GPUs | running 16 h; 159 of 240 rows; missing dir k24 p1, dir k48 p2, rho1 k32, k48, k96 p2 |
| 38515 | dir_nscan | directed branch k = 24, N = 5000 to 320000 | running |
| 39213 | hetero | ER vs random regular k = 24, 48, 96 (resubmitted: first run OOM-killed building ER graphs in networkx; now numpy ER above N = 20000, esfc.graphs._er_numpy, validated) | queued |
| 39215 | o6_fhn | FitzHugh-Nagumo O6 rerun with sweep from 4 x reference | queued |
| 38518 | z1_period_all | cycle-averaged indicators, every scenario | queued |
| 38220 | branch_A4 | undirected size scan k = 48, 96, 192 (4 GPUs, 38 CPUs; starts only when all else is done) | queued |
| done | z2_noclip (38512), o6_sparse (all), o6_fhn (old, invalid), z1_fhn_period, window_N, window_long, zoo_size, zoo_robust | pulled locally |

| 39222 | o10_validate | O10 validation: und k = 24, N = 10000; Kuramoto, van der Pol, FHN; Kuramoto at 2 x T | queued |

O10: script reviewed 2026-10-25 and fixed before submission: FHN now relaxes 2 x longer (T 1000, survival judged
on 750 to 1000; branch burn-in 700) because its start-up transient lasts about 500 (O6 SKIP); dead line removed.
After validation: production stages fold, branch, hetero, window (one sbatch each, 1 GPU).

Local background runs (restarted 20:45 after the reboot; logs in the session scratchpad):
- `esfc run configs/paperB_intraday_nospy.yaml` (intraday study without SPY) -> results/paperB_intraday_nospy/
- `oscillators/z2_noise_theory.py` (exact noisy fold) -> oscillators/data/z2_noise_theory.csv

## 4. Findings (numbers in docs/ADA_RUN_2026-10-24.md; pages research/index.html P3 to P6)

Paper A (adaptive Kuramoto on sparse graphs):
- Rician closure excluded in all 60 branch cells.
- Undirected branch shift extrapolates to dynamic+reaction within 1 to 3% at 2 and 3 x fold, 5 to 9% short near the
  fold (A4 decides). Directed within about 5 to 10% (mean 1.06); dir_nscan tests finite N.
- Reciprocity: shift linear in mutual fraction; slope 123% (k = 24) to 106% (k = 48) of the predicted reaction.
- Window: no critical connectivity down to k = 3 (closure predicts closing near 6 to 8); robust to N (widens with
  N) and waiting time. Reciprocity rescales the coupling: und/dir ratio equal for fold and forward threshold within
  about 1% (p = 1, k = 4 to 24). Directed fold below the all-to-all fold at small k (0.93 at k = 4), independent of N:
  the directed fold is non-monotonic in k.
- Structure: clustering raises the fold slightly; ER lowers it strongly.
- NEW THEORY (docs/THEORY.md section 7; esfc.hetero; results/paperA_theory_hetero): degree-based mean field gives
  lambda_b/lambda_b(inf) = 1 - (p+2)/(2(p+1)^2) CV^2 exactly at leading order (derived symbolically, checked to
  1e-7). Predicts the ER fold at k = 24 within its interval at both p (0.82 vs 0.79; 1.64 vs 1.63), not at k = 12.
Oscillator studies:
- Z1: with N >= 6400 or slower ramps every transition warns (lowest AUC 0.70) except FitzHugh-Nagumo herding at
  noise 0.1 (AUC 0.03; warns at noise 0.05). Cause: event-based phase ripple; cycle-averaging restores 0.86.
- Z2/A3: escape lag reaches the deterministic ramp delay 0.75% above N ~ 6400; large-N variance ratio is an
  artifact of an N-independent floor. RESOLVED by z2_noclip: without the +-60 clip N * var_far = 0.070 at every
  N (400 to 819200) and the ratio rises to 2.52; lag unchanged 0.75%. Caveat: zoo.lorentz_q clips too, so
  Lorentzian Z1 variance indicators at N >= 25600 are conservative (run_zoo.py: Kuramoto and theta width 0.5,
  0.53% of units at the clip; Stuart-Landau width 0.1, 0.11%).
- A8 explained (in progress): exact Gaussian-noise fold (continued fraction, Sakaguchi) lies 2.05% (sigma 0.3) and
  4.89% (sigma 0.5) below the OA effective-width fold; measured escape approaches it with N (sigma 0.3: +0.12,
  -1.19, -1.55% at N = 100, 400, 1600; extrapolation about -1.8%).
- O5: correlation-induced ES for Kuramoto, Stuart-Landau, van der Pol; Roessler equally hysteretic when shuffled;
  FHN none. O6 (N = 20000, k to 192): Kuramoto about 1.3 to 1.4 and Stuart-Landau about 1.2 to 1.35 (same law),
  van der Pol about 1.45 to 1.6, Roessler climbs 0.49 to 1.0 (not 1/k). FHN invalid twice: first the start-up
  transient, then the sweep started below the local-herding fold (reference from the global-herding kernel); rerun
  39215 sweeps from 4 x.
Markets (Paper B): negative and underpowered (AUC 0.587, MDE 0.70). Found: panel is 22 stocks plus SPY (an ETF
treated as a stock); design promised 40 large caps. Rerun without SPY running locally. Deviations to disclose.
Corrections made: docs/PRIOR_ART.md and research/build_index.py no longer attribute the Rician closure to Ichinomiya.

## 5. Literature (docs/LITREVIEW_A_SPARSE_THEORY.md, LITREVIEW_OSCILLATOR_EWS.md, LITREVIEW_MARKETS.md; PDFs in
references/12, 13, 14 with SOURCES.md)

- Sparse theory: mechanisms are known (Metz and Perez Castillo 2026; Metz PRL 2025, our directed theory is the 1/k
  expansion of his Eq. 7; Tarabolo and Dall'Asta 2025 closest method; Restrepo Ott Hunt 2005; Aonishi Kurata Okada
  2002). New: explicit adaptive coefficients, closure excluded, no critical connectivity, directed fold below
  all-to-all, reciprocity rescaling, closed-form heterogeneity law.
- Oscillator EWS: Kato and Masuda 2026 (circular evaluation, no null), Leyva 2012 (no shuffled control), Leyva 2025,
  Rahjerdi 2022 (read in full still needed), Bury 2020 (Hopf ACF rule overgeneralised; our Z3), Berglund-Gentz and
  Tyulkina 2018 (Z2 mostly known). Event-based phase ripple masking EWS not reported anywhere.
- Markets: Lee et al. PNAS 2025: wrong p-values (0.012 and 0.0015, not < 0.001), robustness claim contradicted by
  its own Table S5, post hoc lag-2, 204 correlations uncorrected, n = 39 one crisis.

## 6. Paper plan

1. Paper A: finite-connectivity corrections to explosive sync with adaptive coupling (needs A4, kscan, dir_nscan,
   hetero; credit Metz, Tarabolo, Aonishi).
2. Closed-form degree-heterogeneity law + ER test (section of A or short paper).
3. No critical connectivity, reciprocity rescaling, directed non-monotonic fold (letter).
4. Oscillator EWS benchmark with strict null (Chaos); event-based phase ripple as a short letter.
5. Herding folds across families (phase reduction, O6, O10, exact noisy fold).
6. Markets: replication/critique of Lee et al.; power/design note.

Verification workflow wf_ab886108-0ca (2026-10-25): skeptical checks of 9 claims (hetero law, Lee p-values, Lee
S5, Metz link, window, reciprocity/branch, O6, A8, EWS novelty) plus a critic; results to be folded in here.

## 7. Next steps (in order)

1. Check the two local background runs (intraday no-SPY; z2_noise_theory) and record results here and in
   docs/ADA_RUN_2026-10-24.md.
2. DONE: z2_noclip analysed (docs/ADA_RUN A3). 3. DONE: O10 validation submitted (39222); then production.
4. When kscan, dir_nscan, hetero, o6_fhn, z1_period_all, A4 finish: pull, analyse, update pages
   (research P1 to P6, oscillators section C), figures, run note, this file; rebuild and republish the dashboard.
5. Add the literature reviews to references/README.md and to the dashboard Publishable section.
