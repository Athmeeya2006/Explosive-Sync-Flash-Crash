# Reference library

139 open-access PDF files, grouped by topic. Folders 12 to 14 (55 files, added 2026-10-25) hold the literature reviews; of the 81 entries in `fetch_references.py`, 71 are available;
the rest are listed at the end. Most were resolved through the arXiv API with a title check;
the rest were downloaded from open-access publisher pages. Re-download or extend with
`python references/fetch_references.py` (it skips files already present and writes `index.csv`).

**Read first** (marked ★): the papers your contribution has to be positioned against.

## 01 Kuramoto foundations
| File | Paper | Why you need it |
|---|---|---|
| `rodrigues2016` | Rodrigues, Peron, Ji, Kurths, *The Kuramoto model in complex networks*, Phys. Rep. 2016 | Standard review; notation, mean-field methods |
| `arenas2008` | Arenas et al., *Synchronization in complex networks*, Phys. Rep. 2008 | Network synchronization background |
| `ott_antonsen2008` | Ott & Antonsen, Chaos 2008 | Low-dimensional reduction; route to analytic theory |
| `pazo2005` | Pazó, PRE 2005 | First-order transition already in all-to-all Kuramoto (uniform g) |

## 02 Explosive synchronization
| File | Paper | Why you need it |
|---|---|---|
| ★ `gomezgardenes2011` | Gómez-Gardeñes, Gómez, Arenas, Moreno, PRL 106, 128701 (2011) | Original ES; reproduced in `research/s1` |
| ★ `zhang2015_adaptive` | Zhang, Boccaletti, Guan, Liu, PRL 114, 038701 (2015) | Adaptive (local-order-parameter) coupling = our herding mechanism; reproduced in `research/s2` |
| ★ `lee2024_es_proximity_crises` | Lee et al., bioRxiv 2024 (now published, PMC12595435) | Closest prior work: ES proximity → collapse/recovery in brain & 39 stock markets. Your work must differ from it |
| ★ `boccaletti2016_review` | Boccaletti et al., Phys. Rep. 2016 | The ES review |
| `dsouza2019_review` | D'Souza et al., Adv. Phys. 2019 | Explosive phenomena review |
| `zou2014_basin` | Zou et al., PRL 2014 | Basin of attraction sets hysteresis (relevant to crash-type diagram) |
| `leyva2012`, `leyva2013_weighted`, `peron2012`, `coutinho2013`, `ji2013` | Classic ES extensions | Frequency-degree correlation, mean-field critical couplings, cluster ES |
| `hu2014_exact_first_order`, `exact_es_time_delay2018` | Sci. Rep. | Exact forward/backward thresholds (analytics template) |
| `danziger2019`, `synchronization_bombs2022`, `adaptive_competing2026`, `typeI_neurons2025` | Recent ES mechanisms | Multilayer, self-organized, adaptive competition |

## 03 Higher-order interactions
`skardal_arenas2019` (PRL), `skardal_arenas2020` (Commun. Phys.), `millan2020` (PRL), `battiston2020` (Phys. Rep.),
`zhang2023_hyper_vs_simplicial` (Nat. Commun.), `hoi_random_hypergraphs2025`. Group (3-body) herding is a natural
extension: a trader follows a *group* of peers, not pairs.

## 04 Inertia / cascades
`olmi2014` (hysteretic Kuramoto with inertia), `multiplex_inertia2017`. Power-grid style cascading desynchronization analogues.

## 05 Early-warning signals & tipping
| File | Paper | Why |
|---|---|---|
| ★ `kato_masuda2026` | Kato & Masuda, arXiv 2608.28320 (Aug 2026) | EWS from partial observations; **continuous transitions only**, no ES, no hysteresis |
| ★ `ordinal_predictors2025` | Leyva et al., arXiv 2501.05202 | Ordinal-pattern entropy at sentinel hubs predicts ES |
| ★ `hagstrom2021_phase_transitions_ews` | Hagstrom & Levin | Why critical slowing down is a 2nd-order-transition signal |
| `scheffer2009` | Scheffer et al., Nature 2009 | Classic EWS paper |
| `dakos2012_methods` | Dakos et al., PLoS ONE 2012 | Methods (detrending, Kendall tau) used in `research/s6` |
| `ashwin2012_tipping`, `kuehn2011` | Tipping theory | B-, N-, R-tipping definitions (`research/s7`) |
| `forced_desync_ews2020`, `persistent_entropy2026`, `pseudo_coherence2026` | Related indicators | |
| `ghanavati2014_power_ews` | Ghanavati, Hines, Lakoba, IEEE TCAS-I 2014 | Autocorrelation is often late before power-system (swing-equation) transitions; compare `oscillators` Z1 and Z3 |

## 06 Finance
| File | Paper | Why |
|---|---|---|
| ★ `chen_petukhov_wang2018_dark_side_circuit_breakers` | Chen, Petukhov, Wang, *The Dark Side of Circuit Breakers* | Finance-side motivation for `research/s5` (magnet effect) |
| `filimonov_sornette2012` | Quantifying reflexivity / flash crashes | Endogeneity of crashes |
| `harmon2011_panic` | Harmon et al., *Predicting economic market crises using measures of collective panic* | Co-movement as crisis signal |
| `johnson2013_ultrafast` | Johnson et al., Sci. Rep. 2013 | Ultrafast machine ecology / mini flash crashes |
| `flash_crash_contagion2018`, `bouchaud2010_endogenous`, `sornette2003_crashes`, `synchronization_market_asymmetry2006` | Crash mechanisms | |

## 07 Novelty checks
`mitra2018_nodal_robustness`, `resilience_targeted_attacks2019` (targeted perturbations in multistable networks, prior art for S4c/S5c),
`subsystem_resetting_kuramoto2026` (resetting part of a Kuramoto population ≈ partial halt: check before claiming S5c),
`rtipping_oscillatory_sync2023` (R-tipping with oscillatory responses; prior art for S7).

## 11 Oscillator zoo (the families of `oscillators/`)
| File | Paper | Why |
|---|---|---|
| ★ `pikovsky_rosenblum2015_global_coupling` | Pikovsky & Rosenblum, Chaos 25, 097616 (2015) | Review of globally coupled oscillators: Stuart-Landau, chaotic, Winfree; the setting of every population in `oscillators/zoo.py` |
| ★ `ariaratnam_strogatz2001_winfree` | Ariaratnam & Strogatz, PRL 86, 4278 (2001) | Phase diagram of the Winfree model, incl. oscillator death (`theory_zoo.winfree_death`) |
| `pazo_montbrio2014_winfree_oa` | Pazo & Montbrio, PRX 4, 011009 (2014) | Ott-Antonsen reduction of pulse-coupled (Winfree) populations |
| ★ `montbrio2015_qif_meanfield` | Montbrio, Pazo & Roxin, PRX 5, 021028 (2015) | Exact mean field of spiking (QIF / theta) networks; the theta-neuron bistability |
| ★ `gupta2014_inertia_review` | Gupta, Campa & Ruffo, J. Stat. Mech. (2014) | Kuramoto with inertia: the incoherence instability used in `o3_phase_reduction.py` |
| `acebron2000_inertia` | Acebron, Bonilla & Spigler, PRE 62, 3437 (2000) | Inertial Kuramoto populations, hysteresis |
| `bi2014_explosive_oscillation_death` | Bi et al., EPL (2014) | Explosive oscillation death in coupled Stuart-Landau oscillators |
| `imai2021_chaotic_phase_description` | Imai, Suetani & Aoyagi (2021) | Phase description of chaotic (Roessler) oscillators; the systematic version of the a1 = 1/2 approximation |

## 12 Literature review: sparse-graph theory (Paper A)
17 arXiv preprints; review in `docs/LITREVIEW_A_SPARSE_THEORY.md`, citations and URLs in `12_lit_review_sparse_theory/SOURCES.md`.
★ `metz2025` (PRL 134, 037401): our directed theory is the 1/k expansion of its Eq. 7. ★ `metz_perezcastillo2026`
(dynamical cavity, continuous time). ★ `tarabolo_dallasta2025_gecam` (closest method). `restrepo_ott_hunt2005_onset`,
`restrepo_ott_hunt2006_directed` (mean field on networks), `aonishi_kurata_okada2002` (the reaction term). What is
new against them: explicit adaptive coefficients, the closure excluded, no critical connectivity, a directed fold below
all-to-all, reciprocity as a coupling rescaling, and the closed-form heterogeneity law.

## 13 Literature review: early warnings across oscillator families
17 arXiv preprints; review in `docs/LITREVIEW_OSCILLATOR_EWS.md`, sources in `13_lit_review_oscillator_ews/SOURCES.md`.
★ `kato_masuda2026` (evaluation without a null), ★ `leyva2012_chaotic_es` (no shuffled control), `leyva2025_ordinal`,
`chen2012_fhn_es`, `ehstand2023_percolation`, Bury et al. 2020 (the Hopf autocorrelation rule, which our Z3 shows is
overgeneralised), Berglund and Gentz and Tyulkina et al. 2018 (most of Z2 is known). The event-based phase ripple that
hides warnings (Z1) is not reported in any of them.

## 14 Literature review: markets (Paper B)
21 open-access files; review in `docs/LITREVIEW_MARKETS.md`, sources in `14_lit_review_markets/SOURCES.md`.
★ `lee2025_es_proximity` (PNAS 2025, with SI and bioRxiv version): its reported p-values are about 0.012 and 0.0015, not
"< 0.001"; the robustness claim is contradicted by its own Table S5; lag 2 chosen post hoc; 204 correlations
uncorrected; one crisis, n = 39. Also `preis2012_correlations_stress`, `sandoval_franca2012_crisis_correlation`.

## Not downloaded (paywalled / not on arXiv), get via your library
- Nakao, *Phase reduction approach to synchronisation of nonlinear oscillators*, Contemp. Phys. 57, 188 (2016): the method of `o3_phase_reduction.py`
- Koseska, Volkov & Kurths, *Oscillation quenching mechanisms: amplitude vs. oscillation death*, Phys. Rep. 531, 173 (2013)
- Mirollo & Strogatz, *Amplitude death in an array of limit-cycle oscillators*, J. Stat. Phys. 60, 245 (1990): the death window of `theory_zoo.sl_death_window`
- Matthews, Mirollo & Strogatz, *Dynamics of a large system of coupled nonlinear oscillators*, Physica D 52, 293 (1991)
- Luke, Barreto & So, *Complete classification of the macroscopic behavior of a heterogeneous network of theta neurons*, Neural Comput. 25, 3207 (2013): the theta-neuron coupling of `zoo.theta_pop`
- Rosenblum, Pikovsky & Kurths, *Phase synchronization of chaotic oscillators*, PRL 76, 1804 (1996)
- Tanaka, Lichtenberg & Oishi, *First order phase transition resulting from finite inertia in coupled oscillator systems*, PRL 78, 2104 (1997)
- Winfree, *Biological rhythms and the behavior of populations of coupled oscillators*, J. Theor. Biol. 16, 15 (1967)
- FitzHugh, Biophys. J. 1, 445 (1961); Nagumo, Arimoto & Yoshizawa, Proc. IRE 50, 2061 (1962)
- Bury, Bauch & Anand, *Detecting and distinguishing tipping points using spectral early warning signals*, J. R. Soc. Interface 17, 20200482 (2020)
- Acebrón et al., *The Kuramoto model: a simple paradigm*, Rev. Mod. Phys. 77, 137 (2005)
- Dörfler & Bullo, *Synchronization in complex networks of phase oscillators: a survey*, Automatica 2014
- Rohden et al., PRL 109, 064101 (2012), power grids
- Bury et al., *Deep learning for early warning signals of tipping points*, PNAS 2021
- Haldane & May, *Systemic risk in banking ecosystems*, Nature 2011
- Kirilenko, Kyle, Samadi, Tuzun, *The Flash Crash: High-frequency trading in an electronic market*, J. Finance 2017
- Christie & Huang (1995) and Chang, Cheng & Khorana (2000): herding via cross-sectional dispersion (used in S3b)
- Kritzman et al. (2011), *Principal components as a measure of systemic risk* (absorption ratio)
- Chaos, Solitons & Fractals (2022) PII S0960077922008359, EWS for continuous vs explosive synchronization (fetch was blocked; **check it before claiming S6 results as new**)
