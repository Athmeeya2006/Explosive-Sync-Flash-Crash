# Literature review: early warnings, slow passage and explosive synchronisation across oscillator families

Date: 2026-10-25. Scope: the oscillator studies Z1 (early-warning benchmark), Z2/Z3 (slow passage, Gaussian
noise fold, exact autocorrelation of a damped oscillator) and the O-series (phase reduction of the herding
fold, sparse 1/k law, frequency-degree correlation on Barabasi-Albert networks). PDFs of the open-access papers
read for this review are in `references/13_lit_review_oscillator_ews/` (listed in `SOURCES.md` there). Papers
already in other reference folders are pointed to, not duplicated, except the four closest ones, which were
copied so the folder is self-contained.

How this was done: arXiv API and abstract pages, web search (standard and extended), PubMed Central for open
journal versions. 33 relevant papers were identified; 17 PDFs were downloaded or copied; 9 were read in full
(Kato and Masuda 2026, Leyva et al. 2025, Leyva et al. 2012, Chen et al. 2013, Tyulkina et al. 2018,
Boettiger and Hastings 2012 (methods sections), Berglund and Gentz 2002 (introduction and main theorems),
Ghanavati et al. 2014 (oscillator section), Bury et al. 2020 (Hopf section via the PMC full text)); the others
were read at abstract, methods and results level. Where only an abstract was available, this is stated.

---

## (a) Table of all relevant papers

Legend for "file": D = downloaded or copied into `references/13_lit_review_oscillator_ews/`; L = already in
the library (folder given); no = paywalled or not needed in full. "Closeness" is to this project: 3 = directly
overlapping claim, 2 = same question in a narrower setting, 1 = background or method.

| # | Key | Paper (short) | Venue, year | arXiv | Closeness | File |
|---|---|---|---|---|---|---|
| 1 | kato_masuda2026 | Kato, Masuda: EWS for synchronization transitions from partial observations | arXiv preprint 2026 | 2608.28320 | 3 | D (also L 05) |
| 2 | leyva2025 | Leyva, Almendral, Letellier, Sendina-Nadal: Local predictors of explosive synchronization with ordinal methods | Entropy 27, 113 (2025) | 2501.05202 | 3 | D (also L 05) |
| 3 | rahjerdi2022 | Rahjerdi, Ramamoorthy, Nazarimehr, Rajagopal, Jafari: Indicating the synchronization bifurcation points using EWS, continuous and explosive synchronization (Kuramoto and Roessler, skewness and kurtosis) | Chaos Solitons Fractals 164, 112656 (2022) | none | 3 | no (paywalled; abstract only) |
| 4 | leyva2012 | Leyva et al.: Explosive first-order transition to synchrony in networked chaotic oscillators | PRL 108, 168702 (2012) | 1203.6508 | 3 | D (also L 02) |
| 5 | chen2013 | Chen, He, Huang, Shen, Hou: Explosive synchronization transitions in complex neural networks (FitzHugh-Nagumo, BA, frequency-degree) | Chaos 23, 033124 (2013) | 1204.1816 | 3 | D |
| 6 | bury2020 | Bury, Bauch, Anand: Detecting and distinguishing tipping points using spectral EWS | J. R. Soc. Interface 17, 20200482 (2020) | none | 3 | no (open on PMC7536046, read there) |
| 7 | ghanavati2014 | Ghanavati, Hines, Lakoba, Cotilla-Sanchez: Understanding early indicators of critical transitions in power systems from autocorrelation functions | IEEE TCAS-I 61, 2747 (2014) | 1309.7306 | 2 | L 05 |
| 8 | ditlevsen2026 | S. Ditlevsen, P. Ditlevsen: EWS can vanish or amplify, dimensionality in complex systems (inertia, red noise) | arXiv preprint 2026 | 2609.01164 | 2 | D |
| 9 | hobden2026 | Hobden, Ritchie, Ashwin: EWS through vector autoregression, multiscale systems (fold, subcritical and singular Hopf) | arXiv preprint 2026 | 2605.28260 | 2 | D |
| 10 | tyulkina2018 | Tyulkina, Goldobin, Klimenko, Pikovsky: Dynamics of noisy oscillator populations beyond the Ott-Antonsen ansatz | PRL 120, 264101 (2018) | 1804.05326 | 3 | D |
| 11 | goldobin2018 | Goldobin, Tyulkina, Klimenko, Pikovsky: Collective mode reductions for populations of coupled noisy oscillators | Chaos 28, 101101 (2018) | 1808.07833 | 2 | no |
| 12 | tonjes2020 | Tonjes, Pikovsky: Low-dimensional description for ensembles of identical phase oscillators subject to Cauchy noise | PRE 102, 052315 (2020) | 2009.04725 | 2 | D |
| 13 | sonnenschein2013 | Sonnenschein, Schimansky-Geier: Approximate solution to the stochastic Kuramoto model | PRE 88, 052111 (2013) | 1308.5629 | 2 | no |
| 14 | berglund2002 | Berglund, Gentz: The effect of additive noise on dynamical hysteresis | Nonlinearity 15, 605 (2002) | math/0107199 | 3 | D |
| 15 | bergeot2026 | Bergeot, Berglund, Zogheib: The dynamic saddle-node bifurcation with noise on the slow variable | Nonlinearity 39, 075004 (2026) | 2512.10460 | 1 | no |
| 16 | ritchie2016 | Ritchie, Sieber: Early-warning indicators for rate-induced tipping | Chaos 26, 093116 (2016) | 1509.01696 | 2 | D |
| 17 | ritchie2016b | Ritchie, Sieber: Early-warning indicators in the dynamic regime (saddle-node with linear drift) | arXiv preprint 2016 | 1609.07271 | 2 | no |
| 18 | pavithran2021 | Pavithran, Sujith: Effect of rate of change of parameter on EWS (Rijke tube, dynamic Hopf) | Chaos 31, 013116 (2021) | 2101.11811 | 2 | D |
| 19 | radhakrishnan2026 | Radhakrishnan, Kumar, Pavithran, Sujith: EWS for primary and secondary bifurcation to oscillatory instabilities | arXiv preprint 2026 | 2603.24068 | 1 | D |
| 20 | boettiger2012 | Boettiger, Hastings: Quantifying limits to detection of early warning for critical transitions | J. R. Soc. Interface 9, 2527 (2012) | 1204.6231 | 2 | D |
| 21 | dakos2012 | Dakos et al.: Methods for detecting early warnings of critical transitions in time series | PLoS ONE 7, e41010 (2012) | none | 1 | L 05 |
| 22 | scheffer2009 | Scheffer et al.: Early-warning signals for critical transitions | Nature 461, 53 (2009) | none | 1 | L 05 |
| 23 | kuehn2011 | Kuehn: A mathematical framework for critical transitions | Physica D 240, 1020 (2011) | 1101.2899 | 1 | L 05 |
| 24 | weinans2021 | Weinans, Quax, van Nes, van de Leemput: Evaluating the performance of multivariate indicators of resilience loss | Sci. Rep. 11, 9148 (2021) | none | 2 | no (open on PMC8080839) |
| 25 | bury2021 | Bury et al.: Deep learning for early warning signals of tipping points | PNAS 118, e2106140118 (2021) | none | 1 | no (open on PMC8488604) |
| 26 | rodriguezsanchez2020 | Rodriguez-Sanchez, van Nes, Scheffer: EWS for desynchronization in periodically forced systems | arXiv preprint 2020 | 2003.11595 | 2 | D (also L 05) |
| 27 | leyva_masoller2020 | Leyva, Masoller: Inferring connectivity and anticipating transition to synchrony through lag-time analysis | Chaos Solitons Fractals 133, 109604 (2020) | 2001.08195 | 2 | D |
| 28 | ehstand2023 | Ehstand, Donner, Lopez, Hernandez-Garcia: Network percolation provides early warnings of abrupt changes in coupled oscillatory systems (noisy FitzHugh-Nagumo ring) | PRE 108, 054207 (2023) | 2305.05450 | 2 | D |
| 29 | ghosh2022 | Ghosh: Early detection of synchrony in coupled oscillator model (mutual-information index) | Eur. Phys. J. Plus 137, 897 (2022) | 2208.04905 | 1 | D |
| 30 | fan2021 | Fan, Kong, Lai, Wang: Anticipating synchronization with machine learning (incl. explosive hysteresis) | PRR 3, 023237 (2021) | 2103.13358 | 1 | no |
| 31 | liu2024 | Liu et al.: Early predictor for the onset of critical transitions in networked dynamical systems | PRX 14, 031009 (2024) | none | 1 | no |
| 32 | pecora1998 | Pecora, Carroll: Master stability functions for synchronized coupled systems | PRL 80, 2109 (1998) | none | 1 | no |
| 33 | gengel2022 | Gengel, Pikovsky: Phase reconstruction with iterated Hilbert transform embeddings, benefits and limitations | Physica D 429, 133070 (2022) | 2111.10300 | 1 | no |
| 34 | kralemann2007 | Kralemann, Cimponeriu, Rosenblum, Pikovsky, Mrowka: Uncovering interaction of coupled oscillators from data (protophase to phase) | PRE 76, 055201 (2007) | none | 1 | no |
| 35 | gomezgardenes2011 | Gomez-Gardenes, Gomez, Arenas, Moreno: Explosive synchronization transitions in scale-free networks | PRL 106, 128701 (2011) | 1102.4823 | 1 | L 02 |
| 36 | vlasov2015 | Vlasov, Zou, Pereira: Explosive synchronization is discontinuous | PRE 92, 012904 (2015) | 1411.6873 | 1 | no |
| 37 | masuda2026 | Masuda: A theory of spatial EWS for tipping points on complex networks | arXiv preprint 2026 | 2608.15476 | 1 | no |
| 38 | roberts2017 | Roberts, Are: EWS for multivariate time series with sparse and non-uniform sampling (ratio of deviations, Hopf) | arXiv preprint 2017 | 1706.06201 | 1 | no |

---

## (b) Detailed notes on the closest papers

Each entry: citation, what they did, how it differs from ours, then problems. "Mistake" means a statement that
is wrong or circular as written; "untested claim" means a claim the paper's own evidence does not support;
"limitation" means a restriction of scope that the authors may or may not state.

### 1. Kato and Masuda 2026 (closest on Z1)

Y. Kato, N. Masuda, "Early warning signals for synchronization transitions from partial observations",
arXiv:2608.28320 [physics.soc-ph] (28 Aug 2026), no DOI yet.

What they did. Noisy Kuramoto model on 3 model networks (N = 100) and 6 empirical networks (43 to 217 nodes),
Gaussian frequencies, sigma = 0.15. Quasi-static chained up-sweeps in K (steps 0.01, 150 time units burn-in,
150 recorded). Three indicators: local order parameter <R_S>, its variance Var(R_S), and a new one, the
variance of linearly detrended unwrapped phases V_S (Eq. 5 and 6). Six sentinel-selection rules (S1 all nodes,
S2/S3 dynamics based at 0.9 Kc or 0.5 Kc, S4 hubs, S5 low degree, S6 random). Scored by Kendall tau of the
indicator against K over [K0, Kc] (Sec. II F) and by TIPMOC detection rate (Sec. II G). Main results
(Tables II and III): with all nodes every indicator has tau 0.79 to 0.89 and 100% detection; with ln N
sentinels Var(R_S) fails (tau about -0.27 to 0.02, detection at most 5%), while <R_S> and V_S keep working.

How it differs from ours. One family (Kuramoto), one transition type (continuous onset of synchronisation), no
explosive or herding transition, no desynchronisation, N = 100 only, no ramp (stationary runs per K), no
single-unit indicators in our sense (a single oscillator's own time series), no null model. Their headline
"variance of the collective order parameter works with full observation but variance from a few nodes fails"
is the Kuramoto special case of our Z1 "collective vs single unit" contrast, and it appeared two months
before our runs. That is real overlap and must be cited.

Problems.
- Mistake (circular evaluation). Kc is defined from <R>_t itself: the smallest K where <R>_t has covered 20%
  of the distance from its K0 value to 1 (Eq. 3). For the full-observation set S1, <R_S>_t is <R>_t, so the
  reported tau = 0.87 and 100% TIPMOC detection for <R_S>_t under S1 (Tables II, III) measure that a quantity
  rises before the point where, by definition, it has risen by 20%. This cannot fail and is not an early
  warning. The same circularity weakens the S2/S3 entries, where sentinels are chosen by phase locking to the
  global mean field at 0.9 Kc, which uses all-node information near the transition (the authors note the
  practical cost in Sec. IV, but not the circularity).
- Untested claim (no false-alarm rate). Kendall tau over [K0, Kc] and TIPMOC detection rates are computed
  only on runs that do approach the transition. There is no null (for example the same sweep length at
  couplings well below Kc, or sweeps away from Kc), so a positive tau or a TIPMOC detection has no stated
  false-positive rate. This is exactly the problem Boettiger and Hastings (2012, "Problematic null models")
  warn about. Our Z1 uses a strict null (ramp of equal size away from the transition) and reports ROC AUC,
  which is what their Table II would need.
- Limitation, partly a mistake of interpretation. V_S is the variance of an unwrapped phase after removing a
  straight line. Below Kc the oscillators drift through the mean field with nonuniform rotation and noise
  driven phase slips, and the detrended phase of a noisy oscillator has a variance that grows with the window
  length (phase diffusion). V_S therefore rises when phase slips become slower and larger as K grows; that is
  the saddle-node-on-invariant-circle bottleneck of each oscillator relative to the mean field, not the
  critical slowing down of the collective mode that the paper invokes (Sec. II D, "owing to critical slowing
  down"). The paper does not separate the two.
- Limitation. Single noise level, single window length, single N; finite-size scaling is not checked, so
  "about ln N sentinels suffice" rests on networks of 43 to 217 nodes.

### 2. Leyva, Almendral, Letellier, Sendina-Nadal 2025 (closest on explosive-synchronisation warnings)

I. Leyva, J. A. Almendral, C. Letellier, I. Sendina-Nadal, "Local predictors of explosive synchronization with
ordinal methods", Entropy 27(2), 113 (2025), doi:10.3390/e27020113, arXiv:2501.05202.

What they did. Ordinal pattern transition entropy H_T (Eq. 3 to 6), D = 3, T = 2000 samples, computed at hubs
and leaves of star networks (N = 31) and scale-free networks (N = 100 or 500) of Kuramoto oscillators, Chialvo
maps, y-coupled Roessler oscillators (a = 0.165, b = 0.4, c = 8.5) with a linear frequency-degree correlation,
and the N = 6 electronic star of Leyva 2012. Compared visually with standard deviation and lag-1
autocorrelation (Eq. 8) for the Kuramoto star only (Fig. 1b). Claim: hub H_T rises well before the explosive
jump and "surpasses traditional EWS".

How it differs from ours. Explosive synchronisation onset by frequency-degree correlation, not explosive
desynchronisation by herding; local (hub) observable, not the collective signal; no Stuart-Landau, van der Pol,
FitzHugh-Nagumo, theta, Winfree families; no null; no ROC.

Problems.
- Untested claim. "Surpasses traditional EWS" (abstract and Sec. IV) is supported by one panel (Fig. 1b,
  Kuramoto star, 10 instances, no error bars) and visual comparison. No detection statistic, no false-alarm
  rate, no comparison for the Roessler, Chialvo or experimental cases.
- Limitation that affects the interpretation. In the Kuramoto star the authors themselves note that the
  hub's fluctuation size grows linearly with coupling because the beat amplitude grows (Sec. III A). A hub
  driven by many detuned leaves sees a forcing proportional to d, so any amplitude or complexity measure of
  the hub trends with d whether or not a transition is near. The obvious control, the same sweep with
  frequencies shuffled (no explosive transition), is not shown for any system. Without it, the hub H_T rise
  may be a forcing effect, not a precursor.
- Event-based phase. For Chialvo maps the phase is defined by linear interpolation between spikes (Eq. 11),
  and R is computed from it. They do not apply variance-type indicators to R there, so they do not meet the
  spike-period ripple we found, but their R(d) diagrams inherit it.
- Minor. In Fig. 1b the hub's standard deviation peaks near the backward (desynchronisation) coupling and the
  authors call this a false alarm for the forward transition. That is a correct and useful observation; it
  is the same phenomenon our strict null is designed to catch.

### 3. Leyva et al. 2012 (Roessler explosive synchronisation): was there a shuffled control?

I. Leyva, R. Sevilla-Escoboza, J. M. Buldu, I. Sendina-Nadal, J. Gomez-Gardenes, A. Arenas, Y. Moreno,
S. Gomez, R. Jaimes-Reategui, S. Boccaletti, "Explosive first-order transition to synchrony in networked
chaotic oscillators", Phys. Rev. Lett. 108, 168702 (2012), doi:10.1103/PhysRevLett.108.168702, arXiv:1203.6508.

What they did. N = 1000 piecewise-linear Roessler units (Eq. 1 and 2, not the standard Roessler), x-coupled,
on configuration-model scale-free networks (gamma = 2.2, 2.5, 3.0, <k> = 6), with alpha_i = alpha (1 +
Delta_alpha (k_i - 1)/N) (Eq. 3), Delta_alpha = 6 to 10, so hub frequencies differ from leaf frequencies by
tens of percent up to about a factor 2. Forward and backward quasi-static sweeps, steps 3e-4 or 5e-5. Phase
by arctan(y/x). Fig. 1: hysteresis for some (gamma, R) and none for others; Fig. 2: (d, R) phase diagram;
Figs. 3 and 4: an electronic star of 6 circuits (hub 3333 Hz, leaves 2240 +- 200 Hz).

Answer to the question. No. The paper contains no simulation with the frequencies decoupled from degree
(shuffled, random, or uncorrelated). Every figure uses Eq. (3) or the frequency-ordered star. Yet the
conclusion states: "We have shown that the correlation between oscillation frequencies and structural
heterogeneity is a necessary condition for such an explosive phenomenon." This is an untested claim: the
necessity is asserted, not shown. The paper's own data even point the other way in part: at fixed
correlation the order of the transition switches with the chaotic regime R (Fig. 2, regions I and II) and in
the experiment with R_exp at fixed frequencies (Fig. 4), which shows that the node dynamics, not only the
correlation, decide whether hysteresis appears. A shuffled control is the only way to separate the two.
Other problems:
- Limitation. One network realisation per curve, no error bars, N = 1000 only; no finite-size scaling of the
  jump or the hysteresis width, which is how Vlasov, Zou and Pereira (2015) later tested whether "explosive"
  transitions are truly discontinuous.
- Limitation. The experiment has 6 oscillators. A hysteresis loop in 6 coupled circuits is evidence of
  bistability, not of a first-order transition in the statistical-physics sense used in the title.
- Limitation. The phase is computed by arctan(y/x) in the simulations but by Poincare-section interpolation in
  the experiment; the claim that the two are "fully equivalent" for phase-coherent systems is asserted with a
  reference, not checked on their data.
- Later papers (Leyva 2025, Leyva and Masoller 2020, Boccaletti 2016 review, D'Souza 2019 review) cite
  Leyva 2012 as establishing frequency-degree-correlation ES in chaotic oscillators; none that we found adds
  the missing control. Chen et al. (2013), for FitzHugh-Nagumo, did add it (their Fig. 4).

What our O5 result does and does not show. Our Roessler (standard Roessler a = 0.15, b = 0.2, c = 10, x-coupled,
BA network N = 1000, frequency spread 2% per standard deviation of degree, continuous ramp T = 4000, 2 seeds)
is as hysteretic with shuffled frequencies (largest gap 0.68, area 0.006) as with correlated ones (0.78,
0.007). That is the control Leyva 2012 did not run, and it says that in our setting the hysteresis is not
caused by the correlation. It does not refute Leyva 2012, because (i) their unit is the piecewise Roessler,
(ii) their frequency heterogeneity is about 10 to 50 times larger than ours, (iii) they swept quasi-statically,
we used a continuous ramp, and (iv) our down-sweep starts from identical states (r0 = 1); weakly coupled
phase-coherent Roessler oscillators have very small phase diffusion, so a synchronised start can persist far
below the locking threshold on the time scale of the ramp. Our hysteresis may therefore be a transient
memory effect, not bistability. A stationarity test is needed before this goes in a paper (see section d).

### 4. Chen, He, Huang, Shen, Hou 2013 (FitzHugh-Nagumo with frequency-degree correlation)

H. Chen, G. He, F. Huang, C. Shen, Z. Hou, "Explosive synchronization transitions in complex neural networks",
Chaos 23, 033124 (2013), doi:10.1063/1.4818543, arXiv:1204.1816.

What they did. FitzHugh-Nagumo eps x' = x - x^3 - y + C sum A_ij (x_j - x_i), y' = x + a_i + xi (Eq. 1 and 2),
eps = 0.01, a_i = 0.99 - delta (k_i - k_min)/(k_max - k_min) (Eq. 3), noise D = 0.005, N = 200, <k> = 6, BA and
ER. Event-based phase from threshold crossings x = 1 (Eq. 4). Result: hysteresis in BA for delta >= 0.25
(Figs. 1 and 2), none in ER, none with shuffled a_i or negative correlation (Fig. 4), and assortativity
suppresses it (Fig. 5).

How it differs from ours. Our O5 FitzHugh-Nagumo shows no hysteresis with or without correlation (largest gap
0.14 vs 0.13, areas 4e-4 both). Their units sit at a_i close to 0.99, right next to the Hopf point |a| = 1
where the frequency is extremely sensitive to a_i and where coupling can switch units between spiking and
quiescence; ours use an input-current parameterisation spread over a wide range inside the oscillatory
regime, with N = 1000. So the two results are about different regimes and do not contradict each other.

Problems.
- Untested alternative mechanism. Near |a| = 1 the low-degree units (a close to 0.99) are barely oscillatory;
  with diffusive coupling to quiescent or slower neighbours some may stop spiking. The event-based phase of
  Eq. (4) is undefined for a unit that stops spiking, and the paper does not report the fraction of spiking
  units along the sweeps. Part of the "explosive" jump in r could be a switch in the number of active units
  (an excitability transition) rather than phase locking.
- Limitation. N = 200, one network per curve, no finite-size test; hysteresis area only in Fig. 2.
- Positive point: the shuffled and negatively correlated controls (Fig. 4) are there, which makes this the
  methodological standard Leyva 2012 did not meet.

### 5. Bury, Bauch, Anand 2020 (Hopf autocorrelation) and Ghanavati et al. 2014 (damped oscillator)

T. M. Bury, C. T. Bauch, M. Anand, "Detecting and distinguishing tipping points using spectral early warning
signals", J. R. Soc. Interface 17, 20200482 (2020), doi:10.1098/rsif.2020.0482 (read on PMC7536046).
G. Ghanavati, P. D. H. Hines, T. I. Lakoba, E. Cotilla-Sanchez, "Understanding early indicators of critical
transitions in power systems from autocorrelation functions", IEEE Trans. Circuits Syst. I 61, 2747 (2014),
doi:10.1109/TCSI.2014.2332246, arXiv:1309.7306.

What they did. Bury et al. derive Ornstein-Uhlenbeck approximations of the power spectrum and autocorrelation
before fold, transcritical, pitchfork, flip, Hopf and Neimark-Sacker bifurcations. For Hopf (Table 1, Sec.
2.1) they use rho(tau) = exp(mu |tau|) cos(omega_0 tau) and state: "At sufficiently low lag times (tau < T/4),
autocorrelation increases as the bifurcation is approached", "for lag times near to half the period ... the
autocorrelation decreases", and recommend several lags. Ghanavati et al. linearise a single-machine power
system to a damped harmonic oscillator d'' + 2 gamma d' + omega_0^2 d = -f eta (Eq. 5), derive the exact
autocorrelation and show that its trend depends on the lag (Sec. on SMIB, Fig. 3); there the bifurcation is a
saddle-node where omega_0 goes to zero at fixed damping.

How it differs from ours. Our Z3 treats the other oscillatory route: damping goes to zero at fixed stiffness
(the linear picture of a Hopf point), with force noise and the position observed. The exact position
autocorrelation is exp(-g tau/2) [cos w' tau + (g/2w') sin w' tau], and its derivative with respect to g at
g = 0 is (sin x - x cos x)/(2 w0) > 0 for 0 < x = w0 tau < 4.49, so as damping vanishes the position's lag-tau
autocorrelation falls at every sampling interval below the Nyquist limit. The velocity rises only for
x < 2.03, and the Bury normal form rises only for x < pi/2.

Problem exposed in Bury et al. (overgeneralisation, a limitation rather than an error in their algebra). The
rule "tau < T/4 gives an increase before a Hopf bifurcation" is exact for the Hopf normal form with isotropic
noise observed in one coordinate, which is the model of Table 1. It is presented as the behaviour "prior to
Hopf bifurcations". For a second-order (mechanical) oscillator driven by force noise and observed in position,
which is the standard model of power-system swings, structural modes and the Rijke tube, the rule is wrong at
every lag: the autocorrelation falls. Which way lag-1 autocorrelation moves before an oscillatory instability
therefore depends on the observable and on where the noise enters, not only on the lag relative to the period.
Ditlevsen and Ditlevsen (2026) make the related point for inertia in a saddle-node (stiffness to zero), not
for vanishing damping. We did not find this exact statement for the damping route in any paper.

### 6. Tyulkina, Goldobin, Klimenko, Pikovsky 2018 (Gaussian vs Cauchy noise in Ott-Antonsen)

I. V. Tyulkina, D. S. Goldobin, L. S. Klimenko, A. Pikovsky, "Dynamics of noisy oscillator populations beyond
the Ott-Antonsen ansatz", Phys. Rev. Lett. 120, 264101 (2018), doi:10.1103/PhysRevLett.120.264101,
arXiv:1804.05326.

What they did. Rewrite the Fourier hierarchy of a noisy Kuramoto population (Eq. 4) in circular cumulants
(Eq. 8 to 10). The Ott-Antonsen manifold is kappa_m = 0 for m > 1; with Gaussian intrinsic noise the cumulants
scale as kappa_m ~ sigma^(2(m-1)). Truncating after kappa_2 gives a closed 2-cumulant model (Eq. 13) with
accuracy O(sigma^4), against O(sigma^2) for the naive replacement gamma -> gamma + sigma^2 (their noise
normalisation <xi xi> = 2 delta, so their sigma^2 is our sigma^2/2). Fig. 1 confirms the error orders for the
standard Kuramoto steady state.

How it relates to ours. Our Z2 finding that with Gaussian noise the herding fold sits 1.3% (sigma = 0.3) and
4.0% (sigma = 0.5) below the prediction with effective width gamma + sigma^2/2 is the expected O(sigma^2)
error of the naive Ott-Antonsen treatment; the qualitative point (exact for Cauchy noise, approximate for
Gaussian; Tonjes and Pikovsky 2020 is the Cauchy reference) is known. What is not in the literature, as far
as we found, is (i) the application to an explosive fold (a saddle-node of the mean-field equation produced
by order-parameter-dependent coupling), and (ii) the sign and size of the correction there. The 2-cumulant
model makes a concrete prediction we can test: substituting h = (K R^p / 2) Z into Eq. (13) gives a corrected
fold location. If it reproduces 1.3% and 4.0%, the Z2 result becomes a validated theory instead of an
anomaly.

Problems. None found in the paper. Limitation: the closure is checked only for the stationary Kuramoto state
and a chimera; no saddle-node of the mean field and no slow passage are examined. Sonnenschein and
Schimansky-Geier (2013), the Gaussian ansatz, are reported (Goldobin et al. 2018) to be accurate only at high
synchrony, which is exactly where the herding fold is not.

### 7. Berglund and Gentz 2002 (slow passage through a saddle-node with noise)

N. Berglund, B. Gentz, "The effect of additive noise on dynamical hysteresis", Nonlinearity 15, 605-632
(2002), doi:10.1088/0951-7715/15/3/305, arXiv:math/0107199.

What they did. Rigorous pathwise analysis of a periodically forced overdamped double-well Langevin equation:
deterministic delay of the jump past the saddle-node of order eps^(2/3) (Eq. 1.2), and three noise regimes with
a critical noise intensity sigma_c ~ (eps a0)^(1/2) (Theorems 2.3 to 2.6, Case IIa/IIb/III): below it, the
jump happens at the deterministic delayed point; above it, noise triggers the jump early and the hysteresis
area depends to leading order on sigma alone.

How it relates to ours. Our Z2 result (escape lag shrinks from 1.34% at N = 100 to 0.75% at N = 6400 and stays
at 0.75% up to N = 819200; "delay ~ speed^0.65" in `z2_fold_scaling.py`) is this theory applied to a mean-field
saddle-node whose effective noise is finite-size noise, sigma_eff ~ N^(-1/2). The measured exponent 0.65 is
the Berglund-Gentz 2/3, and the crossover at N about 6400 is the regime boundary sigma_eff ~ eps^(1/2). This
is a confirmation, not a discovery. It does give a falsifiable prediction we have not tested: the crossover
size should scale as N* ~ 1/eps (inverse ramp speed). Bergeot, Berglund and Zogheib (2026) extend the theory to
noise on the slow variable.

Problems. None. Limitation: one-dimensional normal form with additive noise; finite-size noise in a
population is multiplicative and correlated with the order parameter, so the constants (not the exponents)
need checking.

### 8. Boettiger and Hastings 2012 (nulls and ROC)

C. Boettiger, A. Hastings, "Quantifying limits to detection of early warning for critical transitions",
J. R. Soc. Interface 9, 2527-2539 (2012), doi:10.1098/rsif.2012.0125, arXiv:1204.6231.

What they did. Argue that Kendall tau tests on rolling-window statistics have no valid non-parametric null
("Problematic null models"), fit a time-heterogeneous OU model and a constant OU model, simulate both, and
compare indicators by ROC curves of the false-alarm vs detection trade-off.

How it relates to ours. Our Z1 scoring (ROC AUC of Kendall tau between ramps towards the transition and ramps
of equal size away from it) is a simulation-based version of their approach. The method is theirs; what we
add is the "equal size away" null, which controls for trends caused by moving the parameter at all, and its
use across 18 transitions in 11 families. Kato and Masuda 2026 and Leyva et al. 2025 do not use any null,
which is the gap their paper predicted.

### Shorter notes on the rest

- Rodriguez-Sanchez, van Nes, Scheffer 2020 (arXiv:2003.11595): desynchronisation from a periodic forcing is a
  saddle-node of cycles; raw oscillatory data do not show critical slowing down directly, but the phase
  difference to the forcing does. This is the closest prior statement of our spike-ripple point (the
  oscillation must be removed before computing CSD indicators), but for a single forced oscillator, not for a
  population order parameter built from event-based phases.
- Ehstand et al. 2023 (PRE 108, 054207): noisy FitzHugh-Nagumo ring, percolation of correlation networks as
  precursor of switches; separates mean-field trends from synchronisation of individual fluctuations. Related to
  collective vs individual, different indicator class, no ROC null.
- Leyva and Masoller 2020 (CSF 133, 109604): lag-time distributions between pairs anticipate gradual and
  explosive Kuramoto transitions (N = 50, frequency-gap networks) and in the 28-node Roessler electronic data.
  Pairwise, not collective; no null.
- Rahjerdi et al. 2022 (CSF 164, 112656): skewness and kurtosis of synchronisation measures before continuous
  and explosive transitions in Kuramoto and Roessler networks. Abstract only (paywalled); it is the earliest
  paper we found that applies classical EWS to explosive synchronisation. Must be cited and, if possible, read
  before submission, because it overlaps our Z1 indicator list (skewness) and our Roessler family.
- Hobden, Ritchie, Ashwin 2026 (arXiv:2605.28260): VAR(1) eigenvalue estimation for fold, subcritical and
  singular Hopf; states that variance and AR(1) "can be obscured by oscillatory behaviour" and shows AR(1)
  decreasing before a singular Hopf (their Fig. 6C). Our Z1 includes VAR(1) and the AR(2) decay rate, so the
  indicator idea is theirs; the multi-family benchmark is ours.
- Ditlevsen and Ditlevsen 2026 (arXiv:2609.01164): inertia and red noise modify EWS before a saddle-node;
  observations misaligned with the critical direction can hide EWS. Closest general statement of "single
  observables can show the opposite of a warning", but no oscillator populations.
- Pavithran and Sujith 2021 (Chaos 31, 013116): faster ramps shorten warning time (inverse power law) before a
  subcritical Hopf in a Rijke tube and a noisy Hopf model. Consistent with our "fast ramps weaken the warning".
- Weinans et al. 2021 (Sci. Rep. 11, 9148): no multivariate indicator wins everywhere; which is best depends
  on the scenario. Our Z1 reaches the same verdict for oscillators.
- Ritchie and Sieber 2016 (Chaos 26, 093116) and 2016b (arXiv:1609.07271): indicators lag the forcing and are
  biased by non-stationarity under a drifting parameter, the theory behind our ramp-speed dependence.
- Fan et al. 2021 (PRR 3, 023237) and Liu et al. 2024 (PRX 14, 031009): machine-learning anticipation of
  synchronisation, including explosive hysteresis; different method class.
- Gengel and Pikovsky 2022, Kralemann et al. 2007: phase estimates from observables (protophases) differ from
  the true phase by a periodic distortion; for spiking units, event-based interpolation is a protophase. This
  is the theoretical reason for the ripple we found, but none of these papers looks at what the distortion
  does to variance indicators of the order parameter.

---

## (c) What this project adds, claim by claim

1. Z1, multi-family benchmark with a strict null. New as a whole. No paper we found compares the same
   indicators across 11 oscillator families and 18 transitions, scored by ROC AUC against a null of equal
   parameter change away from the transition. Overlap: the scoring idea is Boettiger and Hastings (2012) and
   Dakos et al. (2012); the indicator set is standard (VAR from Hobden et al. 2026, AR(2) is close to Bury 2020's
   spectral approach).
2. Z1, collective variance warns, single units often show the opposite. Partly overlapping. Kato and Masuda
   2026 show for Kuramoto that Var(R) from all nodes warns while Var(R_S) from ln N nodes does not (their
   Table II), and Ditlevsen and Ditlevsen 2026 show that misaligned observables hide EWS. New here: the same
   contrast across 11 families, and that single units can move against the warning (not just fail to warn).
3. Z1, the FitzHugh-Nagumo herding failure and its cause. New as far as we found. No paper reports that an
   event-based phase puts a ripple at the spiking period into R, that this ripple dominates the detrended
   variance (84% of power), that it shrinks as units drift apart so the collective variance falls before the
   collapse (AUC 0.03), and that averaging R over one spike period restores the warning (AUC 0.86).
   Rodriguez-Sanchez et al. (2020) make the general point that the oscillation must be removed first; the
   protophase literature explains why event-based phases are distorted. Leyva 2025 and Chen 2013 use exactly
   this event-based phase and never test variance indicators on it. This is the most clearly novel item.
4. Z3, exact lag-tau autocorrelation of a noise-driven damped oscillator as damping vanishes. The formula
   itself is textbook (Ghanavati 2014 Eq. 14 has the same family); the derivative statement (position falls at
   every lag below Nyquist, velocity rises only for w0 tau < 2.03) and its contradiction of the "tau < T/4"
   rule of Bury 2020 for this observable are, as far as we found, new. Small but clean.
5. Z2, slow passage through the explosive-desynchronisation fold. Mostly confirmation: delay exponent 0.65
   against Berglund-Gentz 2/3, and the noise-to-deterministic crossover at N about 6400 is their regime
   boundary with sigma_eff ~ N^(-1/2). New: the measurement in an adaptive (herding) Kuramoto fold, up to
   N = 819200, with the clipping artefact identified. Honest framing: an application of known theory.
6. Z2, Gaussian noise fold 1.3% to 4% below the gamma + sigma^2/2 prediction. The existence of an O(sigma^2)
   error is known (Tyulkina et al. 2018; exact only for Cauchy noise, Tonjes and Pikovsky 2020). New would be:
   the 2-cumulant correction to an explosive fold, if derived and matched to these numbers. Currently we have
   only the measurement.
7. O3/O6, phase reduction maps each family's herding fold; Kuramoto and Stuart-Landau share one 1/k
   coefficient (1.36, 1.35), van der Pol about 18% higher, Roessler not a 1/k law. No direct prior work found
   on phase reduction of order-parameter-dependent (herding) coupling across families or on the sparse-graph
   1/k coefficient across families. Overlap risk with Paper A's own prior-art folder (10_prior_art_adaptive),
   not with the EWS literature.
8. O5, frequency-degree correlation across families. New in combination: Kuramoto, Stuart-Landau and van der
   Pol become explosive with the correlation and lose it when shuffled; FitzHugh-Nagumo shows none either way;
   Roessler is equally hysteretic with shuffled frequencies. The Roessler control is the one Leyva 2012 did
   not run; the FitzHugh-Nagumo null disagrees in outcome with Chen 2013 but in a different regime (see b.4).
   These two contrasts are publishable only after the checks in (d).
9. O5, divergence of the x-coupled Roessler network when coupling times degree exceeds about 4.6. Known: the
   bounded master stability window of x-coupled Roessler oscillators (Pecora and Carroll 1998). Our
   contribution is only the practical warning that hubs in scale-free networks leave the window at modest
   coupling, which any heterogeneous Roessler ES study (Leyva 2012 uses piecewise Roessler with different
   parameters) should check.

---

## (d) Novelty risks

- Kato and Masuda (Aug 2026) already published the Kuramoto version of "collective variance works, few-node
  variance fails". Any paper on Z1 must cite it and position Z1 as the multi-family, null-controlled
  extension, and should point out their circular use of <R> (b.1).
- Rahjerdi et al. (2022) applied skewness and kurtosis to continuous and explosive synchronisation in Kuramoto
  and Roessler networks. Not read in full (paywalled). If they also tested variance and autocorrelation with
  any null, our Z1 novelty claim narrows. Get the PDF through a library before writing.
- The Roessler O5 contrast with Leyva 2012 is fragile: our hysteresis may be a ramp-speed and initial-condition
  artefact (identical-state down-sweep, slow phase diffusion, continuous ramp, 2 seeds, small frequency spread).
  Required before claiming anything: quasi-static sweeps (or 3 ramp speeds, extrapolated as in O3c), random
  initial phases on the down-sweep at fixed coupling inside the loop with long relaxation, Leyva's frequency
  spread (Delta_alpha = 6 to 10) and, ideally, the piecewise Roessler of Eq. (1).
- The FitzHugh-Nagumo O5 null disagrees in outcome with Chen 2013. Without reproducing their regime
  (a_i near 0.99, eps = 0.01, N = 200) we can only say "not in our regime". Reproducing their Fig. 1(b) and
  then shuffling, plus counting spiking units, would turn this into either a confirmation or a correction.
- Z2 slow passage and Gaussian-noise offset are applications of Berglund-Gentz and the circular-cumulant
  theory. Presenting them as new physics would be wrong; as validation of a model and of the GPU code they are
  fine.
- The event-ripple mechanism (c.3) rests on one scenario (FitzHugh-Nagumo herding, sigma = 0.1). Before it is
  a general claim it needs: theta neurons and Winfree with event-based phase (they are spiking too), a
  Hilbert or protophase-to-phase corrected phase as an alternative fix, and an analytic estimate of the ripple
  amplitude vs N and dispersion.
- Z3 is short and the general idea (sign of AC trend depends on lag) is in Bury 2020 and Ghanavati 2014; it
  works as a section or a short note, not as a stand-alone paper.

---

## (e) BibTeX

```bibtex
@article{kato_masuda2026,
  author  = {Kato, Yusuke and Masuda, Naoki},
  title   = {Early warning signals for synchronization transitions from partial observations},
  journal = {arXiv preprint},
  eprint  = {2608.28320},
  year    = {2026}
}
@article{leyva2025,
  author  = {Leyva, Inmaculada and Almendral, Juan A. and Letellier, Christophe and Sendi{\~n}a-Nadal, Irene},
  title   = {Local Predictors of Explosive Synchronization with Ordinal Methods},
  journal = {Entropy},
  volume  = {27},
  number  = {2},
  pages   = {113},
  year    = {2025},
  doi     = {10.3390/e27020113},
  eprint  = {2501.05202}
}
@article{rahjerdi2022,
  author  = {Karimi Rahjerdi, Bahareh and Ramamoorthy, Ramesh and Nazarimehr, Fahimeh and Rajagopal, Karthikeyan and Jafari, Sajad},
  title   = {Indicating the synchronization bifurcation points using the early warning signals in two case studies: Continuous and explosive synchronization},
  journal = {Chaos, Solitons \& Fractals},
  volume  = {164},
  pages   = {112656},
  year    = {2022},
  doi     = {10.1016/j.chaos.2022.112656}
}
@article{leyva2012,
  author  = {Leyva, I. and Sevilla-Escoboza, R. and Buld{\'u}, J. M. and Sendi{\~n}a-Nadal, I. and G{\'o}mez-Garde{\~n}es, J. and Arenas, A. and Moreno, Y. and G{\'o}mez, S. and Jaimes-Re{\'a}tegui, R. and Boccaletti, S.},
  title   = {Explosive First-Order Transition to Synchrony in Networked Chaotic Oscillators},
  journal = {Physical Review Letters},
  volume  = {108},
  pages   = {168702},
  year    = {2012},
  doi     = {10.1103/PhysRevLett.108.168702},
  eprint  = {1203.6508}
}
@article{chen2013,
  author  = {Chen, Hanshuang and He, Gang and Huang, Feng and Shen, Chuansheng and Hou, Zhonghuai},
  title   = {Explosive synchronization transitions in complex neural networks},
  journal = {Chaos},
  volume  = {23},
  pages   = {033124},
  year    = {2013},
  doi     = {10.1063/1.4818543},
  eprint  = {1204.1816}
}
@article{bury2020,
  author  = {Bury, Thomas M. and Bauch, Chris T. and Anand, Madhur},
  title   = {Detecting and distinguishing tipping points using spectral early warning signals},
  journal = {Journal of the Royal Society Interface},
  volume  = {17},
  pages   = {20200482},
  year    = {2020},
  doi     = {10.1098/rsif.2020.0482}
}
@article{ghanavati2014,
  author  = {Ghanavati, Goodarz and Hines, Paul D. H. and Lakoba, Taras I. and Cotilla-Sanchez, Eduardo},
  title   = {Understanding Early Indicators of Critical Transitions in Power Systems From Autocorrelation Functions},
  journal = {IEEE Transactions on Circuits and Systems I: Regular Papers},
  volume  = {61},
  number  = {9},
  pages   = {2747--2760},
  year    = {2014},
  doi     = {10.1109/TCSI.2014.2332246},
  eprint  = {1309.7306}
}
@article{ditlevsen2026,
  author  = {Ditlevsen, Susanne and Ditlevsen, Peter},
  title   = {Early Warning Signals Can Vanish or Amplify: Dimensionality in Complex Systems},
  journal = {arXiv preprint},
  eprint  = {2609.01164},
  year    = {2026}
}
@article{hobden2026,
  author  = {Hobden, Bryony and Ritchie, Paul and Ashwin, Peter},
  title   = {Early warnings of critical transitions through vector autoregression: lessons from multiscale systems},
  journal = {arXiv preprint},
  eprint  = {2605.28260},
  year    = {2026}
}
@article{tyulkina2018,
  author  = {Tyulkina, Irina V. and Goldobin, Denis S. and Klimenko, Lyudmila S. and Pikovsky, Arkady},
  title   = {Dynamics of Noisy Oscillator Populations beyond the {Ott-Antonsen} Ansatz},
  journal = {Physical Review Letters},
  volume  = {120},
  pages   = {264101},
  year    = {2018},
  doi     = {10.1103/PhysRevLett.120.264101},
  eprint  = {1804.05326}
}
@article{goldobin2018,
  author  = {Goldobin, Denis S. and Tyulkina, Irina V. and Klimenko, Lyudmila S. and Pikovsky, Arkady},
  title   = {Collective mode reductions for populations of coupled noisy oscillators},
  journal = {Chaos},
  volume  = {28},
  pages   = {101101},
  year    = {2018},
  doi     = {10.1063/1.5053576},
  eprint  = {1808.07833}
}
@article{tonjes2020,
  author  = {T{\"o}njes, Ralf and Pikovsky, Arkady},
  title   = {Low-dimensional description for ensembles of identical phase oscillators subject to {Cauchy} noise},
  journal = {Physical Review E},
  volume  = {102},
  pages   = {052315},
  year    = {2020},
  doi     = {10.1103/PhysRevE.102.052315},
  eprint  = {2009.04725}
}
@article{sonnenschein2013,
  author  = {Sonnenschein, Bernard and Schimansky-Geier, Lutz},
  title   = {Approximate solution to the stochastic {Kuramoto} model},
  journal = {Physical Review E},
  volume  = {88},
  pages   = {052111},
  year    = {2013},
  doi     = {10.1103/PhysRevE.88.052111},
  eprint  = {1308.5629}
}
@article{berglund2002,
  author  = {Berglund, Nils and Gentz, Barbara},
  title   = {The effect of additive noise on dynamical hysteresis},
  journal = {Nonlinearity},
  volume  = {15},
  pages   = {605--632},
  year    = {2002},
  doi     = {10.1088/0951-7715/15/3/305},
  eprint  = {math/0107199}
}
@article{bergeot2026,
  author  = {Bergeot, Baptiste and Berglund, Nils and Zogheib, Israa},
  title   = {The dynamic saddle-node bifurcation with noise on the slow variable},
  journal = {Nonlinearity},
  volume  = {39},
  pages   = {075004},
  year    = {2026},
  doi     = {10.1088/1361-6544/ae7ec0},
  eprint  = {2512.10460}
}
@article{ritchie2016,
  author  = {Ritchie, Paul and Sieber, Jan},
  title   = {Early-warning indicators for rate-induced tipping},
  journal = {Chaos},
  volume  = {26},
  pages   = {093116},
  year    = {2016},
  doi     = {10.1063/1.4963012},
  eprint  = {1509.01696}
}
@article{ritchie2016b,
  author  = {Ritchie, Paul and Sieber, Jan},
  title   = {Early-warning indicators in the dynamic regime},
  journal = {arXiv preprint},
  eprint  = {1609.07271},
  year    = {2016}
}
@article{pavithran2021,
  author  = {Pavithran, Induja and Sujith, R. I.},
  title   = {Effect of rate of change of parameter on early warning signals for critical transitions},
  journal = {Chaos},
  volume  = {31},
  pages   = {013116},
  year    = {2021},
  doi     = {10.1063/5.0025533},
  eprint  = {2101.11811}
}
@article{radhakrishnan2026,
  author  = {Radhakrishnan, Rohit and Kumar, Prasana and Pavithran, Induja and Sujith, R. I.},
  title   = {Early warning signals for primary and secondary bifurcation to oscillatory instabilities},
  journal = {arXiv preprint},
  eprint  = {2603.24068},
  year    = {2026}
}
@article{boettiger2012,
  author  = {Boettiger, Carl and Hastings, Alan},
  title   = {Quantifying limits to detection of early warning for critical transitions},
  journal = {Journal of the Royal Society Interface},
  volume  = {9},
  number  = {75},
  pages   = {2527--2539},
  year    = {2012},
  doi     = {10.1098/rsif.2012.0125},
  eprint  = {1204.6231}
}
@article{dakos2012,
  author  = {Dakos, Vasilis and Carpenter, Stephen R. and Brock, William A. and Ellison, Aaron M. and Guttal, Vishwesha and Ives, Anthony R. and K{\'e}fi, Sonia and Livina, Valerie and Seekell, David A. and van Nes, Egbert H. and Scheffer, Marten},
  title   = {Methods for Detecting Early Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data},
  journal = {PLoS ONE},
  volume  = {7},
  pages   = {e41010},
  year    = {2012},
  doi     = {10.1371/journal.pone.0041010}
}
@article{scheffer2009,
  author  = {Scheffer, Marten and Bascompte, Jordi and Brock, William A. and Brovkin, Victor and Carpenter, Stephen R. and Dakos, Vasilis and Held, Hermann and van Nes, Egbert H. and Rietkerk, Max and Sugihara, George},
  title   = {Early-warning signals for critical transitions},
  journal = {Nature},
  volume  = {461},
  pages   = {53--59},
  year    = {2009},
  doi     = {10.1038/nature08227}
}
@article{kuehn2011,
  author  = {Kuehn, Christian},
  title   = {A mathematical framework for critical transitions: Bifurcations, fast-slow systems and stochastic dynamics},
  journal = {Physica D},
  volume  = {240},
  pages   = {1020--1035},
  year    = {2011},
  doi     = {10.1016/j.physd.2011.02.012},
  eprint  = {1101.2899}
}
@article{weinans2021,
  author  = {Weinans, Els and Quax, Rick and van Nes, Egbert H. and van de Leemput, Ingrid A.},
  title   = {Evaluating the performance of multivariate indicators of resilience loss},
  journal = {Scientific Reports},
  volume  = {11},
  pages   = {9148},
  year    = {2021},
  doi     = {10.1038/s41598-021-87839-y}
}
@article{bury2021,
  author  = {Bury, Thomas M. and Sujith, R. I. and Pavithran, Induja and Scheffer, Marten and Lenton, Timothy M. and Anand, Madhur and Bauch, Chris T.},
  title   = {Deep learning for early warning signals of tipping points},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {118},
  number  = {39},
  pages   = {e2106140118},
  year    = {2021},
  doi     = {10.1073/pnas.2106140118}
}
@article{rodriguezsanchez2020,
  author  = {Rodr{\'i}guez-S{\'a}nchez, Pablo and van Nes, Egbert H. and Scheffer, Marten},
  title   = {Early warning signals for desynchronization in periodically forced systems},
  journal = {arXiv preprint},
  eprint  = {2003.11595},
  year    = {2020}
}
@article{leyva_masoller2020,
  author  = {Leyva, Inmaculada and Masoller, Cristina},
  title   = {Inferring the connectivity of coupled oscillators and anticipating their transition to synchrony through lag-time analysis},
  journal = {Chaos, Solitons \& Fractals},
  volume  = {133},
  pages   = {109604},
  year    = {2020},
  doi     = {10.1016/j.chaos.2020.109604},
  eprint  = {2001.08195}
}
@article{ehstand2023,
  author  = {Ehstand, No{\'e}mie and Donner, Reik V. and L{\'o}pez, Crist{\'o}bal and Hern{\'a}ndez-Garc{\'i}a, Emilio},
  title   = {Network percolation provides early warnings of abrupt changes in coupled oscillatory systems: An explanatory analysis},
  journal = {Physical Review E},
  volume  = {108},
  pages   = {054207},
  year    = {2023},
  doi     = {10.1103/PhysRevE.108.054207},
  eprint  = {2305.05450}
}
@article{ghosh2022,
  author  = {Ghosh, Anupam},
  title   = {Early detection of synchrony in coupled oscillator model},
  journal = {The European Physical Journal Plus},
  volume  = {137},
  pages   = {897},
  year    = {2022},
  doi     = {10.1140/epjp/s13360-022-03122-7},
  eprint  = {2208.04905}
}
@article{fan2021,
  author  = {Fan, Huawei and Kong, Ling-Wei and Lai, Ying-Cheng and Wang, Xingang},
  title   = {Anticipating synchronization with machine learning},
  journal = {Physical Review Research},
  volume  = {3},
  pages   = {023237},
  year    = {2021},
  doi     = {10.1103/PhysRevResearch.3.023237},
  eprint  = {2103.13358}
}
@article{liu2024,
  author  = {Liu, Zijia and Zhang, Xiaozhu and Ru, Xiaolei and Gao, Ting-Ting and Moore, Jack Murdoch and Yan, Gang},
  title   = {Early Predictor for the Onset of Critical Transitions in Networked Dynamical Systems},
  journal = {Physical Review X},
  volume  = {14},
  pages   = {031009},
  year    = {2024},
  doi     = {10.1103/PhysRevX.14.031009}
}
@article{pecora1998,
  author  = {Pecora, Louis M. and Carroll, Thomas L.},
  title   = {Master Stability Functions for Synchronized Coupled Systems},
  journal = {Physical Review Letters},
  volume  = {80},
  pages   = {2109--2112},
  year    = {1998},
  doi     = {10.1103/PhysRevLett.80.2109}
}
@article{gengel2022,
  author  = {Gengel, Erik and Pikovsky, Arkady},
  title   = {Phase reconstruction from oscillatory data with iterated {Hilbert} transform embeddings: Benefits and limitations},
  journal = {Physica D},
  volume  = {429},
  pages   = {133070},
  year    = {2022},
  doi     = {10.1016/j.physd.2021.133070},
  eprint  = {2111.10300}
}
@article{kralemann2007,
  author  = {Kralemann, Bj{\"o}rn and Cimponeriu, Laura and Rosenblum, Michael and Pikovsky, Arkady and Mrowka, Ralf},
  title   = {Uncovering interaction of coupled oscillators from data},
  journal = {Physical Review E},
  volume  = {76},
  pages   = {055201},
  year    = {2007},
  doi     = {10.1103/PhysRevE.76.055201}
}
@article{gomezgardenes2011,
  author  = {G{\'o}mez-Garde{\~n}es, Jes{\'u}s and G{\'o}mez, Sergio and Arenas, Alex and Moreno, Yamir},
  title   = {Explosive Synchronization Transitions in Scale-Free Networks},
  journal = {Physical Review Letters},
  volume  = {106},
  pages   = {128701},
  year    = {2011},
  doi     = {10.1103/PhysRevLett.106.128701},
  eprint  = {1102.4823}
}
@article{vlasov2015,
  author  = {Vlasov, Vladimir and Zou, Yong and Pereira, Tiago},
  title   = {Explosive synchronization is discontinuous},
  journal = {Physical Review E},
  volume  = {92},
  pages   = {012904},
  year    = {2015},
  doi     = {10.1103/PhysRevE.92.012904},
  eprint  = {1411.6873}
}
@article{masuda2026,
  author  = {Masuda, Naoki},
  title   = {A theory of spatial early warning signals for tipping points on complex networks},
  journal = {arXiv preprint},
  eprint  = {2608.15476},
  year    = {2026}
}
@article{roberts2017,
  author  = {Roberts, Andrew and Are, Sasanka},
  title   = {An early warning system for multivariate time series with sparse and non-uniform sampling},
  journal = {arXiv preprint},
  eprint  = {1706.06201},
  year    = {2017}
}
```

Notes on BibTeX: the Chaos 23, 033124 reference for Chen et al. is taken from the DOI 10.1063/1.4818543 on the
arXiv record; the arXiv metadata lists four authors while the PDF lists five (Gang He included); the PDF author
list is used. Vlasov et al. PRE 92, 012904 and Kralemann et al. PRE 76, 055201 are from memory of the journal
records and should be checked against Crossref before submission.

---

## (f) Proposed papers

### Paper Z-1: "Which early warnings survive a strict null? A benchmark across eleven oscillator families"

Scope: Z1 as it stands (18 transitions, 11 families, collective vs single unit, ROC AUC against the
equal-ramp-away null), Z3 as a section explaining why lag-1 autocorrelation fails for oscillatory observables,
and the FitzHugh-Nagumo ripple as the main mechanistic result. Target: Chaos or PRE.
Extra work:
- Read Rahjerdi et al. 2022 in full and position against it and Kato and Masuda 2026; reanalyse our Kuramoto
  runs with their V_S and <R_S> and with their Kc definition to show the circularity numerically (tau for <R>
  under their scoring vs AUC under ours).
- Extend the ripple analysis to theta neurons and Winfree (also spiking) and replace event interpolation by a
  protophase-to-phase transform (Kralemann et al. 2007) as a second fix besides period averaging.
- Finish A1 (N = 6400, 25600) so that the "N >= 6400" statement rests on completed runs; report the noise
  dependence (sigma = 0.05 vs 0.1) as a figure.
- Bootstrap confidence intervals on every AUC (already in `ewslib.py`) in the main table.

### Paper Z-2: "Event-based phases hide critical slowing down in spiking populations"

Scope: a short letter on c.3 alone: theory of the spike-period ripple in R built from interpolated event
phases (amplitude vs N, frequency dispersion and coupling), its dominance of the detrended variance, why it
shrinks before a desynchronising transition (so the variance falls), and two fixes (period averaging,
phase correction). Target: PRE Letter or Chaos Fast Track.
Extra work: an analytic ripple amplitude for a population with spread spiking periods (the interpolated phase
of each unit is piecewise linear, so R carries harmonics of the mean period weighted by locking); check
against FitzHugh-Nagumo, theta, Winfree and Chialvo maps (the last to connect with Leyva 2025); one real
spiking dataset if available (for example published multi-electrode recordings) to show the ripple exists in
data.

### Paper O-1: "Does frequency-degree correlation cause explosive synchronisation beyond phase oscillators? Controls for five families"

Scope: O5 with proper controls, framed around the missing shuffled control in Leyva 2012 and the regime
dependence in Chen 2013. Target: PRE.
Extra work (all required before submission):
- Quasi-static sweeps or three ramp speeds with extrapolation (as O3c), at least 10 seeds.
- Down-sweeps from random initial phases and stationarity tests inside the loop (hold coupling, long runs).
- Reproduce Leyva 2012 Fig. 1 with their piecewise Roessler and Delta_alpha, then shuffle; reproduce Chen 2013
  Fig. 1(b), then shuffle and count spiking units.
- Keep every hub inside the master stability window (lambda k_max < 4.6 for x-coupled Roessler) or switch to
  y-coupling as Leyva 2025 does, and say which.

### Paper O-2 (with Z2): "Herding folds across oscillator families: phase reduction, sparse graphs and slow passage"

Scope: O3/O3e/O6 (phase-reduction prediction of each family's herding fold, Stuart-Landau amplitude
correction of 10%, the shared 1/k coefficient for Kuramoto and Stuart-Landau, van der Pol 18% higher,
Roessler outside the 1/k law) plus Z2 (slow passage up to N = 819200, Berglund-Gentz exponent and crossover).
Target: Chaos or Physica D.
Extra work:
- Derive the 2-cumulant (Tyulkina et al. 2018) correction for the herding fold with Gaussian noise and test it
  against the 1.3% and 4.0% offsets; this turns the Gaussian-noise anomaly into a validated prediction.
- Test the Berglund-Gentz crossover N* ~ 1/eps with two more ramp speeds.
- Rerun Z2 without the frequency clip (job 38512) and replace the large-N variance numbers.
- Explain the van der Pol 18% excess from its higher Fourier harmonics of the phase interaction function.
