# Literature review: synchrony and correlation as early warnings of market crashes (Paper B)

Date: 2026-10-25. Scope: the market half of the project ("Paper B": if a crash is modelled as explosive
desynchronisation, does synchrony measured from prices warn of crashes?). Companion to
`docs/PRIOR_ART.md` (which covers Paper A, the oscillator theory) and to `docs/PREREGISTRATION_B.md`.

PDFs of the open-access papers are in `references/14_lit_review_markets/` and are listed with URLs in
`references/14_lit_review_markets/SOURCES.md`.

## How this review was done, and its limits

1. Web searches (general, arXiv-indexed, PLoS, Nature, NBER, RePEc) on: explosive synchronisation and
   markets; correlation, absorption ratio and eigenvalue measures as crash warnings; Kuramoto and Hilbert
   phase synchrony of returns; intraday co-movement and the 2010 flash crash; VPIN; critiques
   (volatility bias in correlations, prosecutor's fallacy in early-warning studies, multiple testing).
   The arXiv API (`export.arxiv.org`) returned HTTP 429 (rate limit) for the whole session, so arXiv was
   searched through web search only.
2. Reading depth is stated per paper in the table (column "read"). "Full" means the full text was read
   for the parts used here; "partial" means abstract plus targeted sections; "abstract" means only
   abstract or secondary summaries; "memory" means details recalled from prior knowledge and not
   re-verified against the paper in this session. Anything marked "memory" must be checked before it
   goes into a manuscript.
3. Lee et al. (2025) was read in full: main text (PMC12595435 full-text XML and the Oxford ORA copy of
   the PNAS PDF), the full SI Appendix (Europe PMC supplementary bundle), and the authors' public code
   (`github.com/pangyuj/explosive-sync-proximity`, folder `stock_market_analysis/Code`).
4. Not a systematic review. Google Scholar forward citations of Lee et al. (2025), Kritzman et al.
   (2011) and Preis et al. (2012) were not followed. Do that before submission.

## (a) Table of relevant papers

Columns: **vol ctrl** = does the paper control for volatility when it attributes signal to correlation
or synchrony; **OOS** = out-of-sample (held-out period or events) evaluation; **read** = depth (see above).

| # | key | citation | what it does | vol ctrl | OOS | relation to Paper B | read |
|---|---|---|---|---|---|---|---|
| 1 | lee2025 | Lee, Kim, Kim, Oh, Joo, Park, Pal, Tracey, Warnaby, Sleigh, Mashour, PNAS 122(44) e2505434122 (2025); bioRxiv 2024.11.28.625924 | "ES proximity" (kurtosis of the lag-2 autocorrelation of a Hilbert-phase Kuramoto order parameter) in 2006 predicts collapse and recovery speed of 39 national markets in 2008 | no | no | Closest conceptual prior: explosive synchronisation applied to markets. Different question (speed given a crisis, cross-section of countries) and different data | full, plus SI and code |
| 2 | kritzman2011 | Kritzman, Li, Page, Rigobon, J. Portfolio Management 37(4) 112-126 (2011) | Absorption ratio: variance share of the top fifth of principal components; standardized 15-day vs 1-year shift flags fragility before drawdowns | no | no (full-sample) | Our daily study includes it as a benchmark (B1.4). Standard industry precedent for "concentration of co-movement as a warning" | abstract + memory |
| 3 | preis2012 | Preis, Kenett, Stanley, Helbing, Ben-Jacob, Sci. Rep. 2, 752 (2012) | Mean correlation of DJIA components (1939 to 2010, point-in-time membership) rises linearly with normalized index return over the same window | no | n/a (descriptive) | The contemporaneous fact our test has to get beyond: co-movement is high during stress | partial |
| 4 | sandoval2012 | Sandoval Junior, De Paula Franca, Physica A 391(1) 187-208 (2012) | Eigen-structure of correlation matrices of world indices around 1987, 1998, 2001, 2008; high volatility goes with high correlation | no (states the link) | n/a | Daily, cross-market analogue of our daily study; descriptive | partial |
| 5 | forbesrigobon2002 | Forbes, Rigobon, J. Finance 57(5) 2223-2261 (2002); NBER w7267 | Shows correlation is mechanically biased upward when volatility rises; adjusted correlations find "no contagion, only interdependence" in 1987, 1994, 1997 | yes (parametric adjustment) | n/a | The theoretical reason our primary test is volatility-matched | partial |
| 6 | corsetti2005 | Corsetti, Pericoli, Sbracia, J. Int. Money Finance 24(8) 1177-1199 (2005) | The Forbes-Rigobon adjustment itself biases towards "interdependence" when shocks are partly idiosyncratic | critique of adjustment | n/a | Reason to prefer a non-parametric matched null over a parametric correction | memory |
| 7 | pollet2010 | Pollet, Wilson, J. Financial Economics 96(3) 364-380 (2010) | Splits market variance into average correlation and average stock variance; average correlation forecasts quarterly market returns | yes (decomposition) | partly | Prior art for "correlation beyond volatility"; outcome is expected returns, not crashes | abstract |
| 8 | longin2001 | Longin, Solnik, J. Finance 56(2) 649-676 (2001) | Extreme-value approach: correlation rises in bear markets, not with large absolute returns per se | yes (conditional on tails) | n/a | Asymmetry: correlation is linked to down moves, which matters for "crash" vs "ordinary peak" contrasts | memory |
| 9 | reigneron2011 | Reigneron, Allez, Bouchaud, Physica A 390(17) 3026-3035 (2011) | Index leverage effect decomposed into a volatility part and a correlation part: index drops raise future correlation | yes (decomposes) | n/a | Mechanism by which falls create correlation (reverse causation for a warning) | abstract |
| 10 | harmon2015 | Harmon, Lagi, de Aguiar, Chinellato, Braha, Epstein, Bar-Yam, PLoS ONE 10(7) e0131871 (2015); arXiv 1102.2620 | "Market mimicry" from a model fit to the fraction of stocks moving together; claims 4 of 4 crash clusters anticipated, 0 false positives (1985 to 2010) | compared, not controlled | no | Closest "co-movement predicts crashes" claim on US stocks | partial |
| 11 | pharasi2018 | Pharasi, Sharma, Chatterjee, Chakraborti, Leyvraz, Seligman, New J. Phys. 20, 103041 (2018); arXiv 1809.00885 | Clusters 20-day correlation matrices into "market states"; the state adjacent to the crash state is a precursor (S&P 500, Nikkei, 1985 to 2016) | no | no | Correlation-structure precursor; survivorship explicit (only stocks present for the whole period) | partial |
| 12 | zheng2012 | Zheng, Podobnik, Feng, Li, Sci. Rep. 2, 888 (2012) | Steepest rise of the first principal component of sector indices as a systemic-risk indicator | no | no | Same family as the absorption ratio and our `lead_eig_share` | partial |
| 13 | kenett2011 | Kenett, Shapira, Madi, Bransburg-Zabary, Gur-Gershgoren, Ben-Jacob, PLoS ONE 6(4) e19378 (2011) | Index cohesive force (correlation vs partial correlation given the index) for S&P 500, 1999 to 2010; transition to a "stiff" state after 2001 | no | no | Index-driven correlation as a fragility state | abstract |
| 14 | kenett2012 | Kenett, Raddant, Lux, Ben-Jacob, PLoS ONE 7(2) e31144 (2012) | Uniformity and volatility across six world markets 2000 to 2010 | descriptive | no | Shows how tightly correlation and volatility co-move | abstract |
| 15 | billio2012 | Billio, Getmansky, Lo, Pelizzon, J. Financial Economics 104(3) 535-559 (2012); NBER w16223 | PCA and Granger-causality networks of hedge funds, banks, brokers, insurers (monthly, 1994 to 2008) as systemic-risk measures | partly | partly (pre-2007 ranks vs crisis losses) | Systemic-risk connectedness; one crisis, monthly data | abstract + memory |
| 16 | guttal2016 | Guttal, Raghavendra, Goel, Hoarau, PLoS ONE 11(1) e0144198 (2016) | No critical slowing down before 1929, 1987, 2000, 2008 crashes in five indices; variance rises instead; crashes look like stochastic transitions | n/a | partial (checks windows away from crashes) | Supports our finding that volatility, not a bifurcation signature, carries the warning | partial |
| 17 | boettiger2012 | Boettiger, Hastings, Proc. R. Soc. B 279(1748) 4734-4739 (2012); arXiv 1210.1204 | Prosecutor's fallacy: selecting series because a transition happened inflates false-positive warning signals | n/a | n/a | Critique that applies to every "signal rose before crash X" study, including Lee and Kritzman | partial |
| 18 | harvey2016 | Harvey, Liu, Zhu, Rev. Financial Studies 29(1) 5-68 (2016); NBER w20592 | Multiple testing in empirical finance; recommends t above about 3 for new factors | n/a | n/a | Basis for our BH correction and for the multiplicity critique of Lee | memory |
| 19 | bailey2017 | Bailey, Borwein, Lopez de Prado, Zhu, J. Computational Finance 20(4) 39-69 (2017) | Probability of backtest overfitting | n/a | n/a | Thresholds chosen in-sample overfit; supports our train/test split | memory |
| 20 | calcagnile2018 | Calcagnile, Bormetti, Treccani, Marmi, Lillo, Quantitative Finance 18(2) 237-247 (2018); arXiv 1505.00704 | 1-minute data, 140 liquid US stocks per year (2001 to 2013): co-jumps (simultaneous over-threshold moves, thresholds relative to local volatility) became more synchronised; Hawkes model | yes (jumps scaled by local volatility) | n/a | Closest intraday synchrony measure; a benchmark we have not yet run | partial |
| 21 | borland2010 | Borland, Hassid, arXiv 1010.4917 (2010) | Intraday cross-sectional statistics: correlation rises in panics, memory in correlation, dispersion and correlation both jump on 6 May 2010 | no | n/a | Intraday evidence that co-movement is coincident with panic | abstract |
| 22 | kirilenko2017 | Kirilenko, Kyle, Samadi, Tuzun, J. Finance 72(3) 967-998 (2017); CFTC working version 2014 | Trader-level E-mini data on 6 May 2010: HFTs did not cause the crash and did not change behaviour, but amplified it by demanding immediacy | n/a | n/a | Microstructure account; our preregistration cites it for "no causal claim" | partial |
| 23 | easley2011 | Easley, Lopez de Prado, O'Hara, J. Portfolio Management 37(2) 118-128 (2011); also Rev. Financial Studies 25(5) 1457-1493 (2012) | VPIN (order-flow toxicity) reported elevated hours before the flash crash | no | no | Microstructure early-warning claim; disputed (row 24) | memory |
| 24 | andersen2014 | Andersen, Bondarenko, J. Financial Markets 17, 1-46 (2014); CREATES rp 2011-50 | VPIN reaches its historical high only after the crash; its forecasting content comes mainly from a mechanical link to trading intensity and volatility | yes (the point of the paper) | yes | Template for a critique: re-run the indicator against a volatility benchmark and look at timing | partial |
| 25 | araujo2004 | Araujo, Louca, arXiv cond-mat/0403333 (2004) | S&P 500 sector synchronisation and desynchronisation during 1987 and 2001 crises | no | n/a | Early descriptive "synchronisation during crises" | abstract |
| 26 | song2024 | Song, Li, EPJ Data Science 13, 16 (2024) | Multiplex recurrence-network indicators as early warnings for crashes in Chinese and US constituents | not checked | not checked | Recent EWS claim of the same family (already in `references/06_finance/`) | abstract |
| 27 | demirer2019 | Demirer, Leggio, Lien, Finance Research Letters 31 (2019) | No herding before the 2010 flash crash at one-minute resolution | n/a | n/a | Cited in `research/index.html`; not re-verified here | not read |

Count: 27 papers, of which 9 are judged closest (rows 1, 2, 3, 5, 10, 11, 16, 20, 24) and are discussed
below. Rows 17 to 19 are the methodological critiques the closest papers are measured against.

## (b) Detailed notes on the closest papers

Terminology. A **mistake** is an error that changes or misstates what the evidence shows (wrong numbers,
claims contradicted by the paper's own tables, choices made after seeing outcomes and not disclosed). A
**limitation** is a design constraint the paper either states or that is unavoidable with its data;
it narrows what can be concluded but is not an error.

### 1. Lee et al. (2025), PNAS: the closest prior work

**Citation.** UnCheol Lee, Hyoungkyu Kim, Minkyung Kim, Gabjin Oh, Pangyu Joo, Ayoung Park, Dinesh Pal,
Irene Tracey, Catherine E. Warnaby, Jamie Sleigh, George A. Mashour. "Proximity to explosive
synchronization determines network collapse and recovery trajectories in neural and economic crises."
PNAS 122(44), e2505434122 (2025), doi:10.1073/pnas.2505434122, PMC12595435. Preprint bioRxiv
10.1101/2024.11.28.625924 (posted 3 Dec 2024). Code: github.com/pangyuj/explosive-sync-proximity.

**What they did.**
- Model: networks of feedback-modulated Stuart-Landau oscillators (78-node connectome, 1,000-node
  random, small-world and scale-free graphs) where a feedback strength Z moves the synchronisation
  transition from second order towards first order ("ES proximity"). At the critical point (peak of the
  pair correlation function) they show that the distribution of windowed autocorrelations (ACF) of the
  order parameter becomes heavy-tailed as Z grows, and that its kurtosis (KACF) correlates with how
  fast the network loses and regains its critical state after a perturbation (Figs. 2 and 3).
- EEG: 16 subjects under propofol; band-passed alpha (8 to 13 Hz) Hilbert phases; KACF at lag 50 in
  10-s windows; Spearman rho = -0.67 with induction time and 0.59 with recovery time (Fig. 4).
- Markets (Fig. 5; SI S3; code `stock_market_analysis/Code/*.m`): daily returns of individual stocks
  (Compustat Global and CRSP) for 39 countries, July 2005 to December 2006. Each stock's raw daily
  return series is Hilbert-transformed (`F_cal_GlobalOrderParameter1.m`: `hilbert(X(:,c))`, no band-pass),
  unit phasors are averaged across stocks to give a daily order parameter r(t), its lag-2 ACF is computed
  in 120-day windows advanced one day at a time, and the kurtosis of those roughly 250 ACF values over
  2006 is the country's ES proximity. Outcomes come from the national index: collapse rate R1 and
  recovery rate R2 over alpha = 100 days after the index maximum and minimum inside OECD recession dates,
  normalised by the peak-to-trough depth, then response time = ln(1/R1), recovery time = ln(1/R2).
  Results: Spearman rho = -0.40 (response) and 0.49 (recovery), both reported "P < 0.001"; KACF also
  correlates with GDP per capita (rho = -0.36, P < 0.05).

**How it differs from Paper B.**
- Question: they ask how *fast* a market falls and recovers *given* a crisis, across countries. We ask
  whether synchrony *warns of the timing* of crashes within a market. Their design contains no timing
  prediction, no false alarms and no alarm rule.
- Events: one crisis (2008), one observation per country (n = 39). We have 35 intraday onsets over 24
  years (plus 324 daily onsets in 23 markets).
- Indicator: a second-order statistic (kurtosis of windowed autocorrelations of a phase order parameter)
  measured in a calm baseline year. Ours are level indicators (mean correlation, leading-eigenvalue
  share, correlation residual given volatility) measured in the 60 minutes before the event.
- Null: they compare against nothing (no surrogate, no volatility benchmark). Our primary test is against
  volatility-matched ordinary peaks.
- Theory: both papers use oscillator models with bistability. Theirs predicts slow recovery from
  hysteresis; ours predicts that synchrony should rise before a desynchronising cascade and tests it.

**Mistakes (in order of importance).**
1. *Reported significance does not match the paper's own SI or the arithmetic.* Fig. 5C and D (and the
   Results text) give rho = -0.40, P < 0.001 and rho = 0.49, P < 0.001. With n = 39 the two-sided
   p-values are about 0.012 and 0.0015 (t approximation; computed here), and SI Table S5 (row KACF2,
   alpha = 100) itself lists p = 0.01 and 0.00. Neither main-text P < 0.001 is supported; the response
   result is p of about 0.01.
2. *"Not sensitive to alpha" is contradicted by SI Table S5.* The Methods state the results "were not
   sensitive to alpha". In Table S5 the KACF2 response correlation is not significant at alpha = 40, 50
   and 60 days (rho = -0.13, -0.15, -0.27; p = 0.45, 0.36, 0.10). Table S4 shows the same pattern for all
   three baseline periods.
3. *Undisclosed specification search and outcome-dependent choice of indicator.* SI Table S5 alone
   reports 4 indicators (KPCF, KACF at lags 1, 2, 3) x 9 alpha values x 2 outcomes = 72 correlations;
   Table S4 adds 54 and Table S6 14. KACF1 is null in all 18 of its cells; KACF3 is null for recovery
   in all 9; KPCF is null for recovery in 8 of 9. The main text reports only lag 2 and alpha = 100, the SI
   says "We found that the KACF with a time lag of 2 showed a significant correlation", i.e. the lag was
   chosen because it worked, and the main text says the ACF was preferred over the PCF because it "aligns
   more consistently with the real-world data we tested". No correction for multiplicity is applied and
   the selection is not presented as exploratory. The code (`Main_Make_Result_StockMarket_2511.m`,
   `f_cal_Response_Recovery.m`) computes 6 indicators x 17 alpha values x 2 outcomes = 204 correlations.
4. *Effective sample size of the indicator is ignored.* The SI general method says windows move with
   "half-size window overlap", but the stock analysis (SI S3.2 and `F_cal_criticality.m`) steps the
   120-day window by one day. Over 2006 that gives about 250 ACF values of which roughly two are
   independent, and the kurtosis of a strongly autocorrelated sequence of about two independent values
   is a very noisy statistic. No uncertainty for KACF per country is reported.
5. Minor reporting errors: SI Tables S4 and S5 captions call the parenthesised values "t-statistics"
   although they are p-values; the main text dates the Lehman collapse to September 2007 (it was
   September 2008).

**Limitations (design constraints; some stated by the authors).**
1. *One shock, non-independent observations.* All 39 markets respond to the same 2008 shock; index
   maxima cluster in mid to late 2007 and minima on 24 Oct 2008 to 9 Mar 2009 (SI Table S3). The Spearman
   p-values treat the 39 countries as independent, which overstates significance. Not acknowledged.
2. *No volatility, liquidity or market-size control.* Emerging markets were more volatile and less
   liquid in 2006 and fell faster in 2008. KACF may proxy for any of these. The number of stocks per
   country ranges from 12 (Russia) to 2,664 (Japan) (Table S3); the finite-size fluctuations of a phase
   order parameter scale as 1/sqrt(N), and the distribution of r(t) and its ACF depend on N. Across the
   39 rows of Table S3 the number of stocks has Spearman 0.26 with GDP per capita and medians of 70
   (emerging) vs 147 (developed) (computed here). This does not show N drives the result, but it is an
   uncontrolled path to the GDP correlation.
3. *The phase is not an oscillator phase.* For EEG the signal is band-passed before the Hilbert
   transform; for stocks the raw daily return series, which is close to white noise, is transformed
   directly. The resulting "instantaneous phase" has no clear oscillator meaning (Bedrosian conditions
   fail for broadband signals), so r(t) is closer to a weighted cross-sectional sign agreement than to a
   Kuramoto order parameter. The link from the Stuart-Landau model to markets is therefore by analogy.
   The authors state the KACF to ES link "lacks analytical justification" (Limitations).
4. *Shared denominator in the outcomes.* Both R1 and R2 are divided by the same peak-to-trough depth
   (`f_cal_Response_Recovery.m`), so response and recovery times are not independent outcomes and both
   inherit any relation between KACF and drawdown depth.
5. *Researcher-defined crisis windows.* OECD recession dates per country, US dates for countries without
   an OECD indicator, Denmark and Greece excluded for timing reasons (SI S3.1), alpha chosen as "an
   appropriate period".
6. *No out-of-sample test, no preregistration, no power analysis.* At n = 39, the smallest Spearman
   correlation detectable with 80% power at two-sided alpha 0.05 is about 0.44 (Fisher z; computed
   here), so rho = -0.40 is below the design's own detection limit even before multiplicity.
7. *Survivorship* is less of a problem than in many studies: the baseline uses firms trading in 2005 to
   2006 and the outcome uses published national indices. Stated here for fairness.

**Our own prior engagement.** `research/index.html`, section E3, reports an implementation of their
estimator on constituent-level Yahoo data in 23 markets: the 2008 result did not reproduce (n = 53,
collapse rho = -0.207, CI [-0.435, 0.015]; recovery rho = -0.017), while across 262 other events the
recovery relation survived weakly (rho = 0.133, CI [0.038, 0.215]). That analysis is not an exact
replication (different data, survivorship in Yahoo constituents) and was not preregistered.

### 2. Kritzman, Li, Page, Rigobon (2011), absorption ratio

**Citation.** Mark Kritzman, Yuanzhen Li, Sebastien Page, Roberto Rigobon. "Principal components as a
measure of systemic risk." Journal of Portfolio Management 37(4), 112-126 (2011),
doi:10.3905/jpm.2011.37.4.112. PDF not downloaded (journal paywalled; the SSRN working version could
not be fetched by script).

**What they did** (from abstract, secondary sources and memory; verify before citing details): absorption
ratio AR = fraction of total variance explained by a fixed number of eigenvectors (about one fifth of
the number of assets) of the covariance matrix of US industry returns, estimated in a 500-day
exponentially weighted window; a "standardized shift" compares the 15-day and 1-year averages of AR in
units of its standard deviation. They report that most large US drawdowns were preceded by AR spikes and
that conditional returns after large upward shifts are low; a timing strategy that de-risks on spikes
beats buy-and-hold in-sample.

**Differences.** Daily, US industries, full-sample; no event-level discrimination test. Our daily study
computes AR as a benchmark (B1.4); our intraday `lead_eig_share` is the single-eigenvector analogue.

**Mistakes and weaknesses.**
- Conditioning on the outcome (limitation bordering on mistake): "most drawdowns were preceded by
  spikes" is P(signal | crash). Secondary reports quote the paper itself saying not all rises were
  followed by drawdowns. What matters is P(crash | signal) and the false-alarm rate, which is the
  prosecutor's fallacy pointed out by Boettiger and Hastings (2012).
- No volatility control: AR is a covariance-based variance share; when a common factor's variance jumps,
  AR rises mechanically (Forbes and Rigobon 2002 logic). Whether AR adds to plain volatility was not
  tested.
- Thresholds (1 sigma), window lengths and the number of eigenvectors were chosen within the same sample
  used for evaluation (limitation; no out-of-sample period).

### 3. Preis, Kenett, Stanley, Helbing, Ben-Jacob (2012)

**Citation.** Tobias Preis, Dror Y. Kenett, H. Eugene Stanley, Dirk Helbing, Eshel Ben-Jacob.
"Quantifying the behavior of stock correlations under market stress." Scientific Reports 2, 752 (2012),
doi:10.1038/srep00752.

**What they did.** Daily closes of the 30 DJIA components with point-in-time membership (their Fig. 1
tracks index changes), March 1939 to December 2010. For windows of 10 to 60 days, mean pairwise
correlation is plotted against the normalised index return of the same window; the data collapse on
two straight lines with a minimum near zero return (Figs. 3 and 4). Shuffling tests remove the effect.

**Differences.** Contemporaneous, not predictive; no crash events.

**Mistakes and weaknesses.**
- Partly tautological relation (unacknowledged): the index return over a window is the average of
  component returns, so a large absolute index return over that window requires the components to move
  together. The variance of an equal-weight index is approximately (average variance) x (1/N + (1 - 1/N)
  x mean correlation). Conditioning correlation on the index return of the same window therefore builds
  in part of the result. The shuffling tests (Fig. 4) destroy all dependence and do not separate the
  mechanical part from any behavioural part.
- The paper's suggestion that the relation can help anticipate diversification breakdown is not tested
  forward in time.
- Strength: no survivorship bias (historical constituents), long sample.

### 4. Forbes and Rigobon (2002)

**Citation.** Kristin J. Forbes, Roberto Rigobon. "No contagion, only interdependence: measuring stock
market comovements." Journal of Finance 57(5), 2223-2261 (2002), doi:10.1111/0022-1082.00494; NBER
Working Paper 7267 (1999).

**What they did.** Show (NBER w7267 eq. 7 and Appendix A) that under a linear model the conditional
correlation rho measured in a high-volatility period relates to the unconditional rho* by
rho = rho* x sqrt((1 + delta) / (1 + delta x rho*^2)), with delta the relative rise in variance of the
source market, so measured correlation rises with volatility even when the transmission mechanism is
unchanged. After adjustment, the 1987, 1994 and 1997 crises show no rise in correlation.

**Relevance.** This is the reason Paper B's primary hypothesis is "correlation beyond volatility". Our
volatility matching is a non-parametric version of their adjustment, and our `corr_given_vol` residual
is a regression version.

**Weaknesses (from the literature, not our reading).** Corsetti, Pericoli and Sbracia (2005) show the
adjustment assumes the crisis shock is purely common; with idiosyncratic shocks it over-corrects and is
biased towards "interdependence". Crisis windows are chosen by the analyst. Implication for us: matching
on index volatility could also over-correct if part of the extra volatility before crashes is itself a
product of rising correlation. See Risks.

### 5. Harmon et al. (2015), "market mimicry"

**Citation.** Dion Harmon, Marco Lagi, Marcus A. M. de Aguiar, David D. Chinellato, Dan Braha, Irving R.
Epstein, Yaneer Bar-Yam. "Anticipating economic market crises using measures of collective panic."
PLoS ONE 10(7), e0131871 (2015), doi:10.1371/journal.pone.0131871; arXiv 1102.2620 (2011).

**What they did.** Fit a model of the daily distribution of the fraction of stocks moving up (Russell
3000 stocks on NYSE and Nasdaq) to obtain a mimicry parameter U; a crash is "predicted" in the year
following a two-standard-deviation drop in U after a period of increase. Over 1985 to 2010 they report
4 true positives, 0 false positives and 0 false negatives for four clusters of large one-day drops, with
a combinatorial p = 0.000067 against a random yearly classifier (their eq. 4 and Table 2). Volatility and
mean correlation, given a similar "sharp increase" rule, score 3 TP, 2 FP, 1 FN (volatility p = 0.014);
average covariance anticipates all events but also flags 1991.

**Differences.** Yearly prediction windows, daily data, four events; we use minute data, 35 events and a
fixed alarm horizon of 60 minutes.

**Mistakes and weaknesses.**
- The decision rule (two standard deviations, "after periods of increase", a one-year window, sign-change
  condition) was designed on the same 25 years and four events it is evaluated on, and the p-value
  treats the rule as fixed in advance (mistake: the reported p does not account for rule selection).
- Eight crashes were merged into four clusters by the authors; the evaluation unit is the analyst's
  choice (event-definition degree of freedom).
- n = 4: one hit or miss changes the conclusion. No out-of-sample period.
- The comparison with volatility is the right instinct but uses a different functional form for each
  indicator rather than a matched or incremental test.

### 6. Pharasi et al. (2018), market states

**Citation.** Hirdesh K. Pharasi, Kiran Sharma, Rakesh Chatterjee, Anirban Chakraborti, Francois
Leyvraz, Thomas H. Seligman. "Identifying long-term precursors of financial market crashes using
correlation patterns." New Journal of Physics 20, 103041 (2018), doi:10.1088/1367-2630/aae7e0; arXiv
1809.00885.

**What they did.** 20-day correlation matrices (power-mapped to reduce noise) of 194 S&P 500 and 165
Nikkei 225 stocks, 1985 to 2016; k-means on similarity matrices defines market states; the state that
precedes the "crash state" is proposed as a precursor.

**Mistakes and weaknesses.**
- Survivorship bias (stated, but consequences not discussed): only "stocks which are present in the
  indices for the entire duration" (their Section 2) are used, a list fixed in 2016. Survivors tend to
  be firms that did not collapse in the crises being studied.
- The number of states and clustering parameters are chosen on the full sample; states are labelled
  with hindsight. No false-alarm rate, no held-out period.
- No volatility control: market states differ mainly in mean correlation, which tracks volatility.

### 7. Guttal et al. (2016), no critical slowing down

**Citation.** Vishwesha Guttal, Srinivas Raghavendra, Nikunj Goel, Quentin Hoarau. "Lack of critical
slowing down suggests that financial meltdowns are not critical transitions, yet rising variability
could signal systemic risk." PLoS ONE 11(1), e0144198 (2016), doi:10.1371/journal.pone.0144198.

**What they did.** Detrended Dow Jones, S&P 500, Nasdaq, DAX, FTSE before 1929, 1987, 2000, 2008 and
others; lag-1 autocorrelation did not rise, variance and low-frequency power did; a stochastic model
with noise-induced jumps reproduces this. They test robustness to window and bandwidth and check
indicator trends away from crashes, and use surrogate data (phase-randomised and AR(1)).

**Relevance and weaknesses.** Their result agrees with ours (volatility is the strong unmatched
predictor; nothing bifurcation-specific beyond it). Weaknesses: a small number of crashes, Kendall-tau
trend statistics depend on several tuning parameters (they scan them, which is good practice), and the
"rising variance" signal is itself retrospective, with no alarm rule or false-alarm count.

### 8. Calcagnile et al. (2018), co-jump synchronisation

**Citation.** Lucio Maria Calcagnile, Giacomo Bormetti, Michele Treccani, Stefano Marmi, Fabrizio Lillo.
"Collective synchronization and high frequency systemic instabilities in financial markets."
Quantitative Finance 18(2), 237-247 (2018), doi:10.1080/14697688.2017.1403141; arXiv 1505.00704.

**What they did.** 1-minute data, 140 highly liquid Russell 3000 stocks re-selected each year,
2001 to 2013. Returns are scaled by an intraday-pattern and local-volatility estimate; a co-jump of
multiplicity M is M stocks exceeding a threshold in the same minute. High-multiplicity co-jumps became
more frequent after 2001, fewer than 40% coincide with scheduled macro news, and a multivariate Hawkes
model reproduces the cross-excitation.

**Relevance.** The closest intraday "synchrony" measure, and it is volatility-scaled by construction.
Descriptive: no crash forecasting. We have not yet run it as a benchmark (flagged in
`research/index.html`).

### 9. Andersen and Bondarenko (2014), the VPIN critique

**Citation.** Torben G. Andersen, Oleg Bondarenko. "VPIN and the flash crash." Journal of Financial
Markets 17, 1-46 (2014); CREATES Research Paper 2011-50. Replies: Easley, Lopez de Prado, O'Hara, J.
Financial Markets 17, 47-52 (2014); Andersen and Bondarenko, ibid. 53-64.

**What they did.** Recomputed VPIN on E-mini data; it reaches its historical high after the 6 May 2010
crash, not before; its predictive content for short-run volatility is weak once trading intensity and
realised volatility are controlled for.

**Relevance.** The same structure as our critique of synchrony: an intuitive indicator, a single famous
event, and a mechanical link to volume or volatility. It is the precedent for publishing a critique of a
crash-warning indicator in a good journal, and for the volatility benchmark.

### Short notes on the rest

- **Sandoval and Franca (2012)**: descriptive; daily world indices are non-synchronous across time
  zones (they include lagged indices to handle it); correlation and volatility are shown to co-move,
  which they report rather than control.
- **Kirilenko et al. (2017)**: the microstructure account of 6 May 2010 relies on trader-level data that
  synchrony indicators cannot see; it argues against reading price synchrony as a causal mechanism.
- **Zheng et al. (2012), Kenett et al. (2011, 2012), Billio et al. (2012)**: principal-component and
  network connectedness measures that rise around 2008. One crisis dominates each evaluation; none
  matches on volatility; Billio et al. use hedge-fund indices whose databases have survivorship and
  backfill biases (their own data discussion; verify).
- **Pollet and Wilson (2010)**: the cleanest prior separation of correlation from variance, applied to
  expected returns. Any claim that "correlation beyond volatility" is a new idea would be wrong.

## (c) What this project adds, and where it overlaps

**Adds (relative to the papers above).**
1. *A volatility-matched null at the event level.* Crash onsets are compared with ordinary local peaks
   matched one to one on pre-event volatility (Hungarian assignment, caliper 0.25 in log volatility).
   None of Lee, Kritzman, Preis, Harmon or Pharasi has a volatility benchmark of any kind; Forbes and
   Rigobon and Pollet and Wilson control for volatility but in other settings (contagion, expected
   returns). Andersen and Bondarenko do it for VPIN.
2. *A design fixed before the corrected run, with stated power.* Primary hypothesis, event rule, horizon,
   matching and multiplicity correction (BH across all indicator by test cells) were written before the
   run (`docs/PREREGISTRATION_B.md`), and the minimum detectable AUC is reported (0.70 matched, 0.63
   unmatched). Caveat: the document is not timestamped and earlier versions of the analysis had been
   seen (its own disclosure section), so this is a pre-specified design, not a registered report.
3. *Out-of-sample alarm evaluation with false alarms counted.* Thresholds from training minutes (before
   2016), recall and false alarms per crash caught on test minutes, realised alarm rates. Lee, Kritzman,
   Harmon and Pharasi report no false-alarm rate on held-out data.
4. *Minute data across 24 years.* Intraday onsets rather than one or a few historic crises.
5. *Oscillator-model grounding.* The indicators are the observables a desynchronisation-cascade model
   would predict, so a null is informative about the model, not just about an indicator.

**Our result, stated plainly.** 35 onsets (23 train, 12 test). Unmatched, volatility (AUC 0.986) and mean
correlation (0.915) both separate crashes from ordinary peaks; crash windows have 4.5 times the
volatility of peak windows. Against volatility-matched peaks (25 pairs) the primary indicator
`corr_given_vol` has AUC 0.587 [0.432, 0.746] and mean correlation 0.602 [0.445, 0.757]: not detected,
and the design could only detect AUC of about 0.70 or more. This is "not detected at this power", not
evidence of absence. Volatility alone is the strong, simple predictor.

**Overlap, honestly.**
- "Correlation rises in crises" is old (Longin and Solnik 2001; Sandoval and Franca 2012; Preis et al.
  2012). "Correlation is biased by volatility" is old (Forbes and Rigobon 2002). "Variance, not slowing
  down, rises before crashes" is published (Guttal et al. 2016). "Separate correlation from variance" is
  published for returns (Pollet and Wilson 2010). Our contribution is the combination at the event level
  with a matched null and pre-specified out-of-sample alarms on minute data, not any of these ideas.
- "Explosive synchronisation in markets" is now claimed by Lee et al. (2025) in PNAS. Any Paper B
  manuscript must cite it prominently and explain that it tests a different question.
- A negative, underpowered result is hard to publish alone. Its value rises if framed as a methods
  contribution (how to test crash warnings) or paired with a replication or critique.

## (d) Risks

1. **Power.** 25 matched pairs and 12 test onsets, 9 of which fall in four weeks of March to April 2020.
   The effective number of independent test events is closer to four episodes (Dec 2018, Mar 2020,
   Nov 2022, Apr 2025). A reviewer can dismiss the null as uninformative. The MDE statement helps but
   does not fix it.
2. **Deviation from the written design.** `PREREGISTRATION_B.md` says 40 fixed large caps; the run in
   `results/paperB_intraday/` has 23 tickers. This must be disclosed as a deviation with its reason.
3. **No timestamp.** The design document is not committed or deposited. Reviewers will treat the study as
   exploratory unless the next analysis is registered (OSF, or a tagged commit) before it is run.
4. **Matching may over-correct (Corsetti et al. 2005 logic).** If rising correlation is what raises index
   volatility before a crash, matching on index volatility removes part of the very signal tested. A
   matched null is conservative for "correlation beyond volatility" by design; the paper must say so, and
   a sensitivity analysis that matches on average single-stock volatility (which is not mechanically
   driven by correlation) should be added.
5. **Caliper selection.** 25 of 35 onsets were matched; the dropped ones are presumably the most volatile
   (largest) crashes. The matched estimand is therefore "crashes from moderately volatile states", which
   must be stated. Residual imbalance remains (volatility itself has matched AUC 0.53).
6. **Survivorship.** Today's large caps (stated in the design). Direction of bias is argued but not
   measured.
7. **Overlap risk with Lee et al.** A referee may see our oscillator-model framing as derivative of Lee
   et al. (and of Filatrella-style feedback models). The distinction (timing warnings vs collapse speed;
   desynchronisation cascade vs ES proximity) must be in the abstract.
8. **Errors in this review.** Details marked "memory" (Kritzman parameters, Corsetti, Longin and Solnik,
   Harvey et al., Bailey et al., Demirer et al.) were not re-verified this session.
9. **Tone.** The Lee critique concerns a PNAS paper by an established group. Any public critique must
   quote the SI tables exactly, distinguish mistakes from limitations as above, and be sent to the
   authors first (they posted code, which makes a constructive replication possible).

## (e) Proposed papers and notes

**P1. Replication and critique of Lee et al. (2025) with a volatility-matched null** (highest value).
- Claim to test: pre-crisis KACF predicts collapse and recovery speed beyond pre-crisis volatility,
  liquidity and number of stocks, and beyond the specification chosen.
- Work: (i) reproduce their numbers from their posted code and processed data (`GlobalOrderParameter.mat`,
  `ACF_Window120.mat` are on GitHub); (ii) report the full 72-cell (or 204-cell) grid with BH or Holm
  correction and a specification curve; (iii) add partial Spearman controls for 2006 index volatility,
  zero-return share, number of stocks and drawdown depth; (iv) surrogate test: phase-randomise or
  time-shuffle stocks independently to get the null distribution of KACF for each country's N and
  volatility; (v) cluster-robust inference (regions or common-shock factor) and a leave-one-region-out
  check; (vi) out-of-sample: the same rule on the 2020 COVID crash and the 2011 euro crisis with KACF
  measured in 2019 and 2010; (vii) add a band-passed phase variant to show sensitivity to the Hilbert
  choice.
- Data: their GitHub files; for new crises, constituent daily returns for the same 39 countries
  (Compustat Global or Refinitiv through a university licence; Yahoo only as a fallback with
  survivorship stated). National index levels and OECD recession dates are free.
- Output: a short paper (PNAS Brief Report or a comment, or Physica A / Chaos regular article). Contact
  the authors first.

**P2. Power and design note for crash early-warning studies.**
- Content: closed-form and simulated minimum detectable AUC as a function of event count, matched pairs
  and event clustering (Hanley-McNeil plus a cluster bootstrap); the effective number of events when
  crashes cluster in time; how volatility matching changes power; a checklist (matched null, held-out
  thresholds, false alarms per hit, realised alarm rate, event-definition robustness, multiplicity).
  Apply it retrospectively to Lee (n = 39, one shock), Harmon (n = 4), Kritzman and our own study.
- Data: none new; simulations plus published numbers. Work: about two weeks. Venue: a methods note
  (for example Quantitative Finance letters, or Royal Society Open Science).

**P3. Confirmatory, registered extension of Paper B.**
- Register (OSF) before running: the same T1 and T2 tests on a larger, point-in-time universe (S&P 500
  historical constituents) and on a new period, with the matching on average single-stock volatility
  added as a co-primary (Risk 4), and Calcagnile co-jumps and the absorption ratio as benchmarks.
- Data: 1-minute bars for historical constituents (paid: e.g. TAQ through WRDS, or the HF Data Library
  already used, extended). Work: the pipeline exists; the main cost is data and the registration. Power
  target: enough onsets for MDE AUC of 0.60 (on the order of 100 matched pairs, which likely requires a
  broader universe or a lower fall threshold set in advance).

**P4. Negative-results paper (current data).** "Price synchrony does not warn of intraday crashes beyond
volatility: a volatility-matched test on 24 years of minute data", reporting the primary null with MDE,
the daily 22-market study and the oscillator indicators (mean correlation, absorption ratio, Hilbert
phase synchrony). Venue: PLoS ONE, Royal Society Open Science or Journal of Financial Econometrics
replication section. Needs: the deviation (23 vs 40 stocks) disclosed, Risk 5 sensitivity, and
P2's design checklist as the framing.

## (f) BibTeX

```bibtex
@article{lee2025esproximity,
  author  = {Lee, UnCheol and Kim, Hyoungkyu and Kim, Minkyung and Oh, Gabjin and Joo, Pangyu and Park, Ayoung and Pal, Dinesh and Tracey, Irene and Warnaby, Catherine E. and Sleigh, Jamie and Mashour, George A.},
  title   = {Proximity to explosive synchronization determines network collapse and recovery trajectories in neural and economic crises},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {122},
  number  = {44},
  pages   = {e2505434122},
  year    = {2025},
  doi     = {10.1073/pnas.2505434122},
  note    = {bioRxiv 10.1101/2024.11.28.625924; PMC12595435}
}
@article{kritzman2011absorption,
  author  = {Kritzman, Mark and Li, Yuanzhen and Page, Sebastien and Rigobon, Roberto},
  title   = {Principal Components as a Measure of Systemic Risk},
  journal = {Journal of Portfolio Management},
  volume  = {37},
  number  = {4},
  pages   = {112-126},
  year    = {2011},
  doi     = {10.3905/jpm.2011.37.4.112}
}
@article{preis2012stress,
  author  = {Preis, Tobias and Kenett, Dror Y. and Stanley, H. Eugene and Helbing, Dirk and Ben-Jacob, Eshel},
  title   = {Quantifying the Behavior of Stock Correlations Under Market Stress},
  journal = {Scientific Reports},
  volume  = {2},
  pages   = {752},
  year    = {2012},
  doi     = {10.1038/srep00752}
}
@article{sandoval2012crisis,
  author  = {Sandoval Junior, Leonidas and De Paula Franca, Italo},
  title   = {Correlation of financial markets in times of crisis},
  journal = {Physica A},
  volume  = {391},
  number  = {1},
  pages   = {187-208},
  year    = {2012},
  doi     = {10.1016/j.physa.2011.07.023},
  note    = {arXiv:1102.1339}
}
@article{forbes2002nocontagion,
  author  = {Forbes, Kristin J. and Rigobon, Roberto},
  title   = {No Contagion, Only Interdependence: Measuring Stock Market Comovements},
  journal = {Journal of Finance},
  volume  = {57},
  number  = {5},
  pages   = {2223-2261},
  year    = {2002},
  doi     = {10.1111/0022-1082.00494},
  note    = {NBER Working Paper 7267}
}
@article{corsetti2005some,
  author  = {Corsetti, Giancarlo and Pericoli, Marcello and Sbracia, Massimo},
  title   = {Some contagion, some interdependence: More pitfalls in tests of financial contagion},
  journal = {Journal of International Money and Finance},
  volume  = {24},
  number  = {8},
  pages   = {1177-1199},
  year    = {2005},
  doi     = {10.1016/j.jimonfin.2005.08.012}
}
@article{pollet2010average,
  author  = {Pollet, Joshua M. and Wilson, Mungo},
  title   = {Average correlation and stock market returns},
  journal = {Journal of Financial Economics},
  volume  = {96},
  number  = {3},
  pages   = {364-380},
  year    = {2010},
  doi     = {10.1016/j.jfineco.2010.02.011}
}
@article{longin2001extreme,
  author  = {Longin, Francois and Solnik, Bruno},
  title   = {Extreme Correlation of International Equity Markets},
  journal = {Journal of Finance},
  volume  = {56},
  number  = {2},
  pages   = {649-676},
  year    = {2001},
  doi     = {10.1111/0022-1082.00340}
}
@article{reigneron2011leverage,
  author  = {Reigneron, Pierre-Alain and Allez, Romain and Bouchaud, Jean-Philippe},
  title   = {Principal regression analysis and the index leverage effect},
  journal = {Physica A},
  volume  = {390},
  number  = {17},
  pages   = {3026-3035},
  year    = {2011},
  doi     = {10.1016/j.physa.2011.04.002},
  note    = {arXiv:1011.5810}
}
@article{harmon2015panic,
  author  = {Harmon, Dion and Lagi, Marco and de Aguiar, Marcus A. M. and Chinellato, David D. and Braha, Dan and Epstein, Irving R. and Bar-Yam, Yaneer},
  title   = {Anticipating Economic Market Crises Using Measures of Collective Panic},
  journal = {PLoS ONE},
  volume  = {10},
  number  = {7},
  pages   = {e0131871},
  year    = {2015},
  doi     = {10.1371/journal.pone.0131871},
  note    = {arXiv:1102.2620}
}
@article{pharasi2018precursors,
  author  = {Pharasi, Hirdesh K. and Sharma, Kiran and Chatterjee, Rakesh and Chakraborti, Anirban and Leyvraz, Francois and Seligman, Thomas H.},
  title   = {Identifying long-term precursors of financial market crashes using correlation patterns},
  journal = {New Journal of Physics},
  volume  = {20},
  pages   = {103041},
  year    = {2018},
  doi     = {10.1088/1367-2630/aae7e0},
  note    = {arXiv:1809.00885}
}
@article{zheng2012crosscorr,
  author  = {Zheng, Zeyu and Podobnik, Boris and Feng, Ling and Li, Baowen},
  title   = {Changes in Cross-Correlations as an Indicator for Systemic Risk},
  journal = {Scientific Reports},
  volume  = {2},
  pages   = {888},
  year    = {2012},
  doi     = {10.1038/srep00888}
}
@article{kenett2011icf,
  author  = {Kenett, Dror Y. and Shapira, Yoash and Madi, Asaf and Bransburg-Zabary, Sharron and Gur-Gershgoren, Gitit and Ben-Jacob, Eshel},
  title   = {Index Cohesive Force Analysis Reveals That the US Market Became Prone to Systemic Collapses Since 2002},
  journal = {PLoS ONE},
  volume  = {6},
  number  = {4},
  pages   = {e19378},
  year    = {2011},
  doi     = {10.1371/journal.pone.0019378}
}
@article{kenett2012uniformity,
  author  = {Kenett, Dror Y. and Raddant, Matthias and Lux, Thomas and Ben-Jacob, Eshel},
  title   = {Evolvement of Uniformity and Volatility in the Stressed Global Financial Village},
  journal = {PLoS ONE},
  volume  = {7},
  number  = {2},
  pages   = {e31144},
  year    = {2012},
  doi     = {10.1371/journal.pone.0031144}
}
@article{billio2012econometric,
  author  = {Billio, Monica and Getmansky, Mila and Lo, Andrew W. and Pelizzon, Loriana},
  title   = {Econometric measures of connectedness and systemic risk in the finance and insurance sectors},
  journal = {Journal of Financial Economics},
  volume  = {104},
  number  = {3},
  pages   = {535-559},
  year    = {2012},
  doi     = {10.1016/j.jfineco.2011.12.010},
  note    = {NBER Working Paper 16223}
}
@article{guttal2016nocsd,
  author  = {Guttal, Vishwesha and Raghavendra, Srinivas and Goel, Nikunj and Hoarau, Quentin},
  title   = {Lack of Critical Slowing Down Suggests that Financial Meltdowns Are Not Critical Transitions, yet Rising Variability Could Signal Systemic Risk},
  journal = {PLoS ONE},
  volume  = {11},
  number  = {1},
  pages   = {e0144198},
  year    = {2016},
  doi     = {10.1371/journal.pone.0144198}
}
@article{boettiger2012prosecutor,
  author  = {Boettiger, Carl and Hastings, Alan},
  title   = {Early warning signals and the prosecutor's fallacy},
  journal = {Proceedings of the Royal Society B},
  volume  = {279},
  number  = {1748},
  pages   = {4734-4739},
  year    = {2012},
  doi     = {10.1098/rspb.2012.2085},
  note    = {arXiv:1210.1204}
}
@article{harvey2016cross,
  author  = {Harvey, Campbell R. and Liu, Yan and Zhu, Heqing},
  title   = {... and the Cross-Section of Expected Returns},
  journal = {Review of Financial Studies},
  volume  = {29},
  number  = {1},
  pages   = {5-68},
  year    = {2016},
  doi     = {10.1093/rfs/hhv059},
  note    = {NBER Working Paper 20592}
}
@article{bailey2017pbo,
  author  = {Bailey, David H. and Borwein, Jonathan M. and Lopez de Prado, Marcos and Zhu, Qiji Jim},
  title   = {The probability of backtest overfitting},
  journal = {Journal of Computational Finance},
  volume  = {20},
  number  = {4},
  pages   = {39-69},
  year    = {2017},
  doi     = {10.21314/JCF.2016.322}
}
@article{calcagnile2018cojumps,
  author  = {Calcagnile, Lucio Maria and Bormetti, Giacomo and Treccani, Michele and Marmi, Stefano and Lillo, Fabrizio},
  title   = {Collective synchronization and high frequency systemic instabilities in financial markets},
  journal = {Quantitative Finance},
  volume  = {18},
  number  = {2},
  pages   = {237-247},
  year    = {2018},
  doi     = {10.1080/14697688.2017.1403141},
  note    = {arXiv:1505.00704}
}
@misc{borland2010panic,
  author        = {Borland, Lisa and Hassid, Yoan},
  title         = {Market panic on different time-scales},
  year          = {2010},
  eprint        = {1010.4917},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.ST}
}
@article{kirilenko2017flash,
  author  = {Kirilenko, Andrei and Kyle, Albert S. and Samadi, Mehrdad and Tuzun, Tugkan},
  title   = {The Flash Crash: High-Frequency Trading in an Electronic Market},
  journal = {Journal of Finance},
  volume  = {72},
  number  = {3},
  pages   = {967-998},
  year    = {2017},
  doi     = {10.1111/jofi.12498}
}
@article{easley2011flashcrash,
  author  = {Easley, David and Lopez de Prado, Marcos M. and O'Hara, Maureen},
  title   = {The Microstructure of the ``Flash Crash'': Flow Toxicity, Liquidity Crashes, and the Probability of Informed Trading},
  journal = {Journal of Portfolio Management},
  volume  = {37},
  number  = {2},
  pages   = {118-128},
  year    = {2011},
  doi     = {10.3905/jpm.2011.37.2.118}
}
@article{easley2012toxicity,
  author  = {Easley, David and Lopez de Prado, Marcos M. and O'Hara, Maureen},
  title   = {Flow Toxicity and Liquidity in a High-frequency World},
  journal = {Review of Financial Studies},
  volume  = {25},
  number  = {5},
  pages   = {1457-1493},
  year    = {2012},
  doi     = {10.1093/rfs/hhs053}
}
@article{andersen2014vpin,
  author  = {Andersen, Torben G. and Bondarenko, Oleg},
  title   = {VPIN and the flash crash},
  journal = {Journal of Financial Markets},
  volume  = {17},
  pages   = {1-46},
  year    = {2014},
  doi     = {10.1016/j.finmar.2013.05.005}
}
@misc{araujo2004sync,
  author        = {Araujo, Tanya and Louca, Francisco},
  title         = {Complex behavior of stock markets: processes of synchronization and desynchronization during crises},
  year          = {2004},
  eprint        = {cond-mat/0403333},
  archivePrefix = {arXiv}
}
@article{song2024ews,
  author  = {Song, Shijia and Li, Handong},
  title   = {Early warning signals for stock market crashes: empirical and analytical insights utilizing nonlinear methods},
  journal = {EPJ Data Science},
  volume  = {13},
  pages   = {16},
  year    = {2024},
  doi     = {10.1140/epjds/s13688-024-00457-2}
}
@article{demirer2019herding,
  author  = {Demirer, Riza and Leggio, Karyl B. and Lien, Donald},
  title   = {Herding and flash events: Evidence from the 2010 Flash Crash},
  journal = {Finance Research Letters},
  volume  = {31},
  year    = {2019},
  note    = {Not verified in this review; pages and DOI to be checked}
}
@article{hanley1982auc,
  author  = {Hanley, James A. and McNeil, Barbara J.},
  title   = {The meaning and use of the area under a receiver operating characteristic (ROC) curve},
  journal = {Radiology},
  volume  = {143},
  number  = {1},
  pages   = {29-36},
  year    = {1982},
  doi     = {10.1148/radiology.143.1.7063747}
}
```

DOIs and page ranges marked "memory" in table (a) (Corsetti, Longin and Solnik, Reigneron, Billio,
Harvey, Bailey, Kirilenko, Easley, Andersen and Bondarenko, Hanley and McNeil) should be checked against
Crossref before use.
