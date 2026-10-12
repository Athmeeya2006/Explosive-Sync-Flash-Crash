# Pre-registration: synchronization-based crash warnings (empirical study, version 1)

Written on 2026-10-13 **before** any corrected analysis was run. It is **not** under version control:
at the author's request nothing from this work has been committed or tagged, so there is no
independent timestamp. Anyone relying on the ordering has only this statement and the file times in
`results/paperB_intraday/`. To make the ordering verifiable, commit and tag this file (or deposit it,
for example on OSF) before any further analysis is run.

## Disclosure

This is not a blind pre-registration. Earlier versions of these analyses (E1, E7, E8 at commit
`fc98f02`) were run and seen. An audit found defects in them (overnight contamination, look-ahead in a
regression, an alarm-scoring rule that could miss persistent alarms, zero-filled returns, a misnamed
indicator). What is fixed here is the corrected design, written before running it. The earlier results
suggest the direction "no usable warning"; the design below is chosen so that either outcome is
interpretable.

## Data

| set | source | span | cross-section | note |
|---|---|---|---|---|
| Intraday | HF Data Library, 1-minute closes | 2002 to 2026 | 40 fixed large caps | Survivorship: chosen from today's large caps. Bias direction: favours calmer, surviving names, which should lower measured co-movement in old crises |
| Daily | Yahoo, via `fetch_markets.py` | 2000 to 2026 | 23 markets, current constituents | Same survivorship caveat. Robustness run restricted to names listed over each full window |

Train: timestamps before 2016-01-01 (US/Eastern). Test: from 2016-01-01. Never shuffled.

## Intraday definitions

1. **Session.** One calendar day in US/Eastern. Returns are 1-minute log returns **within a session**.
   The first return of every session (which spans the overnight gap) is discarded. Returns with
   absolute value above 0.25 are treated as data errors and discarded.
2. **Index.** Equal-weight mean of available stock returns each minute, cumulated within the session
   and reset at each open.
3. **Crash onset.** Minute `t`, at least 30 minutes after the session's first bar, such that the index
   at `t` equals its maximum over the surrounding 60 minutes of the same session and falls by at least
   3% below that level within the next 60 minutes of the same session. At most one onset per session
   (the first).
4. **Ordinary peak (null event).** Same local-maximum rule, fall within the next 60 minutes smaller
   than 1%, in a session with no crash onset.
5. **Windows.** Every indicator at minute `t` uses the trailing 30 one-minute returns of the **same
   session**. No window straddles an open, so no indicator exists in the first 30 minutes.

## Indicators

| name | definition |
|---|---|
| `volatility` | standard deviation of index returns in the window |
| `mean_corr` | mean pairwise correlation over stocks with complete data in the window (at least 12) |
| `lead_eig_share` | largest eigenvalue of that correlation matrix divided by the number of stocks |
| `corr_given_vol` | residual of an ordinary least squares fit of `mean_corr` on log `volatility`, **fitted on the training period only** |

Daily (E1 corrected) adds: Kritzman absorption ratio (variance share of the top N/5 eigenvectors of
the covariance matrix), cross-sectional dispersion, breadth flicker, index variance, index lag-1
autocorrelation. The quantity previously called "absorption" is renamed `lead_eig_share`.
Correlations use pairwise-complete observations; returns are never zero-filled.

## Tests

**T1. Event discrimination.** One value per event: the median indicator over the 60 minutes before
the event. AUC of crash onsets against ordinary peaks, (a) unmatched, (b) volatility-matched by
optimal one-to-one assignment (Hungarian algorithm) on absolute log-volatility distance with a caliper
of 0.25, pairs outside the caliper dropped. Confidence intervals: bootstrap over events for (a), over
matched pairs for (b), 4000 resamples, 95%.

**T2. Alarm performance out of sample.** For each indicator and each nominal budget in
{0.5%, 1%, 2%, 5%, 10%}: threshold = the corresponding upper quantile of the indicator on training
minutes. On test minutes the alarm state is ON when the indicator is at or above the threshold.
- A crash is **caught** if the state is ON at any minute in `[onset - LEAD, onset)`, LEAD = 60.
- An ON **episode** is a maximal run of consecutive ON minutes within a session. An episode is a
  **false alarm** if no crash onset lies in `(start, end + LEAD]`.
- Reported: recall, number of false-alarm episodes, false alarms per crash caught, precision over
  episodes, and the **realised** alarm rate (fraction of test minutes ON), which need not equal the
  nominal budget. Confidence intervals by bootstrap over sessions, 2000 resamples.
- By construction the caught set can only grow as the threshold is lowered. A violation is a bug.

**Primary hypothesis H1.** `corr_given_vol` discriminates crash onsets from volatility-matched
ordinary peaks: AUC in T1(b) above 0.5, one-sided, alpha = 0.05.
**Secondary H2.** At the 1% and 5% budgets, `mean_corr` has higher recall than `volatility` at an equal
or lower realised alarm rate.
Multiplicity: Benjamini-Hochberg at q = 0.05 across all indicator by test combinations reported, with
the number of tests stated.

## Power

The number of test-period crashes is small (on the order of ten). The report must state the exact
number, give Clopper-Pearson intervals for recall, and state the minimum detectable AUC (the smallest
AUC distinguishable from 0.5 at 80% power for the realised event counts, from the Hanley-McNeil
variance). A null result will be described as "not detected at this power", not as "absent".

## Robustness (reported regardless of outcome)

LEAD in {30, 60, 120}; onset exclusion zone of 0 minutes instead of 30; fall threshold 2% and 4%.

## Not claimed

No causal statement. Kirilenko et al. (2017) found high-frequency traders' behaviour did not change
during the 2010 crash. Synchrony in the model is direction-agnostic; a "crash" arises only because a
sell shock or bias selects the direction.
