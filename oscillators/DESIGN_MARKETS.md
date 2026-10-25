# Design: oscillator-motivated crash warnings on real markets (study M1)

Written on 2026-10-21, before `m1_market_ews.py` was run on any market. Not committed or timestamped
(the author has asked that nothing be committed), so the ordering rests on this statement and the file
times in `oscillators/data/`.

## Why these indicators

The simulation benchmark (Z1 to Z3) asks which indicator warns before which oscillator transition.
Three indicators come out of the oscillator picture and have not, as far as a literature search found,
been tested across many stock markets with null events:

| name | definition (computed in a trailing 60-day window, every 5 trading days) | oscillator it comes from |
|---|---|---|
| `ar2_rate` | AR(2) fit to the index's daily log returns; `ln max|root|` (the decay rate of the slowest mode per day; rises toward 0 as stability is lost) | damped harmonic oscillator, Stuart-Landau below Hopf |
| `var1_eig` | the constituents' returns are standardised and projected on the top 3 principal components of the window; a VAR(1) is fitted to the 3 component series; `max|eig|` of its matrix | multivariate dynamic eigenvalue of a coupled system |
| `hilbert_R` | each constituent's log price is band-passed inside a trailing 120-day window (difference of Gaussian smooths with widths 3 and 20 days); Hilbert phases are taken in that window only; Kuramoto order parameter `R(t) = |mean_j exp(i phi_j(t))|`, averaged over the last 20 days of the window | Kuramoto phase synchrony |

No indicator uses data after its date: the Hilbert transform is applied to the trailing window only, so
the end effects of the transform fall on the last days, never on future ones.

## Data, events and nulls

Unchanged from E1c (`research/empirical/e1c_ews_corrected.py`, `docs/PREREGISTRATION_B.md`):
23 markets (daily closes, current constituents, 2000 to 2026), crash onset = 20-day high followed by a
fall of at least 10% (30% for crypto) within 30 trading days, at most one per 120 days; three null sets
(all non-crash dates at least 250 days from an event, volatility-matched dates, 20-day highs not
followed by a crash).

## Scores and tests

For each indicator, two scores at each date: `trend` (Kendall tau over the prior 250 days) and
`level` (z-score of today's value against days t-500 to t-60). AUC of events against each null set,
95% CI and one-sided p-value from the cluster bootstrap over calendar quarters on both sides
(2000 resamples). 3 indicators x 2 scores x 3 nulls = 18 tests, Benjamini-Hochberg at q = 0.05.

**Primary hypothesis H1.** `var1_eig` trend discriminates crash onsets from peaks not followed by a crash
(AUC > 0.5 against the `peak` null).
**Secondary.** The other 17 combinations, reported in full with the BH correction.

The E1c indicators (variance, AC1, mean correlation, absorption ratio) are re-scored on the same events
for comparison only; they are not part of the 18 tests.

## What would count

A positive result needs a BH-surviving AUC against the `peak` null (the null that already went up and
did not crash). An AUC against the `all` null alone mostly says that crashes start from high-volatility
regimes, which variance already shows.

## Deviation recorded while the run was in progress

E1c counted 23 markets because it also scored `world_indices.pkl`, whose "constituents" are national
indices. In `m1_market_ews.py` that panel is excluded (`marketdata.market_names`), so M1 has 22 stock
markets; the world panel is analysed separately, with a time-zone-robust correlation, in M3. This was
decided when the code was written, before any M1 result was seen, and is stated here because the text
above says 23.
