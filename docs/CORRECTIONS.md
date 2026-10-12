# Corrections and withdrawals

Every claim this project has made and later withdrawn or downgraded, newest first. Nothing is deleted
from this list.

| date | claim | where it was made | what replaced it | evidence |
|---|---|---|---|---|
| 2026-10-13 | The finite-connectivity coefficient `(p+2)/2` is exact | README, index S14, reference step 21 | Exact for the Rician closure only. Direct simulation is below the closure in every run | `docs/CLAIMS_AUDIT.md` |
| 2026-10-13 | The closure predicts the measured fold to 0.05% at k = 48 and 0.6 to 3.7% at k = 12 | index S14 | Withdrawn: the source files were never committed and a re-run disagrees | `docs/CLAIMS_AUDIT.md` |
| 2026-10-13 | Predicted 12.5% against measured 11.6% at k = 12 | README, S14 docstring | Withdrawn: one unsaved grid-read run | `docs/CLAIMS_AUDIT.md` |
| 2026-10-13 | The jump is real and grows with N | README, index S11 | Not evidence of a discontinuity: the statistic is the largest grid step of one curve | `docs/CLAIMS_AUDIT.md` |
| 2026-10-13 | False alarms per crash caught (E8 counts) | README, index E8 | Withdrawn pending the corrected, pre-registered study: look-ahead in the regression and a scoring rule that missed persistent alarms | `docs/PREREGISTRATION_B.md` |
| 2026-10-13 | "Absorption ratio" in E1 | E1 | Renamed `lead_eig_share`; the Kritzman ratio is a different quantity | `src/esfc/ews.py` |
| 2026-10-13 | Largest Lyapunov exponent as a diagnostic of the onset of chaos | old README | Withdrawn: unseeded tangent vector, neutral direction not projected out | `legacy/README.md` |
| 2026-10-13 | "525 simulations", "95% coverage", cross-layer reproducibility, 0.15 s C++ run | old README | Withdrawn or restated as legacy-only | `legacy/README.md` |
| earlier | S13: the `lam_b(p)` family and `R*(p)` are new | index S13 | Zou and Wang, PRE 102, 012219 (2020) with `alpha = p + 1` | index S13, S14 |
| earlier | S13: critical clustering C* of about 0.35 | index S13 | Confounded by a fixed coupling grid that no longer contained the transition | index S13 |
| earlier | E5: 24 times excess clustering; collapse of the window is structural | index E5 | 3.8 times against a size-matched baseline; 72% of the collapse is finite size | index E5 |
| earlier | A3: estimate lambda from market data | index A3 | Withdrawn: lambda is not identifiable from R | index A3 |
| earlier | 30x C++ speedup (0.15 s against 4.50 s) | old README | Not reproducible; measured 2.15 s against 2.98 s | old README |
| earlier | Finite-size scaling converges within 2% of `K_c` | old README | Withdrawn: onset near K of 2, not 1.5957, with `K/d_i` coupling | old README |
| earlier | Degree-normalised coupling shows explosive synchronization | old README | It does not: hysteresis gap 0.06 against 0.79 unnormalised | `research/s1_reproduce_gg2011.py` |
| earlier | A frequency-degree rank correlation is sufficient for explosive synchronization | old README | The tail of g(omega) must match the tail of P(k) | `research/s8c_correlation_corrected.py` |
