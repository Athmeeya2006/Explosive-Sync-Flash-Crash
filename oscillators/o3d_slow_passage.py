"""O3d: the delay of explosive desynchronisation, predicted with no free parameter.

Z2 and O3c found that a population swept down through its herding fold collapses late, by an amount
proportional to (sweep speed)^(2/3). For the Lorentzian Kuramoto model the Ott-Antonsen equation
    R' = -gamma' R + (lam(t) / 2) R^(p+1) (1 - R^2),   gamma' = gamma + sigma^2 / 2,
is exact for N -> infinity, so the delay itself can be predicted: integrate it along exactly the ramps of
Z2 and O3c and apply exactly their detection rules. Near the fold (p = 1) it reduces to the saddle-node
normal form x' = F_lam (lam - lam_b) + F_RR x^2 / 2 with F_lam = R_b^2 (1 - R_b^2) / 2 = 1/9 and
F_RR = -lam_b; for a ramp of speed v the trajectory escapes (x -> -infinity) at
    lam_b - lam_escape = Omega_0 v^(2/3) (2 / (F_lam lam_b))^(1/3),   Omega_0 = 2.33811 (first zero of Ai),
the classic slow-passage law (see Kuehn, Physica D 240, 1020 (2011)); the detection rules of the
simulations fire before that escape, which the full integration takes into account.
Compared with: Z2 ramp runs (data/z2_ramp.csv, detection: last time R >= R_b - 0.05) and the O3c
Kuramoto runs (data/o3c_rate.csv, detection: R falls below 0.5, both p = 1 and p = 2).
Result columns: oa_delay (prediction), measured_delay (relative to the N -> infinity fold), offset (their
difference; constant across speeds at fixed N if the speed dependence is predicted exactly), and for O3c
delay_from_finite_N_fold (relative to the fold O3c extrapolated for its own N = 500 population).
Output: data/o3d_slow_passage.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

DATA = Path(__file__).resolve().parent / "data"
GAMMA, SIG = 0.5, 0.1
GP = GAMMA + SIG ** 2 / 2
OMEGA0 = 2.338107410459767


def lam_b(p: float) -> float:
    return GP * (p + 2) * ((p + 2) / p) ** (p / 2)


def branch(lam: float, p: float) -> float:
    Rb = np.sqrt(p / (p + 2))
    return brentq(lambda R: (lam / 2) * R ** p * (1 - R * R) - GP, Rb, 1 - 1e-12)


def oa_ramp(lam0: float, lam1: float, T: float, p: float, rule: str) -> float:
    """Integrate the OA equation along lam(t) = lam0 + (lam1 - lam0) t / T from the stable branch at lam0;
    return lam at the moment the detection rule fires."""
    def rhs(t, y):
        lam = lam0 + (lam1 - lam0) * min(t / T, 1.0)
        R = y[0]
        return [-GP * R + (lam / 2) * R ** (p + 1) * (1 - R * R)]
    Rb = np.sqrt(p / (p + 2))
    thr = Rb - 0.05 if rule == "leave" else 0.5
    ev = lambda t, y: y[0] - thr      # noqa: E731
    ev.terminal = True
    ev.direction = -1
    sol = solve_ivp(rhs, (0, T), [branch(lam0, p)], events=ev, rtol=1e-10, atol=1e-12, max_step=T / 2000)
    t_hit = sol.t_events[0][0]
    return lam0 + (lam1 - lam0) * t_hit / T


def normal_form_delay(v: float, p: float = 1.0) -> float:
    lb = lam_b(p)
    Rb = np.sqrt(p / (p + 2))
    F_lam = Rb ** p * (1 - Rb ** 2) / 2
    F_RR = (lb / 2) * ((p + 1) * p * Rb ** (p - 1) - (p + 3) * (p + 2) * Rb ** (p + 1))
    return OMEGA0 * v ** (2 / 3) * (2 / (F_lam * abs(F_RR))) ** (1 / 3)


def main() -> None:
    rows = []
    z2 = pd.read_csv(DATA / "z2_ramp.csv").dropna(subset=["eps_esc"])
    lb1 = lam_b(1.0)
    for (N, T), g in z2.groupby(["N", "T_ramp"]):
        lam0, lam1 = 1.5 * lb1, 0.8 * lb1
        pred = oa_ramp(lam0, lam1, T, 1.0, "leave")
        v = (lam0 - lam1) / T
        rows.append({"source": "Z2", "p": 1, "N": N, "T": T, "speed": v, "measured_delay": float((-g.eps_esc * lb1).mean()),
                     "measured_sd": float((g.eps_esc * lb1).std()), "runs": len(g), "oa_delay": lb1 - pred,
                     "normal_form_escape_delay": normal_form_delay(v)})
    o3c = pd.read_csv(DATA / "o3c_rate.csv")
    o3c = o3c[o3c.family == "kuramoto"].dropna(subset=["fold"])
    for (p, T), g in o3c.groupby(["p", "T"]):
        pr = g.pred_fold.iloc[0]
        lam0, lam1 = 1.4 * pr, 0.5 * pr
        lb = lam_b(float(p))
        pred = oa_ramp(lam0, lam1, T, float(p), "below05")
        v = (lam0 - lam1) / T
        rows.append({"source": "O3c", "p": p, "N": 500, "T": T, "speed": v, "measured_delay": float(lb - g.fold.mean()),
                     "measured_sd": float(g.fold.std()), "runs": len(g), "oa_delay": lb - pred,
                     "normal_form_escape_delay": normal_form_delay(v, float(p))})
    d = pd.DataFrame(rows)
    d["measured_over_oa"] = d.measured_delay / d.oa_delay
    # The OA prediction is for N -> infinity with noise entering only as gamma' = gamma + sigma^2 / 2. A
    # finite population has its own fold, offset by a small amount that does not depend on the speed, so
    # the speed dependence is tested by offset = oa_delay - measured_delay being constant at fixed N.
    d["offset"] = d.oa_delay - d.measured_delay
    ex = pd.read_csv(DATA / "o3c_extrapolated.csv").set_index(["family", "p"]).fold_extrapolated
    o3 = d.source == "O3c"
    d.loc[o3, "delay_from_finite_N_fold"] = [ex.loc[("kuramoto", int(p))] - (lam_b(float(p)) - md)
                                             for p, md in zip(d.loc[o3, "p"], d.loc[o3, "measured_delay"])]
    d.to_csv(DATA / "o3d_slow_passage.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
