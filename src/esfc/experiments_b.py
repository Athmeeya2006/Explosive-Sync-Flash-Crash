"""Empirical experiments (Paper B), implementing docs/PREREGISTRATION_B.md."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import alarms, intraday
from .runner import REPO
from .stats import auc, benjamini_hochberg

EMP = REPO / "research" / "empirical" / "data"


def load_minutes() -> pd.DataFrame:
    """Minute closes as parquet. The original cache is a pickle (not portable, unsafe to load from
    untrusted sources); it is converted once, locally, from the file this repository's own fetcher wrote."""
    pq = EMP / "e7_minute.parquet"
    if not pq.exists():
        pd.read_pickle(EMP / "e7_minute_cache.pkl").to_parquet(pq)
    return pd.read_parquet(pq)


def _boot_auc(pos: np.ndarray, neg: np.ndarray, rng, n_boot: int, paired: bool) -> tuple[float, float, float, float]:
    a = auc(pos, neg)
    b = np.empty(n_boot)
    for i in range(n_boot):
        if paired:
            j = rng.integers(0, pos.size, pos.size)
            b[i] = auc(pos[j], neg[j])
        else:
            b[i] = auc(pos[rng.integers(0, pos.size, pos.size)], neg[rng.integers(0, neg.size, neg.size)])
    return a, float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), float(np.mean(b <= 0.5))


def _pre(sig: np.ndarray, sess: np.ndarray, events: list[int], lead: int) -> np.ndarray:
    out = []
    for t in events:
        lo = max(0, t - lead)
        w = sig[lo:t][sess[lo:t] == sess[t]]
        w = w[np.isfinite(w)]
        out.append(np.median(w) if w.size else np.nan)
    return np.array(out)


def _session_boot(state, sess_te, onsets_te, lead, rng, n_boot):
    """Bootstrap over sessions of recall and false alarms per catch. All counts are additive over sessions."""
    ids, inv = np.unique(sess_te, return_inverse=True)
    r = alarms.score(state, sess_te, onsets_te, lead)
    crash_s = np.zeros(ids.size)
    caught_s = np.zeros(ids.size)
    for o, c in zip(onsets_te, r["caught_mask"]):
        crash_s[inv[o]] += 1
        caught_s[inv[o]] += c
    false_s = np.zeros(ids.size)
    for s, e in alarms.episodes(state, sess_te):
        if not any(s < o <= e + lead and sess_te[o] == sess_te[s] for o in onsets_te):
            false_s[inv[s]] += 1
    rec, fpc = [], []
    for _ in range(n_boot):
        j = rng.integers(0, ids.size, ids.size)
        c, k, f = crash_s[j].sum(), caught_s[j].sum(), false_s[j].sum()
        rec.append(k / c if c else np.nan)
        fpc.append(f / k if k else np.inf)
    q = lambda v: [float(x) for x in np.nanquantile(np.array(v, float), [0.025, 0.975])]  # noqa: E731
    return r, q(rec), q(fpc)


def intraday_study(cfg: dict, run_dir: Path) -> None:
    t0 = time.time()
    rng = np.random.default_rng(cfg["root_seed"])
    px = load_minutes()
    sess = intraday.session_ids(px.index)
    ret = intraday.session_returns(px, sess)
    _, lvl = intraday.session_index(ret, sess)
    ind = intraday.rolling_indicators(ret, sess, cfg["win"], cfg["stride"], cfg["min_stocks"])
    et = px.index.tz_convert("US/Eastern")
    train = np.asarray(et < pd.Timestamp(cfg["split"], tz="US/Eastern"))
    mos = intraday.minute_of_session(sess)

    # corr | vol: OLS fitted on TRAIN only (the earlier E8 fitted it on the whole sample)
    ok = np.isfinite(ind["mean_corr"]) & np.isfinite(ind["volatility"]) & (ind["volatility"] > 0)
    A = np.vstack([np.ones(ok.sum()), np.log(ind["volatility"][ok])]).T
    tr_ok = train[ok]
    beta, *_ = np.linalg.lstsq(A[tr_ok], ind["mean_corr"][ok][tr_ok], rcond=None)
    resid = np.full(ret.shape[0], np.nan)
    resid[ok] = ind["mean_corr"][ok] - A @ beta
    ind["corr_given_vol"] = resid
    names = ["volatility", "mean_corr", "lead_eig_share", "corr_given_vol"]

    def analyse(drop: float, exclude: int, lead: int, tag: str) -> dict:
        onsets, peaks = intraday.find_events(lvl, sess, drop=drop, calm=cfg["calm"], horizon=cfg["horizon"],
                                             exclude=exclude)
        on_tr = [o for o in onsets if train[o]]
        on_te = [o for o in onsets if not train[o]]
        out: dict = {"tag": tag, "drop": drop, "exclude": exclude, "lead": lead, "n_onsets": len(onsets),
                     "n_train": len(on_tr), "n_test": len(on_te), "n_peaks": len(peaks),
                     "onsets_first_hour": int(np.sum(mos[onsets] < 60)) if onsets else 0,
                     "test_dates": [str(et[o])[:16] for o in on_te]}
        # ---- T1: event discrimination, all events (train and test), one value per event
        lv_c = np.log(_pre(ind["volatility"], sess, onsets, lead))
        lv_p = np.log(_pre(ind["volatility"], sess, peaks, lead))
        gc, gp = np.isfinite(lv_c), np.isfinite(lv_p)
        pairs = intraday.optimal_match(lv_c[gc], lv_p[gp], cfg["caliper"])
        out["n_matched_pairs"] = len(pairs)
        out["vol_ratio_crash_over_peak"] = float(np.exp(np.nanmedian(lv_c) - np.nanmedian(lv_p)))
        out["mde_auc_unmatched"] = alarms.min_detectable_auc(int(gc.sum()), int(gp.sum()))
        out["mde_auc_matched"] = alarms.min_detectable_auc(len(pairs), len(pairs)) if pairs else None
        t1 = []
        for nm in names:
            c = _pre(ind[nm], sess, onsets, lead)[gc]
            p = _pre(ind[nm], sess, peaks, lead)[gp]
            m = np.isfinite(c)
            a, lo, hi, pv = _boot_auc(c[m], p[np.isfinite(p)], rng, cfg["n_boot"], paired=False)
            row = {"indicator": nm, "null": "unmatched", "auc": a, "ci_lo": lo, "ci_hi": hi, "p_one_sided": pv,
                   "n_pos": int(m.sum()), "n_neg": int(np.isfinite(p).sum())}
            t1.append(row)
            if len(pairs) >= 5:
                ci, pi = np.array([i for i, _ in pairs]), np.array([j for _, j in pairs])
                cc, pp = c[ci], p[pi]
                mm = np.isfinite(cc) & np.isfinite(pp)
                a, lo, hi, pv = _boot_auc(cc[mm], pp[mm], rng, cfg["n_boot"], paired=True)
                t1.append({"indicator": nm, "null": "vol_matched", "auc": a, "ci_lo": lo, "ci_hi": hi,
                           "p_one_sided": pv, "n_pos": int(mm.sum()), "n_neg": int(mm.sum())})
        rej, adj = benjamini_hochberg([r["p_one_sided"] for r in t1], 0.05)
        for r, rj, ad in zip(t1, rej, adj):
            r["p_bh"], r["bh_reject"] = float(ad), bool(rj)
        out["t1"] = t1
        # ---- T2: alarms out of sample
        te = ~train
        sess_te = sess[te]
        pos_te = np.cumsum(te) - 1
        on_te_local = [int(pos_te[o]) for o in on_te]
        t2 = []
        for nm in names:
            sig = ind[nm]
            tr_vals = sig[train & np.isfinite(sig)]
            for budget in cfg["budgets"]:
                thr = float(np.quantile(tr_vals, 1 - budget))
                state = (np.nan_to_num(sig[te], nan=-np.inf) >= thr)
                r, rec_ci, fpc_ci = _session_boot(state, sess_te, on_te_local, lead, rng, cfg["n_boot_sessions"])
                cp = alarms.clopper_pearson(r["caught"], r["n_crash"]) if r["n_crash"] else (np.nan, np.nan)
                t2.append({"indicator": nm, "budget": budget, "threshold": thr, "caught": r["caught"],
                           "n_crash": r["n_crash"], "recall": r["recall"], "recall_cp": list(cp),
                           "recall_boot": rec_ci, "episodes": r["episodes"], "false_episodes": r["false_episodes"],
                           "false_per_catch": r["false_per_catch"] if np.isfinite(r["false_per_catch"]) else None,
                           "false_per_catch_boot": [x if np.isfinite(x) else None for x in fpc_ci],
                           "precision_episodes": r["precision_episodes"], "realised_rate": r["alarm_rate"]})
        out["t2"] = t2
        for nm in names:      # the invariant the old scoring violated
            c = [r["caught"] for r in t2 if r["indicator"] == nm]
            assert c == sorted(c), (nm, c)
        return out

    primary = analyse(cfg["drop"], cfg["exclude"], cfg["lead"], "primary")
    robust = [analyse(cfg["drop"], cfg["exclude"], ld, f"lead_{ld}") for ld in cfg["robust_leads"]]
    robust += [analyse(cfg["drop"], 0, cfg["lead"], "no_exclusion")]
    robust += [analyse(d, cfg["exclude"], cfg["lead"], f"drop_{d}") for d in cfg["robust_drops"]]
    old = pd.read_csv(EMP / "e7_events.csv")
    old_t = pd.to_datetime(old.date, utc=True).dt.tz_convert("US/Eastern")
    old_first30 = int(((old_t.dt.hour * 60 + old_t.dt.minute) < 10 * 60).sum())
    summary = {"n_tickers": int(px.shape[1]), "n_bars": int(px.shape[0]), "n_sessions": int(sess.max() + 1),
               "span": [str(px.index[0])[:10], str(px.index[-1])[:10]],
               "regression_slope_train": float(beta[1]), "wall_seconds": round(time.time() - t0, 1),
               "old_e7_onsets": int(len(old)), "old_e7_onsets_before_10am": old_first30,
               "primary": primary, "robustness": robust}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
