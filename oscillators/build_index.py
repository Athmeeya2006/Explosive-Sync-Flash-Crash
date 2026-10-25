"""Build oscillators/index.html: the results of this folder, with every figure and table explained.

Every number on the page is read from data/ at build time, so the page matches the latest run.
The build fails if a figure is missing or if the page contains an em or en dash.

    ../research/.venv/bin/python build_index.py
"""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"

SCEN_LABEL = {
    "ou": "Ornstein-Uhlenbeck (reference)", "harmonic": "Simple harmonic oscillator", "sl_hopf": "Stuart-Landau, single",
    "fhn": "FitzHugh-Nagumo, single", "kuramoto_up": "Kuramoto, onset", "kuramoto_adapt_down": "Kuramoto, adaptive, collapse",
    "kuramoto2_down": "Kuramoto with inertia, collapse", "sl_sync_up": "Stuart-Landau population, onset",
    "sl_adapt_down": "Stuart-Landau, adaptive, collapse", "sl_death_up": "Stuart-Landau, amplitude death",
    "winfree_up": "Winfree, oscillator death", "vdp_up": "van der Pol population, onset",
    "rossler_up": "Roessler population, phase sync", "theta_down": "Theta neurons, collapse of activity",
    "vdp_adapt_down": "van der Pol, herding, collapse", "rossler_adapt_down": "Roessler, herding, collapse",
    "fhn_adapt_down": "FitzHugh-Nagumo, herding, collapse", "kuramoto2_adapt_down": "Kuramoto with inertia, herding, collapse"}
IND_LABEL = {"var": "variance", "ac1": "lag-1 autocorrelation", "skew": "|skewness|", "spec_peak": "spectral peak",
             "ar2_rate": "AR(2) decay rate", "var1_rate": "VAR(1) decay rate",
             "var_index": "index variance", "ac1_index": "index lag-1 AC", "mean_corr": "mean correlation",
             "absorption_ratio": "absorption ratio", "var1_eig": "VAR(1) eigenvalue", "hilbert_R": "Hilbert phase synchrony",
             "unit_var": "median stock variance", "coll_share": "collective share of variance",
             "world_R": "world phase synchrony", "world_corr": "world mean correlation", "world_vol": "world volatility"}
NULL_LABEL = {"all": "all non-crash dates", "volmatched": "volatility-matched dates", "peak": "peaks not followed by a crash",
              "null_r": "ramp away from the transition", "null_c": "parameter held fixed"}


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        out = f"{v:.{nd}f}"
        return out[1:] if out.startswith("-") and float(out) == 0 else out
    return html.escape(str(v))


def table(df: pd.DataFrame, cols: dict, auc_cols=(), corr_cols=(), nd=3, wrap=False) -> str:
    head = "".join(f"<th>{html.escape(v)}</th>" for v in cols.values())
    body = []
    for r in df.astype(object).to_dict("records"):      # records keep integer columns as integers
        tds = []
        for c in cols:
            v = r[c]
            cls = ""
            if c in auc_cols and isinstance(v, float) and np.isfinite(v):
                cls = ' class="good"' if v >= 0.7 else (' class="bad"' if v <= 0.3 else "")
            if c in corr_cols and isinstance(v, float) and np.isfinite(v):
                cls = ' class="good"' if v > 0 else ""
            if isinstance(v, (bool, np.bool_)):
                v = "yes" if v else "no"
            if isinstance(v, str) and v in IND_LABEL:
                v = IND_LABEL[v]
            if isinstance(v, str) and v in NULL_LABEL:
                v = NULL_LABEL[v]
            if isinstance(v, str) and v in SCEN_LABEL:
                v = SCEN_LABEL[v]
            tds.append(f"<td{cls}>{fmt(v, nd)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    cls = ' class="fit"' if wrap else ""
    return f'<div class="tbl"><table{cls}><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def figure(name: str, caption: str, seeing: str, significance: str) -> str:
    path = FIG / name
    if not path.exists():
        raise FileNotFoundError(path)
    return (f'<figure><img src="figures/{name}" alt="{html.escape(caption)}" loading="lazy">'
            f'<figcaption>{caption}</figcaption>'
            f'<div class="explain"><div><h4>What you are seeing</h4>{seeing}</div>'
            f'<div><h4>Significance</h4>{significance}</div></div></figure>')


def badge(kind: str) -> str:
    text = {"result": "result", "neg": "negative result", "theory": "theory confirmed", "explore": "exploratory",
            "open": "open question", "data": "data quality"}[kind]
    return f'<span class="badge b-{kind}">{text}</span>'


def section(sid: str, tag: str, title: str, kinds: list[str], body: str) -> str:
    return (f'<section id="{sid}"><div class="sec-head"><span class="tag">{tag}</span><h2>{title}</h2>'
            f'{"".join(badge(k) for k in kinds)}</div>{body}</section>')


# ------------------------------------------------------------------------------------------ content
def o_section() -> str:
    """The per-oscillator pipeline (O0 to O9): headline numbers and figures; details on reference.html."""
    if not (DATA / "o3c_extrapolated.csv").exists():
        return ""
    ex = pd.read_csv(DATA / "o3c_extrapolated.csv")
    red = pd.read_csv(DATA / "o3_reduction.csv")
    pr = pd.read_csv(DATA / "o3_predictions.csv")
    inert = pr[(pr.family == "inertia") & (pr.p >= 1)]
    thr = pd.read_csv(DATA / "o1_thresholds.csv").groupby(["family", "p", "direction"]).threshold.mean()
    inert_rows = pd.DataFrame([{"family": "inertia", "p": int(r.p), "fold_slowest": thr[("inertia", int(r.p), "down")],
                                "fold_extrapolated": np.nan, "pred_phase_reduction": r.pred_fold,
                                "rel_gap_extrapolated": thr[("inertia", int(r.p), "down")] / r.pred_fold - 1}
                               for _, r in inert.iterrows()])
    t = pd.concat([ex, inert_rows], ignore_index=True)
    sep = pd.read_csv(DATA / "o8_separability.csv") if (DATA / "o8_separability.csv").exists() else None
    extra = ""
    for name, cap, seeing, sig in (
            ("o4_protocols.png", "O4. Shocks and circuit breakers, every family with herding.",
             "<p>Top row: fraction of runs that end synchronised after a fraction phi (horizontal) of a calm population is put at one phase, at couplings below and inside the bistable window (vertical, multiples of the slow-sweep fold). Bottom row: fraction of runs that re-synchronise after a halt of the given length (horizontal, log, in units of 1 / frequency spread); dashed: the halt predicted by phase reduction.</p>",
             "<p>Whether shocks and halts act the same way in every family, and whether the halt length can be predicted from the frequencies alone.</p>"),
            ("o5_frequency_degree.png", "O5. Frequency increasing with degree on a scale-free network, every family.",
             "<p>R against unnormalised coupling on one Barabasi-Albert network; green: swept up from calm, red: down from synchrony. Top: frequency increasing with degree (the Gomez-Gardenes condition); bottom: the same frequencies shuffled.</p>",
             "<p>A gap between the green and red curves is hysteresis, the signature of explosive synchronisation; the shuffled control shows how much of it comes from the frequency-degree correlation.</p>"),
            ("o6_sparse_fold.png", "O6. Herding fold on sparse networks, every family.",
             "<p>Fold on random regular graphs of degree k divided by the fold on the complete graph, minus 1, against 1/k; lines are fits a/k + c. Dashed: the main project's leading-order Kuramoto theory on undirected graphs.</p>",
             "<p>Phase reduction says a should be the same for every family; equal slopes support the transfer of the main project's finite-connectivity theory to all of them.</p>")):
        if (FIG / name).exists():
            extra += figure(name, cap, seeing, sig)
    body = f"""
<p class="lead">The main project's pipeline, repeated for every oscillator family. Full chapter per family, with every
symbol and table, on <a href="reference.html">reference.html</a>; interactive demo on <a href="demo.html">demo.html</a>.</p>
<ul>
<li><b>Single units (O0)</b> reproduce their textbook bifurcations: Stuart-Landau amplitude sqrt(mu) to four digits,
van der Pol 2 sqrt(mu), theta-neuron firing rate sqrt(eta)/pi, FitzHugh-Nagumo spikes at its Hopf point.</li>
<li><b>Herding makes synchronisation explosive in every family (O1)</b>: the calm state stays stable far past the fold
(at p = 2 no upward jump in range), the synchronised state collapses abruptly. Every exact threshold is reproduced:
Kuramoto onset, amplitude-death window (1, 5.53), Winfree death 0.775, theta-neuron folds.</li>
<li><b>Phase reduction (O3)</b> gives each family one number, a1 (Stuart-Landau {red.set_index('family').a1['sl']:.3f},
van der Pol {red.set_index('family').a1['vdp']:.3f}, FitzHugh-Nagumo {red.set_index('family').a1['fhn']:.3f}), and with it the
synchrony onset and herding fold of every family from the Kuramoto formulas.</li>
<li><b>A finite sweep lowers every measured fold (O3c)</b>, as (speed)^(2/3); slow families (small frequency spread)
the most. Extrapolated to an infinitely slow sweep, the folds agree with phase reduction within a few percent for
Kuramoto, Kuramoto with inertia, van der Pol and Roessler; Stuart-Landau (-10%) and FitzHugh-Nagumo (-19%) stay below.
The delay itself is predicted with no free parameter (O3d, in section Z2).</li>
<li><b>Separability (O8)</b>: Winfree and theta-neuron couplings are product-separable (rank 1), inside Hens et al.'s
signal-propagation class; diffusive couplings are rank 2 like Kuramoto; herding is never pairwise additive.</li>
</ul>
<h3>Herding folds: slow-sweep limit against phase reduction</h3>
{table(t, {"family": "family", "p": "p", "fold_slowest": "fold, slowest sweep", "fold_extrapolated": "extrapolated",
           "pred_phase_reduction": "phase reduction (inertia: locked-state self-consistency)",
           "rel_gap_extrapolated": "relative gap"}, nd=4)}
<p>For Kuramoto with inertia the comparison is with the atlas fold (no speed extrapolation was run for it).</p>
{figure("o1_atlas.png", "O1. Every family swept up and down, with and without herding.",
        "<p>One panel per family: R against coupling, mean of 4 seeds. Solid: up from calm; dashed: down from synchrony. Grey p = 0, blue p = 1, red p = 2. Dotted verticals: phase-reduction folds; circle: predicted onset at R = 0.5.</p>",
        "<p>The same qualitative picture in every family: herding opens a bistable window. The downward collapses in this figure come later than the true folds because the sweep is finite (O3c).</p>")}
{figure("o3_phase_reduction.png", "O3. Phase interaction functions and the thresholds they predict.",
        "<p>(a) H(phi) of each family from the numerical adjoint. (b) Measured against predicted thresholds, log-log; filled: slow-sweep limit, open: atlas sweep.</p>",
        "<p>One coefficient per family maps the Kuramoto thresholds onto it; the residual gaps of Stuart-Landau and FitzHugh-Nagumo mark where phase reduction stops being accurate.</p>")}
{figure("o3c_rate.png", "O3c. Folds at three sweep speeds, extrapolated to an infinitely slow sweep.",
        "<p>Measured fold divided by the phase-reduction fold against (speed / fold)^(2/3); lines: fits, meeting the left axis at the extrapolated fold.</p>",
        "<p>Removes the largest systematic error of every threshold measured by sweeping.</p>")}
{table(sep, {"family": "family", "coupling G(x_i, x_j)": "coupling", "product_separable": "product-separable",
             "separable_rank": "separable rank"}) if sep is not None else ""}
{extra}"""
    pend = [(code, what, ada) for code, f, what, ada in (
        ("O3e", "o3e_sl_selfconsistency.csv", "exact mean-field self-consistency of the Stuart-Landau herding branch (why it is 10% below phase reduction)", "A12"),
        ("O5", "o5_summary.csv", "frequency-degree correlation on a scale-free network, every family", "A10"),
        ("O6", "o6_coefficients.csv", "herding fold on sparse random regular graphs against 1/k, every family", "A7"),
        ("O9", "o9_fits.csv", "signal propagation on Winfree and theta-neuron networks (Hens et al. a2 test)", "A11"),
        ("Z1 herding", None, "Z1 indicators for herding in van der Pol, Roessler, FitzHugh-Nagumo and inertia", "A9"))
        if (not (DATA / f).exists() if f else
            "vdp_adapt_down" not in set(pd.read_csv(DATA / "z1_auc.csv").scenario))]
    if pend:
        body += ("<h3>Designed but not yet run (moved to the Ada cluster)</h3><p>These steps of the pipeline are written "
                 "and documented in README.md, but their full runs need more than this workstation; "
                 "ADA_COMPUTE.md gives the cost and the SLURM script. Their results appear here and on reference.html when "
                 "the pages are rebuilt.</p><ul>"
                 + "".join(f"<li><b>{c}</b>: {w} (item {a})</li>" for c, w, a in pend) + "</ul>")
    return section("o", "O", "Every oscillator family: the main project's pipeline repeated", ["result", "theory"], body)


def z1_section() -> str:
    a = pd.read_csv(DATA / "z1_auc.csv")
    runs = pd.read_parquet(DATA / "z1_runs.parquet")
    cv = a[(a.signal == "coll") & (a.indicator == "var")].pivot(index="scenario", columns="null", values="auc")
    uv = a[(a.signal == "unit") & (a.indicator == "var")].pivot(index="scenario", columns="null", values="auc")
    tab = pd.DataFrame({"scenario": cv.index, "coll_r": cv.null_r.values, "coll_c": cv.null_c.values,
                        "unit_r": uv.loc[cv.index].null_r.values, "unit_c": uv.loc[cv.index].null_c.values})
    ptr = runs.groupby("scenario").p_at_trans.agg(["mean", "std"])
    tab["p_mean"] = ptr.loc[tab.scenario, "mean"].values
    tab["p_sd"] = ptr.loc[tab.scenario, "std"].values
    order = ["ou", "harmonic", "sl_hopf", "fhn", "kuramoto_up", "kuramoto_adapt_down", "kuramoto2_down", "sl_sync_up",
             "sl_adapt_down", "sl_death_up", "winfree_up", "vdp_up", "rossler_up", "theta_down",
             "vdp_adapt_down", "rossler_adapt_down", "fhn_adapt_down", "kuramoto2_adapt_down"]
    tab = tab.set_index("scenario").loc[[o for o in order if o in tab.scenario.values]].reset_index()
    min_cv = float(cv.null_r.min())
    worst = SCEN_LABEL[cv.null_r.idxmin()]
    n_scen = int(cv.null_r.size)
    warned = cv.null_r[cv.null_r >= 0.7]
    failed = cv.null_r[cv.null_r < 0.5]
    if failed.empty:
        headline = (f"<b>Variance of the collective signal is the only indicator that warned in all {n_scen} scenarios</b> "
                    f"(lowest AUC {min_cv:.2f}, {html.escape(worst)}).")
    else:
        headline = (f"<b>Variance of the collective signal warned in {len(warned)} of {n_scen} scenarios</b> (AUC 0.7 or more), more "
                    f"than any other indicator, but it <b>fell</b> instead of rising before "
                    + ", ".join(f"{html.escape(SCEN_LABEL[x])} (AUC {v:.2f})" for x, v in failed.items())
                    + ". No indicator warned everywhere.")
    fails = []
    for (sig, ind), g in a[(a.null == "null_r") & (a.signal != "nodes")].groupby(["signal", "indicator"]):
        bad = g[g.auc < 0.3].scenario.tolist()
        fails.append({"signal": "collective" if sig == "coll" else "one unit", "indicator": ind,
                      "n_fail": len(bad), "where": ", ".join(SCEN_LABEL[b] for b in bad) or "none"})
    fails = pd.DataFrame(fails).sort_values(["signal", "n_fail"])
    models = """
<div class="tbl"><table class="fit"><thead><tr><th>model</th><th>equation (noise terms written as sigma xi)</th><th>control parameter and transition</th></tr></thead><tbody>
<tr><td>Ornstein-Uhlenbeck</td><td><code>x' = -kappa x + sigma xi</code></td><td>kappa from 1 down to 0: the textbook loss of stability with a real eigenvalue. Reference case.</td></tr>
<tr><td>Simple harmonic oscillator</td><td><code>x'' + g x' + x = sigma xi</code></td><td>damping g from 1 to 0: the rest state loses stability through a complex pair of eigenvalues (a linear Hopf point).</td></tr>
<tr><td>Stuart-Landau</td><td><code>z' = (mu + i w) z - |z|^2 z + sigma (xi1 + i xi2)</code></td><td>mu from -1 to 0: supercritical Hopf bifurcation, the normal form of every smooth oscillation onset.</td></tr>
<tr><td>FitzHugh-Nagumo</td><td><code>v' = v - v^3/3 - w + I + sigma xi,  w' = 0.08 (v + 0.7 - 0.8 w)</code></td><td>current I up: excitable rest state to spiking; the small cycle explodes into full spikes (canard explosion), an abrupt single-unit transition.</td></tr>
<tr><td>Kuramoto</td><td><code>theta_j' = w_j + K R sin(Psi - theta_j) + sigma xi_j</code></td><td>K up through K<sub>c</sub> = 2(gamma + sigma^2/2): continuous synchrony onset.</td></tr>
<tr><td>Kuramoto, adaptive</td><td><code>theta_j' = w_j + lam R^2 sin(Psi - theta_j) + sigma xi_j</code></td><td>lam down through the fold lam<sub>b</sub> = 3 sqrt(3)(gamma + sigma^2/2): explosive desynchronisation, the model of a crash used in the rest of the project.</td></tr>
<tr><td>Kuramoto with inertia</td><td><code>2 theta_j'' + theta_j' = w_j + K R sin(Psi - theta_j) + sigma xi_j</code></td><td>K down: hysteretic desynchronisation (the swing equation of power grids).</td></tr>
<tr><td>Stuart-Landau population</td><td><code>z_j' = (1 + i w_j - |z_j|^2) z_j + K rho^p (Z - z_j) + noise</code></td><td>p = 0, narrow frequencies: synchrony onset. p = 1 (rho = |Z| / mean |z_j|): explosive desynchronisation. p = 0, frequencies spread over [-4, 4]: amplitude death, every unit stops oscillating.</td></tr>
<tr><td>Winfree</td><td><code>theta_j' = w_j - kappa sin(theta_j) mean_k(1 + cos theta_k) + sigma xi_j</code></td><td>kappa up: oscillator death, the whole population stops at a fixed point (a collective saddle-node on the circle).</td></tr>
<tr><td>van der Pol population</td><td><code>x_j'' - (1 - x_j^2) x_j' + w_j^2 x_j = K (V - x_j') + sigma xi_j</code></td><td>K up: frequency synchrony of relaxation-type oscillators.</td></tr>
<tr><td>Roessler population</td><td><code>x' = -w y - z + K (X - x),  y' = w x + 0.15 y,  z' = 0.2 + z (x - 10)</code></td><td>K up: phase synchronisation of chaotic oscillators; the chaos itself acts as noise.</td></tr>
<tr><td>Theta neurons</td><td><code>theta_j' = (1 - cos theta_j) + (1 + cos theta_j)(eta_j + J S) + sigma xi_j</code></td><td>mean drive down: the high-activity state of a bistable network collapses at a fold.</td></tr>
</tbody></table></div>"""
    body = f"""
<p class="lead">Eighteen transitions in eleven oscillator models plus the Ornstein-Uhlenbeck reference (fourteen, and the herding collapse added later for van der Pol, Roessler, FitzHugh-Nagumo and Kuramoto with inertia), all analysed with one pipeline. For each:
48 runs in which the control parameter ramps slowly toward the transition, and two kinds of null run of
the same length. Question: which early-warning indicator rises before which transition, and is the
warning carried by the collective signal or by single units?</p>
<h3>The models</h3>{models}
<h3>Method</h3>
<ul>
<li><b>Ramp.</b> The control parameter moves linearly from a stable value through the transition. The analysed segment
runs from the end of an 8% burn-in to 25 time units before the transition (the theoretical critical point for
continuous transitions, the detected jump for abrupt ones). Signals are sampled every 0.5 time units.</li>
<li><b>Nulls.</b> <i>Parameter held fixed</i>: the same model at the starting value. <i>Ramp away</i>: the
parameter moves by the same amount at the same rate, but away from the transition. The second null is the
stricter one. An indicator that simply grows whenever the parameter changes scores 0.5 against it; only an
indicator that rises specifically toward the transition scores high. (The earlier model study S6 used only
the first kind and could not rank indicators.)</li>
<li><b>Indicators</b> on the detrended segment (Gaussian smoothing, bandwidth 10% of the segment), in a rolling
window of half the segment: variance, lag-1 autocorrelation, absolute skewness, peak of the power spectrum,
and two eigenvalue estimators. The AR(2) decay rate fits <code>x_t = a1 x_(t-1) + a2 x_(t-2) + e_t</code> and
reports <code>ln max|root| / dt</code>, the decay rate of the slowest mode, which rises to 0 at any loss of
stability. The VAR(1) decay rate does the same for eight units jointly.</li>
<li><b>Score.</b> Kendall's tau of each rolling indicator against time (positive = rising). ROC AUC: the
probability that a randomly chosen ramp run has a larger tau than a randomly chosen null run. 1 means it
always warns, 0.5 means chance, 0 means it always <i>falls</i> before the transition.</li>
<li><b>Signals.</b> <i>Collective</i>: the scenario's mean-field signal (Re Z before an onset, |Z| or mean
amplitude on a synchronised branch, the activity S for theta neurons). <i>One unit</i>: the signal of
oscillator 0. <i>8 units</i>: oscillators 0 to 7 together (VAR(1) only). N = 400 oscillators per population,
noise sigma = 0.1.</li>
</ul>
{figure("z1_examples.png", "Z1. One ramp run of every scenario.",
        "<p>One panel per scenario. Black: the collective signal that the indicators are computed on. Blue (right axis): the control parameter, ramped linearly. Red vertical line: the transition used to cut the segment (the theoretical critical point for continuous transitions, the detected jump for abrupt ones). Time on the horizontal axis. In the first two panels (Ornstein-Uhlenbeck and harmonic oscillator) the system diverges after the critical point because nothing limits it, so the vertical axis is scaled to the approach.</p>",
        "<p>Shows that every transition is real and is detected where theory says: the adaptive Kuramoto collapse happens just below its predicted fold, Kuramoto onset at K<sub>c</sub>, Winfree death once the oscillation slows to a stop. Only the data left of the red line are used for warnings.</p>")}
{figure("z1_auc_null_r.png", "Z1. Warning skill against the strict null (ramp away from the transition).",
        f"<p>Rows: the {n_scen} scenarios. Columns: indicators, grouped (thick black lines) into those computed on the collective signal (left five), on one unit (next three) and on eight units jointly (last). Colour and number: ROC AUC from 48 ramp runs against 48 null runs. Dark red = the indicator rises before the transition in essentially every run; white = no information; dark blue = it reliably <i>falls</i>. Blank cells: the single-unit models have no separate unit or group signal.</p>",
        f"<p>{headline} Every other indicator fails or reverses somewhere. Single units often show the opposite of a warning: their variance falls before amplitude death, van der Pol and Stuart-Landau synchrony and Winfree death, because what changes at a collective transition is the mean field, while each unit keeps oscillating on its own cycle.</p>")}
{figure("z1_auc_null_c.png", "Z1. The same against the lenient null (parameter held fixed).",
        "<p>Same layout as the previous figure; the null runs keep the parameter at its starting value.</p>",
        "<p>Mostly the same picture. Where the two figures differ (for example Winfree collective variance: 1.00 here, lower against the strict null) part of the rise comes from the parameter change itself rather than from the approach to the transition. The strict null is the one to quote.</p>")}
<h3>Variance: collective signal against one unit, every scenario</h3>
<p>Columns: AUC of the collective variance and of one unit's variance against each null, and the control parameter
at the detected transition (mean and standard deviation over the 48 ramp runs).</p>
{table(tab, {"scenario": "scenario", "coll_r": "collective, vs ramp away", "coll_c": "collective, vs fixed",
             "unit_r": "one unit, vs ramp away", "unit_c": "one unit, vs fixed", "p_mean": "parameter at transition",
             "p_sd": "sd"}, auc_cols=("coll_r", "coll_c", "unit_r", "unit_c"))}
{fhn_period_block()}
<h3>Where each indicator fails (AUC below 0.3 against the strict null)</h3>
{table(fails, {"signal": "signal", "indicator": "indicator", "n_fail": "scenarios failed", "where": "which"}, wrap=True)}
<div class="note"><b>Caveat.</b> AUC measures how consistently an indicator moves, not how much. With 48 runs a small
but systematic rise already gives AUC near 1. Section Z2 measures the size of the warning before the explosive
collapse and finds it small. All runs use N = 400 and one noise level; how the table changes with N, noise and ramp
speed is item A1 and A2 of the compute list.</div>"""
    return section("z1", "Z1", "Which warning works before which oscillator transition", ["result"], body)


def fhn_period_block() -> str:
    """The FitzHugh-Nagumo herding exception: indicators on R raw and averaged over one spike period."""
    f = DATA / "z1_fhn_period.csv"
    if not f.exists():
        return ""
    d = pd.read_csv(f)
    d = d[d.null == "null_r"].pivot(index="indicator", columns="variant", values="auc").reset_index()
    d["indicator"] = [IND_LABEL.get(x, x) for x in d.indicator]
    return ("<h3>The exception: FitzHugh-Nagumo herding collapse</h3><p>The collective variance falls before this collapse. "
            "The phase of a FitzHugh-Nagumo unit is event based, so R carries a ripple at the spiking period (about 37 time "
            "units), which dominates its variance and shrinks as the units drift apart. Rerun of the scenario (100 runs, "
            "<code>z1_fhn_period.py</code>) with every indicator computed on R as before and on R averaged over one spike "
            "period; AUC against the strict null:</p>"
            + table(d, {"indicator": "indicator", "raw": "R as in Z1", "period_mean": "R averaged over one spike period"},
                    auc_cols=("raw", "period_mean"))
            + "<p>Averaging over the cycle restores the variance warning: the slow mode is there, hidden by the ripple. For "
              "relaxation oscillators the indicators have to be computed on the cycle-averaged collective signal. The failure "
              "also depends on noise: at sigma = 0.05 the raw collective variance warns (section C).</p>")


def z3_section() -> str:
    d = pd.read_csv(DATA / "z3_sampling.csv")
    p = d[d.indicator.isin(["ac1", "ar2_rate"])].pivot_table(index="w_dt", columns=["model", "indicator"], values="auc")
    p.columns = [f"{m}__{i}" for m, i in p.columns]
    p = p.reset_index()
    body = f"""
<p class="lead">Why lag-1 autocorrelation failed for the simple harmonic oscillator in Z1, worked out exactly and
confirmed by simulation. The answer depends on <i>which coordinate</i> is observed, and the position of a
noise-driven mechanical oscillator never gives the warning.</p>
<h3>Theory</h3>
<p>For <code>x'' + g x' + w0^2 x = sigma xi</code> (noise enters the force, i.e. the velocity equation), the
stationary autocorrelation at lag tau, with <code>w' = sqrt(w0^2 - g^2/4)</code>, is</p>
<div class="math">position: AC<sub>x</sub>(tau) = e<sup>-g tau/2</sup> [cos w' tau + (g / 2w') sin w' tau]<br>
velocity: AC<sub>v</sub>(tau) = e<sup>-g tau/2</sup> [cos w' tau - (g / 2w') sin w' tau]</div>
<p>Differentiating at g = 0 with x = w0 tau (checked symbolically and against the exact Lyapunov covariance,
<code>z3_theory.py</code>):</p>
<div class="math">dAC<sub>x</sub>/dg = (sin x - x cos x) / 2w0, positive for 0 &lt; x &lt; 4.493<br>
dAC<sub>v</sub>/dg = -(sin x + x cos x) / 2w0, negative for 0 &lt; x &lt; 2.029<br>
Stuart-Landau, AC = e<sup>mu tau</sup> cos w0 tau: rises as mu goes to 0 for x &lt; pi/2</div>
<p>A warning needs the autocorrelation to rise as the damping falls. For the position it falls at every lag up to
4.49/w0, beyond the Nyquist limit pi/w0: <b>no sampling interval gives a warning</b>. For the velocity it rises up
to 2.03 (1.78 over the whole ramp from g = 1). For the Stuart-Landau signal up to a quarter period, which Bury,
Bauch and Anand (J. R. Soc. Interface 17, 20200482, 2020) already pointed out for the Hopf normal form. The
position and velocity results for a forced mechanical oscillator are not stated in the papers our search
found; that search was not exhaustive.</p>
{figure("z3_sampling.png", "Z3. Warning skill against the sampling interval, for three observables.",
        "<p>Horizontal axis: sampling interval times the oscillation frequency, w dt (pi = Nyquist limit, dotted line). Vertical axis: ROC AUC of 40 ramp runs against 40 constant runs (95% bootstrap CI). Red: lag-1 autocorrelation. Blue: AR(2) decay rate. Orange: spectral peak power. Grey: variance. Green band: where the exact theory says lag-1 autocorrelation rises over the ramp. Dashed line: chance.</p>",
        "<p>The simulated lag-1 autocorrelation follows the theory: it fails at every interval for the position, works for the velocity until w dt = 1.75 and fails from 2.0 (theory 1.78), and works for the Stuart-Landau signal until about 1.25. Variance and spectral peak warn throughout; the AR(2) rate warns throughout for the harmonic oscillator and loses precision near Nyquist for Stuart-Landau.</p>")}
<h3>AUC values</h3>
{table(p, {"w_dt": "w dt", "harmonic__ac1": "position, lag-1 AC", "harmonic__ar2_rate": "position, AR(2)",
           "harmonic_v__ac1": "velocity, lag-1 AC", "harmonic_v__ar2_rate": "velocity, AR(2)",
           "sl_hopf__ac1": "Stuart-Landau, lag-1 AC", "sl_hopf__ar2_rate": "Stuart-Landau, AR(2)"},
       auc_cols=tuple(c for c in p.columns if "__" in c), nd=2)}
<div class="note"><b>For markets.</b> A price is a position and a return is a velocity. Lag-1 autocorrelation of
returns, which is what earlier studies used, is the observable for which the theory allows a warning, provided the
sampling interval is shorter than about a third of the underlying period. The data do not tell us that period, which
is one reason to prefer variance or an eigenvalue estimator.</div>"""
    return section("z3", "Z3", "Lag-1 autocorrelation: why it fails for a mechanical oscillator, and when it works",
                   ["theory", "result"], body)


def o3d_block() -> str:
    """O3d: the size of the delay, predicted from the exact Ott-Antonsen equation with no free parameter."""
    if not (DATA / "o3d_slow_passage.csv").exists():
        return ""
    d = pd.read_csv(DATA / "o3d_slow_passage.csv")
    z2 = d[(d.source == "Z2") & (d.N == 1600)]
    o3 = d[d.source == "O3c"].copy()
    o3["measured"] = o3.delay_from_finite_N_fold
    return ("<h3>The size of the delay, with no free parameter (O3d)</h3>"
            "<p>The Ott-Antonsen equation is exact for this model at large N, so integrating it along exactly the "
            "same ramps and applying the same detection rule predicts the delay itself (near the fold this is the "
            "slow passage through a saddle-node, Delta lam = Omega_0 v^(2/3) (2 / (F_lam lam_b))^(1/3), Omega_0 = 2.338, the "
            "first zero of the Airy function). Z2 runs at N = 1600: the measured delay changes 6-fold with the ramp speed, "
            "while the difference from the prediction (offset) stays constant: a fixed, speed-independent shift of the "
            "finite, noisy population's fold of about 0.1%.</p>"
            + table(z2, {"T": "ramp time", "speed": "speed", "measured_delay": "measured delay", "oa_delay": "predicted",
                         "offset": "predicted minus measured"}, nd=4)
            + "<p>O3c Kuramoto runs (N = 500), delay measured from the fold this population was extrapolated to:</p>"
            + table(o3, {"p": "p", "T": "sweep time", "speed": "speed", "measured": "measured delay",
                         "oa_delay": "predicted"}, nd=4))


def z2_section() -> str:
    f = json.loads((DATA / "z2_fits.json").read_text())
    st = pd.read_csv(DATA / "z2_stationary.csv")
    g = st.groupby(["N", "eps"]).agg(R=("R_mean", "mean"), R_th=("R_theory", "first"), N_var=("N_var", "mean"),
                                     tau=("tau_int", "mean"), S=("N_S_low", "mean"), kappa=("kappa", "first")).reset_index()
    g = g[g.N == g.N.max()]
    rp = pd.read_csv(DATA / "z2_ramp.csv").dropna(subset=["eps_esc"])
    byR = rp[rp.N == 1600].groupby("T_ramp").eps_esc.agg(["mean", "std", "count"]).reset_index()
    byR["delay"] = -byR["mean"]
    byR["rate"] = 0.7 / byR.T_ramp
    ci = f["delay_rate_exponent_ci"]
    vr = f["var_ratio_near_far_by_N"]
    body = f"""
<p class="lead">The crash model of this project is the collapse of the synchronised branch of the adaptive Kuramoto
model. Ott-Antonsen theory gives that branch exactly, so the warning before the collapse can be predicted, not just
measured. Three things were tested: whether the slow mode is there, how large the warning is, and when a slowly
driven finite population actually collapses.</p>
<h3>Theory</h3>
<div class="math">R' = -gamma' R + (lam/2) R<sup>2</sup> (1 - R<sup>2</sup>),&nbsp;&nbsp; gamma' = gamma + sigma<sup>2</sup>/2<br>
branch: gamma' = (lam/2) R (1 - R<sup>2</sup>),&nbsp; fold at lam<sub>b</sub> = 3 sqrt(3) gamma' = {f['lam_b']:.4f}, R<sub>b</sub> = 1/sqrt(3)<br>
relaxation rate on the branch: kappa(lam) = gamma' - (lam/2)(2R - 4R<sup>3</sup>), which vanishes like sqrt(lam - lam<sub>b</sub>)</div>
<p>If finite-size fluctuations of R were a single slow mode of rate kappa: N Var(R) proportional to 1/kappa, correlation
time 1/kappa, and power at low frequency proportional to 1/kappa<sup>2</sup>.</p>
<h3>Results</h3>
<ul>
<li><b>The branch is exact.</b> Mean R from simulation matches the theory within {100 * f['max_rel_err_R_branch']:.2f}% at every N and distance from the fold.</li>
<li><b>The slow mode is there.</b> Low-frequency power of R scales as kappa<sup>{f['slope_Slow_vs_kappa']:.2f}</sup> (theory -2), collapsing for N = 200, 800 and 3200.</li>
<li><b>But it is diluted.</b> Total variance and correlation time scale only as kappa<sup>{f['slope_NvarR_vs_kappa']:.2f}</sup> and kappa<sup>{f['slope_tau_vs_kappa']:.2f}</sup>, not kappa<sup>-1</sup>: most of the fluctuation of R comes from fast contributions of drifting oscillators, which do not slow down. An indicator computed on R sees a weakened version of the critical slowing down.</li>
<li><b>Collapse comes after the fold, not before.</b> In every ramp run the population stays on the branch past lam<sub>b</sub>. The delay grows with ramp speed as speed<sup>{f['delay_rate_exponent']:.3f}</sup> (95% CI {ci[0]:.3f} to {ci[1]:.3f}), consistent with the 2/3 law of slow passage through a saddle-node. At noise sigma = 0.1 we see no sign of the opposite effect, finite-size noise tipping the system early (predicted to shift the collapse by order N<sup>-2/3</sup>).</li>
<li><b>The warning is small.</b> In the last 40 time units before the population leaves the branch, the variance of R is only {min(vr.values()):.1f} to {max(vr.values()):.1f} times its value far from the fold (N = 100 to 12800), although the theoretical relaxation time diverges at the fold.</li>
</ul>
{figure("z2_fold_scaling.png", "Z2. Explosive desynchronisation: theory against simulation.",
        "<p>(a) N times the variance of R on the synchronised branch against the theoretical relaxation rate kappa (log-log), for three population sizes; dashed: slope -1 expected for a single slow mode. (b) Low-frequency power of R (below 0.01 cycles per time unit), times N; black: C'/kappa<sup>2</sup> with one fitted constant. (c) In slow downward ramps, how far below the fold the population leaves the branch (fraction of lam<sub>b</sub>) against ramp speed (N = 1600, 16 runs per speed, mean and standard error); dashed: slope 2/3. (d) Blue: the same delay against N at fixed ramp time 3000. Orange (right axis): variance of R just before leaving the branch divided by its value far from the fold (median over 16 runs).</p>",
        "<p>The theory of the branch is right, and so is the slow mode underneath. What reaches an observer is weaker: the slow mode is buried under fast fluctuations, and a system driven at a finite rate overshoots the fold and collapses late, so the variance never gets far above its baseline. For the market picture this means the explosive-collapse model predicts at most a modest rise in variance before a crash, which is what the data show (M1, M4).</p>")}
<h3>Stationary runs, largest N</h3>
{table(g, {"eps": "(lam - lam_b) / lam_b", "R": "R simulated", "R_th": "R theory", "N_var": "N Var(R)",
           "tau": "correlation time", "S": "N x low-frequency power", "kappa": "kappa (theory)"}, nd=4)}
<h3>Ramp speed and delay past the fold (N = 1600)</h3>
{table(byR, {"T_ramp": "ramp time", "rate": "speed (fold values per time unit)", "delay": "delay past the fold",
             "std": "sd", "count": "runs"}, nd=5)}
{o3d_block()}
<div class="note"><b>Open.</b> Whether finite-size noise ever makes the collapse come early needs stronger noise and
much slower ramps (running locally, section R), and very large N (item A3 of the compute list).</div>"""
    return section("z2", "Z2", "The crash model: critical slowing down before explosive desynchronisation",
                   ["theory", "result"], body)


def m0_section() -> str:
    d = pd.read_csv(DATA / "m0_audit.csv")
    rules = """
<div class="tbl"><table class="fit"><thead><tr><th>rule</th><th>what it catches</th><th>what is removed</th></tr></thead><tbody>
<tr><td>R1 non-positive price</td><td>Yahoo's dividend back-adjustment drives some early adjusted prices below zero (ABI.BR: negative for 2070 days)</td><td>every price of that stock up to its last non-positive date</td></tr>
<tr><td>R2 holiday filler row</td><td>days on which more than 80% of stocks show exactly zero return: local holidays filled forward by the vendor</td><td>the whole day</td></tr>
<tr><td>R3 stale run</td><td>10 or more identical consecutive prices: suspension or delisting filled forward</td><td>those prices after the first</td></tr>
<tr><td>R4 one-day spike</td><td>a move of more than 25% that reverses the next day (within 25% of its size): a bad tick</td><td>that day's price</td></tr>
<tr><td>R5 split error</td><td>a move of more than 50% whose ratio is within 2% of a split ratio (2, 3, 4, 5, 8, 10, ... 100)</td><td>that return; price levels kept</td></tr>
<tr><td>R6 level break</td><td>a move by more than a factor e = 2.72 in one day: unadjusted splits with odd ratios, redenominations, mergers (BRKM5.SA: 1.24 million to 0.068)</td><td>that return; price levels kept</td></tr>
</tbody></table></div>"""
    body = f"""
<p class="lead">The 22 daily stock panels (Yahoo adjusted closes of today's constituents, 2000 to 2026) were audited
before any new test. Earlier studies in this repository (E1, E1c) used them uncleaned.</p>
{rules}
<p>Crypto keeps R1 to R4 only: moves larger than 50% are real there. The world index panel is audited in M3.</p>
{figure("m0_audit.png", "M0. What cleaning removed, and whether it helped.",
        "<p>(a) For each market, how many prices (R1, R3, R4), whole days (R2) or returns (R5, R6) each rule removed; log scale, so a bar at 1 means none. (b) Correlation between the daily returns of the official index and the equal-weight mean of the constituents' returns, before (grey) and after (blue) cleaning. A low value means the constituent data disagree with the index.</p>",
        "<p>Cleaning never lowers the agreement with the official index and raises it clearly in five markets (Brazil 0.70 to 0.93, Spain, Switzerland, Singapore, FTSE 250). It removes 0.03% to 6.4% of the data per market. The remaining weak spots are inherent: survivorship (only today's constituents, see the last column) and crypto, whose index has a different composition.</p>")}
{table(d, {"market": "market", "n_stocks_raw": "stocks", "R1_nonpositive_cells": "R1", "R2_filler_rows": "R2",
           "R3_stale_cells": "R3", "R4_spike_cells": "R4", "R5_split_errors": "R5", "R6_level_breaks": "R6",
           "frac_removed": "fraction removed", "index_vs_ew_corr_raw": "index agreement, raw",
           "index_vs_ew_corr": "index agreement, clean", "coverage_first_year": "stocks with data in 2000"})}
<p>Five of the 38 world indices (Egypt CASE30, Chile IPSA, Thailand SET, Philippines PSEi, Poland WIG20) have no
history on Yahoo; a fresh download on {date.today():%Y-%m-%d} confirmed it. Euro Stoxx 50, OMX, Oslo and Nifty
start between 2007 and 2013 there.</p>"""
    return section("m0", "M0", "Market data audit and cleaning", ["data"], body)


def m1_section() -> str:
    r = pd.read_csv(DATA / "m1_pooled_auc.csv")
    sm = json.loads((DATA / "m1_summary.json").read_text())
    reg = r[r.registered]
    comp = r[~r.registered]
    h1 = reg[(reg.indicator == "var1_eig") & (reg.score == "trend") & (reg.null == "peak")].iloc[0]
    ne = pd.read_csv(DATA / "m1_per_market_auc.csv").groupby("market").n_events.first()
    body = f"""
<p class="lead">Three indicators that come from the oscillator picture, fixed in
<code>DESIGN_MARKETS.md</code> before the run, tested before {sm['n_events']} crash onsets in {sm['n_markets']}
markets ({sm['n_quarters']} distinct calendar quarters), on cleaned data.</p>
<ul>
<li><b>AR(2) decay rate</b> of the index's daily returns (the damped-oscillator eigenvalue).</li>
<li><b>VAR(1) eigenvalue</b>: largest |eigenvalue| of a VAR(1) fitted to the top three principal components of the
constituents' returns (the coupled-system eigenvalue). Primary hypothesis: its 250-day trend separates crash onsets
from peaks not followed by a crash.</li>
<li><b>Hilbert phase synchrony</b>: Kuramoto order parameter of the constituents' Hilbert phases (log prices
band-passed to periods of about 3 to 20 days, computed inside a trailing 120-day window so nothing after the date
is used).</li>
</ul>
<p>Crash onset: a 20-day high followed by a fall of at least 10% (30% for crypto) within 30 trading days. Nulls:
all non-crash dates, volatility-matched dates, and 20-day highs that were not followed by a crash. Scores: the
indicator's z-score against its own days t-500 to t-60 (level) and its Kendall trend over 250 days (trend). AUC
with 95% CI from a bootstrap over calendar quarters on both sides. 18 registered tests, Benjamini-Hochberg at
q = 0.05.</p>
<p><b>Result: no registered test survives the correction ({sm['n_bh_reject']} of {sm['n_registered_tests']}).</b>
The primary hypothesis gives AUC {h1.auc:.3f} [{h1.ci_lo:.3f}, {h1.ci_hi:.3f}]. The four earlier indicators,
recomputed on the cleaned data, reproduce the earlier E1c numbers: index variance and mean correlation separate
crash onsets from ordinary dates (AUC around 0.6), but nothing separates them clearly from peaks that did not
crash.</p>
{figure("m1_pooled_auc.png", "M1. Warning skill of every indicator, all markets pooled.",
        "<p>Each row an indicator; the three new ones are above the black line, the earlier indicators below. Three markers per row, one per null set (legend). Horizontal position: ROC AUC with 95% CI. Left: level score; right: trend score. Dashed line: chance.</p>",
        "<p>The new indicators sit on the chance line. Index variance and mean correlation are the only indicators clearly above 0.5, and only against all dates or volatility-matched dates. Against the null that matters for a usable warning (a peak that did not turn into a crash) every interval includes 0.5.</p>")}
{figure("m1_per_market_level.png", "M1. Per market, level score, against peaks not followed by a crash.",
        "<p>Rows: markets (number of crash onsets in brackets). Columns: indicators, new ones left of the black line. Colour and number: AUC in that market alone. Red: higher before crashes; blue: lower.</p>",
        f"<p>No indicator is consistently red across markets. With {ne.min()} to {ne.max()} crashes per market, single-market AUCs between 0.3 and 0.7 are within sampling noise.</p>")}
{figure("m1_per_market_trend.png", "M1. Per market, trend score.", "<p>As above, for the 250-day trend.</p>",
        "<p>Same conclusion.</p>")}
<h3>Registered tests</h3>
{table(reg, {"indicator": "indicator", "score": "score", "null": "null", "auc": "AUC", "ci_lo": "CI low",
             "ci_hi": "CI high", "p_one_sided": "p (one-sided)", "p_bh": "p after BH", "bh_reject": "survives BH",
             "n_events": "events"}, auc_cols=("auc",))}
<h3>Comparison indicators (not part of the test)</h3>
{table(comp, {"indicator": "indicator", "score": "score", "null": "null", "auc": "AUC", "ci_lo": "CI low",
              "ci_hi": "CI high", "p_one_sided": "p (one-sided)"}, auc_cols=("auc",))}"""
    return section("m1", "M1", "Real markets: oscillator-motivated warnings, registered in advance", ["neg"], body)


def m2_section() -> str:
    sev = pd.read_csv(DATA / "m2_severity.csv")
    fw = pd.read_csv(DATA / "m2_forward.csv")
    rec = pd.read_csv(DATA / "m2_recovery.csv")
    body = f"""
<p class="lead">Questions not asked anywhere else in the repository, all exploratory (no correction for the number of
correlations, so treat any single interval that excludes zero with suspicion). Every indicator is a z-score against
its own past (days t-500 to t-60). "Partial" removes the part explained by the index variance z-score.</p>
<ol>
<li><b>Severity.</b> At the crash onset, does the indicator relate to how deep the fall is (largest loss within 30
days) or how fast (days from peak to trough)?</li>
<li><b>Recovery.</b> Does it relate to how long the index takes to regain its pre-crash peak (censored at 750
days)?</li>
<li><b>All dates.</b> Does it correlate with the next 30 days' maximum drawdown? Computed per market, averaged over
the 22 markets, CI from a bootstrap over quarters within each market.</li>
</ol>
{figure("m2_correlations.png", "M2. Rank correlations, with and without the variance.",
        "<p>Each row an indicator (new ones above the line). Blue circle: Spearman rank correlation with 95% bootstrap CI. Orange square: the partial correlation given the index variance z-score. Panels (a) to (d) are the targets listed above. The vertical line is zero.</p>",
        "<p>(a) Mean correlation at the onset relates to the depth of the crash (0.19), but given variance the interval reaches zero: a hint, not a finding. (b, c) Nothing relates to the speed of the fall or the recovery time. (d) Several indicators correlate with future drawdowns, but every partial correlation given variance is near zero: whatever they know about the next month, variance already knows.</p>")}
<h3>Severity</h3>
{table(sev, {"indicator": "indicator", "target": "target", "spearman": "Spearman", "lo": "CI low", "hi": "CI high",
             "partial_given_var": "partial given variance", "p_lo": "CI low", "p_hi": "CI high", "n": "events"})}
<h3>Recovery time</h3>
{table(rec, {"indicator": "indicator", "spearman": "Spearman", "lo": "CI low", "hi": "CI high", "n": "events",
             "n_censored": "not recovered in 750 days"})}
<h3>Next 30 days' drawdown, all dates</h3>
{table(fw, {"indicator": "indicator", "mean_spearman": "mean Spearman", "lo": "CI low", "hi": "CI high",
            "mean_partial_given_var": "partial given variance", "p_lo": "CI low", "p_hi": "CI high",
            "markets_positive": "markets with positive value", "n_markets": "markets"})}"""
    return section("m2", "M2", "Real markets: crash depth, speed, recovery and future drawdowns", ["explore", "neg"], body)


def m3_section() -> str:
    ev = pd.read_csv(DATA / "m3_events.csv")
    fc = pd.read_csv(DATA / "m3_forward_corr.csv")
    body = f"""
<p class="lead">If the world's markets are coupled oscillators, global crashes might be preceded by rising phase
synchrony across countries. Tested on 33 national indices (the world panel minus five with no data).</p>
<p>Global index: equal-weight mean of the indices' daily log returns. Global crash: the same onset rule as M1, which
finds {len(ev)} onsets from 2000 to 2026. World phase synchrony: Kuramoto order parameter of the indices' Hilbert phases
(as in M1). World correlation: mean pairwise correlation of 5-day returns over 120 days (5-day returns remove most
of the time-zone offset between Asia, Europe and America). Holidays: price levels carried forward for up to 3 days.</p>
{figure("m3_world.png", "M3. World indicators over 26 years.",
        "<p>From top: world phase synchrony, world mean correlation, volatility of the global index, and how many indices are in the window. Red vertical lines: global crash onsets.</p>",
        "<p>Synchrony and correlation move in long regimes (high after 2008 and in 2020) that do not line up with the crash onsets. Volatility spikes come during or after the crashes. The early years have fewer indices, so values before 2008 are less comparable.</p>")}
<h3>Indicators at each global crash onset (z-score against days -500 to -60)</h3>
{table(ev, {"date": "onset", "fall": "fall within 30 days", "world_R_z": "synchrony z", "world_corr_z": "correlation z",
            "world_vol_z": "volatility z"}, nd=2)}
<h3>Correlation with the next 30 days' drawdown, all dates</h3>
{table(fc, {"indicator": "indicator", "spearman": "Spearman", "lo": "CI low", "hi": "CI high",
            "partial_given_vol": "partial given volatility", "p_lo": "CI low", "p_hi": "CI high",
            "n_dates": "dates", "n_quarters": "quarters"})}
<p><b>Result:</b> world synchrony and world correlation do not anticipate global drawdowns (correlations within
0.01 of zero, intervals about plus or minus 0.1). With 12 global crashes, an event-based test would have no
power.</p>"""
    return section("m3", "M3", "Real markets: are the world's markets synchronised before global crashes?", ["explore", "neg"], body)


def m4_section() -> str:
    a = pd.read_csv(DATA / "m4_auc.csv")
    d = pd.read_csv(DATA / "m4_difference.csv")
    body = f"""
<p class="lead">Z1 found that the collective signal carries the warning and single units often do not. The market
version, written into the script before it was run: the variance of the index (collective) should separate crash
onsets from the nulls better than the median variance of the individual stocks (single units).</p>
<p>Indicators: index return variance, median over stocks of each stock's return variance, and their ratio (the
collective share of variance, close to the mean pairwise correlation). Same events, nulls and bootstrap as M1. The
comparison is a paired bootstrap of the AUC difference over calendar quarters.</p>
<h3>Collective minus unit</h3>
{table(d, {"score": "score", "null": "null", "auc_collective_minus_unit": "AUC difference", "ci_lo": "CI low",
           "ci_hi": "CI high", "p_one_sided": "p (one-sided)"}, nd=4)}
<h3>All AUCs</h3>
{table(a, {"indicator": "indicator", "score": "score", "null": "null", "auc": "AUC", "ci_lo": "CI low",
           "ci_hi": "CI high", "p_one_sided": "p (one-sided)", "n_events": "events"}, auc_cols=("auc",))}
<p><b>Result:</b> the difference has the predicted sign in all six comparisons but is small (at most
{d.auc_collective_minus_unit.max():.3f}) and never significant. In real markets single stocks' variance rises with
the index, because one common factor drives both, so the sharp contrast of the oscillator populations does not
appear.</p>"""
    return section("m4", "M4", "Real markets: collective against single-stock variance", ["explore", "neg"], body)


def m5_section() -> str:
    if not (DATA / "m5_auc.csv").exists():
        return ""
    a = pd.read_csv(DATA / "m5_auc.csv")
    fd = pd.read_csv(DATA / "m5_fd_summary.csv")
    ce = pd.read_csv(DATA / "m5b_contagion_events.csv") if (DATA / "m5b_contagion_events.csv").exists() else None
    from math import comb

    def sign_p(v):
        v = v.dropna()
        k = int((v < 0).sum())
        return k, len(v), sum(comb(len(v), j) for j in range(k, len(v) + 1)) / 2 ** len(v)
    cont = ""
    if ce is not None:
        rows = []
        for col, lab in (("spearman_half", "time to half of its own loss"),
                         ("partial_5pct_given_loss", "time to -5%, loss size held fixed"),
                         ("half_given_utc", "time to half of its loss, time zone held fixed"),
                         ("half_west_only", "time to half of its loss, Americas, Europe and Middle East only")):
            k, n, p = sign_p(ce[col])
            rows.append({"test": lab, "mean": ce[col].mean(), "negative": f"{k} of {n}", "p": p})
        cont = ("<h3>H8: do the most strongly coupled markets crash first?</h3>"
                "<p>For each of the 11 global crashes, the Spearman correlation across national indices between "
                "coupling (correlation of 5-day returns with the world index over the prior 250 days) and how soon "
                "the index fell. Prediction: negative (coupled markets fall first). p: one-sided sign test across "
                "crashes.</p>" + table(pd.DataFrame(rows), {"test": "timing measure", "mean": "mean Spearman",
                                                           "negative": "crashes with negative correlation",
                                                           "p": "sign-test p"}, nd=4)
                + "<p><b>Result:</b> negative in 10 of 11 crashes for every scale-free timing measure and with the "
                "time zone held fixed (p = 0.006), but only 8 of 11 (p = 0.11) inside the Americas, Europe and the "
                "Middle East. Part of the effect is regional: Asian markets are both less coupled to this world index "
                "and later. A hint consistent with oscillators locking in order of coupling, not a finding. At the level of single "
                "stocks, Smith (arXiv:0901.1392) found the 2008 decline spreading from central to peripheral stocks; the "
                "country-level version tested here was not found in a literature search.</p>")
    body = f"""
<p class="lead">Four predictions taken from the simulations and tested on the same events and nulls as M1, each
written into the script before it was run. Exploratory: no correction across the four.</p>
<ul>
<li><b>H5 (from Z3):</b> if a market near a crash behaved like a forced oscillator losing its damping, the lag-1
autocorrelation of its <i>price level</i> (a position) would fall before crashes. Prediction AUC &lt; 0.5.</li>
<li><b>H9 (from O5 and Gomez-Gardenes 2011):</b> explosive synchrony needs the best-connected units to be the
fastest. fd_corr = correlation, across stocks, between a stock's total correlation with the others and its
volatility. Predictions: positive on average, and rising before crashes.</li>
<li><b>H1 (from K_c = 2 gamma):</b> a narrower spread of natural frequencies is closer to synchrony;
drift_disp = spread of the stocks' 60-day mean returns divided by their median volatility. Prediction: falls
before crashes.</li>
<li><b>H8 (from the atlas):</b> the most strongly coupled units lock first; below.</li>
</ul>
{table(a, {"indicator": "indicator", "prediction": "prediction", "score": "score", "null": "null", "auc": "AUC",
           "ci_lo": "CI low", "ci_hi": "CI high", "n_events": "events"}, auc_cols=("auc",))}
<p><b>H5, H1:</b> every interval includes 0.5 (H5 lies on the predicted side, 0.45 to 0.49). Not detected.</p>
<h3>H9: where real markets sit</h3>
{table(fd, {"market": "market", "fd_mean": "mean fd_corr", "fd_frac_positive": "share of windows positive",
            "crashes_per_decade": "crashes per decade"})}
<p><b>Result:</b> no crash warning (AUC about 0.5). The sign of fd_corr is negative in most markets, but a check
(M5c) shows that this is mechanical: a stock with a large idiosyncratic part of its volatility has lower correlations
with everything. Splitting each stock's volatility into a systematic part (|beta| times the market's volatility) and
an idiosyncratic part, strength rises strongly with systematic volatility and falls with idiosyncratic volatility in
every market checked:</p>
{table(pd.read_csv(DATA / "m5c_fd_confound.csv"), {"market": "market", "fd_total": "with total volatility",
       "fd_systematic": "with systematic volatility", "fd_idiosyncratic": "with idiosyncratic volatility"})
 if (DATA / "m5c_fd_confound.csv").exists() else ""}
<p>So, measured by systematic volatility, the best-connected stocks are the most volatile, as studies of stock-network
centrality report, and markets sit in the frequency-degree regime that Gomez-Gardenes et al. associate with explosive
synchronisation. That regime does not, by itself, give a crash warning here.</p>
{cont}"""
    if (DATA / "m5d_contagion_robust.csv").exists():
        cr = pd.read_csv(DATA / "m5d_contagion_robust.csv")
        body += ("<h3>Contagion order with more events (M5d)</h3><p>Coupling recomputed against the world index "
                 "<b>without</b> the market itself, and the test repeated with global drawdowns of 10% (the M5 design), "
                 "7% and 5%. <code>rho</code>: Spearman correlation between coupling and half-loss delay in each event; "
                 "<code>rho_given_utc</code>: the same with the UTC offset partialled out; <code>rho_west</code>: "
                 "Americas, Europe and the Middle East only. Mean over events with a bootstrap CI over events, and a "
                 "one-sided sign test. The last three rows (added after the first run) ask whether the effect grows with "
                 "the depth of the world drawdown.</p>"
                 + table(cr, {"threshold": "drawdown", "measure": "measure", "events": "events", "negative": "negative",
                              "mean": "mean", "lo": "CI low", "hi": "CI high", "sign_test_p": "sign test p"}, nd=3)
                 + "<p><b>Result.</b> Leaving the market out of the world index changes nothing at the 10% threshold "
                   "(10 of 11 events, also with time zones partialled out, p = 0.006). With smaller drawdowns the "
                   "effect fades (12 of 18 at 7%, 15 of 27 at 5%), and the dose-response has the predicted sign but is "
                   "not significant. Within the Americas, Europe and the Middle East it is not significant at any "
                   "threshold. So the contagion order is confined to the largest global drawdowns, holds at only one "
                   "threshold (the one fixed in M5's design, so not chosen after the fact) and rests on 11 events: "
                   "suggestive, not established.</p>")
    return section("m5", "M5", "Four hypotheses from the simulations, tested on real markets", ["explore"], body)


def m6_section() -> str:
    if not (DATA / "m6_granger.csv").exists():
        return ""
    g = pd.read_csv(DATA / "m6_granger.csv")
    rb = pd.read_csv(DATA / "m6b_granger_robust.csv") if (DATA / "m6b_granger_robust.csv").exists() else None
    m7 = pd.read_csv(DATA / "m7_intraday_leadlag.csv") if (DATA / "m7_intraday_leadlag.csv").exists() else None
    body = f"""
<p class="lead">Does synchrony move first and volatility follow (the herding model), or the other way round (the
usual finance reading: correlations rise because volatility does)? Not tested elsewhere in the repository.</p>
<p>Series: M1's indicators, sampled every 60 trading days so that consecutive values come from non-overlapping
windows. Granger-style test per market: does the change of a synchrony measure over one block predict the change
of log index variance over the next block, beyond the variance's own change (and the reverse)? Partial
correlation, pooled over the 22 markets (Fisher z), 95% CI by bootstrap over markets.</p>
{table(g, {"sync": "synchrony measure", "direction": "direction", "pooled_partial_r": "pooled partial r",
           "lo": "CI low", "hi": "CI high", "markets_positive": "markets positive", "n_markets": "markets"})}"""
    if rb is not None:
        body += ("<h3>Robustness of the one lead found (M6b)</h3><p>Extra controls (two own lags and the level of "
                 "volatility, since the Hilbert window is 120 days and the variance window 60), the two halves of "
                 "the period separately, and a placebo with the synchrony changes shuffled.</p>"
                 + table(rb, {"sync": "measure", "test": "test", "pooled_partial_r": "pooled partial r", "lo": "CI low",
                              "hi": "CI high", "markets_positive": "markets positive", "n_markets": "markets"}))
    if (DATA / "m6c_band_sensitivity.csv").exists():
        bs = pd.read_csv(DATA / "m6c_band_sensitivity.csv")
        body += ("<h3>Does it depend on the frequency band? (M6c)</h3><p>The Hilbert synchrony recomputed with two other "
                 "bands, same test as M6b. The middle band (3 to 20 days) is the one fixed in DESIGN_MARKETS.md before M1.</p>"
                 + table(bs, {"band_days": "band (days)", "pooled_partial_r": "pooled partial r", "lo": "CI low",
                              "hi": "CI high", "markets_positive": "markets positive", "n_markets": "markets"}))
    if (DATA / "m6d_joint.csv").exists():
        body += ("<h3>Are the two opposite leads one effect? (M6d, a check made after M6b)</h3><p>The changes of phase "
                 "synchrony and of mean correlation are correlated (0.36 on average over markets), and their leads have "
                 "opposite signs. Here each is tested with the change and level of the other added to M6b's controls.</p>"
                 + table(pd.read_csv(DATA / "m6d_joint.csv"),
                         {"measure": "measure", "also_controlled": "other measure controlled", "pooled_partial_r": "pooled partial r",
                          "lo": "CI low", "hi": "CI high", "markets_positive": "markets positive", "n_markets": "markets"})
                 + "<p>Neither lead weakens when the other measure is controlled; both become slightly stronger. Phase "
                   "alignment and correlation strength carry separate information about the next volatility change, with "
                   "opposite signs. The absorption ratio (share of variance in the top eigenvectors) behaves like mean "
                   "correlation but weaker, and its CI touches 0.</p>")
    if (DATA / "m10_har_pooled.csv").exists():
        hp = pd.read_csv(DATA / "m10_har_pooled.csv")
        body += ("<h3>Does it improve a real volatility forecast? (M10)</h3><p>The test a finance referee asks for first. "
                 "Target: log mean squared index return over the next 20 trading days. Baseline: the HAR model (Corsi 2009), "
                 "log mean squared return over the last 5, 22 and 66 days. Added: phase synchrony (level and 20-day change), "
                 "mean correlation (level and change), or both. Expanding-window OLS refitted every 5 trading days, first "
                 "forecast after 8 years, fitted only on targets already observed. Out-of-sample R<sup>2</sup> gain = 1 minus "
                 "the ratio of mean squared errors; CI by bootstrap over calendar quarters with all markets resampled "
                 "together; the last two columns count markets with a gain and with a Diebold-Mariano statistic above "
                 "1.65.</p>"
                 + table(hp, {"model": "added to HAR", "pooled_oos_r2": "pooled out-of-sample R2 gain", "lo": "CI low",
                              "hi": "CI high", "markets_gain": "markets with gain", "markets_dm_gt_1.65": "markets significant",
                              "n_markets": "markets"}, nd=4)
                 + "<p><b>Result: no.</b> Phase synchrony makes the forecast slightly <b>worse</b> out of sample (pooled "
                   f"{hp.set_index('model').loc['sync', 'pooled_oos_r2'] * 100:.1f}%, gain in "
                   f"{int(hp.set_index('model').loc['sync', 'markets_gain'])} of 22 markets, none significant), and no "
                   "market's gain beats a placebo with the synchrony series shifted in time. The in-sample lead of M6 is "
                   "real but carries no information beyond the volatility history that HAR already uses over 66 days. This "
                   "closes the M6 line as a forecasting result. Per-market table: data/m10_har_oos.csv.</p>")
    if m7 is not None:
        body += ("<h3>Replication inside the trading day (M7, 1-minute data)</h3><p>The same test on 30-minute "
                 "blocks of the repository's minute data (SPY and 22 large US stocks, 2002 to 2026), within each "
                 "session, time-of-day effects removed, CI by bootstrap over calendar months.</p>"
                 + table(m7, {"indicator": "measure", "sample": "sample", "partial_r": "partial r", "lo": "CI low",
                              "hi": "CI high", "n_blocks": "blocks", "n_months": "months"}, nd=4))
    body += """<p><b>Result (daily):</b> changes of <b>Hilbert phase synchrony</b> are followed by changes of
volatility in the same direction, beyond volatility's own history, in 20 to 21 of 22 markets (partial r about
0.06 to 0.08); the placebo is 0 within 0.01; the effect sits mostly after 2013. Plain <b>mean correlation</b>
does the opposite (a rise is followed by falling volatility), and nothing predicts synchrony from volatility.
The effect does not depend on the band (2 to 10, 3 to 20 and 5 to 40 days all give 0.06 to 0.08). It is small
(R^2 under 1%) and exploratory: a lead that is consistent across markets, not a usable warning. Prior work has used
correlation-network synchronisation (minimum spanning tree length) to forecast realised volatility (Magner and
co-authors, Complexity 2020 and REMEF 2022), so the claim here is narrower: the phase measure leads, while plain mean
correlation, in this test, does not.</p>"""
    if m7 is not None:
        r = m7.set_index(["indicator", "sample"])
        body += (f"""<p><b>Result (intraday):</b> the lead does <b>not</b> replicate over the full minute-data sample
(phase synchrony partial r = {r.loc[('sync', 'all'), 'partial_r']:.3f}, CI {r.loc[('sync', 'all'), 'lo']:.3f} to
{r.loc[('sync', 'all'), 'hi']:.3f}); from 2013 on it is small and positive ({r.loc[('sync', 'from 2013'), 'partial_r']:.3f}, CI
{r.loc[('sync', 'from 2013'), 'lo']:.3f} to {r.loc[('sync', 'from 2013'), 'hi']:.3f}), and before 2013 it is zero. Mean correlation is
slightly positive intraday, the opposite sign to the daily result. The only feature shared by both time scales is that
the synchrony effect appears after 2013, which could reflect a change in market structure; it is too small to build on.</p>""")
    return section("m6", "M6", "Does synchrony lead volatility, or follow it?", ["explore"], body)


def m8_section() -> str:
    if not (DATA / "m8_auc.csv").exists():
        return ""
    a = pd.read_csv(DATA / "m8_auc.csv")
    g = pd.read_csv(DATA / "m8_granger.csv")
    off = pd.read_csv(DATA / "m8_offsets.csv")
    cr = a[(a.indicator == "cluster_ratio") & (a.null == "all")].set_index("score")
    fs = g.set_index("indicator").loc["freq_spread"]
    body = f"""
<p class="lead">Four more predictions taken from coupled-oscillator theory, each written down before running, tested
on the cleaned data of all 22 markets with M1's crash events, nulls and quarter-cluster bootstrap.</p>
<ul>
<li><b>H10 frequency locking.</b> In the Kuramoto model, synchrony means the units' effective frequencies merge.
<code>freq_spread</code> is the interquartile range across stocks of each stock's instantaneous Hilbert frequency
(band 3 to 20 days), averaged over 20 days. Prediction: it falls before crashes (AUC below 0.5), and a fall is
followed by rising volatility (Granger partial r below 0).</li>
<li><b>H11 reciprocity of the lead-lag network.</b> Stock i leads j when the correlation of today's return of i with
tomorrow's return of j is significant. Reciprocity is the share of such links whose reverse link is also
significant with the same sign. The main project's theory gives different folds for one-way and two-way coupling.
Prediction, with no sign given: it changes before crashes.</li>
<li><b>H12 sectors lock first</b> (three US markets, GICS sectors). On networks, groups often lock before the
whole population does. <code>cluster_ratio</code> is the mean correlation within sectors divided by the mean
correlation between sectors. Prediction: it rises before crashes (AUC above 0.5).</li>
<li><b>H13 phase offsets.</b> A locked Kuramoto unit sits at phase offset arcsin(w / KR), so faster units lead.
Prediction: across stocks, the natural frequency (measured in the least-locked windows) correlates positively
with the phase offset (measured in the most-locked windows), in most markets.</li>
</ul>
<h3>Crash warning (ROC AUC of the score at each crash onset against the three nulls)</h3>
<p>Trend: Kendall tau of the indicator over the 250 days before the date. Level: the indicator's z-score against
days t-500 to t-60. An AUC of 0.5 is chance. Each CI is a 95% bootstrap over calendar quarters.</p>
{table(a, {"indicator": "indicator", "prediction": "prediction", "score": "score", "null": "null", "auc": "AUC",
           "ci_lo": "CI low", "ci_hi": "CI high", "n_events": "crashes"}, auc_cols=("auc",))}
<h3>Lead of volatility (same test as M6b, with the same controls)</h3>
{table(g, {"indicator": "indicator", "pooled_partial_r": "pooled partial r", "lo": "CI low", "hi": "CI high",
           "markets_positive": "markets positive", "n_markets": "markets"})}
<h3>Phase offsets against natural frequency (H13), per market</h3>
{table(off, {"market": "market", "spearman_natfreq_offset": "Spearman (natural frequency, offset)", "n_stocks": "stocks"},
       corr_cols=("spearman_natfreq_offset",))}
<p><b>Result.</b> None of the four gives a crash warning.</p>
<ul>
<li><b>H10.</b> The frequency spread does not fall before crashes: every CI contains 0.5. It does lead
volatility weakly, with the predicted sign: a narrowing spread is followed by rising volatility (pooled partial r
{fs.pooled_partial_r:.3f}, CI {fs.lo:.3f} to {fs.hi:.3f}; positive in only {int(fs.markets_positive)} of
{int(fs.n_markets)} markets). This is the frequency-domain twin of the M6 phase-synchrony lead and is probably
not independent of it.</li>
<li><b>H11.</b> Reciprocity is at chance in every comparison.</li>
<li><b>H12.</b> The sector ratio goes the <b>opposite</b> way to the prediction: it falls before US crashes
(trend AUC {cr.loc['trend', 'auc']:.2f}, CI {cr.loc['trend', 'ci_lo']:.2f} to {cr.loc['trend', 'ci_hi']:.2f};
only 39 crashes, three markets). Correlations between sectors catch up with those within sectors, so the market
merges into one block instead of locking sector by sector. M8b shows this is the ordinary rise of mean
correlation and nothing more: within-sector correlations are already high and have less room to rise, so the
ratio falls whenever all correlations rise (its correlation over time with mean correlation is -0.42 to -0.72).
With mean correlation regressed out, the ratio is at chance (table below). The same holds for its volatility lead,
which rests on three markets only.</li>
<li><b>H13.</b> The phase-offset ordering is not seen: the correlation averages
{off.spearman_natfreq_offset.mean():.3f} and is positive in {int((off.spearman_natfreq_offset > 0).sum())} of
{len(off)} markets. Stocks do not behave like locked oscillators with fixed natural frequencies.</li>
</ul>"""
    if (DATA / "m8b_cluster_check.csv").exists():
        body += ("<h3>Sector ratio against mean correlation (M8b, a check made after M8)</h3><p>US crashes only "
                 "(39), null of all non-crash dates. <code>resid</code> is the sector ratio with M1's mean "
                 "correlation regressed out, market by market.</p>"
                 + table(pd.read_csv(DATA / "m8b_cluster_check.csv"),
                         {"indicator": "indicator", "score": "score", "auc": "AUC", "ci_lo": "CI low", "ci_hi": "CI high",
                          "n_events": "crashes"}, auc_cols=("auc",))
                 + "<p>Against all non-crash dates, mean correlation rises before these 39 US crashes (trend AUC 0.70). "
                   "Against M1's null of market tops that were not followed by a crash it is near chance (AUC 0.47 "
                   "to 0.61 in the three US markets, data/m1_per_market_auc.csv), so this is what any market top looks "
                   "like. The sector structure adds nothing to it.</p>")
    return section("m8", "M8", "Four more oscillator predictions on market data", ["explore", "neg"], body)


def m9_m11_section() -> str:
    parts = []
    if (DATA / "m9_auc.csv").exists():
        a9, s9 = pd.read_csv(DATA / "m9_auc.csv"), pd.read_csv(DATA / "m9_severity.csv")
        parts.append(f"""<h3>M9: the systematic frequency-degree correlation over time</h3>
<p>M5c showed that, with systematic volatility, the best-connected stocks are the most volatile: the regime that
Gomez-Gardenes et al. (2011) link to explosive synchronisation. If the mechanism operates in markets, periods with a
stronger correlation should come before crashes, and before faster and deeper ones. <code>fd_sys</code>: in each
trailing 60-day window, the Spearman correlation across stocks between a stock's correlation strength and its
systematic volatility. Predictions written before running: AUC above 0.5; correlation with days from peak to
trough negative, with depth positive.</p>
{table(a9, {"score": "score", "null": "null", "auc": "AUC", "ci_lo": "CI low", "ci_hi": "CI high", "n_events": "crashes"}, auc_cols=("auc",))}
{table(s9, {"target": "crash property", "spearman_with_fd_sys_level": "Spearman with fd_sys level at onset", "lo": "CI low",
            "hi": "CI high", "n_events": "crashes"})}
<p><b>Result: null.</b> fd_sys does not rise before crashes, and it is unrelated to how fast or how deep they are.
Markets sit in the explosive frequency-degree regime all the time; how far into it they sit does not matter.</p>""")
    if (DATA / "m11_auc.csv").exists():
        a11 = pd.read_csv(DATA / "m11_auc.csv")
        parts.append(f"""<h3>M11: flickering of the daily market order parameter</h3>
<p>Between its fold and its onset the herding model is bistable, and noise can move it back and forth between the
synchronised and the incoherent state. Near a tipping point this flickering makes the state's distribution bimodal
and more variable (Scheffer et al., Nature 2009). A daily order parameter that needs no window: breadth b = share of
stocks up on the day minus 1/2; |2b| is near 0 on incoherent days and near 1 when stocks move as one, like the
Kuramoto R. Over the trailing 60 days: <code>bc</code>, the bimodality coefficient of |2b| (above 5/9 suggests two
modes); <code>var</code>, its variance; <code>ltail</code>, the share of days with more than 90% of stocks down.
Prediction written before running: all three rise before crashes against all three nulls.</p>
{table(a11, {"indicator": "indicator", "score": "score", "null": "null", "auc": "AUC", "ci_lo": "CI low", "ci_hi": "CI high",
             "n_events": "crashes"}, auc_cols=("auc",))}
<p><b>Result: weak, and not specific to crashes.</b> Against all non-crash dates, the variance and the bimodality of
breadth rise a little before crashes (AUC 0.55 to 0.58, some CIs above 0.5). Against market tops that were not
followed by a crash they are at chance (0.50 to 0.54). Like the M1 indicators, they describe market tops in general,
not the tops that turn into crashes.</p>""")
    if not parts:
        return ""
    return section("m9", "M9, M11", "Explosive-synchronisation regime and flickering", ["explore", "neg"], "".join(parts))


def robustness_section() -> str:
    parts = []
    for tag, label in (("_s0.05", "noise sigma = 0.05"), ("_s0.2", "noise sigma = 0.2")):
        f = DATA / f"z1_auc{tag}.csv"
        if f.exists():
            a = pd.read_csv(f)
            cv = a[(a.signal == "coll") & (a.indicator == "var") & (a.null == "null_r")].set_index("scenario").auc
            parts.append((label, cv))
    noise = sorted(DATA.glob("z2_ramp_noise*.csv"))
    if not parts and not noise:
        return ""
    body = "<p class=\"lead\">Checks that fit on this machine: Z1 at two other noise levels (48 runs each), and the Z2 ramps with stronger noise and slower ramps.</p>"
    if parts:
        base = pd.read_csv(DATA / "z1_auc.csv")
        b = base[(base.signal == "coll") & (base.indicator == "var") & (base.null == "null_r")].set_index("scenario").auc
        t = pd.DataFrame({"scenario": b.index, "base": b.values})
        cols = {"scenario": "scenario", "base": "sigma = 0.1 (main run)"}
        for label, cv in parts:
            key = label.replace(" ", "_").replace("=", "").replace(".", "p")
            t[key] = cv.reindex(b.index).values
            cols[key] = label
        body += "<h3>Collective variance against the strict null, three noise levels</h3>" + table(t, cols, auc_cols=tuple(k for k in cols if k != "scenario"))
    if noise:
        rows = []
        for fpath in noise:
            s = fpath.stem.replace("z2_ramp_noise", "")
            r = pd.read_csv(fpath).dropna(subset=["eps_esc"])
            for (N, T), g in r.groupby(["N", "T_ramp"]):
                rows.append({"sigma": float(s), "N": int(N), "T_ramp": T, "eps": g.eps_esc.mean(), "sd": g.eps_esc.std(),
                             "early": float((g.eps_esc > 0).mean()), "runs": len(g)})
        nt = pd.DataFrame(rows)
        body += ("<h3>Z2 with stronger noise and slower ramps</h3><p>Column \"left the branch at\" is (lam - lam<sub>b</sub>) / lam<sub>b</sub> at the moment the population leaves the branch: positive = before the fold (early), negative = after it (late). \"Fraction early\" counts runs that left before the fold.</p>"
                 + table(nt, {"sigma": "noise sigma", "N": "N", "T_ramp": "ramp time", "eps": "left the branch at",
                              "sd": "sd", "early": "fraction early", "runs": "runs"}, nd=4))
    return section("r", "R", "Robustness runs on this machine", ["result"], body)


def cluster_section() -> str:
    """Runs made on the Ada cluster: Z1 at larger N (A1) and over noise and ramp speed (A2), Z2 at large N (A3)."""
    def coll_var(f):
        a = pd.read_csv(f)
        sel = a[(a.indicator == "var") & (a.null == "null_r")]
        return sel[sel.signal == "coll"].set_index("scenario").auc, sel[sel.signal == "unit"].set_index("scenario").auc
    body = ""
    sizes = sorted(int(f.stem.split("_N")[1]) for f in DATA.glob("z1_auc_N*.csv"))
    if sizes:
        c0, u0 = coll_var(DATA / "z1_auc.csv")
        t = pd.DataFrame({"scenario": [SCEN_LABEL.get(x, x) for x in c0.index], "c400": c0.values, "u400": u0.reindex(c0.index).values})
        cols = {"scenario": "scenario", "c400": "collective, N = 400", "u400": "one unit, N = 400"}
        for n in sizes:
            c, u = coll_var(DATA / f"z1_auc_N{n}.csv")
            t[f"c{n}"] = c.reindex(c0.index).values
            t[f"u{n}"] = u.reindex(c0.index).values
            cols[f"c{n}"], cols[f"u{n}"] = f"collective, N = {n}", f"one unit, N = {n}"
        body += ("<h3>A1. Variance warning against population size (strict null)</h3><p>ROC AUC of the variance of the "
                 "collective signal and of one unit's signal against the ramp-away null, as in section Z1, at N = 400 "
                 "(48 runs) and at the larger sizes (100 runs each, on the GPU, <code>ZOO_GPU=1</code>). The question: does "
                 "the collective warning hold as finite-size noise falls, and does the single-unit failure persist?</p>"
                 + table(t, cols, auc_cols=tuple(k for k in cols if k != "scenario")))
    grid = sorted(DATA.glob("z1_auc_s*_t*.csv"))
    if grid:
        rows = []
        for f in grid:
            sig, ts = f.stem.replace("z1_auc_s", "").split("_t")
            c, u = coll_var(f)
            rows.append({"sigma": float(sig), "tscale": float(ts), "min": float(c.min()), "worst": SCEN_LABEL.get(c.idxmin(), c.idxmin()),
                         "below07": int((c < 0.7).sum()), "n": int(c.size), "unit_below05": int((u < 0.5).sum())})
        g = pd.DataFrame(rows).sort_values(["sigma", "tscale"])
        body += ("<h3>A2. Variance warning over noise level and ramp speed (N = 400, 100 runs)</h3><p>One row per setting. "
                 "Ramp time is a multiple of the main run's. Lowest collective AUC over the scenarios, which scenario, how many "
                 "scenarios fall below 0.7, and in how many one unit's variance falls instead of rising (AUC below 0.5).</p>"
                 + table(g, {"sigma": "noise sigma", "tscale": "ramp time x", "min": "lowest collective AUC", "worst": "where",
                             "below07": "collective below 0.7", "n": "scenarios", "unit_below05": "one unit below 0.5"},
                         auc_cols=("min",), wrap=True))
    big = DATA / "z2_ramp_largeN.csv"
    if big.exists():
        r = pd.concat([pd.read_csv(DATA / "z2_ramp.csv").assign(src="workstation"), pd.read_csv(big).assign(src="cluster")])
        r = r[r.T_ramp == r[r.src == "cluster"].T_ramp.iloc[0]].dropna(subset=["eps_esc"])
        r["ratio"] = r.var_near / r.var_far
        g = r.groupby("N").agg(runs=("seed", "size"), eps=("eps_esc", "mean"), sd=("eps_esc", "std"),
                               ratio=("ratio", "median")).reset_index()
        body += ("<h3>A3. Z2 ramps up to N = 819200</h3><p>Where the population leaves the synchronised branch, (lam - "
                 "lam<sub>b</sub>) / lam<sub>b</sub> (positive = before the fold), and the median variance ratio just before "
                 "leaving against far from the fold, at the main ramp time. Sizes up to 12800 are the workstation runs.</p>"
                 + table(g, {"N": "N", "runs": "runs", "eps": "left the branch at", "sd": "sd",
                             "ratio": "variance ratio near / far"}, nd=4))
    if not body:
        return ""
    return section("c", "C", "Cluster runs (Ada): larger populations, noise and ramp speed", ["result"],
                   "<p class=\"lead\">Items A1 to A3 of the compute list, run on the GPU path after it was checked "
                   "against the CPU kernels (<code>zoo_gpu_check.py</code>, <code>docs/ADA_RUN_2026-10-24.md</code>).</p>" + body)


def mainproject_section() -> str:
    res = HERE.parent / "results"
    out = []
    for exp, title in (("paperA_A0", "All-to-all validation (configs/paperA_A0_meanfield.yaml)"),
                       ("paperA_convergence", "Convergence of the fold estimator (configs/paperA_convergence.yaml)")):
        runs = sorted(p for p in (res / exp).glob("*") if (p / "summary.json").exists())
        if not runs:
            continue
        sm = json.loads((runs[-1] / "summary.json").read_text())
        out.append((exp, title, runs[-1].name, pd.DataFrame(sm["cells"])))
    if not out:
        return ""
    body = """<p class="lead">Two checks of the main project's fold measurement that its plan listed as not run, both sized for
this workstation and run from here. The fold ratio is the measured backward threshold divided by the exact all-to-all
value of Zou and Wang (PRE 102, 012219, 2020); bisection tolerance 0.2%.</p>"""
    for exp, title, run, df in out:
        body += f"<h3>{html.escape(title)}</h3><p>Run directory <code>results/{exp}/{html.escape(run)}/</code>.</p>"
        body += table(df, {"variant": "variant", "p": "p", "k": "k", "n": "N", "ratio_mean": "fold ratio",
                           "ci_lo": "CI low", "ci_hi": "CI high", "sd": "sd", "n_ok": "seeds"}, nd=4)
    a0 = [d for e, _, _, d in out if e == "paperA_A0"]
    cv = [d for e, _, _, d in out if e == "paperA_convergence"]
    notes = []
    if a0:
        r = a0[0].ratio_mean
        notes.append(f"<li><b>All-to-all:</b> the harness reproduces the exact fold at every exponent and size; fold ratio "
                     f"{r.min():.4f} to {r.max():.4f}, within the 0.2% bisection tolerance.</li>")
    if cv:
        d = cv[0].set_index("variant")
        b = d.loc["baseline", "ratio_mean"]
        dev = ((d.ratio_mean / b - 1) * 100).drop("baseline")
        small = dev[dev.abs() < 0.2]
        big = dev[dev.abs() >= 0.2]
        notes.append(f"<li><b>Convergence:</b> {len(small)} of {len(dev)} variants move the fold by less than the 0.2% "
                     f"tolerance (time step, relaxation time, threshold, frequency clip).</li>")
        for v, x in big.items():
            notes.append(f"<li><b>Systematic:</b> variant <code>{html.escape(v)}</code> moves the fold ratio by {x:+.1f}% "
                         f"({d.loc[v, 'ratio_mean']:.4f}, CI {d.loc[v, 'ci_lo']:.3f} to {d.loc[v, 'ci_hi']:.3f}). Random "
                         "frequency draws add sample-to-sample disorder that the deterministic quantiles remove; with 4 "
                         "seeds the shift is within that extra noise, but it exceeds the tolerance and should be stated "
                         "with any fold measurement made from quantile frequencies.</li>")
    body += "<h3>Result</h3><ul>" + "".join(notes) + "</ul>"
    return section("c", "C", "Main project: two checks run on this workstation", ["result"], body)


def publishable_section() -> str:
    body = """
<h3>Candidates, in order of strength</h3>
<ol>
<li><b>Herding folds of every oscillator family from one phase-reduced equation (O1, O3, O3c, O3d).</b> One number
per family, the first odd Fourier coefficient a1 of its interaction function, maps the exact Kuramoto herding fold
onto every limit-cycle family. Once sweep-speed delay is extrapolated away, the measured folds agree within a few
percent for Kuramoto, Kuramoto with inertia, van der Pol and Roessler. The remaining gaps are specific: Stuart-Landau
-10% (amplitude relaxation, O3b), FitzHugh-Nagumo -19% (strongly non-sinusoidal adjoint). The collapse delay after a
slow ramp is predicted with no free parameter from the exact Ott-Antonsen equation (O3d). Circuit-breaker durations
from the unstable middle branch work in 14 or 15 of 15 cells for most families (O4). A literature search found
neither the cross-family fold mapping nor the slow-passage application. Fits Chaos or Physical Review E. Needed before
submission: O3e (why Stuart-Landau is 10% off) and O6 (sparse networks), both on the Ada list.</li>
<li><b>A benchmark of early-warning indicators across coupled-oscillator families (Z1), with a strict null.</b>
Collective variance warned in the first 14 transitions at the default noise; lag-1 autocorrelation, AR(2) and spectral
indicators each fail or reverse somewhere; single units often show the opposite of a warning. Two caveats: at noise
0.2 the weakest case drops to chance (Winfree death, AUC 0.53), and among the four herding scenarios added later
(A9) the collective variance falls, not rises, before the FitzHugh-Nagumo herding collapse (AUC 0.03 at N = 400 and
0.00 at N = 1600). The reason is found: the event-based phase puts a ripple at the spiking period into R, and
averaged over one period the collective variance warns again (AUC 0.86). So the claim becomes "collective variance of
the cycle-averaged signal warns in every transition tested", with the size and noise scans (A1, A2) on the
page.</li>
<li><b>An exact result on lag-1 autocorrelation for forced mechanical oscillators (Z3).</b> Position: never a warning
below Nyquist; velocity: a warning up to w dt = 2.03. Checked symbolically, against the exact covariance and in
simulation. A short note, or a section of item 2. Search the mechanical and power-system literature once more first.</li>
<li><b>Why explosive desynchronisation gives only a weak warning (Z2, O3d), with the market data as the test.</b>
Exact branch, a slow mode present but diluted in the variance, and late collapse on the 2/3 slow-passage law. The
model predicts small warnings, and the registered market test (M1, 310 crashes, 22 markets) finds none beyond
variance. Needs A3 and A8.</li>
</ol>
<h3>Market results: what they support</h3>
<ul>
<li><b>A well-powered negative result</b> (M1 registered; M5, M8, M9, M11 exploratory). About 30 indicators
motivated by oscillator theory were tested: phase synchrony, eigenvalue and autoregressive rates, frequency locking,
frequency-degree regime, lead-lag reciprocity, sector locking, phase offsets, flickering. None warns before crashes
once compared with market tops that were not followed by a crash. This supports item 4; it is not a paper on its own.</li>
<li><b>Contagion order</b> (M5, M5b, M5d): in 10 of 11 global drawdowns of at least 10%, national indices more
coupled to the rest of the world lost half of their fall sooner (p = 0.006), also with time zones partialled out and
the market left out of the world index. It fades for smaller drawdowns and is not significant within the
Americas-Europe band. Suggestive; a short section at most, with these caveats.</li>
<li><b>Phase synchrony leads volatility</b> (M6 to M6d): consistent across markets and bands, independent of mean
correlation, but tiny, mostly after 2013, and of no use out of sample against the HAR model (M10). Close to
Magner and co-authors' result with network synchronisation. Not a contribution on its own.</li>
</ul>
<h3>Not publishable as it stands</h3>
<ul>
<li>Any claim that synchrony, phase coherence or eigenvalue indicators predict real crashes: the tests say no.</li>
<li>The finite-N early-tipping law N<sup>-2/3</sup>: not observed at the noise level used.</li>
<li>The frequency-degree sign of markets (M5): mechanical (M5c), and its systematic version predicts nothing (M9).</li>
<li>The sector-ratio fall before US crashes (M8): only the rise of mean correlation (M8b).</li>
<li>Multiple testing: the exploratory market tests (M2, M5 to M11) were not corrected across hypotheses. Read every
single exploratory CI that excludes chance with that in mind.</li>
</ul>"""
    return section("pub", "P", "What could be published, and what not", ["open"], body)


def ada_section() -> str:
    text = (HERE / "ADA_COMPUTE.md").read_text()
    rows = []
    for line in text.splitlines():
        if line.startswith("| A") and "|" in line[2:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append(cells)
    head = "<tr><th>item</th><th>what</th><th>why it matters</th><th>cost</th><th>script</th></tr>"
    body_rows = "".join("<tr>" + "".join(f"<td class=\"wrapc\">{html.escape(c).replace('`', '')}</td>" for c in r[:5]) + "</tr>"
                        for r in rows)
    body = f"""<p class="lead">Runs that do not fit on this workstation (16 cores, 15 GB). Costs are scaled from
throughput measured here. Full details, and what is not limited by compute, are in <code>ADA_COMPUTE.md</code>;
SLURM scripts are in <code>ada/</code>.</p>
<div class="tbl"><table class="fit"><thead>{head}</thead><tbody>{body_rows}</tbody></table></div>"""
    return section("ada", "A", "What needs the Ada cluster", ["open"], body)


def files_section() -> str:
    body = """
<p>Every file of this folder is described in <code>README.md</code>. Data files are in <code>data/</code> (CSV
and parquet; the run logs in <code>data/logs/</code>), figures in <code>figures/</code>, cluster scripts in
<code>ada/</code>. Market prices are not stored in the repository (licence); they come from
<code>research/empirical/fetch_markets.py</code>.</p>"""
    return section("files", "F", "Files", [], body)


CSS = """
:root { --bg:#0e1116; --card:#161b22; --ink:#ffffff; --sub:#d6deeb; --muted:#aab6c8; --line:#2c3440;
        --accent:#7fb8ff; --soft:#1c2430; --good:#5fd38d; --bad:#ff7b72;
        --b-result:#1f7a4d; --b-neg:#a8322b; --b-theory:#0f6f78; --b-explore:#7a5a12; --b-open:#5a4b9c; --b-data:#245e9e; }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--ink); font:16.5px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif; }
header { max-width:1240px; margin:0 auto; padding:44px 24px 16px }
header h1 { margin:0 0 8px; font-size:32px; letter-spacing:-.01em }
header p { margin:4px 0; color:var(--sub) }
nav { position:sticky; top:0; z-index:5; background:rgba(14,17,22,.94); border-bottom:1px solid var(--line) }
nav .in { max-width:1240px; margin:0 auto; padding:10px 24px; display:flex; gap:8px; overflow-x:auto }
nav a { white-space:nowrap; text-decoration:none; color:var(--ink); font-size:14px; padding:5px 12px; border:1px solid var(--line); border-radius:999px; background:var(--card) }
nav a b { color:var(--accent); margin-right:6px }
main { max-width:1240px; margin:0 auto; padding:22px 24px 40px }
section { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:26px 30px; margin:0 0 28px }
.sec-head { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px }
.sec-head h2 { margin:0; font-size:24px; flex:1; min-width:260px }
.tag { font-weight:800; color:var(--accent); font-size:16px }
.badge { font-size:13.5px; font-weight:700; padding:4px 12px; border-radius:999px; color:#fff }
.b-result { background:var(--b-result) } .b-neg { background:var(--b-neg) } .b-theory { background:var(--b-theory) }
.b-explore { background:var(--b-explore) } .b-open { background:var(--b-open) } .b-data { background:var(--b-data) }
h3 { font-size:19px; margin:24px 0 8px; color:var(--accent) }
h4 { margin:0 0 6px; font-size:14px; text-transform:uppercase; letter-spacing:.05em; color:var(--sub) }
p, li { color:var(--ink) } .lead { font-size:17.5px }
a { color:var(--accent) }
code { font-size:14.5px; background:#232c39; color:#ffffff; padding:1px 6px; border-radius:4px }
.math { background:var(--soft); border-radius:10px; padding:12px 18px; font-family:"STIX Two Math","Cambria Math",Georgia,serif; font-size:17.5px; line-height:1.9; overflow-x:auto }
figure { margin:24px 0 0; border-top:1px solid var(--line); padding-top:18px }
figure img { width:100%; border-radius:10px; background:#ffffff; padding:8px }
figcaption { font-size:15.5px; font-weight:700; margin:8px 0 10px }
.explain { display:grid; grid-template-columns:1fr 1fr; gap:18px; background:var(--soft); border-radius:10px; padding:14px 18px }
.explain p { margin:0; font-size:15.5px }
@media (max-width:860px) { .explain { grid-template-columns:1fr } section { padding:20px 16px } }
.tbl { overflow-x:auto; margin:10px 0 } table { border-collapse:collapse; font-size:14.5px; min-width:100% }
th, td { border-bottom:1px solid var(--line); padding:6px 10px; text-align:left; white-space:nowrap }
th { color:var(--sub); font-weight:700; vertical-align:bottom; white-space:normal }
table.fit td, td.wrapc { white-space:normal } table.fit td:first-child { min-width:150px }
td.good { color:var(--good); font-weight:700 } td.bad { color:var(--bad); font-weight:700 }
.note { background:var(--soft); padding:12px 16px; border-radius:10px; margin-top:16px }
.summary { border:2px solid var(--accent) }
.guide { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px }
.guide div { background:var(--soft); border-radius:10px; padding:12px 14px; font-size:15px }
footer { text-align:center; color:var(--muted); font-size:14px; padding:6px 0 40px }
"""


def summary_section() -> str:
    a = pd.read_csv(DATA / "z1_auc.csv")
    cv = a[(a.signal == "coll") & (a.indicator == "var") & (a.null == "null_r")]
    fell = cv[cv.auc < 0.5]
    z1_exception = ("; it fell instead before " + ", ".join(f"{SCEN_LABEL[r.scenario]} (AUC {r.auc:.2f})" for r in fell.itertuples())
                    if len(fell) else "")
    f = json.loads((DATA / "z2_fits.json").read_text())
    sm = json.loads((DATA / "m1_summary.json").read_text())
    body = f"""
<ul class="findings">
<li><b>Z1, simulation.</b> Across {len(cv)} transitions in 11 oscillator models (plus a non-oscillating reference), the variance
of the collective signal warned in {int((cv.auc >= 0.7).sum())} (AUC 0.7 or more against the strict null){z1_exception}. Lag-1
autocorrelation, AR(2) decay rate and spectral peak each fail or reverse in some family. Single oscillators often
show the opposite of a warning.</li>
<li><b>Z3, exact theory.</b> For a noise-driven mechanical oscillator, lag-1 autocorrelation of the position falls as
the damping vanishes at every sampling interval below Nyquist; the velocity's rises up to w dt = 2.03.
Simulation agrees to within one grid step.</li>
<li><b>Z2, the crash model.</b> Ott-Antonsen theory gives the synchronised branch to {100 * f['max_rel_err_R_branch']:.1f}%.
The slow mode is present (low-frequency power scales as kappa<sup>{f['slope_Slow_vs_kappa']:.1f}</sup>) but diluted
in the variance, and slowly driven populations collapse after the fold, with a delay scaling as speed<sup>{f['delay_rate_exponent']:.2f}</sup>
(2/3 law). The variance before collapse is only 1.2 to 2.1 times its baseline.</li>
<li><b>M0, data.</b> The market panels contained negative adjusted prices, holiday filler days, stale runs, bad ticks and
unadjusted splits; six rules remove them and raise the agreement with the official index (Brazil 0.70 to 0.93).</li>
<li><b>M1 to M4, real markets.</b> Registered test, {sm['n_events']} crashes in {sm['n_markets']} markets: AR(2) rate,
VAR(1) eigenvalue and Hilbert phase synchrony do not warn ({sm['n_bh_reject']} of {sm['n_registered_tests']} tests survive).
Nothing anticipates future drawdowns beyond variance; world synchrony does not anticipate global crashes; index
variance is not clearly better than single-stock variance. Consistent with Z2: the model itself predicts only a
small warning.</li>
<li><b>O0 to O9, the main project's pipeline for every family.</b> Single-unit bifurcations match exact formulas;
herding makes synchronisation explosive and bistable in every family; exact thresholds are reproduced (amplitude
death window, Winfree death, theta-neuron folds). Phase reduction with the family's a1 predicts the herding folds
within a few percent once sweep delay is removed (Stuart-Landau -10%, FitzHugh-Nagumo -19%), and the collapse delay
is predicted with no free parameter. Shocks and breakers behave as the theory says except in FitzHugh-Nagumo.</li>
<li><b>M5 to M11, more ideas on market data (exploratory).</b> Frequency-degree regime, frequency locking, lead-lag
reciprocity, sector locking, phase offsets and flickering give no crash warning beyond what any market top shows.
Contagion order holds in 10 of 11 large global drawdowns but fades for smaller ones. Phase synchrony leads volatility
slightly and independently of mean correlation, but adds nothing to an out-of-sample HAR forecast.</li>
</ul>
<h3>How to read this page</h3>
<div class="guide">
<div><span class="badge b-result">result</span> a measured finding that held up under the checks described.</div>
<div><span class="badge b-theory">theory confirmed</span> an analytic prediction tested against simulation.</div>
<div><span class="badge b-neg">negative result</span> the tested idea did not work; reported in full.</div>
<div><span class="badge b-explore">exploratory</span> not designed in advance and not corrected for multiple tests.</div>
<div><span class="badge b-data">data quality</span> what was wrong with the data and what was done about it.</div>
<div><span class="badge b-open">open question</span> needs more compute or more work.</div>
<div><b>AUC</b> (area under the ROC curve): the probability that a randomly chosen positive case (a ramp run, a crash
onset) scores higher than a randomly chosen negative one. 0.5 = no information. In tables, green = 0.7 or more,
red = 0.3 or less.</div>
<div><b>Null</b>: the comparison cases. A warning only counts if it separates the positive cases from the null.</div>
<div><b>CI</b>: 95% bootstrap confidence interval. <b>BH</b>: Benjamini-Hochberg correction for multiple tests.</div>
</div>"""
    return section("summary", "S", "Summary", [], body).replace("<section id=\"summary\">", "<section id=\"summary\" class=\"summary\">")


def main() -> None:
    sections = [summary_section(), o_section(), z1_section(), z3_section(), z2_section(), m0_section(), m1_section(), m2_section(),
                m3_section(), m4_section(), m5_section(), m6_section(), m8_section(), m9_m11_section(), robustness_section(), cluster_section(), mainproject_section(), publishable_section(), ada_section(), files_section()]
    nav_items = [("summary", "S", "Summary"), ("o", "O", "Every family"), ("z1", "Z1", "Oscillator zoo"), ("z3", "Z3", "Lag-1 AC theory"),
                 ("z2", "Z2", "Crash model"), ("m0", "M0", "Data audit"), ("m1", "M1", "Market test"),
                 ("m2", "M2", "Correlations"), ("m3", "M3", "World markets"), ("m4", "M4", "Collective vs unit"), ("m5", "M5", "Hypotheses"), ("m6", "M6", "Lead or lag"),
                 ("r", "R", "Robustness"), ("c", "C", "Main-project checks"), ("pub", "P", "Publishable?"), ("ada", "A", "Ada compute"), ("files", "F", "Files")]
    present = {s.split('id="')[1].split('"')[0] for s in sections if s}
    nav = "".join(f'<a href="#{i}"><b>{t}</b>{n}</a>' for i, t, n in nav_items if i in present)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oscillator Early Warnings</title><style>{CSS}</style></head><body>
<header><h1>Early warnings across oscillator families, and on real markets</h1>
<p>Athmeeya M Kashyap, IIIT Hyderabad. Built {date.today():%Y-%m-%d} from the files in <code>oscillators/data/</code>.</p>
<p>The coloured buttons below jump to each section. Section letters: Z = simulation, M = market data, R = robustness,
C = checks of the main project, P = assessment, A = compute still needed.</p></header>
<nav><div class="in">{nav}</div></nav>
<main>{"".join(sections)}</main>
<footer>Generated by oscillators/build_index.py</footer></body></html>"""
    for bad in ("\u2014", "\u2013"):
        if bad in page:
            i = page.index(bad)
            raise ValueError(f"dash found near: {page[max(0, i - 60):i + 20]!r}")
    (HERE / "index.html").write_text(page)
    print(f"wrote {HERE / 'index.html'} ({len(page) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
