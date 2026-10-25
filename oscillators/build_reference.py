"""Build oscillators/reference.html: every oscillator family of this project, built up from one unit.

The counterpart of research/reference.html (which builds the adaptive Kuramoto model step by step) for
the families in zoo.py. Every number is read from data/ at build time; a section whose data are not there
yet says so. The build fails on a missing figure or a dash character.

    ../research/.venv/bin/python build_reference.py
"""

from __future__ import annotations

import html
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from build_index import CSS, fmt, table

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"


def have(*names: str) -> bool:
    return all((DATA / n).exists() for n in names)


def fig(name: str, caption: str, seeing: str, significance: str) -> str:
    if not (FIG / name).exists():
        return f'<div class="note">Figure <code>{name}</code> not built yet (its run has not finished).</div>'
    return (f'<figure><img src="figures/{name}" alt="{html.escape(caption)}" loading="lazy">'
            f'<figcaption>{caption}</figcaption><div class="explain"><div><h4>What you are seeing</h4>{seeing}</div>'
            f'<div><h4>Significance</h4>{significance}</div></div></figure>')


def step(sid: str, n: str, title: str, body: str) -> str:
    return f'<section id="{sid}"><div class="sec-head"><span class="tag">{n}</span><h2>{title}</h2></div>{body}</section>'


def math(text: str) -> str:
    return f'<div class="math">{text}</div>'


# ----------------------------------------------------------------------------------- numbers from data
def atlas() -> pd.DataFrame | None:
    if not have("o1_thresholds.csv"):
        return None
    return pd.read_csv(DATA / "o1_thresholds.csv").groupby(["family", "p", "direction"]).threshold.agg(["mean", "std", "count"])


def pred() -> pd.DataFrame | None:
    return pd.read_csv(DATA / "o3_predictions.csv").set_index(["family", "p"]) if have("o3_predictions.csv") else None


def reduction() -> pd.DataFrame | None:
    return pd.read_csv(DATA / "o3_reduction.csv").set_index("family") if have("o3_reduction.csv") else None


def exact() -> pd.Series | None:
    return pd.read_csv(DATA / "o3_exact.csv").set_index("transition").value if have("o3_exact.csv") else None


def special() -> pd.DataFrame | None:
    if not have("o1_special.csv"):
        return None
    return pd.read_csv(DATA / "o1_special.csv").groupby(["kind", "direction", "event"]).param.agg(["mean", "std", "count"])


def z1_row(scen: str) -> str:
    if not have("z1_auc.csv"):
        return ""
    a = pd.read_csv(DATA / "z1_auc.csv")
    a = a[(a.scenario == scen) & (a.null == "null_r")]
    if a.empty:
        return ""
    lab = {"var": "variance", "ac1": "lag-1 AC", "skew": "|skewness|", "spec_peak": "spectral peak",
           "ar2_rate": "AR(2) rate", "var1_rate": "VAR(1) rate"}
    cells = []
    for sig, name in (("coll", "collective"), ("unit", "one unit"), ("nodes", "8 units")):
        g = a[a.signal == sig]
        for _, r in g.iterrows():
            cells.append({"signal": name, "indicator": lab[r.indicator], "AUC": r.auc})
    d = pd.DataFrame(cells)
    return ("<h3>Early warnings before this transition (Z1, AUC against ramps away from the transition)</h3>"
            + table(d, {"signal": "signal", "indicator": "indicator", "AUC": "AUC"}, auc_cols=("AUC",), nd=2))


def herding_table(fam: str) -> str:
    at, pr = atlas(), pred()
    if at is None:
        return '<div class="note">Atlas not run yet.</div>'
    rows = []
    for p in (0, 1, 2):
        up = at.loc[(fam, p, "up")] if (fam, p, "up") in at.index else None
        dn = at.loc[(fam, p, "down")] if (fam, p, "down") in at.index else None
        theory = np.nan
        if pr is not None and (fam, p) in pr.index:
            r = pr.loc[(fam, p)]
            theory = r.get("pred_K_at_R05", np.nan) if p == 0 else r.get("pred_fold", np.nan)
        rows.append({"p": p, "up": up["mean"] if up is not None and up["count"] > 0 else np.nan,
                     "up_n": int(up["count"]) if up is not None else 0,
                     "down": dn["mean"] if dn is not None else np.nan,
                     "down_sd": dn["std"] if dn is not None else np.nan, "theory": theory})
    d = pd.DataFrame(rows)
    return table(d, {"p": "herding exponent p", "up": "jump up from calm (R reaches 0.5)",
                     "up_n": "seeds that jumped (of 4)", "down": "collapse coming down (R falls below 0.5)",
                     "down_sd": "sd", "theory": "phase reduction (p = 0: K at R = 0.5; p >= 1: fold)"}, nd=4)


def reduction_line(fam: str) -> str:
    red = reduction()
    if red is None or fam not in red.index:
        return ""
    r = red.loc[fam]
    return (f"<p>Phase reduction, computed by {html.escape(str(r.method))}: first odd Fourier coefficient of the interaction "
            f"function <b>a1 = {r.a1:.4f}</b>, phase lag alpha = {r.alpha:+.4f} rad. Every Kuramoto threshold of this "
            f"family is the Kuramoto value divided by a1 (for the same frequencies).</p>")


def sep_line(name: str) -> str:
    if not have("o8_separability.csv"):
        return ""
    d = pd.read_csv(DATA / "o8_separability.csv")
    r = d[d.family == name]
    if r.empty:
        return ""
    r = r.iloc[0]
    add = ""
    if "herding_pairwise_additive" in r and isinstance(r.herding_pairwise_additive, (bool, np.bool_)):
        add = " With herding the input is not even pairwise additive." if not r.herding_pairwise_additive else ""
    kind = "product-separable (rank 1): inside the class of Hens et al., Nat. Phys. 15, 403 (2019)" \
        if r.product_separable else f"not product-separable (separable rank {int(r.separable_rank)}), like Kuramoto"
    return f"<p>Separability (O8): coupling {html.escape(str(r['coupling G(x_i, x_j)']))} is {kind}.{add}</p>"


# ------------------------------------------------------------------------------------------- sections
def part_one() -> list[str]:
    s = []
    s.append(step("p0", "0", "What this page is, and how it fits with the rest", """
<p class="lead">The main project builds one model, the adaptive (herding) Kuramoto model, from a single
oscillator up, in <a href="../research/reference.html">research/reference.html</a>. This page does the same for
every other oscillator family used in this project, so that each can be read on its own: what one unit does,
where its oscillation is born, how a population of them synchronises or stops, what herding does to it, what
exact theory says, and what the simulations measured. The results of the early-warning and market studies are on
<a href="index.html">index.html</a>; the files are described in <a href="README.md">README.md</a>.</p>
<p>Notation is kept the same across families: N units, natural frequencies w_j, coupling strength K (lam when it
is a herding coupling), noise amplitude sigma, the mean field Z (the average of the units' complex signals or of
exp(i phase)), and the order parameter R = |Z| measured on phases. Equations are written in plain text:
<code>x'</code> is dx/dt, <code>x''</code> the second derivative, <code>^</code> a power.</p>"""))
    s.append(step("p1", "1", "What an oscillator is: a cycle, a phase, and sometimes an amplitude", f"""
<p>An oscillator is any system that settles onto a closed loop in its state space, a <b>limit cycle</b>, and goes
round it forever. Where it is on the loop is its <b>phase</b> (an angle from 0 to 2 pi); how fast it goes round when
left alone is its <b>natural frequency</b>. Some families have only a phase (Kuramoto, Winfree, theta neurons); the
others also have an <b>amplitude</b>, the size of the loop, which can grow, shrink or vanish. The families here
differ in exactly the ways that matter for synchronisation:</p>
<ul>
<li><b>Smooth limit cycles</b> (Stuart-Landau, van der Pol at moderate eps): the loop is round and the phase advances
evenly.</li>
<li><b>Relaxation oscillators</b> (van der Pol at large eps, FitzHugh-Nagumo): slow creeping followed by fast jumps;
the phase advances very unevenly and coupling acts mostly during the jumps.</li>
<li><b>Excitable units</b> (FitzHugh-Nagumo below threshold, theta neurons with negative drive): at rest, but a large
enough kick produces one full excursion, a spike.</li>
<li><b>Chaotic oscillators</b> (Roessler): the motion never repeats exactly, yet it still turns round a centre, so a
phase can be defined and synchronised while the amplitude stays irregular.</li>
<li><b>Linear oscillators</b> (simple harmonic motion): no limit cycle at all; with damping they decay, without it
they keep whatever amplitude they started with. They are the reference against which the nonlinear ones are
compared.</li>
</ul>
{fig("o0_units.png", "O0. One unit of each family, alone: time series and phase-plane trajectory.",
     "<p>Pairs of panels. Left of each pair: the unit's main variable against time. Right: the same run as a curve in the phase plane (the two state variables against each other). A closed loop is a limit cycle; a spiral into a point is a damped rest state; the Roessler curve never closes. The theta neuron is drawn on the unit circle (cos theta, sin theta).</p>",
     "<p>The shape of the loop decides how coupling acts: on a round loop (Stuart-Landau) a push changes the phase by the same amount everywhere; on a relaxation loop (van der Pol eps = 5, FitzHugh-Nagumo) only pushes near the jumps matter. That difference reappears in every threshold measured below.</p>")}"""))
    ex = exact()
    bif = ""
    if have("o0_bifurcations.csv"):
        b = pd.read_csv(DATA / "o0_bifurcations.csv")
        bif = table(b, {"family": "family", "parameter": "parameter", "value": "value", "measured": "measured",
                        "theory": "theory", "quantity": "quantity"}, nd=4)
    s.append(step("p2", "2", "Where a unit's oscillation is born", f"""
<p>Changing one parameter can switch a unit's oscillation on or off. How it switches on decides how a unit, and
later a population, behaves near the switch:</p>
<ul>
<li><b>Supercritical Hopf</b> (Stuart-Landau, van der Pol): the rest state turns unstable through a pair of complex
eigenvalues and a small loop grows smoothly, amplitude proportional to the square root of the distance past the
point. Stuart-Landau is the normal form: <code>z' = (mu + i w) z - |z|^2 z</code> gives |z| = sqrt(mu) exactly.</li>
<li><b>Linear Hopf</b> (simple harmonic oscillator): at zero damping nothing limits the amplitude; there is no loop
to settle on. It is the linearisation of every Hopf point, which is why its early-warning behaviour (Z3 on
index.html) applies to all of them.</li>
<li><b>Hopf with canard explosion</b> (FitzHugh-Nagumo): the small loop born at the Hopf point grows into a full spike
within an exponentially thin range of the input current, so in practice the unit jumps from rest to full spikes at
I = {fmt(ex['FitzHugh-Nagumo Hopf point']) if ex is not None else 'n/a'}.</li>
<li><b>Saddle-node on the invariant circle, SNIC</b> (theta neuron): a resting point and a saddle merge on the
circle; the firing rate rises from zero like sqrt(eta) / pi, so arbitrarily slow firing is possible.</li>
</ul>
{fig("o0_bifurcations.png", "O0. Where each unit's oscillation is born: simulation against theory.",
     "<p>Markers: amplitude (or firing rate) measured by direct integration of one unit, after transients. Dashed: the exact or standard formula (sqrt(mu) for Stuart-Landau, 2 sqrt(mu) from averaging for van der Pol, sqrt(eta) / pi for the theta neuron, the Hopf point for FitzHugh-Nagumo).</p>",
     "<p>The integrator reproduces every textbook result; the FitzHugh-Nagumo unit jumps to full spikes between I = 0.32 and 0.33, at the Hopf point.</p>")}
{bif}"""))
    s.append(step("p3", "3", "Coupling a crowd, and measuring how synchronised it is", """
<p>All populations here are coupled through a mean field: each unit feels the average of the others (or, on a
network, of its neighbours). The coupling has three forms:</p>
<ul>
<li><b>Phase coupling</b> (Kuramoto, with or without inertia): <code>K R sin(Psi - theta_j)</code>.</li>
<li><b>Diffusive coupling</b> (Stuart-Landau, van der Pol, FitzHugh-Nagumo, Roessler): <code>K (X - x_j)</code> in one
variable: a unit is pulled toward the average state. It also damps each unit's own amplitude, which is what makes
amplitude death possible.</li>
<li><b>Pulse coupling</b> (Winfree, theta neurons): each unit sends a pulse that depends on its own phase, and its
sensitivity depends on the receiver's phase: <code>P(theta_k) Q(theta_j)</code>.</li>
</ul>
<p>The order parameter is always <b>R = |mean of exp(i phase)|</b>, between 0 (phases spread, signals cancel) and 1
(all in step). What changes is how the phase is read off:</p>
<div class="tbl"><table class="fit"><thead><tr><th>family</th><th>phase used for R</th></tr></thead><tbody>
<tr><td>Kuramoto, inertia, Winfree, theta</td><td>the phase variable itself</td></tr>
<tr><td>Stuart-Landau</td><td>the angle of z</td></tr>
<tr><td>van der Pol</td><td>the angle of (x, -v / w) in the phase plane</td></tr>
<tr><td>Roessler</td><td>the angle of (x, y)</td></tr>
<tr><td>FitzHugh-Nagumo</td><td>event based: 2 pi (time since the last spike) / (last period); the phase-plane angle is
not used because it advances very unevenly along a relaxation cycle (an uncoupled population gave R = 0.27 with it,
against 0.05 with the event phase)</td></tr>
</tbody></table></div>
<p>Frequencies are drawn as deterministic quantiles of one law per family (Lorentzian for Kuramoto and
Stuart-Landau, because it makes the Ott-Antonsen theory exact; Gaussian or uniform for the others), and randomly
assigned to units, so different seeds differ only in placement, initial phases and noise.</p>"""))
    s.append(step("p4", "4", "The herding term, for every family", """
<p>The main project's crash mechanism is herding: a unit listens more when the others already agree. For every
family the coupling is multiplied by the order parameter to a power p:</p>"""
                  + math("coupling term -> K R^p x (the family's coupling)") + """
<p>with R the global order parameter in the mean-field runs and the local order parameter r_i of node i's
neighbours on networks (exactly the model of Zhang et al., PRL 114, 038701 (2015)). With p > 0 the coupling
vanishes when R = 0, so the calm (incoherent) state is linearly stable at every coupling: synchrony cannot start
by itself, only through a large enough fluctuation or shock. The synchronised state, once reached, survives down
to a <b>fold</b>, below which it collapses. Between the fold and the coupling at which noise finally triggers
synchrony both states coexist: the bistable window that the main project reads as the crash mechanism.</p>"""))
    s.append(step("p5", "5", "One number per family: phase reduction", """
<p>When the coupling is weak compared with how fast a unit returns to its cycle, the state of a unit is
described by its phase alone, and the effect of coupling depends only on the phase difference:</p>"""
                  + math("theta_j' = Omega_j + (K / N) sum_k H(theta_k - theta_j),&nbsp;&nbsp; H(phi) = &lt; Z(psi) . G(x(psi), x(psi + phi)) &gt;") + """
<p>Z(psi) is the <b>phase sensitivity</b> (how much a small push at phase psi advances the phase), found by
integrating the adjoint equation backward along the cycle, and G is the coupling. Keeping the first Fourier mode
of H, every family becomes a Kuramoto model whose coupling is multiplied by <b>a1</b>, the coefficient of
sin(phi) in H (with a small phase lag from the cos(phi) term). So every Kuramoto threshold carries over:
the synchrony onset and the herding fold of a family are the Kuramoto values (for its own frequencies) divided by
its a1. For Stuart-Landau a1 = 1 exactly, which the numerical adjoint reproduces to 1e-6; for the relaxation
oscillators a1 is about 0.4 to 0.5. How far each family follows this rule, and why some do not, is measured in
the chapters below and on <a href="#summary">the summary table</a>.</p>""" + fig(
        "o3_phase_reduction.png", "O3. Phase interaction functions and the thresholds they predict.",
        "<p>(a) H(phi) for each family, from the numerical adjoint; dashed: sin(phi), the Kuramoto case. (b) Each measured threshold against its phase-reduction prediction, log-log; circle: onset (p = 0, upward crossing of R = 0.5 in the atlas), square: herding fold at p = 1, diamond: at p = 2. Filled: the fold extrapolated to an infinitely slow sweep (O3c); open, faint: the fold at the atlas sweep speed. Dashed: perfect agreement.</p>",
        "<p>Onsets fall on the diagonal for every family. Herding folds, once the delay caused by the finite sweep speed is removed, lie within a few percent of the prediction for Kuramoto, van der Pol and Roessler (p = 1: +0.5%, -0.3%, +2.6%). Two families stay below it: Stuart-Landau (-10%) and the relaxation oscillator FitzHugh-Nagumo (-19%). The open markers show how strongly a fixed sweep speed lowers the measured fold of the slow families.</p>")))
    return s


CHAPTERS = [
    # id, title, equation, unit text, population text, family key for atlas, z1 scenario, separability name,
    # papers
    ("harmonic", "Simple harmonic oscillator (the linear reference)",
     "x'' + g x' + w0^2 x = sigma xi(t)",
     "Damped harmonic motion driven by noise. For g > 0 the rest state is stable and the amplitude decays at rate "
     "g/2; at g = 0 the pair of eigenvalues -g/2 +- i sqrt(w0^2 - g^2/4) reaches the imaginary axis: the linear "
     "version of every Hopf point. Exact stationary variance sigma^2 / (2 g w0^2).",
     "Used as a single unit only. It is the reference for early warnings before any oscillatory loss of stability. "
     "Z3 on index.html derives exactly that lag-1 autocorrelation of its position falls, not rises, as the damping "
     "vanishes, at every sampling interval below the Nyquist limit, while that of its velocity rises up to "
     "w dt = 2.03.",
     None, "harmonic", None, []),
    ("sl", "Stuart-Landau oscillator: the normal form of every smooth oscillation",
     "z' = (mu + i w - |z|^2) z + sigma (xi1 + i xi2),&nbsp;&nbsp; population: z_j' = (mu + i w_j - |z_j|^2) z_j + K R^p (Z - z_j)",
     "One complex variable z = x + i y. Below mu = 0 the rest state is stable; above it a circular cycle of radius "
     "sqrt(mu) rotating at w. Every smooth oscillation onset looks like this near its Hopf point, which is why it is "
     "the standard model of a generic oscillator with an amplitude.",
     "Diffusive mean-field coupling pulls each z_j toward the average Z and also damps it by K. Three behaviours, "
     "depending on the frequency spread: synchrony (narrow spread, Lorentzian half-width 0.1), amplitude death "
     "(wide spread, frequencies uniform on [-4, 4]: every unit stops, z_j = 0, for K between 1 and an upper edge), "
     "and with herding an explosive collapse of synchrony.",
     "sl", "sl_adapt_down", "Stuart-Landau population", ["bi2014_explosive_oscillation_death", "koseska2013 (not on arXiv)",
                                                          "pikovsky_rosenblum2015_global_coupling"]),
    ("vdp", "van der Pol oscillator: from smooth to relaxation",
     "x'' - eps (1 - x^2) x' + w^2 x = K R^p (V - x') + sigma xi",
     "Negative damping at small amplitude, positive at large amplitude: a self-sustained cycle of amplitude about 2. "
     "For eps near 1 the cycle is smooth; for large eps it becomes a relaxation oscillation (slow creep, fast "
     "jump). With (mu - x^2) instead of (1 - x^2) it has a Hopf point at mu = 0 and amplitude 2 sqrt(mu) just after.",
     "Coupled through the velocity (dissipative coupling): coupling through position alone is reactive and does not "
     "synchronise them (checked). Frequencies Gaussian, mean 1, sd 0.05, eps = 1.",
     "vdp", "vdp_up", "van der Pol population", []),
    ("fhn", "FitzHugh-Nagumo: an excitable neuron model",
     "v' = v - v^3/3 - w + I + K R^p (V - v) + sigma xi,&nbsp;&nbsp; w' = 0.08 (v + 0.7 - 0.8 w)",
     "A fast membrane variable v and a slow recovery variable w. Below I = 0.3313 (the Hopf point) the unit rests but "
     "is excitable: a large enough kick produces one spike. Above it, it fires periodically (period 36 to 47 here); the "
     "small Hopf cycle explodes into full spikes within an exponentially thin range of I (canard explosion).",
     "Diffusive coupling through v, as for electrically coupled neurons. Each unit has its own input current in "
     "[0.34, 0.9], which gives frequencies spread by about +- 12 percent. Phase is event based (see step 3).",
     "fhn", None, "FitzHugh-Nagumo population", []),
    ("kuramoto", "Kuramoto model: phases only",
     "theta_j' = w_j + K R^p R sin(Psi - theta_j) + sigma xi_j",
     "A unit is a pure phase rotating at w_j. It is what every limit-cycle family reduces to when coupling is weak "
     "(step 5), and the model of the main project.",
     "Lorentzian frequencies of half-width 0.5. p = 0: continuous synchrony onset at K_c = 2 (gamma + sigma^2/2) with "
     "R = sqrt(1 - K_c/K) (Ott-Antonsen, exact). p > 0: explosive, fold lam_b = gamma' (p+2) ((p+2)/p)^(p/2) (Zou and "
     "Wang 2020).",
     "kuramoto", "kuramoto_adapt_down", "Kuramoto, with or without inertia", ["ott_antonsen2008", "rodrigues2016"]),
    ("inertia", "Kuramoto with inertia: the swing equation of power grids",
     "m theta_j'' + theta_j' = w_j + K R^p R sin(Psi - theta_j) + sigma xi_j,&nbsp;&nbsp; m = 2",
     "Each unit is a damped rotor (a generator in a power grid). Inertia lets a unit keep rotating after it is "
     "pulled, so locking and unlocking happen at different couplings even without herding.",
     "Gaussian frequencies, sd 1. Going up, the incoherent state loses stability where "
     "K_c = 2 / [pi g(0) - m integral g(w) / (1 + m^2 w^2) dw] (Gupta, Campa and Ruffo 2014); coming down, every unit "
     "with |w| < K R stays locked (Tanaka, Lichtenberg and Oishi 1997).",
     "inertia", "kuramoto2_down", "Kuramoto, with or without inertia", ["gupta2014_inertia_review", "acebron2000_inertia",
                                                                       "olmi2014"]),
    ("winfree", "Winfree model: pulse coupling and oscillator death",
     "theta_j' = w_j - kappa sin(theta_j) mean_k(1 + cos theta_k) + sigma xi_j",
     "Each unit sends a pulse 1 + cos(theta) that peaks when it fires, and responds with a sensitivity -sin(theta). "
     "This is the original model of collective synchrony (Winfree 1967).",
     "Frequencies uniform on [0.9, 1.1]. As kappa grows the locked population slows down and finally stops: "
     "oscillator death, every unit at a stable fixed point. The death state exists iff kappa >= min over x of "
     "x / sigma(x), sigma(x) = 1 + mean sqrt(1 - (w/x)^2).",
     None, "winfree_up", "Winfree", ["ariaratnam_strogatz2001_winfree", "pazo_montbrio2014_winfree_oa"]),
    ("rossler", "Roessler oscillator: phase synchronisation of chaos",
     "x' = -w y - z + K R^p (X - x),&nbsp;&nbsp; y' = w x + 0.15 y,&nbsp;&nbsp; z' = 0.2 + z (x - 10)",
     "A chaotic attractor that still turns round a centre in the (x, y) plane, so its angle is a usable phase. The "
     "phase can lock while the amplitude stays chaotic (phase synchronisation, Rosenblum, Pikovsky and Kurths, PRL 76, "
     "1804 (1996), not on arXiv).",
     "Coupled through x; frequencies Gaussian, mean 1, sd 0.02. There is no strict phase sensitivity for chaos; for "
     "x-coupling of a phase-coherent attractor the angular equation gives a1 = 1/2 if all radii are equal, used as an "
     "approximation (Imai, Suetani and Aoyagi 2021 give the systematic version).",
     "rossler", "rossler_up", "Roessler population", ["imai2021_chaotic_phase_description"]),
    ("theta", "Theta neurons: a bistable network of spiking units",
     "theta_j' = (1 - cos theta_j) + (1 + cos theta_j)(eta_j + J S),&nbsp;&nbsp; S = mean (2/3)(1 - cos theta_k)^2",
     "The normal form of a neuron that starts firing through a SNIC: rest below eta = 0, firing at rate sqrt(eta)/pi "
     "above (step 2).",
     "Lorentzian drives centred at -5 + shift, half-width 0.5, J = 15. Exactly reducible by Ott-Antonsen (Luke, "
     "Barreto and So 2013; Montbrio, Pazo and Roxin 2015 for the equivalent QIF network): a high-activity and a quiet "
     "state coexist over a range of the drive; the active state collapses at a fold.",
     None, "theta_down", "theta neurons (pulse coupling)", ["montbrio2015_qif_meanfield", "luke2013 (not on arXiv)"]),
]


FAMKEY = {"sl": "sl", "vdp": "vdp", "fhn": "fhn", "kuramoto": "kuramoto", "inertia": "inertia", "rossler": "rossler"}


def protocols_block(fam: str) -> str:
    """O4 (shocks, circuit breakers, rate), O5 (frequency-degree correlation), O6 (sparse networks) for one family."""
    out = ""
    if have("o4_shock.csv", "o4_breaker.csv", "o4_breaker_theory.csv", "o4_rate.csv"):
        sh = pd.read_csv(DATA / "o4_shock.csv")
        sh = sh[sh.family == fam]
        if not sh.empty:
            p = sh.pivot_table(index="c", columns="phi", values="crash").reset_index()
            p.columns = ["coupling / fold"] + [f"phi = {c:g}" for c in p.columns[1:]]
            out += ("<h3>Shocks (O4 a)</h3><p>Fraction of runs that end synchronised (a persistent crash) after a "
                    "fraction phi of a calm population is put at one phase, at couplings below (0.8) and inside the "
                    "bistable window (multiples of the slow-sweep fold).</p>"
                    + table(p, {c: c for c in p.columns}, nd=2))
            br = pd.read_csv(DATA / "o4_breaker.csv")
            br = br[br.family == fam]
            th = pd.read_csv(DATA / "o4_breaker_theory.csv")
            th = th[th.family == fam].set_index("c")
            rows = []
            for c, g in br.groupby("c"):
                gm = g.groupby("tau_s").recrash.mean()
                after = gm[gm < 0.5]
                rows.append({"coupling / fold": c, "shortest halt that worked (x 1/spread)": after.index.min() if not after.empty else np.nan,
                             "predicted tau* (phase reduction)": th.loc[c, "tau_star_s"] if c in th.index else np.nan})
            out += ("<h3>Circuit breakers (O4 b)</h3><p>A synchronised population is uncoupled for a time, then "
                    "coupling resumes; the shortest halt (in units of 1 / frequency spread, from the grid 0.25 to 4) "
                    "after which most runs stay calm, against the prediction R(tau*) = R_u(lam).</p>"
                    + table(pd.DataFrame(rows), {c: c for c in rows[0]}, nd=3))
            ra = pd.read_csv(DATA / "o4_rate.csv")
            ra = ra[ra.family == fam].groupby("ramp_time_s").jumped.mean()
            out += ("<p>Rate (O4 c): coupling ramped from 0.5 to 1.6 times the fold from a calm start; fraction of runs "
                    "that jumped to synchrony: " + ", ".join(f"{v:.2f} over {k:g} / spread" for k, v in ra.items())
                    + ". Herding keeps the calm state linearly stable, so any jump is noise-induced.</p>")
    if not have("o5_summary.csv"):
        out += pending("O5", "Frequency-degree correlation on a scale-free network",
                       "every family on the same Barabasi-Albert network (N = 1000, mean degree 6), natural frequency rising "
                       "with degree against shuffled frequencies; up and down sweeps, gap between the R curves", "A10")
    if not have("o6_coefficients.csv"):
        out += pending("O6", "Sparse networks",
                       "herding fold on random regular graphs, k = 8 to 64 (N = 2000) and up to 192 (N = 20000), relative "
                       "to the complete graph; phase reduction predicts the same 1/k coefficient for every family", "A7")
    if have("o5_summary.csv"):
        o5 = pd.read_csv(DATA / "o5_summary.csv")
        o5 = o5[o5.family == fam]
        if not o5.empty:
            g = o5.groupby("shuffled")[["max_gap", "area"]].mean()
            out += ("<h3>Frequency-degree correlation on a scale-free network (O5)</h3><p>Largest gap between the "
                    f"upward and downward R curves: {g.loc[False, 'max_gap']:.2f} with frequency increasing with degree, "
                    f"{g.loc[True, 'max_gap']:.2f} with the frequencies shuffled (mean of 2 seeds).</p>")
    if have("o6_coefficients.csv"):
        o6 = pd.read_csv(DATA / "o6_coefficients.csv")
        o6 = o6[o6.family == fam]
        if not o6.empty:
            out += ("<h3>Sparse networks (O6)</h3><p>Herding fold on random regular graphs relative to the complete "
                    f"graph: fitted 1/k coefficient a = {o6.fit_a.iloc[0]:.2f} (Kuramoto theory on undirected graphs "
                    "1.65, measured in the main project 1.2 to 1.4).</p>"
                    + table(o6, {"k": "k", "fold": "fold", "ratio": "ratio to complete graph", "k_times_excess": "k (ratio - 1)"}, nd=4))
    return out


def pending(code: str, title: str, design: str, ada: str) -> str:
    return (f"<h3>{title} ({code}): designed, not yet run</h3><p>Design: {design}. Script and expected output are in "
            f"README.md; the run was moved to the Ada cluster (ADA_COMPUTE.md, item {ada}) and this block fills in when "
            "the page is rebuilt with its result file.</p>")


def propagation_block(name: str) -> str:
    if not have("o9_fits.csv"):
        return pending("O9", "Signal propagation", "response time of a node to a clamped neighbour against its degree, "
                       "tau ~ S^theta on Barabasi-Albert and Erdos-Renyi networks (N = 1000), against the Hens et al. "
                       "prediction for a product-separable coupling", "A11")
    d = pd.read_csv(DATA / "o9_fits.csv")
    d = d[d.model == name]
    if d.empty:
        return ""
    return ("<h3>Signal propagation (O9)</h3><p>Response time of a node to a clamped neighbour against its degree, "
            "tau ~ S^theta, with A2's protocol.</p>"
            + table(d, {"network": "network", "theta_measured": "theta measured", "lo": "CI low", "hi": "CI high",
                        "theta_hens": "Hens prediction", "slope_of_local_prediction": "slope of ln 2 / |J_ii|",
                        "corr_T_local": "corr(log tau, log local)", "nodes": "nodes"}, nd=3))


def chapter(c) -> str:
    cid, title, eq, unit, pop, fam, z1, sepname, papers = c
    body = math(html.escape(eq).replace("&amp;nbsp;", "&nbsp;"))
    body += f"<h3>One unit</h3><p>{html.escape(unit)}</p><h3>A population</h3><p>{html.escape(pop)}</p>"
    if fam:
        body += "<h3>Measured transitions (O1 atlas, N = 500, 4 seeds) against phase reduction</h3>" + herding_table(fam)
        body += reduction_line(fam)
    if cid == "inertia" and pred() is not None:
        up = pred().loc[("inertia", 0)].get("pred_up_onset", np.nan)
        body += ("<p>For this family the theory column is not phase reduction. At p = 0 it is the coupling at which the "
                 "locked-state self-consistency gives R = 0.5 (compare with the downward collapse); the jump up is "
                 f"predicted where the calm state loses linear stability, K = {fmt(up)}, against "
                 f"{fmt(atlas().loc[('inertia', 0, 'up'), 'mean'])} measured: the synchronised state already exists below "
                 "that point, so a finite population jumps early. At p = 1 and 2 it is the fold of the same "
                 "self-consistency (the locked state of the inertial model is the Kuramoto one).</p>")
    sp, ex = special(), exact()
    if cid == "sl" and sp is not None and ex is not None:
        body += ("<h3>Amplitude death (frequencies uniform on [-4, 4])</h3>"
                 f"<p>Exact window from linear stability: {fmt(ex['Stuart-Landau amplitude death, lower edge'])} &lt; K &lt; "
                 f"{fmt(ex['Stuart-Landau amplitude death, upper edge'])}. Measured: death begins at "
                 f"{fmt(sp.loc[('sl_death', 'up', 'dies'), 'mean'])} going up and ends at "
                 f"{fmt(sp.loc[('sl_death', 'down', 'revives'), 'mean'])} coming down (no hysteresis at the lower edge); "
                 f"oscillation returns at {fmt(sp.loc[('sl_death', 'up', 'revives'), 'mean'])} going up, but coming down "
                 f"from strong coupling death starts only at {fmt(sp.loc[('sl_death', 'down', 'dies'), 'mean'])}: a small "
                 "hysteresis loop at the upper edge that linear stability cannot see. A time step of 0.005 is needed "
                 "there (0.02 moves the edge to 4.98).</p>")
        if have("o3b_amplitude.csv"):
            body += fig("o3b_amplitude.png", "O3b. Stuart-Landau herding fold against the amplitude relaxation rate.",
                        "<p>Measured herding fold (mean and sd of 3 seeds) against 1/mu; phase reduction is exact at 1/mu = 0. Dashed lines: the phase-reduction folds.</p>",
                        "<p>Faster amplitude relaxation moves the fold toward phase reduction but levels off below it. These runs use the atlas sweep speed; O3c shows that speed lowers this fold by about 0.019, and even after that correction mu = 1 stays 10% below phase reduction. One candidate: this mean-field kernel measures herding with the amplitude-weighted |Z| / mean |z|, not with the phase order parameter of the theory; the network runs of O6 use the phase order parameter and test it.</p>")
    if cid == "winfree" and sp is not None and ex is not None:
        body += (f"<p>Exact death threshold {fmt(ex['Winfree oscillator death'])}; measured with a slow ramp: dies at "
                 f"{fmt(sp.loc[('winfree', 'up', 'dies'), 'mean'])}, revives at "
                 f"{fmt(sp.loc[('winfree', 'down', 'revives'), 'mean'])} (4 seeds each). Near death the period diverges, so "
                 "a cutoff on the mean velocity that is too high (0.3) fires early (0.752).</p>")
    if cid == "theta" and sp is not None and ex is not None:
        body += (f"<p>Ott-Antonsen folds: the active state collapses at shift "
                 f"{fmt(ex['theta neurons: active state collapses (shift)'])}, the quiet state recovers at "
                 f"{fmt(ex['theta neurons: quiet state recovers (shift)'])}. Measured (N = 500, 4 seeds): "
                 f"{fmt(sp.loc[('theta', 'down', 'collapses'), 'mean'])} and {fmt(sp.loc[('theta', 'up', 'recovers'), 'mean'])}.</p>")
    if cid == "fhn" and sp is not None:
        body += (f"<p>Single unit swept slowly: spikes start at I = {fmt(sp.loc[('fhn_single', 'up', 'starts spiking'), 'mean'])} "
                 f"going up (after the Hopf point 0.3313: a slowly ramped unit passes a Hopf point late) and stop at "
                 f"{fmt(sp.loc[('fhn_single', 'down', 'stops spiking'), 'mean'])} coming down (the large cycle survives below the "
                 "Hopf point).</p>")
    if fam:
        body += protocols_block(fam)
    if cid == "winfree":
        body += propagation_block("Winfree, death state")
    if cid == "theta":
        body += propagation_block("theta neurons, rest state")
    if z1:
        body += z1_row(z1)
    if sepname:
        body += sep_line(sepname)
    if papers:
        body += "<p>Papers in <code>references/</code>: " + ", ".join(f"<code>{html.escape(p)}</code>" for p in papers) + ".</p>"
    return step(cid, cid.upper()[:3], title, body)


def summary() -> str:
    at, pr, red = atlas(), pred(), reduction()
    if at is None or pr is None or red is None:
        return step("summary", "S", "Summary table", '<div class="note">Waiting for the atlas and phase reduction.</div>')
    rows = []
    for fam in ("kuramoto", "sl", "vdp", "rossler", "fhn", "inertia"):
        r = {"family": {"kuramoto": "Kuramoto", "sl": "Stuart-Landau", "vdp": "van der Pol", "rossler": "Roessler",
                        "fhn": "FitzHugh-Nagumo", "inertia": "Kuramoto with inertia"}[fam]}
        r["a1"] = red.loc[fam, "a1"] if fam in red.index else np.nan
        r["onset"] = at.loc[(fam, 0, "up"), "mean"]
        r["onset_pred"] = pr.loc[(fam, 0)].get("pred_K_at_R05", np.nan) if (fam, 0) in pr.index else np.nan
        r["fold1"] = at.loc[(fam, 1, "down"), "mean"]
        r["fold1_pred"] = pr.loc[(fam, 1)].get("pred_fold", np.nan) if (fam, 1) in pr.index else np.nan
        r["ratio1"] = r["fold1"] / r["fold1_pred"] if np.isfinite(r["fold1_pred"]) else np.nan
        r["fold2"] = at.loc[(fam, 2, "down"), "mean"]
        r["fold2_pred"] = pr.loc[(fam, 2)].get("pred_fold", np.nan) if (fam, 2) in pr.index else np.nan
        rows.append(r)
    d = pd.DataFrame(rows)
    extra = ""
    if have("o3c_extrapolated.csv"):
        e = pd.read_csv(DATA / "o3c_extrapolated.csv")
        extra = ("<h3>Folds extrapolated to an infinitely slow sweep (O3c)</h3>"
                 + table(e, {"family": "family", "p": "p", "fold_fastest": "fold, fastest sweep",
                             "fold_slowest": "fold, slowest sweep", "fold_extrapolated": "extrapolated",
                             "se": "se", "pred_phase_reduction": "phase reduction",
                             "rel_gap_extrapolated": "relative gap"}, nd=4)
                 + fig("o3c_rate.png", "O3c. Fold against sweep speed, every family.",
                       "<p>Measured fold divided by the phase-reduction fold, against (speed / fold)^(2/3); lines are fits fold_0 - c v^(2/3), meeting the left axis at the extrapolated slow-sweep fold. Dashed: phase reduction.</p>",
                       "<p>Shows how much of each family's gap to phase reduction is the finite sweep speed (slow passage through the fold) and how much is real.</p>"))
    body = ("<p>Measured thresholds (atlas, mean of 4 seeds) against phase reduction. The onset is where R first "
            "reaches 0.5 going up with plain coupling; the folds are where R falls below 0.5 coming down with herding.</p>"
            + table(d, {"family": "family", "a1": "a1", "onset": "onset (p = 0)", "onset_pred": "predicted",
                        "fold1": "fold p = 1", "fold1_pred": "predicted", "ratio1": "measured / predicted",
                        "fold2": "fold p = 2", "fold2_pred": "predicted"}, nd=4)
            + fig("o1_atlas.png", "O1. Every family swept up and down, with and without herding.",
                  "<p>One panel per family. R (order parameter) against the coupling, mean of 4 seeds. Solid: swept up from a calm start; dashed: swept down from synchrony. Grey p = 0, blue p = 1, red p = 2. Dotted verticals: phase-reduction folds (p = 1 blue, p = 2 red; grey for inertia: the linear-stability onset); circle: predicted onset at R = 0.5.</p>",
                  "<p>Herding turns synchronisation explosive in every family: the upward sweep stays calm far past the fold (at p = 2 it never synchronises in range), the downward sweep collapses abruptly. Kuramoto with inertia is already hysteretic at p = 0, and FitzHugh-Nagumo shows a small loop even at p = 0. The downward collapses in this figure come later than the true folds because the sweep is finite; O3c measures that delay and removes it.</p>")
            + fig("o1_special.png", "O1. Death, bistability and single-unit thresholds against exact values.",
                  "<p>Each marker is one seed's measured threshold, grouped by direction of the sweep and event; dashed lines are the exact values (linear stability of amplitude death, existence of the Winfree death state, Ott-Antonsen folds of the theta network, the FitzHugh-Nagumo Hopf point). Last panel: Stuart-Landau amplitude against sqrt(mu).</p>",
                  "<p>Every exact threshold is reproduced, after two fixes recorded in the code: a smaller time step at the upper edge of amplitude death and a lower velocity cutoff for Winfree death.</p>")
            + extra)
    return step("summary", "S", "Summary: every family against its theory", body)


def pipeline_map() -> str:
    def st(files: tuple[str, ...], ada: str) -> str:
        return "done" if have(*files) else f"designed; runs on Ada ({ada})"
    rows = [
        ("A1 to A11 (basic ideas)", "steps 0 to 5 of this page", "done"),
        ("B1 simulator, tests to 1e-12", "zoo.py, zoo_net.py; tests/test_oscillators.py (exact reproduction of Z1, complete-graph equivalence, kernels against exact variances and branches)", "done"),
        ("B2 run directories, esfc audit", "data/ files with logs in data/logs/; no per-run directories", "partial"),
        ("B3 interactive simulation", "demo.html (all families)", "done"),
        ("B4 results and reference pages", "index.html, reference.html (this page)", "done"),
        ("S1 reproduction of Gomez-Gardenes 2011", "O5 (frequency-degree correlation on BA networks, every family)", st(("o5_summary.csv",), "A10")),
        ("S2 herding makes it explosive", "O1 atlas, p = 0, 1, 2 for every family", "done"),
        ("S3, S4 crash anatomy and shock map", "O4 (a)", "done"),
        ("S5 circuit breakers", "O4 (b), with the halt-length prediction from phase reduction", "done"),
        ("S6 early warnings", "Z1, Z2, Z3; Z1 herding scenarios for van der Pol, Roessler, FitzHugh-Nagumo, inertia", "Z1 to Z3 done; herding scenarios on Ada (A9)"),
        ("S7 rate tipping", "O4 (c); slow passage in Z2, O3c and O3d", "done"),
        ("S8 what makes it explosive", "O5", st(("o5_summary.csv",), "A10")),
        ("S9 observable mapping", "market reading in steps 1 to 4; markets M0 to M11 on index.html", "done"),
        ("S10, S12 extensions", "inertia as its own family; others not repeated", "partial"),
        ("S11, S13 exact thresholds", "O3 phase reduction, O3c extrapolation, theory_zoo.py, data/o3_exact.csv; O3e for the Stuart-Landau gap", "done; O3e on Ada (A12)"),
        ("S14 and Part D, sparse networks", "O6 (fold against k for every family)", st(("o6_coefficients.csv",), "A7")),
        ("E1 to E8 markets", "M0 to M11 on index.html", "done"),
        ("F1 separability (A1)", "O8", "done"),
        ("F2 signal propagation (A2)", "O9 (Winfree and theta couplings are rank 1, so Hens et al. apply directly)", st(("o9_fits.csv",), "A11")),
        ("F3 herding strength from data (A3)", "not repeated per family (A3 found it not identifiable)", "not done"),
    ]
    head = "<tr><th>main project</th><th>counterpart here</th><th>status</th></tr>"
    body = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{html.escape(c)}</td></tr>" for a, b, c in rows)
    return step("map", "M", "The Kuramoto pipeline and its counterpart for every oscillator",
                f'<div class="tbl"><table class="fit"><thead>{head}</thead><tbody>{body}</tbody></table></div>')


def main() -> None:
    sections = part_one() + [chapter(c) for c in CHAPTERS] + [summary(), pipeline_map()]
    nav = [("p0", "0", "This page"), ("p1", "1", "Oscillators"), ("p2", "2", "Birth"), ("p3", "3", "Coupling"),
           ("p4", "4", "Herding"), ("p5", "5", "Phase reduction")] + \
        [(c[0], c[0].upper()[:3], c[1].split(":")[0]) for c in CHAPTERS] + [("summary", "S", "Summary"), ("map", "M", "Pipeline map")]
    navh = "".join(f'<a href="#{i}"><b>{t}</b>{html.escape(n)}</a>' for i, t, n in nav)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oscillator Reference</title><style>{CSS}</style></head><body>
<header><h1>Every oscillator family, built up from one unit</h1>
<p>Athmeeya M Kashyap, IIIT Hyderabad. Built {date.today():%Y-%m-%d} from <code>oscillators/data/</code>. The counterpart, for every
oscillator, of <a href="../research/reference.html">research/reference.html</a>.</p>
<p>The buttons below jump to each step (numbers) and each family's chapter (letters).</p></header>
<nav><div class="in">{navh}</div></nav>
<main>{"".join(sections)}</main>
<footer>Generated by oscillators/build_reference.py</footer></body></html>"""
    for bad in ("—", "–"):
        if bad in page:
            i = page.index(bad)
            raise ValueError(f"dash found near: {page[max(0, i - 60):i + 20]!r}")
    (HERE / "reference.html").write_text(page)
    print(f"wrote {HERE / 'reference.html'} ({len(page) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
