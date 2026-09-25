"""Build research/index.html: the complete, presentable record of every simulation, analytic result and
real-data test in research/.

    research/.venv/bin/python research/build_index.py      # then open research/index.html

Content lives in SECTIONS below. Every figure carries "What you are seeing" and "Significance".
Numbers that come from result files (A2 exponents, E1/E2 tables) are read at build time, so the page
always matches the latest run. Equations ($...$ inline, $$...$$ display) are rendered to static SVG at
build time by tools/tex2svg.cjs (MathJax under Node), so the page needs no JavaScript and no internet.
The build fails on any TeX error, unrendered '$', unbalanced HTML tag or missing image.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

VERDICT = {"new": ("new result", "v-new"), "repro": ("reproduction", "v-repro"), "neg": ("negative result", "v-neg"),
           "corr": ("partly confirmed", "v-open"), "tool": ("mechanism", "v-tool"), "data": ("real data", "v-data"),
           "theory": ("analytic", "v-theory")}


# ------------------------------------------------------------------------------------------------ helpers
def csv_table(path: Path, cols: list[str] | None = None, rename: dict | None = None, query: str | None = None,
              fmt: str = "{:.3f}", auc_cols: tuple = ()) -> str:
    if not path.exists():
        return '<div class="missing">table not available yet (run the script)</div>'
    df = pd.read_csv(path)
    if query:
        df = df.query(query)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if rename:
        df = df.rename(columns=rename)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                cls = ""
                if c in auc_cols and pd.notna(v):
                    cls = ' class="pos"' if v >= 0.6 else (' class="negv"' if v <= 0.4 else "")
                tds.append(f"<td{cls}>{'-' if pd.isna(v) else fmt.format(v)}</td>")
            else:
                tds.append(f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def fig(path: str, seeing: str, significance: str) -> dict:
    return {"path": path, "seeing": seeing, "significance": significance}


def render_fig(f: dict) -> str:
    p = ROOT / f["path"]
    img = (f'<a href="{f["path"]}" target="_blank"><img loading="eager" src="{f["path"]}" alt="{html.escape(f["path"])}"></a>'
           if p.exists() else f'<div class="missing">{html.escape(f["path"])} not generated yet</div>')
    return f"""<figure>{img}
  <figcaption><code>{html.escape(f['path'])}</code> · click to enlarge</figcaption>
  <div class="explain"><div><h4>What you are seeing</h4>{f['seeing']}</div>
  <div><h4>Significance</h4>{f['significance']}</div></div></figure>"""


def render_section(s: dict) -> str:
    label, cls = VERDICT[s["verdict"]]
    findings = "".join(f"<li>{b}</li>" for b in s.get("findings", []))
    return f"""
<section id="{s['id']}">
  <div class="sec-head"><span class="tag">{s['id'].upper()}</span><h2>{s['title']}</h2><span class="verdict {cls}">{label}</span></div>
  <div class="grid2">
    <div><h3>Question</h3><p>{s['question']}</p></div>
    <div><h3>Setup</h3><p>{s['setup']}</p><p class="script">Script: <code>{s['script']}</code></p></div>
  </div>
  {('<h3>Model / derivation</h3><div class="math">' + s['math'] + '</div>') if s.get('math') else ''}
  <h3>Results</h3><ul class="findings">{findings}</ul>
  {s.get('extra', '')}
  {''.join(render_fig(f) for f in s.get('figures', []))}
</section>"""


# ------------------------------------------------------------------------------------------------ dynamic A2 text
def a2_blocks() -> tuple[list[str], str]:
    exp = ROOT / "analytics" / "data" / "a2_exponents.csv"
    nl = ROOT / "analytics" / "data" / "a2_nonlinear_check.csv"
    if not exp.exists():
        return (["A2 has not finished running yet; the exponent table and figures appear here after "
                 "<code>analytics/a2_signal_propagation.py</code> completes."], "")
    df = pd.read_csv(exp)
    df["abs_err"] = (df.theta_measured - df.theta_pred).abs()
    df["pred_in_ci"] = (df.theta_pred >= df.ci_lo) & (df.theta_pred <= df.ci_hi)
    n_in = int(df.pred_in_ci.sum())
    finds = [f"Across {len(df)} model/network combinations the measured exponent lies within 95% CI of the prediction in "
             f"<b>{n_in}/{len(df)}</b> cases; median |measured − predicted| = <b>{df.abs_err.median():.2f}</b>, "
             f"max = {df.abs_err.max():.2f}."]
    for _, r in df.sort_values(["model", "network"]).iterrows():
        mark = "✔" if r.pred_in_ci else "✘"
        finds.append(f"{mark} <b>{html.escape(r.model)}</b> on {r.network}: predicted θ = {r.theta_pred:+.2f}, measured "
                     f"{r.theta_measured:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] (S ≥ 6 only: {r['theta_measured_S>=6']:+.2f}), "
                     f"degrees {r.S_range}.")
    if nl.exists():
        n = pd.read_csv(nl)
        finds.append(f"Linear response vs full nonlinear simulation ({len(n)} sources): median |log₁₀(T_nonlinear/T_linear)| "
                     f"≤ {n.median_abs_log10_ratio.max():.3f} for every model, i.e. linear response is exact to "
                     f"≈{100 * (10 ** n.median_abs_log10_ratio.max() - 1):.1f}%.")
    table = csv_table(exp, ["model", "network", "theta_pred", "theta_measured", "ci_lo", "ci_hi", "theta_measured_S>=6", "n_nodes", "S_range"],
                      {"theta_pred": "θ predicted", "theta_measured": "θ measured", "ci_lo": "95% CI low", "ci_hi": "95% CI high",
                       "theta_measured_S>=6": "θ measured (S≥6)", "n_nodes": "nodes", "S_range": "degree range"}, fmt="{:+.3f}")
    steady = ROOT / "analytics" / "data" / "a2_steady_states.csv"
    if steady.exists():
        st = pd.read_csv(steady)
        bad = st[(st.steady_residual >= 1e-8) | (~st.linearly_stable)]
        for _, r in bad.iterrows():
            finds.append(f"Excluded: <b>{html.escape(r.model)}</b> on {r.network} has no locked, stable steady state "
                         f"(residual {r.steady_residual:.1e}). For Sakaguchi this is physical: the phase lag creates degree-dependent "
                         f"frequency shifts ≈ λ⟨S⟩sin α that degree-1 nodes (pull ≤ λ) cannot balance.")
    return finds, "<h3>Exponent table</h3>" + table


def a1_block() -> str:
    p = ROOT / "analytics" / "a1_separability.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    # interaction and explicit decomposition written in TeX (x = x_i, y = x_j); D1 value reported by SymPy
    tex = {
        "SIS epidemic (Hens E)": (r"(1-x)\,y", r"M_1=1-x,\ M_2=y"),
        "Regulatory Michaelis-Menten (Hens R)": (r"\dfrac{y^h}{1+y^h}", r"M_1=1,\ M_2=\dfrac{y^h}{1+y^h}"),
        "Mutualistic population (Hens M)": (r"\dfrac{x\,y}{1+y}", r"M_1=x,\ M_2=\dfrac{y}{1+y}"),
        "Kuramoto (GG2011, S1)": (r"\sin(y-x)", r"\sin y\cos x-\cos y\sin x"),
        "Sakaguchi-Kuramoto (phase lag)": (r"\sin(y-x-\alpha)", r"\begin{aligned}&\cos\alpha\,(\sin y\cos x-\cos y\sin x)\\&-\sin\alpha\,(\cos y\cos x+\sin y\sin x)\end{aligned}"),
        "Linear diffusive / Stuart-Landau coupling": (r"y-x", r"1\cdot y+(-x)\cdot 1"),
        "tanh diffusive (saturating)": (r"\tanh(y-x)", None),
    }
    rows = []
    for r in d["pairwise"]:
        g, dec = tex[r["model"]]
        d1 = r["D1 = G*G_xy - G_x*G_y"]
        d1_cell = f"$D_1={d1}$" if d1 in ("0", "1") else "$D_1\\neq 0$"
        dec_cell = f"${dec}$" if dec is not None else "no finite decomposition (tanh is not an exponential polynomial)"
        rank = r["separable_rank"]
        rank_txt = "infinite (Levi-Civita)" if isinstance(rank, str) or rank is None else str(rank)
        sep = "yes" if r["product_separable (Hens eq. 3)"] else "<b>no</b>"
        rows.append(f"<tr><td>{html.escape(r['model'])}</td><td>${g}$</td><td>{sep}</td><td>{rank_txt}</td>"
                    f"<td>{d1_cell}</td><td class=\"wrap\">{dec_cell}</td></tr>")
    herd = d["herding_additivity"]
    labels = {"kuramoto": r"Kuramoto: $F_i=\sum_j\sin(\theta_j-\theta_i)$",
              "herding r_i": r"Herding: $F_i=r_i\sum_j\sin(\theta_j-\theta_i)$",
              "herding r_i^2": r"Herding variant: $F_i=r_i^2\sum_j\sin(\theta_j-\theta_i)$"}
    hrows = []
    for k, lab in labels.items():
        v = herd[k]
        hrows.append(f"<tr><td>{lab}</td><td>{v['max |d2F/dth1 dth2|']:.3g}</td>"
                     f"<td>{'yes' if v['pairwise_additive'] else '<b>no</b>'}</td></tr>")
    tri = herd["r_i^2 variant == explicit triadic (hypergraph) sum"]
    jac = d["herding_jacobian"]["max abs error closed form vs autodiff (incl. row sum)"]
    return ("<h3>Pairwise interactions: product-separability test</h3>"
            '<div class="tbl"><table class="fit"><thead><tr><th>model</th><th>interaction $G(x,y)$, $x=x_i$, $y=x_j$</th>'
            "<th>product-separable (Hens eq. 3)?</th><th>separable rank</th><th>test value</th><th>explicit decomposition</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            "<h3>Herding coupling: is the input additive over neighbours?</h3>"
            "<p>Test (k = 3 neighbours, 40 random configurations): a pairwise-additive input must have "
            "$\\partial^2F_i/\\partial\\theta_1\\partial\\theta_2=0$ identically.</p>"
            '<div class="tbl"><table><thead><tr><th>input to node i</th><th>max $|\\partial^2F_i/\\partial\\theta_1\\partial\\theta_2|$</th>'
            f"<th>pairwise-additive?</th></tr></thead><tbody>{''.join(hrows)}</tbody></table></div>"
            f"<p>$r_i^2$ variant equals the explicit triadic sum $k_i^{{-2}}\\sum_{{l,m,j}}\\cos(\\theta_l-\\theta_m)\\sin(\\theta_j-\\theta_i)$ "
            f"(SymPy identity): <b>{'verified' if tri else 'NOT verified'}</b>. "
            f"Closed-form herding Jacobian vs automatic differentiation (25 random configurations, k = 5, including the zero row sum): "
            f"max error <b>{jac:.1e}</b>.</p>")


def e_tables() -> tuple[str, str, str]:
    ed = ROOT / "empirical" / "data"
    markets = csv_table(ed / "e1_markets.csv", ["market", "index_source", "n_constituents", "start", "end", "n_events", "n_events_scored",
                                                "median_stocks_in_window"],
                        {"index_source": "index", "n_constituents": "constituents", "n_events": "crashes detected",
                         "n_events_scored": "crashes scored", "median_stocks_in_window": "median stocks per window"}, fmt="{:.0f}")
    e1 = csv_table(ed / "e1_pooled_auc.csv", ["indicator", "score", "auc", "ci_lo", "ci_hi", "event_median", "null_median"],
                   {"auc": "AUC", "ci_lo": "95% CI low", "ci_hi": "95% CI high", "event_median": "median score (crash peaks)",
                    "null_median": "median score (ordinary peaks)"}, query="null == 'peak'", fmt="{:.4f}", auc_cols=("AUC",))
    e2 = csv_table(ed / "e2_summary.csv", ["comovement", "n_crash", "median_gap_crash", "ci_lo", "ci_hi", "share_crashes_gap_positive",
                                          "n_placebo", "median_gap_placebo", "p_perm_one_sided", "positive_control_planted",
                                          "positive_control_recovered", "positive_control_ci_lo", "positive_control_ci_hi"],
                   {"comovement": "co-movement measure", "n_crash": "crashes", "median_gap_crash": "median gap (crashes)",
                    "ci_lo": "95% CI low", "ci_hi": "95% CI high", "share_crashes_gap_positive": "share gap > 0",
                    "n_placebo": "placebo spikes", "median_gap_placebo": "median gap (placebo)", "p_perm_one_sided": "p (crash > placebo)",
                    "positive_control_planted": "control: planted", "positive_control_recovered": "control: recovered",
                    "positive_control_ci_lo": "control CI low", "positive_control_ci_hi": "control CI high"}, fmt="{:.4f}")
    return markets, e1, e2


# ------------------------------------------------------------------------------------------------ math + validation
MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)


def render_math(page: str) -> tuple[str, int]:
    """Replace every $$...$$ / $...$ in <body> by MathJax SVG rendered at build time."""
    head, sep, body = page.partition("<body>")
    matches = list(MATH_RE.finditer(body))
    items = [{"tex": (m.group(1) if m.group(1) is not None else m.group(2)).strip(), "display": m.group(1) is not None} for m in matches]
    if not items:
        return page, 0
    proc = subprocess.run(["node", str(ROOT / "tools" / "tex2svg.cjs")], input=json.dumps(items), capture_output=True,
                          text=True, cwd=ROOT / "tools", check=True)
    out = json.loads(proc.stdout)
    errors = [(it["tex"], o["error"]) for it, o in zip(items, out) if "error" in o]
    if errors:
        raise SystemExit("TeX errors:\n" + "\n".join(f"  {e}  in: {t[:120]}" for t, e in errors))
    pieces, last = [], 0
    for m, o in zip(matches, out):
        pieces.append(body[last:m.start()])
        pieces.append(o["svg"])
        last = m.end()
    pieces.append(body[last:])
    return head + sep + "".join(pieces), len(items)


class _TagCheck(HTMLParser):
    VOID = {"meta", "img", "br", "hr", "input", "link", "source", "area", "base", "col", "embed", "param", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack, self.problems, self.imgs = [], [], []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.imgs.append(dict(attrs).get("src", ""))
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href and not href.startswith(("#", "http://", "https://")):
                self.imgs.append(href)
        if tag not in self.VOID:
            self.stack.append((tag, self.getpos()))

    def handle_startendtag(self, tag, attrs):
        if tag == "img":
            self.imgs.append(dict(attrs).get("src", ""))

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
            return
        idx = next((k for k in range(len(self.stack) - 1, -1, -1) if self.stack[k][0] == tag), None)
        if idx is None:
            self.problems.append(f"stray </{tag}> at line {self.getpos()[0]}")
        else:
            for t, pos in self.stack[idx + 1:]:
                self.problems.append(f"unclosed <{t}> opened at line {pos[0]}")
            del self.stack[idx:]


def validate(page: str) -> dict:
    body = page.partition("<body>")[2]
    text_only = re.sub(r"<mjx-container.*?</mjx-container>", "", body, flags=re.S)
    text_only = re.sub(r"<(style|script|code)[^>]*>.*?</\1>", "", text_only, flags=re.S)
    leftover = [m.start() for m in re.finditer(r"\$", text_only)]
    checker = _TagCheck()
    checker.feed(page)
    problems = checker.problems + [f"unclosed <{t}> opened at line {pos[0]}" for t, pos in checker.stack if t not in ("html", "body")]
    missing = [src for src in checker.imgs if not (ROOT / src).exists()]
    if [m for m in missing if m.startswith("../")]:
        raise SystemExit(f"broken local links: {[m for m in missing if m.startswith('../')]}")
    report = {"unrendered_dollar_signs": len(leftover), "html_problems": problems, "missing_images": missing,
              "images": len(checker.imgs)}
    if leftover:
        ctx = text_only[max(0, leftover[0] - 80):leftover[0] + 80]
        raise SystemExit(f"unrendered $ found ({len(leftover)}), first context: ...{ctx}...")
    if problems:
        raise SystemExit("HTML structure problems:\n  " + "\n  ".join(problems[:20]))
    return report

# ------------------------------------------------------------------------------------------------ content
GLOSSARY = [
    ("θ<sub>i</sub>(t)", "phase of trader/strategy i: position in its buy-sell cycle"),
    ("q<sub>i</sub> = sin θ<sub>i</sub>", "net order flow of i (+ buy, − sell); Q = ⟨q<sub>i</sub>⟩ is aggregate imbalance and moves the price"),
    ("ω<sub>i</sub>", "natural frequency: how fast i cycles on its own (its trading horizon); spread Δω = strategy diversity"),
    ("A<sub>ij</sub>, k<sub>i</sub> = S<sub>i</sub>", "network (who observes whom) and degree of node i"),
    ("λ", "herding / coupling strength: how strongly traders copy observed order flow"),
    ("r<sub>i</sub> = |Σ<sub>j</sub>A<sub>ij</sub>e<sup>iθ<sub>j</sub></sup>|/k<sub>i</sub>", "local agreement among i's neighbours (0 = split, 1 = unanimous)"),
    ("f", "fraction of consensus-sensitive (herding) traders whose coupling is multiplied by r<sub>i</sub>"),
    ("R = |⟨e<sup>iθ</sup>⟩|", "global synchrony (Kuramoto order parameter); crash = R jumps to ≈1 with Q ≈ −1"),
    ("λ<sub>f</sub>, λ<sub>b</sub>", "forward (crash onset) and backward (recovery) thresholds; λ<sub>b</sub> < λ < λ<sub>f</sub> is the bistable window"),
    ("σ", "noise intensity (idiosyncratic news)"),
    ("τ, τ*", "trading-halt duration and minimum safe halt predicted by theory"),
    ("τ<sub>i</sub> ~ S<sub>i</sub><sup>θ</sup>", "local response time of node i to a neighbour's perturbation and its degree exponent (Hens et al. 2019)"),
    ("AUC", "probability that a random crash scores higher than a random non-crash (0.5 = no skill)"),
    ("Kendall τ (trend)", "rank correlation of an indicator with time over the year before a date"),
]

SECTIONS = [
    {"id": "s1", "verdict": "repro", "title": "Explosive synchronization (Gómez-Gardeñes et al., PRL 2011)",
     "script": "s1_reproduce_gg2011.py",
     "question": "Does the simulator reproduce the original explosive (first-order, hysteretic) transition, and does the repository's earlier degree-normalized model show it?",
     "setup": "N = 1000, ⟨k⟩ = 6, ω<sub>i</sub> = k<sub>i</sub> (frequency-degree correlation), two networks per case, adiabatic sweeps (each coupling starts from the previous final state, 20 + 20 time units), Heun integrator dt = 0.004.",
     "math": r"$$\dot\theta_i=\omega_i+\lambda\,c_i\sum_j A_{ij}\sin(\theta_j-\theta_i),\qquad \omega_i=k_i,\qquad c_i=\begin{cases}1 & \text{GG2011}\\ 1/k_i & \text{repo (python/analysis)}\end{cases}$$",
     "findings": ["BA, unnormalized: forward jump at λ<sub>f</sub> ≈ 1.56, backward at λ<sub>b</sub> ≈ 1.31, max hysteresis gap <b>0.79</b>.",
                  "ER control: continuous transition, gap 0.02, matching the paper.",
                  "Repo variant (K/k<sub>i</sub>): continuous, gap <b>0.06</b>. Degree normalization destroys explosive synchronization, so the repository README's explosive-sync claim does not hold for that code."],
     "figures": [fig("figures/s1_gg2011_reproduction.png",
                     "Order parameter R (0 = incoherent, 1 = fully synchronized) versus coupling. Blue: coupling increased step by step; red: decreased. Orange shading marks where the two branches differ (bistability). Panels: original model on scale-free BA, same on ER, and the repo's degree-normalized version (its x-axis is K because the coupling is K/k<sub>i</sub>).",
                     "Left panel reproduces the hallmark of explosive synchronization: a jump of ΔR ≈ 0.8 and a hysteresis loop. The right panel shows the repository's old model lacks it, so all later work uses the unnormalized or adaptive forms.")]},
    {"id": "s2", "verdict": "repro", "title": "Herding feedback makes the market transition explosive (Zhang et al., PRL 2015)",
     "script": "s2_herding_explosive.py",
     "question": "If a fraction f of traders copy neighbours more strongly when those neighbours already agree, does the transition become explosive, and where is the bistable window?",
     "setup": "N = 500, ⟨k⟩ = 12, Gaussian ω (σ<sub>ω</sub> = 1), f ∈ {0, 0.25, 0.5, 0.75, 1}, noise σ ∈ {0, 0.3}, 3 networks each, λ ∈ [0, 6] in 49 steps.",
     "math": r"$$\dot\theta_i=\omega_i+\frac{\lambda}{\langle k\rangle}\,\alpha_i\sum_j A_{ij}\sin(\theta_j-\theta_i)+\sigma\xi_i(t),\qquad \alpha_i=\begin{cases}r_i & \text{herding trader (prob. } f)\\ 1&\text{otherwise}\end{cases}$$ Mean-field (ER, f = 1): $\dot\theta_i\approx\omega_i+\lambda R^2\sin(\Psi-\theta_i)$, so $R=H(u)=u\int_{-\pi/2}^{\pi/2}\cos^2\phi\,g(u\sin\phi)\,d\phi$ with $u=\lambda R^2$ and $\lambda(u)=u/H(u)^2$; the minimum of λ(u) is λ<sub>b</sub>.",
     "findings": ["ER: the window opens only for f ≥ 0.75; at f = 1, <b>λ ∈ [3.04, 3.88]</b> (σ = 0) and [3.21, 3.96] (σ = 0.3).",
                  "Scale-free BA has a much narrower window (0.17 vs 0.83 at f = 1): heterogeneous networks resist herding-induced bistability, the opposite of the frequency-degree mechanism in S1.",
                  "Mean-field λ<sub>b</sub> = 2.75 vs simulated 3.04 (≈10% low; the annealed approximation ignores finite-degree fluctuations of r<sub>i</sub>)."],
     "figures": [fig("figures/s2_herding_hysteresis.png",
                     "Each small panel is R versus herding λ for one network type (row) and herding fraction f (column). Solid blue/red: forward/backward sweeps without noise; dotted blue: forward sweep with noise σ = 0.3; orange: bistable region. Rightmost panels summarize the thresholds λ<sub>f</sub> (onset) and λ<sub>b</sub> (recovery) as functions of f.",
                     "Shows the explosive transition is produced by the herding mechanism itself (no frequency-degree correlation needed), and fixes the window [λ<sub>b</sub>, λ<sub>f</sub>] used by every crash experiment below.")]},
    {"id": "s3", "verdict": "tool", "title": "Anatomy of a crash: flash, persistent and endogenous",
     "script": "s3_crash_anatomy.py",
     "question": "Trader by trader, how does a crash unfold in the herding model, and which observable 'herding thermometers' react?",
     "setup": "ER N = 500, f = 1, σ = 0.3, identical network, traders and noise in all scenarios. A: λ below λ<sub>b</sub> plus a sell shock (ε = 1.2 for 8 time units); B: λ inside the window plus the same shock; C: λ ramped slowly past λ<sub>f</sub>, no shock. All share a weak bearish news term (ε = 0.02).",
     "math": r"$$\dot\theta_i=\dots+\varepsilon(t)\sin\!\big(-\tfrac{\pi}{2}-\theta_i\big),\qquad dx=\big[\beta Q-\kappa x\big]dt+\eta\,dB,\quad P=100\,e^{x},\quad \beta=0.01,\ \kappa=0.05$$",
     "findings": ["A: synchrony spike, −7.6% V-shaped flash crash, full recovery once the shock ends.",
                  "B: the <b>same shock</b> locks the market into selling: −16.2%, price stays at 87 (hysteresis).",
                  "C: no shock needed: −19.2%; bursts of partial synchrony (flicker) precede the jump.",
                  "Without any directional news the locked herd's common phase drifts from selling to buying (boom-bust whipsaw), which is why the news term is included."],
     "figures": [fig("figures/s3_crash_anatomy.png",
                     "Columns = scenarios A, B, C. Row 1: herding λ(t) with the bistable window (orange band) and the shock (red bar). Row 2: synchrony R (blue) and net order flow Q (orange; −1 = everyone selling). Row 3: market price with the fundamental value (dashed). Row 4: every trader's order flow over time, traders sorted by natural frequency; red = selling, blue = buying.",
                     "Makes the mechanism visible: a crash is the heatmap turning uniformly red. Whether it heals (A) or persists (B) depends only on where λ sits relative to λ<sub>b</sub>, not on the shock."),
                 fig("figures/s3b_herding_measures.png",
                     "Rows = scenarios B and C. Left: cross-sectional distribution of 10-period asset returns at three moments (calm, onset, crash) with the share of assets falling. Right: rolling herding measures, dispersion CSSD (green), breadth |mean sign| (orange), mean pairwise return correlation (purple) and the unobservable synchrony R (grey).",
                     "Breadth (85% of assets falling) and dispersion react only at the onset, not before. Pairwise correlation stays ≈ 0 even in a locked crash, because a constant selling rate does not co-fluctuate. Correlation-based systemic-risk measures can therefore miss a synchronized crash.")]},
    {"id": "s4", "verdict": "new", "title": "Crash-type phase diagram and shock targeting",
     "script": "s4_crash_phase_diagram.py",
     "question": "For herding λ and shock size ε, is a shock absorbed, a flash crash, or a lasting crash? Does it matter which traders are hit?",
     "setup": "ER N = 500, f = 1, σ = 0.3. Grid 20 λ × 16 ε × 4 seeds (settle 60, shock 8, relax 150). Targeting: λ = 3.51, ε = 2, shock applied to a fraction φ of traders chosen by degree, natural frequency, or at random; 10 fractions × 5 strategies × 12 seeds.",
     "math": "Classes: absorbed (R never exceeds 0.5), flash (exceeds 0.5 but relaxes), persistent (R ≥ 0.5 at the end), already crashed (R ≥ 0.5 before the shock).",
     "findings": ["The flash/persistent boundary lies between λ = 3.1 (no lasting crash for any ε) and λ = 3.4 (lasting crash in every run for ε ≥ 0.23), bracketing <b>λ<sub>b</sub> = 3.21</b> from S2; above the smallest shock it does not depend on ε.",
                  "Inside the window the second-smallest shock (ε = 0.23) already gives a lasting crash; with the smallest (ε = 0.1) the probability climbs from 0 at λ = 3.4 to 1 at λ = 3.8.",
                  "From λ = 3.8 some markets crash before any shock (25% of runs), rising to 100% at λ = 4.3 (noise-induced escape near λ<sub>f</sub> = 3.96).",
                  "Hitting 10-15% of traders is enough. At φ = 0.10 the probability of a persistent crash is 0.83 (hubs), 0.67 (random), 0.50 (slow traders), 0.17 (fast traders), 0.08 (periphery); at φ = 0.15 it is 1.00 for hubs, random and slow traders, 0.67 fast, 0.58 periphery. Hubs are the most and the periphery the least dangerous target; hubs vs random is within noise (12 seeds, SE ≈ 0.14), hubs vs periphery is not."],
     "figures": [fig("figures/s4_crash_phase_diagram.png",
                     "(a) Majority outcome over seeds for each (shock strength, herding) cell; dashed lines are λ<sub>b</sub> and λ<sub>f</sub> from S2. (b) Probability the market is still crashed at the end. (c) Probability of a persistent crash versus the fraction of traders hit (log axis), for five selection rules.",
                     "Shows that crash persistence is a property of the market state (the hysteresis branch), not of the size of the trigger. A regulator estimating λ relative to λ<sub>b</sub> would know whether the next shock heals or sticks.")]},
    {"id": "s5", "verdict": "corr", "title": "Circuit breakers as desynchronization control",
     "script": "s5_circuit_breakers.py + theory.py",
     "question": "How long must a trading halt be to stop a locked-in crash, and which trigger works better?",
     "setup": "Halt = coupling set to 0 (nobody sees anyone's trades) for τ; phases dephase at their own frequencies. (a) 8 λ × 16 τ × 8 seeds from a locked crash; (b) full price paths, 30 seeds, drawdown-7% vs breadth-triggered breakers, up to 4 halts; (c) partial halts.",
     "math": r"During a halt from the locked state (locked phases $\sin(\theta-\Psi)=\omega/u$): $$R_{\rm halt}(t)=\Big|\int_{-u}^{u}g(\omega)\,e^{\,i[\arcsin(\omega/u)+\omega t]}\,d\omega\Big|\,e^{-\sigma^2t/2}$$ Reduced criterion: trading resumes safely iff $R_{\rm halt}(\tau)<R_u(\lambda)$ (the unstable branch = basin boundary), giving $\tau^*(\lambda)$. Because $\lambda_b\propto\Delta\omega$ and the dephasing time $\propto 1/\Delta\omega$, $\tau^*=f(\lambda/\lambda_b)/\Delta\omega$.",
     "findings": ["Theory reproduces the trend and order of magnitude but lies <b>16-44% below</b> the simulated 50% boundary (theory/simulation = 0.56-0.84): 0.28 vs 0.49 at λ/λ<sub>b</sub> = 1.02, 0.47 vs 0.68 at 1.06, 0.57 vs 0.68 at 1.10, 0.65 vs 0.93 at 1.15, 0.73 vs 0.93 at 1.19. The simulated boundary is resolved only to the τ-grid step (factor 1.38). <i>(Corrects an earlier claim of close agreement.)</i>",
                  "Halts lose effectiveness close to λ<sub>f</sub>: the lowest re-crash probability over all τ ≤ 12 is 0 up to λ/λ<sub>b</sub> = 1.10, 0.12-0.25 at 1.15-1.19, and 0.75-1.00 from 1.24 (λ ≥ λ<sub>f</sub>), where noise re-triggers the crash.",
                  "Drawdown-7% breaker: τ = 0.5 re-fires (1.97 halts on average) and the market is still crashed in 97% of runs; τ = 1: 10%; τ = 2 and 4: 0%; τ = 8: 3%. Max drawdown falls from 16.0% without a breaker to 7.3-8.2%.",
                  "Breadth-triggered breaker (|mean sign of 10-period returns| > 0.6): max drawdown 4.1-4.6% for τ ≥ 1 (still crashed in 0-10% of runs), at the cost of more halts (1.1-4.0).",
                  "Partial halts (τ = 6): halting 20% of traders gives recovery probability 1.00 for every selection rule, as effective as a full halt (0.92-1.00). Hubs work with the fewest halted: at φ = 0.05 recovery is 0.58 (hubs) vs 0.00 (slow traders, random); at φ = 0.10 it is 0.92, 0.75 and 0.58."],
     "figures": [fig("figures/s5_circuit_breakers.png",
                     "(a) Colour = probability the market re-crashes after trading resumes, versus herding (relative to λ<sub>b</sub>) and halt duration (log). Black curve: theory τ*(λ); white dots: simulated 50% boundary; dotted line: λ<sub>f</sub>. (b) For two trigger rules, probability the market is still crashed (solid) and the absolute max drawdown (dotted) versus halt duration; grey lines = no breaker. (c) Recovery probability when only a fraction of traders is halted.",
                     "Provides a mechanistic, testable rule: halts must exceed a minimum that grows with herding and shrinks with strategy diversity, and they largely fail near λ<sub>f</sub> (re-crash probability ≥ 0.75 for every tested halt length). This connects to the real March 2020 sequence of four market-wide halts within eight days.")]},
    {"id": "s6", "verdict": "neg", "title": "Early-warning signals in the model",
     "script": "s6_early_warning.py",
     "question": "In the model, do classic early-warning signals fail before explosive transitions (as often claimed) while flickering works?",
     "setup": "40 ramp/null pairs for explosive (f = 1, λ 2.0 → 4.6) and continuous (f = 0, λ 0.6 → 2.6) transitions; indicators on 100-unit windows up to 20 units before the jump; Kendall τ trend; ROC AUC ramp vs null.",
     "findings": ["<b>No.</b> var(R), AC1(R), index variance and AC1 all reach AUC ≈ 0.96-1.0 for <i>both</i> transition types.",
                  "Design limitation: nulls keep coupling fixed while ramps sweep it widely, so any indicator that grows with coupling scores well. This test cannot rank indicators.",
                  "Retracted: the earlier statement that flickering beats classic signals is not supported by the model."],
     "figures": [fig("figures/s6_early_warning.png",
                     "(a) AUC of each indicator for continuous (blue) and explosive (orange) transitions; top block needs the unobservable R, bottom block uses simulated asset returns. (b, c) One example run of each type: R (grey), rolling variance (orange, scaled) and lag-1 autocorrelation (blue).",
                     "A negative, cautionary result: in this setup the explosive transition is not harder to anticipate than the continuous one. The design flaw is stated so the result is not over-read.")]},
    {"id": "s7", "verdict": "neg", "title": "Does the speed of crowding matter? (rate-induced tipping)",
     "script": "s7_rate_tipping.py",
     "question": "Does herding that builds up quickly crash the market when a slow build-up to the same level does not?",
     "setup": "Herding ramps from λ<sub>b</sub> − 0.8 to λ<sub>hi</sub> inside the window over T<sub>r</sub> ∈ [0.5, 500], then holds for 300 units; 4 levels × 7 ramp times × 24 seeds.",
     "findings": ["<b>No rate effect</b>: crash probability is flat in T<sub>r</sub> within binomial error (±0.08).",
                  "It depends only on where λ ends: ≈0.2 at 30% into the window, ≈0.8 at 60%, ≈1.0 at 80-95%.",
                  "Inside the window the calm state is metastable: noise alone triggers crashes (e.g. 21% within 300 units at 30% depth)."],
     "figures": [fig("figures/s7_rate_tipping.png",
                     "Probability of a crash within 300 units after the ramp (y) versus ramp duration (x, log). Each colour is a final herding level expressed as its depth into the bistable window; error bars are binomial standard errors.",
                     "Flat lines rule out rate-induced tipping in this model: the level of herding, not its speed, matters. This is consistent with mean-field theory (S2), in which the calm (incoherent) state stays linearly stable for every λ, so there is no stability boundary that a fast ramp could outrun; crashes inside the window are noise-induced escapes whose rate depends on λ, not on how λ got there.")]},
    {"id": "a1", "verdict": "theory", "title": "Is the model's differential equation separable? (Hens et al. framework)",
     "script": "analytics/a1_separability.py (SymPy)",
     "question": "Hens et al. (Nat. Phys. 2019) predict how signals spread through networks, but only for dynamics of the form below with a product-separable interaction; they explicitly exclude G = M(x<sub>j</sub> − x<sub>i</sub>), i.e. Kuramoto. Where do our models stand?",
     "setup": "Symbolic tests on each interaction G(x, y) (x = x<sub>i</sub>, y = x<sub>j</sub>), each confirmed numerically at random points. Hens' own models are included as controls.",
     "math": r"$$\text{Hens eq. (3):}\quad \dot x_i=M_0(x_i)+\sum_j A_{ij}\,M_1(x_i)\,M_2(x_j)$$ Product separability: $G=M_1(x)M_2(y)\iff D_1\equiv G\,G_{xy}-G_xG_y=0$. Separable rank r (minimal number of product terms): the $(r+1)\times(r+1)$ matrix $\big[\partial_x^a\partial_y^bG\big]_{a,b=0,\dots,r}$ is identically singular while the $r\times r$ one is not. For Kuramoto $G=\sin(y-x)$: $G_x=-\cos(y-x),\ G_y=\cos(y-x),\ G_{xy}=\sin(y-x)$, hence $$D_1=\sin^2(y-x)+\cos^2(y-x)=1\neq0,$$ so Kuramoto is <b>never</b> factorizable, but $\sin(y-x)=\sin y\cos x-\cos y\sin x$ has rank 2. Difference kernels $M(y-x)$ have finite rank iff M is an exponential polynomial (Levi-Civita), so $\tanh(y-x)$ has infinite rank. Herding input $F_i=r_i\sum_j\sin(\theta_j-\theta_i)$ is not even additive: $\partial^2F_i/\partial\theta_j\partial\theta_l\neq0$; with $\alpha_i=r_i^2$ it equals exactly $k_i^{-2}\sum_{l,m,j}\cos(\theta_l-\theta_m)\sin(\theta_j-\theta_i)$, a 3-body (hypergraph) interaction. Exact herding Jacobian: $$J_{ii}=-\tfrac{\lambda}{\langle k\rangle}k_ir_i^2\cos(\psi_i-\theta_i),\qquad J_{ij}=\tfrac{\lambda}{\langle k\rangle}r_i\big[\cos(\theta_j-\theta_i)-\sin(\theta_j-\psi_i)\sin(\psi_i-\theta_i)\big]$$ (non-reciprocal, $J_{ij}\neq J_{ji}$, rows sum to 0).",
     "findings": ["Hens' SIS, regulatory and mutualistic models: product-separable (rank 1), D₁ = 0 as they must be.",
                  "<b>Kuramoto, Sakaguchi, linear diffusive/Stuart-Landau: NOT product-separable</b> (D₁ ≡ 1), separable rank exactly 2.",
                  "tanh diffusive: not separable at any finite rank.",
                  "<b>Herding coupling: not pairwise-additive at all</b> (cross-derivative up to 0.47); its r<sub>i</sub>² version is exactly a triadic hypergraph interaction.",
                  "Therefore Hens' theory does not apply to any of our phase models as published. This is the gap confirmed in Hens et al. 2019 (text excludes M(x<sub>j</sub> − x<sub>i</sub>), citing Kuramoto) and in Meena et al., Nat. Phys. 2023 (power/synchronization dynamics tested only numerically, “not covered by our analytical framework”)."],
     "extra": "", "figures": []},
    {"id": "a2", "verdict": "theory", "title": "Signal propagation beyond factorizable interactions: generalized theory and test",
     "script": "analytics/a2_signal_propagation.py",
     "question": "Can Hens' degree-scaling law τ<sub>i</sub> ~ S<sub>i</sub><sup>θ</sup> be extended to non-factorizable (Kuramoto-type), infinite-rank, and non-additive (herding) interactions, with exponents predicted before simulation?",
     "setup": "Scale-free (BA, m = 3) and ER (⟨k⟩ = 6) networks, N = 1000. Steady state by integration plus Newton (phase models in the co-rotating frame). A source node is clamped at x*<sub>j</sub> + δ; T(j → i) = time to half the final response (Hens protocol). τ<sub>i</sub> = median T over sampled neighbours; θ = log-log slope with bootstrap CI. 160 sources per model/network; 4 per model re-run with the full nonlinear equations. Controls: Hens' SIS (θ = −1), regulatory a = 1 (θ = 0), a = 0.4 (θ = 3/2).",
     "math": r"Generalization: for any neighbour-additive $G$, configuration-model mean-field gives $\sum_jA_{ij}G(x_i,x_j)\approx S_iF(x_i)$ with $F(x)=\mathbb{E}_y[G(x,y)]$, which is Hens' eq. (8) with $M_1\bar M_2\to F$. Hence $$R(x)=-\frac{F(x)}{M_0(x)},\quad Y(x)=\Big(\frac{d[F R]}{dx}\Big)^{-1},\quad Y(R^{-1}(\lambda))\sim\lambda^{\Gamma(0)},\quad \theta=-2-\Gamma(0).$$ Phase oscillators ($M_0=\omega_i-\Omega$ constant): $\tau_i=1/\sqrt{(c_iS_iR_{nb})^2-(\omega_i-\Omega)^2}$ ⇒ θ = −1 for $c_i$ = const and θ = 0 for $c_i=K/S_i$. Herding: $|J_{ii}|=\tfrac{\lambda}{\langle k\rangle}S_ir_i^2\cos(\cdot)$ with $r_i^2\approx R_{nb}^2+(1-R_{nb}^2)/S_i$ ⇒ θ → −1 (crossover at $S^*=(1-R^2)/R^2$). Ratio interaction $G=y/(x+y)$, $M_0=-x^a$ (infinite separable rank): $x\sim S^{1/(a+1)}$ and $$\theta=\frac{1-a}{1+a}$$ (+1/3 for a = 0.5, −1/3 for a = 2), a value outside all Hens classes.",
     "findings": [], "figures": [
         fig("analytics/figures/a2_theta_pred_vs_measured.png",
             "Each point is one model on one network: x = exponent predicted by theory before simulation, y = exponent measured from the simulations (bars = 95% bootstrap CI). Blue circles: scale-free; orange squares: ER. Dashed diagonal = perfect agreement.",
             "Points on the diagonal validate the generalized theory; Hens' own models (SIS, regulatory) act as controls showing the pipeline reproduces the published exponents. Off-diagonal points identify where finite degree range or the approximations fail."),
         fig("analytics/figures/a2_tau_vs_degree.png",
             "One panel per model. Dots: local response time τ<sub>i</sub> of individual nodes versus their degree S<sub>i</sub> (log-log), blue = scale-free, orange = ER. Lines have the <i>predicted</i> slope θ; only their height is matched to the data. Titles give predicted and measured θ.",
             "The slope is the propagation fingerprint: θ < 0 means hubs react fastest (composite regime), θ = 0 means all nodes react alike (distance-limited), θ > 0 means hubs are bottlenecks (degree-limited). The Kuramoto, Sakaguchi and herding models are classified here for the first time; the repo's K/S<sub>i</sub> model falls in a different class from the unnormalized one."),
         fig("analytics/figures/a2_regimes_T_vs_L.png",
             "Distribution of propagation time T(j → i) versus network distance L between source and target on the scale-free network, for three models representing θ = 0, θ < 0 and θ > 0 (boxes = interquartile range, line = median, log scale).",
             "Shows the three regimes predicted from θ alone: arrival times ordered by distance (θ = 0), weakly distance-dependent with fast hub paths (θ < 0), and dominated by slow hubs regardless of distance (θ > 0).")]},
    {"id": "e1", "verdict": "neg", "title": "Real data: do early-warning signals precede actual crashes?",
     "script": "empirical/e1_ews_all_markets.py (+ fetch_markets.py)",
     "question": "At a market peak, can any indicator tell whether a ≥10% crash (30% for crypto) follows within 30 trading days?",
     "setup": "23 markets: US large/mid/small cap, UK 100/250, Germany, France, Spain, Switzerland, Netherlands, Italy, Sweden, Euro Stoxx 50, Japan, Hong Kong, India, Australia, Canada, Brazil, Korea, Singapore, 38 world indices (as nodes) and 40 cryptocurrencies; ≈2,900 constituents, daily 2000-2026 (current constituents: survivorship bias). Onset = true peak before the fall. Six indicators on 60-day windows using only past data. Three comparison sets; the fairest is peaks NOT followed by a crash. 95% CIs by cluster bootstrap over calendar quarters (a global crash hits many markets at once).",
     "math": r"Indicators on a trailing 60-day window (r = daily log return, i = stock, t = day): $$\mathrm{Var}(r_{\rm idx}),\qquad \mathrm{AC}_1(r_{\rm idx}),\qquad \bar\rho=\frac{2}{N(N-1)}\sum_{i<j}\rho_{ij},\qquad \frac{\lambda_1(C)}{N},$$ $$\mathrm{CSSD}=\big\langle\,\mathrm{std}_i\,r_{i,t}\big\rangle_t,\qquad \mathrm{flicker}=\Pr_t\!\Big[\,\big|\langle \mathrm{sign}\,r_{i,t}\rangle_i\big|\ge 0.6\,\Big].$$ Scores at date d: level $z=(v_d-\bar v_{[d-500,\,d-60]})/s_{[d-500,\,d-60]}$ and trend = Kendall τ of the indicator against time over the preceding 250 days.",

     "findings": ["324 scored crash onsets in 23 markets, 69 distinct calendar quarters.",
                  "Against random dates several indicators look informative (AUC 0.59-0.63), but against ordinary peaks <b>every indicator's 95% CI touches or includes 0.5</b>: best AUC 0.571 (dispersion trend), variance level 0.567, flicker 0.547-0.552, correlation/absorption 0.52-0.54, lag-1 autocorrelation ≤ 0.52.",
                  "The event study shows why: all co-movement measures are flat until the peak and rise only after the fall begins. They react to crashes; they do not anticipate them.",
                  "Per-market AUCs (level score, ordinary-peak comparison) scatter from 0.31 to 0.71 with 8-22 scored crashes per market, consistent with noise over 23 × 6 = 138 tests.",
                  "Model-derived flicker is no better than plain variance."],
     "figures": [fig("empirical/figures/e1_detected_crashes.png",
                     "Log-scale index for each market with every detected crash onset (red line). Titles give the index source and number of crashes.",
                     "Sanity check of the event definition: onsets coincide with known episodes (2000-02, 2008, Apr 2010, Jul 2011, Aug 2015, 2018, Feb 2020, 2022, 2025) in every market."),
                 fig("empirical/figures/e1_pooled_auc.png",
                     "For each indicator, the AUC (dot) and 95% cluster-bootstrap CI (bar) for separating crash peaks from: all dates (grey), volatility-matched dates (blue), ordinary peaks (orange). Left: indicator level at the peak; right: its trend over the prior year. Dashed line = 0.5 (no skill).",
                     "The orange bars, the only fair comparison, all reach 0.5. The apparent skill against random dates is a peak effect, not a crash warning."),
                 fig("empirical/figures/e1_event_study.png",
                     "Median path (thick line) and interquartile band of each indicator, as a z-score against its own past, from 250 days before to 60 days after the peak. Coloured: crash onsets; grey: ordinary peaks.",
                     "No indicator separates from the grey band before day 0; all rise after the fall starts. This is the visual core of the negative result."),
                 fig("empirical/figures/e1_per_market_level.png",
                     "Heatmap of AUC (level score, ordinary-peak comparison) for each market (row, number of crashes in brackets) and indicator (column). Red > 0.5, blue < 0.5.",
                     "Heterogeneity without a consistent pattern: no indicator is reliably red across markets, and signs flip between similar markets."),
                 fig("empirical/figures/e1_per_market_trend.png",
                     "Same as above for the trend score.",
                     "Confirms the level result.")]},
    {"id": "e2", "verdict": "neg", "title": "Real data: is there hysteresis in crashes?",
     "script": "empirical/e2_hysteresis.py",
     "question": "The model predicts crashes are bistable. Observable consequence: at the same volatility, co-movement should be higher on the way out of a crash than on the way in. Is it, beyond what ordinary volatility spikes show?",
     "setup": "Same 23 markets. Daily 20-day volatility and co-movement. For each crash: pre branch [onset − 120, trough], post branch (trough, trough + 250]; volatility binned into 6 common bins; gap = mean post − mean pre co-movement. Placebo: high-volatility spikes ≥ 250 days from crashes. Positive control: the same real series with +0.03 added to co-movement after every trough.",
     "math": r"$$\mathrm{gap}=\frac{1}{|B|}\sum_{b\in B}\Big[\overline{c}_{\rm post}(v\in b)-\overline{c}_{\rm pre}(v\in b)\Big]$$",
     "findings": ["347 crashes, 62 placebo spikes, 23 markets.",
                  "Median gap 0.0006 (correlation), 0.0028 (absorption), 0.0031 (flicker), each with a 95% CI containing 0 (widest [−0.013, +0.014]); only 50-53% of crashes have a positive gap; crash vs placebo p = 0.53-0.77.",
                  "Positive control: a planted +0.03 is recovered as 0.031-0.033 with CI excluding 0, so the test has power, and any real co-movement hysteresis is smaller than ≈0.013.",
                  "Caveat: in the model's price mapping (S3b) return correlation does not track synchrony, so this rejects co-movement hysteresis in data but is not a direct refutation of the model's R-hysteresis."],
     "figures": [fig("empirical/figures/e2_hysteresis_gaps.png",
                     "Histograms of the per-event gap for crashes (orange) and non-crash volatility spikes (grey), for three co-movement measures. Dashed lines: medians. x = 0 means no hysteresis.",
                     "Both distributions are centred on zero and overlap: real crashes show no systematic co-movement hysteresis beyond ordinary volatility episodes."),
                 fig("empirical/figures/e2_example_loops.png",
                     "For six S&P 500 crashes, mean pairwise correlation versus annualized 20-day volatility; blue = before and into the trough, orange = after.",
                     "Loops go both ways; in 2020 correlation at a given volatility was higher on the way in. Individual crashes do not follow the bistable ordering.")]},
]

REFS = [
    ("Hens et al., Spatiotemporal signal propagation in complex networks, Nat. Phys. 15, 403 (2019)", "../references/08_signal_propagation/hens2019_with_SI__arxiv_1801.08854.pdf"),
    ("Meena et al., Emergent stability in complex network dynamics, Nat. Phys. (2023)", "../references/08_signal_propagation/meena2023_emergent_stability__natphys.pdf"),
    ("Ji et al., Signal propagation in complex networks, Phys. Rep. 1017 (2023)", "../references/08_signal_propagation/ji2023_signal_propagation_review__physrep1017.pdf"),
    ("Gómez-Gardeñes et al., Explosive synchronization transitions in scale-free networks, PRL 106, 128701 (2011)", "../references/02_explosive_sync/gomezgardenes2011__1102.4823.pdf"),
    ("Zhang et al., Explosive synchronization in adaptive and multilayer networks, PRL 114, 038701 (2015)", "../references/02_explosive_sync/zhang2015_adaptive__1410.2986.pdf"),
    ("Lee et al., Proximity to explosive synchronization determines network collapse and recovery (2024)", "../references/02_explosive_sync/lee2024_es_proximity_crises__biorxiv_2024.11.28.625924.pdf"),
    ("Kato & Masuda, Early warning signals for synchronization transitions from partial observations (2026)", "../references/05_early_warning/kato_masuda2026__2608.28320.pdf"),
    ("Leyva et al., Local predictors of explosive synchronization with ordinal methods (2025)", "../references/05_early_warning/ordinal_predictors2025__2501.05202.pdf"),
    ("Scheffer et al., Early-warning signals for critical transitions, Nature (2009)", "../references/05_early_warning/scheffer2009__nature_08227.pdf"),
    ("Chen, Petukhov & Wang, The dark side of circuit breakers", "../references/06_finance/chen_petukhov_wang2018_dark_side_circuit_breakers.pdf"),
]


def summary_card() -> str:
    return """
<section class="summary">
<h2>Bottom line</h2>
<div class="grid2">
<div><h3>What is new (candidate contributions)</h3><ol>
<li><b>Separability classification (A1).</b> Kuramoto-type couplings are provably non-factorizable (D₁ ≡ 1, rank 2) and the herding coupling is non-additive (an exact 3-body interaction), so the Hens et al. signal-propagation theory does not cover them. Hens 2019 and Meena 2023 state this gap themselves.</li>
<li><b>Generalized propagation theory (A2).</b> Mean-field reduction F(x) = E<sub>y</sub>G(x,y) extends the degree-scaling exponent to any additive interaction, gives closed-form τ<sub>i</sub> for phase oscillators and the exact herding Jacobian, and predicts a new exponent θ = (1−a)/(1+a) for an infinite-rank interaction. Validation numbers are in A2.</li>
<li><b>Crash type set by the hysteresis branch (S4)</b> and <b>minimum trading-halt duration (S5)</b>: a semi-quantitative theory (16-44% below simulation) predicting τ* = f(λ/λ<sub>b</sub>)/Δω, i.e. safe halts scale inversely with strategy diversity, and halts that fail near λ<sub>f</sub>.</li>
</ol></div>
<div><h3>What did not hold up</h3><ul>
<li><b>No early warning in real data (E1):</b> 324 crashes, 23 markets; no indicator beats ordinary peaks (best AUC 0.57, CI reaches 0.5).</li>
<li><b>No co-movement hysteresis in real data (E2):</b> gap ≈ 0 ± 0.013 despite a positive control that detects 0.03.</li>
<li>Model early-warning ranking (S6) and rate-induced tipping (S7): negative.</li>
<li>The repository's original degree-normalized model has no explosive synchronization (S1).</li>
</ul></div></div>
<p class="note">Publishability: A1 + A2 form a self-contained theory paper <i>if</i> A2's exponents match (see its table). S4/S5 support a model paper on crash persistence and circuit-breaker design. E1/E2 are rigorous negative results that belong in such a paper as the empirical reality check.</p>
</section>"""


def build() -> Path:
    a2_find, a2_table = a2_blocks()
    for s in SECTIONS:
        if s["id"] == "a2":
            s["findings"] = a2_find
            s["extra"] = a2_table
        if s["id"] == "a1":
            s["extra"] = a1_block()
    markets, e1t, e2t = e_tables()
    for s in SECTIONS:
        if s["id"] == "e1":
            s["extra"] = "<h3>Markets used</h3>" + markets + "<h3>Pooled skill against ordinary peaks (fair comparison)</h3>" + e1t
        if s["id"] == "e2":
            s["extra"] = "<h3>Summary with positive control</h3>" + e2t
    gloss = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in GLOSSARY)
    refs = "".join(f'<li><a href="{u}" target="_blank">{html.escape(t)}</a></li>' for t, u in REFS)
    nav = "".join(f'<a href="#{s["id"]}"><span>{s["id"].upper()}</span>{html.escape(s["title"].split(":")[0].split("(")[0].strip())}</a>' for s in SECTIONS)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explosive Synchronization and Market Crashes</title>
<style>
:root {{ --bg:#f6f5f1; --card:#fff; --ink:#1c1c1e; --muted:#5d6166; --line:#e2dfd7; --accent:#0b5cad; --soft:#eef3fa;
        --new:#0a7d4f; --repro:#4a5fc1; --neg:#b3261e; --open:#9a6b00; --tool:#6b4fa0; --data:#0b5cad; --theory:#00707a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141518; --card:#1d1f24; --ink:#ececec; --muted:#a2a6ad; --line:#30333a; --accent:#7fb3ff; --soft:#1f2a38; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15.5px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif; }}
header {{ max-width:1200px; margin:0 auto; padding:44px 24px 18px }}
header h1 {{ margin:0 0 6px; font-size:32px; letter-spacing:-.02em }}
header p {{ margin:0; color:var(--muted) }}
nav {{ position:sticky; top:0; z-index:5; background:color-mix(in srgb, var(--bg) 92%, transparent); backdrop-filter:blur(6px); border-bottom:1px solid var(--line) }}
nav .in {{ max-width:1200px; margin:0 auto; padding:9px 24px; display:flex; gap:7px; overflow-x:auto }}
nav a {{ white-space:nowrap; text-decoration:none; color:var(--ink); font-size:12.5px; padding:4px 10px; border:1px solid var(--line); border-radius:999px; background:var(--card) }}
nav a span {{ color:var(--accent); font-weight:700; margin-right:5px }}
main {{ max-width:1200px; margin:0 auto; padding:22px 24px }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:24px 28px; margin:0 0 26px }}
section.summary {{ border:2px solid var(--accent) }}
.sec-head {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:6px }}
.sec-head h2 {{ margin:0; font-size:22px; letter-spacing:-.01em; flex:1 }}
h2 {{ font-size:22px }} h3 {{ font-size:15px; margin:16px 0 6px; color:var(--accent) }} h4 {{ margin:0 0 4px; font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted) }}
.tag {{ font-weight:800; color:var(--accent); font-size:13px }}
.verdict {{ font-size:12px; font-weight:700; padding:3px 10px; border-radius:999px; color:#fff }}
.v-new {{ background:var(--new) }} .v-repro {{ background:var(--repro) }} .v-neg {{ background:var(--neg) }} .v-open {{ background:var(--open) }}
.v-tool {{ background:var(--tool) }} .v-data {{ background:var(--data) }} .v-theory {{ background:var(--theory) }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:22px }}
@media (max-width:820px) {{ .grid2 {{ grid-template-columns:1fr }} }}
.math {{ background:var(--soft); border-radius:10px; padding:10px 16px; overflow-x:auto }}
mjx-container {{ color:inherit }}
mjx-container[jax="SVG"] {{ direction:ltr }}
mjx-container[jax="SVG"] > svg {{ overflow:visible; min-height:1px; min-width:1px }}
mjx-container[display="true"] {{ display:block; text-align:center; margin:.7em 0; overflow-x:auto; overflow-y:hidden; padding:2px 0 }}
.script {{ color:var(--muted); font-size:13px }}
code {{ font-size:12.5px; background:color-mix(in srgb, var(--accent) 10%, transparent); padding:1px 5px; border-radius:4px }}
.findings li {{ margin:5px 0 }}
figure {{ margin:22px 0 0; border-top:1px solid var(--line); padding-top:16px }}
figure img {{ width:100%; border-radius:8px; border:1px solid var(--line); background:#fff }}
figcaption {{ font-size:12px; color:var(--muted); margin:4px 0 8px }}
.explain {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; background:var(--soft); border-radius:10px; padding:12px 16px }}
@media (max-width:820px) {{ .explain {{ grid-template-columns:1fr }} }}
.missing {{ padding:16px; border:1px dashed var(--line); border-radius:8px; color:var(--muted); font-size:13px }}
.tbl {{ overflow-x:auto; margin:8px 0 }} table {{ border-collapse:collapse; font-size:12.5px; min-width:100% }}
th, td {{ border-bottom:1px solid var(--line); padding:5px 9px; text-align:left; white-space:nowrap }}
td.wrap {{ white-space:normal; min-width:280px }}
table.fit th, table.fit td {{ white-space:normal }}
.tbl th {{ white-space:normal; vertical-align:bottom }}
table.fit td:first-child {{ min-width:150px }}
th {{ color:var(--muted); font-weight:600 }} td.pos {{ color:var(--new); font-weight:700 }} td.negv {{ color:var(--neg); font-weight:700 }}
.note {{ background:var(--soft); padding:10px 14px; border-radius:8px }}
footer {{ text-align:center; color:var(--muted); font-size:12px; padding:6px 0 40px }}
</style></head><body>
<header><h1>Explosive synchronization &amp; market crashes</h1>
<p>Simulations, analytic results and real-data tests. Every figure explains what is shown and why it matters; verdict tags state plainly whether a result holds.</p></header>
<nav><div class="in"><a href="#summary"><span>★</span>Bottom line</a><a href="#glossary"><span>§</span>Symbols</a>{nav}<a href="#refs"><span>¶</span>References</a></div></nav>
<main>
<div id="summary">{summary_card()}</div>
<section id="glossary"><h2>Symbols and terms</h2><div class="tbl"><table><tbody>{gloss}</tbody></table></div></section>
{''.join(render_section(s) for s in SECTIONS)}
<section id="refs"><h2>Key references (local PDFs)</h2><ul>{refs}</ul><p>Full library: <code>references/README.md</code>.</p></section>
</main>
<footer>Generated {datetime.now():%Y-%m-%d %H:%M} by research/build_index.py</footer>
</body></html>"""
    page, n_math = render_math(page)
    report = validate(page)
    out = ROOT / "index.html"
    out.write_text(page)
    print(f"wrote {out}: {n_math} equations rendered to SVG, {report["images"]} local images/links "
          f"({len(report['missing_images'])} not generated yet: {report['missing_images']}), HTML structure OK")
    return out


if __name__ == "__main__":
    build()
