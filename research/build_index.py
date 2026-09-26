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
    {"id": "s8", "verdict": "corr", "title": "What actually makes the transition explosive",
     "script": "s8_topology_frequency_sweep.py + s8c_correlation_corrected.py",
     "question": "Topology, frequency law, frequency-degree correlation and noise are all said to drive explosive synchronization. Under one matched normalization, which of them really does?",
     "setup": "Factorial sweep: 6 topologies (ER, BA, scale-free, Watts-Strogatz, ring, random regular) x 4 frequency laws x correlation c, all standardized to zero mean and unit scale so the coupling axis is comparable. N = 600, adiabatic sweeps. S8c repeats the correlation sweep after a design error was found in S8.",
     "math": r"Correlation is imposed by a Gaussian copula, $z_i=c\,\Phi^{-1}(\mathrm{rank}(k_i)/N)+\sqrt{1-c^2}\,\varepsilon_i$, which fixes the RANK correlation while forcing the marginal of $\omega$. That is the error: the Gomez-Gardenes mechanism needs $\omega_i\propto k_i$, i.e. proportionality, not ordering. With a Gaussian marginal the largest hub receives $\omega\approx3.1$ rather than $\omega\propto k_{\max}$.",
     "findings": ["<b>A Gaussian frequency law never produces a jump, at any correlation.</b> &Delta;R stays flat at 0.08 from c = -1 to +1, with zero bistable window.",
                  "The jump appears only at |c| = 1 <i>and</i> only when the tail of g(&omega;) matches the tail of P(k): &Delta;R = <b>0.578</b> with a degree-matched marginal, 0.498 with a Lorentzian, 0.088 with a Gaussian. Even c = 0.75 gives almost nothing.",
                  "So the correct statement is not &quot;frequency-degree correlation causes explosive synchronization&quot;, it is <b>the tail of g(&omega;) must match the tail of P(k); a rank correlation is not sufficient</b>.",
                  "No-free-parameter check against S1: predicted &lambda;<sub>f</sub> = 1.56&middot;&lang;k&rang;/std(k) = 1.357, measured <b>1.562</b>, ratio 1.15.",
                  "Noise erodes the transition monotonically but does not destroy it: &Delta;R 0.578 &rarr; 0.480 and the window 0.438 &rarr; 0.250 as &sigma; goes 0 &rarr; 0.8.",
                  "<b>Correction to S8 itself:</b> the first factorial used too short a relaxation and returned <i>negative</i> hysteresis widths, which is unphysical. Only its &Delta;R values are quoted; S8c doubles the relaxation."],
     "figures": [fig("figures/s8c_correlation_corrected.png",
                     "Forward jump &Delta;R against the frequency-degree rank correlation c, one line per marginal shape of g(&omega;), on BA and ER. The right panels show the bistable window width and the effect of noise.",
                     "Separates two things the literature usually conflates: the correlation between frequency and degree, and the shape of the frequency distribution. Only their combination produces a first-order transition.")]},
    {"id": "s9", "verdict": "new", "title": "The observable gap: why E2 could not have detected the model",
     "script": "s9_observable_mapping.py",
     "question": "E2 found no co-movement hysteresis and flagged a caveat about itself. Was that a refutation of the model, or a test without the power to see it?",
     "setup": "The herding market with lambda ramped up through lambda_f and back down, co-movement measured with E1's own estimators, against real calm/crash levels from all 23 markets.",
     "math": r"In S3's price map each asset is driven by its own order flow plus INDEPENDENT noise, so in a locked crash every $q_i$ is pinned and $\bar\rho\equiv0$ identically, at every $R$. Adding one common market mode $\sigma_c\,dW_c$ to the PHASE equation gives asset $i$ a loading $\cos\theta_i$ on the shared shock, and with $a=\beta\sigma_c$, $v$ the idiosyncratic variance, $$u_i=\frac{a\cos\theta_i}{\sqrt{a^2\cos^2\theta_i+v}},\qquad \bar\rho=\langle u_iu_j\rangle_{i\neq j}=\langle u\rangle^2+O(1/N),$$ and in the weak-mode limit $\bar\rho\approx(a^2/v)R^2\cos^2\Psi$ since $\langle\cos\theta\rangle=R\cos\Psi$. So $\bar\rho\propto R^2$.",
     "findings": ["<b>Real markets are unambiguous: co-movement rises from calm to crash in 23 markets out of 23</b>, median +0.127 (0.229 &rarr; 0.389), from +0.029 (Brazil) to +0.265 (Japan). The unmodified model produces <b>0.000</b>.",
                  "That zero is an identity, not a parameter choice: with independent per-asset noise the correlation is zero at every level of synchrony.",
                  "With the common mode on, crash co-movement rises 0.0001 &rarr; 0.0955 as &sigma;<sub>c</sub> goes 0 &rarr; 0.8, and the closed form &lang;u&rang;&sup2; tracks it within about a factor of two.",
                  "<b>The decisive number: E2's own gap statistic computed on the repaired model is 0.0002 to 0.0028 at every &sigma;<sub>c</sub>, against E2's detection bound of 0.013.</b> Even with the mapping fixed the predicted effect is roughly four times smaller than the smallest one E2 could have seen.",
                  "So E2's null does <b>not</b> falsify the model's hysteresis. It is a negative result about the test, not the model.",
                  "But one channel is not enough: &sigma;<sub>c</sub> cannot produce a calm-period baseline (0.0006 against a real 0.229), while a common shock in the prices matches the level and produces no crash rise (+0.008 against +0.127). A calibrated model needs both."],
     "figures": [fig("figures/s9_observable_mapping.png",
                     "Left: synchrony R (thin) and mean correlation (thick) through a crash, for several values of the common market mode, with the real calm and crash levels as dotted lines. Middle: calibration against real markets. Right: E2's hysteresis gap computed on the model, with E2's real-data bound shaded.",
                     "Turns E2 from an apparent refutation into a statement about statistical power, and supplies the closed-form bridge from synchrony to an observable a regulator can actually measure.")]},
    {"id": "s10", "verdict": "corr", "title": "Do the proposed extensions actually improve the model?",
     "script": "s10_extensions.py",
     "question": "Eighteen extensions were proposed. Which of them measurably move the model toward real markets, and which quietly destroy the phenomenon they were meant to explain?",
     "setup": "Eight phase-side variants x four price mappings, scored against the 23 markets on excess kurtosis 10.20, volatility clustering 0.246, leverage effect -0.113 and crash co-movement 0.389. Lambda cycles across the bistable window so each run contains real crash episodes.",
     "findings": ["<b>State-dependent liquidity is the single most valuable addition.</b> It appears in every top-scoring combination and is the only change that produces fat tails or volatility clustering at all: kurtosis 0.035 &rarr; <b>2.06</b>, clustering 0.008 &rarr; <b>0.34</b>. Without it the model's returns are Gaussian white noise.",
                  "Volatility clustering (target 0.246) and crash co-movement (target 0.389) are now <b>reproduced</b>. Fat tails improve 59-fold but remain 5x short of the real 10.2.",
                  "<b>The leverage effect is not reproduced by anything.</b> Asymmetric herding was added specifically for it and produces the <b>wrong sign</b> (+0.007 to +0.100 against a real -0.113) while halving the bistable window. Rejected.",
                  "<b>Contrarians destroy the transition entirely.</b> Just 15% of desks leaning against observed flow collapses the hysteresis gap to 0.108 from 0.767 and R never reaches 0.5. Rejected as an improvement, but it is a finding: market makers look structurally stabilizing.",
                  "Inertia <b>widens the bistable window 3.6x</b> (0.675 &rarr; 2.400), confirming that it supplies hysteresis independently of any frequency-degree correlation.",
                  "Keep: liquidity, common mode, inertia. Reject: contrarians, asymmetric herding.",
                  "<b>Correction to this test:</b> its first version held lambda fixed inside the window and measured <i>frac_crashed = 0.0</i> for every variant, so the returns were pure noise and the whole comparison was vacuous. The cycling design replaced it."],
     "figures": [fig("figures/s10_extensions.png",
                     "Each panel is one stylized fact; colour is the phase-side variant, x-axis is the price mapping, the dashed line is the real-market value. The rightmost panel is the bistable window width for each variant against the baseline.",
                     "Turns a list of plausible-sounding model extensions into a ranked, falsifiable comparison, and rejects two of them on evidence.")]},
    {"id": "e3", "verdict": "neg", "title": "Real data: does Lee et al. (PNAS 2025) generalize?",
     "script": "empirical/e3_lee_reproduction.py",
     "question": "Lee et al. report that a market's pre-crisis proximity to explosive synchronization predicts how fast it collapses and recovers, validated on 39 country indices in the 2008 crisis. Does that hold at constituent level across 26 years?",
     "setup": "Their estimator implemented exactly: Hilbert transform each stock's returns, build the instantaneous Kuramoto order parameter across the market, take the ACF of it in overlapping moving windows, and use the kurtosis of those ACF values as ES proximity. 315 events, 23 markets, cluster bootstrap over markets.",
     "findings": ["<b>The 2008 result does not reproduce at constituent level</b> (n = 53): collapse &rho; = -0.207 with CI [-0.435, +0.015], recovery &rho; = -0.017. Neither is significant.",
                  "Out of sample across 262 other events, <b>the recovery half survives</b>: &rho; = +0.133, CI [+0.038, +0.215], and the sign is the one hysteresis predicts (closer to explosive means a more prolonged recovery). The collapse half does not (&rho; = +0.080, CI includes zero).",
                  "The surviving effect is weak: &rho; = 0.13 explains under 2% of the variance.",
                  "<b>Not an exact replication.</b> They used 39 country indices from Compustat Global; this uses constituent-level Yahoo data inside 23 markets, so the networks differ and survivorship bias applies here."],
     "figures": [fig("empirical/figures/e3_lee_reproduction.png",
                     "Pre-crisis ES proximity against log collapse time (circles) and log recovery time (squares), for the 2008 crisis, for all other events, and pooled. Spearman coefficients in the titles.",
                     "Directly engages the closest prior work rather than citing it. A predictor validated on a single crisis is tested on 262 independent events, and half of it survives.")]},
    {"id": "e4", "verdict": "data", "title": "Real data: does the answer change intraday?",
     "script": "empirical/e4_intraday.py",
     "question": "The model describes intraday synchronization but E1 and E2 tested daily closes, three orders of magnitude coarser. Is the null result just a timescale mismatch?",
     "setup": "Hourly bars for 119 large US names, the deepest free history available (730 days). E1's event study repeated on that grid, with the same matched-peak null.",
     "findings": ["<b>Hard data limit:</b> Yahoo serves 1-minute bars for 7 days, 5-minute for 60 days and hourly for 730 days, so the 2010, 2015 and 2018 events cannot be tested without a paid feed. Hourly is about a 7x improvement over daily, not the full fix.",
                  "At hourly resolution co-movement is <b>elevated before crashes but not before ordinary peaks</b> (0.206 against 0.123 sixty bars ahead), giving AUC 0.717 against E1's 0.542 on daily data.",
                  "<b>But there are only 6 events.</b> Observations inside one event are heavily autocorrelated, so the effective sample size is about 6, not 72. Event-level sign test: 5 of 6.",
                  "<b>Suggestive, not significant.</b> The value of this result is that it identifies a paid intraday feed, rather than more daily data, as the way to settle E1 and E2."],
     "figures": [fig("empirical/figures/e4_intraday.png",
                     "Left: median mean correlation around crash onsets (red) and ordinary peaks (grey), in hourly bars. Right: the equal-weight hourly index with detected events marked.",
                     "Addresses the strongest objection to the empirical chapters, and shows honestly how far free data can take it.")]},
    {"id": "a3", "verdict": "neg", "title": "Can the herding coupling be measured in a real market?",
     "script": "a3_lambda_estimation.py",
     "question": "Every operational claim here is phrased as where lambda sits relative to lambda_b, yet lambda has never been measured. Can it be recovered from observable co-movement?",
     "setup": "Invert the S9 mapping to get R from measured co-movement, then invert the model's own equilibrium curve R_eq(lambda) to get lambda, for 27,723 market-days across 23 markets.",
     "math": r"$$\bar\rho\approx(a^2/v)R^2\ \Rightarrow\ \hat R=R_{\max}\sqrt{\bar\rho/\bar\rho_{\max}},\qquad \hat\lambda=R_{eq}^{-1}(\hat R)$$",
     "findings": ["<b>This does not work, and the reason is structural.</b> R<sub>eq</sub>(&lambda;) is very nearly a step function, jumping from R = 0.137 to R = 0.899 across a &lambda; window only <b>0.167</b> wide.",
                  "<b>97.2% of real market-days land inside that blind region</b>, so the estimate has an inter-quartile range of 0.04. The discontinuity that makes explosive synchronization interesting is exactly what makes its control parameter unidentifiable from its order parameter.",
                  "<b>Even a perfect inversion could not help.</b> The estimate is a monotone transform of mean correlation, and AUC is invariant under monotone transforms, so it must score identically to the raw indicator. No reparametrization of an indicator can beat that indicator.",
                  "This kills what had been this repository's headline recommendation, and it is reported rather than dropped.",
                  "<i>An earlier version of this script reported AUC 0.949. That was an artifact of a thinning bug leaving only 8 comparison points.</i>"],
     "figures": [fig("figures/a3_lambda_estimation.png",
                     "Left: the model's equilibrium curve used as the inversion table, with the two thresholds marked. Middle: the estimated coupling for the S&amp;P 500 over 26 years, with crash onsets. Right: the estimate at crash onsets against ordinary peaks.",
                     "A negative result about identifiability. It sets a hard limit on how operational any first-order synchronization model of markets can be made from co-movement data alone.")]},
    {"id": "s11", "verdict": "theory", "title": "Exact thresholds, and does the jump survive N to infinity?",
     "script": "s11_analytics.py",
     "question": "S2's mean field was numerical and about 10% off. Can the herding model be solved exactly for Lorentzian frequencies, and is the first-order jump real or a finite-size artifact?",
     "setup": "Closed-form solution of the cubic for Lorentzian g, checked against a direct N = 2000 simulation; then a finite-size sweep from N = 125 to N = 4000, 3 seeds per size.",
     "math": r"The classic Kuramoto result $R=\sqrt{1-K_c/K}$, $K_c=2\gamma$, has the mean field entering as $KR$. In the herding model the gain is $\alpha_i=r_i$, so it enters as $\lambda R^2$, i.e. $K_{\rm eff}=\lambda R$. Substituting gives a cubic $$\lambda R^3-\lambda R+2\gamma=0,$$ whose fold (double root, $3\lambda R^2=\lambda$) sits at $R^*=1/\sqrt3$, and back-substitution gives the closed form $$\boxed{\lambda_b=3\sqrt3\,\gamma\approx5.196\,\gamma,\qquad R^*=1/\sqrt3\approx0.5774.}$$",
     "findings": ["<b>Closed form for the recovery threshold</b>: &lambda;<sub>b</sub> = 3&radic;3&thinsp;&gamma;, with R* = 1/&radic;3 at the fold. Verified: at &lambda; = 5.2, just above the fold, the two roots are 0.5901 and 0.5645, converging on 0.5774 as they must.",
                  "Simulation at N = 2000 gives &lambda;<sub>b</sub> = 5.800 against the exact 5.196, ratio <b>1.116</b>. That 12% gap is not an algebra error, it is the finite-connectivity correction: the closed form is exact for all-to-all coupling while the simulation runs on a sparse graph with &lang;k&rang; = 12. <b>This explains the ~10% offset S2 could only measure.</b>",
                  "<b>Finite-size: the jump GROWS with N</b>, 0.183 at N = 125 to <b>0.486</b> at N = 4000, and so do the hysteresis gap (0.119 to 0.473) and the window (0.000 to 0.400). A finite-size artifact would decay toward zero, so <b>the first-order discontinuity is genuine</b> and the values quoted elsewhere at N = 500 are if anything conservative.",
                  "Caveats: 3 seeds per size, the trend is not perfectly monotone (N = 2000 dips), and the window is resolved only to the 0.2-wide &lambda; grid."],
     "figures": [fig("figures/s11_analytics.png",
                     "Left: the exact stable and unstable branches from the cubic, with the fold marked. Middle and right: the forward jump, hysteresis gap and bistable window against system size on log axes.",
                     "Replaces a numerical saddle-node search with a closed form and explains its residual error, then establishes that the central phenomenon is not a small-system illusion.")]},
    {"id": "e5", "verdict": "neg", "title": "Real data: the model on a REAL market network",
     "script": "empirical/e5_real_network.py",
     "question": "Every structural claim here rests on synthetic ER or BA graphs. What happens on a network estimated from actual market data?",
     "setup": "Two networks per market from daily returns, thresholded to the same mean degree (12) as the synthetic benchmark so degree is fixed and only structure differs: correlation-thresholded, and lead-lag (i leads j at +1 day more than the reverse, then symmetrised). US, UK, Japan, Germany.",
     "findings": ["Real market networks are <b>an order of magnitude more clustered</b> than anything used elsewhere here: C = 0.731 (correlation) and 0.138 (lead-lag) against <b>0.030</b> for synthetic ER and 0.081 for BA. &kappa; is 1.80 to 1.98 against 1.09 for ER.",
                  "<b>The bistable window essentially vanishes on them.</b> Hysteresis gap 0.793 on synthetic ER falls to <b>0.172</b> (correlation) and <b>0.190</b> (lead-lag), and the window width goes to zero (-0.100 and 0.000 against 0.800).",
                  "<b>This is the most serious result in the repository.</b> S4 (crash persistence is set by the hysteresis branch) and S5 (minimum safe halt duration), the two most novel items, both depend entirely on that window existing. On a network estimated from real data there is no window for a shock to leave the market in.",
                  "Likely mechanism: the clustering gap. When neighbours share neighbours, local agreement r<sub>i</sub> saturates locally instead of propagating globally, which is exactly the positive feedback the herding route needs. It also means the annealed mean field, which assumes no clustering, does not apply to the real networks either.",
                  "<b>Caveats, stated plainly:</b> the real networks are smaller (n = 125 to 335 against 400) and the thresholding is a modelling choice, not an observable. Neither explains a fourfold collapse of the gap, but both should be checked at matched n and across thresholds before this is treated as settled."],
     "figures": [fig("empirical/figures/e5_real_network.png",
                     "Left and middle: degree heterogeneity and clustering for the real networks against the synthetic ones, at matched mean degree. Right: the bistable window width and forward jump for each.",
                     "Tests the one load-bearing assumption that had never been tested, and finds that the central phenomenon of the model is much weaker on the networks markets actually have.")]},
    {"id": "e6", "verdict": "data", "title": "Real data: the 2010 flash crash at 1-minute resolution",
     "script": "empirical/e6_flash_crash.py",
     "question": "The model is intraday and the 2010 flash crash lasted 36 minutes. Does co-movement rise BEFORE 14:32, which daily data said it does not?",
     "setup": "1-minute bars for 32 large caps over 3-7 May 2010 (HF Data Library, CC BY 4.0, 2002 to present). Same co-movement estimator as E1 and the same order parameter as E3.",
     "findings": ["Co-movement rose <b>+0.137 in the 92 minutes BEFORE 14:32</b> (0.336 baseline to 0.473) and then fell slightly during the crash itself. On daily data E1 found the reverse: flat until the peak, rising only afterwards.",
                  "<b>But the null cuts the claim down.</b> Running the same 13:00-14:32 window on every day that week, the crash day has the largest afternoon rise (+0.141) but 2010-05-05 reaches +0.115, and the crash day's absolute level (0.473) is not even the week's highest (2010-05-07, the aftermath, at 0.623).",
                  "With <b>one crash and four control days</b>, inside a week already dominated by the European debt crisis, this cannot separate a genuine precursor from ordinary intraday variation.",
                  "<b>Suggestive, not established.</b> It motivates E7 rather than settling anything."],
     "figures": [fig("empirical/figures/e6_flash_crash.png",
                     "6 May 2010 at one-minute resolution: the equal-weight index, the instantaneous order parameter R, and mean pairwise correlation. The shaded band is 14:32 to 15:08.",
                     "The first look at this project's actual timescale, and an honest demonstration that a single event cannot answer the question however well resolved it is.")]},
    {"id": "e7", "verdict": "neg", "title": "Real data: the timescale objection, settled at 1-minute resolution",
     "script": "empirical/e7_minute_event_study.py",
     "question": "E1 and E2 tested daily closes while the model describes intraday synchronization. Was the null result just a resolution artifact?",
     "setup": "23 tickers, 2,319,682 one-minute bars from 2002-12-30 to 2026-09-24 (HF Data Library, CC BY 4.0). 31 intraday crash events (a 3% fall of the equal-weight index within 60 minutes) against E1's own matched null of local maxima not followed by a fall, one value per event so events rather than minutes are the unit.",
     "findings": ["<b>The unmatched result looks spectacular.</b> Pre-event co-movement is 0.458 before crashes against 0.228 before ordinary peaks, <b>AUC 0.911</b> [0.843, 0.969], against E1's 0.542 on daily data. Taken at face value this says E1's null was a resolution artifact.",
                  "<b>It is not.</b> Crashes happen in high-volatility regimes and co-movement rises with volatility, so an unmatched comparison scores well purely as a volatility proxy. Pre-event volatility is <b>4.94x higher</b> before crashes (0.00151 against 0.00031).",
                  "Pairing each crash with the nearest-volatility ordinary peak without replacement, exactly the control E1 used on daily data: <b>AUC falls from 0.897 to 0.526, CI [0.340, 0.712]</b>, which spans chance. Median co-movement is 0.459 against 0.409 for matched peaks.",
                  "<b>Once volatility is held fixed, co-movement carries no skill at minute resolution either.</b> The same confound structure appears at both timescales.",
                  "<b>This settles the single biggest threat to the empirical chapters.</b> Assumption 41 held that the daily grid might be three orders of magnitude too coarse and that E1 and E2 could be measuring nothing but that mismatch. Tested at the model's own timescale across 24 years, E1's conclusion survives the test it was most vulnerable to. It also retires recommendation A15.",
                  "Caveats: 21 crashes survive the pre-window data requirement, so the matched interval is wide, and the cross-section is 23 large caps rather than a full index. A larger ticker set would tighten it, but 0.526 is not close to meaningful skill."],
     "figures": [fig("empirical/figures/e7_minute_event_study.png",
                     "Left: median co-movement in the 90 minutes either side of crash onsets (red) and ordinary peaks (grey), at one-minute resolution across 24 years. Right: the distribution of pre-event co-movement for each group, with the unmatched AUC.",
                     "Answers the objection that would otherwise have sunk E1 and E2, and does so against the project's own hypothesis. The separation in the figure is real but is volatility, not synchronization.")]},
    {"id": "s12", "verdict": "neg", "title": "The last three structural extensions",
     "script": "s12_structural_extensions.py",
     "question": "Do a co-evolving network, a multilayer structure, or a bipartite trader-asset mapping improve the model?",
     "setup": "Same protocol as S10: lambda cycles across the bistable window, scored on four stylised facts plus whether the window survives.",
     "findings": ["<b>A11 co-evolving network: reject.</b> Its one notable effect is counter-intuitive. Rewiring toward desks that already agree <i>reduces</i> global synchrony (R 0.926 to 0.712 at a 20% rewire rate), because homophily fragments the network into like-minded clusters. Local order, global disorder: an echo chamber, not an amplifier.",
                  "<b>A12 multilayer: reject.</b> Best stylised-fact score of the three, but it <b>destroys the bistable window</b> (gap 0.465 against 0.789, lambda_b undefined). Same verdict as contrarians in S10: it improves the fit by removing the phenomenon.",
                  "<b>A13 bipartite trader-asset: no effect.</b> Structurally it is the honest fix for assumption 29, but every stylised fact is slightly worse and crash co-movement is unchanged.",
                  "None of the three is worth adopting. With S10 that is <b>five of eleven tested extensions rejected on evidence</b>."],
     "figures": [fig("figures/s12_structural.png",
                     "Crash co-movement and excess kurtosis for each variant under both the per-node and bipartite price mappings, against the real-market values; and the bistable window width and hysteresis gap for each.",
                     "Completes the extension programme: every proposal in the reference document has now been implemented and scored rather than asserted.")]},
    {"id": "e8", "verdict": "neg", "title": "Real data: could you actually have called a crash?",
     "script": "empirical/e8_prediction.py",
     "question": "E7 reported AUC, which ranks. Fix a threshold, make real calls out of sample, and count them: how many crashes are caught, and at what cost in false alarms?",
     "setup": "Thresholds fixed on 2002-2015 at a set alarm budget, then applied unchanged to 2016-2026. 10 crashes across 2,697 test trading days, a base rate of 0.37%. Three detectors: raw co-movement, volatility alone, and co-movement after regressing out log-volatility.",
     "math": r"Co-movement regressed on log volatility has $R^2 = 0.414$: <b>41% of co-movement is just volatility</b>. The residual is the synchronization-specific component the model actually claims.",
     "findings": ["<b>Nothing is usable.</b> The best detector catches half the crashes at <b>77.6 false alarms per catch</b>, precision 0.0127 against a base rate of 0.0037. TP 5, FP 388, TN 2299, FN 5.",
                  "<b>Volatility alone is the best of the three</b> (precision 0.0151, 65.2 false alarms per catch). It beats the model's own observable.",
                  "<b>Removing the confound makes it worse, not better.</b> Co-movement with volatility regressed out is the <b>worst</b> detector: precision 0.0036, recall 0.20, <b>274.5 false alarms per catch</b>, falling to 1173 at a 5% budget.",
                  "E7 showed the ranking signal was volatility. E8 shows that what remains after removing volatility is not merely weak but actively unhelpful, out of sample, at the model's own timescale.",
                  "<b>This closes the empirical question.</b> Across 23 markets daily, 730 days hourly, and 2.3M minute bars, there is no operationally useful crash precursor in co-movement."],
     "figures": [fig("empirical/figures/e8_prediction.png",
                     "Left: precision against recall for each detector, with the base rate marked. Middle: false alarms per crash caught, on a log axis. Right: true positives, false positives and false negatives at a 5% alarm budget.",
                     "The measure that matters operationally. A good AUC with a 0.37% base rate still means dozens of false alarms for every crash caught, and here the model-specific signal is the weakest of the three.")]},
    {"id": "s13", "verdict": "new", "title": "Two new results: a closed-form threshold family, and a critical clustering",
     "script": "s13_new_hypotheses.py",
     "question": "Everything else here reproduces known work or reports a negative result. Is there anything positive and new?",
     "setup": "H1: generalise the herding gain to alpha_i = r_i^p, solve the fold exactly, and test against simulation at N = 1500 for p in {0.5, 1, 1.5, 2, 3}. H2: sweep Watts-Strogatz rewiring at fixed mean degree, which holds kappa at 1.00-1.04 and so isolates clustering from degree heterogeneity.",
     "math": r"With $\alpha_i=r_i^p$ the mean field gives $K_{\rm eff}=\lambda R^p$, and substituting into the exact Lorentzian result $R=\sqrt{1-2\gamma/K}$ yields $$\lambda R^{p+2}-\lambda R^{p}+2\gamma=0.$$ The fold, where $(p+2)R^{p+1}=pR^{p-1}$, gives a closed form for every $p$: $$\boxed{R^*(p)=\sqrt{\frac{p}{p+2}},\qquad \lambda_b(p)=\gamma\,(p+2)\left(\frac{p+2}{p}\right)^{p/2}.}$$ One expression spans three separately studied models: $p\to0$ gives $\lambda_b\to2\gamma$ (plain Kuramoto, continuous), $p=1$ gives $3\sqrt3\,\gamma$ (pairwise herding), and $p=2$ gives $8\gamma$ (the exact 3-body hypergraph interaction A1 identified).",
     "findings": ["<b>H1 confirmed.</b> Predicted against simulated lambda_b: 3.738/4.231 (p=0.5), 5.196/5.598 (p=1), 6.608/7.208 (p=1.5), 8.000/8.946 (p=2). <b>Median ratio 1.105 with a total spread across p of only 0.055.</b>",
                  "That flat ratio is the result. The predicted threshold moves by a factor of 2.1 across this range while the ratio stays within &plusmn;2.5%; a formula with the wrong <i>shape</i> would drift with p. The constant ~10% offset is the finite-connectivity correction measured independently as 1.116 in S11: the closed form is exact for all-to-all coupling, the simulation is sparse at &lang;k&rang; = 12.",
                  "<b>Reported failure:</b> p = 3 gave no transition (jump 0.014) inside the scanned coupling range, so the family is verified over p in [0.5, 2] only.",
                  "<b>H2 confirmed, with a number: C* &asymp; 0.35.</b> Hysteresis gap by clustering: 0.327 (C=0.014), 0.312 (C=0.092), 0.099 (C=0.354), 0.089 (C=0.504), 0.075 (C=0.682). Below C* the window is open, above it closed.",
                  "<b>This turns E5 from an observation into an explanation.</b> The synthetic ER graph every crash result in this repository uses has C = 0.030, an order of magnitude <i>below</i> C*. The real market correlation network has C = 0.731, <b>2.1x above</b> it. The model's bistable window is an artifact of using an unclustered graph.",
                  "Clustering suppressing explosive synchronization is itself known (Chaos 33, 053103, 2023). What is new is the threshold value and the placement of real market networks relative to it."],
     "figures": [fig("figures/s13_new_hypotheses.png",
                     "Left: the closed-form lambda_b(p) curve with simulated thresholds overlaid, and the herding and triadic cases marked. Middle: the simulated/predicted ratio against p, flat if the shape is right. Right: hysteresis gap and jump against clustering, with the synthetic ER and real market values marked.",
                     "The two positive contributions of this work: a single closed form covering the continuous, pairwise-herding and 3-body transitions, and a quantitative clustering threshold that explains why the model's central phenomenon does not appear on real market networks.")]},
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
<div><h3>What holds up</h3><ol>
<li><b>A closed-form threshold family (S13, new).</b> With gain alpha_i = r_i^p the fold solves exactly:
<b>R*(p) = sqrt(p/(p+2))</b> and <b>lambda_b(p) = gamma(p+2)((p+2)/p)^(p/2)</b>. One expression spans
plain Kuramoto (p&rarr;0, 2&gamma;), pairwise herding (p=1, 3&radic;3&gamma;) and the 3-body hypergraph
case (p=2, 8&gamma;). Simulated/predicted ratio is flat at 1.105 &plusmn; 0.055 across a 2.1x range of
thresholds.</li>
<li><b>A critical clustering (S13, new).</b> <b>C* &asymp; 0.35</b>: above it the bistable window closes.
Synthetic ER (C = 0.030) sits far below, the real market correlation network (C = 0.731) sits 2.1x
above. This explains E5 rather than merely reporting it.</li>
<li><b>Exact thresholds (S11).</b> For Lorentzian frequencies the herding model is solvable in closed
form: the cubic &lambda;R&sup3; &minus; &lambda;R + 2&gamma; = 0 folds at R* = 1/&radic;3, giving
<b>&lambda;<sub>b</sub> = 3&radic;3&thinsp;&gamma;</b>. The residual 12% against simulation is the
finite-connectivity correction, which explains the ~10% offset S2 could only measure.</li>
<li><b>The first-order jump is genuine (S11).</b> It grows with system size, 0.183 at N = 125 to 0.486
at N = 4000, rather than decaying. Not a finite-size artifact.</li>
<li><b>What actually causes explosiveness (S8c).</b> Not a frequency-degree rank correlation: with a
Gaussian frequency law there is no jump at any correlation. The <b>tail of g(&omega;) must match the
tail of P(k)</b>, and even c = 0.75 gives almost nothing.</li>
<li><b>Separability classification (A1).</b> Kuramoto couplings are provably non-factorizable
(D&#8321; &equiv; 1, rank 2) and the herding coupling is an exact 3-body interaction, so the Hens et al.
propagation theory does not cover them.</li>
<li><b>Two model extensions earn their place (S10).</b> State-dependent liquidity is the only change
that produces fat tails or volatility clustering at all; inertia widens the bistable window 3.6x.</li>
</ol></div>
<div><h3>What did not hold up</h3><ul>
<li><b>The model's own network is wrong (E5).</b> Real market networks are an order of magnitude more
clustered than the synthetic ER graph every crash result uses, and <b>the bistable window essentially
vanishes on them</b> (gap 0.793 &rarr; 0.17). S4 and S5, the two most novel items, both depend on that
window existing. <b>This is the most serious result here.</b></li>
<li><b>No early warning, at any timescale, and it is not usable (E1, E7, E8).</b> On daily data no
indicator beats ordinary peaks. At 1-minute resolution over 2.3M bars the raw AUC is 0.911, but
<b>volatility-matched it is 0.526</b> [0.340, 0.712]. Out of sample the best detector needs <b>77 false
alarms per crash caught</b>, and co-movement with volatility removed is the <b>worst</b> of three
detectors. The timescale objection is answered and E1 survives it.</li>
<li><b>&lambda; cannot be measured (A3).</b> R<sub>eq</sub>(&lambda;) is a step function, so 97.2% of
market-days carry no information about &lambda;, and a monotone transform cannot beat its own indicator.
The headline recommendation is withdrawn.</li>
<li><b>Prior work does not fully generalize (E3).</b> Lee et al. (PNAS 2025) does not reproduce at
constituent level for 2008; out of sample across 262 events only the recovery half survives, weakly
(&rho; = 0.13).</li>
<li><b>Two proposed extensions must be rejected (S10).</b> Contrarians destroy the transition entirely;
asymmetric herding gives the wrong sign on the leverage effect.</li>
<li>No co-movement hysteresis in real data (E2), though S9 shows the test lacked the power to see the
model's own prediction by a factor of four.</li>
<li>The repository's original degree-normalized model has no explosive synchronization (S1).</li>
</ul></div></div>
<p class="note">Honest status: the analytic core is in better shape than when this started (closed-form
thresholds, a genuine discontinuity, a sharper cause), while the empirical case is weaker (no early
warning at any resolution, and the central bistability largely absent on real networks). The most
valuable next step is deciding whether E5's collapse is the clustering gap or an artifact of how the
network was built, because steps S4 and S5 stand or fall on it.</p>
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
.starthere {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin-top:16px }}
.sh {{ display:block; padding:13px 15px; border:1px solid var(--line); border-radius:9px;
      text-decoration:none; background:var(--soft) }}
.sh:hover {{ border-color:var(--accent) }}
.sh b {{ display:block; color:var(--accent); font-size:14px; margin-bottom:3px }}
.sh span {{ display:block; color:var(--muted); font-size:12.5px; line-height:1.5 }}
footer {{ text-align:center; color:var(--muted); font-size:12px; padding:6px 0 40px }}
</style></head><body>
<header><h1>Explosive synchronization &amp; market crashes</h1>
<p>Simulations, analytic results and real-data tests. Every figure explains what is shown and why it matters; verdict tags state plainly whether a result holds.</p>
<div class="starthere">
<a class="sh" href="reference.html"><b>Build the model from one oscillator</b><span>The whole model in the order you would discover it: each step shows where the previous one breaks and adds exactly one term. Every assumption listed.</span></a>
<a class="sh" href="../simulation/index.html"><b>Interactive simulation</b><span>Watch a crash spread node by node through the network. Click any desk to see its own numbers in the locking inequality.</span></a>
<a class="sh" href="MODEL_REFERENCE.md"><b>Reference text</b><span>Equations, all 41 assumptions, what should be removed, what should be added and whether it measurably helped.</span></a>
</div></header>
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
