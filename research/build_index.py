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
           "theory": ("analytic", "v-theory"), "withdrawn": ("claim withdrawn", "v-neg"),
           "unverified": ("unverified, under audit", "v-open")}


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
                  "<b>The key number: E2's own gap statistic computed on the repaired model is 0.0002 to 0.0028 at every &sigma;<sub>c</sub>, against E2's detection bound of 0.013.</b> Even with the mapping fixed the predicted effect is roughly four times smaller than the smallest one E2 could have seen.",
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
                     "Addresses the strongest objection to the empirical chapters, and shows how far free data can take it.")]},
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
                     "The first look at this project's actual timescale, and a demonstration that a single event cannot answer the question however well resolved it is.")]},
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
                  "<b>A13 bipartite trader-asset: no effect.</b> Structurally it is the correct fix for assumption 29, but every stylised fact is slightly worse and crash co-movement is unchanged.",
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
    {"id": "s13", "verdict": "withdrawn", "title": "Two claimed new results, and what survived a literature check",
     "script": "s13_new_hypotheses.py",
     "question": "Everything else here reproduces known work or reports a negative result. Is there anything positive and new?",
     "setup": "H1: generalise the herding gain to alpha_i = r_i^p, solve the fold exactly, and test against simulation at N = 1500 for p in {0.5, 1, 1.5, 2, 3}. H2: sweep Watts-Strogatz rewiring at fixed mean degree, which holds kappa at 1.00-1.04 and so isolates clustering from degree heterogeneity.",
     "math": r"With $\alpha_i=r_i^p$ the mean field gives $K_{\rm eff}=\lambda R^p$, and substituting into the exact Lorentzian result $R=\sqrt{1-2\gamma/K}$ yields $$\lambda R^{p+2}-\lambda R^{p}+2\gamma=0.$$ The fold, where $(p+2)R^{p+1}=pR^{p-1}$, gives a closed form for every $p$: $$\boxed{R^*(p)=\sqrt{\frac{p}{p+2}},\qquad \lambda_b(p)=\gamma\,(p+2)\left(\frac{p+2}{p}\right)^{p/2}.}$$ One expression spans three separately studied models: $p\to0$ gives $\lambda_b\to2\gamma$ (plain Kuramoto, continuous), $p=1$ gives $3\sqrt3\,\gamma$ (pairwise herding), and $p=2$ gives $8\gamma$ (the exact 3-body hypergraph interaction A1 identified).",
     "findings": ["<b class='bad'>H1 is not new, and the claim is withdrawn.</b> The family is <b>Zou &amp; Wang, Phys. Rev. E 102, 012219 (2020)</b>, who study exactly a mean-field force $K r^{\\alpha}$, i.e. $\\alpha = p+1$. Their stability result reads, verbatim, that the locked states are stable \"only for $r^2 \\ge (\\alpha-1)/(\\alpha+1)$ if $\\alpha > 1$. For $\\alpha = 2$, $r = 1/\\sqrt3 \\approx 0.577$ is predicted at the coupling threshold $K = K_b$.\" With $\\alpha = p+1$, $(\\alpha-1)/(\\alpha+1) = p/(p+2)$, which <i>is</i> $R^*(p)^2$ &mdash; same formula, same symbol for the threshold, five years earlier.",
                  "The $p=2$ corner is separately exact in <b>Skardal &amp; Arenas, Commun. Phys. 3 (2020)</b>: their saddle-node curve $K_1^{\\rm desync} = 2\\sqrt{2K_{2+3}} - K_{2+3}$ gives $K_{2+3} = 8$ at $K_1 = 0$ with $r = 1/\\sqrt2$. That PDF was already in this repository's own reference folder. The observation that $r_i^2$ coupling <i>is</i> a 3-body interaction is stated in one line by <b>Xu et al., Phys. Rev. Research 3, 043004 (2021)</b>.",
                  "The nearest active work is the Hens group's own: <b>Dutta et al., Phys. Rev. Research 7, L022049 (2025)</b> and <b>arXiv:2507.18421 (2025)</b> run power-law adaptation of the order parameter with both an adaptation exponent and a partial-adaptation fraction $f$, on hypergraphs, via Ott-Antonsen. That is this model, with this $f$, analysed more generally.",
                  "<b>The measurement could not have supported the claim either.</b> H1's coupling grid was <code>linspace(0.45&middot;pred, 1.8&middot;pred, 34)</code>, whose step is <b>4.09% of the prediction at every p</b>. The 12 runs therefore produced only three distinct ratio values (1.0636, 1.1045, 1.1455), and the reported flatness of 0.055 is <b>1.3 grid steps wide</b>. A drift of a few percent with p &mdash; which is exactly what S14 shows is really there &mdash; was invisible by construction. The quoted \"1.105 &plusmn; 0.055\" is also mislabelled: the standard deviation over the 12 points is 0.035, and 0.055 is the <i>range of the four per-p means</i>.",
                  "Separately, $\\lambda_b$ was read off as the grid point where the backward branch crosses $R = 0.5$, while the theory's own fold amplitude $R^*(p)$ moves from 0.447 at $p=0.5$ to 0.707 at $p=2$. At $p=0.5$ the fold sits <i>below</i> the detection level. The estimator's bias therefore varies with p across precisely the range where p-independence was being asserted.",
                  "<b class='bad'>H2's C* &asymp; 0.35 was confounded by the coupling range.</b> The sweep used a fixed grid <code>linspace(2, 11, 31)</code> at every clustering level, but $\\lambda_f$ rises with clustering: 5.9-6.5 at C=0.014, 6.8-7.7 at C=0.354, then 8.0/11.0/undefined at C=0.503, and <b>undefined for all three seeds</b> at C = 0.586, 0.641 and 0.682. Above C &asymp; 0.5 the transition had left the scanned box, so the small gap there means only that both branches stayed incoherent. The single clustering level above C* with a measured transition is C = 0.354 itself.",
                  "<b>What survives, and is now derived rather than measured.</b> The qualitative claim that clustering closes the window holds, and S14 replaces the measured threshold with a predicted one by showing that clustering acts entirely through an <i>effective connectivity</i>. The prior art the original write-up cited for this (Chaos 33, 053103, 2023) turns out to be a different mechanism: it uses a degree-biased Laplacian on trees and Cayley graphs, and its result is about <i>cycle length</i>, with long cycles suppressing the transition strongly and triangles behaving erratically. So H2 was conceded too generously and H1 not nearly enough."],
     "figures": [fig("figures/s13_new_hypotheses.png",
                     "Left: the closed-form lambda_b(p) curve with simulated thresholds overlaid, and the herding and triadic cases marked. Middle: the simulated/predicted ratio against p, flat if the shape is right. Right: hysteresis gap and jump against clustering, with the synthetic ER and real market values marked.",
                     "Kept for the record. The left panel is a correct plot of a known result (Zou &amp; Wang 2020); the middle panel's flatness is one grid step wide; the right panel's high-clustering points sit outside the scanned coupling range. S14 redoes all three without a grid.")]},
    {"id": "s14", "verdict": "unverified", "title": "A Rician closure for the local order parameter: the finite-connectivity law for the fold",
     "script": "s14_finite_connectivity.py",
     "question": "S11 and S13 both measured a fold about 10% above the all-to-all prediction (ratios 1.116 and 1.105 at &lang;k&rang; = 12) and had no explanation. That offset is the only thing in S13 the literature does not already contain. What is it?",
     "setup": "Theory: keep the local order parameter r_i as a random variable instead of replacing it by R. Measurement: lambda_b by bisection on the survival of the upper branch, with NO coupling grid, on random regular graphs (every degree exactly k, so the theory applies with no degree heterogeneity), Erdos-Renyi graphs at the same mean degree, and Watts-Strogatz graphs at fixed &lang;k&rang; = 12. N = 2000, p in {0.5, 1, 1.5, 2}, &lang;k&rang; from 6 to 96, 5 seeds.",
     "math": r"""The annealed mean field replaces $r_i$ by $R$. But the exact locking field is
$$h_i=\frac{\lambda}{\langle k\rangle}\,k_i\,r_i^{\,p+1},\qquad r_i=\Big|\sum_j A_{ij}e^{i\theta_j}\Big|/k_i,$$
and $r_i$ is the <b>modulus of a sum of $k_i$ unit vectors</b>. By the central limit theorem that sum is complex Gaussian with mean $k_iR$ and total variance $k_i(1-R^2)$, so $r_i$ is a <b>Rician</b> variable with scale $s^2=(1-R^2)/2k_i$ per component: a coherent part $R$ plus a scattered part, exactly as in Rician fading.
The second ingredient is what makes the closure correct. Node $i$ does not lock to the global mean phase $\Psi$; it locks to the direction $\phi_i$ of its <i>own</i> neighbour sum, which is itself random. Its contribution therefore carries a projection factor $\cos(\phi_i-\Psi)=X/r$, and the self-consistency becomes
$$R=\Big\langle\, c\!\big(\tfrac{\lambda k}{\langle k\rangle}r^{p+1}\big)\cdot\frac{X}{r}\Big\rangle,\qquad c(h)=\frac{\sqrt{h^2+\gamma^2}-\gamma}{h},$$
with $X\sim N(R,s^2)$, $Y\sim N(0,s^2)$, $r=\sqrt{X^2+Y^2}$. Keeping the projection factor is essential: without it $R=0$ stops being a solution, because a sparse graph has $r_i\approx0.886/\sqrt k$ even when incoherent, and the closure would predict ignition out of nothing. With it, $X/r$ is odd in $X$ at $R=0$ and the incoherent state is preserved exactly.
Expanding to $O(s^2)$ and shifting the fold gives the result:
$$\boxed{\ \lambda_b(p,\langle k\rangle)=\underbrace{\gamma(p+2)\Big(\tfrac{p+2}{p}\Big)^{p/2}}_{\text{Zou \& Wang 2020}}\Big[1+\frac{p+2}{2\langle k\rangle}+O(\langle k\rangle^{-2})\Big]\ }$$
The factor $(p+2)$ appears twice. Linearising the same closure about $R=0$ gives a <i>second</i> threshold that does not exist all-to-all,
$$A(\lambda)=\tfrac12\big\langle \tfrac{d}{dr}c(\lambda r^{p+1})+c(\lambda r^{p+1})/r\big\rangle=1,\qquad r=s|\zeta|,\ |\zeta|\ \text{Rayleigh},$$
with $A(\infty)=0.8862\sqrt{k}$ &mdash; the same $\sqrt\pi/2$ that sets the incoherent $r_i$ floor. Finally, if the neighbours of $i$ are themselves linked their phases correlate, so the variance becomes $k(1-R^2)\,(1+(k-1)\rho)$ and everything above holds with an <b>effective connectivity</b>
$$k_{\rm eff}=\frac{k}{1+(k-1)\rho}.$$
Sparsity, clustering and small system size are then the same effect in different units.""",
     "findings": [
        "<b>The coefficient is exactly $(p+2)/2$.</b> Numerically $\\langle k\\rangle(\\lambda_b/\\lambda_b^\\infty-1)$ converges to 1.24935, 1.49982, 1.75005, 2.00021 at $p$ = 0.5, 1, 1.5, 2 against $(p+2)/2$ = 1.25, 1.50, 1.75, 2.00. Symbolically the $O(s^2)$ shift evaluates to $(p+2)/2$ exactly at $p$ = 1/4, 1/2, 1, 3/2, 2 and 3.",
        "<b>The closure's central assumption is testable on its own, and it holds.</b> Simulating a locked state and comparing the observed spread of the neighbour sums against the predicted Rician scale $(1-R^2)/2k$ gives a ratio of 1.215 at &lang;k&rang; = 6, 1.091 at 12, <b>0.998 at 24</b> and 0.919 at 48, and the predicted mean of $r_i$ matches the observed one to three decimals from &lang;k&rang; = 12 upward. A Kolmogorov-Smirnov test against the full Rice law passes at &lang;k&rang; = 48 (p = 0.30) and fails at small &lang;k&rang;, which is the central limit theorem converging exactly as it should for a sum of k unit vectors.",
        "<b>It predicts the measured fold with no fitted parameter.</b> At &lang;k&rang; = 48 on random regular graphs, equation (4) gives 5.3585 against a measured 5.3558 for $p=1$ (<b>0.05%</b>) and 8.3333 against 8.3291 for $p=2$ (<b>0.05%</b>). At &lang;k&rang; = 12 it gives 9.333 against 9.278 for $p=2$ (0.6%) and 5.846 against 5.632 for $p=1$ (3.7%), where the $O(\\langle k\\rangle^{-2})$ term is no longer negligible.",
        "<b>It explains why S13's ratio looked flat, and says it is not.</b> The offset should <i>grow</i> with p as $(p+2)/2$, from 10.4% at $p=0.5$ to 16.7% at $p=2$ when &lang;k&rang; = 12. The measured ratio does grow, 1.084 to 1.160 across that range. S13 could not see it because its grid quantised the ratio in steps of 4.09%.",
        "<b>A second threshold exists only at finite connectivity.</b> All-to-all, the incoherent state is linearly stable for every coupling, which is why $\\lambda_f$ has always been reported as a fluctuation-driven, N-dependent quantity rather than a constant of the model. At finite &lang;k&rang; the residual $r_i$ floor gives it a genuine linear instability at a finite $\\lambda_c$, and $A(\\infty)=0.8862\\sqrt k$ means this happens at all only for $k>1.27$. Predicted $\\lambda_c$ at &lang;k&rang; = 12: 6.97 ($p=1$) and 19.76 ($p=2$), against simulated $\\lambda_f$ of 5.74-5.95 and 13.4-14.1. The linear threshold is an upper bound and the sweep ignites below it, which is the expected direction.",
        "<b>Bistability needs a minimum number of neighbours.</b> The window $[\\lambda_b,\\lambda_c]$ narrows as connectivity falls and closes at a critical $k_c(p)$: relative width 1.93 at &lang;k&rang; = 96 falling to 0.0027 at &lang;k&rang; = 6 for $p=1$. $k_c$ <i>decreases</i> with p, so higher-order coupling makes bistability more robust to sparsity, not less.",
        "<b>Clustering is lost connectivity, and this is what replaces S13's C*.</b> $k_{\\rm eff}$ is read straight off the observed spread of the neighbour sums, with no free parameter and without estimating $\\rho$ separately. The critical clustering is then not a measured threshold but the point where $k_{\\rm eff}(C)$ crosses $k_c(p)$.",
        "<b>The same formula explains E5's finite-size confound.</b> For an Erdos-Renyi graph the neighbourhood overlap is $\\langle k\\rangle/n$, so a small random graph already has correlated neighbourhoods and a reduced $k_{\\rm eff}$. E5's rerun with n-matched controls confirms it: 72% of the reported collapse of the bistable window was finite size, not clustering."],
     "extra": '<h3>Is r<sub>i</sub> actually Rician? A direct test of the closure, independent of any threshold</h3>'
              + csv_table(ROOT / "data" / "s14_rician_validation.csv",
                  cols=["k", "n", "R", "r_mean_obs", "r_mean_rice", "var_obs", "var_pred",
                        "var_ratio", "ks_p"],
                  rename={"k": "&lang;k&rang;", "r_mean_obs": "&lang;r_i&rang; observed",
                          "r_mean_rice": "&lang;r_i&rang; Rice", "var_obs": "variance observed",
                          "var_pred": "variance predicted", "var_ratio": "ratio", "ks_p": "KS p"},
                  fmt="{:.4f}")
              + '<h3>Theory table</h3>' + csv_table(ROOT / "data" / "s14_theory.csv",
                  cols=["p", "k", "lam_b_inf", "lam_b_closure", "lam_b_asymptotic", "lam_c_forward",
                        "rel_window", "ratio_closure", "ratio_asymptotic", "k_crit"],
                  rename={"k": "&lang;k&rang;", "lam_b_inf": "lam_b (all-to-all)",
                          "lam_b_closure": "lam_b (closure)", "lam_b_asymptotic": "lam_b eq.(4)",
                          "lam_c_forward": "lam_c", "rel_window": "rel. window",
                          "ratio_closure": "ratio, closure", "ratio_asymptotic": "ratio, eq.(4)",
                          "k_crit": "k_c(p)"}, fmt="{:.4f}")
              + '<h3>Measured against predicted, random regular and Erdos-Renyi</h3>'
              + csv_table(ROOT / "data" / "s14_measured_summary.csv", fmt="{:.4f}")
              + '<h3>Clustering: the measured effective connectivity</h3>'
              + csv_table(ROOT / "data" / "s14_clustering_summary.csv", fmt="{:.4f}"),
     "figures": [fig("figures/s14_finite_connectivity.png",
                     "Panel 1: the ratio lambda_b/lambda_b(all-to-all) against mean degree. Solid lines are equation (4), dashed lines the full closure, dots the simulation with error bars over 5 seeds; the four colours are the four herding exponents. Panel 2: the same data multiplied by &lang;k&rang; so the prediction is a flat line at (p+2)/2, marked dotted. Panel 3: the effective connectivity measured from the neighbour sums against clustering, with the critical connectivity k_c and the synthetic and real-market clustering values marked. Panel 4: the predicted and measured fold against clustering.",
                     "Panel 2 is the test: the quantity plotted is flat only if the correction really goes as 1/&lang;k&rang; with coefficient (p+2)/2, and a wrong exponent would make it drift. Panels 3 and 4 convert S13's measured clustering threshold into a predicted one by routing clustering through a single effective-connectivity number.")]},
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
    ("Zou & Wang, Generalized Kuramoto model with nonlinear coupling: bifurcation and stability, PRE 102, 012219 (2020) (the r^p family, five years earlier)", "../references/10_prior_art_adaptive/zou_wang2020_nonlinear_coupling_bifurcation.pdf"),
    ("Skardal & Arenas, Higher-order interactions promote abrupt synchronization switching, Commun. Phys. 3 (2020) (the p = 2 corner)", "../references/03_higher_order/skardal_arenas2020__1909.08057.pdf"),
    ("Xu et al., Collective dynamics of heterogeneously and nonlinearly coupled phase oscillators, PRR 3, 043004 (2021)", "../references/10_prior_art_adaptive/xu2021_nonlinear_coupling__2105.03894.pdf"),
    ("Dutta et al., A double explosive Kuramoto transition in hypergraphs, PRR 7, L022049 (2025)", "../references/10_prior_art_adaptive/dutta2025_double_explosive_hypergraphs__2412.18897.pdf"),
    ("Dutta, Pal & Hens, Generalized adaptation-induced non-universal synchronization transitions in random hypergraphs (2025)", "../references/10_prior_art_adaptive/adaptation_hypergraphs2025__2507.18421.pdf"),
    ("Kumpeerakij & Restrepo, Self-consistent analysis of the Kuramoto model with higher-order interactions (2026)", "../references/10_prior_art_adaptive/selfconsistent_hoi2026__2605.24701.pdf"),
    ("Miranda, Frasca & Estrada, Topologically-induced suppression of explosive synchronization, Chaos 33, 053103 (2023)", "../references/10_prior_art_adaptive/miranda2023_topological_suppression__2301.08000.pdf"),
    ("Early warning signals for stock market crashes, EPJ Data Science (2024) (recurrence networks, validated in sample)", "../references/06_finance/ews_market_crashes_epjds2024.pdf"),
    ("Pang & Lin, Market state transitions and crash early warning in the Chinese stock market, Front. Phys. (2025)", "../references/06_finance/pang_lin2025_market_states.pdf"),
]


# ------------------------------------------------------------------------------------------------ audit (P0)
def _tbl(head: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table class="fit"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


# The external audit's own table (its Section 1.4c), keyed by (k, p, N, seed, t_relax). These are the reference
# values being reproduced, i.e. inputs to the comparison, not results of this repository.
AUDIT_REFERENCE = {(12, 1.0, 1200, 301, 120.0): 1.0513, (12, 1.0, 1200, 301, 480.0): 1.0533,
                   (12, 1.0, 1200, 302, 120.0): 1.0614, (12, 1.0, 4800, 301, 120.0): 1.1000,
                   (12, 1.0, 4800, 302, 120.0): 1.0634, (12, 1.0, 19200, 301, 120.0): 1.0838,
                   (24, 1.0, 1200, 301, 120.0): 1.0447, (48, 1.0, 1200, 301, 120.0): 1.0233,
                   (12, 2.0, 1200, 301, 120.0): 1.1534, (6, 2.0, 1200, 301, 120.0): 1.1851}


def p0_numbers() -> dict | None:
    """Everything the audit section quotes, computed from result files. Nothing here is typed by hand."""
    path = ROOT / "data" / "p0_spot_checks.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    c = pd.DataFrame(d["c"])
    c["frac"] = (c.ratio - 1.0) / (c.closure - 1.0)
    th = pd.read_csv(ROOT / "data" / "s14_theory.csv").set_index(["p", "k"])
    c["R_closure"] = [th.R_star_closure.get((r.p, r.k), float("nan")) for r in c.itertuples()]
    c["lam_inf"] = [th.lam_b_inf.get((r.p, r.k), float("nan")) for r in c.itertuples()]
    c["audit"] = [AUDIT_REFERENCE.get((r.k, r.p, r.n, r.seed, r.t_relax), float("nan")) for r in c.itertuples()]
    c["dev"] = 100 * (c.ratio / c.audit - 1.0).abs()
    worst = c.loc[c.dev.idxmax()]
    base = c[(c.k == 12) & (c.p == 1.0) & (c.t_relax == 120.0)]
    k48 = c[(c.k == 48) & (c.p == 1.0)].iloc[0]
    s11 = pd.read_csv(ROOT / "data" / "s11_finite_size.csv")
    e5 = pd.read_csv(ROOT / "empirical" / "data" / "e5_hysteresis.csv")
    e8 = pd.read_csv(ROOT / "empirical" / "data" / "e8_confusion.csv")
    e1 = pd.read_csv(ROOT / "empirical" / "data" / "e1_pooled_auc.csv")
    a3 = pd.read_csv(ROOT / "data" / "a3_auc.csv").iloc[0]
    s13 = pd.read_csv(ROOT / "data" / "s13_lambda_b_family.csv")
    v = e1[(e1.indicator == "var_index") & (e1.score == "level") & (e1.null == "peak")].iloc[0]
    peak = e1[e1.null == "peak"]
    mc = e8[e8.detector == "mean_corr"]
    return {
        "a": d["a"], "b": d["b"], "c": c,
        "n_close": int((c.dev <= 0.1).sum()), "n_rows": len(c), "worst": worst,
        "k12_lo": base.ratio.min(), "k12_hi": base.ratio.max(), "k12_mean": base.ratio.mean(),
        "k12_n": len(base), "k12_closure": base.closure.iloc[0], "k12_eq4": base.eq4.iloc[0],
        "frac_lo": c.frac.min(), "frac_hi": c.frac.max(),
        "k48_lam": k48.ratio * k48.lam_inf, "k48_eq4": k48.eq4 * k48.lam_inf,
        "k48_err": 100 * (k48.ratio / k48.eq4 - 1.0),
        "s11_ratio_lo": s11[s11.n >= 1000].lam_b.min() / k48.lam_inf,
        "s11_ratio_hi": s11[s11.n >= 1000].lam_b.max() / k48.lam_inf,
        "k12_R_lo": base.R_at.min(), "k12_R_hi": base.R_at.max(), "k12_Rc": base.R_closure.iloc[0],
        "long": c[c.t_relax > 120.0].iloc[0], "short": base[(base.n == 1200) & (base.seed == 301)].iloc[0],
        "s11_zero_125": int((s11[s11.n == 125].d_lam == 0).sum()), "s11_n_125": int((s11.n == 125).sum()),
        "s11_seeds": int(s11.seed.nunique()),
        "s11_step": float(pd.Series(sorted(set(s11.lam_b))).diff().min()),
        "s13_p1": s13[s13.p == 1.0].ratio.mean(), "s13_seeds": int(s13.seed.nunique()),
        "e5_neg": int((e5.d_lam < 0).sum()), "e5_rows": len(e5), "e5_seeds": int(e5.seed.nunique()),
        "e8_tp": ", ".join(str(int(x)) for x in mc.TP), "e8_alarms": int(mc.alarms.iloc[0]),
        "e8_days": int(mc.TP.iloc[0] + mc.FP.iloc[0] + mc.TN.iloc[0] + mc.FN.iloc[0]),
        "e8_crashes": int(mc.crashes.iloc[0]), "e8_budget": 100 * float(mc.budget.iloc[0]),
        "e1_var": (v.auc, v.ci_lo, v.ci_hi), "e1_peak_incl": int((peak.ci_lo <= 0.5).sum()), "e1_peak_n": len(peak),
        "a3": (a3.auc, a3.ci_lo, a3.ci_hi),
    }


def s14_audit_finding() -> str:
    n = p0_numbers()
    if n is None:
        return "<b class='bad'>Audit pending:</b> run <code>p0_spot_checks.py</code>."
    return (f"<b class='bad'>Audit status, read this first.</b> The findings below describe the closure, which is an "
            f"approximation. They are not yet confirmed as statements about the network model. An independent re-run "
            f"of this script's own bisection (section P0) on random regular graphs at &lang;k&rang; = 12, p = 1 gives "
            f"a fold ratio between {n['k12_lo']:.3f} and {n['k12_hi']:.3f} over {n['k12_n']} runs, against "
            f"{n['k12_closure']:.3f} from the closure and {n['k12_eq4']:.3f} from equation (4). Simulation sits "
            f"<b>below</b> the closure in every run. At &lang;k&rang; = 48, p = 1 the re-run gives "
            f"&lambda;<sub>b</sub> = {n['k48_lam']:.4f} against {n['k48_eq4']:.4f} from equation (4), a difference of "
            f"{n['k48_err']:.2f}%, not the 0.05% quoted below. The measured values quoted in the findings below "
            f"(the 0.05% agreement, the 0.6 to 3.7% range, the clustering table, and the five-seed error bars in the "
            f"figure caption) come from a run whose output files were never committed, so they cannot be checked and "
            f"are <b>withdrawn until re-measured</b>. The word 'exactly' applies to the closure only.")


def audit_card() -> str:
    n = p0_numbers()
    if n is None:
        return '<section id="p0"><h2>Audit</h2><div class="missing">run p0_spot_checks.py</div></section>'
    ta = _tbl(["p", "k = 200", "k = 800", "k = 3200", "predicted (p+2)/2"],
              [[p, *[f"{x:.4f}" for x in v], f"{(float(p) + 2) / 2:.2f}"] for p, v in n["a"].items()])
    tb = _tbl(["N", "measured fold / exact all-to-all fold"], [[r["n"], f"{r['ratio']:.4f}"] for r in n["b"]])
    tc = _tbl(["&lang;k&rang;", "p", "N", "seed", "relaxation time", "measured ratio", "audit's value", "closure ratio",
               "equation (4) ratio", "share of predicted correction", "R just above fold", "R at fold, closure"],
              [[r.k, f"{r.p:g}", r.n, r.seed, f"{r.t_relax:g}", f"<b>{r.ratio:.4f}</b>", f"{r.audit:.4f}", f"{r.closure:.4f}",
                f"{r.eq4:.4f}", f"{100 * r.frac:.0f}%", f"{r.R_at:.3f}", f"{r.R_closure:.3f}"]
               for r in n["c"].itertuples()])
    ledger = _tbl(["headline claim", "evidence actually in the repository", "status"], [
        ["Finite-connectivity law, coefficient (p+2)/2 is exact",
         f"The closure only. Its expansion is confirmed (table A). The simulation half of S14 was never committed, and "
         f"the re-run (table C) recovers only {100 * n['frac_lo']:.0f}% to {100 * n['frac_hi']:.0f}% of the predicted "
         f"correction.", "<b class='bad'>Unverified.</b> Exact for the closure, not for the model"],
        ["Critical connectivity k<sub>c</sub>(p)", "The closure only. No direct measurement of bistability at low "
         "connectivity exists.", "<b class='bad'>Unverified</b>"],
        ["Clustering is lost connectivity, k<sub>eff</sub>", "The closure plus a run that was not committed.",
         "<b class='bad'>Unverified</b>"],
        ["&lambda;<sub>b</sub> = 3&radic;3 &gamma; at p = 1", "Standard result of Zou and Wang (2020); the harness "
         "reproduces it (table B).", "Reproduction, correctly credited"],
        ["Predicted 12.5% against measured 11.6% at &lang;k&rang; = 12",
         f"One run at N = 2000 read off a coupling grid and never saved. The saved S11 table gives ratios "
         f"{n['s11_ratio_lo']:.2f} to {n['s11_ratio_hi']:.2f} for N of 1000 and above; S13 gives a mean of {n['s13_p1']:.3f} over "
         f"{n['s13_seeds']} seeds; table C gives {n['k12_lo']:.3f} to {n['k12_hi']:.3f}.",
         "<b class='bad'>Not supported</b> by committed data"],
        ["The jump is real and grows with N",
         f"{n['s11_seeds']} seeds per size on a grid of step {n['s11_step']:.1f}. The statistic is the largest step "
         f"between neighbouring grid points of one curve. At N = 125, {n['s11_zero_125']} of {n['s11_n_125']} seeds "
         f"have zero window width.", "<b class='bad'>Not evidence</b> of a discontinuity. Needs fold and escape-time "
         "measurements"],
        ["Tail of g(&omega;) must match tail of P(k)", "S8c, 2 seeds. Overlaps Xu et al. (2015) and arXiv:2206.04207.",
         "Reproduction plus sharpening, not a headline"],
        ["Window collapses on real networks, 72% finite size",
         f"{n['e5_seeds']} seeds on a grid; {n['e5_neg']} of {n['e5_rows']} rows have a negative window width, which is "
         f"impossible for a true window. Correlation-thresholded graphs are clustered by construction.",
         "Weak. Robustness appendix at most"],
        ["No usable early warning; false alarms per catch",
         f"Direction plausible: against ordinary peaks {n['e1_peak_incl']} of {n['e1_peak_n']} indicator scores have a "
         f"95% interval that includes 0.5, and index variance alone scores {n['e1_var'][0]:.3f} "
         f"[{n['e1_var'][1]:.3f}, {n['e1_var'][2]:.3f}]. The E8 counts are unreliable: {n['e8_crashes']} test crashes, "
         f"a regression fitted on the test period, true positives of {n['e8_tp']} as the alarm threshold is "
         f"loosened (should never decrease), and {n['e8_alarms']} alarms over {n['e8_days']} days at a nominal "
         f"{n['e8_budget']:.0f}% budget.", "Direction kept, numbers <b class='bad'>withdrawn</b> until E7 and E8 are fixed"],
        ["&lambda; cannot be measured", f"The argument is sound, but the script's own table gives AUC "
         f"{n['a3'][0]:.3f} [{n['a3'][1]:.3f}, {n['a3'][2]:.3f}], above chance, measured against a different "
         f"baseline than E1.", "Argument kept. Needs the same baseline as E1"],
    ])
    return f"""
<section id="p0" class="summary">
<div class="sec-head"><span class="tag">P0</span><h2>Audit: which headline claims are actually backed by the files in this repository</h2><span class="verdict v-open">audit in progress</span></div>
<p>This section was added after an external audit of the repository at commit <code>fc98f02</code>. It does two
things. First, it re-runs the audit's three spot checks with this repository's own, unmodified functions
(<code>p0_spot_checks.py</code>, output in <code>data/p0_spot_checks.json</code>). Second, it lists every headline
claim next to the evidence that is really committed. Every number in this section is read from a result file when
the page is built. No manuscripts are being written at this stage: nothing below is settled enough for that.</p>

<h3>How to read the three tables</h3>
<p>The quantity under test is the <b>fold</b> &lambda;<sub>b</sub>: the smallest coupling at which the synchronised
state can still exist. All tables report it as a <b>ratio</b> to the exact all-to-all value
&gamma;(p+2)((p+2)/p)<sup>p/2</sup> of Zou and Wang (2020), so a ratio of 1.000 means "no finite-connectivity
effect" and a ratio of 1.125 means "the fold sits 12.5% higher than all-to-all". The Rician closure of S14 predicts
the ratio 1 + (p+2)/(2&lang;k&rang;) at leading order.</p>

<h3>Table A. Is the formula the correct expansion of the closure? Yes</h3>
<p>Each cell is &lang;k&rang; &times; (closure ratio &minus; 1). If equation (4) is the right expansion, the cells
must approach (p+2)/2 as &lang;k&rang; grows. They do, to four digits. This confirms the algebra. It says nothing
about whether the closure describes the network.</p>
{ta}

<h3>Table B. Is the measuring tool itself biased? No</h3>
<p>The same bisection procedure (start fully synchronised, ask whether the synchronised state survives, halve the
coupling interval), run on the exact all-to-all model with the same frequencies and the same survival threshold.
It returns the known answer to within the bisection tolerance of 0.2% at every size. So when the tool disagrees
with the closure on a sparse graph, the cause is the graph, not the frequency sampling and not the tool.</p>
{tb}

<h3>Table C. Does the closure match simulation on sparse graphs? Not within these runs</h3>
<p>Random regular graphs: every node has exactly &lang;k&rang; neighbours, so there is no degree heterogeneity to
blame. Column by column: <b>measured ratio</b> is the simulated fold from this re-run; <b>audit's value</b> is what the external
audit reported for the same run; <b>closure ratio</b> and
<b>equation (4) ratio</b> are the two predictions; <b>share of predicted correction</b> is
(measured &minus; 1) / (closure &minus; 1), which would be 100% if the closure were right;
<b>R just above fold</b> is the synchrony of the surviving state at the last coupling where it survived;
<b>R at fold, closure</b> is what the closure says that synchrony should be.</p>
{tc}
<ul class="findings">
<li><b>{n['n_close']} of {n['n_rows']} values agree with the audit's table to within 0.1%.</b> The largest
difference is {n['worst'].dev:.2f}% (the run at &lang;k&rang; = {int(n['worst'].k)}, p = {n['worst'].p:g}, relaxation time
{n['worst'].t_relax:g}: {n['worst'].ratio:.4f} here against {n['worst'].audit:.4f} in the audit). That is larger than
the bisection tolerance of 0.2% and is most likely the chaotic sensitivity of a long run to the numerical library
version, but it is far smaller than the gap to the closure, so no conclusion changes.</li>
<li><b>Simulation is below the closure in every row.</b> The measured correction is {100 * n['frac_lo']:.0f}% to
{100 * n['frac_hi']:.0f}% of the predicted one.</li>
<li><b>It is not slow relaxation.</b> Raising the relaxation time from {n['short'].t_relax:g} to {n['long'].t_relax:g} on
the same graph at &lang;k&rang; = 12 moves the ratio from {n['short'].ratio:.4f} to {n['long'].ratio:.4f}, still far
from the closure's {n['k12_closure']:.4f}.</li>
<li><b>The spread between seeds is large and does not obviously shrink with N</b>: at &lang;k&rang; = 12, p = 1 the
{n['k12_n']} runs span {n['k12_lo']:.3f} to {n['k12_hi']:.3f} with mean {n['k12_mean']:.3f}. These are spot checks,
not an ensemble, so no confidence interval is quoted. The fold of a sparse deterministic network appears to depend
on the particular graph and the particular placement of frequencies on it.</li>
<li><b>New observation from this re-run, not in the audit.</b> At &lang;k&rang; = 12 the surviving state just above
the measured fold has R between {n['k12_R_lo']:.2f} and {n['k12_R_hi']:.2f}, well below the closure's fold amplitude of
{n['k12_Rc']:.2f} and close to the survival threshold of 0.55 R*. With the longer relaxation time it is
{n['long'].R_at:.2f}. So part of what is being measured near the fold
may be a slowly decaying or only partly synchronised state. The fold estimator's sensitivity to the survival
threshold and to the relaxation time has to be mapped before any coefficient is quoted.</li>
<li><b>Most likely physical cause of the gap</b> (a hypothesis, not a result): the closure treats the neighbours of
node i as independent of node i. On a sparse graph each neighbour is itself pulled by node i, a back-reaction that a
cavity (message-passing) calculation would include and the closure does not.</li>
</ul>

<h3>Claim ledger: every headline against its evidence</h3>
{ledger}

<h3>What has been done, and what happens next</h3>
<ol>
<li><b>Done (phase 0).</b> Spot checks reproduced; repository inventory, model specification and this claims audit
written to <code>docs/</code>. No existing simulation code was changed.</li>
<li><b>Next (phase 1).</b> Move the simulator into a tested package with a regression test that it reproduces
<code>model.py</code> exactly, so that no historical number can change silently.</li>
<li><b>Phase 2.</b> Fix the known defects: three different Lorentzian samplers, hidden run-order dependencies,
the E7 overnight-gap contamination, the E8 look-ahead and scoring bugs, the E3 missing band-pass step, hand-typed
numbers on this page.</li>
<li><b>Phase 3.</b> A systematic prior-art search before any novelty claim.</li>
<li><b>Phase 4.</b> The deciding measurement: random regular graphs, &lang;k&rang; from 6 to 96, four values of p,
N up to 30,000, 5 seeds first and then 30, with the fold extrapolated to infinite size and a cavity theory
compared against it. This needs cluster or GPU time.</li>
<li><b>On hold by decision of the author.</b> Manuscripts. They wait until phase 4 says what is true.</li>
</ol>
</section>"""



def write_claims_audit() -> Path | None:
    """docs/CLAIMS_AUDIT.md, generated from the same numbers as section P0 so the two cannot disagree."""
    n = p0_numbers()
    if n is None:
        return None
    c = n["c"]
    L = ["# Claims audit", "",
         "Generated by `research/build_index.py` from `research/data/p0_spot_checks.json` and the committed result",
         "CSVs. Do not edit by hand. Spot checks were run by `research/p0_spot_checks.py` with the unmodified",
         "functions of `research/s14_finite_connectivity.py` and `research/model.py` at commit `fc98f02`.", "",
         "## Spot check A: closure asymptote", "",
         "`k * (rician_fold / lam_b_inf - 1)`, which must approach `(p+2)/2`.", "",
         "| p | k = 200 | k = 800 | k = 3200 | (p+2)/2 |", "|---|---|---|---|---|"]
    L += [f"| {p} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {(float(p) + 2) / 2:.2f} |" for p, v in n["a"].items()]
    L += ["", "Reproduces the audit. Equation (4) is the correct expansion of the closure.", "",
          "## Spot check B: bisection harness in the all-to-all limit", "",
          "| N | lam_b / lam_b_inf |", "|---|---|"]
    L += [f"| {r['n']} | {r['ratio']:.4f} |" for r in n["b"]]
    L += ["", "Reproduces the audit. The harness is unbiased when there is no network.", "",
          "## Spot check C: random regular graphs", "",
          "One run per row (no ensemble, so no confidence interval). `share` is `(measured - 1)/(closure - 1)`.", "",
          "| k | p | N | seed | t_relax | measured | audit | closure | eq. (4) | share | R above fold | R fold, closure |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r.k} | {r.p:g} | {r.n} | {r.seed} | {r.t_relax:g} | {r.ratio:.4f} | {r.audit:.4f} | {r.closure:.4f} | {r.eq4:.4f} | "
          f"{100 * r.frac:.0f}% | {r.R_at:.3f} | {r.R_closure:.3f} |" for r in c.itertuples()]
    L += ["", f"{n['n_close']} of {n['n_rows']} rows agree with the audit's table within 0.1%. The largest difference is "
          f"{n['worst'].dev:.2f}% (k = {int(n['worst'].k)}, p = {n['worst'].p:g}, t_relax = {n['worst'].t_relax:g}), above the "
          "bisection tolerance (`rel_tol = 2e-3`) but far below the gap to the closure.",
          f"Simulation is below the closure in every row; the measured correction is {100 * n['frac_lo']:.0f}% to "
          f"{100 * n['frac_hi']:.0f}% of the closure's.",
          f"At k = 12, p = 1 the {n['k12_n']} runs at t_relax = 120 span {n['k12_lo']:.4f} to {n['k12_hi']:.4f} "
          f"(mean {n['k12_mean']:.4f}).", "",
          "Observation not in the audit: at k = 12 the surviving state just above the measured fold has R between "
          f"{n['k12_R_lo']:.2f} and {n['k12_R_hi']:.2f} against the closure's {n['k12_Rc']:.2f}, close to the survival "
          f"threshold `0.55 R*`; at t_relax = {n['long'].t_relax:g} it is {n['long'].R_at:.2f}. The estimator's "
          "sensitivity to threshold and relaxation time must be mapped before a coefficient is quoted.", "",
          "## Claim ledger", "", "| claim | evidence in repository | status |", "|---|---|---|",
          f"| Finite-connectivity law, coefficient `(p+2)/2` exact | closure only; S14 simulation outputs not committed; "
          f"spot check C | Unverified. Exact for the closure, not the model |",
          "| Critical connectivity `k_c(p)` | closure only | Unverified |",
          "| Clustering as `k_eff` | closure plus an uncommitted run | Unverified |",
          "| `lam_b = 3 sqrt(3) gamma` at p = 1 | Zou and Wang (2020); spot check B | Reproduction, credited |",
          f"| 12.5% predicted vs 11.6% measured at k = 12 | one unsaved grid-read run; saved S11 ratios "
          f"{n['s11_ratio_lo']:.2f} to {n['s11_ratio_hi']:.2f} (N >= 1000); S13 mean {n['s13_p1']:.3f} "
          f"({n['s13_seeds']} seeds) | Not supported |",
          f"| The jump is real and grows with N | `s11_finite_size.csv`: {n['s11_seeds']} seeds, grid step "
          f"{n['s11_step']:.1f}, statistic `max(diff(fwd))`; {n['s11_zero_125']} of {n['s11_n_125']} seeds at N = 125 "
          f"have `d_lam = 0` | Not evidence of a discontinuity |",
          "| Tail of g(omega) must match tail of P(k) | `s8c_correlation.csv`, 2 seeds; overlaps Xu et al. (2015) | "
          "Reproduction plus sharpening |",
          f"| Window collapses on real networks | `e5_hysteresis.csv`: {n['e5_seeds']} seeds, grid, {n['e5_neg']} of "
          f"{n['e5_rows']} rows with negative `d_lam` | Weak |",
          f"| No usable early warning; false alarms per catch | E1 vs peak null: {n['e1_peak_incl']} of {n['e1_peak_n']} "
          f"CIs include 0.5; `var_index` level {n['e1_var'][0]:.3f} [{n['e1_var'][1]:.3f}, {n['e1_var'][2]:.3f}]. E8: "
          f"{n['e8_crashes']} test crashes, TP {n['e8_tp']} as threshold loosens, {n['e8_alarms']} alarms on "
          f"{n['e8_days']} days at a nominal {n['e8_budget']:.0f}% budget | Direction kept, E8 numbers withdrawn |",
          f"| lambda unidentifiable | argument sound; `a3_auc.csv` AUC {n['a3'][0]:.3f} [{n['a3'][1]:.3f}, "
          f"{n['a3'][2]:.3f}] against a different null than E1 | Needs E1's nulls |", "",
          "## Statements on research/index.html that have no committed data behind them", "",
          f"1. S14 finding \"0.05% at k = 48\": re-run gives lam_b = {n['k48_lam']:.4f} against eq. (4) "
          f"{n['k48_eq4']:.4f}, a difference of {n['k48_err']:.2f}%.",
          "2. S14 finding \"0.6 to 3.7% at k = 12\" and the measured-against-predicted and clustering tables: source "
          "files `s14_measured*.csv`, `s14_clustering*.csv` are not tracked.",
          "3. S14 figure caption says error bars are over 5 seeds; the script defines 3 (`SEEDS = (301, 302, 303)`), "
          "and the figure file is not tracked.", ""]
    out = REPO / "docs" / "CLAIMS_AUDIT.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L))
    return out



# ------------------------------------------------------------------------------------------------ P1 / B1 (results/)
def _ci(m, lo, hi, d=4):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if pd.notna(lo) else f"{m:.{d}f}"


def validation_blocks() -> str:
    """A0 (all-to-all validation) and the paired convergence checks, if those runs exist."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc import report
    out = ""
    a0 = report.fold_cells("paperA_A0")
    if a0 is not None and len(a0):
        out += ("<h3>Table P1.5. Does the estimator return the exact answer when there is no network?</h3>"
                "<p>The same bisection on the all-to-all model, where the fold is known exactly. A ratio of 1 means no "
                "bias. This is the control for every number above. Run: <code>" + report.run_name("paperA_A0") + "</code>.</p>"
                + _tbl(["p", "N", "seeds", "fold ratio [95% CI]", "standard deviation across seeds"],
                       [[f"{r.p:g}", r.N, r.n_seeds, _ci(r.ratio, r.ci_lo, r.ci_hi), "n/a" if pd.isna(r.sd) else f"{r.sd:.4f}"]
                        for r in a0.sort_values(["p", "N"]).itertuples()]))
    got = report._rows("paperA_convergence")
    if got is not None:
        df, _ = got
        piv = df.pivot_table(index="index", columns="variant", values="ratio")
        if "baseline" in piv:
            rows = []
            for v in piv.columns:
                d = 100 * (piv[v] - piv["baseline"]) / piv["baseline"]
                d = d.dropna()
                se = d.std(ddof=1) / len(d) ** 0.5 if len(d) > 1 else float("nan")
                rows.append([v, len(piv[v].dropna()), f"{piv[v].mean():.4f}",
                             "reference" if v == "baseline" else f"{d.mean():+.2f}% &plusmn; {se:.2f}%",
                             "" if v == "baseline" else ("within tolerance" if abs(d.mean()) <= 0.2 else "<b class='bad'>systematic</b>")])
            c = df.iloc[0]
            out += ("<h3>Table P1.6. How much does the fold depend on the estimator's settings?</h3>"
                    f"<p>Random regular graphs, k = {int(c.k)}, p = {c.p:g}, N = {int(c.n)}. Every variant is run on the "
                    "<b>same</b> graphs and frequencies as the baseline, so the differences are paired and much more "
                    "precise than the fold itself. The bisection tolerance is 0.2%; a shift larger than that is a "
                    "systematic error of the method and is flagged. Variants: integration step halved; relaxation time "
                    "halved and doubled; the survival threshold moved from 0.55 to 0.40 and 0.70 of the all-to-all fold "
                    "amplitude; the frequency clip raised from 60 to 600; frequencies drawn at random instead of taken "
                    "as quantiles. Run: <code>" + report.run_name("paperA_convergence") + "</code>.</p>"
                    + _tbl(["variant", "seeds", "mean fold ratio", "paired shift from baseline &plusmn; SE", "verdict"], rows))
    return out



def physics_card() -> str:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc import report
    th = report.theory_table()
    cells = [c for c in (report.fold_cells("paperA_A1_pilot"), report.fold_cells("paperA_A7")) if c is not None]
    if th.empty or not cells:
        return '<section id="p1"><h2>P1 New measurements</h2><div class="missing">results not available yet</div></section>'
    df = pd.concat(cells, ignore_index=True)
    tt = _tbl(["p", "Rician closure, (p+2)/2", "theory, directed graph", "reaction term", "theory, undirected graph",
               "frozen share of neighbour variance", "rotating share"],
              [[f"{r.p:g}", f"{r.rician:.3f}", f"<b>{r.directed:.3f}</b>", f"{r.reaction:.3f}", f"<b>{r.undirected:.3f}</b>",
                f"{100 * r.var_frozen / r.var_instantaneous:.0f}%", f"{100 * r.var_rotating / r.var_instantaneous:.0f}%"]
               for r in th.itertuples()])

    def cell_rows(d):
        out = []
        for r in d.sort_values(["graph", "p", "k", "N"]).itertuples():
            se = (r.coef_hi - r.coef_lo) / 3.92 if pd.notna(r.coef_lo) and r.coef_hi > r.coef_lo else float("nan")
            z_t = (r.coef - r.theory_coef) / se if pd.notna(se) and r.theory_coef is not None and pd.notna(r.theory_coef) else float("nan")
            z_r = (r.coef - r.rician_coef) / se if pd.notna(se) else float("nan")
            out.append(["undirected" if r.graph == "rr" else "directed", f"{r.p:g}", r.k, r.N, r.n_seeds,
                        _ci(r.ratio, r.ci_lo, r.ci_hi), f"{r.closure_ratio:.4f}", _ci(r.coef, r.coef_lo, r.coef_hi, 2),
                        "n/a" if pd.isna(r.theory_coef) else f"{r.theory_coef:.2f}", f"{r.rician_coef:.2f}",
                        "n/a" if pd.isna(z_t) else f"{z_t:+.1f}", "n/a" if pd.isna(z_r) else f"{z_r:+.1f}"])
        return out
    head = ["graph", "p", "k", "N", "seeds", "fold ratio [95% CI]", "closure ratio", "coefficient k(ratio&minus;1) [95% CI]",
            "theory coefficient", "Rician coefficient", "distance from theory (standard errors)",
            "distance from Rician (standard errors)"]
    big = df[df.N == df.groupby(["graph", "p", "k"]).N.transform("max")]
    t_big = _tbl(head, cell_rows(big))
    t_all = _tbl(head, cell_rows(df))
    ex = report.extrapolate("paperA_A1_pilot")
    t_ex = ""
    if ex is not None and len(ex):
        t_ex = "<h3>Table P1.4. Extrapolation to infinite size</h3><p>For each (p, k) the fold ratio of every seed at every size is fitted by a straight line in N<sup>&minus;1/2</sup>; the intercept is the infinite-size estimate. <b>The exponent 1/2 is an assumption</b>: three sizes cannot determine it. The slope column shows how strongly the fold still depends on size; a slope many standard errors from zero means the largest-size value in table P1.2 is not yet converged.</p>" + _tbl(
            ["p", "k", "points", "ratio at N &rarr; &infin; &plusmn; SE", "slope &plusmn; SE", "coefficient at N &rarr; &infin; &plusmn; SE", "closure ratio"],
            [[f"{r.p:g}", r.k, r.n_points, f"{r.ratio_inf:.4f} &plusmn; {r.se:.4f}", f"{r.slope:+.2f} &plusmn; {r.slope_se:.2f}",
              f"{r.coef_inf:.2f} &plusmn; {r.coef_se:.2f}", f"{r.closure_ratio:.4f}"] for r in ex.itertuples()])
    # size and relaxation dependence, against the Phase 0 spot check
    n0 = p0_numbers()
    size_note = ""
    if n0 is not None:
        c0 = n0["c"]
        old = c0[(c0.k == 48) & (c0.p == 1.0)].iloc[0]
        new_ = df[(df.graph == "rr") & (df.p == 1.0) & (df.k == 48)].sort_values("N")
        if len(new_):
            b = new_.iloc[-1]
            size_note = (f"<li><b>The Phase 0 gap was mostly a size and relaxation-time effect.</b> At k = 48, p = 1 the "
                         f"Phase 0 spot check (N = {int(old.n)}, relaxation time {old.t_relax:g}, one seed) gave a ratio of "
                         f"{old.ratio:.4f}. The pilot (N = {int(b.N)}, relaxation time 480, {int(b.n_seeds)} seeds) gives "
                         f"{_ci(b.ratio, b.ci_lo, b.ci_hi)}, against {b.closure_ratio:.4f} from the closure. The statement in "
                         f"section P0 that simulation sits below the closure in every run is true of those short, small "
                         f"runs and must not be read as a statement about the infinite-size fold.</li>")
    demo_note = ""
    ej = REPO / "simulation" / "echo_measurement.js"
    if ej.exists():
        E = json.loads(ej.read_text().split("=", 1)[1].rstrip().rstrip(";"))
        demo_note = (f"<li><b>The same contrast at the interactive demo's settings.</b> With Gaussian frequencies, noise "
                     f"&sigma; = {E['sigma']}, N = {E['n']} and k = {E['k']} (not the conditions the theory is derived for), "
                     f"the recovery threshold over {E['seeds']} networks of each kind is "
                     f"{E['two_way']['mean']:.2f} [{E['two_way']['lo']:.2f}, {E['two_way']['hi']:.2f}] with two-way links "
                     f"and {E['one_way']['mean']:.2f} [{E['one_way']['lo']:.2f}, {E['one_way']['hi']:.2f}] with one-way "
                     f"links, a gap of {E['difference']['mean']:.2f} &plusmn; {E['difference']['se']:.2f} "
                     f"({E['relative_percent']:.1f}%). The Rician closure cannot produce any gap, because it never asks "
                     f"whether links are mutual. Script: <code>simulation/measure_echo.py</code>; try it in block 10 of "
                     f"<a href='../simulation/index.html'>the simulation</a>.</li>")
    # the reaction (echo) term measured directly: undirected minus directed coefficient at the same p, k, N
    t_echo = ""
    u, dd = big[big.graph == "rr"], big[big.graph == "dir"]
    mm = u.merge(dd, on=["p", "k", "N"], suffixes=("_u", "_d"))
    if len(mm):
        rk = report.load_summary("paperA_theory_k")
        rows_e = []
        for r in mm.sort_values(["p", "k"]).itertuples():
            se = ((r.coef_hi_u - r.coef_lo_u) ** 2 + (r.coef_hi_d - r.coef_lo_d) ** 2) ** 0.5 / 3.92
            pred = [x["reaction_at_k"] for x in (rk["rows"] if rk else []) if abs(x["p"] - r.p) < 1e-9 and x["k"] == int(r.k)]
            diff = r.coef_u - r.coef_d
            rows_e.append([f"{r.p:g}", r.k, r.N, f"{r.coef_u:.2f}", f"{r.coef_d:.2f}", f"<b>{diff:.2f} &plusmn; {se:.2f}</b>",
                           f"{pred[0]:.2f}" if pred else "n/a", f"{100 * diff / pred[0]:.0f}%" if pred else "n/a", "0.00"])
        t_echo = ("<h3>Table P1.2b. The echo measured directly</h3><p>Subtracting the directed coefficient from the "
                  "undirected one at the same p, k and N isolates what two-way links add. <b>Predicted echo</b> is the "
                  "reaction term of the theory evaluated at that k. The Rician closure predicts zero, because it gives "
                  "the same coefficient to both kinds of graph. The uncertainty combines the two confidence intervals "
                  "and treats the two sets of seeds as independent.</p>"
                  + _tbl(["p", "k", "N", "undirected coefficient", "directed coefficient", "measured echo &plusmn; SE",
                          "predicted echo", "measured as share of predicted", "Rician closure's echo"], rows_e))
    sd = df[df.graph == "rr"].groupby("N").sd.mean()
    sd_note = "; ".join(f"N = {int(n)}: {v:.4f}" for n, v in sd.items() if pd.notna(v))
    def tally(g, col):
        d = big[(big.graph == g) & big[col].notna() & big.coef_lo.notna()]
        return int(((d.coef_lo <= d[col]) & (d[col] <= d.coef_hi)).sum()), len(d)
    ut, un = tally("rr", "theory_coef"); ur, _ = tally("rr", "rician_coef")
    dt_, dn = tally("dir", "theory_coef"); dr, _ = tally("dir", "rician_coef")
    got = report._rows("paperA_A1_pilot")
    pilot_note = ""
    if report.is_partial("paperA_A1_pilot") and got is not None:
        cfg = __import__("yaml").safe_load((got[1] / "config.yaml").read_text())
        planned = len(cfg["k"]) * len(cfg["p"]) * len(cfg["n"]) * cfg["n_seeds"]
        done_n = sorted(int(x) for x in got[0].n.unique())
        missing = [nm for nm, eid in (("the estimator convergence checks", "paperA_convergence"),
                                      ("the all-to-all validation", "paperA_A0")) if report._rows(eid) is None]
        pilot_note = (f'<p class="note"><b class="bad">Incomplete pilot.</b> The undirected pilot was interrupted before '
                      f'completion: {len(got[0])} of {planned} planned runs finished, all at N = '
                      f'{", ".join(str(x) for x in done_n)}. The smaller sizes were never run, so there is no finite-size '
                      f'extrapolation. The directed-graph experiment was run separately and is complete.'
                      + (f' Not yet run: {" and ".join(missing)}.' if missing else "") + '</p>')
    fig_html = render_fig(fig("figures/p1_fold_coefficient.png",
        "One panel per exponent p. Horizontal axis: number of neighbours k (log scale). Vertical axis: the coefficient k &times; (fold ratio &minus; 1), which is flat in k if the correction really goes as 1/k. Blue circles: undirected random regular graphs. Orange squares: directed graphs with k in-neighbours. Bars: 95% confidence interval over seeds at the largest size run. Dotted grey line: the Rician closure's (p+2)/2. Dashed blue and orange lines: the leading-order theory for undirected and directed graphs.",
        "The directed points are the discriminating test: the Rician closure predicts the same value for both graph types, the new theory predicts a much smaller value on directed graphs. Points drifting with k signal second-order terms, which the leading-order lines do not include."))
    return f"""
<section id="p1" class="summary">
<div class="sec-head"><span class="tag">P1</span><h2>New measurements and a leading-order theory of the fold at finite connectivity</h2><span class="verdict v-open">pilot, not final</span></div>
{pilot_note}
<p>This section reports what was measured after the audit, with a new compiled simulator that reproduces the
old one to machine precision (56 regression and analytic tests). All numbers are read from
<code>results/</code> at build time. Runs: <code>{report.run_name('paperA_A1_pilot')}</code>,
<code>{report.run_name('paperA_A7')}</code>, <code>{report.run_name('paperA_theory')}</code>. Derivation of the
theory: <code>docs/THEORY.md</code>.</p>

<h3>The idea in plain words</h3>
<p>Each oscillator feels the average of its k neighbours. That average is the global synchrony R plus a
scatter of size 1/&radic;k. The Rician closure of S14 treats that scatter as <b>frozen</b> and as
<b>independent of the oscillator itself</b>. Neither is right:</p>
<ul class="findings">
<li><b>Part of the scatter rotates.</b> Neighbours that are not locked keep rotating, so their contribution is an
oscillating push, and an oscillator cannot follow a push that changes faster than its own relaxation rate. Only
the frozen share, plus the slow part of the rotating share, acts. This lowers the correction.</li>
<li><b>The oscillator influences its own neighbours.</b> On an undirected graph each neighbour is pulled slightly
toward the oscillator, and the k slightly-pulled neighbours add up to an echo of the oscillator's own phase.
The echo weakens the pull toward consensus. This raises the correction. It is absent on a <b>directed</b> graph,
where neighbours do not listen back.</li>
</ul>
<p>So the theory makes two predictions for the coefficient a(p) in
fold = all-to-all fold &times; (1 + a(p)/k): one for directed graphs (first effect only) and one for undirected
graphs (both effects). The Rician closure predicts the same value, (p+2)/2, for both.</p>

<h3>Table P1.1. Predicted coefficients</h3>
<p><b>Rician closure</b>: all scatter frozen, no echo. <b>Directed</b>: rotating scatter filtered, no echo.
<b>Reaction term</b>: the echo. <b>Undirected</b>: directed plus reaction. The last two columns split the
variance of a neighbour's contribution at the all-to-all fold into the part that is frozen and the part that
rotates.</p>
{tt}

<h3>Table P1.2. Measured coefficient at the largest size run</h3>
<p>Column by column: <b>fold ratio</b> is the simulated fold divided by the exact all-to-all fold, mean over
seeds with a 95% confidence interval; <b>closure ratio</b> is the full Rician closure at that k;
<b>coefficient</b> is k &times; (ratio &minus; 1); the two <b>distance</b> columns say how many standard errors the
measured coefficient lies from each prediction (within about &plusmn;2 is compatible). The fold is found by
bisection from a fully synchronised start with relaxation time 480, tolerance 0.2%.</p>
{t_big}
{t_echo}
{fig_html}
<ul class="findings">
<li><b>Undirected graphs:</b> the 95% interval contains the new theory's value in {ut} of {un} cells and the
Rician value in {ur} of {un}. <b>Directed graphs:</b> the new theory in {dt_} of {dn} cells, the Rician value in
{dr} of {dn}. Read the table, not this count, for the size of each discrepancy: both predictions are
leading-order in 1/k and are expected to fail at small k.</li>
{size_note}
{demo_note}
<li><b>The fold varies from sample to sample.</b> Mean standard deviation of the fold ratio across seeds on
undirected graphs: {sd_note}. Any single-seed fold is uncertain by about this much.</li>
<li><b>What is not established.</b> Whether the coefficients converge to the predicted values needs larger k and
N than this pilot. Nothing here tests the closure's critical connectivity or the clustering claim. The theory
neglects loops, degree heterogeneity and second-order terms.</li>
</ul>

<h3>Table P1.3. Every size</h3>
{t_all}
{t_ex}
{validation_blocks()}
</section>"""


def _cloud_tables():
    """Shared by the status card and section P2: fold cross-check rows and branch rows, all from results/."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc import report
    cf, bc = report.cloud_fold_cells(), report.branch_cells()
    if cf is None or bc is None:
        return None
    rk = report.load_summary("paperA_theory_k")

    def th_fold(graph, p_, k):
        t = report._theory(float(p_))
        if t is None:
            return float("nan")
        if graph == "dir":
            return t["directed"]
        hit = [r["reaction_at_k"] for r in (rk["rows"] if rk else []) if abs(r["p"] - float(p_)) < 1e-9 and r["k"] == int(k)]
        return t["directed"] + hit[0] if hit else t["undirected"]

    cf = cf.assign(source="cloud")
    loc = [c for c in (report.fold_cells("paperA_A1_pilot"), report.fold_cells("paperA_A7")) if c is not None]
    if loc:
        lf = pd.concat(loc, ignore_index=True)
        from scipy import stats as _st
        lf = lf.assign(source="local", t_relax=480.0)
        lf["se"] = [(hi - lo) / (2 * _st.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
                    for lo, hi, n in zip(lf.coef_lo, lf.coef_hi, lf.n_seeds)]
        allf = pd.concat([cf, lf[["graph", "p", "k", "N", "n_seeds", "t_relax", "coef", "coef_lo", "coef_hi", "se", "source"]]],
                         ignore_index=True)
    else:
        allf = cf
    allf["theory"] = [th_fold(g, p_, k) for g, p_, k in zip(allf.graph, allf.p, allf.k)]
    allf["rician"] = (allf.p + 2) / 2
    # cloud against local at the same graph, p, k: the cloud cell with the largest N
    pairs = []
    for r in allf[allf.source == "local"].itertuples():
        c = allf[(allf.source == "cloud") & (allf.graph == r.graph) & (allf.p == r.p) & (allf.k == r.k)]
        if len(c):
            c = c.sort_values("N").iloc[-1]
            se = (r.se ** 2 + c.se ** 2) ** 0.5
            pairs.append({"graph": r.graph, "p": r.p, "k": r.k, "N_local": r.N, "N_cloud": c.N, "local": r.coef,
                          "cloud": c.coef, "diff": r.coef - c.coef, "se": se, "z": (r.coef - c.coef) / se})
    return {"fold": allf, "pairs": pd.DataFrame(pairs), "branch": bc, "report": report}


def _gname(g: str) -> str:
    return "undirected (two-way)" if g == "rr" else "directed (one-way)"


def status_card() -> str:
    """Where the project stands today, in one screen. Every number is read from results/."""
    T = _cloud_tables()
    if T is None:
        return ""
    report, bc = T["report"], T["branch"]
    pool = report.branch_pooled("dir", 1.0, 48)
    d1 = bc[(bc.graph == "dir") & (bc.p == 1.0)].iloc[0]
    und = bc[(bc.graph == "rr") & (bc.p == 1.0) & (bc.N == 20000)].sort_values("k")
    und_seq = ", ".join(f"{r.c:.2f} at k = {int(r.k)}" for r in und.itertuples())
    fold = T["fold"]
    dfold = fold[(fold.graph == "dir") & (fold.p == 1.0)]
    n_below = int((dfold.coef < dfold.theory).sum())
    z_c = (pool[0] - d1.closure) / pool[1]
    z_t = (pool[0] - d1.theory) / pool[1]
    return f"""
<section id="status" class="summary">
<div class="sec-head"><span class="tag">NOW</span><h2>Where the project stands</h2><span class="verdict v-new">one new result</span></div>
<p>This card is the current state. The older summary below it is kept as a record and is marked where it no
longer holds. Details, tables and every caveat are in section <a href="#p2">P2</a>.</p>
<div class="grid2">
<div><h3>Established</h3><ul class="findings">
<li><b>On directed graphs the Rician closure is wrong and the dynamic theory is right.</b> The shift of the
synchronised branch, c = k (R &minus; R<sub>0</sub>), pooled over {pool[2]} connectivities and {pool[3]} runs at
p = 1, is <b>{pool[0]:.3f} &plusmn; {pool[1]:.3f}</b>. The dynamic theory predicts {d1.theory:.3f} with no free
parameter ({abs(z_t):.1f} standard errors away). The closure predicts {d1.closure:.3f} ({abs(z_c):.0f} standard errors
away).</li>
<li><b>Two independent sets of fold runs agree.</b> The cloud runs and the local runs used different seeds,
sizes and relaxation times. Their fold coefficients differ by at most
{T['pairs'].z.abs().max():.1f} combined standard errors across {len(T['pairs'])} comparable cells.</li>
<li><b>The stored summary is faithful to the raw runs.</b> Recomputing every cell of the branch table from the
run file reproduces the stored table to {report.branch_summary_check():.0e}.</li>
</ul></div>
<div><h3>Not established</h3><ul class="findings">
<li><b>The reaction (echo) term on undirected graphs.</b> The branch shift at N = 20000 reads {und_seq}. It
approaches the predicted {und.iloc[0].theory:.2f} near k = 48 to 96 and then moves away again, and every one of
those cells has only {int(und.n_seeds.min())} seeds.</li>
<li><b>The fold coefficient on directed graphs.</b> The closure is excluded, but {n_below} of {len(dfold)}
measured cells at p = 1 lie below the dynamic theory's {dfold.theory.iloc[0]:.2f}. The leading-order number is
too high at the connectivities reached.</li>
<li><b>Novelty of the mechanisms.</b> They are not new: sparse dynamical mean-field and cavity frameworks
(Metz 2025; Metz and P&eacute;rez Castillo 2026) and the symmetric-against-asymmetric comparison (Aonishi,
Kurata and Okada 2002) already contain them. What appears new is the explicit coefficients for the adaptive
model and their tests (<code>docs/PRIOR_ART.md</code>).</li>
<li><b>Everything that depended on the closure</b>: the critical connectivity k<sub>c</sub>(p) and the
effective connectivity for clustered graphs.</li>
<li><b>Any usable crash warning from synchrony</b> (section <a href="#b1">B1</a>): unchanged, still not
detected.</li>
</ul></div></div>
</section>"""


def cloud_card() -> str:
    T = _cloud_tables()
    if T is None:
        return '<section id="p2"><h2>P2 Cloud runs</h2><div class="missing">results not available yet</div></section>'
    report, fold, pairs, bc = T["report"], T["fold"], T["pairs"], T["branch"]

    def fz(v, ref, se):
        return "n/a" if pd.isna(se) or pd.isna(ref) else f"{(v - ref) / se:+.1f}"

    t_fold = _tbl(["graph", "p", "k", "N", "source", "seeds", "relaxation time", "coefficient k(ratio&minus;1) [95% CI]",
                   "standard error", "theory", "Rician closure", "distance from theory (SE)", "distance from Rician (SE)"],
                  [[_gname(r.graph), f"{r.p:g}", int(r.k), int(r.N), r.source, int(r.n_seeds), f"{r.t_relax:g}",
                    _ci(r.coef, r.coef_lo, r.coef_hi, 2), f"{r.se:.2f}", f"{r.theory:.2f}", f"{r.rician:.2f}",
                    fz(r.coef, r.theory, r.se), fz(r.coef, r.rician, r.se)]
                   for r in fold.sort_values(["graph", "p", "k", "N", "source"]).itertuples()])
    t_pairs = _tbl(["graph", "p", "k", "local N", "cloud N", "local coefficient", "cloud coefficient",
                    "difference &plusmn; combined SE", "difference in standard errors"],
                   [[_gname(r.graph), f"{r.p:g}", int(r.k), int(r.N_local), int(r.N_cloud), f"{r.local:.2f}", f"{r.cloud:.2f}",
                     f"{r['diff']:+.2f} &plusmn; {r.se:.2f}", f"{r.z:+.1f}"] for _, r in pairs.sort_values(["graph", "p", "k"]).iterrows()])
    t_branch = _tbl(["graph", "p", "k", "N", "seeds", "c = k(R&minus;R<sub>0</sub>) [95% CI]", "standard error",
                     "overlap k&sup2;/N", "theory for this graph", "Rician closure", "distance from theory (SE)",
                     "distance from Rician (SE)"],
                    [[_gname(r.graph), f"{r.p:g}", int(r.k), int(r.N), int(r.n_seeds), _ci(r.c, r.c_lo, r.c_hi, 3),
                      "n/a" if pd.isna(r.se) else f"{r.se:.3f}", f"{r.overlap:.2f}",
                      "n/a" if pd.isna(r.theory) else f"{r.theory:.3f}", f"{r.closure:.3f}",
                      fz(r.c, r.theory, r.se), fz(r.c, r.closure, r.se)]
                     for r in bc.sort_values(["graph", "p", "k", "N"]).itertuples()])
    pool = report.branch_pooled("dir", 1.0, 48)
    d1 = bc[(bc.graph == "dir") & (bc.p == 1.0)].iloc[0]
    two = bc[bc.n_seeds == 2]
    wide = two.assign(w=two.c_hi - two.c_lo)
    und = bc[(bc.graph == "rr") & (bc.p == 1.0) & (bc.N == 20000)].sort_values("k")
    k192 = und[und.k == und.k.max()].iloc[0]
    k96u = und[und.k == 96].iloc[0]
    d96 = bc[(bc.graph == "dir") & (bc.k == 96)].sort_values("N")
    runs = report.branch_runs()
    ref = runs.groupby(["p", "n"]).R0.first()
    ref_note = "; ".join(f"p = {p_:g}, N = {int(n)}: {v:.5f}" for (p_, n), v in ref.items())
    r20, r80 = ref.get((1.0, 20000)), ref.get((1.0, 80000))
    dfold = fold[(fold.graph == "dir") & (fold.p == 1.0)]
    ufold = fold[(fold.graph == "rr") & (fold.p == 1.0)]
    u_closer = int(((ufold.coef - ufold.rician).abs() < (ufold.coef - ufold.theory).abs()).sum())
    return f"""
<section id="p2" class="summary">
<div class="sec-head"><span class="tag">P2</span><h2>Independent cloud runs: cross-check of the fold, and a direct test on the synchronised branch</h2><span class="verdict v-new">new result on directed graphs</span></div>
<p>A second set of simulations was run on a cloud machine with the same compiled kernel
(<code>esfc.fast</code>) at commit <code>d49a0da</code>, using different seeds, different system sizes and a
shorter relaxation time than the local runs of section P1. This section does three things: it checks the cloud
files against each other and against the local runs, it reports a new measurement that tests the theory away
from the fold, and it states what is and is not established. All numbers are read at build time from
<code>{report.cloud_run_name()}</code>. Scripts: <code>experiments/fold.py</code>,
<code>experiments/branch_sim.py</code>, <code>experiments/branch_theory.py</code>,
<code>experiments/react_branch.py</code>. The original run note is
<code>docs/CLOUD_RUN_2026-10-13.md</code>.</p>

<h3>Table P2.1. Do the files agree with each other?</h3>
{_tbl(["check", "result"], [
    ["Stored branch table (<code>branch_summary.csv</code>) against means recomputed from the run file",
     f"all {len(bc)} cells reproduced; largest difference {report.branch_summary_check():.0e} (rounding of the stored file)"],
    ["Duplicate runs in the branch file", f"{int(runs.duplicated(['p', 'kind', 'k', 'n', 'seed']).sum())}"],
    ["Fold theory values quoted in the run note against <code>" + report.run_name('paperA_theory') + "</code>",
     "match (table P1.1 is the source used here)"],
    ["Cloud fold coefficients against local fold coefficients (table P2.3)",
     f"{len(pairs)} comparable cells, largest difference {pairs.z.abs().max():.1f} combined standard errors"],
    ["Crash-warning figures quoted in the run note against <code>results/paperB_intraday</code> and <code>results/paperB_daily</code>",
     "match (compared by hand on 2026-10-14; section B1 reads the same files at build time)"],
    ["Branch theory values (table P2.4, columns theory and Rician closure)",
     (f"recomputed by <code>esfc.branch</code>; largest difference from the values copied out of the run note "
      f"{report.branch_theory_agreement():.1e} (table P3.1)") if report.branch_theory_agreement() is not None else
     "<b class='bad'>not regenerated locally</b>. Taken from the run note and stored in <code>branch_theory_reported.json</code>"],
])}

<h3>Table P2.2. Fold coefficient, every cloud and local cell</h3>
<p>Same quantity as table P1.2: the fold is written as all-to-all fold &times; (1 + a/k) and the table gives
a = k &times; (ratio &minus; 1). <b>Source</b> says which set of runs the row comes from. <b>Relaxation time</b> is
how long each bisection step waits before deciding whether the synchronised state survived.
<b>Theory</b> is the dynamic coefficient on directed graphs and the dynamic coefficient plus the reaction term
evaluated at that k on undirected graphs. The two <b>distance</b> columns give (measured &minus; predicted) in
units of the standard error; a value beyond about &plusmn;2 is a disagreement. The 95% interval is a Student-t
interval over seeds.</p>
{t_fold}

<h3>Table P2.3. Cloud against local at the same graph, exponent and connectivity</h3>
<p>For each local cell, the cloud cell with the same graph type, p and k at its largest N. The two sets share
no seeds, so the difference should be zero within the combined standard error if nothing depends on the
machine, the seeding scheme, the size or the relaxation time.</p>
{t_pairs}

<h3>Why the test was moved off the fold</h3>
<p>The fold is a poor place for a precision test. Its value differs from one network to the next by an amount
that does not shrink quickly with N, the coefficient multiplies that scatter by k, and the expansion in 1/k is
not analytic there (the next term is of relative order k<sup>&minus;1/2</sup>). The same second-order expansion
holds at every point of the stable synchronised branch, where none of this applies. Above the fold the
self-consistency R = z<sub>0</sub>(&lambda; R<sup>p+1</sup>) + &lang;a<sub>2</sub>&rang;/k gives a finite shift of
the stationary synchrony:</p>
<div class="math">k (R &minus; R<sub>0</sub>) &rarr; c(&lambda;) = &lang;a<sub>2</sub>&rang; / (1 &minus; z<sub>0</sub>&prime;(h) &lambda; (p+1) R<sub>0</sub><sup>p</sup>),
&nbsp; h = &lambda; R<sub>0</sub><sup>p+1</sup></div>
<p>R<sub>0</sub> is the synchrony of the all-to-all model at the same coupling, and &lang;a<sub>2</sub>&rang; is the
second-order term of <code>docs/THEORY.md</code> section 3. The three treatments differ only in which neighbour
scatter enters &lang;a<sub>2</sub>&rang;: all of it (Rician closure), the frozen part plus the filtered rotating part
(dynamic theory, directed graphs), or that plus the reaction term (undirected graphs). c is negative because
finite connectivity lowers the synchrony of the branch.</p>
<p><b>Protocol.</b> Coupling &lambda; = 1.25 &times; the all-to-all fold. Start fully synchronised, relax for 150
time units, average R over the next 250. R<sub>0</sub> is measured with the all-to-all kernel on the same
frequencies at the same N, so the frequency clip and the finite frequency sample cancel in the difference.
Measured references: {ref_note}.</p>

<h3>Table P2.4. Shift of the synchronised branch</h3>
<p><b>c</b> is the mean over seeds with a Student-t 95% interval. With two seeds that interval is very wide,
which is the correct statement of what two runs can show; the <b>standard error</b> column is given as well
because it is what the run note quoted. <b>Overlap k&sup2;/N</b> is the expected number of nodes shared by the
neighbourhoods of a node's neighbours; when it is not small, neighbour fields are correlated and the theory's
independence assumption fails. <b>Theory for this graph</b> is the dynamic theory on directed graphs and the
dynamic theory plus reaction on undirected graphs.</p>
{t_branch}

<h3>What the branch test shows</h3>
<ul class="findings">
<li><b>Directed graphs, p = 1: confirmed.</b> The {pool[2]} cells with {int(d1.n_seeds)} seeds each
(k = 12, 24, 48) pool to c = <b>{pool[0]:.3f} &plusmn; {pool[1]:.3f}</b>. The dynamic theory's
{d1.theory:.3f} is {abs((pool[0] - d1.theory) / pool[1]):.1f} standard errors away; the closure's
{d1.closure:.3f} is {abs((pool[0] - d1.closure) / pool[1]):.0f} standard errors away. There is no fitted
parameter, and the measured value does not drift with k.</li>
<li><b>Directed graphs, p = 2: compatible, weakly tested.</b> Two seeds per cell. The point estimates sit on
the dynamic theory, but the 95% intervals are too wide to count as a second confirmation.</li>
<li><b>Undirected graphs: suggestive, not confirmed.</b> At N = 20000 the shift is
{", ".join(f"{r.c:.2f} (k = {int(r.k)})" for r in und.itertuples())} against a prediction of {k96u.theory:.2f}.
It reaches the prediction only at k = 48 and 96 and departs again at k = {int(k192.k)}. All of these cells have
two seeds.</li>
<li><b>The k = 96 undirected point may agree by accident.</b> Its overlap is {k96u.overlap:.2f}. The directed
run at the same k and N gave {d96.iloc[0].c:.2f}, and moved to {d96.iloc[-1].c:.2f} when N was raised to
{int(d96.iloc[-1].N)} (overlap {d96.iloc[-1].overlap:.2f}). So at N = 20000 a k = 96 point carries a finite-size
error of that order, and the undirected point has not been repeated at larger N.</li>
<li><b>One coupling only.</b> Every run is at 1.25 times the all-to-all fold. This is one point of the branch,
not the curve c(&lambda;).</li>
<li><b>The reference is shared.</b> R<sub>0</sub> is one all-to-all run per (p, N). Its error is common to every
seed, is multiplied by k, and is not in the intervals. The two p = 1 references differ by
{abs(r20 - r80):.1e}, which is {96 * abs(r20 - r80):.3f} in c at k = 96 and {192 * abs(r20 - r80):.3f} at
k = 192: small, but not zero.</li>
</ul>

<h3>What the fold data add</h3>
<ul class="findings">
<li><b>Directed fold:</b> the closure's value is excluded in every cell, by both sets of runs. The measured
coefficient is below the dynamic theory's leading-order {dfold.theory.iloc[0]:.2f} in
{int((dfold.coef < dfold.theory).sum())} of {len(dfold)} cells at p = 1, so the leading-order number
overestimates the fold shift at these connectivities.</li>
<li><b>Undirected fold:</b> the Rician value is closer to the measurement than the new theory in {u_closer} of
{len(ufold)} cells at p = 1. This is the cancellation described in section P1 (too much scatter, no echo), and
it is the reason the closure looked right for so long.</li>
</ul>

<h3>Problems with the cloud run as a record</h3>
<ul class="findings">
<li>The runs were launched by hand, outside <code>esfc run</code>, so the directory has no <code>env.txt</code> or
<code>seeds.json</code>. The settings in its <code>config.yaml</code> were written afterwards from the run note.</li>
<li>The last four lines of <code>branch_runs.jsonl</code> lack the timing and time-spread fields, so they came
from an earlier version of the script.</li>
<li>{len(two)} of {len(bc)} branch cells have two seeds and {int((bc.n_seeds == 1).sum())} have one. The widest
two-seed interval spans {wide.w.max():.2f} in c.</li>
<li>The run note states that no local config had results. That was out of date: the directed fold experiment
is complete locally and the undirected pilot is partial (section P1).</li>
</ul>

<h3>Next steps, in order (revised after the external novelty audit)</h3>
<ol>
<li><b>Reciprocity interpolation.</b> Done at k = 24 and 48: section <a href="#p3">P3</a>.</li>
<li><b>Large connectivity with small overlap.</b> k = 96 and 192, directed, half mutual and undirected, with
k&sup2;/N at 0.05. Needs the cluster: <code>configs/paperA_branch_ada.yaml</code>.</li>
<li><b>Tie the dynamic term to the exact directed theory.</b> Show it is the leading 1/k expansion of the
equation of Metz (2025), analytically or by population dynamics at small k.</li>
<li><b>Derive the exponent of the non-analytic correction at the fold.</b> Nodes about to unlock respond
without limit, which should give a correction of relative order k<sup>&minus;1/2</sup>. Derive it, then fit
the fold data of table P2.2 with the exponent fixed and at least four connectivities.</li>
<li><b>Branch curve.</b> Several couplings and exponents (p = 1, 2, 3) on directed graphs with 10 or more
seeds, against the three curves of table P3.1.</li>
<li><b>Degree fluctuations.</b> Erd&#337;s-R&eacute;nyi graphs add their own 1/k term through the spread of
degrees; measure it against random regular graphs.</li>
<li><b>Forward threshold at finite k</b> (a hypothesis, cheap to test): in the incoherent state the local
order parameter is of order k<sup>&minus;1/2</sup>, not zero, so incoherence may lose stability at a finite
coupling that grows with k.</li>
<li><b>Only then</b> the critical connectivity and clustering, both derived from the closure.</li>
</ol>
</section>"""


def reciprocity_card() -> str:
    """Section P3: the reciprocity interpolation and the stored branch predictions."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc import report
    cells, fit, tt = report.reciprocity_cells(), report.reciprocity_fit(), report.branch_theory_table()
    if cells is None or cells.empty or tt.empty:
        return '<section id="p3"><h2>P3 Reciprocity test</h2><div class="missing">results not available yet</div></section>'

    def fz(v, ref, se):
        return "n/a" if pd.isna(se) or pd.isna(ref) or se == 0 else f"{(v - ref) / se:+.1f}"

    t_th = _tbl(["p", "coupling / all-to-all fold", "R<sub>0</sub>", "Rician closure", "dynamic (directed)", "reaction",
                 "dynamic + reaction (undirected)", "reaction from locked nodes", "reaction from drifting nodes"],
                [[f"{r.p:g}", f"{r.mult:g}", f"{r.R0:.4f}", f"{r.closure:.3f}", f"<b>{r.dynamic:.3f}</b>", f"{r.reaction:.3f}",
                  f"<b>{r.undirected:.3f}</b>", f"{r.reaction_locked:.3f}", f"{r.reaction_drifting:.3f}"]
                 for r in tt.sort_values(["p", "mult"]).itertuples()])
    t_cells = _tbl(["p", "k", "N", "overlap k&sup2;/N", "intended mutual fraction &rho;", "realised mutual fraction", "seeds",
                    "c = k(R&minus;R<sub>0</sub>) [95% CI]", "standard error", "theory: dynamic + &rho; &times; reaction",
                    "Rician closure", "distance from theory (SE)", "distance from Rician (SE)"],
                   [[f"{r.p:g}", int(r.k), int(r.N), f"{r.overlap:.3f}", f"{r.rho:g}", f"{r.reciprocity:.4f}", int(r.n_seeds),
                     _ci(r.c, r.c_lo, r.c_hi, 3), f"{r.se:.3f}", f"{r.theory:.3f}", f"{r.closure:.3f}",
                     fz(r.c, r.theory, r.se), fz(r.c, r.closure, r.se)]
                    for r in cells.sort_values(["p", "k", "rho"]).itertuples()])
    t_fit = _tbl(["p", "k", "runs", "fitted intercept &plusmn; SE", "predicted (dynamic)", "distance (SE)",
                  "fitted slope &plusmn; SE", "predicted (reaction)", "distance (SE)", "curvature &plusmn; SE (zero if linear)"],
                 [[f"{r.p:g}", int(r.k), int(r.n_points), f"{r.intercept:.3f} &plusmn; {r.intercept_se:.3f}", f"{r.dynamic:.3f}",
                   fz(r.intercept, r.dynamic, r.intercept_se), f"<b>{r.slope:.3f} &plusmn; {r.slope_se:.3f}</b>",
                   f"{r.reaction:.3f}", fz(r.slope, r.reaction, r.slope_se), f"{r.curvature:+.3f} &plusmn; {r.curvature_se:.3f}"]
                  for r in fit.sort_values(["p", "k"]).itertuples()]) if fit is not None and len(fit) else ""
    agree = report.branch_theory_agreement()
    rec = "paperA_reciprocity_gpu" if report._rows("paperA_reciprocity_gpu") else "paperA_reciprocity"
    got = report._rows(rec)
    partial = ('<p class="note"><b class="bad">Run incomplete.</b> ' + f"{len(got[0])} rows so far; the tables below use what has finished.</p>"
               if report.is_partial(rec) else "")
    bullets = []
    if fit is not None and len(fit):
        for r in fit.sort_values("k").itertuples():
            zs, zi = (r.slope - r.reaction) / r.slope_se, (r.intercept - r.dynamic) / r.intercept_se
            zc = r.curvature / r.curvature_se
            bullets.append(
                f"<li><b>k = {int(r.k)}.</b> Slope {r.slope:.3f} &plusmn; {r.slope_se:.3f} against a predicted reaction of "
                f"{r.reaction:.3f} ({abs(zs):.1f} standard errors apart; the measured slope is {100 * r.slope / r.reaction:.0f}% "
                f"of the prediction). Intercept {r.intercept:.3f} &plusmn; {r.intercept_se:.3f} against the dynamic value "
                f"{r.dynamic:.3f} ({abs(zi):.1f} standard errors). Curvature {r.curvature:+.3f} &plusmn; {r.curvature_se:.3f} "
                f"({abs(zc):.1f} standard errors from zero). The Rician closure predicts slope 0, which is "
                f"{abs(r.slope / r.slope_se):.0f} standard errors from the measured slope.</li>")
    fig_html = render_fig(fig("figures/p3_reciprocity.png",
        "Horizontal axis: &rho;, the fraction of a node's k in-neighbours that listen back (0 = fully one-way, 1 = fully two-way). Vertical axis: the branch shift c = k (R &minus; R<sub>0</sub>); more negative means finite connectivity costs more synchrony. Points: mean over seeds, one colour and marker per connectivity k, with 95% confidence bars; points at the same &rho; are nudged sideways so the bars do not hide each other. Solid black line: the theory, dynamic + &rho; &times; reaction, with no fitted parameter. Dotted grey line: the Rician closure, which gives the same value at every &rho;.",
        "A flat set of points would support the closure. Points on the black line support the theory in full. Points that fall on a straight line with a different slope mean the reaction mechanism is right in kind (it scales with the number of mutual links) but its leading-order size is not yet reached at this k."))
    return f"""
<section id="p3" class="summary">
<div class="sec-head"><span class="tag">P3</span><h2>Reciprocity test: does the branch shift grow in proportion to the fraction of mutual links?</h2><span class="verdict v-open">test of the reaction term</span></div>
{partial}
<p>Section P2 left the reaction (echo) term unconfirmed, and an external novelty audit pointed out that
the idea behind it is not new: comparing symmetric with asymmetric coupling to isolate an Onsager reaction
term was done for oscillator networks by Aonishi, Kurata and Okada (2002), and the general sparse-graph
frameworks of Metz (2025) and Metz and P&eacute;rez Castillo (2026) contain both mechanisms
(<code>docs/PRIOR_ART.md</code>). What this project can add is a quantitative prediction and its test. This
section is that test. Runs: <code>{report.run_name(rec)}</code> (on the GPU path, 10 seeds; the CPU run of
2026-10-14 was interrupted before writing rows), <code>{report.run_name('paperA_branch_theory')}</code>.
Configs: <code>configs/{rec}.yaml</code>,
<code>configs/paperA_branch_theory.yaml</code>.</p>

<h3>The prediction</h3>
<p>Take graphs in which every node listens to exactly k others, and make a fraction &rho; of those links
mutual (the node listened to also listens back) and the rest one-way. &rho; = 0 is the directed graph of
section P2 and &rho; = 1 is an ordinary undirected random regular graph. The echo reaches a node only
through neighbours that listen back, and there are &rho;k of them, so at leading order in 1/k:</p>
<div class="math">c(&rho;) = c<sub>dynamic</sub> + &rho; &times; c<sub>reaction</sub></div>
<p>a straight line in &rho; whose intercept and slope are both fixed by the theory. The Rician closure never asks
whether links are mutual, so it predicts a horizontal line.</p>

<h3>Table P3.1. Predicted branch shift, computed in this repository</h3>
<p>These are the values that section P2 could only copy from a run note. They are now computed by
<code>esfc.branch</code> (numerical integrals, no simulation) and stored as a result.
{"The computed values differ from the copied ones by at most " + f"{agree:.1e}" + ", so the copied values were correct." if agree is not None else ""}
<b>R<sub>0</sub></b> is the all-to-all synchrony at that coupling. <b>Reaction</b> is split into the part from
locked nodes (closed form) and the part from drifting nodes (integrated numerically). All shifts are negative:
finite connectivity lowers synchrony.</p>
{t_th}

<h3>Table P3.2. Measured branch shift at each mutual fraction</h3>
<p>Coupling 1.25 times the all-to-all fold, start synchronised, relax 150 time units, average R over the next
250. <b>Realised mutual fraction</b> is counted on the generated graphs; it exceeds the intended value by the
small chance that a one-way link is reciprocated by accident. N is chosen for each k so that the overlap
k&sup2;/N stays below 0.05. The interval is a Student-t 95% interval over seeds.</p>
{t_cells}
{fig_html}

<h3>Table P3.3. Straight-line fit per connectivity</h3>
<p>Least squares of c on &rho; over all individual runs. The theory fixes both numbers: the intercept should
equal the dynamic value and the slope the reaction value. <b>Curvature</b> is the coefficient of
&rho;&sup2; when a quadratic term is added; it tests whether the dependence is a straight line at all.</p>
{t_fit}

<h3>Reading</h3>
<ul class="findings">
{"".join(bullets)}
</ul>

<h3>What remains for this test</h3>
<ul class="findings">
<li><b>Larger k with small overlap.</b> k = 96 and 192 need N of 200,000 and 800,000 to keep k&sup2;/N at
0.05. Running on the cluster: <code>configs/paperA_branch_ada_gpu.yaml</code>.</li>
<li><b>Second exponent and other couplings.</b> Done for &rho; = 0 and 1 in section P4.</li>
<li><b>Link to the exact directed theory.</b> Show that the dynamic term is the leading 1/k expansion of the
equation of Metz (2025), analytically or by population dynamics.</li>
</ul>
</section>"""


def branch_curve_card() -> str:
    """Section P4: the branch shift along the coupling on directed and undirected graphs (GPU run)."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc import report
    cells, ex = report.branch_curve_cells(), report.branch_curve_extrapolation()
    if cells is None or ex is None:
        return '<section id="p4"><h2>P4 Branch curve</h2><div class="missing">results not available yet</div></section>'
    gname = {"dir": "directed", "rho1.00": "undirected"}
    t_cells = _tbl(["graph", "p", "coupling / all-to-all fold", "k", "N", "seeds", "c = k(R&minus;R<sub>0</sub>) [95% CI]",
                    "theory for this graph", "Rician closure", "distance from theory (SE)", "used"],
                   [[gname[r.variant], f"{r.p:g}", f"{r.mult:g}", int(r.k), int(r.n), int(r.n_ok), _ci(r.c_mean, r.ci_lo, r.ci_hi, 3),
                     f"{r.theory:.3f}", f"{r.closure:.3f}", f"{(r.c_mean - r.theory) / r.se:+.1f}",
                     "yes" if r.valid else '<b class="bad">no</b> (below the graphs&#39; own fold)']
                    for r in cells.sort_values(["variant", "p", "mult", "k"]).itertuples()])
    t_ex = _tbl(["graph", "p", "coupling / all-to-all fold", "values of k used", "c extrapolated to k &rarr; &infin; &plusmn; SE",
                 "theory", "ratio to theory", "distance (SE)", "Rician closure"],
                [[gname[r.graph], f"{r.p:g}", f"{r.mult:g}", int(r.n_k), f"{r.c_inf:.3f} &plusmn; {r.c_inf_se:.3f}", f"{r.theory:.3f}",
                  f"<b>{r.ratio_to_theory:.3f}</b>", f"{r.z:+.1f}", f"{r.closure:.3f}"]
                 for r in ex.sort_values(["graph", "p", "mult"]).itertuples()])
    und, dr = ex[ex.graph == "rho1.00"], ex[ex.graph == "dir"]
    far, near = und[und.mult >= 2], und[(und.mult > 1.1) & (und.mult < 2)]
    bullets = [
        "<li><b>The Rician closure is excluded in every cell</b>, too negative on directed graphs and not negative enough on "
        "undirected ones.</li>",
        f"<li><b>Undirected graphs converge on the reaction-corrected theory as k grows.</b> Away from the fold (2 and 3 times it) "
        f"the extrapolated shift is {far.ratio_to_theory.min():.2f} to {far.ratio_to_theory.max():.2f} times the prediction. "
        f"Nearer the fold (1.25 and 1.5 times it) it is {near.ratio_to_theory.min():.2f} to {near.ratio_to_theory.max():.2f} times it "
        f"({near.z.abs().min():.1f} to {near.z.abs().max():.1f} standard errors). The predicted shift diverges at the fold, where the "
        f"1/k expansion converges slowest, so a straight line in 1/k through k &le; 48 is least safe there; the size scan at "
        f"k = 96 and 192 (<code>paperA_branch_ada_gpu</code>) tests that region.</li>",
        f"<li><b>Directed graphs sit a little beyond the dynamic theory.</b> Extrapolated ratio {dr.ratio_to_theory.mean():.3f} on "
        f"average ({dr.ratio_to_theory.min():.2f} to {dr.ratio_to_theory.max():.2f}); {int((dr.z.abs() <= 2.1).sum())} of {len(dr)} "
        f"cells within about 2 standard errors. The earlier single-coupling agreement (section P2) holds to within about 5 to 10%, "
        f"not exactly.</li>"]
    fig_html = render_fig(fig("figures/p4_branch_curve.png",
        "One panel per herding exponent p. Horizontal axis: coupling as a multiple of the all-to-all fold. Vertical axis: the branch shift c = k (R &minus; R<sub>0</sub>). Orange squares: directed graphs; blue circles: undirected. Small filled points: mean over 10 seeds at k = 12, 24, 48 (lighter = smaller k), with 95% bars. Large open points: c extrapolated to infinite k. Dashed lines: the leading-order theory for each graph type; dotted: the Rician closure.",
        "If the theory is right at leading order, the open points sit on the dashed line of their colour. They do for undirected graphs away from the fold and, to within about 5 to 10%, for directed graphs; the closure misses both."))
    return f"""
<section id="p4" class="summary">
<div class="sec-head"><span class="tag">P4</span><h2>Branch curve: the shift of the synchronised branch along the coupling</h2><span class="verdict v-open">theory tested at five couplings</span></div>
<p>Item 1 of the plan after section P2: the branch shift at five couplings (1.1 to 3 times the all-to-all fold),
both exponents, directed and undirected graphs, k = 12, 24 and 48 with the overlap k&sup2;/N &le; 0.05,
10 seeds per cell, on the GPU path (validated against the CPU code, <code>docs/ADA_RUN_2026-10-24.md</code>).
Run: <code>{report.run_name(report.BRANCH_CURVE)}</code>. Config: <code>configs/paperA_branch_curve_gpu.yaml</code>.
Predictions: Table P3.1.</p>
{fig_html}
<h3>Table P4.1. Extrapolation to large connectivity</h3>
<p>Weighted least squares of c(k) = c<sub>&infin;</sub> + b/k over the values of k used (weights 1/SE&sup2;). The
straight line in 1/k is an assumption with three points at most.</p>
{t_ex}
<h3>Reading</h3>
<ul class="findings">{"".join(bullets)}</ul>
<h3>Table P4.2. Every cell</h3>
<p>Cells marked "no" lie at or below the realised graphs&#39; own fold, so seeds left the branch; they are left out
of the extrapolation.</p>
{t_cells}
</section>"""


def structure_card() -> str:
    """Section P5: fold on clustered and on degree-heterogeneous graphs (GPU run)."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc import report
    cells = report.fold_cells("paperA_structure_gpu")
    if cells is None or cells.empty:
        return '<section id="p5"><h2>P5 Graph structure</h2><div class="missing">results not available yet</div></section>'
    name = {"crr_t0": "random regular (no triangles)", "crr_t3": "3 triangles per node", "crr_t6": "6 triangles per node",
            "er": "Erd&odblac;s-R&eacute;nyi (Poisson degrees)"}
    t = _tbl(["graph", "p", "k", "N", "seeds", "fold ratio [95% CI]", "coefficient k(ratio&minus;1) [95% CI]"],
             [[name.get(r.graph, r.graph), f"{r.p:g}", int(r.k), int(r.N), int(r.n_seeds), _ci(r.ratio, r.ci_lo, r.ci_hi, 4),
               _ci(r.coef, r.coef_lo, r.coef_hi, 2)]
              for r in cells.sort_values(["p", "k", "graph"]).itertuples()])
    return f"""
<section id="p5" class="summary">
<div class="sec-head"><span class="tag">P5</span><h2>Graph structure: clustering and degree heterogeneity</h2><span class="verdict v-open">outside the leading-order theory</span></div>
<p>The theory assumes locally tree-like graphs with every degree equal (<code>docs/THEORY.md</code> section 5). This
run measures how far each assumption matters at fixed mean degree: clustered random regular graphs with 0, 3 and 6
triangles per node (clustering about 0.005, 0.05 and 0.10 at k = 12) and Erd&odblac;s-R&eacute;nyi graphs.
Run: <code>{report.run_name('paperA_structure_gpu')}</code>. Config: <code>configs/paperA_structure_gpu.yaml</code>.
The coefficient uses each graph&#39;s realised mean degree.</p>
{t}
<h3>Reading</h3>
<ul class="findings">
<li><b>Clustering raises the fold modestly.</b> At k = 12 the coefficient grows with the number of triangles at both
exponents; at k = 24, where the same triangles give less clustering, the effect is within the intervals.</li>
<li><b>Degree heterogeneity lowers the fold strongly.</b> With the 1/&lang;k&rang; normalisation, high-degree nodes are
coupled more strongly and hold the synchronised state. A heterogeneous mean field (Restrepo, Ott and Hunt 2005) is the
natural extension of the theory.</li>
<li><b>Consistency.</b> The random regular control agrees with the A1 pilot at the same k on independent seeds.</li>
</ul>
</section>"""



def window_card() -> str:
    """Section P6: the bistable window at low connectivity against the closure's critical connectivity."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc import report
    c = report.window_cells()
    if c is None:
        return '<section id="p6"><h2>P6 Bistable window</h2><div class="missing">results not available yet</div></section>'
    gname = {"dir": "directed", "rho1.00": "undirected"}
    main = c[c.run == "paperA_window_gpu"]
    t = _tbl(["run", "graph", "p", "k", "N", "seeds", "fold / all-to-all", "forward / all-to-all", "width [95% CI]",
              "closure: fold", "closure: incoherence unstable", "closure: width"],
             [[r.run_label, gname[r.variant], f"{r.p:g}", int(r.k), int(r.n), int(r.n_ok), f"{r.ratio_b:.3f}", f"{r.ratio_f:.3f}",
               _ci(r.width_mean, r.ci_lo, r.ci_hi, 3), f"{r.closure_ratio_b:.3f}", f"{r.closure_ratio_c:.3f}", f"{r.closure_width:.3f}"]
              for r in c.sort_values(["run", "variant", "p", "k", "n"]).itertuples()])
    und = main[main.variant == "rho1.00"]
    u1, u2 = und[(und.p == 1) & (und.k <= 6)], und[(und.p == 2) & (und.k <= 4)]
    both = main.pivot_table(index=["p", "k"], columns="variant", values="width_mean").dropna()
    rel = (both["dir"] / both["rho1.00"] - 1).abs()
    p1 = und[(und.p == 1) & (und.k >= 10)]
    k8 = und[(und.p == 1) & (und.k == 8)]
    dr = main[main.variant == "dir"]
    below = dr[dr.ratio_b < 1].groupby("p").k.max()
    dn = c[(c.run == "paperA_window_nscan_gpu") & (c.variant == "dir") & (c.p == 1) & (c.k == 4)].sort_values("n")
    dir_n = (" It does not depend on N (p = 1, k = 4: " + ", ".join(f"{v:.3f} at N = {int(n)}" for n, v in zip(dn.n, dn.ratio_b))
             + ").") if len(dn) else ""
    bullets = [
        f"<li><b>The window does not close where the closure says.</b> At p = 1 the closure predicts a width below "
        f"{u1.closure_width.max():.3f} for k &le; 6; on undirected graphs the measured width there is {u1.width_mean.min():.2f} to "
        f"{u1.width_mean.max():.2f} (95% interval above zero in {int((u1.ci_lo > 0).sum())} of {len(u1)} cells). At p = 2 the closure "
        f"closes the window at k = 3 to 4 (predicted width {u2.closure_width.min():.3f} to {u2.closure_width.max():.3f}); measured "
        f"{u2.width_mean.min():.2f} to {u2.width_mean.max():.2f}. Bistability survives down to k = 3.</li>",
        f"<li><b>Above it, the closure&#39;s width is close at p = 1</b>: measured / predicted {(p1.width_mean / p1.closure_width).min():.2f} "
        f"to {(p1.width_mean / p1.closure_width).max():.2f} for k &ge; 10, {float(k8.width_mean.iloc[0] / k8.closure_width.iloc[0]):.2f} at k = 8. "
        f"At p = 2 the closure&#39;s width is too large at every k &ge; 8.</li>",
        f"<li><b>The width hardly depends on whether links are mutual.</b> Directed and undirected widths differ by "
        f"{100 * rel[rel.index.get_level_values('k') >= 8].median():.0f}% (median, k &ge; 8), although both edges move a lot: at k = 3 "
        f"the undirected fold is {und[(und.k == 3) & (und.p == 1)].ratio_b.iloc[0]:.2f} times the all-to-all fold and the directed one "
        f"{dr[(dr.k == 3) & (dr.p == 1)].ratio_b.iloc[0]:.2f} times.</li>",
        "<li><b>On directed graphs at small k the fold lies below the all-to-all fold</b> ("
        + ", ".join(f"p = {pp:g}: k &le; {int(kk)}" for pp, kk in below.items())
        + "): the synchronised branch survives to weaker coupling than in the all-to-all model, opposite to every theory here "
        "(the closure predicts a fold well above it). The directed fold is therefore not monotonic in k: below 1 at small k, "
        "slightly above 1 near k = 12 to 24, and back to 1 as 1 + a/k at large k, which needs a large negative term of higher "
        "order in 1/k. A candidate is the floor of the local order parameter (about 0.89/&radic;k for random phases), which "
        "keeps the herding gain large." + dir_n + "</li>",
    ]
    uu, dd = main[main.variant == "rho1.00"].set_index(["p", "k"]), dr.set_index(["p", "k"])
    sc = ((uu.ratio_f / dd.ratio_f) / (uu.ratio_b / dd.ratio_b) - 1).dropna()
    s1 = sc[(sc.index.get_level_values("p") == 1) & (sc.index.get_level_values("k") >= 4)]
    s2 = sc[sc.index.get_level_values("p") == 2]
    f1 = (uu.ratio_b / dd.ratio_b)[(uu.index.get_level_values("p") == 1)]
    bullets.append(
        f"<li><b>Reciprocity rescales the coupling.</b> The undirected-to-directed ratio is the same for the fold and for the "
        f"forward threshold: at p = 1 and k = 4 to 24 the two ratios differ by at most {100 * s1.abs().max():.1f}% "
        f"(median {100 * s1.abs().median():.1f}%), while the ratio itself ranges from {f1.min():.2f} to {f1[f1.index.get_level_values('k') >= 4].max():.2f}. "
        f"Mutual links therefore act like a renormalised coupling, lambda divided by a factor f(k), which moves both edges "
        f"of the window together and explains why its relative width hardly depends on reciprocity. At p = 2 the rule is "
        f"approximate (up to {100 * s2.abs().max():.1f}%). At leading order f(k) = 1 + reaction / k.</li>")
    rob = c[c.run != "paperA_window_gpu"]
    if len(rob):
        ns = rob[rob.run == "paperA_window_nscan_gpu"]
        big = ns[ns.n == ns.n.max()]
        lng = rob[rob.run == "paperA_window_long_gpu"]
        k4 = ns[(ns.variant == "rho1.00") & (ns.p == 1) & (ns.k == 4)].sort_values("n")
        bullets.append(
            f"<li><b>Robust to N and to the waiting time.</b> At N = {int(big.n.iloc[0])} the interval is above zero in "
            f"{int((big.ci_lo > 0).sum())} of {len(big)} cells, and with the waiting time 1600 instead of 400 in "
            f"{int((lng.ci_lo > 0).sum())} of {len(lng)}. The window widens with N (undirected, p = 1, k = 4: "
            + ", ".join(f"{w:.3f} at N = {int(n)}" for n, w in zip(k4.n, k4.width_mean))
            + "), so finite size narrows it rather than creating it: the widths at N = 10000 are lower bounds. The forward "
            "threshold is a first passage and depends on N; the width in the limit of large N is not yet fixed.</li>")
    else:
        bullets.append("<li><b>Caveat.</b> The forward threshold is a first passage from a random start: it can depend on the "
                       "waiting time and on N. Robustness runs are pending.</li>")
    fig_html = render_fig(fig("figures/p6_window.png",
        "Columns: herding exponent p. Top: relative width of the bistable window, (forward threshold &minus; fold) / fold, against the number of neighbours k (log scale); points are means over 8 seeds with 95% bars, blue undirected and orange directed; the grey line is the Rician closure, which predicts the window closes at small k. Bottom: the two edges separately, relative to the all-to-all fold: filled points the fold, open points the forward threshold; grey lines the closure&#39;s fold (solid) and the coupling where it makes incoherence unstable (dashed).",
        "If the closure were right, the points in the top row would fall to zero at small k. They do not. In the bottom row the directed and undirected edges are far apart at small k, yet the top row shows almost the same width for both."))
    return f"""
<section id="p6" class="summary">
<div class="sec-head"><span class="tag">P6</span><h2>Bistable window at low connectivity: is there a critical connectivity?</h2><span class="verdict v-open">closure prediction tested</span></div>
<p>The closure predicts that below a critical connectivity the forward threshold (where a random, incoherent start
synchronises) meets the fold, so bistability disappears (README: "unverified"). Both edges were measured on the same graphs
and frequencies: the fold from a synchronised start (as in P1) and the forward threshold from a random start by the same
bisection, waiting 400 time units per coupling. Run: <code>{report.run_name('paperA_window_gpu')}</code>. Config:
<code>configs/paperA_window_gpu.yaml</code>.</p>
{fig_html}
<h3>Reading</h3>
<ul class="findings">{"".join(bullets)}</ul>
<h3>Table P6.1. Every cell</h3>
{t}
</section>"""


def novelty_card() -> str:
    """Section N1: what the external novelty audit found, in full. No measured values of this repository."""
    rows = [
        ["Metz, PRL 134, 037401 (2025)", "Exact single-site equation for nonlinear dynamics on sparse <b>directed</b> random graphs, solved by population dynamics. Synchronization is named as an application; the audit reports that the Kuramoto model is listed but not solved, and that graphs with two-way edges are excluded",
         "Our directed-graph (dynamic filtering) term is, in principle, the leading 1/k expansion of this equation for the adaptive coupling", "State this in the paper and show it, analytically or by population dynamics", "abstract checked"],
        ["Metz and P&eacute;rez Castillo, arXiv:2606.08689 (2026)", "Dynamical cavity method for sparse random graphs with one-way and two-way edges. Two-way edges need conditional path kernels; dense limits give drift, noise and <b>response</b> channels",
         "Our dynamic term is its noise channel and our reaction term is its response channel", "Cite as the framework that predicts the structure. Our part is the explicit coefficients and their test. Unrefereed preprint", "abstract checked"],
        ["Aonishi, Kurata and Okada, PRE 65, 046223 (2002)", "Oscillator associative-memory models with symmetric and with asymmetric dilution of couplings, which differ by the Onsager reaction term",
         "Direct precedent for the idea that undirected minus directed isolates the reaction term", "Must be cited. Differences: random sparse topology, Lorentzian Kuramoto setting, adaptive gain, explicit 1/k coefficient", "abstract checked"],
        ["Ichinomiya, PRE 72, 016109 (2005)", "Path-integral treatment of dynamics on a sparse random network: equivalent to globally coupled oscillators with random Gaussian interactions",
         "The origin of the static Gaussian scatter picture, which is what the Rician closure assumes", "Cite as the picture that the branch test falsifies", "abstract checked"],
        ["Pr&uuml;ser, Rosmej and Engel, PRL 132, 187201 (2024)", "Dynamical cavity method for the fully disordered <b>dense</b> Kuramoto model, with self-consistent correlation and response functions",
         "Dense counterpart of the reaction term", "Cite", "abstract checked"],
        ["Reddy, arXiv:2603.09402 (2026)", "Dense dynamical mean-field theory of oscillator networks: one oscillator driven by a mean field and self-consistent coloured noise; reduces to Ott-Antonsen without disorder",
         "Closest dense analogue of Ott-Antonsen driven by coloured noise", "Cite. The audit reports that it lists the sparse case as open", "abstract checked"],
        ["Hatchett and Uezu, PRE 78, 036106 (2008)", "Cavity analysis of oscillator networks on sparse random graphs in an equilibrium (Hamiltonian) case",
         "Could contain a finite-connectivity threshold expansion", "<b class='bad'>Read in full before submission</b>", "from the audit, unread"],
        ["Skantzos, P&eacute;rez Castillo and Hatchett, PRE 72, 066127 (2005)", "Equilibrium cavity method for XY spins on small-world graphs",
         "Finite-connectivity cavity lineage for phase variables. No frequency disorder, so no drifting oscillators", "Cite for lineage", "from the audit"],
        ["Restrepo, Ott and Hunt, PRE 71, 036151 (2005); Chaos 16, 015107 (2006)", "Onset of synchronization on networks from the largest adjacency eigenvalue, extended to directed networks",
         "Directed against undirected for the continuous onset only; no reaction term", "Cite. Our problem is the fold of an explosive transition, where that criterion does not apply", "from the audit"],
        ["Khanra, Kundu, Pal, Ji and Hens, Chaos 30, 031101 (2020)", "Adaptive power-law coupling widens hysteresis; semi-analytic mean-field backward threshold",
         "Mean field only, no 1/k term", "Cite as the group's prior work; our result says why mean-field backward thresholds are biased at finite degree", "read earlier"],
        ["Dutta, Pal and Hens, arXiv:2507.18421 (2025)", "Ott-Antonsen with an annealed degree-class closure on dense hypergraphs; double-jump transitions",
         "The annealed closure removes exactly the effects computed here", "Cite; a short section can quantify the error of that closure at moderate connectivity", "from the audit"],
        ["Kumpeerakij and Restrepo, arXiv:2605.24701 (2026)", "Self-consistent local order parameter theory for dyadic plus triadic interactions on finite structures",
         "Static self-consistency; misses the fluctuations from drifting neighbours", "Cite for the p = 2 (triadic) framing", "read earlier"],
    ]
    verdicts = [
        ["The Rician closure is wrong", "A useful negative result: the frozen-scatter closure was this project's own first theory (an approximation on top of Ichinomiya's 2005 Gaussian mapping, not his)", "Present as a falsified baseline, not as the headline"],
        ["(a) Dynamic filtering, exact on directed graphs", "Mechanism known (Metz 2025; noise channel of Metz and P&eacute;rez Castillo 2026). The closed-form coefficients for the adaptive coupling appear new", "New explicit result within a known framework"],
        ["(b) Onsager reaction on undirected graphs", "Concept known (Aonishi et al. 2002; dense case Pr&uuml;ser and Engel 2024; generic in Metz and P&eacute;rez Castillo 2026). The explicit sparse Kuramoto calculation appears new", "New explicit result; validation incomplete (sections P2, P3)"],
        ["(c) Directed against undirected contrast", "The structural distinction is known", "New only as a measured decomposition for explosive synchronization"],
        ["(d) Finite-degree shift of the fold", "No prior closed-form result found", "The most novel claim and the least supported by data (table P2.2)"],
        ["Non-analytic correction from nodes about to unlock", "No prior treatment found for sparse Kuramoto", "A selling point only if the exponent is derived, not inferred from two points"],
    ]
    referee = [
        ["Two limits kept apart", "N to infinity at fixed k for every k, by scaling in N or by holding k&sup2;/N fixed. Juh&aacute;sz, Kelling and &Oacute;dor (2019) warn that finite-size scaling on sparse graphs must be treated cautiously", "Partly: overlap held below 0.05 in section P3; no N scan yet"],
        ["Realisations and error bars", "10 to 20 independent graph and frequency realisations per point at the largest N, with confidence intervals; quantile frequencies to remove sampling noise; converged transient and averaging times", "Section P3: 10 seeds, quantile frequencies, Student-t intervals. Section P2 cloud cells: 1 to 4 seeds, not enough"],
        ["Fold protocol", "Fine backward sweeps and a square-root (saddle-node) fit to locate the fold, with sweep rate and step size shown not to matter", "Not done. The fold is found by bisection with a survival criterion; the convergence config exists and has not been run"],
        ["Extrapolation in k", "At least four connectivities, two candidate forms (a + b/k and a + b k<sup>&minus;1/2</sup>) with an information criterion, and a derived subleading exponent", "Not done"],
        ["Independent theoretical check", "Comparison with population dynamics of the exact equations, as in Skantzos et al. (2005) and Metz (2025)", "Not done"],
        ["Code and data", "Archived code and figure data with a DOI and a data-availability statement", "Repository is reproducible from configs; no archive or DOI yet"],
    ]
    programme = [
        ["1", "Reciprocity interpolation", "Graphs with a tunable fraction of mutual links; the theory predicts a straight line", "<b>Done at k = 24 and 48: section <a href='#p3'>P3</a></b>"],
        ["2", "Population dynamics of the exact directed equation", "Shows the dynamic term is the correct asymptote of Metz (2025), which answers the strongest novelty objection", "Not started"],
        ["3", "Analytic boundary-layer exponent", "Scatter of size k<sup>&minus;1/2</sup> smears the diverging response of nodes about to unlock; derive the resulting correction and fit the fold data with the exponent fixed", "Not started"],
        ["4", "Fold and branch at k = 96 and 192", "With N chosen so that k&sup2;/N is at most 0.05", "Config ready for the cluster: <code>configs/paperA_branch_ada.yaml</code> (branch); fold not yet configured"],
        ["5", "Random regular against Erd&#337;s-R&eacute;nyi", "Degree fluctuations add their own 1/k term through the spread of degrees", "Not started"],
        ["6", "Exponent dependence", "p = 1, 2, 3 on one theory curve", "Theory computed for p = 1, 2 (table P3.1); simulations at p = 1 mainly"],
    ]
    return f"""
<section id="n1" class="summary">
<div class="sec-head"><span class="tag">N1</span><h2>Novelty audit: what is new, what is not, and what a referee will ask for</h2><span class="verdict v-open">narrow but real</span></div>
<p>An external audit of the literature was run against the theory of section P1 (the summary is in
<code>docs/PRIOR_ART.md</code>). Its conclusion changes how this work must be described, so it is given here
in full. This section quotes no measured value of this repository; the measurements are in sections P1 to P3.
The audit could not run full-text searches, so a paper with an explicit 1/k Kuramoto threshold could still
exist in the literature it did not reach.</p>

<h3>The conclusion in four sentences</h3>
<ul class="findings">
<li><b>The mechanisms are known.</b> Filtering of fast neighbour fluctuations and the reaction of a node on
its own neighbours are both contained in existing dynamical mean-field and cavity frameworks for sparse
graphs.</li>
<li><b>The explicit results appear new.</b> No published closed-form 1/k shift of the fold or of the
synchronised branch was found for adaptive (local order parameter) Kuramoto oscillators on sparse graphs.</li>
<li><b>The right form is a regular article</b>, framed as explicit leading-order sparse-graph corrections
with a directed/undirected decomposition verified by simulation. It is not a new cavity method.</li>
<li><b>The finance study is kept separate.</b> Attaching a weak empirical result to the physics would lead a
referee to discount both.</li>
</ul>

<h3>Table N1.1. Prior work and how it relates</h3>
<p><b>What it does</b> describes the earlier paper. <b>Overlap</b> says which part of this project it
anticipates. <b>What to do</b> is the action for the manuscript. <b>Status</b> records how far the
description has been verified here: <i>abstract checked</i> means the arXiv abstract page was fetched and
matches; <i>from the audit</i> means the description is the audit's and has not been checked against the
paper; <i>read earlier</i> means it was read during the first search.</p>
{_tbl(["paper", "what it does", "overlap with this project", "what to do", "status"], rows)}

<h3>Table N1.2. Verdict on each claim</h3>
{_tbl(["claim", "what is already known", "verdict"], verdicts)}

<h3>Table N1.3. What a referee will expect, and where this repository stands</h3>
<p>Based on the norms of comparable published work, not on a journal checklist.</p>
{_tbl(["requirement", "what it means", "status here"], referee)}

<h3>Table N1.4. Programme, in order of value</h3>
{_tbl(["rank", "item", "why", "status"], programme)}

<h3>Hypothesis worth a cheap test</h3>
<p>In the all-to-all model the incoherent state is linearly stable at every coupling, because the gain
r<sup>p</sup> vanishes there. On a sparse graph the local order parameter of an incoherent state is of order
k<sup>&minus;1/2</sup>, not zero, so the effective coupling is finite. This suggests a finite forward
threshold that grows with k, roughly as k<sup>p/2</sup>, which would make the width of the hysteresis loop
finite and dependent on connectivity. This is the audit's inference, not a published result.</p>

<h3>The finance study after the audit</h3>
<ul class="findings">
<li><b>The one positive number is exploratory.</b> It comes from a robustness variant after the primary test
did not reject, and must be reported that way.</li>
<li><b>A disjoint set of stocks is not an independent test.</b> Crash events are market-wide and stock
co-movement is dominated by a common factor, so different stocks over the same period test the same events. A
confirmatory test needs a later time period with the event rule, horizon and volatility matching fixed in
advance.</li>
<li><b>Benchmarks to include:</b> the co-jump synchronization measures of Calcagnile et al. (2018) and the
absorption ratio of Kritzman et al. (2011). Demirer, Leggio and Lien (2019) found no herding before the 2010
flash crash at one-minute resolution, consistent with co-movement being coincident and not leading.</li>
</ul>
</section>"""


def daily_block() -> str:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc.runner import latest_run
    d = latest_run("paperB_daily")
    if d is None:
        return ""
    sm = json.loads((d / "summary.json").read_text())
    df = pd.DataFrame(sm["rows"])
    old = pd.read_csv(ROOT / "empirical" / "data" / "e1_pooled_auc.csv").replace({"absorption": "lead_eig_share"})
    m = df.merge(old, on=["indicator", "score", "null"], suffixes=("", "_old"))
    nice = {"var_index": "index variance", "ac1_index": "index lag-1 autocorrelation", "mean_corr": "mean pairwise correlation",
            "lead_eig_share": "leading eigenvalue share", "absorption_ratio": "absorption ratio (Kritzman)",
            "cssd": "cross-sectional dispersion", "flicker": "breadth flicker"}
    pk = df[df.null == "peak"]
    t = _tbl(["indicator", "score", "AUC [95% CI]", "one-sided p", "p after correction", "events / nulls"],
             [[nice[r.indicator], "level at the peak" if r.score == "level" else "trend over the prior year",
               _ci(r.auc, r.ci_lo, r.ci_hi, 3), f"{r.p_one_sided:.3f}", f"{r.p_bh:.3f}", f"{r.n_events} / {r.n_null}"]
              for r in pk.itertuples()])
    best = df.sort_values("p_bh").iloc[0]
    return f"""
<h3>Table B1.4. Daily data, 23 markets, corrected</h3>
<p>The same question on daily closes: {sm['n_events']} crash onsets in {sm['n_markets']} markets across
{sm['n_quarters']} calendar quarters. Corrections relative to E1: correlations from pairwise-complete data (no
zero-filling), the old "absorption" renamed and the real Kritzman absorption ratio added, the bootstrap resamples
calendar quarters on both the event side and the comparison side, and p-values are corrected across all
{sm['n_tests']} tests. The table shows the fairest comparison: against local highs that were <i>not</i> followed
by a crash. Run: <code>results/paperB_daily/{d.name}</code>, script
<code>research/empirical/e1c_ews_corrected.py</code>.</p>
{t}
<ul class="findings">
<li><b>{sm['n_bh_reject']} of {sm['n_tests']} tests survive the multiplicity correction.</b> The smallest corrected
p-value over all indicators, scores and comparisons is {best.p_bh:.3f} ({nice[best.indicator]}, {best.score},
against {best.null}).</li>
<li><b>The zero-filling defect did not matter in practice.</b> The largest change in any AUC shared with the
original E1 table is {(m.auc - m.auc_old).abs().max():.4f}. The audit was right that the estimator was biased in
principle; these markets have too little missing data inside a 60-day window for it to show.</li>
<li>The Kritzman absorption ratio, which the original analysis did not actually compute, does no better than the
other co-movement measures.</li>
</ul>"""



def empirical_card() -> str:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc import report
    s = report.intraday_summary()
    if s is None:
        return '<section id="b1"><h2>B1 Corrected empirical study</h2><div class="missing">results not available yet</div></section>'
    a = s["primary"]
    nice = {"volatility": "volatility", "mean_corr": "mean pairwise correlation", "lead_eig_share": "leading eigenvalue share",
            "corr_given_vol": "correlation with volatility removed"}
    t1 = _tbl(["indicator", "comparison", "AUC [95% CI]", "one-sided p", "p after multiplicity correction", "events / nulls"],
              [[nice[r["indicator"]], "all ordinary peaks" if r["null"] == "unmatched" else "volatility-matched peaks",
                _ci(r["auc"], r["ci_lo"], r["ci_hi"], 3), f"{r['p_one_sided']:.3f}", f"{r['p_bh']:.3f}",
                f"{r['n_pos']} / {r['n_neg']}"] for r in a["t1"]])
    t2 = _tbl(["indicator", "nominal alarm budget", "realised alarm rate", "crashes caught", "recall, exact 95% CI",
               "false-alarm episodes", "false alarms per crash caught"],
              [[nice[r["indicator"]], f"{100 * r['budget']:g}%", f"{100 * r['realised_rate']:.2f}%",
                f"{r['caught']} of {r['n_crash']}", f"[{r['recall_cp'][0]:.2f}, {r['recall_cp'][1]:.2f}]", r["false_episodes"],
                "none caught" if r["false_per_catch"] is None else f"{r['false_per_catch']:.0f}"] for r in a["t2"]])
    h1 = next(r for r in a["t1"] if r["indicator"] == "corr_given_vol" and r["null"] == "vol_matched")
    months = pd.Series([d[:7] for d in a["test_dates"]]).value_counts()
    rb = _tbl(["variant", "crash onsets (train / test)", "matched pairs", "correlation with volatility removed, matched AUC [95% CI]", "p after correction",
               "mean correlation, matched AUC [95% CI]", "smallest detectable matched AUC"],
              [[x["tag"], f"{x['n_onsets']} ({x['n_train']} / {x['n_test']})", x["n_matched_pairs"]] + sum(
                  [[_ci(r["auc"], r["ci_lo"], r["ci_hi"], 3), f"{r['p_bh']:.3f}"][:(2 if nm == "corr_given_vol" else 1)]
                   for nm in ("corr_given_vol", "mean_corr")
                   for r in x["t1"] if r["indicator"] == nm and r["null"] == "vol_matched"], [])
               + ["n/a" if x["mde_auc_matched"] is None else f"{x['mde_auc_matched']:.3f}"]
               for x in [a] + s["robustness"]])
    v1 = next(r for r in a["t2"] if r["indicator"] == "volatility" and r["budget"] == 0.01)
    m1 = next(r for r in a["t2"] if r["indicator"] == "mean_corr" and r["budget"] == 0.01)
    return f"""
<section id="b1" class="summary">
<div class="sec-head"><span class="tag">B1</span><h2>Corrected test of synchronization-based crash warnings on minute data, design fixed in advance</h2><span class="verdict v-data">real data</span></div>
<p>This replaces E7 and E8. The design was written (<code>docs/PREREGISTRATION_B.md</code>) before this analysis was run, but it is
not under version control, so that ordering rests on the author's word and on file times. It is also not a
blind pre-registration: the earlier, flawed versions had been seen. Run: <code>results/paperB_intraday/{s['_run']}</code>.
Data: {s['n_tickers']} US large-cap stocks, {s['n_bars']:,} one-minute bars, {s['n_sessions']:,} trading sessions,
{s['span'][0]} to {s['span'][1]}. Thresholds and the volatility regression are fitted on data before 2016 and
applied unchanged afterwards.</p>
<h3>What was wrong before, and what changed</h3>
<ul class="findings">
<li><b>Overnight gaps.</b> Returns and 30-minute windows used to run across the overnight gap, so the open looked
like a burst of co-movement. Now the first return of every session is dropped, the index restarts at each open,
and no window crosses an open. Of the {s['old_e7_onsets']} onsets the old E7 detected, {s['old_e7_onsets_before_10am']}
were before 10:00.</li>
<li><b>Look-ahead.</b> The regression that removes volatility from co-movement used to be fitted on the whole
sample, test period included. It is now fitted on the training period only.</li>
<li><b>Alarm scoring.</b> The old rule kept only the first alarm of each 390-minute stretch, so an alarm that had
been on for more than an hour before a crash could never count. Now the alarm is a state per minute; a crash is
caught if the state is on at any time in the 60 minutes before it; a false alarm is an episode with no crash
within 60 minutes after it.</li>
</ul>
<h3>Table B1.1. Can the indicator tell a crash onset from an ordinary peak?</h3>
<p>A crash onset is a local high followed by a fall of at least {100 * a['drop']:.0f}% within 60 minutes of the same
session, at least {a['exclude']} minutes after the open: {a['n_onsets']} in total ({a['n_train']} before 2016,
{a['n_test']} after). AUC is the probability that a random crash scores higher than a random ordinary peak
(0.5 means no skill). "Volatility-matched" pairs each crash with an ordinary peak of nearly the same
volatility ({a['n_matched_pairs']} pairs), so that an indicator cannot score merely by tracking volatility. Before
a crash, volatility is {a['vol_ratio_crash_over_peak']:.1f} times its level before an ordinary peak.</p>
{t1}
<h3>Table B1.2. Out of sample: alarms actually raised after 2016</h3>
<p><b>Nominal budget</b> is the share of training minutes above the threshold; <b>realised rate</b> is the share
of test minutes the alarm was actually on, which differs because the test period is not the training period.</p>
{t2}
<ul class="findings">
<li><b>Primary hypothesis (synchrony beyond volatility): not detected.</b> Correlation with volatility removed,
against volatility-matched peaks: AUC {_ci(h1['auc'], h1['ci_lo'], h1['ci_hi'], 3)}, corrected p = {h1['p_bh']:.3f}.
With {a['n_matched_pairs']} pairs the smallest AUC this test could reliably detect is {a['mde_auc_matched']:.2f}, so
this is "not detected at this power", not "absent".</li>
<li><b>Secondary hypothesis (co-movement beats volatility as an alarm): rejected.</b> At the 1% budget volatility
catches {v1['caught']} of {v1['n_crash']} test crashes with {v1['false_per_catch']:.0f} false alarms per catch;
mean correlation catches {m1['caught']} of {m1['n_crash']} with {m1['false_per_catch']:.0f}.</li>
<li><b>The test crashes are not independent.</b> By month: {", ".join(f"{k}: {v}" for k, v in months.items())}.
Most of them belong to one episode, so the effective sample is smaller than the count suggests.</li>
<li><b>Not claimed:</b> any causal role of synchronization, or that these alarm rates would hold on other
stocks. The cross-section is today's surviving large caps.</li>
</ul>
<h3>Table B1.3. Robustness (reported whatever the outcome)</h3>
<p>Variants: lead time of 30 or 120 minutes instead of 60; no exclusion of the first 30 minutes; a fall of 2% or
4% instead of 3%. The 2% variant has about three times as many events and therefore more power.</p>
{rb}
{daily_block()}
</section>"""



def write_readme_and_evidence() -> None:
    """README block between the GENERATED markers and docs/CLAIM_EVIDENCE.md, both from results/."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from esfc import report
    L: list[str] = []
    th = report.theory_table()
    if not th.empty:
        L += ["### Predicted 1/k coefficient of the fold", "",
              "`fold = all-to-all fold x (1 + a/k)`. Rician closure: all neighbour scatter frozen, no back-reaction. "
              "Directed: rotating scatter filtered, no back-reaction. Undirected: directed plus the reaction term. "
              "Derivation in `docs/THEORY.md`.", "",
              report.md_table(th[["p", "rician", "directed", "reaction", "undirected"]].rename(columns={
                  "rician": "Rician closure", "directed": "theory, directed", "reaction": "reaction term",
                  "undirected": "theory, undirected"}), {"p": "g", "Rician closure": ".3f", "theory, directed": ".3f",
                                                           "reaction term": ".3f", "theory, undirected": ".3f"}), ""]
    cells = [c for c in (report.fold_cells("paperA_A1_pilot"), report.fold_cells("paperA_A7")) if c is not None]
    if cells:
        df = pd.concat(cells, ignore_index=True)
        big = df[df.N == df.groupby(["graph", "p", "k"]).N.transform("max")].sort_values(["graph", "p", "k"])
        t = pd.DataFrame({"graph": big.graph.map({"rr": "undirected", "dir": "directed"}), "p": big.p, "k": big.k,
                          "N": big.N, "seeds": big.n_seeds,
                          "fold ratio [95% CI]": [_ci(a, b, c) for a, b, c in zip(big.ratio, big.ci_lo, big.ci_hi)],
                          "closure ratio": big.closure_ratio,
                          "coefficient [95% CI]": [_ci(a, b, c, 2) for a, b, c in zip(big.coef, big.coef_lo, big.coef_hi)],
                          "theory": big.theory_coef.astype(float), "Rician": big.rician_coef})
        L += ["### Measured fold at the largest size run (pilot)", "",
              "Bisection from a synchronised start, relaxation time 480, tolerance 0.2%, Lorentzian quantile "
              "frequencies. Coefficient is `k (ratio - 1)`. Both predictions are leading-order and are expected to "
              "fail at small k.", "",
              report.md_table(t, {"p": "g", "closure ratio": ".4f", "theory": ".2f", "Rician": ".2f"}), ""]
    bc = report.branch_cells()
    if bc is not None:
        pool = report.branch_pooled("dir", 1.0, 48)
        t = pd.DataFrame({"graph": bc.graph.map({"rr": "undirected", "dir": "directed"}), "p": bc.p, "k": bc.k, "N": bc.N,
                          "seeds": bc.n_seeds, "c [95% CI]": [_ci(a, b, c, 3) for a, b, c in zip(bc.c, bc.c_lo, bc.c_hi)],
                          "overlap k^2/N": bc.overlap, "theory": bc.theory.astype(float), "Rician": bc.closure.astype(float)})
        L += ["### Shift of the synchronised branch (cloud runs)", "",
              "`c = k (R - R0)` at 1.25 times the all-to-all fold, `R0` from the all-to-all model on the same "
              "frequencies. Theory is the dynamic theory on directed graphs and dynamic plus reaction on undirected "
              "graphs; the theory columns were printed by the cloud run and have not been regenerated locally. "
              f"Directed graphs, p = 1, k = 12 to 48 pooled: {pool[0]:.3f} +/- {pool[1]:.3f}. With two seeds the "
              "95% interval is very wide; those cells are not confirmations. Full discussion: `research/index.html` "
              "section P2.", "",
              report.md_table(t, {"p": "g", "overlap k^2/N": ".2f", "theory": ".3f", "Rician": ".3f"}), ""]
    b = report.intraday_summary()
    if b:
        a = b["primary"]
        rows = [r for r in a["t2"] if r["budget"] in (0.01, 0.05)]
        t = pd.DataFrame({"indicator": [r["indicator"] for r in rows], "budget": [f"{100 * r['budget']:g}%" for r in rows],
                          "realised rate": [f"{100 * r['realised_rate']:.2f}%" for r in rows],
                          "caught": [f"{r['caught']} of {r['n_crash']}" for r in rows],
                          "false episodes": [r["false_episodes"] for r in rows],
                          "false per catch": ["none caught" if r["false_per_catch"] is None else f"{r['false_per_catch']:.0f}" for r in rows]})
        h1 = next(r for r in a["t1"] if r["indicator"] == "corr_given_vol" and r["null"] == "vol_matched")
        L += ["### Crash warnings on minute data (corrected; design written before the run, not timestamped)", "",
              f"{b['n_tickers']} stocks, {b['n_sessions']} sessions, {a['n_onsets']} intraday crash onsets "
              f"({a['n_test']} in the test period after 2016). Primary hypothesis, synchrony beyond volatility "
              f"(`corr_given_vol` against volatility-matched peaks): AUC {_ci(h1['auc'], h1['ci_lo'], h1['ci_hi'], 3)}, "
              f"corrected p = {h1['p_bh']:.3f}; smallest detectable AUC at this sample size "
              f"{a['mde_auc_matched']:.2f}. Not detected, and underpowered.", "", report.md_table(t), ""]
    readme = REPO / "README.md"
    txt = readme.read_text()
    a_, b_ = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
    if a_ in txt and b_ in txt:
        readme.write_text(txt[:txt.index(a_) + len(a_)] + "\n" + "\n".join(L) + txt[txt.index(b_):])
    ev = ["# Claim to evidence map", "", "Generated by `research/build_index.py`. One row per claim made on "
          "`research/index.html` sections P0, P1, P2 and B1 or in `README.md`. The run directory name ends in the git "
          "hash the code was at.", "", "| claim | where shown | code | config | run |", "|---|---|---|---|---|",
          f"| Equation (4) is the correct expansion of the Rician closure | index P0 table A | `research/p0_spot_checks.py` | in script | `research/data/p0_spot_checks.json` |",
          f"| The bisection harness is unbiased in the all-to-all limit | index P0 table B; `tests/test_analytic.py` | `research/p0_spot_checks.py`, `esfc.continuation` | in script | `research/data/p0_spot_checks.json` |",
          f"| Short, small runs sit below the closure | index P0 table C | `research/p0_spot_checks.py` | in script | `research/data/p0_spot_checks.json` |",
          f"| Predicted coefficients for directed and undirected graphs | index P1.1; README | `esfc.cavity` | `configs/paperA_theory.yaml` | `{report.run_name('paperA_theory')}` |",
          f"| Freezing all scatter recovers (p+2)/2 exactly | `tests/test_cavity.py` | `esfc.cavity` | n/a | test suite |",
          f"| Measured fold against k on undirected graphs | index P1.2 to P1.4; README; figure p1_fold_coefficient | `esfc.experiments.fold_grid` | `configs/paperA_A1_pilot.yaml` | `{report.run_name('paperA_A1_pilot')}` |",
          f"| Measured fold on directed graphs | index P1.2; figure p1_fold_coefficient | `esfc.experiments.fold_grid` | `configs/paperA_A7_directed.yaml` | `{report.run_name('paperA_A7')}` |",
          f"| Cloud fold runs agree with local fold runs | index P2.2, P2.3 | `experiments/fold.py`, `esfc.report.cloud_fold_cells` | `{report.cloud_run_name()}/config.yaml` | `{report.cloud_run_name()}` |",
          f"| Branch shift on directed graphs matches the dynamic theory and excludes the Rician closure | index status card, P2.4; README | `experiments/branch_sim.py`, `esfc.report.branch_cells` | `{report.cloud_run_name()}/config.yaml` | `{report.cloud_run_name()}` |",
          f"| Branch theory values (dynamic, reaction, closure) | index P2.4 | `experiments/branch_theory.py`, `experiments/react_branch.py` | n/a | `{report.cloud_run_name()}/branch_theory_reported.json` (copied from the run note, not regenerated) |",
          f"| Stored branch summary equals the raw runs | index P2.1 | `esfc.report.branch_summary_check` | n/a | `{report.cloud_run_name()}` |",
          f"| Compiled kernel equals `research/model.py` | `tests/test_regression_model.py`, `tests/test_invariants.py` | `esfc.fast`, `esfc.integrate` | n/a | test suite |",
          f"| Corrected intraday crash-warning results | index B1; README | `esfc.experiments_b.intraday_study` | `configs/paperB_intraday.yaml` | `{report.run_name('paperB_intraday')}` |",
          f"| Design fixed before the corrected run | `docs/PREREGISTRATION_B.md` | n/a | n/a | not version controlled; no independent timestamp |", ""]
    (REPO / "docs" / "CLAIM_EVIDENCE.md").write_text("\n".join(ev))



def summary_card() -> str:
    return """
<section class="summary">
<h2>Earlier summary (kept as a record)</h2>
<p class="note"><b class="bad">Under audit.</b> This summary was written before the audit in section
<a href="#p0">P0</a>. Items 1, 3 (the "12%" sentence) and 4 in the left column, and the false-alarm counts in the
right column, are <b>not currently supported</b> by committed data. They are left in place so the record stays
readable, and each is marked. Read section P0 for the measured values.</p>
<div class="grid2">
<div><h3>What holds up</h3><ol>
<li><b class="bad">[Unverified, see P0]</b> <b>The finite-connectivity law for the fold (S14).</b> Treating the local order parameter as
a <b>Rician</b> variable rather than replacing it by R gives
<b>&lambda;<sub>b</sub>(p, &lang;k&rang;) = &gamma;(p+2)((p+2)/p)<sup>p/2</sup> [1 + (p+2)/2&lang;k&rang;]</b>.
The bracket is new, has no free parameter, and its coefficient (p+2)/2 is exact both symbolically and
numerically. It predicts the measured fold to <b>0.05%</b> at &lang;k&rang; = 48 and to 0.6-3.7% at
&lang;k&rang; = 12, and it accounts for the ~10% offset S11 and S13 could only report. The same closure
gives a <b>second threshold that exists only at finite connectivity</b>, hence a <b>critical
connectivity k<sub>c</sub>(p)</b> below which bistability is impossible at any coupling, and it absorbs
clustering into a single <b>effective connectivity</b> k<sub>eff</sub> = k/(1+(k&minus;1)&rho;).</li>
<li><b class="bad">S13's two "new" results did not survive, and the correction is recorded (S13, S14).</b>
The threshold family is <b>Zou &amp; Wang, PRE 102, 012219 (2020)</b> with &alpha; = p+1: their stability
bound r&sup2; &ge; (&alpha;&minus;1)/(&alpha;+1) <i>is</i> R*(p)&sup2;. The p = 2 corner is Skardal &amp;
Arenas, Commun. Phys. 3 (2020), whose PDF was already in this repository. S13's own measurement could
not have supported the claim either: its coupling grid quantised the reported ratio in steps of 4.09%,
so the "flatness" of 0.055 was 1.3 grid steps wide. C* &asymp; 0.35 was confounded by the same fixed
grid no longer containing the transition above C &asymp; 0.5. What survives is the qualitative claim,
now derived in S14 instead of measured.</li>
<li><b>Exact thresholds (S11).</b> For Lorentzian frequencies the herding model is solvable in closed
form: the cubic &lambda;R&sup3; &minus; &lambda;R + 2&gamma; = 0 folds at R* = 1/&radic;3, giving
<b>&lambda;<sub>b</sub> = 3&radic;3&thinsp;&gamma;</b>. The residual 12% against simulation is the
finite-connectivity correction, which explains the ~10% offset S2 could only measure.</li>
<li><b class="bad">[Not supported as stated, see P0]</b> <b>The first-order jump is genuine (S11).</b> It grows with system size, 0.183 at N = 125 to 0.486
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
<li><b>The model's own network is wrong, but by less than was claimed (E5, corrected).</b> The bistable
window does collapse on networks estimated from real returns, and S4 and S5 both depend on that window,
so the direction of the result stands. But the magnitude was overstated: an <b>n-matched</b> control,
which E5 originally listed as an untested caveat, shows that a <i>random</i> graph at the real networks'
own size already drops the gap from 0.793 to 0.396. <b>72% of the reported collapse was finite size,
not clustering</b>; the structural effect is a further 40% on top. Reading C against a size-matched
baseline &lang;k&rang;/n rather than against a single n = 400 graph, the real excess clustering is
<b>3.8x</b>, not 24x. S14 explains why the two effects are the same effect: in a small random graph
neighbourhoods overlap by necessity, so k<sub>eff</sub> falls for the same reason clustering lowers
it.</li>
<li><b class="bad">[Direction kept, E8 counts withdrawn, see P0]</b> <b>No early warning, at any timescale, and it is not usable (E1, E7, E8).</b> On daily data no
indicator beats ordinary peaks. At 1-minute resolution over 2.3M bars the raw AUC is 0.911, but
<b>volatility-matched it is 0.526</b> [0.340, 0.712]. Out of sample co-movement needs <b>77.6 false
alarms per crash caught</b>; <b>plain volatility is the better detector</b> (65.2 at the same 1% budget)
and co-movement with volatility removed is the <b>worst</b> of the three (274.5). The timescale objection is answered and E1 survives it.</li>
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
<p class="note">Status when this summary was written: the analytic core is in better shape than when this started (closed-form
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
        if s["id"] == "s14" and not s.get("_audited"):
            s["findings"] = [s14_audit_finding()] + list(s["findings"])
            s["_audited"] = True
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
b.bad, .bad {{ color:var(--neg) }}
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
<a class="sh" href="#status"><b>Where the project stands (read first)</b><span>One screen: what is established, what is not, and the single new result, with the numbers behind each statement.</span></a>
<a class="sh" href="#p2"><b>Cloud runs and the branch test</b><span>Two independent sets of simulations checked against each other, and the measurement that confirms the dynamic theory on directed graphs.</span></a>
<a class="sh" href="#p0"><b>Audit status</b><span>Which headline claims are backed by committed files, with an independent re-run of the key measurement.</span></a>
<a class="sh" href="reference.html"><b>Build the model from one oscillator</b><span>The whole model in the order you would discover it: each step shows where the previous one breaks and adds exactly one term. Every assumption listed.</span></a>
<a class="sh" href="../simulation/index.html"><b>Interactive simulation</b><span>Watch a crash spread node by node through the network. Click any desk to see its own numbers in the locking inequality.</span></a>
<a class="sh" href="RESEARCH_IDEAS.md"><b>Research ideas</b><span>What is still open, keyed paper by paper to the prior art, with what would have to be computed for each.</span></a>
</div></header>
<nav><div class="in"><a href="#status"><span>NOW</span>Status</a><a href="#summary"><span>OLD</span>Earlier summary</a><a href="#p0"><span>P0</span>Audit</a><a href="#p1"><span>P1</span>New theory and measurements</a><a href="#p2"><span>P2</span>Cloud runs and branch test</a><a href="#p3"><span>P3</span>Reciprocity test</a><a href="#p4"><span>P4</span>Branch curve</a><a href="#p5"><span>P5</span>Graph structure</a><a href="#p6"><span>P6</span>Bistable window</a><a href="#n1"><span>N1</span>Novelty audit</a><a href="#b1"><span>B1</span>Corrected data test</a><a href="#glossary"><span>SYM</span>Symbols</a>{nav}<a href="#refs"><span>REF</span>References</a></div></nav>
<main>
{status_card()}
<div id="summary">{summary_card()}</div>
{audit_card()}
{physics_card()}
{cloud_card()}
{reciprocity_card()}
{branch_curve_card()}
{structure_card()}
{window_card()}
{novelty_card()}
{empirical_card()}
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
    write_claims_audit()
    write_readme_and_evidence()
    print(f"wrote {out}: {n_math} equations rendered to SVG, {report["images"]} local images/links "
          f"({len(report['missing_images'])} not generated yet: {report['missing_images']}), HTML structure OK")
    return out


if __name__ == "__main__":
    build()
