// Builds presentation/Explosive_Synchronization.pptx: black background, white text, LaTeX equations.
// Run: node presentation/build_deck.js   (equation and figure images come from make_equations.py and
// make_dark_figures.py, written to presentation/assets/).
const path = require("path");
const pptxgen = require("pptxgenjs");
const sizeOf = require("image-size").imageSize || require("image-size");

const A = (f) => path.join(__dirname, "assets", f);
const REPO = path.join(__dirname, "..");
const FIG_SYNC = A("fig_sync_dark.png");
const FIG_HERD = A("fig_herding_dark.png");
const FIG_FOLD = A("fig_fold_dark.png");
const OUT = path.join(__dirname, "Explosive_Synchronization.pptx");

const W = 13.333, MX = 0.65, CW = W - 2 * MX, TOP = 1.65, BOTTOM = 6.85;
const WHITE = "FFFFFF", GREY = "BFBFBF", DIM = "7A7A7A", RULE = "4D4D4D";
const BODY = 18, SMALL = 15, FONT = "Calibri";
const EM = (20 * 400) / 72; // pixel size of one em in the equation images

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Explosive Synchronization";
pres.author = "Athmeeya M Kashyap";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

let page = 0;
const warn = (msg) => console.log("WARNING:", msg);

function lines(text, w, pt) {
  const cpl = Math.max(1, Math.floor(w / (pt * 0.0072)));
  return text.split("\n").reduce((n, para) => n + Math.max(1, Math.ceil(para.length / cpl)), 0);
}
function textH(text, w, pt, spacing = 1.22) { return (lines(text, w, pt) * pt * spacing) / 72; }

function text(s, str, x, y, w, opts = {}) {
  const pt = opts.fontSize || BODY;
  const h = opts.h || textH(typeof str === "string" ? str : str.map((r) => r.text).join(""), w, pt) + 0.05;
  s.addText(str, { x, y, w, h, fontFace: FONT, fontSize: pt, color: WHITE, margin: 0, valign: "top",
    isTextBox: true, lineSpacingMultiple: 1.05, ...opts });
  return y + h;
}
function eq(s, name, x, y, pt) {
  const d = sizeOf(A(`eq_${name}.png`));
  const w = (d.width / EM) * (pt / 72), h = (d.height / EM) * (pt / 72);
  s.addImage({ path: A(`eq_${name}.png`), x, y, w, h, altText: `Equation ${name}` });
  return { w, h, bottom: y + h };
}
let lastSection = null;
function base(section, title, notes, trackerActive) {
  if (section !== lastSection) { pres.addSection({ title: section }); lastSection = section; }
  const s = pres.addSlide({ sectionTitle: section });
  s.background = { color: "000000" };
  page += 1;
  if (trackerActive) {
    const items = ["Motivation", "Background", "Literature", "Market model", "Crash and recovery"];
    s.addText(items.map((t, i) => ({ text: t + (i < items.length - 1 ? "      " : ""),
      options: { color: t === trackerActive ? WHITE : DIM, bold: t === trackerActive } })),
      { x: MX, y: 0.35, w: 8, h: 0.3, fontFace: FONT, fontSize: 14, margin: 0, isTextBox: true });
  }
  if (title) s.addText(title, { x: MX, y: 0.75, w: CW, h: 0.65, fontFace: FONT, fontSize: 32, bold: true,
    color: WHITE, margin: 0, isTextBox: true, valign: "top" });
  if (page > 1) s.addText(String(page), { x: W - MX - 0.6, y: 6.98, w: 0.6, h: 0.3, fontFace: FONT,
    fontSize: 12, color: DIM, align: "right", margin: 0, isTextBox: true });
  if (notes) s.addNotes(notes);
  s._name = title || "cover";
  return s;
}
function check(s, y) { if (y > BOTTOM + 0.01) warn(`slide ${page} "${s._name}" ends at ${y.toFixed(2)} in`); }
function footer(s, str) {
  s.addText(str, { x: MX, y: 6.98, w: 10, h: 0.3, fontFace: FONT, fontSize: 12, color: GREY, margin: 0, isTextBox: true });
}
// rows of [equation symbol | sentence]; returns bottom y
function termRows(s, rows, x, y, symW, w, symPt = 24, pt = BODY, gap = 0.14) {
  for (const r of rows) {
    const tw = w - symW;
    const th = textH(r.text, tw, pt);
    let ih = 0;
    if (r.eq) {
      const d = sizeOf(A(`eq_${r.eq}.png`)); ih = (d.height / EM) * (symPt / 72);
      const rowH = Math.max(ih, th);
      eq(s, r.eq, x, y + (rowH - ih) / 2, symPt);
      text(s, r.text, x + symW, y + (rowH - th) / 2, tw, { fontSize: pt });
      y += rowH + gap;
    } else {
      const lh = textH(r.label, symW - 0.2, pt);
      const rowH = Math.max(lh, th);
      text(s, r.label, x, y, symW - 0.2, { fontSize: pt, bold: true });
      text(s, r.text, x + symW, y, tw, { fontSize: pt });
      y += rowH + gap;
    }
  }
  return y;
}

// ---------------------------------------------------------------- 1 Title
{
  const s = base("Opening", null, "Introduce yourself and your advisor. I will explain how a group of interacting units can switch suddenly from disorder to order, starting from the basic model of synchronization, and what this means for financial networks.");
  s.addText("Explosive Synchronization", { x: MX, y: 2.2, w: CW, h: 1.1, fontFace: FONT, fontSize: 54, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Adaptive coupling in financial networks", { x: MX, y: 3.3, w: CW, h: 0.6, fontFace: FONT, fontSize: 26, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Athmeeya M Kashyap", { x: MX, y: 4.6, w: CW, h: 0.45, fontFace: FONT, fontSize: 22, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Advisor: Chittaranjan Hens", { x: MX, y: 5.1, w: CW, h: 0.4, fontFace: FONT, fontSize: 18, color: GREY, margin: 0, isTextBox: true });
  s.addText("IIIT Hyderabad", { x: MX, y: 5.5, w: CW, h: 0.4, fontFace: FONT, fontSize: 18, color: GREY, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 2 Motivation
{
  const s = base("Motivation", "Motivation", "Think about fireflies flashing together, or an audience gradually finding the same clapping rhythm. Each individual starts with its own timing, but interactions can bring them into step. Traders in a market watch and copy each other in a similar way. What interests me is that this change is not always gradual. Sometimes the system switches very suddenly, and the structure of the network decides how.", "Motivation");
  let y = text(s, "Many natural and social systems are made of units that each have their own rhythm but influence one another.", MX, TOP, CW);
  y += 0.3;
  const cards = [
    ["Fireflies", "In parts of Southeast Asia, thousands of fireflies flash at the same moment."],
    ["Applause", "A clapping audience often falls into a common rhythm."],
    ["Neurons", "Groups of neurons fire together, both in normal brain rhythms and in epileptic seizures."],
    ["Financial networks", "Traders watch one another and often copy each other’s decisions."],
  ];
  const cw = (CW - 0.6) / 2;
  cards.forEach(([h, t], i) => {
    const x = MX + (i % 2) * (cw + 0.6), yy = y + Math.floor(i / 2) * 1.35;
    s.addShape(pres.shapes.LINE, { x, y: yy, w: cw, h: 0, line: { color: RULE, width: 1 } });
    text(s, h, x, yy + 0.15, cw, { fontSize: 22, bold: true });
    text(s, t, x, yy + 0.55, cw, { color: "D9D9D9" });
  });
  y += 2.85;
  y = text(s, "In some of these systems the change from disorder to order is gradual, and in others it is sudden. This talk explains the sudden case, and how the network that connects the units affects it.", MX, y, CW);
  check(s, y);
}

// ---------------------------------------------------------------- 3 Oscillators
{
  const s = base("Background", "Oscillators", "An oscillator is anything that repeats a cycle. Its phase theta is where it is in the cycle, an angle between 0 and 360 degrees. Its natural frequency omega is the speed it would have on its own, and in a real population every oscillator has a slightly different one. The coupling strength K sets how strongly they pull on each other. For a pendulum the phase is the angle of the swing, for a heart cell the point in the beat, for a firefly the time since its last flash, and for a trader it can be read as the position in a cycle of buying and selling.", "Background");
  let y = text(s, "An oscillator is anything that repeats a cycle. Three quantities describe it.", MX, TOP, CW);
  y += 0.3;
  const lw = 7.4;
  const y1 = termRows(s, [
    { label: "Phase θ", text: "The phase is the position of the oscillator in its cycle, measured as an angle from 0 to 360°." },
    { label: "Natural frequency ω", text: "The natural frequency is the speed the oscillator would have if it were left alone. Every oscillator has a slightly different one." },
    { label: "Coupling K", text: "The coupling strength sets how strongly the oscillators pull on one another." },
  ], MX, y, 2.6, lw, 24, BODY, 0.3);
  const rx = MX + lw + 0.5, rw = CW - lw - 0.5;
  s.addShape(pres.shapes.LINE, { x: rx - 0.25, y, w: 0, h: 3.4, line: { color: RULE, width: 1 } });
  let y2 = text(s, "EXAMPLES", rx, y, rw, { fontSize: 14, bold: true, color: GREY, charSpacing: 2 });
  for (const t of ["For a pendulum, the phase is the angle of the swing.", "For a heart cell, it is the point reached in the beat.", "For a firefly, it is the time since its last flash.", "For a trader, it can be read as the position in a cycle of buying and selling."]) {
    y2 = text(s, t, rx, y2 + 0.18, rw);
  }
  check(s, Math.max(y1, y2));
}

// ---------------------------------------------------------------- 4 Kuramoto
{
  const s = base("Background", "The Kuramoto model (1975)", "The idea behind the Kuramoto model is fairly simple. Every oscillator has its own natural frequency, but it also feels the pull of the others. The sine term says how the phase difference sets that pull: if a neighbour is ahead you speed up, if it is behind you slow down, and in step there is no pull. When the coupling is weak, the oscillators mostly follow their own rhythms. As we increase the coupling, they start locking to one another.", "Background");
  let y = eq(s, "k_eq", MX, TOP, 28).bottom + 0.25;
  const colW = (CW - 0.5) / 2;
  const yl = termRows(s, [
    { eq: "k_th", text: "is the phase of oscillator i." },
    { eq: "k_dth", text: "is the rate at which that phase changes, which is the actual speed of the oscillator." },
    { eq: "k_om", text: "is its natural frequency." },
    { eq: "k_K", text: "is the coupling strength." },
  ], MX, y, 1.5, colW, 24, BODY, 0.08);
  const yr = termRows(s, [
    { eq: "k_N", text: "is the number of oscillators, so dividing by N turns the sum into an average." },
    { eq: "k_sin", text: "is the pull from oscillator j. It is positive when j is ahead, so i speeds up, and negative when j is behind, so i slows down. It is zero when the two are in step." },
  ], MX + colW + 0.5, y, 2.1, colW, 24, BODY, 0.08);
  y = Math.max(yl, yr) + 0.1;
  y = text(s, "With weak coupling every oscillator runs at its own pace. With strong enough coupling they lock together.", MX, y, CW, { color: "D9D9D9" });
  check(s, y);
}

// ---------------------------------------------------------------- 5 Order parameter
{
  const s = base("Background", "The order parameter", "To measure how synchronized the oscillators are, we use the order parameter. Each oscillator is represented by an arrow of length one pointing at its phase. Averaging all the arrows gives a new arrow. Its length R is 0 when the phases are spread out and the arrows cancel, and 1 when all oscillators are in step. Its direction psi is the average phase of the group. The local order parameter r i is the same average taken only over the neighbours of node i, and it will be important later.", "Background");
  let y = eq(s, "o_eq", MX, TOP, 28).bottom + 0.3;
  const colW = (CW - 0.5) / 2;
  const yl = termRows(s, [
    { eq: "o_e", text: "is an arrow of length one pointing at the phase of oscillator j, which is the point (cos θ, sin θ) on a circle. Here i is the imaginary unit." },
    { eq: "o_avg", text: "averages these arrows over all N oscillators." },
    { eq: "o_psi", text: "is the direction of the average arrow, which is the average phase of the group." },
  ], MX, y, 1.4, colW, 24, BODY, 0.12);
  const yr = termRows(s, [
    { eq: "o_R", text: "is the length of the average arrow. It is 0 when the phases are spread out and 1 when all oscillators are in step." },
    { eq: "o_r", text: "is the same average taken only over the neighbours of node i. It is called the local order parameter and is used later in the talk." },
  ], MX + colW + 0.5, y, 1.0, colW, 24, BODY, 0.12);
  check(s, Math.max(yl, yr));
}

// ---------------------------------------------------------------- 6 Mean-field form
{
  const s = base("Background", "The mean-field form of the model", "The order parameter makes the model much simpler. Multiply its definition by the complex conjugate of oscillator i's arrow and take the imaginary part. The average of the sines of theta j minus theta i becomes R times the sine of psi minus theta i. So the Kuramoto model becomes d theta i by d t equals omega i plus K R sine of psi minus theta i. Each oscillator only feels the group, through a pull of strength K R towards the average phase psi. Because the pull is proportional to R, synchrony reinforces itself. But R is also produced by the oscillators themselves, so it must be consistent with the motion it causes. That condition is solved on the next slide.", "Background");
  let y = text(s, "Multiplying the definition of R by the complex conjugate of the arrow of oscillator i, and taking the imaginary part, turns the sum over all oscillators into a single term.", MX, TOP, CW, { color: "D9D9D9" });
  y = eq(s, "m_eq1", MX, y + 0.2, 28).bottom + 0.25;
  y = text(s, "The Kuramoto model therefore becomes the following equation.", MX, y, CW, { color: "D9D9D9" });
  y = eq(s, "m_eq2", MX, y + 0.2, 28).bottom + 0.3;
  y = text(s, "Each oscillator is pulled towards the average phase ψ with a strength KR. The pull grows as the group becomes more synchronized, so synchrony reinforces itself. R must also be consistent with the motion it produces, which is the condition solved on the next slide.", MX, y, CW);
  check(s, y);
}

// ---------------------------------------------------------------- 7 Locked and drifting
{
  const s = base("Background", "Locked and drifting oscillators", "Measured relative to the group, an oscillator is locked when its phase stops changing. Setting its speed to zero in the mean-field equation gives sine of theta i minus psi equals omega i over K R. A sine can never be larger than one, so this only works when the size of the natural frequency is at most K R. Those oscillators are slow enough to be held by the pull, and they lock. Faster ones keep drifting around the circle, and their contributions cancel on average, so only locked oscillators add to R. R must then equal what the locked oscillators produce, which is the self-consistency equation, where g is the distribution of natural frequencies. Besides R equal to zero, it has a solution only above a critical coupling, 2 over pi times g of zero. For a Lorentzian distribution of width gamma the threshold is K equal to 2 gamma, and above it R is the square root of one minus 2 gamma over K. This is why ordinary synchronization starts at a threshold and then grows gradually. Locked does not mean that every oscillator has the same phase. Locked oscillators rotate together with the group but keep fixed, different phase offsets, with slower oscillators closer to the group phase.", "Background");
  const lw = 6.4, rx = MX + lw + 0.6, rw = CW - lw - 0.6;
  let y = text(s, "Measured relative to the group, an oscillator is locked when its phase stops changing. Setting its speed to zero in the mean-field equation gives the condition below.", MX, TOP, lw, { color: "D9D9D9" });
  y = eq(s, "l_eq1", MX, y + 0.15, 26).bottom + 0.15;
  y = text(s, "A sine cannot be larger than 1, so locking is possible only when the following holds.", MX, y, lw);
  y = eq(s, "l_eq2", MX, y + 0.12, 26).bottom + 0.15;
  y = text(s, "Oscillators that satisfy this condition lock to the group. Faster ones keep drifting around the circle and, on average, do not contribute to R.", MX, y, lw);
  s.addShape(pres.shapes.LINE, { x: rx - 0.3, y: TOP, w: 0, h: 4.9, line: { color: RULE, width: 1 } });
  let y2 = text(s, "SELF-CONSISTENCY", rx, TOP, rw, { fontSize: 14, bold: true, color: GREY, charSpacing: 2 });
  y2 = text(s, "R must equal the contribution of the locked oscillators, where g(ω) is the distribution of natural frequencies.", rx, y2 + 0.15, rw);
  y2 = eq(s, "l_int", rx, y2 + 0.2, 22).bottom + 0.2;
  y2 = text(s, "A solution with R > 0 exists only above a critical coupling. For a Lorentzian distribution of width γ, the threshold and the solution above it are given below.", rx, y2, rw);
  y2 = eq(s, "l_lor", rx, y2 + 0.2, 22).bottom;
  check(s, Math.max(y, y2));
}

// ---------------------------------------------------------------- 8 Networks
{
  const s = base("Background", "Networks", "So far every oscillator has felt every other one. In a network, each oscillator sits on a node, and a link means that one node feels the pull of another. The degree of a node is its number of links. In a random, or Erdos-Renyi, network the links are placed at random and most nodes have a similar degree. In a scale-free, or Barabasi-Albert, network a few hubs have very many links and most nodes have only a few. In a random regular network every node has exactly k neighbours chosen at random. Finally, a link can be two-way, where both nodes listen to each other, or one-way, where only one of them listens, as when a small trader follows a large fund.", "Background");
  const y = termRows(s, [
    { label: "Nodes and links", text: "Each oscillator sits on a node, and a link means that one node feels the pull of another." },
    { label: "Degree k", text: "The degree of a node is its number of links, which is the number of neighbours it listens to." },
    { label: "Random networks", text: "In a random (Erdős-Rényi) network, links are placed at random and most nodes have a similar degree." },
    { label: "Scale-free networks", text: "In a scale-free (Barabási-Albert) network, a few hubs have very many links and most nodes have only a few." },
    { label: "Random regular networks", text: "In a random regular network, every node has exactly k neighbours chosen at random." },
    { label: "One-way and two-way links", text: "A link can be two-way, where both nodes listen to each other, or one-way, where only one of them listens." },
  ], MX, TOP, 3.6, CW, 24, BODY, 0.2);
  check(s, y);
}

// ---------------------------------------------------------------- 9 Financial networks
{
  const s = base("Background", "Financial networks as networks of oscillators", "The same model can be read as a financial network, and I use this picture throughout the talk. Each oscillator is a trader or a trading desk. The phase is the trader's position in a cycle of buying and selling, and the sine of the phase is its net order, positive when it buys and negative when it sells. The natural frequency stands for the trader's own strategy and speed, and the coupling for how strongly traders copy each other. The network says who watches whom, and a small trader following a large fund that does not follow back is a one-way link. The order parameter is market consensus: near zero, buy and sell orders balance, and near one, most traders act together. Herding, where a trader copies more when the traders it watches already agree, is exactly the adaptive coupling introduced in the literature review. This is a modelling abstraction. Real traders do not literally follow sinusoidal cycles; the picture is used to interpret the model, and testing it on market data is future work.", "Background");
  let y = TOP;
  text(s, "IN THE MODEL", MX, y, 3.6, { fontSize: 14, bold: true, color: GREY, charSpacing: 2 });
  text(s, "IN A FINANCIAL NETWORK", MX + 3.6, y, 6, { fontSize: 14, bold: true, color: GREY, charSpacing: 2 });
  y = termRows(s, [
    { label: "Oscillator", text: "Each oscillator is a trader or a trading desk." },
    { label: "Phase θ", text: "The phase is the trader’s position in a cycle of buying and selling. The net order is sin θ, which is positive when the trader buys and negative when it sells." },
    { label: "Natural frequency ω", text: "The natural frequency stands for the trader’s own strategy and trading speed." },
    { label: "Coupling", text: "The coupling measures how strongly traders copy one another." },
    { label: "Network links", text: "The links describe who watches whom. A small trader may follow a large fund that does not follow back, which is a one-way link." },
    { label: "Order parameter R", text: "R measures market consensus. Near 0 the buy and sell orders balance, and near 1 most traders act together." },
    { label: "Adaptive coupling", text: "Adaptive coupling describes herding, in which a trader copies more when the traders it watches already agree. It is introduced in the literature review." },
  ], MX, y + 0.38, 3.6, CW, 24, 17, 0.07);
  check(s, y);
}

// ---------------------------------------------------------------- 10 Financial research
{
  const s = base("Background", "Synchronization in financial research", "This model connects to an existing line of financial research. In 1999 Mantegna used the correlations between stock returns to build a network of stocks, a minimum spanning tree that shows the hierarchical structure of the market, and describing a market as a network is now a standard tool. In 2000 Cont and Bouchaud showed that herding, where traders form groups that act together, can produce the large price fluctuations seen in real markets. Most directly, in 2025 Lee and co-workers in PNAS measured how close 39 national stock markets were to explosive synchronization before the 2008 financial crisis, and found that this proximity was associated with how quickly the markets collapsed and recovered. This is an association, not proof of cause. Together these works describe markets as networks of coupled units that can switch suddenly. The Kuramoto model with adaptive coupling is a simple version of this picture that can be solved exactly, which is what the rest of the talk does.", "Background");
  let y = TOP;
  const items = [
    ["Markets as networks", "Mantegna, Eur. Phys. J. B 11, 193 (1999)", "Mantegna used the correlations between stock returns to build a network of stocks, a minimum spanning tree that reveals the hierarchical structure of the market. Describing a market as a network is now a standard tool."],
    ["Herding", "Cont and Bouchaud, Macroecon. Dyn. 4, 170 (2000)", "Cont and Bouchaud showed that herding, in which traders form groups that act together, can produce the large price fluctuations seen in real markets."],
    ["Explosive synchronization in markets", "Lee et al., PNAS (2025)", "Lee and co-workers measured how close 39 stock markets were to explosive synchronization before the 2008 financial crisis, and found that this proximity was associated with how quickly the markets collapsed and recovered."],
  ];
  const lw = 4.0, tw = CW - lw;
  for (const [h, c, t] of items) {
    s.addShape(pres.shapes.LINE, { x: MX, y, w: CW, h: 0, line: { color: RULE, width: 1 } });
    let yl = text(s, h, MX, y + 0.12, lw - 0.3, { bold: true, fontSize: 20 });
    yl = text(s, c, MX, yl + 0.04, lw - 0.3, { fontSize: SMALL, color: GREY });
    const yt = text(s, t, MX + lw, y + 0.12, tw);
    y = Math.max(yl, yt) + 0.25;
  }
  y = text(s, "These works describe markets as networks of coupled units that can switch suddenly. The Kuramoto model with adaptive coupling is a simple, exactly solvable version of this picture.", MX, y + 0.05, CW, { bold: true });
  check(s, y);
}

// ---------------------------------------------------------------- 11 Simulation
{
  const s = base("Background", "Synchronization in a simulation", "This is a simulation of the Kuramoto model with the coupling above the critical value. The oscillators start with random phases, so R is about 0.1. As they pull on each other, groups of locked oscillators form and R rises to about 0.7. R keeps fluctuating because the fastest oscillators never lock. In the Kuramoto model the final value of R grows gradually as the coupling is increased. The next part of the talk looks at how this growth can instead be sudden.", "Background");
  const ds = sizeOf(FIG_SYNC), iw = 7.6, ih = iw * ds.height / ds.width;
  s.addImage({ path: FIG_SYNC, x: MX, y: TOP + 0.1, w: iw, h: ih, altText: "Order parameter against time rising from about 0.1 to about 0.7" });
  const rx = MX + iw + 0.5, rw = CW - iw - 0.5;
  let y = TOP + 0.1;
  for (const t of ["The oscillators start with random phases, so R is close to 0.1.", "Each oscillator is pulled towards the others, and groups of locked oscillators form.", "After some time most oscillators are locked and R settles near 0.7.", "The fastest oscillators never lock, which is why R keeps fluctuating.", "In the Kuramoto model, R grows gradually as the coupling is increased."]) {
    y = text(s, t, rx, y, rw) + 0.02;
  }
  footer(s, "Source: legacy/figures/order_param.png, a Kuramoto run from the earlier code in legacy/.");
  check(s, Math.max(y, TOP + 0.1 + ih));
}

// ---------------------------------------------------------------- 12 Adaptive coupling
{
  const s = base("Literature", "Adaptive coupling (Zhang et al., 2015)", "So far the coupling strength has been fixed. Zhang and co-authors introduced a different idea: what if the strength of an interaction depends on how much the neighbours already agree? When the neighbours are poorly aligned, the coupling is weak. As they become more synchronized, it gets stronger. That creates a feedback loop, and this is what makes a sudden transition possible. In the equation, r i is the local order parameter of node i's neighbours, p sets how strong the feedback is, with p equal to zero giving back ordinary Kuramoto, and lambda is the overall coupling. In a financial network this is herding.", "Literature");
  let y = eq(s, "a_eq", MX, TOP, 28).bottom + 0.3;
  const colW = (CW - 0.5) / 2;
  const yl = termRows(s, [
    { eq: "a_lam", text: "is the coupling strength." },
    { eq: "a_k", text: "is the number of neighbours of each node, so dividing by k gives an average." },
    { eq: "a_A", text: "describes the network. It is 1 if node i listens to node j and 0 otherwise." },
  ], MX, y, 1.0, colW);
  const yr = termRows(s, [
    { eq: "o_r", text: "is the local order parameter, which measures how much the neighbours of node i agree." },
    { eq: "a_p", text: "is the feedback exponent. Setting p = 0 recovers the ordinary Kuramoto model." },
    { eq: "a_rp", text: "is the adaptive gain. A node listens weakly when its neighbours disagree and strongly when they agree, which is a simple model of herding." },
  ], MX + colW + 0.5, y, 1.0, colW);
  footer(s, "Zhang, Boccaletti, Guan and Liu, Phys. Rev. Lett. 114, 038701 (2015)");
  check(s, Math.max(yl, yr));
}

// ---------------------------------------------------------------- 13 Herding panels
panelSlide("Literature", "Without herding: a smooth rise", "This is one panel from my simulations of the adaptive model, on a random network of 500 traders with an average of 12 links each and Gaussian natural frequencies. Here no trader uses adaptive coupling, so this is the ordinary Kuramoto model on a network. The horizontal axis is the coupling, the vertical axis is synchrony R. The blue curve is measured while the coupling is slowly increased and the orange curve while it is slowly decreased. Synchrony rises smoothly from near zero to near one, and the two curves lie on top of each other: the market moves gradually from disorder to synchrony and back along the same path.", "panel_herding_f0.png", [
  "No trader uses adaptive coupling here, so this is the ordinary Kuramoto model on a network.",
  "Blue: the coupling is slowly increased. Orange: it is slowly decreased.",
  "Synchrony rises smoothly, and the two curves lie on top of each other. The system moves from disorder to synchrony and back along the same path.",
], "Source: research/s2_herding_explosive.py. Random network, 500 nodes, average degree 12, Gaussian frequencies, no noise.", { axes: "Horizontal: coupling λ. Vertical: synchrony R." });

panelSlide("Literature", "With full herding: a sudden jump", "This is the same network, but now every trader uses adaptive coupling with p equal to one. When the coupling is slowly increased, the blue curve, synchrony stays near zero and then jumps to almost one at a coupling of about 3.9. When the coupling is slowly decreased, the orange curve, synchrony stays high and only drops at about 3.0. The shaded area between the curves is the range where the outcome depends on whether the market was calm or synchronized before. In market terms, a crash starts suddenly at a high level of herding, and once it has started, herding must fall well below that level before the market recovers. The numbers 3.88 and 3.04 come from the log of this run.", "panel_herding_f1.png", [
  "Every trader now uses adaptive coupling, with p = 1.",
  "Increasing the coupling (blue), synchrony stays near zero and then jumps to almost 1 at about 3.9.",
  "Decreasing it (orange), synchrony stays high and drops only at about 3.0.",
  "In market terms, a crash starts suddenly, and herding must fall well below that level before the market recovers.",
], "Source: research/s2_herding_explosive.py and research/data/s2.log (jump up at 3.88, drop at 3.04). Same network, all nodes herding, no noise.", { axes: "Horizontal: coupling λ. Vertical: synchrony R." });


function panelSlide(section, title, notes, img, sentences, source, opts = {}) {
  const s = base(section, title, notes, section);
  const d = sizeOf(A(img)), ih = opts.ih || 4.9, iw = ih * d.width / d.height;
  s.addImage({ path: A(img), x: MX, y: TOP - 0.1, w: iw, h: ih, altText: title });
  if (opts.axes) {
    text(s, opts.axes, MX, TOP - 0.1 + ih + 0.05, iw, { fontSize: 13, color: GREY });
  }
  const rx = MX + iw + 0.5, rw = CW - iw - 0.5;
  let y = TOP - 0.1;
  for (const t of sentences) y = text(s, t, rx, y, rw, { fontSize: opts.fs || 18 }) + 0.15;
  footer(s, source);
  check(s, y);
  return s;
}
function crashSlide(title, notes, img, sentences, source) {
  const s = base("Crash and recovery", title, notes, "Crash and recovery");
  const d = sizeOf(A(img)), iw = CW, ih = iw * d.height / d.width;
  text(s, "Synchrony R (blue) and order flow Q (orange)", MX, TOP - 0.15, iw / 2, { fontSize: 15, bold: true, color: GREY });
  text(s, "Market price (dashed line: fundamental value 100)", MX + iw / 2 + 0.1, TOP - 0.15, iw / 2, { fontSize: 15, bold: true, color: GREY });
  s.addImage({ path: A(img), x: MX, y: TOP + 0.2, w: iw, h: ih, altText: title });
  let y = TOP + 0.2 + ih + 0.3;
  const colW = (CW - 0.5) / 2, half = Math.ceil(sentences.length / 2);
  let yl = y, yr = y;
  sentences.slice(0, half).forEach((t) => { yl = text(s, t, MX, yl, colW, { fontSize: 18 }) + 0.12; });
  sentences.slice(half).forEach((t) => { yr = text(s, t, MX + colW + 0.5, yr, colW, { fontSize: 18 }) + 0.12; });
  footer(s, source);
  check(s, Math.max(yl, yr));
  return s;
}
// ================================================================ Market crashes
function figSlide(section, title, notes, fig, sentences, opts = {}) {
  const s = base(section, title, notes, section);
  const d = sizeOf(A(fig)), ar = d.width / d.height;
  if (ar < 2.2) {
    const iw = opts.iw || 7.2, ih = iw / ar;
    s.addImage({ path: A(fig), x: MX, y: TOP - 0.15, w: iw, h: ih, altText: title });
    const rx = MX + iw + 0.4, rw = CW - iw - 0.4;
    let y = TOP - 0.1;
    for (const t of sentences) y = text(s, t, rx, y, rw, { fontSize: opts.fs || 16 }) + 0.12;
    check(s, Math.max(y, TOP - 0.15 + ih));
  } else {
    const ih = Math.min(CW / ar, 3.1), iw = ih * ar;
    s.addImage({ path: A(fig), x: MX + (CW - iw) / 2, y: TOP - 0.15, w: iw, h: ih, altText: title });
    let y = TOP - 0.15 + ih + 0.2;
    const colW = (CW - 0.5) / 2, half = Math.ceil(sentences.length / 2);
    let yl = y, yr = y;
    sentences.slice(0, half).forEach((t) => { yl = text(s, t, MX, yl, colW, { fontSize: opts.fs || 16 }) + 0.08; });
    sentences.slice(half).forEach((t) => { yr = text(s, t, MX + colW + 0.5, yr, colW, { fontSize: opts.fs || 16 }) + 0.08; });
    check(s, Math.max(yl, yr));
  }
  return s;
}

// ---------------------------------------------------------------- M1 Market model
{
  const s = base("Market model", "The herding market model", "This is the full market model used in my simulations. Each trader's phase follows the adaptive Kuramoto equation, with three additions. The shock term epsilon s i is a common sell signal: traders with s i equal to one are pushed towards the selling phase, minus pi over two. The noise term sigma d W i is each trader's own news, different for every trader. The term sigma c d W c is common market news, the same for everyone. The net order flow Q is the average of the sines of the phases: minus one means everyone is selling and plus one means everyone is buying. The log price x is pushed by the order flow with strength beta and pulled back to its fundamental value by value investors at rate kappa, plus its own noise. The price is 100 times e to the x.", "Market crashes");
  let y = eq(s, "m_phase", MX, TOP - 0.1, 22).bottom + 0.15;
  y = eq(s, "m_price", MX, y, 22).bottom + 0.25;
  const colW = (CW - 0.5) / 2;
  const yl = termRows(s, [
    { eq: "m_eps", text: "is a common sell shock. Traders with sᵢ = 1 are pushed towards the selling phase." },
    { eq: "m_sig", text: "is each trader’s own news, which is different for every trader." },
    { eq: "m_sigc", text: "is common market news, which is the same for every trader." },
    { eq: "m_Q", text: "is the net order flow. Q = −1 means everyone sells and Q = +1 means everyone buys." },
  ], MX, y, 1.4, colW, 22, 17, 0.06);
  const yr = termRows(s, [
    { eq: "m_beta", text: "is the price impact: how strongly order flow moves the log price x." },
    { eq: "m_kappa", text: "is the pull of value investors back towards the fundamental value." },
    { eq: "m_eta", text: "is random noise in the price itself." },
  ], MX + colW + 0.5, y, 1.2, colW, 22, 17, 0.06);
  footer(s, "Source: research/model.py (the equations) and research/s3_crash_anatomy.py (parameter values).");
  check(s, Math.max(yl, yr));
}

// ---------------------------------------------------------------- N1 Order flow
{
  const s = base("Market model", "What the phase means for a trader", "In the market model the phase of a trader is not just an angle; it sets the trader's order. The order is the sine of the phase. At 90 degrees, pi over two, the sine is plus one and the trader buys at full strength. At minus 90 degrees, minus pi over two, the sine is minus one and the trader sells at full strength. At 0 or 180 degrees the order is zero: the trader is switching between buying and selling. A trader left alone runs round the circle at its own speed, so it alternates between buying and selling. Q is the average order over all traders, the net order flow of the market. When phases are spread out, buys and sells cancel and Q is close to zero. When most traders are locked near minus pi over two, Q is close to minus one and the whole market sells.", "Market model");
  let y = eq(s, "m_q", MX, TOP - 0.05, 24).bottom + 0.3;
  y = termRows(s, [
    { label: "θ = π/2 (90°)", text: "The order is q = +1. The trader is buying at full strength." },
    { label: "θ = −π/2 (−90°)", text: "The order is q = −1. The trader is selling at full strength." },
    { label: "θ = 0 or π", text: "The order is q = 0. The trader is switching between buying and selling." },
    { label: "A trader on its own", text: "It runs round the circle at its own speed ω, so it alternates between buying and selling." },
    { label: "The whole market", text: "Q is the average order. With phases spread out, buys and sells cancel and Q ≈ 0. With most traders locked near −π/2, Q ≈ −1 and the whole market sells." },
  ], MX, y, 3.2, CW, 22, BODY, 0.2);
  check(s, y);
}

// ---------------------------------------------------------------- N2 Sell shock
{
  const s = base("Market model", "The sell shock: sin(−π/2 − θ)", "The shock term has exactly the same form as the Kuramoto pull, sine of a target minus my own phase, but the target is not another trader. It is the fixed phase minus pi over two, which means full selling. So a trader hit by the shock behaves as if it were watching a phantom trader who always sells, which is how bad news acts. If the trader's phase is behind minus pi over two, the sine is positive and pushes it forward; if it is ahead, the sine is negative and pulls it back. At minus pi over two the term is zero, so the trader is held at full selling. Epsilon is the strength of the news. In the crash simulations a strong shock of 1.2 lasts 8 time units, and a weak bearish background of 0.02 is always on. s i is one for the traders who receive the news and zero for the rest; hitting 10 to 15 percent of traders, especially hubs, can be enough. The weak background is needed because, without any directional news, a locked herd's common phase slowly drifts from selling to buying, a boom-bust whipsaw.", "Market model");
  let y = eq(s, "m_shock", MX, TOP - 0.05, 28).bottom + 0.3;
  y = termRows(s, [
    { label: "Same form as the coupling", text: "It is the Kuramoto pull towards a fixed phase, −π/2, instead of towards another trader. A trader hit by the shock acts as if it watched a phantom trader who always sells." },
    { label: "Direction of the push", text: "Behind −π/2 the term is positive and pushes the phase forward, ahead of it the term is negative, and at −π/2 it is zero. The trader is driven to full selling and held there." },
    { label: "Strength ε", text: "In the crash simulations a strong shock ε = 1.2 lasts 8 time units, and a weak bearish background ε = 0.02 is always on." },
    { label: "Who is hit, sᵢ", text: "sᵢ = 1 for traders who receive the news and 0 for the rest. Hitting 10 to 15% of traders, especially hubs, can be enough for a lasting crash." },
    { label: "Why a weak background", text: "Without any directional news, a locked herd’s common phase slowly drifts from selling to buying, a boom-bust whipsaw. The weak term keeps it on the selling side." },
  ], MX, y, 3.2, CW, 22, 17, 0.14);
  footer(s, "Source: research/s3_crash_anatomy.py (ε = 1.2 for 8 time units, background ε = 0.02) and research/s4_crash_phase_diagram.py (10 to 15% of traders, hubs).");
  check(s, y);
}

// ---------------------------------------------------------------- N3 Noise
{
  const s = base("Market model", "Noise: private news and common news", "Real traders do not move like clockwork; they react to news. The model has two kinds of random news. sigma d W i is private news: at every time step each trader gets its own random kick, with mean zero and a variance proportional to the time step, so the kicks add up like a random walk. sigma c d W c is common news: one random kick shared by every trader, which moves all phases together. Private noise shakes locked traders and blurs the synchronized state. In my runs with sigma equal to 0.3, the lowest herding at which a crash could survive rose from 3.04 to 3.21. Noise can also start a crash on its own: near the crash threshold, random kicks can bring enough traders into line by chance. At herding 3.8 a quarter of simulated markets crashed with no shock at all, and at 4.3 all of them did. Common news is what makes different stocks move together; without it, the model's stocks would be uncorrelated even in a crash. The equations are integrated with a stochastic Heun scheme.", "Market model");
  let y = eq(s, "m_dW", MX, TOP - 0.05, 24).bottom + 0.3;
  y = termRows(s, [
    { eq: "m_sig", text: "is private news. Each trader gets its own random kick at every step, with mean zero and variance proportional to the step, so the kicks add up like a random walk." },
    { eq: "m_sigc", text: "is common news. One random kick is shared by every trader, so it moves all phases together. It is what makes different stocks move together." },
  ], MX, y, 1.7, CW, 26, BODY, 0.2);
  y = termRows(s, [
    { label: "Noise blurs a crash", text: "Private noise shakes locked traders. With σ = 0.3 the lowest herding at which a crash can survive rose from 3.04 to 3.21 in my runs." },
    { label: "Noise can start a crash", text: "Near the crash threshold, random kicks can line up enough sellers by chance. At herding 3.8 a quarter of simulated markets crashed with no shock, and at 4.3 all did." },
  ], MX, y + 0.1, 3.2, CW, 22, BODY, 0.2);
  footer(s, "Source: research/data/s2.log (3.04 without noise, 3.21 with σ = 0.3) and research/s4_crash_phase_diagram.py (crashes with no shock at 3.8 and 4.3).");
  check(s, y);
}

// ---------------------------------------------------------------- N4 Prices
{
  const s = base("Market model", "From orders to prices", "The phases set the orders, and the orders set the price. x is the log price relative to the fundamental value, so the price is 100 times e to the x, and x equal to zero means a price of 100. Beta Q is price impact: net selling, Q below zero, pushes the price down. Minus kappa x is the value investors: they buy when the price is below its fundamental value and sell when it is above, which pulls x back to zero. Eta d B is random noise in the price itself. In the simulations beta is 0.01 and kappa is 0.05. If Q stays constant, the price settles where the two forces balance, beta Q equals kappa x, so x star is beta Q over kappa. If everyone sells, Q equals minus one and the price settles at 100 times e to the minus 0.2, about 82. That is exactly where the simulated crashes end, between 82.9 and 87.", "Market model");
  let y = eq(s, "m_dx", MX, TOP - 0.05, 24).bottom + 0.25;
  y = termRows(s, [
    { label: "x", text: "The log price relative to the fundamental value. P = 100 eˣ, so x = 0 means a price of 100." },
    { label: "Price impact βQ", text: "Order imbalance moves the price. Net selling (Q < 0) pushes it down. In the simulations β = 0.01." },
    { label: "Value investors −κx", text: "They buy when the price is below its fundamental value and sell when it is above, pulling x back to 0. In the simulations κ = 0.05." },
    { label: "Price noise η dB", text: "Random noise in the price itself, separate from the traders’ news." },
  ], MX, y, 3.2, CW, 22, BODY, 0.15);
  y = text(s, "If Q stays constant, the price settles where the two forces balance. When everyone sells, it settles near 82, which is where the simulated crashes end.", MX, y + 0.05, CW, { bold: true });
  y = eq(s, "m_xstar", MX, y + 0.15, 24).bottom;
  footer(s, "Source: research/s3_crash_anatomy.py (β = 0.01, κ = 0.05, final prices 82.9 and 87.0). The 82 follows from the price equation.");
  check(s, y);
}

// ---------------------------------------------------------------- N5 Disorder to synchrony
function physicsMarket(s, physics, market, y0) {
  const colW = (CW - 0.8) / 2;
  let yb = y0;
  [["THE PHYSICS", physics], ["IN THE MARKET", market]].forEach(([h, ts], i) => {
    const x = MX + i * (colW + 0.8);
    s.addShape(pres.shapes.LINE, { x, y: y0, w: colW, h: 0, line: { color: RULE, width: 1 } });
    let y = text(s, h, x, y0 + 0.12, colW, { fontSize: 14, bold: true, color: GREY, charSpacing: 2 });
    for (const t of ts) y = text(s, t, x, y + 0.08, colW, { fontSize: 16 });
    yb = Math.max(yb, y);
  });
  return yb;
}
{
  const s = base("Crash and recovery", "From disorder to synchrony: how a crash starts", "This is the crash, seen first as physics and then as a market. In ordinary Kuramoto the group pulls each oscillator with strength K R, which grows in proportion to R. Even a small cluster pulls a little, so synchrony grows smoothly. With adaptive coupling the pull is lambda R to the p plus one. A small cluster has almost no pull, so the disordered state is stable. But once a large enough cluster forms, every new recruit raises R, which raises the pull faster than R itself, which recruits more. That runaway cascade is the sudden jump. The cluster can form because herding rises far enough, because a shock pushes a group together, or because noise does so by chance. In the market: in calm times traders buy and sell at different times and their orders cancel. While opinions are split, herding stays weak. News or rising herding brings a group of traders to sell together, their agreement makes everyone copy them more strongly, and selling spreads through the network. Q falls towards minus one and the price falls by up to about 18 percent. That is the crash.", "Crash and recovery");
  const y0 = eq(s, "m_pulls", MX, TOP - 0.1, 22).bottom + 0.2;
  const y = physicsMarket(s, [
    "In ordinary Kuramoto the group pulls each oscillator with strength KR. Even a small cluster pulls a little, so synchrony grows smoothly.",
    "With adaptive coupling the pull is the one shown on the right above. It is tiny when R is small, so a small cluster has almost no pull and disorder is stable.",
    "Once a large enough cluster forms, each new member raises R, which raises the pull faster than R, which recruits more. This runaway cascade is the sudden jump.",
    "The cluster can form when herding rises far enough, when a shock pushes a group together, or when noise does so by chance.",
  ], [
    "In a calm market traders buy and sell at different times, their orders cancel, and Q ≈ 0.",
    "Herding means copying more when others agree. While opinions are split, copying stays weak.",
    "News or rising herding brings a group of traders to sell together. Their agreement makes everyone copy them more, and selling spreads through the network.",
    "Q falls towards −1 and the price falls by up to about 18%. That is the crash.",
  ], y0);
  footer(s, "The 18% follows from the price equation: 1 − e^(−0.2) ≈ 0.18 when every trader sells.");
  check(s, y);
}

// ---------------------------------------------------------------- M3 Three crashes
crashSlide("Crash A: a flash crash that heals", "These are my simulations of the market model with 500 traders and noise 0.3. In scenario A herding is low, 2.61. A strong sell shock of strength 1.2 hits for 8 time units. Synchrony R, in blue, spikes to almost one and the order flow Q, in orange, drops to almost minus one: briefly, everyone sells. The price falls 7.6 percent. As soon as the shock ends, herding is too weak to hold the herd together, the traders drift apart, Q returns to zero and the price recovers to its fundamental value. That is a flash crash.", "panel_crash_A.png", [
  "Herding is low (λ = 2.61). A sell shock of strength ε = 1.2 hits for 8 time units.",
  "R jumps to almost 1 and Q to almost −1: for a moment, everyone sells.",
  "The price falls 7.6%.",
  "When the shock ends, herding is too weak to hold the herd together. Q returns to 0 and the price recovers.",
], "Source: research/s3_crash_anatomy.py. 500 traders, average degree 12, noise σ = 0.3, β = 0.01, κ = 0.05, one shared noise realisation.");

crashSlide("Crash B: the same shock, but the crash lasts", "Scenario B uses exactly the same market, traders, noise and shock. The only difference is that herding is higher, 3.47. Now, after the shock ends, the herd holds itself together: R stays near 0.85 and Q stays near minus 0.7. The price falls 16.2 percent and stays down near 87, which matches the price equation, 100 times e to the power beta Q over kappa. So the same trigger can produce a short dip or a lasting crash, depending only on how strongly traders herd.", "panel_crash_B.png", [
  "The same market, traders, noise and shock, but herding is higher (λ = 3.47).",
  "After the shock the herd holds itself together: R stays near 0.85 and Q near −0.7.",
  "The price falls 16.2% and stays near 87, as the price equation predicts for Q ≈ −0.7.",
  "The same trigger gives a short dip or a lasting crash, depending only on how strongly traders herd.",
], "Source: research/s3_crash_anatomy.py. 500 traders, average degree 12, noise σ = 0.3, β = 0.01, κ = 0.05, one shared noise realisation.");

crashSlide("Crash C: a crash with no shock at all", "In scenario C there is no shock, only the weak bearish background news. Instead, herding rises slowly from 1.7 to about 4.8. For a long time the market looks calm, but R starts to flicker as small groups briefly line up. Then, once herding is high enough, the cascade runs away: R jumps to almost one, Q to almost minus one, and the price falls 19.2 percent, ending near 83. This is an endogenous crash, caused by rising herding and noise alone.", "panel_crash_C.png", [
  "There is no shock. Herding rises slowly, from about 1.7 to 4.8.",
  "For a long time the market looks calm, but R flickers as small groups briefly line up.",
  "Once herding is high enough the cascade runs away: R jumps to almost 1 and Q to almost −1.",
  "The price falls 19.2% and ends near 83. The crash is caused by rising herding and noise alone.",
], "Source: research/s3_crash_anatomy.py. 500 traders, average degree 12, noise σ = 0.3, β = 0.01, κ = 0.05, one shared noise realisation.");

// ---------------------------------------------------------------- N6 Synchrony to disorder
{
  const s = base("Crash and recovery", "From synchrony back to disorder: how a crash ends", "Now the reverse transition, the recovery. A synchronized oscillator stays locked only while the group's pull beats its own speed, the size of omega i at most lambda R to the p plus one. Because that pull grows like R to the p plus one, a strongly synchronized group holds itself together: it survives even when herding is lowered below the level that created it. In my runs a crash formed at herding 3.9 but only ended below 3.0. When herding falls far enough, the fastest traders unlock first. R drops, the pull drops even faster, more traders unlock, and synchrony collapses suddenly, just as it formed. If the coupling is switched off altogether, the phases spread out at a rate set by the spread of natural frequencies, gamma, and the noise, sigma: roughly R of t equals R zero times e to the minus gamma plus sigma squared over two, times t. In the market: a crash ends when the selling herd breaks up and buys and sells cancel again. Reducing herding a little is not enough; it must fall well below the level that started the crash. Traders with very different strategies unlock first, so a market with more diverse strategies recovers more easily.", "Crash and recovery");
  const a = eq(s, "m_unlock", MX, TOP - 0.1, 22);
  eq(s, "m_decay", MX + a.w + 1.2, TOP - 0.1, 22);
  const y0 = TOP - 0.1 + Math.max(a.h, 0.5) + 0.25;
  const y = physicsMarket(s, [
    "An oscillator stays locked only while the group’s pull beats its own speed, as in the condition above.",
    "Because the adaptive pull grows faster than R, a strongly synchronized group holds itself together, even below the herding that created it. In my runs a crash formed at 3.9 but only ended below 3.0.",
    "When herding falls far enough, the fastest oscillators unlock first. R drops, the pull drops faster, more unlock, and synchrony collapses suddenly. With the coupling off, R decays as shown above right.",
  ], [
    "A crash ends when the selling herd breaks up and buys and sells cancel again.",
    "Reducing herding a little is not enough. It must fall well below the level that started the crash.",
    "Traders with very different strategies unlock first, so a market with more diverse strategies recovers more easily.",
  ], y0);
  footer(s, "Source: research/data/s2.log (random network, all nodes herding, no noise: jump up at 3.88, drop at 3.04).");
  check(s, y);
}

// ---------------------------------------------------------------- 18 What this means
{
  const s = base("Financial networks", "What this means for financial networks", "To close, here is what the physics means for a network of traders. Each trader responds to the overall market consensus, more strongly when the consensus is stronger. A consensus forms only when copying is strong compared with the diversity of strategies. With herding, the market does not drift gradually to one side; it switches suddenly from balanced to one-sided trading, which is a crash. News reaching a modest fraction of traders, or random news alone when herding is high, can start it. A crash ends only when herding falls well below the level that started it, and markets with more diverse strategies recover more easily. The trader picture is an abstraction, and testing it carefully on market data is the next step.", null);
  let y = text(s, "The trader picture is an abstraction, but within the model each result of the theory has a direct reading for a network of traders.", MX, TOP - 0.1, CW, { color: "D9D9D9" });
  y = termRows(s, [
    { label: "Mean-field pull", text: "Each trader responds to the overall market consensus, and the response is stronger when the consensus is stronger." },
        { label: "Threshold", text: "A consensus forms only when copying is strong compared with the diversity of strategies γ." },
    { label: "Herding", text: "With adaptive coupling, the market does not drift gradually towards one side. It switches suddenly from balanced trading to one-sided trading." },
    { label: "Shocks and noise", text: "News reaching 10 to 15% of traders, or random news alone when herding is high, can be enough to start a crash." },
    { label: "Recovery", text: "A crash ends only when herding falls well below the level that started it. Markets with more diverse strategies recover more easily." },
  ], MX, y + 0.2, 2.8, CW, 24, 17, 0.1);
  footer(s, "Source: research/s4_crash_phase_diagram.py (10 to 15% of traders) and research/data/s2.log (herding levels).");
  check(s, y);
}

// ---------------------------------------------------------------- 25 Thank you
{
  const s = base("Closing", null, "Thank the audience and invite questions. If asked whether this has been applied to real markets: first tests on daily and minute-level stock data are in the repository, but they are not conclusive, mainly because large intraday crashes are rare, so the sample is too small to confirm or rule out the model. In this talk the financial network is the motivation and the way to read the model, and a proper test on market data is future work.");
  s.addText("Thank you", { x: MX, y: 2.6, w: CW, h: 1.1, fontFace: FONT, fontSize: 54, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Questions are welcome.", { x: MX, y: 3.7, w: CW, h: 0.6, fontFace: FONT, fontSize: 26, color: WHITE, margin: 0, isTextBox: true });
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT));
