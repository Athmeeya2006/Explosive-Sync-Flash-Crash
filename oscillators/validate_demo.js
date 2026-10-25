// Runs every model of demo_models.js at the settings in validate_demo_cases.json (shared with
// validate_demo.py) and prints, as JSON, the quantity that the matching zoo.py kernel records as its
// amplitude output, averaged over the second half of each run:
//   phase order parameter R  (Kuramoto, inertia, van der Pol, FitzHugh-Nagumo, Roessler, theta neurons)
//   |Z| = |mean z|           (Stuart-Landau)
//   mean phase velocity      (Winfree)
//     node validate_demo.js
"use strict";
const { MODELS, makeRng, orderParam } = require("./demo_models.js");
const CASES = require("./validate_demo_cases.json");

function observable(model, st, w, m) {
  if (model === "sl") {
    let zr = 0, zi = 0;
    for (let j = 0; j < st.zr.length; j++) { zr += st.zr[j]; zi += st.zi[j]; }
    return Math.hypot(zr, zi) / st.zr.length;
  }
  return orderParam(m.phases(st, w)).R;
}

const out = [];
for (const c of CASES) {
  const m = MODELS[c.model];
  for (let seed = 1; seed <= c.seeds; seed++) {
    const rng = makeRng(1000 * seed + 17);
    const w = m.freq(c.N, rng);
    const st = m.init(c.N, rng, c.sync, w);
    const steps = Math.round(c.T / m.dt);
    let acc = 0, cnt = 0, thMid = null;
    for (let s = 0; s < steps; s++) {
      m.step(st, w, c.K, c.p, m.sigma, m.dt, rng);
      if (c.model === "winfree" && s === Math.floor(steps / 2)) thMid = st.th.slice();
      if (s > steps / 2 && s % 10 === 0) { acc += observable(c.model, st, w, m); cnt++; }
    }
    let value = acc / cnt;
    if (c.model === "winfree") {
      let v = 0;
      for (let j = 0; j < st.th.length; j++) v += (st.th[j] - thMid[j]) / (c.T / 2);
      value = v / st.th.length;
    }
    out.push({ model: c.model, K: c.K, p: c.p, sync: c.sync, seed, value });
  }
}
console.log(JSON.stringify(out));
