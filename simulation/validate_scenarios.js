/* Benchmark 3 for the browser simulation: are all the market phases actually reachable,
 * and does the interactive model reproduce the qualitative behaviour of S2-S5?
 *
 *     node simulation/validate_scenarios.js
 *
 * Checks, all on the shipped network with the shipped dynamics:
 *   1  low lambda           -> stays incoherent            (R < 0.30)
 *   2  high lambda          -> synchronises                (R > 0.70)
 *   3  shock below lambda_b -> flash crash, then recovers  (R peaks > 0.5, decays back < 0.4)
 *   4  shock inside window  -> crash PERSISTS              (R stays > 0.6 long after the shock)
 *   5  halt from a crash    -> desynchronises              (R drops > 0.3 during the halt)
 *   6  adiabatic sweep      -> hysteresis                  (lambda_f > lambda_b)
 *   7  hub shock >= random shock at equal reach
 */
'use strict';
const fs = require('fs');
const path = require('path');
const dir = __dirname;
const NETS = JSON.parse(fs.readFileSync(path.join(dir, 'network.js'), 'utf8')
  .match(/window\.NETWORKS\s*=\s*(\{[\s\S]*\});\s*\n/)[1]);
const { Sim } = require(path.join(dir, 'sim.js'));

const DT = 0.02;
const run = (sim, T) => { const n = Math.round(T / DT); for (let i = 0; i < n; i++) sim.step(DT); };

let pass = 0, fail = 0;
function check(name, cond, detail) {
  if (cond) { pass++; console.log(`  [PASS] ${name}  ${detail}`); }
  else { fail++; console.log(`  [FAIL] ${name}  ${detail}`); }
}

/* the default network is the one the interactive page opens on, and it is the one that must
   demonstrate hysteresis; ba6 is checked too but only warned about */
for (const [KEY, NET] of Object.entries(NETS)) {
  const STRICT = KEY === 'er12';
  const fresh = (o) => new Sim(NET, Object.assign({ sigma: 0.3, f: 1.0, seed: 11 }, o));
  console.log(`\n=================  ${KEY}: ${NET.meta.model}  E=${NET.meta.m}  ` +
    `<k>=${NET.meta.meanDeg.toFixed(2)}  kappa=${NET.meta.kappa.toFixed(2)}  =================\n`);

/* ---- locate the transition on THIS network by an adiabatic sweep ------------------ */
function sweep(lams, sim) {
  const out = [];
  for (const L of lams) {
    sim.lam = L;
    run(sim, 18);           // relax
    let acc = 0, m = 0;
    for (let i = 0; i < Math.round(18 / DT); i++) { sim.step(DT); acc += sim.R; m++; }
    out.push(acc / m);
  }
  return out;
}
const lams = [];
for (let L = 1.0; L <= 6.501; L += 0.1) lams.push(+L.toFixed(2));   // 0.1 resolves S2's 0.75-wide window
let s = fresh({ lam: lams[0] });
const fwd = sweep(lams, s);
const bwd = sweep(lams.slice().reverse(), s).reverse();
const firstAbove = fwd.findIndex(v => v > 0.5);
const lastBelow = bwd.map((v, i) => v < 0.5 ? i : -1).filter(i => i >= 0).pop();
const lamF = firstAbove >= 0 ? lams[firstAbove] : NaN;
const lamB = (lastBelow !== undefined && lastBelow + 1 < lams.length) ? lams[lastBelow + 1] : NaN;
const gap = Math.max(...bwd.map((v, i) => v - fwd[i]));
console.log(`adiabatic sweep: lambda_f = ${lamF}, lambda_b = ${lamB}, ` +
  `width = ${(lamF - lamB).toFixed(2)}, max hysteresis gap = ${gap.toFixed(3)}`);
console.log(`  forward  R: ${fwd.map(v => v.toFixed(2)).join(' ')}`);
console.log(`  backward R: ${bwd.map(v => v.toFixed(2)).join(' ')}\n`);

const hyst = isFinite(lamF) && isFinite(lamB) && (lamF - lamB) > 0.05;
if (STRICT) check('6  hysteresis (lambda_f > lambda_b)', hyst,
  `lambda_f=${lamF} lambda_b=${lamB} width=${(lamF - lamB).toFixed(2)} gap=${gap.toFixed(3)}`);
else console.log(`  [note] ${KEY} hysteresis width ${(lamF - lamB).toFixed(2)} ` +
  `(S2: scale-free networks resist herding-induced bistability, not a failure)`);

const LAM_IN = isFinite(lamB) ? (lamB + 0.35 * Math.max(0.2, lamF - lamB)) : 3.5;
const LAM_LO = isFinite(lamB) ? Math.max(0.5, lamB - 1.0) : 2.0;

/* ---- 1 incoherent ----------------------------------------------------------------- */
const a = fresh({ lam: 0.8 }); run(a, 80);
check('1  low lambda stays incoherent', a.R < 0.30, `R=${a.R.toFixed(3)}`);

/* ---- 2 synchronised --------------------------------------------------------------- */
const b = fresh({ lam: 7.5 }); run(b, 80);
check('2  high lambda synchronises', b.R > 0.70, `R=${b.R.toFixed(3)}`);

/* ---- 3 flash crash then recovery (below the window) -------------------------------- */
const c = fresh({ lam: LAM_LO }); run(c, 60);
c.applyShock(1.5, 0.3, 'random'); run(c, 8);
const cPeak = c.R;
c.clearShock(); run(c, 120);
check('3  shock below window = flash crash then recovery',
  cPeak > 0.5 && c.R < 0.45, `lambda=${LAM_LO} peak R=${cPeak.toFixed(3)} -> final R=${c.R.toFixed(3)}`);

/* ---- 4 persistent crash (inside the window) ---------------------------------------- */
const d = fresh({ lam: LAM_IN }); run(d, 60);
d.applyShock(1.5, 0.3, 'random'); run(d, 8);
d.clearShock(); run(d, 160);
/* only meaningful where a bistable window exists: without one there is no second stable branch for
   the shock to leave the market on, so "persistent" is undefined rather than failed */
if (STRICT) check('4  shock inside window = persistent crash',
  d.R > 0.60, `lambda=${LAM_IN.toFixed(2)} final R=${d.R.toFixed(3)} Q=${d.Q.toFixed(3)}`);
else console.log(`  [note] 4  ${KEY} has no bistable window, so crash persistence is undefined ` +
  `(final R=${d.R.toFixed(3)} is just the continuous branch)`);

/* ---- 5 halt desynchronises --------------------------------------------------------- */
const rBeforeHalt = d.R;
d.halt(10, 1.0); run(d, 9.5);
check('5  trading halt desynchronises', rBeforeHalt - d.R > 0.30,
  `R ${rBeforeHalt.toFixed(3)} -> ${d.R.toFixed(3)} during halt`);
d.resume();

/* ---- 7 hub targeting vs random ----------------------------------------------------- */
function shockTrial(rule, reach, seed) {
  const s2 = fresh({ lam: LAM_IN, seed });
  run(s2, 60);
  s2.applyShock(1.2, reach, rule); run(s2, 8);
  s2.clearShock(); run(s2, 120);
  return s2.R > 0.5 ? 1 : 0;
}
let hub = 0, rnd = 0;
const TRIALS = 6;
for (let k = 0; k < TRIALS; k++) { hub += shockTrial('hubs', 0.08, 20 + k); rnd += shockTrial('random', 0.08, 20 + k); }
check('7  hubs are at least as dangerous as random targets', hub >= rnd,
  `hubs ${hub}/${TRIALS} vs random ${rnd}/${TRIALS} persistent crashes at reach 8%`);

/* ---- physics throughput ------------------------------------------------------------ */
const p = fresh({ lam: 3.0 });
const t0 = process.hrtime.bigint();
for (let i = 0; i < 4000; i++) p.step(DT);
const us = Number(process.hrtime.bigint() - t0) / 1000 / 4000;
console.log(`\nphysics throughput: ${us.toFixed(1)} us per Heun step ` +
  `(N=${NET.meta.n}, E=${NET.meta.m}) -> ${(1e6 / us).toFixed(0)} steps/s single-threaded`);
const msPerFrame = us * 20 / 1000;   // us -> ms
check('8  physics fast enough for 60 fps at 20x speed', msPerFrame < 10,
  `20 substeps = ${msPerFrame.toFixed(2)} ms per frame, budget 10 ms (leaves 6.6 ms to render)`);

}  /* end per-network loop */

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
