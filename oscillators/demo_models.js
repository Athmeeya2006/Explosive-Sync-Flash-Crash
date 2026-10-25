// Oscillator families for demo.html: the same equations, frequency laws and integration scheme as zoo.py
// (mean-field coupling, Euler-Maruyama, exact rotation for Stuart-Landau). Loaded by demo.html and by
// validate_demo.js (Node), which compares time-averaged order parameters with the Python kernels.
(function (root) {
  "use strict";

  // seeded random numbers: mulberry32 for uniforms, Box-Muller for Gaussians
  function makeRng(seed) {
    let a = seed >>> 0;
    let spare = null;
    const uni = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const gauss = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0, v = 0;
      while (u === 0) u = uni();
      v = uni();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    return { uni, gauss };
  }

  // deterministic quantiles, randomly placed (as zoo.lorentz_q, gauss_q, uniform_q)
  function erfinv(x) {
    const a = 0.147, ln = Math.log(1 - x * x), t = 2 / (Math.PI * a) + ln / 2;
    let y = Math.sign(x) * Math.sqrt(Math.sqrt(t * t - ln / a) - t);
    for (let i = 0; i < 2; i++) {               // two Newton steps on erf(y) = x
      const err = erf(y) - x;
      y -= err / (2 / Math.sqrt(Math.PI) * Math.exp(-y * y));
    }
    return y;
  }
  function erf(x) {                             // Abramowitz-Stegun 7.1.26, then refined by erfinv's Newton steps
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  function shuffle(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rng.uni() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
    return arr;
  }
  const quant = {
    lorentz: (n, g, rng) => shuffle(Array.from({ length: n }, (_, i) => Math.max(-60, Math.min(60, g * Math.tan(Math.PI * ((i + 0.5) / n - 0.5))))), rng),
    gauss: (n, m, sd, rng) => shuffle(Array.from({ length: n }, (_, i) => m + sd * Math.SQRT2 * erfinv(2 * (i + 0.5) / n - 1)), rng),
    uniform: (n, lo, hi, rng) => shuffle(Array.from({ length: n }, (_, i) => lo + (hi - lo) * (i + 0.5) / n), rng),
  };

  function orderParam(ph) {
    let c = 0, s = 0;
    for (let i = 0; i < ph.length; i++) { c += Math.cos(ph[i]); s += Math.sin(ph[i]); }
    return { R: Math.hypot(c, s) / ph.length, Psi: Math.atan2(s, c) };
  }

  // Each model: label, coupling range and default, time step, frequencies, initial state, one step,
  // and the phases (and radii, for the circle plot) of the units.
  const MODELS = {
    kuramoto: {
      label: "Kuramoto", Kmax: 8, K: 1.5, dt: 0.02, sigma: 0.1,
      freq: (n, rng) => quant.lorentz(n, 0.5, rng),
      init(n, rng, sync) { return { th: Array.from({ length: n }, () => sync ? 0 : 2 * Math.PI * rng.uni()) }; },
      step(st, w, K, p, sigma, dt, rng) {
        const { R, Psi } = orderParam(st.th);
        const Ke = p > 0 ? K * Math.pow(R, p) : K;
        const zr = R * Math.cos(Psi), zi = R * Math.sin(Psi), sq = Math.sqrt(dt);
        for (let j = 0; j < st.th.length; j++) {
          const t = st.th[j];
          st.th[j] = t + (w[j] + Ke * (zi * Math.cos(t) - zr * Math.sin(t))) * dt + sigma * sq * rng.gauss();
        }
      },
      phases: st => st.th, radii: st => null,
      setPhase(st, j, a) { st.th[j] = a; },
    },
    inertia: {
      label: "Kuramoto with inertia (m = 2)", Kmax: 12, K: 3, dt: 0.02, sigma: 0.1,
      freq: (n, rng) => quant.gauss(n, 0, 1, rng),
      init(n, rng, sync) { return { th: Array.from({ length: n }, () => sync ? 0.1 * rng.gauss() : 2 * Math.PI * rng.uni()), v: new Array(n).fill(0) }; },
      step(st, w, K, p, sigma, dt, rng) {
        const { R, Psi } = orderParam(st.th);
        const Ke = p > 0 ? K * Math.pow(R, p) : K;
        const zr = R * Math.cos(Psi), zi = R * Math.sin(Psi), sq = Math.sqrt(dt), m = 2;
        for (let j = 0; j < st.th.length; j++) {
          const f = w[j] + Ke * (zi * Math.cos(st.th[j]) - zr * Math.sin(st.th[j]));
          st.v[j] += (f - st.v[j]) / m * dt + sigma / m * sq * rng.gauss();
          st.th[j] += st.v[j] * dt;
        }
      },
      phases: st => st.th, radii: st => null,
      setPhase(st, j, a) { st.th[j] = a; },
    },
    sl: {
      label: "Stuart-Landau", Kmax: 1.6, K: 0.3, dt: 0.02, sigma: 0.1,
      freq: (n, rng) => quant.lorentz(n, 0.1, rng),
      init(n, rng, sync) {
        const zr = [], zi = [];
        for (let j = 0; j < n; j++) { const a = sync ? 0 : 2 * Math.PI * rng.uni(); zr.push(Math.cos(a)); zi.push(Math.sin(a)); }
        return { zr, zi };
      },
      step(st, w, K, p, sigma, dt, rng) {
        const n = st.zr.length; let Zr = 0, Zi = 0, ma = 0;
        for (let j = 0; j < n; j++) { Zr += st.zr[j]; Zi += st.zi[j]; ma += Math.hypot(st.zr[j], st.zi[j]); }
        Zr /= n; Zi /= n; ma /= n;
        const Ke = p > 0 ? K * Math.pow(Math.hypot(Zr, Zi) / (ma + 1e-12), p) : K, sq = Math.sqrt(dt);
        for (let j = 0; j < n; j++) {
          const a2 = st.zr[j] ** 2 + st.zi[j] ** 2, g = 1 + (1 - a2 - Ke) * dt, c = Math.cos(w[j] * dt), s = Math.sin(w[j] * dt);
          const xr = g * (c * st.zr[j] - s * st.zi[j]) + Ke * Zr * dt + sigma * sq * rng.gauss();
          const xi = g * (s * st.zr[j] + c * st.zi[j]) + Ke * Zi * dt + sigma * sq * rng.gauss();
          st.zr[j] = xr; st.zi[j] = xi;
        }
      },
      phases: st => st.zr.map((x, j) => Math.atan2(st.zi[j], x)), radii: st => st.zr.map((x, j) => Math.hypot(x, st.zi[j])),
      setPhase(st, j, a) { const r = Math.hypot(st.zr[j], st.zi[j]) || 1; st.zr[j] = r * Math.cos(a); st.zi[j] = r * Math.sin(a); },
    },
    vdp: {
      label: "van der Pol (eps = 1)", Kmax: 1.5, K: 0.25, dt: 0.01, sigma: 0.1,
      freq: (n, rng) => quant.gauss(n, 1, 0.05, rng),
      init(n, rng, sync, w) {
        const x = [], v = [];
        for (let j = 0; j < n; j++) { const a = sync ? 0 : 2 * Math.PI * rng.uni(); x.push(2 * Math.cos(a)); v.push(-2 * w[j] * Math.sin(a)); }
        return { x, v };
      },
      step(st, w, K, p, sigma, dt, rng) {
        const n = st.x.length; let Vm = 0;
        for (let j = 0; j < n; j++) Vm += st.v[j];
        Vm /= n;
        let Ke = K;
        if (p > 0) { Ke = K * Math.pow(orderParam(this.phases(st, w)).R, p); }
        const sq = Math.sqrt(dt);
        for (let j = 0; j < n; j++) {
          st.v[j] += ((1 - st.x[j] ** 2) * st.v[j] - w[j] * w[j] * st.x[j] + Ke * (Vm - st.v[j])) * dt + sigma * sq * rng.gauss();
          st.x[j] += st.v[j] * dt;
        }
      },
      phases(st, w) { return st.x.map((x, j) => Math.atan2(-st.v[j] / w[j], x)); },
      radii(st, w) { return st.x.map((x, j) => Math.hypot(x, st.v[j] / w[j]) / 2); },
      setPhase(st, j, a, w) { st.x[j] = 2 * Math.cos(a); st.v[j] = -2 * w[j] * Math.sin(a); },
    },
    fhn: {
      label: "FitzHugh-Nagumo", Kmax: 0.4, K: 0.06, dt: 0.01, sigma: 0.02,
      freq: (n, rng) => quant.uniform(n, 0.34, 0.9, rng),          // input currents I_j
      init(n, rng, sync, I) {
        const st = { v: [], w: [], vs: [], ws: [], tl: [], per: new Array(n).fill(40), t: 0 };
        for (let j = 0; j < n; j++) {
          let z = 0;
          for (let k = 0; k < 60; k++) z -= (z - z ** 3 / 3 - (z + 0.7) / 0.8 + I[j]) / (1 - z * z - 1 / 0.8);
          st.vs.push(z); st.ws.push((z + 0.7) / 0.8);
          const a = sync ? 0 : 2 * Math.PI * rng.uni();
          st.v.push(z + 2 * Math.cos(a)); st.w.push((z + 0.7) / 0.8 + 0.5 * Math.sin(a)); st.tl.push(-40 * a / (2 * Math.PI));
        }
        return st;
      },
      step(st, I, K, p, sigma, dt, rng) {
        const n = st.v.length; let Vm = 0;
        for (let j = 0; j < n; j++) Vm += st.v[j];
        Vm /= n;
        const Ke = p > 0 ? K * Math.pow(orderParam(this.phases(st)).R, p) : K, sq = Math.sqrt(dt);
        for (let j = 0; j < n; j++) {
          const nv = st.v[j] + (st.v[j] - st.v[j] ** 3 / 3 - st.w[j] + I[j] + Ke * (Vm - st.v[j])) * dt + sigma * sq * rng.gauss();
          st.w[j] += 0.08 * (st.v[j] + 0.7 - 0.8 * st.w[j]) * dt;
          if (st.v[j] < 0 && nv >= 0 && st.t - st.tl[j] > 5) { st.per[j] = st.t - st.tl[j]; st.tl[j] = st.t; }
          st.v[j] = nv;
        }
        st.t += dt;
      },
      phases(st) { return st.v.map((_, j) => 2 * Math.PI * Math.min((st.t - st.tl[j]) / st.per[j], 1)); },
      radii(st) { return st.v.map(v => 0.55 + 0.45 * Math.tanh(v)); },
      setPhase(st, j, a) {
        // put the unit at the start of a spike and set its clock so that its event phase reads a
        st.v[j] = st.vs[j] + 2 * Math.cos(0); st.w[j] = st.ws[j];
        st.tl[j] = st.t - st.per[j] * (((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI);
      },
    },
    winfree: {
      label: "Winfree", Kmax: 1.2, K: 0.4, dt: 0.02, sigma: 0.1,
      freq: (n, rng) => quant.uniform(n, 0.9, 1.1, rng),
      init(n, rng, sync) { return { th: Array.from({ length: n }, () => sync ? 0 : 2 * Math.PI * rng.uni()) }; },
      step(st, w, K, p, sigma, dt, rng) {
        const n = st.th.length; let S = 0;
        for (let j = 0; j < n; j++) S += 1 + Math.cos(st.th[j]);
        S /= n;
        const ke = p > 0 ? K * Math.pow(orderParam(st.th).R, p) : K, sq = Math.sqrt(dt);
        for (let j = 0; j < n; j++) st.th[j] += (w[j] - ke * Math.sin(st.th[j]) * S) * dt + sigma * sq * rng.gauss();
      },
      phases: st => st.th, radii: st => null,
      setPhase(st, j, a) { st.th[j] = a; },
    },
    rossler: {
      label: "Roessler (chaotic)", Kmax: 0.8, K: 0.1, dt: 0.01, sigma: 0.1,
      freq: (n, rng) => quant.gauss(n, 1, 0.02, rng),
      init(n, rng, sync) {
        const st = { x: [], y: [], z: [] };
        for (let j = 0; j < n; j++) {
          st.x.push(sync ? 5 : 10 * (rng.uni() - 0.5)); st.y.push(sync ? 0 : 10 * (rng.uni() - 0.5)); st.z.push(0.1);
        }
        return st;
      },
      step(st, w, K, p, sigma, dt, rng) {
        const n = st.x.length; let Xm = 0;
        for (let j = 0; j < n; j++) Xm += st.x[j];
        Xm /= n;
        const Ke = p > 0 ? K * Math.pow(orderParam(this.phases(st)).R, p) : K, sq = Math.sqrt(dt);
        for (let j = 0; j < n; j++) {
          const dx = -w[j] * st.y[j] - st.z[j] + Ke * (Xm - st.x[j]), dy = w[j] * st.x[j] + 0.15 * st.y[j], dz = 0.2 + st.z[j] * (st.x[j] - 10);
          st.x[j] += dx * dt + sigma * sq * rng.gauss(); st.y[j] += dy * dt; st.z[j] = Math.max(0, st.z[j] + dz * dt);
        }
      },
      phases(st) { return st.x.map((x, j) => Math.atan2(st.y[j], x)); },
      radii(st) { return st.x.map((x, j) => Math.min(1, Math.hypot(x, st.y[j]) / 12)); },
      setPhase(st, j, a) { const r = Math.hypot(st.x[j], st.y[j]) || 6; st.x[j] = r * Math.cos(a); st.y[j] = r * Math.sin(a); },
    },
    theta: {
      label: "theta neurons", Kmax: 30, K: 15, dt: 0.002, sigma: 0.1, couplingName: "J",
      freq: (n, rng) => quant.lorentz(n, 0.5, rng).map(e => e - 5),  // drives eta_j
      init(n, rng, sync) { return { th: Array.from({ length: n }, () => sync ? 0 : 2 * Math.PI * rng.uni()) }; },
      step(st, eta, J, p, sigma, dt, rng) {
        const n = st.th.length; let S = 0;
        for (let j = 0; j < n; j++) { const q = 1 - Math.cos(st.th[j]); S += q * q; }
        S *= (2 / 3) / n;
        const Je = p > 0 ? J * Math.pow(orderParam(st.th).R, p) : J, sq = Math.sqrt(dt);
        for (let j = 0; j < n; j++) {
          const c = Math.cos(st.th[j]);
          st.th[j] += ((1 - c) + (1 + c) * (eta[j] + Je * S)) * dt + sigma * sq * rng.gauss();
        }
      },
      phases: st => st.th, radii: st => null,
      setPhase(st, j, a) { st.th[j] = a; },
    },
  };

  const api = { MODELS, makeRng, quant, orderParam };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Zoo = api;
})(typeof window !== "undefined" ? window : globalThis);
