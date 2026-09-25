/* Herding-oscillator market model, the same dynamics as research/model.py.
 *
 *   dtheta_i = [ omega_i + (lam/<k>) * alpha_i * sum_j A_ij sin(theta_j - theta_i)
 *                + eps(t) * s_i * sin(theta_tgt - theta_i) ] dt
 *              + sigma dW_i + sigma_c dW_c
 *
 *   alpha_i = r_i = |sum_j A_ij e^{i theta_j}| / k_i   for herding nodes (fraction f), else 1
 *   halted nodes (active = 0) neither see nor are seen, and their alpha is 0, so they free-run at omega_i.
 *
 * Integrator: stochastic Heun (same noise increment in predictor and corrector), matching model.py.
 * Runs in the browser (window.Sim) and in node (module.exports) so the physics can be unit-tested.
 */
(function (root) {
  'use strict';

  const TAU = Math.PI * 2;
  const SELL = -Math.PI / 2;

  /* deterministic PRNG so every run is reproducible from a seed */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* standard normal via Box-Muller, one cached spare */
  function gaussian(rand) {
    let spare = null;
    return function () {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0, v = 0, s = 0;
      do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s === 0 || s >= 1);
      const f = Math.sqrt(-2 * Math.log(s) / s);
      spare = v * f;
      return u * f;
    };
  }

  /* inverse standard normal CDF (Acklam), for deterministic Gaussian quantile frequencies */
  function normInv(p) {
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00];
    const pl = 0.02425;
    let q, r;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - pl) {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  class Sim {
    constructor(net, opts) {
      opts = opts || {};
      const n = this.n = net.meta.n;
      /* typed arrays: the adjacency arrives from JSON as plain Arrays, which are several times
         slower to index in the hot drift loop */
      this.off = Int32Array.from(net.nbrOffset);
      this.nbr = Int32Array.from(net.nbrList);
      this.deg = Float64Array.from(net.deg);
      this.meanDeg = net.meta.meanDeg;
      this.scale = 1 / this.meanDeg;              // norm = "mean"

      this.lam = opts.lam !== undefined ? opts.lam : 2.0;
      this.sigma = opts.sigma !== undefined ? opts.sigma : 0.3;
      this.sigmaC = opts.sigmaC !== undefined ? opts.sigmaC : 0.0;
      this.f = opts.f !== undefined ? opts.f : 1.0;
      this.beta = 0.01; this.kappa = 0.05; this.eta = 0.004;
      this.shockEps = 0; this.newsEps = 0;

      this.theta = new Float64Array(n);
      this.omega = new Float64Array(n);
      this.adaptive = new Uint8Array(n);
      this.active = new Uint8Array(n);
      this.shockMask = new Float64Array(n);
      this.cosT = new Float64Array(n);
      this.sinT = new Float64Array(n);
      this.f1 = new Float64Array(n);
      this.f2 = new Float64Array(n);
      this.pred = new Float64Array(n);
      this.noise = new Float64Array(n);
      this.rLocal = new Float64Array(n);
      this.q = new Float64Array(n);

      this.reset(opts.seed !== undefined ? opts.seed : 7);
    }

    reset(seed) {
      const n = this.n;
      this.seed = seed;
      this.rand = mulberry32(seed);
      this.randn = gaussian(this.rand);
      this.t = 0; this.x = 0; this.price = 100;
      this.halted = false; this.haltUntil = -1; this.anyHalted = false;

      /* deterministic Gaussian quantiles, randomly permuted, mean exactly 0 (as gaussian_omegas) */
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) w[i] = normInv((i + 0.5) / n);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(this.rand() * (i + 1));
        const tmp = w[i]; w[i] = w[j]; w[j] = tmp;
      }
      let mean = 0; for (let i = 0; i < n; i++) mean += w[i];
      mean /= n;
      for (let i = 0; i < n; i++) {
        this.omega[i] = w[i] - mean;
        this.adaptive[i] = this.rand() < this.f ? 1 : 0;
        this.active[i] = 1;
        this.shockMask[i] = 0;
        this.theta[i] = this.rand() * TAU;
      }
      this.computeObservables();
    }

    setHerdingFraction(f) {
      this.f = f;
      for (let i = 0; i < this.n; i++) this.adaptive[i] = this.rand() < f ? 1 : 0;
    }

    /* drift into `out`, evaluated at phases `th`.
       Hot loop, so every field is hoisted into a local and the halt test is lifted out of the inner
       loop entirely: while nothing is halted (the overwhelmingly common case) the neighbour sum is
       branch-free, which is worth ~2x on the whole integrator. */
    drift(th, out) {
      const n = this.n, off = this.off, nbr = this.nbr, act = this.active;
      const cosT = this.cosT, sinT = this.sinT, deg = this.deg;
      const omega = this.omega, adaptive = this.adaptive, mask = this.shockMask;
      const anyHalted = this.anyHalted;
      for (let i = 0; i < n; i++) { cosT[i] = Math.cos(th[i]); sinT[i] = Math.sin(th[i]); }
      const lamScale = this.lam * this.scale;
      const shockEps = this.shockEps, newsEps = this.newsEps;
      const eps = shockEps + newsEps;
      for (let i = 0; i < n; i++) {
        let c = 0, s = 0;
        const e = off[i + 1];
        if (anyHalted) {
          for (let p = off[i]; p < e; p++) {
            const j = nbr[p];
            if (act[j]) { c += cosT[j]; s += sinT[j]; }
          }
        } else {
          for (let p = off[i]; p < e; p++) {
            const j = nbr[p];
            c += cosT[j]; s += sinT[j];
          }
        }
        const ci = cosT[i], si = sinT[i];
        const coupling = s * ci - c * si;
        let alpha = adaptive[i] ? Math.sqrt(c * c + s * s) / deg[i] : 1;
        if (anyHalted && !act[i]) alpha = 0;
        let d = omega[i] + lamScale * alpha * coupling;
        if (eps !== 0) {
          const m = shockEps * mask[i] + newsEps;
          if (m !== 0) d += m * (Math.sin(SELL) * ci - Math.cos(SELL) * si);
        }
        out[i] = d;
      }
    }

    /* one stochastic Heun step */
    step(dt) {
      const n = this.n, th = this.theta, noise = this.noise;
      /* derive the halt flag from `active` rather than trusting a flag someone may forget to set:
         a native scan of a Uint8Array costs well under 1 us against ~180 us for the step */
      this.anyHalted = this.active.indexOf(0) !== -1;
      const sq = Math.sqrt(dt) * this.sigma;
      const sqc = Math.sqrt(dt) * this.sigmaC;
      const common = sqc > 0 ? sqc * this.randn() : 0;
      if (sq > 0 || common !== 0) {
        for (let i = 0; i < n; i++) noise[i] = (sq > 0 ? sq * this.randn() : 0) + common;
      } else {
        noise.fill(0);
      }
      this.drift(th, this.f1);
      for (let i = 0; i < n; i++) this.pred[i] = th[i] + dt * this.f1[i] + noise[i];
      this.drift(this.pred, this.f2);
      for (let i = 0; i < n; i++) th[i] += 0.5 * dt * (this.f1[i] + this.f2[i]) + noise[i];
      this.t += dt;

      if (this.halted && this.t >= this.haltUntil) this.resume();

      this.computeGlobal();
      /* index price: dx = (beta Q - kappa x) dt + eta dB */
      this.x += (this.beta * this.Q - this.kappa * this.x) * dt + this.eta * Math.sqrt(dt) * this.randn();
      this.price = 100 * Math.exp(this.x);
    }

    /* per-step: global order parameters only (O(n)) */
    computeGlobal() {
      const n = this.n, th = this.theta;
      let sc = 0, ss = 0, s2c = 0, s2s = 0;
      for (let i = 0; i < n; i++) {
        const c = Math.cos(th[i]), s = Math.sin(th[i]);
        this.cosT[i] = c; this.sinT[i] = s; this.q[i] = s;
        sc += c; ss += s;
        s2c += c * c - s * s;          // cos 2theta
        s2s += 2 * s * c;              // sin 2theta
      }
      sc /= n; ss /= n; s2c /= n; s2s /= n;
      this.R = Math.sqrt(sc * sc + ss * ss);
      this.psi = Math.atan2(ss, sc);
      this.Q = ss;
      this.R2 = Math.sqrt(s2c * s2c + s2s * s2s);
      this.meanCos = sc;
    }

    /* per-frame only: local agreement r_i costs O(E) and is needed for rendering, not for the drift
       (drift computes its own alpha_i inline), so it is kept out of the integration loop */
    computeLocal() {
      const n = this.n, off = this.off, nbr = this.nbr, cosT = this.cosT, sinT = this.sinT;
      for (let i = 0; i < n; i++) {
        let c = 0, s = 0;
        const e = off[i + 1];
        for (let p = off[i]; p < e; p++) { const j = nbr[p]; c += cosT[j]; s += sinT[j]; }
        this.rLocal[i] = Math.sqrt(c * c + s * s) / this.deg[i];
      }
    }

    computeObservables() { this.computeGlobal(); this.computeLocal(); }

    /* common sell shock applied to a fraction of nodes chosen by rule */
    applyShock(eps, fraction, rule) {
      const n = this.n;
      const idx = Array.from({ length: n }, (_, i) => i);
      if (rule === 'hubs') idx.sort((a, b) => this.deg[b] - this.deg[a]);
      else if (rule === 'periphery') idx.sort((a, b) => this.deg[a] - this.deg[b]);
      else for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(this.rand() * (i + 1));
        const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
      }
      this.shockMask.fill(0);
      const cnt = Math.max(1, Math.round(fraction * n));
      for (let i = 0; i < cnt; i++) this.shockMask[idx[i]] = 1;
      this.shockEps = eps;
      this.shockedNodes = cnt;
    }

    clearShock() { this.shockEps = 0; this.shockMask.fill(0); }

    halt(duration, fraction) {
      const n = this.n;
      const cnt = Math.max(1, Math.round((fraction === undefined ? 1 : fraction) * n));
      const idx = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(this.rand() * (i + 1));
        const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
      }
      this.active.fill(1);
      for (let i = 0; i < cnt; i++) this.active[idx[i]] = 0;
      this.halted = true; this.anyHalted = true;
      this.haltUntil = this.t + duration;
    }

    resume() { this.active.fill(1); this.halted = false; this.anyHalted = false; this.haltUntil = -1; }
  }

  const api = { Sim, mulberry32, gaussian, normInv, SELL, TAU };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else { root.Sim = Sim; root.SimLib = api; }
})(typeof window !== 'undefined' ? window : globalThis);
