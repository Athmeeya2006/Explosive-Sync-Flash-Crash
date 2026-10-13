"""Second-order shift of the synchronised branch R(lam) at finite k, for lam above the fold.

Same expansion as esfc.cavity (docs/THEORY.md section 3), evaluated at an arbitrary point (lam, R0) on
the all-to-all stable branch instead of at the fold. Away from the fold the self-consistency
R = z0(lam R^(p+1)) + <a2>/k gives a finite shift

    k (R - R0) -> c(lam) = <a2> / (1 - z0'(h) lam (p+1) R0^p).

Variants: 'closure' (all scatter frozen: must match the Rician closure), 'dynamic' (directed graphs).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def z0(h):
    return (np.sqrt(1 + h * h) - 1) / h


def stable_R0(lam, p):
    Rs = np.linspace(np.sqrt(p / (p + 2)), 0.9999, 4000)
    f = Rs - z0(lam * Rs ** (p + 1))
    i = np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    j = i[-1]
    return brentq(lambda R: R - z0(lam * R ** (p + 1)), Rs[j], Rs[j + 1])


def constants(lam, p):
    R = stable_R0(lam, p)
    h = lam * R ** (p + 1)
    mu = np.sqrt(1 + h * h)
    return dict(lam=lam, R=R, h=h, mu=mu, L=lam * R ** p, al=(1 - R * R) / 2, be=(1 + R * R) / 2,
                dz0dh=(mu - 1) / (mu * h * h))


def variances(c):
    h, mu, R = c["h"], c["mu"], c["R"]
    g = lambda x: (1 / h) / (np.pi * (x * x + 1 / h ** 2))  # noqa: E731

    def harm(x, filt):
        q = x - np.sqrt(x * x - 1)
        W = h * np.sqrt(x * x - 1)
        s = 0.0
        for n in range(1, 4000):
            w = (1 - q * q) ** 2 * q ** (2 * (n - 1)) / 2
            if w < 1e-17:
                break
            s += w * (mu * mu / (mu * mu + (n * W) ** 2) if filt else 1.0)
        return s * g(x)

    cuts = (1.0, 1.02, 1.2, 3.0, 30.0, np.inf)
    rot = lambda filt: 2 * sum(quad(lambda x: harm(x, filt), a, b, limit=200)[0]  # noqa: E731
                               for a, b in zip(cuts[:-1], cuts[1:]))
    frozen = quad(lambda x: (1 - x * x) * g(x), -1, 1)[0] - R * R
    return dict(inst=(1 - R * R) / 2, frozen=frozen, rot=rot(False), rot_f=rot(True))


def a2(c, p, v_resp):
    lam, R, h, mu, L, al, be = (c[k] for k in ("lam", "R", "h", "mu", "L", "al", "be"))
    v_inst = (1 - R * R) / 2
    return (1 / (2 * mu)) * ((1 - R * R) * lam * R ** (p - 1) * (p * (p + 2) / 2) * v_inst
                             - 2 * R * al * L * L * (p + 1) ** 2 * v_resp / mu
                             - 2 * R * be * L * L * v_resp / mu
                             - h * L * L * (al ** 2 * (p + 1) ** 2 - be ** 2) * v_resp / mu ** 2)


def branch_shift(lam, p):
    c = constants(lam, p)
    v = variances(c)
    denom = 1 - c["dz0dh"] * lam * (p + 1) * c["R"] ** p
    return dict(R0=c["R"], closure=a2(c, p, v["inst"]) / denom,
                dynamic=a2(c, p, v["frozen"] + v["rot_f"]) / denom,
                frozen_only=a2(c, p, v["frozen"]) / denom, check=v["frozen"] + v["rot"] - v["inst"])


if __name__ == "__main__":
    from esfc.closure import lam_b_inf, self_consistency
    for p in (1.0, 2.0):
        for m in (1.25, 1.5, 2.0):
            lam = m * lam_b_inf(p)
            b = branch_shift(lam, p)
            # cross-check the 'closure' variant against the full Rician closure at large k
            out = []
            for k in (400, 1600):
                Rk = brentq(lambda R: self_consistency(lam, R, p, k) - R, 0.5 * b["R0"] + 0.3, 0.9999)
                out.append(round(k * (Rk - b["R0"]), 4))
            print(p, m, {k: round(v, 4) for k, v in b.items()}, "rician k*(R-R0) at k=400,1600:", out)
