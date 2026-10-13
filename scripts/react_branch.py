import numpy as np
from esfc import cavity
from esfc.closure import lam_b_inf
import branch_theory as bt

def react_branch(lam, p):
    c = bt.constants(lam, p)
    fc = {"lam": lam, "R": c["R"], "h": c["h"], "mu": c["mu"], "L": c["L"], "alpha": c["al"], "beta": c["be"],
          "dz0_dh": c["dz0dh"], "dz0_dlam": c["R"] ** (p + 1) * c["dz0dh"]}
    orig = cavity._fold_constants
    cavity._fold_constants = lambda p_: fc
    try:
        locked, drift = cavity.reaction_parts.__wrapped__(p)
    finally:
        cavity._fold_constants = orig
    denom = 1 - c["dz0dh"] * lam * (p + 1) * c["R"] ** p
    return locked / denom, drift / denom

if __name__ == "__main__":
    # sanity: at the fold the original code must be reproduced through the same path
    for p in (1.0, 2.0):
        lam = 1.25 * lam_b_inf(p)
        lk, dr = react_branch(lam, p)
        b = bt.branch_shift(lam, p)
        print(f"p={p} mult=1.25 R0={b['R0']:.4f} dyn={b['dynamic']:.4f} react_locked={lk:.4f} react_drift={dr:.4f} undirected={b['dynamic']+lk+dr:.4f} closure={b['closure']:.4f}")
