# Prior-art search for Paper A

Date: 2026-10-13. Question: has anyone published how finite connectivity shifts or closes the bistable
window of Kuramoto oscillators with local-order-parameter (adaptive) coupling?

## Method and its limits

Web search (general and arXiv-indexed) on these query families: "local order parameter" with
"random regular" / "finite degree" / "sparse"; "adaptive coupling" with "mean degree" and hysteresis;
"cavity method" / "message passing" / "population dynamics" / "Bethe" with Kuramoto; "Rician" /
"Rice distribution" with Kuramoto; higher-order Kuramoto with sparsity and saddle-node shift. The
closest hits were then read (abstract, or full text where a PDF was available).

This is **not** a full-text search of arXiv, APS or Google Scholar, which need interactive access. A
negative result here lowers the probability of an existing equivalent result; it does not exclude one.
Before submission, repeat the queries by hand on Google Scholar and the APS search, and follow the
citing-paper lists of Zhang et al. (2015), Zou and Wang (2020) and Restrepo, Ott and Hunt (2005).

## Verdict

No paper was found that derives or measures a finite-degree correction to the fold of the
local-order-parameter model, a critical connectivity for its bistability, or a cavity treatment of it.
Every analytic treatment found is all-to-all, annealed, or dense. Paper A can therefore proceed as a
theory-plus-measurement paper, with the caveat above. The novelty claim must be limited to the
finite-connectivity effect; everything in the all-to-all limit is reproduction.

## Closest papers, one line each on how they differ

| paper | what it does | how it differs from Paper A |
|---|---|---|
| Zhang, Boccaletti, Guan, Liu, PRL 114, 038701 (2015), arXiv:1410.2986 | Introduces coupling scaled by the local order parameter; explosive synchronization without degree-frequency correlation | Numerical; no theory of the thresholds' dependence on mean degree |
| Zou and Wang, PRE 102, 012219 (2020) | Exact bifurcation analysis of global coupling `K r^(alpha-1)` | All-to-all only. Gives our `lam_b_inf(p)` and `R*(p)` with `alpha = p + 1` |
| Khanra, Kundu, Pal, Ji, Hens, Chaos (2021), arXiv:2104.06862 | Adaptive coupling by a power `z` of the order parameter on Sakaguchi-Kuramoto networks; semi-analytic backward threshold | Read in full text: uses the **global** order parameter, the annealed network approximation and the Ott-Antonsen ansatz. No finite-degree term. From the supervisor's own group: must be cited and contrasted |
| Dutta, Pal, Hens, arXiv:2507.18421 (2025); Dutta et al., PRR 7, L022049 (2025) | Generalised adaptation on random hypergraphs; double explosive transition | Dense-limit Ott-Antonsen analysis |
| Kumpeerakij and Restrepo, arXiv:2605.24701 (2026) | Self-consistent local-order-parameter theory for dyadic plus triadic Kuramoto on finite hypergraphs | Read in full text: keeps the specific graph and frequency realisation but explicitly assumes enough neighbours that fluctuations of the local field are negligible; test networks have mean degree `N/40`. No sparse expansion, no `1/k` law |
| Restrepo, Ott, Hunt, PRE 71, 036151 (2005) | Onset of synchronization on networks beyond mean field; discusses fluctuations of the local field | Standard (non-adaptive) Kuramoto, onset threshold, not a fold; no adaptive gain. The natural starting point for a fluctuation expansion, and the reference Kumpeerakij and Restrepo cite for it |
| Ichinomiya, PRE 70, 026116 (2004) | Heterogeneous mean-field onset `K_c <k>/<k^2>` | Annealed |
| Juhasz, Kelling, Odor, arXiv:1902.10422 (2019) | Large-scale simulation of Kuramoto on sparse Erdos-Renyi graphs; strong corrections to scaling | Continuous transition of the standard model; numerical |
| Metz and Perez Castillo, arXiv:2606.08689 (2026) | Continuous-time dynamical cavity method for general pairwise dynamics on sparse random graphs, with population dynamics | General method paper (abstract only read). Does not treat adaptive Kuramoto coupling. Note the gain `r_i^p` is not pairwise, so their framework needs extending. This is the methodological reference for the cavity theory |
| Skardal and Arenas, PRL 122, 248301 (2019); Commun. Phys. 3, 218 (2020) | Higher-order interactions produce abrupt switching | All-to-all |
| Xu et al., PRR 3, 043004 (2021) | Nonlinear coupling as higher-order interaction | All-to-all |
| Malizia et al., arXiv:2307.03519, arXiv:2601.10522 | Hyperedge overlap and nestedness shrink the bistable region | Structural effect of overlap at fixed generalised degree; thresholds found numerically, no connectivity law |
| Adhikari, Restrepo, Skardal, Chaos (2023), arXiv:2208.00909 | Reduced model for hypergraphs with links and triangles | Mean-field in degree classes |

## Consequences recorded

1. The introduction must contrast with Khanra et al. (2021) explicitly: same family of adaptive
   exponents, but global order parameter and annealed theory there, local order parameter and sparse
   graphs here.
2. The "Rician" language is ours; no prior use in this context was found. That is not evidence of
   novelty of the physics, only of the name.
3. If the measured fold turns out to agree with neither the closure nor a cavity theory, the paper is a
   numerical one and should be framed against Juhasz, Kelling and Odor (2019) for methodology.

## Sources

- https://arxiv.org/pdf/1410.2986
- https://arxiv.org/abs/2104.06862
- https://arxiv.org/abs/2605.24701
- https://arxiv.org/abs/2606.08689
- https://arxiv.org/abs/1902.10422
- https://arxiv.org/pdf/1511.07139
- https://www.colorado.edu/amath/sites/default/files/attached-files/physreve_71_036151.pdf
- https://arxiv.org/pdf/2307.03519
- https://arxiv.org/pdf/2601.10522
- https://arxiv.com/abs/2208.00909
