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

## Verdict (revised 2026-10-14 after an external novelty audit)

The first verdict below was too generous about the mechanisms. The corrected position:

1. **The mechanisms are not new.** A dynamical mean-field theory for sparse directed graphs exists
   (Metz 2025), a dynamical cavity method covering reciprocal edges exists (Metz and Perez Castillo
   2026), and isolating an Onsager reaction term by comparing symmetric with asymmetric coupling in
   oscillator networks goes back to Aonishi, Kurata and Okada (2002). The dynamic-filtering term here is
   the coloured-noise channel of those frameworks and the reaction term is their response channel.
2. **What appears to be new** is the explicit evaluation for the adaptive (local order parameter)
   Kuramoto model: closed-form leading coefficients for the fold and for the synchronised branch on
   directed and undirected graphs, the falsification of the static-scatter (Rician) picture that goes
   back to Ichinomiya (2005), and the quantitative tests by simulation, including the reciprocity
   interpolation.
3. **Framing.** A regular article: explicit leading-order sparse-graph corrections for adaptive
   Kuramoto dynamics with a directed/undirected decomposition verified by simulation. Not a new cavity
   method, and the directed/undirected distinction is not claimed as new.
4. **Still unchecked:** Hatchett and Uezu (2008) in full text, the Coolen-group papers on finitely
   connected XY dynamics, and forward citations of Hatchett and Uezu (2008) and Ichinomiya (2005). An
   explicit 1/k Kuramoto threshold could exist there.

Original verdict of 2026-10-13, kept for the record: no paper was found that derives or measures a
finite-degree correction to the fold of the local-order-parameter model, a critical connectivity for
its bistability, or a cavity treatment of it. That statement still stands for the explicit result; it
no longer stands for "a cavity treatment" as a method.

## Frameworks that anticipate the mechanism

"Checked" means the arXiv abstract page was fetched on 2026-10-14 and the title, authors and stated
content match. "From the audit" means the description is taken from the external audit and has not been
verified against the paper here. None of these was read in full text in this repository.

| paper | what it does | relation to this work | status |
|---|---|---|---|
| Metz, PRL 134, 037401 (2025), arXiv:2406.06346 | Exact single-site path-probability equation for nonlinear dynamics on sparse directed random graphs, solved by population dynamics; synchronization named as an application | Our directed-graph term should be the leading 1/k expansion of this equation for the adaptive coupling. To be shown, analytically or by population dynamics. The audit reports that the paper excludes bidirected edges and lists Kuramoto without solving it | abstract checked; details from the audit |
| Metz and Perez Castillo, arXiv:2606.08689 (2026) | Continuous-time dynamical cavity method on sparse random graphs; reciprocal edges need conditional path kernels; dense limits give drift, noise and response channels | Predicts the structure of our decomposition. Our contribution is the explicit coefficients and their test. Unrefereed preprint | abstract checked |
| Aonishi, Kurata, Okada, PRE 65, 046223 (2002), arXiv:cond-mat/0201453 | Oscillator associative-memory models with symmetric and with asymmetric dilution, which differ by the Onsager reaction term | Direct conceptual precedent for "undirected minus directed equals the reaction term". Differences: random sparse topology, Lorentzian Kuramoto setting, adaptive gain, explicit 1/k coefficient | abstract checked |
| Ichinomiya, PRE 72, 016109 (2005), arXiv:cond-mat/0507285 | Path-integral treatment of dynamics on a sparse random network; equivalent to globally coupled oscillators with random Gaussian interactions | Origin of the static Gaussian scatter picture, which is what the Rician closure assumes and what the branch test falsifies | abstract checked |
| Pruser, Rosmej, Engel, PRL 132, 187201 (2024), arXiv:2310.09079; Pruser and Engel, PRE 110, 064214 (2024) | Dynamical cavity method for the fully disordered dense Kuramoto model with self-consistent correlation and response functions | Dense counterpart of the reaction term | first paper: abstract checked; second: from the audit |
| Reddy, arXiv:2603.09402 (2026) | Dense dynamical mean-field theory of oscillator networks: one oscillator driven by a mean field and self-consistent coloured Gaussian noise; reduces to Ott-Antonsen without disorder | Closest dense analogue of "Ott-Antonsen driven by coloured noise". The audit reports it lists the sparse regime as open | abstract checked |
| Hatchett and Uezu, PRE 78, 036106 (2008) | Cavity analysis of oscillator networks on sparse random graphs in a Hamiltonian (equilibrium) case | Must be read in full for any finite-connectivity threshold expansion | from the audit, unread |
| Skantzos, Perez Castillo, Hatchett, PRE 72, 066127 (2005), arXiv:cond-mat/0508609 | Equilibrium cavity method for XY spins on small-world graphs | Finite-connectivity cavity lineage for phase variables; no frequency disorder, so no drifting oscillators | from the audit |
| Restrepo, Ott, Hunt, Chaos 16, 015107 (2006), arXiv:cond-mat/0509255 | Synchronization onset on large directed networks | Directed against undirected for the continuous onset only; no reaction term | from the audit |

## Closest papers, one line each on how they differ

| paper | what it does | how it differs from Paper A |
|---|---|---|
| Zhang, Boccaletti, Guan, Liu, PRL 114, 038701 (2015), arXiv:1410.2986 | Introduces coupling scaled by the local order parameter; explosive synchronization without degree-frequency correlation | Numerical; no theory of the thresholds' dependence on mean degree |
| Zou and Wang, PRE 102, 012219 (2020) | Exact bifurcation analysis of global coupling `K r^(alpha-1)` | All-to-all only. Gives our `lam_b_inf(p)` and `R*(p)` with `alpha = p + 1` |
| Khanra, Kundu, Pal, Ji, Hens, Chaos 30, 031101 (2020), arXiv:2104.06862 | Adaptive coupling by a power `z` of the order parameter on Sakaguchi-Kuramoto networks; semi-analytic backward threshold | Read in full text: uses the **global** order parameter, the annealed network approximation and the Ott-Antonsen ansatz. No finite-degree term. From the supervisor's own group: must be cited and contrasted |
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

1. The introduction must contrast with Khanra et al. (2020) explicitly: same family of adaptive
   exponents, but global order parameter and annealed theory there, local order parameter and sparse
   graphs here.
2. The "Rician" language is ours; no prior use in this context was found. That is not evidence of
   novelty of the physics, only of the name.
3. If the measured fold turns out to agree with neither the closure nor a cavity theory, the paper is a
   numerical one and should be framed against Juhasz, Kelling and Odor (2019) for methodology.

4. The paper must state that its directed-graph term is an evaluation of Metz (2025) and its reaction
   term an evaluation of the response channel of Metz and Perez Castillo (2026), and must cite Aonishi,
   Kurata and Okada (2002) for the symmetric-against-asymmetric idea.
5. The reciprocity interpolation (`configs/paperA_reciprocity.yaml`) is the test that turns a known
   concept into a quantitative prediction: the branch shift must be linear in the mutual fraction.

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
- https://arxiv.org/abs/2406.06346
- https://arxiv.org/abs/cond-mat/0201453
- https://arxiv.org/abs/cond-mat/0507285
- https://arxiv.org/abs/2310.09079
- https://arxiv.org/abs/2603.09402
