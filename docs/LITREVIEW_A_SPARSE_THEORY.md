# Literature review for Paper A: sparse-graph theory of the adaptive Kuramoto fold

Date: 2026-10-25. Scope: the closest published or preprinted work to each claim of Paper A (adaptive
Kuramoto model with gain `r_i^p` on sparse random graphs, leading `1/k` corrections to the fold and to
the synchronised branch, directed against undirected graphs, reciprocity, critical connectivity,
clustering and degree heterogeneity). Builds on `docs/PRIOR_ART.md` (abstract-level search of
2026-10-13/14); this review reads the closest papers in full text.

Method. Web search (general and extended), arXiv API, and full-text reading of downloaded arXiv PDFs
(stored in `references/12_lit_review_sparse_theory/`, listed in its `SOURCES.md`). Limits: no Google
Scholar or APS full-text search, no forward-citation lists, publisher pages behind paywalls not read.
The Consensus search quota was exhausted during the session. A negative result here lowers but does not
exclude the chance that an equivalent result exists.

Notation used below: `k` degree, `p` adaptive exponent, `lambda_b` backward threshold (fold), `rho`
fraction of mutual links, "closure" the Rician (frozen Gaussian scatter) closure, "dynamic" the theory with
rotating neighbour scatter filtered by `mu^2/(mu^2 + W^2)`, "reaction" the Onsager-like echo through mutual
links (THEORY.md sections 2 to 4 and 6).

---

## (a) All relevant papers found

"DL" = PDF downloaded into `references/12_lit_review_sparse_theory/` in this review; "repo" = PDF
already elsewhere in `references/`; "no" = not downloaded (paywalled or no arXiv version found).

| # | paper | one line | DL |
|---|---|---|---|
| 1 | Metz, PRL 134, 037401 (2025), arXiv:2406.06346 | Exact single-site path-probability equation for pairwise dynamics on sparse **directed** random graphs, solved by population dynamics; Kuramoto listed but not solved | DL |
| 2 | Metz and Perez Castillo, arXiv:2606.08689 (2026) | Continuous-time dynamical cavity method on sparse graphs; reciprocal edges need conditional path kernels; dense projection has drift, coloured-noise and response channels | DL |
| 3 | Tarabolo and Dall'Asta, SciPost Phys. 19, 019 (2025), arXiv:2406.14200 | Gaussian Expansion Cavity Method (GECaM): second-order small-coupling truncation of dynamic cavity equations on sparse graphs, with mean, correlation and response cavity fields | DL |
| 4 | Tarabolo, Dall'Asta, Mulet, arXiv:2512.22563 (2025) | GECaM applied to Lotka-Volterra on sparse undirected graphs; a "topological" multistability transition driven purely by finite connectivity; reciprocity of couplings tuned | DL |
| 5 | Machado, Valigi, Tonolo, Angelini, arXiv:2511.17499 (2025/26) | Local Fokker-Planck equations with mean-field closure for Lotka-Volterra on sparse asymmetric graphs | DL |
| 6 | Kumpeerakij and Restrepo, arXiv:2605.24701 (2026) | Self-consistent local-order-parameter theory of dyadic plus triadic Kuramoto on finite hypergraphs; bistability onset; assumes many neighbours | DL (also repo) |
| 7 | Restrepo, Ott, Hunt, PRE 71, 036151 (2005), arXiv:cond-mat/0411202 | Onset on networks via time-averaged local field; Sec. VI heuristic theory of local-field fluctuations: they **raise** the onset at small degree | DL (also repo) |
| 8 | Restrepo, Ott, Hunt, Chaos 16, 015107 (2006), arXiv:cond-mat/0509255 | Extension of the onset theory to directed networks and mixed-sign coupling | DL |
| 9 | Ichinomiya, PRE 72, 016109 (2005), arXiv:cond-mat/0507285 | Path integral: sparse random network approximated by globally coupled network with Gaussian random couplings | DL |
| 10 | Aonishi and Okada, arXiv:cond-mat/9909048 (2001 version) | Onsager reaction term (ORT) isolated by comparing symmetric and asymmetric dilution in oscillator associative memories (SCSNA) | DL |
| 11 | Aonishi, Kurata, Okada, PRE 65, 046223 (2002), arXiv:cond-mat/0201453 | Acceleration effect: curved-isochron clock, and the same symmetric versus asymmetric dilution comparison | DL |
| 12 | Pruser, Rosmej, Engel, PRL 132, 187201 (2024), arXiv:2310.09079 | Dynamical cavity method for dense, fully disordered Kuramoto; self-consistent correlation and response | DL |
| 13 | Skantzos, Perez Castillo, Hatchett, PRE 72, 066127 (2005), arXiv:cond-mat/0508609 | Equilibrium cavity method for XY spins on small-world graphs | DL |
| 14 | Zhang, Boccaletti, Guan, Liu, PRL 114, 038701 (2015), arXiv:1410.2986 | Introduces the local-order-parameter (adaptive) coupling; ES without degree-frequency correlation; mean field with `r_i = R` | repo |
| 15 | Zou and Wang, PRE 102, 012219 (2020) | Exact all-to-all bifurcation analysis of `K r^(alpha-1)` coupling; source of `lambda_b(inf)` | repo |
| 16 | Danziger, Moskalenko, Kurkin, Zhang, Havlin, Boccaletti, Chaos 26, 065307 (2016) | Adaptive fraction `f`: hysteresis for every `f > 0` (contradicts the finite `f_c` of item 14); mean field and simulations at N about 1e6 | no (no arXiv copy found) |
| 17 | Khanra, Kundu, Pal, Ji, Hens, Chaos 30, 031101 (2020), arXiv:2104.06862 | Adaptive gain `R^(z-1)` with the **global** order parameter on Sakaguchi-Kuramoto networks; annealed network + Ott-Antonsen backward threshold | DL |
| 18 | Dutta, Pal, Hens, arXiv:2507.18421 (2025) | Generalised adaptation on random hypergraphs, double explosive transitions, dense Ott-Antonsen | repo |
| 19 | Peron, Rodrigues, Kurths, PRE 87, 032807 (2013), arXiv:1210.2625 | Kuramoto on clustered random networks (Newman-Miller triangles): annealed theory says triangles do not matter for the onset | DL |
| 20 | McGraw and Menzinger, PRE 72, 015101 (2005) | Raising clustering at fixed degree distribution hinders global synchronisation | no |
| 21 | Juhasz, Kelling, Odor, J. Stat. Mech. (2019) 053403, arXiv:1902.10422 | Large-scale simulation of standard Kuramoto on sparse Erdos-Renyi graphs; strong corrections to scaling | DL |
| 22 | Um, Hong, Park, Sci. Rep. 14, 6816 (2024), arXiv:2403.14984 | Validity of the annealed approximation: fails in the incoherent phase even on dense networks | no |
| 23 | Gast, Takasu, Kurths, Kennedy, arXiv:2609.22597 (2026) | Second-order moment closure for weight variance in adaptive oscillator networks; bistability from symmetric adaptation | DL |
| 24 | Reddy, PRE 113, 034222 (2026), arXiv:2603.09402 | Compact dense DMFT of oscillator networks (single oscillator + coloured Gaussian noise; recovers Ott-Antonsen without disorder) | no |
| 25 | Tonolo, Angelini, Azaele, Maritan, Gradenigo, arXiv:2503.20887 (PRX Life 2026) | Symmetric Lotka-Volterra on random regular graphs: cavity/BP, non-Gaussian effects, topological glass | no |
| 26 | Hatchett and Uezu, PRE 78, 036106 (2008) | Cavity analysis of oscillator networks on sparse random graphs (Hamiltonian case) | no (not found online; still unread) |
| 27 | Chiba and Medvedev, arXiv:1612.06493 (and follow-ups) | Onset of synchronisation on graph limits (dense and sparse W-random graphs), eigenvalue of the kernel | no |
| 28 | Brede, arXiv:0809.2117 | Kuramoto on directed small worlds; feed-forward loops and cycles | no |
| 29 | "Dynamical origin of the explosive synchronization with partial adaptive coupling", Chaos Solitons Fractals (2023), Elsevier pii S0960077923004393 | Self-consistent analysis of partial adaptive coupling (abstract only; authors not verified, page paywalled) | no |
| 30 | Ichinomiya, PRE 70, 026116 (2004) | Heterogeneous (annealed) mean-field onset `K_c <k>/<k^2>` | repo |
| 31 | Rodrigues, Peron, Ji, Kurths, Phys. Rep. 610, 1 (2016) | Review of Kuramoto on complex networks, annealed against quenched | repo |
| 32 | Adhikari, Restrepo, Skardal, Chaos (2023), arXiv:2208.00909 | Degree-based reduction for hypergraphs with links and triangles; bistability onset | no |

Closest 10 (ordered by overlap with Paper A): 2, 1, 3, 7, 10/11, 9, 6, 4, 14, 17. Full-text notes
follow for these (items 10 and 11 together).

---

## (b) Detailed notes on the closest papers

### B1. Metz and Perez Castillo (2026), dynamical cavity method on sparse graphs (item 2)

**Citation.** F. L. Metz and I. Perez Castillo, "Dynamical cavity method for continuous-time complex
systems on sparse random graphs", arXiv:2606.08689 (7 June 2026), submitted to SciPost Physics. No DOI
yet.

**What they did.** Path-measure cavity derivation for `x_i' = -f(x_i) + sum_j A_ij g(x_i, x_j) + noise`
with a general **pairwise** kernel `g` (Sec. 2 to 3). On directed graphs the outgoing feedback kernels
normalise away and Metz (2025) is recovered; with reciprocal edges a neighbour's branch is driven by the
imposed history of the root, so the messages become conditional path kernels (Sec. 3, Conclusions).
Causal discretisation and population dynamics (Sec. 5), tested only on an additive-input recurrent neural
network. Sec. 6: high-connectivity projections give drift, coloured-noise and, with reciprocity, a response
(memory) channel. The response channel is parametrised by the reciprocal coupling covariance
`<J_ij J_ji> - <J_ij><J_ji> = rho_J / c` (Eq. 194), and the authors state that the density of reciprocal
partners multiplies its strength.

**Differences from Paper A.** (i) Their kernel is pairwise; our gain `|F_i|^p` multiplies the whole local
sum, so the coupling is not a sum of pairwise terms. The directed construction still applies (it needs only
the input paths), the reciprocal one needs the conditional kernel for a non-additive input. (ii) No
oscillator model is solved. (iii) Their dense projection keeps terms of order one under the scaling
`<J> = mu_J/c`, `<J^2> = gamma_J^2/c`. For deterministic couplings `J = lambda/k` (our case) the coupling
variance is zero, the coloured-noise and response channels are of order `1/k`, and they are exactly the
terms the projection discards. Paper A's result is the next order beyond their dense limit, for a
non-pairwise kernel.

**Shortcomings.** Limitations rather than mistakes: unrefereed; no numerical test for reciprocal graphs
beyond the RNN (they state efficient representations of reciprocal kernels as an open problem); the
response channel in Eq. (194) is written via coupling disorder, so for homogeneous couplings with random
topology the reader has to supply the reciprocal-topology version; whether the dense channels close in
`m, C, R` is left open for nonlinear kernels (Sec. 6). No mistake found in the parts read.

### B2. Metz (2025), sparse directed DMFT (item 1)

**Citation.** F. L. Metz, "Dynamical Mean-Field Theory of Complex Systems on Sparse Directed Networks",
Phys. Rev. Lett. 134, 037401 (2025), arXiv:2406.06346, DOI 10.1103/PhysRevLett.134.037401.

**What they did.** Eq. (7): exact path-probability equation for one node on a sparse directed
configuration-model graph with independent in- and out-degrees, as `N -> infinity`; solved by a population
of trajectories (Fig. 1, `c = 5`, SIS, Lotka-Volterra and neural network, `N = 4000` checks). Phase diagram
of the Sompolinsky neural network in `(c, sigma_J)` (Fig. 2). Supplement: large-`c` limit (Eqs. 54 to 58)
with couplings scaled `mu_J/c`, variance `gamma_J^2/c`; degree fluctuations survive the limit unless the
rescaled in-degree distribution is a delta function.

**Differences.** Table I lists Kuramoto (`f = 0`, `g = sin(x' - x)`) but no oscillator result is shown.
No frequency disorder, no fold, no bidirected edges (stated: bidirected edges make the path dimension grow
exponentially).

**Shortcomings.** Limitations: Kuramoto is advertised in the abstract ("synchronization") but not solved;
the large-`c` expansion (Supplement Eq. 56) keeps only terms that survive with disordered couplings and
therefore says nothing about the `1/c` correction for homogeneous couplings; finite-time population
dynamics of trajectories is expensive near critical slowing down (acknowledged near `c*`). The degree-
fluctuation statement is relevant to our Erdos-Renyi result (C8 below). No mistake found.

### B3. Tarabolo and Dall'Asta (2025), Gaussian expansion cavity method (item 3), and Tarabolo, Dall'Asta, Mulet (2025) (item 4)

**Citations.** M. Tarabolo and L. Dall'Asta, "Gaussian approximation of dynamic cavity equations for
linearly-coupled stochastic dynamics", SciPost Phys. 19, 019 (2025), arXiv:2406.14200, DOI
10.21468/SciPostPhys.19.1.019. M. Tarabolo, L. Dall'Asta, R. Mulet, "Sparse Interactions Reshape Stability
in Random Lotka-Volterra Dynamics", arXiv:2512.22563 (2025).

**What they did.** GECaM: a second-order truncation of the small-coupling expansion of the dynamic cavity
equations on locally tree-like graphs, giving Gaussian cavity marginals and a single-site process with
mean field, coloured noise and an integrated response along each edge (item 4, Eq. 2: the term
`sum_k J_ik J_ki integral R_k\i(t,t') x_i(t') dt'`). Non-linear forces via a self-consistent perturbative
closure (item 3, Sec. 2.4). Item 4 applies it to Lotka-Volterra with couplings of mean `m/K`, variance
`sigma^2/K` and correlation `gamma` between `J_ij` and `J_ji`, assumes a time-translation-invariant fixed
point so that the noise "freezes" to a static variable, and finds a multistability transition at
`sigma = 0` driven by finite connectivity alone (Fig. 2), matching the Kesten-McKay bound
`m_c(K) = -K/(2 sqrt(K-1))`.

**Relation to Paper A.** This is the closest **methodological** overlap. Paper A's expansion is, in
substance, a second-order small-coupling (Gaussian) cavity expansion: the drift term, the coloured-noise
term (our "dynamic" filtering) and the edge response term (our "reaction"). The reciprocity parameter
`gamma` of item 4 plays the role of our `rho`, but as a correlation of coupling values, not of topology.

**Shortcomings.** (i) Item 4's abstract claims a "thermodynamically exact" phase diagram, while the method
is a second-order truncation (item 3 calls it the "primary approximation"); the claim should be read as
"exact within the Gaussian truncation", which is controlled only to leading order in `1/K` (the same order
as Paper A). This is an overstatement, not a computational error. (ii) Item 4 replaces the noise by a static
random variable in the stationary state; that is legitimate at a fixed point but not for oscillators, whose
stationary state contains drifting units. This is precisely the frozen-scatter step that Paper A shows fails
(the Rician closure), so their stationary shortcut cannot be transplanted to oscillators. (iii) The
multiple-equilibria boundary is located by non-convergence of the population-dynamics solver ("the
breakdown of the solver accurately marks the stability boundary"); outside the `sigma = 0` line where the
spectral bound is known, that criterion is not independently checked. Limitations: no oscillators, no
topological reciprocity.

### B4. Restrepo, Ott and Hunt (2005), onset on networks, and its Sec. VI (item 7)

**Citation.** J. G. Restrepo, E. Ott, B. R. Hunt, "Onset of synchronization in large networks of coupled
oscillators", Phys. Rev. E 71, 036151 (2005), arXiv:cond-mat/0411202, DOI 10.1103/PhysRevE.71.036151.

**What they did.** Time-averaged theory (TAT) of the local field; the onset is set by the largest
eigenvalue of `A` (Secs. II to V). Tests use networks with minimum degree 50. Sec. VI ("Effect of
fluctuations"): the coupling term is treated as noise, phases are modelled as `phi + omega t + W(t)` with
`W` a Gaussian random walk, diffusion coefficients `D_nm` are solved self-consistently (Eqs. 39 to 43), and
the growth rate is the leading eigenvalue of `(k/2)A - diag(D_m - i omega_m)` (Eq. 48). Result: local-field
fluctuations **raise** the onset, increasingly so at small degree (tested at `d = 20, 50, 100`, `N = 500`).

**Differences.** Standard Kuramoto, continuous onset of the incoherent state, not a fold of a synchronised
branch; no adaptive gain; no directed/undirected decomposition (Eq. 40 uses the symmetry of `A`).

**Shortcomings.** The authors call Sec. VI heuristic; it is. The Gaussian random-walk ansatz for phase
fluctuations (Eq. 41 onward) is not derived and ignores the coherent (frozen) part of the field; the tests
use `N = 500`, where finite-`N` effects and finite-`d` effects are confounded; no closed `1/d` coefficient is
extracted. These are limitations, honestly flagged in the paper. It remains the earliest quantitative claim
that local-field fluctuations shift a synchronisation threshold, and the reference Kumpeerakij and Restrepo
cite for it.

### B5. Aonishi and Okada (2001), and Aonishi, Kurata, Okada (2002): the Onsager reaction term (items 10, 11)

**Citations.** T. Aonishi and M. Okada, "Acceleration effect caused by the Onsager reaction term in a
frustrated coupled oscillator system", arXiv:cond-mat/9909048 (v6, 2001); published journal version not
verified here. T. Aonishi, K. Kurata, M. Okada, "Acceleration effect of coupled oscillator systems", Phys.
Rev. E 65, 046223 (2002), arXiv:cond-mat/0201453, DOI 10.1103/PhysRevE.65.046223.

**What they did.** Oscillator associative memories with Hebbian couplings `J_ij e^{i beta_ij}`, a uniform
bias `beta_0`, and dilution `c_ij` with fan-in `O(N)` (dilution fraction `c = O(1)`), either symmetric
(`c_ij = c_ji`) or asymmetric. Using the SCSNA, the local field splits into a "pure" field and the ORT
`Gamma s_i` (Eq. 7). The two models have identical order-parameter equations in the renormalised variables
(Eqs. 12, 13) and differ only in `Gamma` (Eqs. 19 against 20). Because `Gamma s_i` is parallel to `s_i`, a
real `Gamma` drops out of the phase equation; the ORT only shifts the observable frequency through
`Omega_tilde = Omega - |Gamma| sin(psi)` (Eq. 9), i.e. only when `beta_0 != 0` (the "acceleration effect").

**Relation to Paper A.** Direct conceptual precedent for "undirected minus directed equals a reaction
term". The physics differs: in Paper A the reaction field (THEORY.md Sec. 4) is **anisotropic** about the
synchronised direction (transverse response 1, longitudinal response `L(p+1) dz0/dh`), so it does not drop
out and it shifts `R` and `lambda_b`, with no bias needed. In Aonishi and Okada an isotropic ORT changes only
the rotation frequency.

**Shortcomings.** (i) SCSNA is heuristic (the authors say so). (ii) The ORT is taken as static ("we assume
that microscopic memory effect can be neglected in the `t -> infinity` limit"), which is not justified for
drifting oscillators; Paper A's reaction for drifting nodes oscillates and is integrated, not frozen.
(iii) The complex-conjugate part of the ORT is dropped (justified only for the Hebbian rule, as stated).
(iv) Dense dilution (fan-in `O(N)`), so no finite-`k` statement. All four are limitations, stated in the
papers; no mistake found.

### B6. Ichinomiya (2005), path integral for sparse random networks (item 9)

**Citation.** T. Ichinomiya, "Path-integral approach to the dynamics in sparse random network", Phys. Rev.
E 72, 016109 (2005), arXiv:cond-mat/0507285, DOI 10.1103/PhysRevE.72.016109.

**What they did.** MSR generating functional averaged over a random network with link probabilities
`p_ij` (Eqs. 5 to 10); the `l = 1` term gives the annealed mean field (Eq. 12); the `l = 2` term, by a
Hubbard-Stratonovich step, gives Gaussian random couplings `r_ij` of variance `p_ij` (Eq. 13 onward), so the
sparse network maps to a globally coupled network with mean `k0/N` plus Gaussian disorder (Eq. 33); odd-`l`
terms are argued to be of order `p_ij^2` (Sec. III B). Sec. IV compares the sparse network and the Gaussian
model numerically (`N = 1000`, `k0 = 10`, Gaussian frequencies, 50 networks): order parameter, velocity
distribution and phase scatter agree qualitatively.

**Relation to Paper A.** It supports the statement in THEORY.md that only second moments of the
neighbour scatter enter at order `1/k` (cumulants of order `l` of a local sum of `k` terms each `O(1/k)` scale
as `k^{1-l}`). **Correction to our own PRIOR_ART.md:** Ichinomiya's Gaussian couplings are static, but the
oscillator dynamics in his globally coupled model is kept in full; solving that model by DMFT would produce
both coloured noise and a response term. The Rician closure (static scatter of the **field**) is therefore a
further approximation, not his result. Paper A should not say the closure "goes back to Ichinomiya"; it can
say Ichinomiya's mapping, solved without the time dependence and without the response, gives the closure.

**Shortcomings.** (i) Possible inconsistency: the ensemble average (Eqs. 6, 7) is a product over ordered
pairs `(i, j)` with independent `a_ij`, which is the directed ensemble; for the undirected Kuramoto network
of Sec. IV the paper then states that "the random Gaussian matrix needs to be considered as symmetric"
without deriving it from Eq. (7). For undirected graphs the average must be over unordered pairs, which
correlates `r_ij` and `r_ji`; that correlation is exactly what generates the reaction term. The final
statement (symmetric Gaussian matrix) is right, but the derivation as written covers the directed case.
(ii) Validation is qualitative: one `k0`, `N = 1000`, and the two order parameters compared are defined
differently (degree-weighted for the sparse network, unweighted for the Gaussian one). (iii) No threshold
shift is predicted. (ii) and (iii) are limitations; (i) is a gap in the derivation.

### B7. Kumpeerakij and Restrepo (2026), self-consistent higher-order Kuramoto (item 6)

**Citation.** C. Kumpeerakij and J. G. Restrepo, "Self-consistent analysis of the Kuramoto model with
higher-order interactions", arXiv:2605.24701 (May 2026), prepared for J. Phys.: Complexity.

**What they did.** Generalised local order parameters for dyadic plus triadic coupling; a hierarchy of
approximations (TAT, frequency-distribution approximation FDA, heterogeneous mean field) for local and
global order parameters; onset of synchronisation and critical triadic coupling for bistability, set by
eigenvector correlations between `A` and the triadic tensor; tests on hypergraphs with mean degree `N/40`.

**Differences.** Their Sec. 4 lists eight assumptions. Assumption 1 (constant local fields, "we assume
that the number of terms in the sums is large") and assumption 6 (self-averaging over neighbours) remove
exactly the `1/k` physics Paper A computes. Their triadic coupling is the closest higher-order analogue of
our `r_i^p` gain (`p = 2` corresponds to a local cubic term), but on dense hypergraphs.

**Shortcomings.** Limitations stated by the authors ("a rigorous characterization of the errors
introduced by these approximations is outside the scope of this paper"); all tests are dense, so the
regime of validity in `k` is not measured. No mistake found.

### B8. Zhang, Boccaletti, Guan, Liu (2015), the model (item 14)

**Citation.** X. Zhang, S. Boccaletti, S. Guan, Z. Liu, "Explosive synchronization in adaptive and
multilayer networks", Phys. Rev. Lett. 114, 038701 (2015), arXiv:1410.2986, DOI
10.1103/PhysRevLett.114.038701.

**What they did.** Coupling of a fraction `f` of nodes multiplied by their local order parameter `r_i`;
ES with hysteresis for `f > f_c` on Erdos-Renyi graphs (`N = 1000`, `<k> = 12`, uniform frequencies);
two-layer version; mean-field analysis with `r_i = R` (Eqs. 5 to 10).

**Shortcomings.** (i) The theory sets `r_i = R` and then writes `R e^{i Psi} = (1/N) sum e^{i theta_j}
= (1/(N<k>)) sum k_j e^{i theta_j}` (text before Eq. 8) as if it were an identity; it is the annealed
approximation, not an equality. Calling the analysis "rigorous" (abstract, intro) overstates it. (ii) The
finite `f_c` was later contradicted by Danziger et al. (2016), who find hysteresis for every `f > 0` at
`N` about 1e6, so the `f_c` of Fig. 1 is plausibly a finite-size effect. (iii) Single network
realisations, no error bars. (iv) Fig. 2(b) (layer II with `<k> = 6`) shows a smaller hysteresis width
than at `<k> = 12`: the only prior numerical hint of a degree dependence of the window, not followed up. The
mean-field `r_i = R` is the `k -> infinity` limit Paper A corrects.

### B9. Khanra, Kundu, Pal, Ji, Hens (2020) (item 17)

**Citation.** P. Khanra, P. Kundu, P. Pal, P. Ji, C. Hens, "Amplification of explosive width in complex
networks", Chaos 30, 031101 (2020), arXiv:2104.06862, DOI 10.1063/5.0003410.

**What they did.** Sakaguchi-Kuramoto on networks with gain `l_i = R^(z-1)`, `R` the **global** order
parameter (Eq. 2), so every node has the same gain; annealed network approximation (Eq. 3) plus
Ott-Antonsen per degree class (Eqs. 4 to 11); semi-analytic backward threshold compared with simulations
on a Barabasi-Albert network (`N = 2000`, `<k> = 12`); Watts-Strogatz and degree-frequency correlated cases
numerically.

**Differences.** Global rather than local gain (no gain fluctuation from node to node); annealed theory,
which has no `1/k` fluctuation term by construction. Same family of exponents as Paper A (`z = p + 1`).

**Shortcomings.** Minor errors: the introduction says "the forward transition curve is perfectly fitted
with our semi-analytical findings", while the abstract and Sec. II say the theory predicts the
**backward** transition (only the backward point is computed); the Watts-Strogatz size is given as
`N = 5 x 10^3` in the text and `N = 500` in the Fig. 4 caption; frequencies are "Lorentzian with mean 0 and
standard deviation 1" (a Lorentzian has no standard deviation; the half width is meant). Limitations:
single realisations, agreement judged visually, and the annealed approximation is not tested against the
quenched network at different `<k>`.

---

## (c) What Paper A adds on top, claim by claim

| claim | closest prior work | overlap | what is new |
|---|---|---|---|
| C1. Leading `1/k` coefficient of the fold `lambda_b(k) = lambda_b(inf)(1 + a/k)` for the adaptive model, three treatments | Metz-Perez Castillo 2026 (channels), GECaM 2025 (second-order cavity), Restrepo-Ott-Hunt 2005 Sec. VI (fluctuations raise onset), Ichinomiya 2005 (Gaussian to `O(1/k)`) | The **method** (second-order cavity expansion with drift, coloured noise, response) is known and must be credited | Closed-form coefficients for a fold (not an onset) of a non-pairwise (adaptive) coupling; the observation that with Lorentzian frequencies the single-site process reduces exactly to one Ott-Antonsen ODE driven by the neighbour field (no such reduction found in the sparse cavity papers read; Reddy 2026 recovers OA only without disorder); the spectral filter `mu^2/(mu^2 + (nW)^2)` from the harmonics of drifting neighbours |
| C2. Rician (frozen-scatter) closure excluded in all 60 branch cells | Tarabolo et al. 2025 freeze the noise at fixed points; Kumpeerakij-Restrepo and Restrepo-Ott-Hunt neglect fluctuations | None: no paper tests a frozen-scatter closure for oscillators | A falsification, with the reason (drifting neighbours rotate). Do not attribute the closure to Ichinomiya (B6) |
| C3. Directed graphs: dynamic theory within about 5 to 10% (mean ratio 1.06) | Metz 2025 (exact directed equation) | The directed theory is the `1/k` expansion of Metz's Eq. (7) generalised to a non-pairwise kernel; Paper A must say so | Explicit evaluation and a quantitative test; a population-dynamics solution of Eq. (7) would settle the residual 6% |
| C4. Undirected: dynamic + reaction within 1 to 3% at 2 and 3 times the fold, 5 to 9% short near the fold | Metz-Perez Castillo 2026 (response channel), Aonishi-Okada (ORT), Pruser et al. 2024 (dense response), GECaM | The existence of a response/reaction channel on reciprocal graphs is known | Explicit anisotropic reaction field for the adaptive model and its effect on `R` and `lambda_b` (unlike the isotropic ORT of Aonishi-Okada, which only shifts frequency); first quantitative oscillator test |
| C5. Branch shift linear in mutual-link fraction `rho`, slope approaching the reaction (123% at k = 24, 106% at k = 48) | Metz-Perez Castillo Eq. (194): response strength multiplied by the density of reciprocal partners; Tarabolo et al. tune coupling correlation `gamma` | Linearity is implied structurally by item 2 at leading order; say so | First measurement with topological reciprocity in an oscillator network; convergence of the slope with `k` |
| C6. Bistable window does not close at small `k` (persists to k = 3, widens with N) | None for oscillators. Zhang 2015 Fig. 2(b) hints the width shrinks from `<k> = 12` to 6; Danziger 2016 (hysteresis for any `f`); Tarabolo et al. find sparsity **creates** multistability in Lotka-Volterra | No equivalent result found | New. It refutes a critical connectivity implied by the closure. Caveat: forward threshold is a first passage; robustness runs pending |
| C7. Directed graphs at small `k`: fold below the all-to-all fold | None found | None | New observation; the leading-order theory does not explain it (needs higher order or population dynamics) |
| C8. Clustering raises the fold modestly | Peron-Rodrigues-Kurths 2013 (triangles irrelevant for onset, annealed theory), McGraw-Menzinger 2005 (clustering hinders global sync) | Direction agrees with McGraw-Menzinger | Quantified for a fold. Note Peron et al.'s conclusion comes from a triangle-blind annealed theory, so it does not contradict a `1/k`-order effect |
| C9. Erdos-Renyi degree heterogeneity lowers the fold strongly (coefficient 0.34 vs 1.07 at k = 12, p = 1) | Ichinomiya 2004, Restrepo-Ott-Hunt 2005 (heterogeneous mean field), Metz 2025 (degree fluctuations survive large `c`), Khanra 2020 (annealed) | The mechanism (degree heterogeneity enters through degree moments) is standard; for Poisson degrees the relative variance is `1/k`, the same order as our fluctuation term | Combination of both `1/k` effects in one coefficient; a heterogeneous-mean-field-plus-fluctuation prediction is still to be done (follow-up P4) |

Honest summary. The mechanisms (coloured noise on directed graphs, response/reaction on reciprocal
graphs, Gaussian closure at order `1/k`) are established in the sparse-DMFT and cavity literature, and in
the older ORT literature. Paper A's contribution is (1) their explicit evaluation for a non-pairwise,
adaptive oscillator coupling, at a fold and along the synchronised branch, made tractable by the
Ott-Antonsen reduction of the single-site process; (2) quantitative large-scale tests that exclude the
frozen-scatter picture and resolve the directed and reciprocal parts separately, including a linear-in-`rho`
interpolation; (3) three numerical facts not anticipated by any paper found: no critical connectivity for
bistability, a directed fold below the all-to-all fold at small `k`, and the size of the clustering and
heterogeneity effects at a fold.

---

## (d) Risks to novelty

1. **Metz and Perez Castillo (2026), Sec. 6, and any follow-up.** If they (or Metz alone) publish a
   Kuramoto application with homogeneous couplings and a `1/c` expansion, C1, C3 and C4 would be
   anticipated for the standard model. Their conclusion names "systematise high-connectivity reductions for
   nonlinear pairwise interactions" as the next step. Mitigation: cite both papers as the framework, state
   that our directed term is the `1/k` expansion of Metz's Eq. (7) and our reaction their response channel,
   and claim only the explicit adaptive-model evaluation and tests.
2. **GECaM (Tarabolo and Dall'Asta 2025).** Their code (GaussianExpansionCavityMethod.jl) could be applied
   to Kuramoto by writing the field in `e^{i theta}`; a Kuramoto GECaM paper from that group is a
   plausible near-term competitor. Their framework is the natural name for our second-order expansion and
   must be cited.
3. **Hatchett and Uezu (2008)** and the Coolen-group finite-connectivity oscillator papers remain
   unread (not found online). An explicit `1/c` Kuramoto threshold could be there for the equilibrium
   (identical-frequency or Hamiltonian) case. Must be read before submission.
4. **Restrepo-Ott-Hunt (2005) Sec. VI** already claims that fluctuations raise the (onset) threshold at small
   degree. The **sign** of the undirected fold shift is therefore not new; only its size, the decomposition
   and the fold setting are.
5. **Reciprocity linearity (C5)** is implied by Metz-Perez Castillo Eq. (194) at leading order. Present it
   as a test, not a discovery.
6. **Ott-Antonsen reduction of the driven single-site process.** Not found in the papers read, but it is a
   short argument (Lorentzian frequency independent of the drive); dense-network analogues (Reddy 2026;
   noise-driven OA papers) may contain it. Search "Ott-Antonsen" with "coloured noise" / "fluctuating mean
   field" before claiming it.
7. **Degree heterogeneity (C9)** is standard in direction. Do not claim it as a finding; present it as a
   control that the leading theory must be extended for.
8. Forward-citation lists of Zhang (2015), Zou and Wang (2020), Restrepo-Ott-Hunt (2005) and Ichinomiya
   (2005) were not checked (no Scholar access). A numerical paper measuring hysteresis width against `<k>`
   for the Zhang model (which would touch C6) could exist there.

---

## (e) Suggested citations (BibTeX)

```bibtex
@article{Metz2025DMFTsparse,
  author  = {Metz, Fernando L.},
  title   = {Dynamical Mean-Field Theory of Complex Systems on Sparse Directed Networks},
  journal = {Phys. Rev. Lett.},
  volume  = {134},
  pages   = {037401},
  year    = {2025},
  doi     = {10.1103/PhysRevLett.134.037401},
  eprint  = {2406.06346},
  archivePrefix = {arXiv}
}

@misc{MetzPerezCastillo2026cavity,
  author  = {Metz, Fernando L. and P{\'e}rez Castillo, Isaac},
  title   = {Dynamical cavity method for continuous-time complex systems on sparse random graphs},
  year    = {2026},
  eprint  = {2606.08689},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.dis-nn}
}

@article{TaraboloDallAsta2025GECaM,
  author  = {Tarabolo, Mattia and Dall'Asta, Luca},
  title   = {Gaussian approximation of dynamic cavity equations for linearly-coupled stochastic dynamics},
  journal = {SciPost Phys.},
  volume  = {19},
  pages   = {019},
  year    = {2025},
  doi     = {10.21468/SciPostPhys.19.1.019},
  eprint  = {2406.14200},
  archivePrefix = {arXiv}
}

@misc{Tarabolo2025LV,
  author  = {Tarabolo, Mattia and Dall'Asta, Luca and Mulet, Roberto},
  title   = {Sparse Interactions Reshape Stability in Random {Lotka-Volterra} Dynamics},
  year    = {2025},
  eprint  = {2512.22563},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.dis-nn}
}

@misc{Machado2025LVsparse,
  author  = {Machado, D. and Valigi, P. and Tonolo, T. and Angelini, M. C.},
  title   = {Local equations for the generalized {Lotka-Volterra} model on sparse asymmetric graphs},
  year    = {2025},
  eprint  = {2511.17499},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.PE}
}

@misc{KumpeerakijRestrepo2026,
  author  = {Kumpeerakij, Chanin and Restrepo, Juan G.},
  title   = {Self-consistent analysis of the {Kuramoto} model with higher-order interactions},
  year    = {2026},
  eprint  = {2605.24701},
  archivePrefix = {arXiv},
  primaryClass  = {nlin.AO}
}

@article{RestrepoOttHunt2005,
  author  = {Restrepo, Juan G. and Ott, Edward and Hunt, Brian R.},
  title   = {Onset of synchronization in large networks of coupled oscillators},
  journal = {Phys. Rev. E},
  volume  = {71},
  pages   = {036151},
  year    = {2005},
  doi     = {10.1103/PhysRevE.71.036151},
  eprint  = {cond-mat/0411202},
  archivePrefix = {arXiv}
}

@article{RestrepoOttHunt2006directed,
  author  = {Restrepo, Juan G. and Ott, Edward and Hunt, Brian R.},
  title   = {Synchronization in large directed networks of coupled phase oscillators},
  journal = {Chaos},
  volume  = {16},
  pages   = {015107},
  year    = {2006},
  doi     = {10.1063/1.2148388},
  eprint  = {cond-mat/0509255},
  archivePrefix = {arXiv}
}

@article{Ichinomiya2005pathintegral,
  author  = {Ichinomiya, Takashi},
  title   = {Path-integral approach to the dynamics in sparse random network},
  journal = {Phys. Rev. E},
  volume  = {72},
  pages   = {016109},
  year    = {2005},
  doi     = {10.1103/PhysRevE.72.016109},
  eprint  = {cond-mat/0507285},
  archivePrefix = {arXiv}
}

@article{Ichinomiya2004,
  author  = {Ichinomiya, Takashi},
  title   = {Frequency synchronization in a random oscillator network},
  journal = {Phys. Rev. E},
  volume  = {70},
  pages   = {026116},
  year    = {2004},
  doi     = {10.1103/PhysRevE.70.026116}
}

@misc{AonishiOkada2001ORT,
  author  = {Aonishi, Toru and Okada, Masato},
  title   = {Acceleration effect caused by the {Onsager} reaction term in a frustrated coupled oscillator system},
  year    = {2001},
  eprint  = {cond-mat/9909048},
  archivePrefix = {arXiv},
  note    = {Journal version not verified}
}

@article{AonishiKurataOkada2002,
  author  = {Aonishi, Toru and Kurata, Koji and Okada, Masato},
  title   = {Acceleration effect of coupled oscillator systems},
  journal = {Phys. Rev. E},
  volume  = {65},
  pages   = {046223},
  year    = {2002},
  doi     = {10.1103/PhysRevE.65.046223},
  eprint  = {cond-mat/0201453},
  archivePrefix = {arXiv}
}

@article{Pruser2024volcano,
  author  = {Pr{\"u}ser, Axel and Rosmej, Sebastian and Engel, Andreas},
  title   = {Nature of the Volcano Transition in the Fully Disordered {Kuramoto} Model},
  journal = {Phys. Rev. Lett.},
  volume  = {132},
  pages   = {187201},
  year    = {2024},
  doi     = {10.1103/PhysRevLett.132.187201},
  eprint  = {2310.09079},
  archivePrefix = {arXiv}
}

@article{Skantzos2005cavityXY,
  author  = {Skantzos, Nikos S. and P{\'e}rez Castillo, Isaac and Hatchett, Jonathan P. L.},
  title   = {Cavity approach for real variables on diluted graphs and application to synchronization in small-world lattices},
  journal = {Phys. Rev. E},
  volume  = {72},
  pages   = {066127},
  year    = {2005},
  doi     = {10.1103/PhysRevE.72.066127},
  eprint  = {cond-mat/0508609},
  archivePrefix = {arXiv}
}

@article{Zhang2015adaptive,
  author  = {Zhang, Xiyun and Boccaletti, Stefano and Guan, Shuguang and Liu, Zonghua},
  title   = {Explosive Synchronization in Adaptive and Multilayer Networks},
  journal = {Phys. Rev. Lett.},
  volume  = {114},
  pages   = {038701},
  year    = {2015},
  doi     = {10.1103/PhysRevLett.114.038701},
  eprint  = {1410.2986},
  archivePrefix = {arXiv}
}

@article{ZouWang2020,
  author  = {Zou, Wei and Wang, Jianwei},
  title   = {Dynamics of the generalized {Kuramoto} model with nonlinear coupling: Bifurcation and stability},
  journal = {Phys. Rev. E},
  volume  = {102},
  pages   = {012219},
  year    = {2020},
  doi     = {10.1103/PhysRevE.102.012219}
}

@article{Danziger2016coexists,
  author  = {Danziger, Michael M. and Moskalenko, Olga I. and Kurkin, Semen A. and Zhang, Xiyun and Havlin, Shlomo and Boccaletti, Stefano},
  title   = {Explosive synchronization coexists with classical synchronization in the {Kuramoto} model},
  journal = {Chaos},
  volume  = {26},
  pages   = {065307},
  year    = {2016},
  note    = {DOI to be verified (believed 10.1063/1.4953345)}
}

@article{Khanra2020amplification,
  author  = {Khanra, Pitambar and Kundu, Prosenjit and Pal, Pinaki and Ji, Peng and Hens, Chittaranjan},
  title   = {Amplification of explosive width in complex networks},
  journal = {Chaos},
  volume  = {30},
  pages   = {031101},
  year    = {2020},
  doi     = {10.1063/5.0003410},
  eprint  = {2104.06862},
  archivePrefix = {arXiv}
}

@misc{DuttaPalHens2025adaptation,
  author  = {Dutta, Sangita and Pal, Pinaki and Hens, Chittaranjan},
  title   = {Generalized Adaptation-Induced Non-universal Synchronization Transitions in Random Hypergraphs},
  year    = {2025},
  eprint  = {2507.18421},
  archivePrefix = {arXiv},
  note    = {Author list taken from docs/PRIOR_ART.md; verify}
}

@article{Peron2013clustered,
  author  = {Peron, Thomas K. DM. and Rodrigues, Francisco A. and Kurths, J{\"u}rgen},
  title   = {Synchronization in clustered random networks},
  journal = {Phys. Rev. E},
  volume  = {87},
  pages   = {032807},
  year    = {2013},
  doi     = {10.1103/PhysRevE.87.032807},
  eprint  = {1210.2625},
  archivePrefix = {arXiv}
}

@article{McGrawMenzinger2005,
  author  = {McGraw, Patrick N. and Menzinger, Michael},
  title   = {Clustering and the synchronization of oscillator networks},
  journal = {Phys. Rev. E},
  volume  = {72},
  pages   = {015101},
  year    = {2005},
  doi     = {10.1103/PhysRevE.72.015101}
}

@article{Juhasz2019sparse,
  author  = {Juh{\'a}sz, R{\'o}bert and Kelling, Jeffrey and {\'O}dor, G{\'e}za},
  title   = {Critical dynamics of the {Kuramoto} model on sparse random networks},
  journal = {J. Stat. Mech.},
  pages   = {053403},
  year    = {2019},
  doi     = {10.1088/1742-5468/ab16c3},
  eprint  = {1902.10422},
  archivePrefix = {arXiv}
}

@article{UmHongPark2024annealed,
  author  = {Um, Jaegon and Hong, Hyunsuk and Park, Hyunggyu},
  title   = {Validity of annealed approximation in a high-dimensional system},
  journal = {Sci. Rep.},
  volume  = {14},
  pages   = {6816},
  year    = {2024},
  doi     = {10.1038/s41598-024-57323-4},
  eprint  = {2403.14984},
  archivePrefix = {arXiv}
}

@misc{Gast2026weightvariance,
  author  = {Gast, Richard and Takasu, Shotaro and Kurths, Juergen and Kennedy, Ann},
  title   = {Deviations from global coupling in adaptive oscillator networks: a mean-field theory for the variance of coupling weights},
  year    = {2026},
  eprint  = {2609.22597},
  archivePrefix = {arXiv},
  primaryClass  = {nlin.AO}
}

@article{Reddy2026compactDMFT,
  author  = {Reddy, Kanishka},
  title   = {Compact Dynamical Mean-Field Theory of Oscillator Networks},
  journal = {Phys. Rev. E},
  volume  = {113},
  pages   = {034222},
  year    = {2026},
  eprint  = {2603.09402},
  archivePrefix = {arXiv}
}

@misc{Tonolo2025LVrrg,
  author  = {Tonolo, T. and Angelini, M. C. and Azaele, S. and Maritan, A. and Gradenigo, G.},
  title   = {Generalized {Lotka-Volterra} model with sparse interactions: non-{Gaussian} effects and topological multiple-equilibria phase},
  year    = {2025},
  eprint  = {2503.20887},
  archivePrefix = {arXiv},
  note    = {Published in PRX Life (2026); title as given by the arXiv listing, verify}
}

@article{HatchettUezu2008,
  author  = {Hatchett, Jonathan P. L. and Uezu, Tatsuya},
  journal = {Phys. Rev. E},
  volume  = {78},
  pages   = {036106},
  year    = {2008},
  note    = {Title not verified; unread}
}

@article{Rodrigues2016review,
  author  = {Rodrigues, Francisco A. and Peron, Thomas K. DM. and Ji, Peng and Kurths, J{\"u}rgen},
  title   = {The {Kuramoto} model in complex networks},
  journal = {Phys. Rep.},
  volume  = {610},
  pages   = {1--98},
  year    = {2016},
  doi     = {10.1016/j.physrep.2015.10.008},
  eprint  = {1511.07139},
  archivePrefix = {arXiv}
}
```

Entries marked "verify" have a field taken from a search snippet or from PRIOR_ART.md rather than from
the paper itself.

---

## (f) Follow-up papers this literature makes possible

**P1. "Sparse dynamical mean-field theory of explosive synchronisation: the adaptive Kuramoto fold at
any degree."** Scope: solve Metz's directed path equation (Eq. 7 of Metz 2025, generalised to the
non-pairwise gain) by population dynamics, using the Ott-Antonsen reduction so each population member is
one complex ODE driven by `k` sampled neighbour trajectories instead of `k` full oscillator populations. This
gives the directed fold at every `k` including `k = 3`, tests whether the 6% residual of C3 is the `1/k^2`
term, and should explain the directed fold below the all-to-all fold (C7). Needs: a population-dynamics
code (GPU friendly, same kernels as `gpu_runner`), convergence checks in population size and time window,
comparison with the existing directed GPU folds; no new network simulation needed.

**P2. "Onsager reaction and reciprocity in sparse oscillator networks."** Scope: the same expansion for
the standard Kuramoto (`p = 0`) and Sakaguchi models: onset, branch shift and (for Sakaguchi) frequency
shift as functions of `k` and the mutual fraction `rho`. With a phase lag the reaction should also produce an
Aonishi-Okada-type frequency shift, giving a second observable. Connects Aonishi-Okada (dense, static ORT)
to Metz-Perez Castillo (sparse response channel) with explicit formulas. Needs: reaction integrals for
`p = 0` and with phase lag; reciprocity runs like `paperA_reciprocity_gpu` for `p = 0` and two lags;
directed/undirected onset measurements (first-passage protocol as in the window runs).

**P3. "No critical connectivity for explosive synchronisation."** Scope: the bistable window down to
`k = 3`, its growth with `N`, and the mechanism of the forward transition on sparse graphs. All-to-all, the
incoherent state is neutrally stable for all couplings (Zou and Wang 2020), so on sparse graphs the forward
jump must be fluctuation driven (nucleation on locally coherent clusters), which explains the dependence on
`N` and waiting time. Contrast with Lotka-Volterra, where sparsity creates multistability (Tarabolo et al.
2025). Needs: the pending window robustness runs (N = 3000 to 30000, waiting 1600) plus larger `N` and
waiting-time scans on Ada, nucleation statistics (first-passage time distributions against `N` and
coupling), a theory of the forward threshold (linear stability of incoherence on a `k`-regular graph via
the Kesten-McKay spectrum, plus a rare-cluster argument).

**P4. "Heterogeneous and clustered sparse corrections to explosive thresholds."** Scope: extend the `1/k`
theory to degree classes (heterogeneous mean field for the drift, per-class fluctuation and reaction
terms) and predict the Erdos-Renyi coefficient (0.34 against 1.07 at `k = 12`); treat triangles with the
Newman-Miller clustered configuration model (reaction through a triangle closes in two steps), to explain
why clustering raises the fold (C8) although the annealed theory of Peron et al. is blind to it. Needs:
derivation of the class-resolved expansion; scale-free and configuration-model runs with prescribed degree
variance at fixed mean (separating the degree-moment effect from fluctuations); clustered runs at more
triangle counts and `k` (the structure run had two).

**P5. "Local against global adaptation on sparse graphs."** Scope: Khanra et al. (2020) and Dutta et al.
(2025) use the global order parameter in the gain; Zhang et al. (2015) and Paper A use the local one. On
sparse graphs the global gain removes the gain fluctuation, so the `1/k` correction comes only from the
field scatter. Comparing the two isolates the contribution of gain fluctuations (the `p`-dependent part of
the coefficient). Needs: the global-gain version of `branch_theory.py` (a few lines), GPU runs at `k = 12`
to 48, directed and undirected, `p = 1, 2`; a short paper.

**P6. "Ott-Antonsen reduction for sparse-graph DMFT of oscillators."** Scope: a methods note: with
Lorentzian frequencies independent of the drive, the single-site process of any sparse cavity theory reduces
to one complex ODE driven by the neighbour field, for Kuramoto, Sakaguchi, Winfree, theta neurons and
higher-order (triadic) coupling. Applied to the dyadic-plus-triadic model of Kumpeerakij and Restrepo (2026)
it would extend their bistability onset to sparse hypergraphs, where their assumption 1 fails. Needs: a
check that the reduction is not already in the noise-driven Ott-Antonsen literature (risk 6), a GECaM-style
second-order implementation, and sparse-hypergraph simulations (the oscillator zoo kernels can be reused).
