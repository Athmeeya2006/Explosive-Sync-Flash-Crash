# Decisions

Recorded as they are made. Newest last.

1. **No commits.** Nothing is committed without a direct instruction. Nine local commits and the tag
   `prereg-v1` made earlier on a branch `publication-refactor` were removed on 2026-10-13; every file
   was kept as an uncommitted change on `main`. Nothing was ever pushed. Run directories created
   before the removal carry hashes of those deleted commits in their names.
2. **No manuscripts for now** (decided 2026-10-13). `paper_A/`, `paper_B/`, LaTeX,
   `numbers.tex` and the Phase 6 work are on hold until the Paper A pilot says what is true.
3. **Phase 0 added files but changed no simulation code.** `research/model.py`, `theory.py` and every
   `s*.py` and `e*.py` script are byte-identical to `fc98f02`. Additions: `research/p0_spot_checks.py`,
   its output `research/data/p0_spot_checks.json`, and `docs/`.
4. **Pages updated.** `research/build_index.py` gained a generated audit
   section (P0) and now also writes `docs/CLAIMS_AUDIT.md`; `research/reference.html` gained step 26;
   `simulation/index.html` gained block 0. Older text was kept and marked rather than deleted, so the
   record stays auditable.
5. **Numbers on the audit section are computed at build time** from `p0_spot_checks.json` and the
   committed CSVs. The audit's own table is stored as `AUDIT_REFERENCE` in `build_index.py` because it
   is the reference being reproduced, not a result of this repository. `reference.html` and
   `simulation/index.html` are static pages, so their new sections quote no measured values and link to
   P0 instead.
6. **S14 verdict changed from "new result" to "unverified, under audit"** on the results page. The
   older hand-typed numbers elsewhere in `build_index.py` are still there; replacing them is Phase 2.
7. **Cloud run files moved into the repository layout** (2026-10-14). `raw/*.jsonl` and
   `branch_summary.csv` went to `results/paperA_cloud/20261013-200800_d49a0da/`, the four scripts from
   `scripts/` to `experiments/`, and `RESULTS.md` to `docs/CLOUD_RUN_2026-10-13.md`. File contents are
   unchanged except for paths in the note. A `config.yaml` was written afterwards from the note,
   because the runs were launched by hand.
8. **Branch predictions are stored as reported, not regenerated.** The values printed by the theory
   scripts on the cloud machine are in `branch_theory_reported.json` and are labelled as such wherever
   they appear. They become a normal result once the scripts are rerun locally.
9. **Two-seed cells are shown with Student-t intervals.** The run note quoted standard errors from two
   seeds. The pages show both, and no two-seed cell is described as a confirmation.
10. **The results page opens with a generated status card.** The older hand-written summary is kept
    below it, retitled as an earlier summary.
11. **Framing after the external novelty audit** (2026-10-14). The physics paper is a regular article
    on explicit leading-order sparse-graph corrections for adaptive Kuramoto dynamics, with a
    directed/undirected decomposition verified by simulation. It is not presented as a new cavity
    method, and the directed/undirected distinction is not claimed as new. The Rician closure is
    presented as a falsified static-scatter baseline. See `docs/PRIOR_ART.md`.
12. **The finance paper stays separate** from the physics paper. Its one positive number comes from a
    robustness variant after the primary test failed and is reported as exploratory.
13. **The planned confirmatory test on a disjoint set of stocks is dropped.** Crash events are
    market-wide, so different stocks over the same period test the same events. A confirmatory test
    needs a later time period, with the event definition, horizon and volatility matching fixed in
    advance.
14. **Branch theory moved into the package.** `esfc.branch` computes the branch shift for all three
    treatments; `cavity` was refactored so its integrals accept the constants of any branch point.
    The fold results are unchanged (`tests/test_cavity.py`). The scripts in `experiments/` remain as the
    record of the cloud run.
15. **Mutual links in very large graphs are paired with numpy** (above 4 million stubs) because
    networkx is too slow there. Smaller graphs keep the networkx generator.

