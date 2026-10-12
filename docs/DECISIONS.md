# Decisions

Recorded as they are made. Newest last.

1. **No commits.** The author asked that nothing be committed. Nine local commits and the tag
   `prereg-v1` made earlier on a branch `publication-refactor` were removed on 2026-10-13; every file
   was kept as an uncommitted change on `main`. Nothing was ever pushed. Run directories created
   before the removal carry hashes of those deleted commits in their names.
2. **No manuscripts for now** (author's instruction, 2026-10-13). `paper_A/`, `paper_B/`, LaTeX,
   `numbers.tex` and the Phase 6 work are on hold until the Paper A pilot says what is true.
3. **Phase 0 added files but changed no simulation code.** `research/model.py`, `theory.py` and every
   `s*.py` and `e*.py` script are byte-identical to `fc98f02`. Additions: `research/p0_spot_checks.py`,
   its output `research/data/p0_spot_checks.json`, and `docs/`.
4. **Pages updated at the author's request.** `research/build_index.py` gained a generated audit
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
