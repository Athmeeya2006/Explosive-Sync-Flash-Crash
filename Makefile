# Pipeline with explicit file dependencies. Results live in results/<ID>/<stamp>_<hash>/.
PY ?= research/.venv/bin/python
ESFC ?= research/.venv/bin/esfc
R := research/data

.PHONY: test test-fast lint audit pilot a0 legacy-test s-chain

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy src/esfc

audit:            ## regenerate research/index.html, docs/CLAIMS_AUDIT.md and the README tables
	$(ESFC) audit

a0:
	$(ESFC) run configs/paperA_A0_meanfield.yaml
pilot:
	$(ESFC) run configs/paperA_A1_pilot.yaml

# The S chain: S3, S4, S5, S7 and S9 read the window that S2 measures. *.npz is not tracked, so on a
# fresh clone S2 must run first. This rule encodes what used to be a hidden run-order dependency.
$(R)/s2_herding.npz: research/s2_herding_explosive.py research/model.py
	$(PY) research/s2_herding_explosive.py
s-chain: $(R)/s2_herding.npz
	for s in s3_crash_anatomy s4_crash_phase_diagram s5_circuit_breakers s7_rate_tipping s9_observable_mapping; do $(PY) research/$$s.py || exit 1; done

legacy-test:
	cd legacy && $(MAKE) test
