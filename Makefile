# Forge — one command rebuilds the asset (DESIGN §4.6). Phase 0 targets only;
# the train/eval/quantize/serve targets land as those phases are built.

CONTRACT ?= contracts/pii_redaction_v1.yaml
PYTHON   ?= python

.PHONY: help install validate gold-sample test lint forge

help:
	@echo "Forge targets (Phase 0):"
	@echo "  make install       # editable install + dev deps"
	@echo "  make validate      # validate the TaskContract + sample gold against the schema"
	@echo "  make gold-sample   # (re)build the illustrative data/gold/sample.jsonl"
	@echo "  make test          # run unit tests"
	@echo "  make lint          # ruff"
	@echo ""
	@echo "  make forge         # END-TO-END rebuild — NOT YET IMPLEMENTED (Phases 1-5)"
	@echo "  CONTRACT=$(CONTRACT)"

install:
	$(PYTHON) -m pip install -e ".[dev,data]"

validate:
	$(PYTHON) scripts/validate_contract.py $(CONTRACT) --gold data/gold/sample.jsonl

gold-sample:
	$(PYTHON) scripts/make_sample_gold.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check forge scripts tests

# The headline reproducibility target. Guard-railed until the pipeline exists so a
# reader can't mistake Phase 0 for a finished system.
forge:
	@echo "make forge is not implemented yet — Phase 0 ships the contract + eval schema only."
	@echo "Pipeline (data -> train -> eval -> quantize -> serve) is built in Phases 1-5."
	@exit 1
