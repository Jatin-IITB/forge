# Forge — one command rebuilds the asset (DESIGN §4.6). Targets land
# incrementally as each phase is built.

CONTRACT ?= contracts/pii_redaction_v1.yaml
GOLD     ?= data/gold/test.jsonl
PREDS    ?= predictions.jsonl
TRAIN    ?= data/train.jsonl
SEEDS    ?= data/gold/dev.jsonl
MODEL    ?= Qwen/Qwen2.5-32B-Instruct
PYTHON   ?= python

.PHONY: help install validate gold gold-sample infer data-engine eval test lint forge

help:
	@echo "Forge targets:"
	@echo "  make install       # editable install + dev deps"
	@echo "  make validate      # validate the TaskContract + sample gold against the schema"
	@echo "  make gold          # (re)build the gold dev/test sets from fixed seed (Faker)"
	@echo "  make gold-sample   # (re)build the illustrative data/gold/sample.jsonl"
	@echo "  make eval          # score PREDS against GOLD, check contract gates"
	@echo "  make test          # run unit tests"
	@echo "  make lint          # ruff"
	@echo ""
	@echo "  make forge         # END-TO-END rebuild — NOT YET IMPLEMENTED (Phases 2-5)"
	@echo "  CONTRACT=$(CONTRACT)  GOLD=$(GOLD)  PREDS=$(PREDS)"

install:
	$(PYTHON) -m pip install -e ".[dev,data]"

validate:
	$(PYTHON) scripts/validate_contract.py $(CONTRACT) --gold data/gold/sample.jsonl

gold:
	$(PYTHON) scripts/build_gold.py

gold-sample:
	$(PYTHON) scripts/make_sample_gold.py

infer:
	$(PYTHON) scripts/run_inference.py $(GOLD) $(PREDS) --model $(MODEL)

data-engine:
	$(PYTHON) scripts/run_data_engine.py --seed-texts $(SEEDS) --gold $(GOLD) --output $(TRAIN) --model $(MODEL)

eval:
	$(PYTHON) scripts/run_eval.py $(GOLD) $(PREDS) --check-gates --contract $(CONTRACT)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check forge scripts tests

# The headline reproducibility target. Guard-railed until the pipeline exists so a
# reader can't mistake Phase 0 for a finished system.
forge:
	@echo "make forge is not fully implemented yet — Phases 2-5 pending."
	@echo "Pipeline (data -> train -> eval -> quantize -> serve) is built incrementally."
	@exit 1
