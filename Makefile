# Forge — one command rebuilds the asset (DESIGN §4.6).
#
# `make forge` is the headline reproducibility claim: a stranger clones the
# repo, sets one API key, and rebuilds the specialist end to end.

CONTRACT ?= contracts/pii_redaction_v2.yaml
GOLD     ?= data/gold/test.jsonl
DEV      ?= data/gold/dev.jsonl
PREDS    ?= data/predictions_student.jsonl
TEACHER_PREDS ?= data/predictions_teacher.jsonl
TRAIN    ?= data/train_v2.jsonl
SEEDS    ?= data/gold/dev.jsonl
RUN      ?= run_002

# Teacher: open-weight GPT-OSS-120B via a hosted endpoint (ADR 0010).
# The provider is fungible — point TEACHER_URL at Groq, or a local vLLM, and
# nothing else changes.
MODEL      ?= gpt-oss-120b
TEACHER_URL ?= https://api.cerebras.ai/v1
TEACHER_KEY_ENV ?= CEREBRAS_API_KEY
TEACHER_RPM ?= 5

BASE     ?= Qwen/Qwen2.5-1.5B-Instruct
CKPT     ?= checkpoints/$(RUN)
MERGED   ?= models/pii-1.5b-merged
GGUF     ?= models/pii-1.5b-gguf
# Prefer the project venv, then python3. A bare `python` does not exist on
# macOS or most modern Linux distros, so defaulting to it made every target
# fail with "No such file or directory" on a fresh clone — including the
# `make forge` reproducibility claim.
PYTHON   ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3 || echo python)

.PHONY: help install validate gold gold-sample infer teacher-baseline data-engine \
        train eval economics error-analysis merge gguf report test lint forge clean-preds

help:
	@echo "Forge targets:"
	@echo "  make install          # editable install + dev deps"
	@echo "  make validate         # validate the TaskContract + sample gold against the schema"
	@echo "  make gold             # (re)build the frozen gold dev/test sets (Faker, seed 42)"
	@echo "  make teacher-baseline # score the teacher on the frozen test set (the parity bar)"
	@echo "  make data-engine      # generate verification-gated training data from the teacher"
	@echo "  make train            # LoRA SFT on verified training data"
	@echo "  make infer            # run the student over the test set"
	@echo "  make eval             # score PREDS against GOLD, check contract gates"
	@echo "  make error-analysis   # cluster student failures into augmentation targets"
	@echo "  make economics        # measure G3 (cost) + G4 (latency)"
	@echo "  make merge            # fold the LoRA adapter into the base model"
	@echo "  make gguf             # quantize the merged model for offline use"
	@echo "  make report           # regenerate the technical report PDF from measured artifacts"
	@echo "  make audit            # mechanical checks on the frozen gold bytes (offsets, dupes, leakage)"
	@echo "  make test / lint"
	@echo ""
	@echo "  make forge            # END-TO-END rebuild (teacher bar -> data -> train -> gates)"
	@echo ""
	@echo "  CONTRACT=$(CONTRACT)"
	@echo "  TEACHER=$(MODEL) via $(TEACHER_URL)  (key from \$$$(TEACHER_KEY_ENV))"
	@echo "  BASE=$(BASE)  RUN=$(RUN)"

install:
	$(PYTHON) -m pip install -e ".[dev,data]"

validate:
	$(PYTHON) scripts/validate_contract.py $(CONTRACT) --gold data/gold/sample.jsonl

gold:
	$(PYTHON) scripts/build_gold.py

gold-sample:
	$(PYTHON) scripts/make_sample_gold.py

# The parity denominator. Must exist before any student claim means anything.
teacher-baseline:
	$(PYTHON) scripts/run_inference.py $(GOLD) $(TEACHER_PREDS) \
		--model $(MODEL) --base-url $(TEACHER_URL) \
		--api-key-env $(TEACHER_KEY_ENV) --rpm $(TEACHER_RPM) \
		--reasoning-effort low --max-tokens 2048 --resume

data-engine:
	$(PYTHON) scripts/run_data_engine.py --seed-texts $(SEEDS) --gold $(GOLD) \
		--output $(TRAIN) --model $(MODEL) --base-url $(TEACHER_URL) \
		--api-key-env $(TEACHER_KEY_ENV) --rpm $(TEACHER_RPM) \
		--reasoning-effort low --resume

train:
	$(PYTHON) scripts/run_train.py --train-data $(TRAIN) --base-model $(BASE) \
		--output-dir $(CKPT) --save-steps 10 --resume

infer:
	$(PYTHON) scripts/run_inference.py $(GOLD) $(PREDS) \
		--model $(BASE) --adapter $(CKPT)/final --resume

eval:
	$(PYTHON) scripts/run_eval.py $(GOLD) $(PREDS) --check-gates --contract $(CONTRACT)

error-analysis:
	$(PYTHON) scripts/error_analysis.py $(GOLD) $(PREDS) --train-data $(TRAIN) \
		--output data/error_analysis_$(RUN).json

economics:
	$(PYTHON) scripts/run_economics.py \
		--teacher-meta $(TEACHER_PREDS:.jsonl=.meta.json) \
		--student-meta $(PREDS:.jsonl=.meta.json) \
		--contract $(CONTRACT) --output reports/economics.md

merge:
	$(PYTHON) scripts/export_model.py merge --base $(BASE) \
		--adapter $(CKPT)/final --output $(MERGED)

gguf: merge
	$(PYTHON) scripts/export_model.py gguf --merged $(MERGED) --output $(GGUF)

report:
	$(PYTHON) scripts/build_report.py

test:
	$(PYTHON) -m pytest -q

# Mechanical checks on the committed gold bytes: offset exactness, span
# disjointness, duplicates, and train/gold leakage. Distinct from `test`,
# which can pass while the data itself is wrong — the ADR 0011 clock defect
# survived a green suite because every test regenerated the data identically.
# NOT human verification (PROTOCOL.md section 5), and must not be called that.
audit:
	$(PYTHON) scripts/audit_gold.py

lint:
	$(PYTHON) -m ruff check forge scripts tests

clean-preds:
	rm -f $(PREDS) $(PREDS:.jsonl=.meta.json)

# ---------------------------------------------------------------------------
# The headline reproducibility target.
#
# Ordering is deliberate: the teacher bar is measured BEFORE the student is
# trained, so the parity threshold cannot be back-fitted to a result. Every
# stage resumes, because these steps take hours and machines sleep.
# ---------------------------------------------------------------------------
forge:
	@if [ -z "$${$(TEACHER_KEY_ENV)}" ]; then \
		echo "ERROR: \$$$(TEACHER_KEY_ENV) is not set."; \
		echo "Get a free key at https://cloud.cerebras.ai and export it:"; \
		echo "    export $(TEACHER_KEY_ENV)=..."; \
		exit 1; \
	fi
	$(MAKE) validate
	$(MAKE) gold
	$(MAKE) teacher-baseline
	$(MAKE) data-engine
	$(MAKE) train
	$(MAKE) infer
	$(MAKE) eval
	$(MAKE) economics
	@echo ""
	@echo "forge complete. Artifacts:"
	@echo "  student adapter : $(CKPT)/final"
	@echo "  gate report     : run 'make eval' output above"
	@echo "  economics       : reports/economics.md"
	@echo "Package for offline use with: make gguf"
