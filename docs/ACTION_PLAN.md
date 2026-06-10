# Forge — Action Plan

A phased plan with **exit gates** (a phase is not "done" until its gate passes), deliverables, the compute/cost budget, and the risk register. Pacing assumes one strong engineer, evenings/weekends; calendar estimates are ranges, not commitments.

> **Rule inherited from Aroha:** every phase ends in a *checkable artifact*, not a vibe. No phase is "done" because it feels done.

---

## Phase 0 — Task contract + frozen gold eval  ⟶ *the most important phase*
**Goal:** decide exactly what we're building and how we'll know it works, before any modeling.

**Steps**
1. **Pick the flagship task.** Selection criteria: (a) narrow & well-defined, (b) high-volume / expensive at frontier prices, (c) a *checkable* metric, (d) privacy/meaning-relevant, (e) teacher ToS + base-model license permit distillation, **(f) uses fully PUBLIC data + open weights — nothing internal/internship-bound (`adr/0003`).**
   - Candidate tasks (all public-data; pick one in this phase):
     - **A. On-device PII detection / redaction** — *Recommended.* Entity/span tagging on text. Meaning is airtight (you can't send PII to an API to find PII → local model is the *only* compliant option; GDPR / India DPDP Act). Public corpora exist (open PII-masking datasets on Hugging Face). Crisp metric (entity P/R/F1). Visceral offline demo. See DESIGN §0.2.
     - **B. Structured field extraction from public documents** — free text → JSON schema (e.g. public invoices/receipts/resumes/filings). Crisp metric (field-F1 + schema-validity); broad utility.
     - **C. Small specialist for an under-served Indian language** — classification/extraction/translation on public Indic corpora (AI4Bharat/IndicNLP/FLORES). Meaning = accessibility/democratization; runs on a phone. Harder eval (generation), so secondary.
2. **Write `TaskContract`** (DESIGN §1) and commit it. Immutable for a run.
3. **Build the gold set:** 300–1,000 human-verified examples drawn/curated from **public** sources, de-duplicated, **leakage-checked** against any synthetic source. Split: dev / test (test stays frozen & untouched until Phase 4 gates).
4. **Choose teacher + base model — both open/independent.** Teacher = an **open-weight** model (e.g. Llama-3.x / Qwen2.5 / DeepSeek) or a public API whose terms permit distillation — **never an internal/company-hosted model you'll lose access to.** Base student = e.g. Qwen2.5-1.5B/3B or Llama-3.2-1B/3B (permissive license). Run the `adr/0003` litmus test before exiting.

**Exit gate:** committed `TaskContract`; a frozen `test.jsonl` (public-sourced) with a documented labeling protocol + inter-annotator spot-check; written teacher/base-model license clearance; **and the `adr/0003` independence litmus test passes.** No modeling has happened yet — correct.

**Deliverables:** `contracts/<task>.yaml`, `data/gold/{dev,test}.jsonl`, `docs/adr/0001`, `docs/adr/0003`, `docs/adr/0005-task-selection.md`.

---

## Phase 1 — Measure the bar
**Goal:** know the number to beat and the gap to close.

**Steps**
1. Score the **teacher** on the gold test set → `teacher_score`, `teacher_cost/1k`, `teacher_p95`. This is the bar.
2. Score **off-the-shelf small models** zero-/few-shot → the starting gap.
3. Build the **eval harness** (one command: model → metric + CI + cost + latency). Reused unchanged in Phase 4.

**Exit gate:** a baseline table — teacher vs. raw small model — with metric, 95% CI, $/1k, p95. The parity target (e.g. `0.98 × teacher_score`) is now a concrete number.

**Deliverables:** `eval/` harness, `reports/baseline.md`.

---

## Phase 2 — Data engine (verification-gated)
**Goal:** produce a *trustworthy* training set, cheaply.

**Steps**
1. Teacher generates labeled examples **with rationales**; diversity sampling across the input space.
2. **Verification gate:** self-consistency (k-sample agreement) + schema/constraint validity + (where cheap) re-derivation; drop everything that fails. Log accept/reject rates.
3. Dedup (near-dup detection) against itself *and* the gold set (no leakage).
4. Start at a modest budget (e.g. 3–10k verified examples); the error loop (Phase 4) will grow only the useful parts.

**Exit gate:** `data/train.jsonl` with a data card: size, accept-rate, dedup stats, leakage check = 0, per-slice coverage.

**Deliverables:** `data_engine/`, `data/train.jsonl`, `reports/data_card.md`.

---

## Phase 3 — Train
**Goal:** a candidate student.

**Steps**
1. **SFT** with LoRA/QLoRA on the verified set (rationale-augmented targets).
2. Set up **constrained decoding** for structured outputs (grammar/schema).
3. (Conditional) **DPO** on pairs: teacher-correct vs. student-failed, if SFT alone misses the gate.
4. Track training curves; seed everything.

**Exit gate:** a saved adapter/checkpoint + a one-command inference path producing schema-valid output on dev.

**Deliverables:** `train/`, `checkpoints/<run_id>/`, `reports/train_<run_id>.md`.

---

## Phase 4 — Evaluate + mine errors (the loop)
**Goal:** pass the parity gate, or learn precisely why not.

**Steps**
1. Run the **Phase-1 harness unchanged** on the frozen test set → student score + CI + cost + latency.
2. If gate **met** → go to Phase 5.
3. If **not met** → cluster failures (by slice/error type), write an error report, and hand the clusters to the **Data Engine (Phase 2)** for targeted generation. Retrain (Phase 3). Repeat.
4. Track each loop iteration's cost so the active-learning ROI is visible.

**Exit gate (the headline gate):** all six gates in `SUCCESS.md` pass on the frozen test set, with CIs.

**Deliverables:** `reports/eval_<run_id>.md`, `reports/error_clusters_<iter>.md`, the loop log.

---

## Phase 5 — Harden + serve
**Goal:** turn the checkpoint into a deployable, honest asset.

**Steps**
1. **Quantize** (AWQ/GGUF); confirm the quantized model still passes the gates (re-run harness).
2. **Serve** via vLLM (GPU) and/or llama.cpp (CPU/laptop); measure *real* $/1k and p95 under load.
3. **Robustness:** OOD probe set, adversarial inputs, input-domain guard + refusal behavior; measure the SAFETY gate.
4. **Model card** (intended use, training data provenance, metrics, limitations, license) + **one-command rebuild** (`make forge`).
5. **Drift-monitor hook** (log input-distribution stats in prod) — stub, not full MLOps.

**Exit gate:** quantized-and-served model passing all gates; `make forge TASK=<contract>` rebuilds the asset end-to-end on a clean machine.

**Deliverables:** `serve/`, `models/<task>/` (+ GGUF), `MODEL_CARD.md`, working `Makefile`.

---

## Phase 6 — Package for impact (portfolio + maybe OSS)
**Goal:** make the result legible and undeniable.

**Steps**
1. **Benchmark table** + a 30-second **demo** (the contrast shot: "GPT-4-class quality, 1% cost, on a laptop").
2. **Writeup** in Aroha's doc style: this `DESIGN.md`, an *honest assessment*, the field comparison, the novelty calibration.
3. (Optional) Open-source the *pipeline* (not just the model) — the contract-driven `make forge` is the reusable artifact others can run on their own task.

**Exit gate:** a stranger can read the writeup, run `make forge`, and reproduce a passing model.

**Deliverables:** `README.md`, `docs/HONEST_ASSESSMENT.md`, demo asset, (optional) public repo.

---

## Compute & cost budget (order-of-magnitude)
| Item | Estimate | Notes |
|---|---|---|
| Teacher tokens — data gen + verification (cold start) | **the dominant cost** | self-consistency multiplies it; error loop concentrates spend later |
| Gold-set labeling | human time, not $$ | the highest-leverage spend |
| Training compute (LoRA/QLoRA, 1–3B) | single 24GB GPU; hours/run | cloud spot or local; cheap relative to data gen |
| Serving benchmark | minutes on 1 GPU + 1 laptop | for $/1k & p95 numbers |
| **Risk control** | set a hard teacher-token budget per loop iteration; the active loop must *justify* each round by closing a measured gap |

---

## Risk register (severity-tagged, Aroha-style)
- **[CRITICAL] Teacher ToS / license** forbids distillation → blocks publishing. *Mitigation:* settle in Phase 0; prefer open-weight teacher + permissive base.
- **[CRITICAL] Leaky/weak gold set** → parity claims are meaningless. *Mitigation:* human curation + leakage check + frozen test.
- **[HIGH] Training on the teacher's confident mistakes** → silent quality cap. *Mitigation:* verification gate (mandatory).
- **[HIGH] Cold-start teacher-token blow-up.** *Mitigation:* hard per-iteration budget + error-driven targeting.
- **[MED] OOD brittleness in deployment.** *Mitigation:* input-domain guard + SAFETY gate.
- **[MED] Drift post-deploy.** *Mitigation:* drift-monitor hook + re-distill cadence (documented, not built in v1).

---

## Definition of done (v1)
A single command rebuilds a quantized specialist that **passes all six gates** on a frozen, human-curated test set, served on commodity hardware, with a model card, an honest writeup, and a benchmark table — for **one** flagship task. Generalization to a second task is a *stretch* proof, not required for v1.
