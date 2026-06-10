# Forge — Task-Specialization Distillation Pipeline

**Codename:** Forge (working name — rename freely).
**Where it lives:** a standalone repo (`forge/`), independent of Prism/Aroha.
**Who uses it:** an ML engineer who has *one expensive, high-volume LLM task* and wants to stop paying frontier-API prices for it.
**One-line summary:** Forge takes a task spec + a teacher model and **manufactures a verified specialist student model** that matches the teacher's task quality within a pre-committed tolerance, at **10–100× lower cost and latency**, runnable fully private — and it does so through an *eval-first, verification-gated, error-driven data loop*, not a one-shot fine-tune script.

This document explains the design from first principles, names the choices that are genuinely state-of-the-art vs. the ones that are solid-but-standard engineering, and compares Forge honestly to the rest of the field. It deliberately mirrors the discipline of the Aroha `AHSI_Code_Indexing_Design.md`.

---

## 0. Why this project exists (the gap it fills)

The flagship system in this engineer's portfolio (Aroha) proves one muscle exceptionally well: **orchestrating frontier LLMs** — prompts, tools, memory, multi-agent reliability, in-context "RL." It proves **zero** of another muscle: **building a model.** No SFT, no distillation, no preference optimization, no inference/serving economics, no GPU work. To a reader, Aroha says *"I can make models behave"* and leaves open *"...but can you make a model, cheaply and privately?"*

Forge is the answer to that open question. It is deliberately **not another agent.** Aroha *calls* GPT-4 at runtime and pays per token forever; Forge *produces a 1–8B model that replaces* GPT-4 for one task. Orchestration vs. **manufacture.** Different toolchain (PyTorch / `transformers` / PEFT / TRL / vLLM / constrained decoding / synthetic-data engineering), different value prop (a cheap private asset vs. a recurring API bill).

The economic premise is concrete: for a *narrow* task at volume, a small fine-tuned model routinely matches a frontier model — the frontier model's edge is *generality*, which a single task does not need. You are buying a sports car to commute one fixed route; Forge builds the commuter.

### 0.1 Hard constraint — independence (this asset must outlive any single job)

Forge depends on **nothing internal** to any employer/internship: open-weight teacher, permissive-license base model, **public datasets** (or synthetic data the open teacher derives from public seeds), and personal/commodity compute. No internal hosted models, no proprietary data, no Aroha/Prism code. The only thing carried over from Aroha is *documentation discipline* — portable practice, not a dependency. **Litmus test:** if access to everything internal were cut tomorrow, a stranger could still clone the repo and rebuild the model end-to-end. See `adr/0003`. This constraint also pushes the flagship task toward problems with strong public corpora — which, conveniently, are also the most *meaningful* (see §0.2).

### 0.2 Why the flagship is on-device PII redaction (meaning + independence align)

The chosen first proof is **detecting and redacting personally-identifiable information in text, fully locally.** It is the task where every constraint reinforces the others:
- **Meaning is airtight.** Sending sensitive text to a frontier API *to find sensitive text* defeats the purpose. So a local specialist isn't merely *cheaper* — for privacy/compliance (GDPR, India's DPDP Act 2023) it is the *only correct option*. The value prop is "redact PII without it ever leaving the machine," not just "save money."
- **It's fully independent.** Strong public datasets exist (e.g. open PII-masking corpora on Hugging Face); no internal data needed.
- **It's crisply measurable.** Entity/span-level precision, recall, F1 — no fuzzy judging.
- **It demos viscerally.** Paste sensitive text → watch it redact → pull the network cable and watch it still work.

Alternatives kept on the table (also public-data, also meaningful): structured field extraction from public documents; and a small specialist for an under-served Indian language (accessibility/democratization angle). Task is locked in Phase 0 (`ACTION_PLAN.md`).

---

## 1. What Forge is

Forge is a **pipeline that compiles a task into a model.** Input: a task contract. Output: a deployable, benchmarked, documented specialist + the evidence that it meets the contract.

A *task contract* is the load-bearing object. It declares, before any training:

```
TaskContract:
  task_id            "txn_field_extraction_v1"
  io_schema          input: free-text; output: JSON conforming to schema S
  metric             primary: field-level F1 (exact-match per field);
                     secondary: schema-validity rate, latency p95, $ / 1k req
  parity_target      student >= 0.98 * teacher_score   (the gate)
  teacher            the frontier reference model (the bar to match)
  constraints        max student size, target hardware, privacy (on-prem/air-gapped?)
  guardrails         in-domain input definition + out-of-domain refusal behavior
```

Everything downstream is derived from the contract. The contract is committed to git **first** and is immutable for a training run — this is Forge's correctness contract (see §4.1).

---

## 2. The pipeline

Forge is a **six-phase loop** (contract → measure → generate → train → evaluate → harden), and the *generate→train→evaluate* segment is a closed loop driven by the student's own errors.

```
Phase 0  CONTRACT + GOLD EVAL  (build the held-out gold set BEFORE anything else;
            │                   freeze the metric and the parity gate)
            ▼
Phase 1  MEASURE THE BAR        (score the teacher on the gold set → the number to beat;
            │                   score off-the-shelf small models zero/few-shot → the gap to close)
            ▼
Phase 2  DATA ENGINE  ◄─────────────────────────┐
            │  teacher generates labeled data     │
            │  + rationales; diversity sampling;   │  error-driven
            │  VERIFICATION GATE (self-consistency │  augmentation
            │  / verifier) drops bad examples;     │  (Phase 4 feeds
            │  dedup; difficulty targeting          │  hard cases back)
            ▼                                      │
Phase 3  TRAIN                                     │
            │  SFT (LoRA/QLoRA) on student;         │
            │  optional DPO on teacher-preferred   │
            │  vs student-failed pairs;            │
            │  constrained-decoding setup           │
            ▼                                      │
Phase 4  EVALUATE + MINE ERRORS  ─────────────────┘
            │  score on FROZEN gold set; CIs / significance;
            │  if parity gate not met → cluster failures →
            │  hand error clusters to the Data Engine (Phase 2)
            ▼  (gate met)
Phase 5  HARDEN + SERVE
            quantize (AWQ/GGUF); serve (vLLM / llama.cpp);
            measure real $ / 1k & p95; robustness + OOD + adversarial;
            input-domain guard; model card; one-command rebuild
```

The loop is the system. A one-shot "generate data → fine-tune → ship" pipeline is a script; Forge's value is the **error-driven closed loop** plus the **verification gate** on training data — the two places where naive distillation silently fails.

### Two non-obvious, load-bearing touches

- **Verification-gated data (the "garbage gate").** Every synthetic example must pass a verifier before it can train the student: self-consistency (teacher agrees with itself across k samples), schema/constraint validity, and (where cheap) a held-out re-derivation. Distillation's #1 failure mode is training on the teacher's *confident mistakes*; the gate is the unglamorous thing that makes the output trustworthy. This is Forge's analogue of AHSI's "no code bodies / disk-truth" discipline — refuse to persist what you can't trust.
- **Error-driven data targeting (the cost-shaper).** After each eval, failures are clustered and the Data Engine generates new examples *concentrated on those clusters*, not uniformly. This is active learning: spend teacher tokens where the student is wrong, not where it's already right. It is the analogue of AHSI's Levenshtein gate — the detail that determines whether the system is *affordable*, because teacher-API tokens for data generation are the dominant cost.

---

## 3. The success formula in plain English

A run is successful iff, on the **frozen gold set**:

```
student_score        >= parity_target  (e.g. 0.98 x teacher_score)   [QUALITY gate]
AND schema_validity   >= 0.999                                        [RELIABILITY gate]
AND cost_per_1k_req   <= teacher_cost / 10                            [ECONOMICS gate]
AND p95_latency       <= teacher_p95 / 5                              [LATENCY gate]
AND runs_on           <= target_hardware (e.g. single 24GB GPU / CPU) [DEPLOYABILITY gate]
AND OOD_refusal_rate  >= threshold on the adversarial/OOD probe set   [SAFETY gate]
```

All gates are **pre-committed in the contract** and measured with confidence intervals. "It seems about as good" is not a result; "98.5% of teacher F1 (95% CI ±0.6), 31× cheaper, 7× faster, on a single 24GB GPU" is. Full rubric in `SUCCESS.md`.

---

## 4. Architectural decisions (each will get an ADR)

### 4.1 Eval-first: the gold set exists before the model (correctness contract)
We build and freeze the held-out gold evaluation set in Phase 0, before generating a single training example, and the parity gate is committed to git up front. This forbids the most common self-deception in fine-tuning — tuning the target after seeing results. Mirrors AHSI's *mandatory index-before-retrieve*: the architecture forbids the gap rather than trusting discipline. → `adr/0001`.

### 4.2 Distill the rationale, not just the label
For reasoning-bearing tasks the teacher emits a short rationale + final answer; the student is trained on rationale-augmented targets (and the rationale can be dropped at inference for speed). Reason: chain-of-thought distillation consistently beats label-only distillation at equal data budget. Standard technique, deliberately chosen.

### 4.3 Constrained decoding closes the format gap
For structured outputs (JSON/enum), the student decodes under a grammar/schema constraint, so it *cannot* emit invalid output. This is how a 1–3B model reaches ~100% schema-validity — a place small models otherwise lose to large ones for free. → `adr/0004`.

### 4.4 Verification gate is mandatory on training data
No unverified teacher output enters the training set (§2). → `adr/0002`.

### 4.5 Economics are first-class metrics, not an afterthought
`$ / 1k requests` and `p95 latency` are measured from Phase 1 and are *gates*, not nice-to-haves — the entire premise is the cost delta, so it is instrumented like quality.

### 4.6 Reproducibility: one command rebuilds the asset
`make forge TASK=<contract>` runs data → train → eval → quantize → serve, seeded, and emits a model card + benchmark table. A model you cannot rebuild deterministically is not an asset.

---

## 5. Genuinely novel vs. solid engineering (calibrated, honest)

**Solid engineering, standard — credited, not claimed as novel:**
1. LoRA / QLoRA SFT — the default efficient fine-tuning method since 2023.
2. Synthetic data generation from a teacher model — widely practiced.
3. Quantization (AWQ/GGUF) + vLLM/llama.cpp serving — standard inference stack.
4. DPO for preference optimization — well-established.
5. Constrained / grammar-guided decoding — mature (Outlines, llama.cpp grammars).

**Where the craft actually lives (the combination, not any single piece):**
1. The **eval-first parity *contract*** with pre-committed gates and statistical rigor — most fine-tuning projects "eyeball it."
2. The **verification gate on training data** — refusing the teacher's confident mistakes.
3. The **error-driven active data loop** — concentrating teacher tokens on the student's failure clusters.
4. Treating **cost & latency as first-class gates** measured end-to-end on real serving.
5. The whole thing as a **reproducible, contract-driven system** that compiles *any* conforming task into a benchmarked model — not a one-off notebook.

Honest verdict on novelty: **no single component is a research contribution; the disciplined, reproducible *combination* is the contribution** — exactly the framing AHSI uses about itself. The portfolio value is that it proves end-to-end model-building competence, not that it invents distillation.

---

## 6. How Forge compares to the field

### vs. a vanilla Hugging Face fine-tuning script
- *Where the script wins:* trivially simpler; fine for a throwaway experiment.
- *Where Forge wins:* the script has no gold-eval contract, no data verification, no error loop, no serving economics, no reproducibility. It produces "a fine-tuned model"; Forge produces *evidence that the model meets a committed contract.*

### vs. OpenPipe / Predibase / "distillation-as-a-service"
- *Where they win:* polished managed UX, hosted infra, scale, support.
- *Where Forge wins:* transparent and self-hosted (runs air-gapped — decisive for fintech/PII), you own the eval + data loop, no per-token lock-in, and — for a portfolio — it *demonstrates you understand the whole pipeline* instead of clicking a SaaS button.

### vs. just prompting a small open model (few-shot, no training)
- *Where prompting wins:* zero training cost, instant.
- *Where Forge wins:* no long few-shot prompt tax (cheaper + faster per call), higher reliability via fine-tuning + constrained decoding, and it actually closes the quality gap that raw few-shot small models leave open.

### vs. full RLHF / RFT
- *Where RL wins:* when the task has no verifiable target and only preference signal, or needs to *exceed* the teacher.
- *Where Forge wins:* for a task with a checkable answer, SFT+DPO distillation reaches parity far cheaper and more stably than online RL. Forge notes explicitly *when* a task warrants escalating to RL (non-verifiable, open-ended) rather than pretending SFT is universal.

Honest framing: Forge sits on the boundary between "a fine-tuning recipe" and "a managed distillation platform" — it is the *transparent, reproducible, contract-driven* point in that space, optimized for trust and for proving competence.

---

## 7. Honest weaknesses, risks, and open questions

1. **Distillation ceiling.** The student cannot exceed the teacher. If the teacher is wrong on a slice, the student inherits it. Forge mitigates with the verification gate but cannot beat the teacher's blind spots.
2. **Gold-set quality is the foundation.** Every parity claim is only as good as the held-out gold set. A weak or leaky gold set makes the whole result meaningless — so Phase 0 (human-curated, de-duplicated, leakage-checked) is the highest-leverage and highest-risk phase.
3. **Teacher ToS / licensing — a real legal caveat.** Some frontier-API terms (e.g. OpenAI) restrict using outputs to train competing models. **Choose a teacher whose terms permit distillation** (open-weight teachers like Llama/Qwen/DeepSeek, or a provider whose ToS allows it) and use a base model with a permissive license. This must be settled in Phase 0; it is non-negotiable for a publishable/portfolio artifact.
4. **Narrowness is the point and the trap.** The specialist is intentionally brittle outside its task; without an input-domain guard it will confidently mishandle out-of-distribution input. The OOD/refusal gate (§3) exists precisely for this.
5. **Distribution drift.** If the task's real input distribution shifts post-deployment, the specialist silently degrades. Production use needs drift monitoring + periodic re-distillation; v1 ships the monitor hook, not the full MLOps loop.
6. **Cold-start teacher-token cost.** Generating + verifying the first dataset is the dominant spend (mirrors AHSI's "LLM-summary cost on first index"). The error-driven loop (§2) is what keeps subsequent rounds cheap; budget the cold start explicitly (`ACTION_PLAN.md` §Budget).

---

## 8. Verdict

**Is Forge state-of-the-art?** No single piece is — and the doc says so plainly. **Is it the right project?** Yes: it fills the exact, glaring gap in the portfolio (model manufacture vs. model orchestration), it has an *obvious buyer and quantifiable ROI* (a 90%+ drop in the per-task inference bill + on-prem privacy), and its **research-grade discipline** (eval-first parity contract, verification-gated data, error-driven active loop, statistical rigor) is what separates it from the thousand "I fine-tuned a model" repos. The artifact a reader walks away with is a single, undeniable sentence: *"a 1–3B model I built that matches GPT-4-class quality on \<task\> at ~1% of the cost, running on a laptop — here is the benchmark table and the one command that rebuilds it."*

---

## Sources / prior art to cite in the writeup
- Knowledge distillation (Hinton et al.); sequence-level & rationale/CoT distillation.
- LoRA / QLoRA (Hu et al.; Dettmers et al.).
- DPO (Rafailov et al.).
- Constrained decoding: Outlines, llama.cpp GBNF grammars.
- Serving: vLLM (PagedAttention), llama.cpp; quantization: AWQ, GPTQ, GGUF.
- Comparable products: OpenPipe, Predibase (for the honest field comparison).
