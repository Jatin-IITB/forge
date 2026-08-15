# Honest Assessment

*What in this project is genuinely novel, what is solid-but-standard engineering, what is
weak, and how it compares to the field. Written to be read by someone looking for reasons to
doubt it.*

Last updated: 2026-08-15, while Phase 4 is still running. Sections that depend on unfinished
runs say so rather than guessing.

---

## 1. Novelty calibration

The honest split between "I invented this" and "I implemented this well."

### Standard practice — no novelty claimed

- **Distillation of a large teacher into a small student.** Textbook since Hinton et al. 2015;
  the modern LLM form is Alpaca/Vicuna-era and thoroughly explored.
- **LoRA / QLoRA fine-tuning.** Hu et al. 2021, Dettmers et al. 2023. Using PEFT is using a
  library, not contributing one.
- **Self-consistency sampling as a quality filter.** Wang et al. 2022.
- **Active learning / error-driven data acquisition.** Decades old. Applying it to LLM
  distillation is current practice, not a new idea.
- **Near-duplicate detection via n-gram Jaccard.** Standard corpus hygiene.

Roughly 85% of this repo is in this bucket. That is the expected ratio for an engineering
portfolio project and is not a criticism — the value is in whether the assembly is rigorous.

### Genuinely uncommon in practice (not in theory)

- **Gates pre-committed in an immutable contract, with version bumps required to change the
  teacher.** The *idea* is standard science. What is rare is the mechanical enforcement: the
  contract is a validated artifact, and when ADR 0010 changed the teacher, the response was a
  new contract version with a programmatic check that all six gate thresholds were byte-identical
  — not an edit. Most portfolio projects (and plenty of production ones) quietly retune the
  target after seeing results.
- **Construction-verified synthetic data as a deliberate alternative to teacher annotation.**
  When error analysis showed the teacher was weakest on exactly the types the student most
  needed (AADHAAR, PAN, DRIVER_LICENSE), the response was to generate data whose labels are
  exact *by construction* — offsets computed at insertion time — rather than trusting a weak
  annotator. The technique is simple; noticing that the verification gate cannot manufacture
  signal the teacher never produced is the actual contribution.
- **Economics as a pass/fail gate with a published cost model, priced against paid tiers while
  developing on a free one.** Most write-ups quote a cost ratio with no derivation. Here every
  input is a CLI flag and the report prints the arithmetic.

### Where the project got *less* novel on contact with reality

The teacher was supposed to be Qwen3-32B. No free host serves it, so the teacher became
GPT-OSS-120B. That is a *larger* teacher and a better headline, but it was a constraint-driven
substitution, not a designed choice — and the honest framing (ADR 0010) says so.

---

## 2. Comparison to the field

| Approach | What it does better than Forge | Where Forge holds up |
|---|---|---|
| **Presidio** (Microsoft) | Mature, multi-language, regex + NER ensemble, battle-tested, free | Forge targets contextual cases regex misses; Presidio needs per-type rule engineering |
| **spaCy / GLiNER NER** | Faster, smaller, no LLM needed, strong on PERSON/ORG/LOC | Forge covers structured identifiers (Aadhaar, PAN, API keys) that generic NER has no notion of |
| **Commercial DLP** | Compliance certifications, support, integrations | Forge runs air-gapped with no vendor; the entire pipeline is auditable |
| **Prompting a frontier model** | Better zero-shot quality, no training | Defeats the purpose — sending PII to an API to find PII is the violation |

**The uncomfortable comparison:** for well-formed identifiers (credit cards, SSN, Aadhaar),
a well-written regex with a checksum is likely to beat a 1.5B model on both precision and
latency, at zero cost. A defensible version of this project must eventually show either
(a) the model winning on *contextual* PII where regex fails, or (b) a hybrid where regex
handles structured types and the model handles the rest. **Neither is measured yet.** Until it
is, "specialist model beats the alternatives" is not a claim this project has earned — only
"specialist model approaches its teacher" is on the table.

---

## 3. Known weaknesses

Ordered by how much they should worry a reader.

1. **No gate has passed except schema validity.** run_001 scored F1 0.52 against a 0.98-parity
   target. run_002 is training. The pipeline is proven to *run*; it is not yet proven to
   *work*.
2. **The evaluation set is synthetic.** Faker values injected into templates. It is
   reproducible and leak-free, but it is not natural text, and scores on it will overstate
   real-world performance — probably substantially, since real PII appears in messier context
   with more ambiguity. An external natural-text benchmark is the single highest-value missing
   piece.
3. **The gold set has never had a documented human verification pass.** The protocol requires
   one (§5). Until it happens, the phrase "human-verified" must not appear in any description
   of this project, and any draft that uses it is wrong.
4. **The frozen gold set was silently drifting until 2026-08-15.** A clock-dependent generator
   made it reproducible only within a single day (ADR 0011). Caught by a test that had been
   failing and was initially dismissed as environmental. The fix reproduces the committed data
   bit-for-bit, so no measurement was invalidated — but the defect existed for eight days
   across several commits, and the lesson is that a red test on a foundational invariant is
   never noise.
5. **Teacher-annotated training data is small (150 records) and was produced by the weak 8B
   development teacher.** It needs regeneration with the 120B before any parity claim.
6. **Single task, single language, single domain.** Nothing here demonstrates the pipeline
   generalizes to a second task, which is the actual claim implied by calling it a "pipeline."
   The DESIGN doc calls this a stretch goal; it is more accurately a gap.
7. **Training runs on Apple Silicon MPS in fp16 because bf16 produces NaN and fp32 segfaults.**
   That is a workaround around a platform limitation, and it constrains batch size and speed
   enough that iteration is slow (~3 min/step, ~5 h per run).
8. **QLoRA and AWQ are unrun** — both require CUDA, which this hardware lacks. Claiming either
   without a rented-GPU run would be false.

---

## 4. What would change my mind about this project's value

- **If run_002 does not substantially beat 0.52**, the error-driven loop's central premise —
  that targeted augmentation fixes measured failures — is unsupported, and the honest
  conclusion is that the imbalance diagnosis was wrong.
- **If a regex baseline beats the model on high-severity types**, the deployment story should
  become a hybrid, and the "specialist model" framing should be retired in favor of
  "specialist system."
- **If the quantized model fails the gates**, the on-device claim collapses to "on-device with
  an asterisk," and the asterisk belongs in the headline.

Each of these is checkable, and each is a real possibility rather than a rhetorical hedge.

---

## 5. What this project does demonstrate, today

Independent of whether the gates ultimately pass:

- An **eval-first discipline** that is mechanically enforced rather than merely asserted:
  frozen test set, pre-committed thresholds, contract versioning, and a bug that was fixed
  in a way that provably did not alter the committed data.
- A **diagnosis-to-intervention loop** that actually closed: run_001's failure was traced to
  per-type imbalance and multi-span blindness, and the intervention targeted exactly those.
- **Infrastructure honesty:** the AWQ path refuses to pretend it ran, the economics harness
  prices the teacher at rates we did not pay, and the model card leads with "NOT RELEASED."

The methodology is the artifact. The model is the test of it.
