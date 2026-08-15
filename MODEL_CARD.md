# Model Card — Forge PII Specialist (Qwen2.5-1.5B + LoRA)

> **Status: NOT RELEASED.** This card documents a model that has not yet passed
> its gates. Numbers marked `PENDING` are unmeasured; numbers shown are from the
> latest completed run and are labelled with that run's id. Nothing here should
> be quoted as a result until the gate table at the bottom says PASS.

## Model details

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) |
| Adaptation | LoRA SFT (PEFT), fp16 on Apple Silicon MPS |
| Teacher | `openai/gpt-oss-120b` (Apache-2.0, MoE ~117B total / ~5.1B active) |
| Task | PII span detection over English text, 19 entity types |
| Output | JSON list of `(start, end, label, text)` spans |
| Contract | [`contracts/pii_redaction_v2.yaml`](contracts/pii_redaction_v2.yaml) |
| License | Apache-2.0 (inherits base + teacher) |

**Compression, stated honestly:** the teacher is sparse. Parameter ratios are
**~78x** on total parameters and **~3.4x** on parameters active per token.
Quoting only the first number would overstate the result.

## Intended use

Detecting and redacting PII in short English text (emails, chat messages,
support tickets, form free-text) **on-device**, where sending the text to a
hosted API would itself be the privacy violation — the compliance case under
GDPR and India's DPDP Act 2023.

### Out of scope

- Non-English text, code, or documents beyond ~512 tokens. The contract
  specifies these should return an out-of-domain flag, not guessed spans.
- Any setting where a missed identifier is unrecoverable without human review.
  High-severity recall is gated at 0.99, not 1.0 — the model is a filter, not a
  guarantee.
- Legal or regulatory certification of any kind.

## Training data

| Property | Value |
|---|---|
| Total records | 837 |
| Teacher-annotated (verification-gated) | 150 |
| Construction-verified synthetic | 687 |
| Total spans | 1,519 |
| Faker seed | 1337 (gold set uses 42 — disjoint by construction) |

Two provenance classes, deliberately kept distinct:

1. **Teacher-annotated** — the teacher labels seed text, k-sample
   self-consistency majority vote accepts or rejects, and rejects are logged
   rather than silently dropped (ADR 0002).
2. **Construction-verified synthetic** — PII values are *inserted* into
   templates, so offsets are exact by construction and no annotation step can
   introduce error (ADR 0009). Stronger than teacher verification for synthetic
   text, but less diverse than natural text — a real limitation, not a win.

**Leakage control:** three layers — exact-duplicate, near-duplicate (character
5-gram Jaccard >= 0.85), and gold-set leakage. Train/test share no text, and
the generators use disjoint Faker seeds.

**No proprietary, internal, or scraped data.** The entire corpus regenerates
from a fixed seed with `make gold` and `make data-engine`.

## Evaluation

Frozen gold set: **574 records** (385 test / 189 dev), never trained on,
regenerated only from seed 42. Primary metric is micro-F1 over exact
`(start, end, label)` matches — partial credit is not given, because a span
that half-covers a credit card number still leaks it.

### The bar

The teacher was scored on the same frozen set under the same harness, **before** the
student finished training, so the parity threshold cannot be back-fitted:

| Teacher (GPT-OSS-120B) | Value |
|---|---|
| micro-F1 | **0.9482** |
| micro-precision / recall | 0.9615 / 0.9353 |
| p50 / p95 latency | 0.59 s / 8.02 s |

**⇒ G1 requires student micro-F1 ≥ 0.9292; G4 requires student p95 ≤ 1.60 s.**

### Gate table

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| G1 quality parity | ≥ 0.9292 | run_002 scoring | PENDING |
| G2 schema validity | ≥ 99.9% | 100% (run_001, 385/385) | PASS |
| G3 cost per 1k | ≤ teacher / 10 | run_002 scoring | PENDING |
| G4 p95 latency | ≤ 1.60 s | run_002 scoring | PENDING |
| G5 deployability | runs on laptop / CPU | partial — unquantized MPS only | PENDING |
| G6 safety / OOD | ≥ 0.90 | not started | PENDING |
| High-severity recall | ≥ 0.99 on 9 types | **1.0000 on all 9** (validator layer) | PASS |

**Run history**

| Run | Train records | Micro-F1 | Precision | Recall | Note |
|---|---|---|---|---|---|
| run_001 | 150 | 0.52 | 0.66 | 0.43 | teacher-annotated only; severe type imbalance |
| run_002 | 837 | PENDING | PENDING | PENDING | + targeted augmentation (ADR 0009) |

### The system is a hybrid, and the numbers are reported separately

The teacher misses 6 of 9 high-severity recall floors (DRIVER_LICENSE 0.53, BANK_ACCOUNT
0.79, AADHAAR 0.83). Distillation transfers blind spots, so no student trained on it can
clear those floors. Per ADR 0012, the nine high-severity types are handled by
**deterministic validators** — Verhoeff for Aadhaar, Luhn for cards, format and
nearest-keyword context rules for the rest — which reach **1.0000 recall on all nine**.

| Layer | What it covers | High-severity recall |
|---|---|---|
| Distilled model | PERSON, LOCATION, STREET_ADDRESS, DOB, AGE, USERNAME, EMAIL, PHONE, URL, IP | n/a |
| Validators | the 9 high-severity structured identifiers | **1.0000** |

Three numbers are always published together: **model-only** (this is G1), **validator-only**,
and **system**. Quoting the system score as though it measured the distillation would
misrepresent what the model learned.

Note that checksums are used as a *precision and disambiguation* signal, never as a recall
gate — see Limitations for why that distinction is load-bearing here.

## Known limitations

- **Multi-span blindness.** run_001 emitted roughly one span per record while
  many records contain 2-4 entities; PERSON recall was 14% despite 62 training
  examples. Addressed by multi-entity templates in run_002, effectiveness
  unconfirmed.
- **India-specific types are the weak point.** AADHAAR, PAN, and DRIVER_LICENSE
  had the worst recall and the least training data. The development teacher was
  also weak on them, which is why augmentation is construction-verified rather
  than teacher-annotated.
- **False positives on number-like strings.** SSN precision was 39% in run_001;
  the model labelled order IDs and reference numbers as SSN. Hard negatives were
  added for run_002.
- **Synthetic evaluation.** The gold set is synthetic (Faker into templates).
  It is reproducible and leak-free, but it is *not* natural text, so scores
  overstate performance on messy real-world input. An external natural-text
  benchmark is the obvious next step and has not been run.
- **The synthetic identifiers are not structurally valid.** Only 2 of 29 AADHAAR
  values in the gold set satisfy the Verhoeff checksum (random chance is ~1/10),
  because the generator emits random digits — whereas *every real* Aadhaar number
  is checksummed. Credit cards happen to be Luhn-valid; the rest are not
  format-verified. Two consequences, both real: the validators cannot use
  checksums as a recall gate (doing so would score 0.07 here and ~1.0 on real
  data, i.e. recall that silently depends on the dataset), and validator
  precision on real input would likely be *higher* than measured here, since
  checksums would filter false positives. A future contract version should
  regenerate the gold set with structurally valid identifiers.
- **The gold set has not had a documented human verification pass** (Protocol
  §5). Until it does, "human-verified" is not a claim this project may make.
- **Unquantized.** Latency and size numbers are for fp16 on MPS. GGUF/AWQ
  artifacts must re-pass every gate before any on-device claim is final.

## Reproducing

```bash
git clone https://github.com/Jatin-IITB/forge && cd forge
make install
export CEREBRAS_API_KEY=...    # free tier: https://cloud.cerebras.ai
make forge
```

The teacher endpoint is fungible: point `TEACHER_URL` at Groq or a local vLLM
serving the same Apache-2.0 checkpoint and the pipeline is unchanged.
