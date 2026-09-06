# ADR 0017 — Task formulation was the ceiling: span extraction becomes token classification

**Status:** accepted (2026-09-07)
**Date:** 2026-09-07
**Depends on:** `forge/token_classifier.py`, `scripts/train_token_classifier.py`, `scripts/bench_serving.py` (`--backend token-classifier`), `reports/bench/tc_v4_cuda_raw.json`
**Related:** ADR 0013 (capacity-not-data diagnosis — this ADR overturns its dichotomy), ADR 0016 (constrained decoding — whose F1 tax this refunds), ADR 0012 (validator layer — demoted to backstop), ROADMAP WP-4 (this closes its exit gate)
**Closes:** ledger row 21; moves rows 5, 14, 19; reopens rows 3, 4

## Context

ROADMAP WP-4 ("Recall repair") was written against a generative student that would not
enumerate. `run_003` added capacity on identical data: loss fell 5× to 0.22, precision rose to
0.86, and **recall collapsed to 0.39** — span ratio 0.46. The model was learning the task and
emitting half the entities anyway.

WP-4 listed an escalation ladder — decode audit, loss reweighting, two-pass decode, DPO — and
one exit gate:

> **Exit gate:** span ratio ≥ 0.9 on dev, or **a written finding that the task formulation is
> the ceiling.**

This ADR is that written finding. None of the four rungs was climbed, because the diagnosis
they share turned out to be wrong.

### The dichotomy this overturns

ADR 0013 and the v3 pre-registration both framed the gap as **capacity or data**. v3 tested the
data arm at 3× volume and the score went *backwards* (0.8803 → 0.8517), which the v3 write-up
correctly attributed to mixture rather than amount. But the pre-registered inference attached
to that failure — "if more data does not help, the constraint is capacity" — was a false
dichotomy on a third axis neither arm varied: **how the task is posed to the model.**

## Decision

Keep the Qwen2.5-1.5B backbone. **Replace the vocabulary head with a 77-label BIOES tagging
head** (19 PII types × {B,I,E,S} + `O`) and reconstruct `PIIRecord` spans from tags in
deterministic code (`forge/token_classifier.py`).

The model no longer *writes* an answer. It labels every input token in one forward pass, and
ordinary code turns labels into spans.

## The measurement

Frozen 385-record test set, model-only (no validator layer), greedy/argmax, raw text input.
Gold carries 695 spans.

| model | data | micro-F1 | precision | recall | pred spans | span ratio |
|---|---|---|---|---|---|---|
| generative student (Q8_0, shipped) | v2 | 0.6360 | 0.6171 | 0.6561 | — | — |
| generative `run_003` (capacity arm) | v2 | — | 0.86 | 0.39 | — | **0.46** |
| token classifier | v2 | 0.8803 | 0.8874 | 0.8734 | 684 | 0.984 |
| token classifier | v3 | 0.8517 | 0.8403 | 0.8633 | 714 | 1.027 |
| **token classifier** | **v4 (v2 ∪ v3)** | **0.9755** | **0.9783** | **0.9727** | **691** | **0.994** |

Teacher (GPT-OSS-120B) scores **0.9482**. G1 needs ≥ 0.9292.

**The v4 student beats its own teacher: ratio 1.0288.**

### The mechanism, stated so it can be falsified

Under-enumeration was never a knowledge failure. It was a property of **sequential generation
under a single output budget**: every additional span competes for the same decode sequence,
and a model that has committed to closing the JSON has no way to reconsider. Recall therefore
degraded with the number of entities per record, and adding capacity made the model *more*
confident about stopping early — which is exactly the precision-up/recall-down trajectory
`run_003` showed.

Token classification removes the competition. Each token's label is decided independently in
one pass, so a record with nine entities costs the same as a record with one. The prediction
this makes is sharp: **span ratio should jump to ≈1.0 and stay there regardless of data
mixture.** It does — 0.984, 1.027, 0.994 across three very different corpora, against 0.46 for
the generative arm. Mixture still moves *which* spans are right (v3 is worse than v2), but it
no longer moves *how many* get emitted.

## Consequences

### Gained

- **G1 passes** — 0.9755 [0.9622, 0.9869] vs a 0.9292 gate; the student exceeds the teacher.
- **G2 becomes structural** — there is no decoder, so no token sequence can be malformed.
  This **refunds the −0.0162 micro-F1** that ADR 0016 paid for constrained decoding: validity
  and F1 are no longer in tension.
- **The decode phase disappears** — 53.4 output tokens become **0**. Since autoregressive
  decoding was 89% of serving time, this is the largest single serving change the project has
  made, and it is the reason ledger row 4's impossibility proof no longer binds.
- **The validator layer is demoted to a backstop** — the model now carries 8 of 9
  high-severity types at 1.0000 recall unaided, where the generative student needed validators
  for all nine. Only `API_KEY` (0.9655 model-only, 1 miss in 29) still depends on it.

### Given up — the honest side of the trade

- **No rationales.** ADR 0007 (rationale-augmented distillation) does not apply to a tagging
  head. A dropped capability, not a deferred one.
- **No unseen labels.** The taxonomy is frozen into the head's output dimension. Adding a
  20th PII type is a retrain, where the generative model could in principle be prompted.
- **Span-only output.** Anything that is not a labelled substring of the input — normalization,
  explanation, abstention phrased in prose — is now out of scope for the model and must live in
  code (`forge/ood.py` already does).
- **No GGUF path.** llama.cpp serves causal LMs; a 77-label tagging head is not one. The
  shipped Q8_0 artifact (row 2/17) is the *generative* model, so **the model that passes G1 and
  the model that passes G5 are currently different models.** This is the project's new top
  problem, and it was created by this decision.
- **The distillation story changes shape.** "120B→1.5B" now means the teacher supplied the
  labels, not the output format. That is still distillation, and the compression ratio is
  unchanged, but a writeup that implies the student learned to imitate the teacher's *behaviour*
  would be overclaiming. It learned from the teacher's *annotations*.

### Not claimed

G3 and G4 are **not** marked passing. Every v4 speed number comes from an RTX 3050 while
`run_economics.py` carries the M1's purchase price and 22 W draw; pricing one against the other
is the substitution `PARALLEL_PLAN.md` names as "the easiest available way to fake this gate."
Both gates stay ❌ pending an M1 measurement of this checkpoint.

The G1 **paired** bootstrap is also not run — `data/predictions_teacher_120b_relat.jsonl` is
gitignored and absent. The ledger quotes a conservative two-interval bound instead (student CI
floor 0.9622 > 0.98 × teacher CI ceiling 0.9448) and labels it as the weaker instrument.

## Alternatives rejected

- **The WP-4 ladder as written** (decode audit → loss reweighting → two-pass decode → DPO).
  Each rung treats under-enumeration as a tuning problem. If the mechanism above is right, all
  four buy fractions of a structural loss; two-pass decode in particular pays a second full
  inference to work around a constraint that simply need not exist.
- **More data.** v3 tested it and lost 0.03 F1. Volume was never the axis.
- **More capacity.** `run_003` tested it and traded recall for precision at flat F1.
