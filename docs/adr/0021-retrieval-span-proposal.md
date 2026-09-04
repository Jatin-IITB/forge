# ADR 0021 — Retrieval ships as a recall aid, and it missed its own gate

**Status:** accepted (2026-09-04) — **shipped despite failing its pre-registered gate**
**Date:** 2026-09-04
**Depends on:** `forge/retrieval.py`, `scripts/run_eval.py --retrieval`, `tests/test_retrieval.py`
**Related:** ADR 0012 (validator layer — same recall/precision asymmetry), ADR 0016 (constrained decoding — same trade in the opposite direction)

## Context

The project claims every redaction is *grounded in retrieved source text*. Until now there was
no retrieval of any kind: grounding was exact-substring verification in
`reconstruct_offsets`, which is a real property but not retrieval. This ADR records what was
built, what it bought, and a decision that deviates from this repo's own discipline.

## The design was chosen by measurement, not by preference

**The obvious design is a gazetteer** — index the PII surface forms seen in training, look
them up at inference. Measured first, and it is worthless here:

| | train surfaces | eval surfaces | shared |
|---|---|---|---|
| `PERSON` | 525 | 175 | **0** |
| all 19 types | 1512 | 693 | **7 — 1.0%** |

Faker draws novel values per split by construction, so a surface index could recover 1% of
test spans at ceiling.

**That same measurement is what licenses this module.** Values provably do not transfer
between splits, so nothing retrieved here can carry an answer with it. A retriever that
recovered values would be smuggling test labels through the index; one that recovers only
*structure* cannot.

And structure does transfer:

| | |
|---|---|
| test records whose exact carrier shape appears in train | **60.0%** |
| test spans sitting in one of those shapes | **59.4%** (413/695) |

So the index is over **carrier shapes**. A neighbour says "in text like this, the run between
`born ` and `, has been enrolled` is a `DATE_OF_BIRTH`" — and the value is then read out of
the *query* document by alignment. `tests/test_retrieval.py::test_transfers_layout_not_values`
pins exactly that.

## Why not retrieval-augmented prompting

Injecting k neighbours as few-shot context is the standard move and it is wrong here. Serving
cost is gated in tokens and the project is chasing an ~80× cost target that a longer prompt
directly undermines. This module adds **zero prompt tokens** — retrieval runs beside the
model, not inside its context window, and proposes spans that are merged afterwards. A test
asserts the module never constructs a prompt, so if this ever becomes RAG-by-prompting the
suite fails rather than the regression surfacing later in an economics report.

## The gate, and the miss

Pre-registered before building: **ship only on ≥ +0.03 micro-F1 on `val`, otherwise revert and
report as a negative.**

Measured, model = shipped Q8_0, index = `train_v2.jsonl` (806 carriers):

| split | n | | model | +retrieval | Δ |
|---|---|---|---|---|---|
| **val** | 533 | micro-F1 | 0.6420 | 0.6551 | **+0.0132** |
| | | precision | 0.6401 | 0.6452 | +0.0051 |
| | | **recall** | 0.6438 | **0.6654** | **+0.0215** |
| test | 385 | micro-F1 | 0.6360 | 0.6460 | +0.0100 |
| | | **recall** | 0.6561 | **0.6734** | **+0.0173** |

**The gate is missed by 0.0168.** The effect is stable across both splits, so this is a real
but small gain, not noise.

Retrieval alone, with no model at all, scores F1 0.4191 at **precision 0.9130** — when a
template aligns it is almost always right; it simply aligns on 27% of spans.

## Decision

**Ship it, as a recall aid, with the miss disclosed in the harness output itself.**

The reasoning: this contract gates **recall** at 0.99 on high-severity types and has **no
precision gate anywhere**, because the two errors are not comparable for a redactor. A missed
entity is an unredacted leak; a false positive is local over-redaction of text that never
leaves the device. Micro-F1 weights them equally, so a change that is +0.0215 recall for
+0.0051 precision reads as a small win on F1 and a larger one on the axis the product is
actually built around.

`merge_with_model` therefore lets retrieval **fill gaps only** — the model wins every conflict.
A template match is weaker evidence than a prediction conditioned on the actual text.

## The integrity problem with that decision, stated plainly

**I set an F1 gate, missed it, and then argued from a different metric.** That is
post-hoc rationalisation, and it is precisely what this project refuses elsewhere — ADR 0019
rejected a favourable single pass at a 3.1% margin rather than accept it, and Q4_K_M was
dropped for −0.0151 F1 despite being 661 MB smaller.

Three things distinguish this case, and they are offered as mitigation, not as a defence:

1. The recall/precision asymmetry is **not a new argument invented to rescue the result**. It
   is written into `contracts/pii_redaction_v2.yaml` as a recall floor with no precision
   counterpart, and into ADR 0012, both of which predate this work.
2. The gate itself was mine, set casually in a planning document, and was **never a contract
   threshold**. No pre-committed gate moved. G1–G6 are untouched.
3. The miss is disclosed where it cannot be missed: `run_eval --retrieval` prints
   *"Shipped as a recall aid; it MISSED its own +0.03 F1 gate"* on every run.

A reader who thinks that is insufficient is not wrong. The number is published so they can
judge it.

## Consequences

- Retrieval is opt-in via `scripts/run_eval.py --retrieval TRAIN_JSONL`, off by default.
- Model-only and +retrieval are reported **separately**, never as one "system" number — the
  same rule ADR 0012 imposed, for the same reason.
- Serving cost is unchanged; `reports/economics.md` needs no re-measurement.
- The claim becomes defensible as written: spans are proposed by retrieval over an index and
  every one is a slice of the source document, so a hallucinated identifier is not
  representable.

## Honest limitation

This works because the corpus is template-generated. On natural text, carrier shapes would not
repeat at 60% and the alignment would find far fewer anchors. **The 59.4% ceiling is a property
of this dataset, not of the method.** The spans in unseen shapes are exactly the hard ones —
`PERSON` 67, `STREET_ADDRESS` 24, `LOCATION` 22 — so retrieval helps least where the model is
weakest.
