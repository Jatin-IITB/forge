# ADR 0016 — Constrained decoding ships, and we publish the worse F1

**Status:** accepted (2026-09-03)
**Date:** 2026-09-03
**Depends on:** `forge/grammar.py`, `scripts/bench_grammar.py`, `reports/bench/grammar_abc.json`
**Related:** ADR 0012 (validator layer — the same recall/precision asymmetry), ADR 0018 (serving stack — the ≤0.01 exit gate)

## Context

`docs/DESIGN.md` has named constrained decoding as the reliability mechanism since the design
was written:

> For structured outputs (JSON/enum), the student decodes under a grammar/schema constraint,
> so it *cannot* emit invalid output. This is how a 1–3B model reaches ~100% schema-validity —
> a place small models otherwise lose to large ones for free.

`ACTION_PLAN.md` Phase 3 step 2 lists it as work. **It was never built.** G2 has been failing at
0.9974 (384/385) ever since — one malformed response, against a threshold with no margin at
this sample size (`0.999 × 385 = 384.6`, so 385/385 is the only passing score).

`forge/grammar.py` now implements it. This ADR decides whether it ships, because the
measurement was not the clean win the design anticipated.

## The measurement

Three arms, one llama-server process at `-np 8 -c 8192 --mlock`, shipped Q8_0 GGUF, full
frozen 385-record test set, greedy decoding. One server and identical flags because batched
Metal decode is not bit-deterministic — measuring arms under different serving configs would
confound the constraint with the config.

| arm | micro-F1 | 95% CI | precision | recall | schema | tok/rec |
|---|---|---|---|---|---|---|
| unconstrained | **0.6360** | [0.5958, 0.6747] | 0.6171 | 0.6561 | 384/385 ❌ | 51.4 |
| grammar (permissive) | 0.6198 | [0.5788, 0.6595] | 0.5795 | 0.6662 | **385/385** ✅ | 52.1 |
| grammar (exact spacing) | 0.6198 | [0.5788, 0.6595] | 0.5795 | 0.6662 | **385/385** ✅ | 52.1 |

**G2 passes by construction**, exactly as designed. And it costs **−0.0162 micro-F1**.

### A hypothesis this ADR recorded in advance, and which failed

Before running, the predicted mechanism was tokenization: grammar-constrained sampling filters
at the *token* level, so optional whitespace lets the decoder accept `{` and `"spans"`
separately when the model's trained continuation is the merged token `{"spans":`. Pinning the
grammar to the exact byte sequence in the system prompt should then recover the loss.

**It recovered nothing.** `grammar_exact` scores identically to `grammar_permissive` — same F1
to four decimals, same interval, same token count. The explanation is that the model already
emits exact spacing, so the permissive grammar's flexibility was never exercised; the two arms
were the same experiment. The hypothesis was not refuted so much as revealed to be untestable
this way, and the −0.0162 has some other source.

### Where the F1 actually goes — the finding that decides this

Diffing predictions record by record (`data/predictions_student_q8_*.jsonl`):

- 41 of 385 records changed.
- **Zero true positives lost.**
- **Seven true positives gained.**

The decomposition:

| | precision | recall |
|---|---|---|
| unconstrained | 0.6171 | 0.6561 |
| grammar | 0.5795 | **0.6662** |
| Δ | **−0.0376** | **+0.0101** |

**The grammar does not lose anything. It raises recall and lowers precision**, and micro-F1 —
which weights the two equally — reports the net as a loss.

The mechanism is now clear: unconstrained, a malformed or truncated response is partially
salvaged by `parse_response`, which recovers whatever spans it can and drops the rest. Under a
grammar the model must emit a complete well-formed object, so everything it wanted to say
arrives. More spans, more correct ones *and* more wrong ones.

## Decision

**Constrained decoding ships as the default decode path. G1 is reported at the lower number,
0.6198 (ratio 0.6537), not the higher one.**

Four reasons, in order of weight:

1. **The errors are not comparable, and the contract already says so.** This is a redactor. A
   false negative is an identifier that leaves the device unredacted. A false positive is text
   over-redacted on a machine that never transmits it. `contracts/pii_redaction_v2.yaml` gates
   **high-severity recall at 0.99 and contains no precision gate anywhere** — the asymmetry is
   pre-committed, not invented here. ADR 0012 built the entire validator layer on the same
   premise.
2. **G2 moves FAIL → PASS, and G1's verdict is unaffected.** G1 fails under both arms: 0.6707
   and 0.6537 against a 0.98 threshold. The gap is ~0.33; the difference between the arms is
   0.0162, roughly one-twentieth of it. No reading of the parity gate turns on this choice.
3. **A structural guarantee beats a probabilistic one for a safety property.** "The decoder
   cannot emit invalid output" is a different kind of claim from "it usually doesn't". G2 is a
   reliability gate; reliability from construction is worth more than the same number reached
   by luck on a particular test set.
4. **It strictly adds information.** Zero true positives lost across 385 records. The change
   is not a trade of one kind of correctness for another; it is more output, of which some is
   right and some is wrong.

## The precedent this has to answer

Four commits earlier this project rejected Q4_K_M for costing **−0.0151 micro-F1** against a
≤0.01 exit gate, despite being 661 MB smaller. Accepting −0.0162 here looks like the same rule
applied twice with opposite results, so the distinction has to be stated rather than glossed.

| | costs | buys | gate verdicts changed |
|---|---|---|---|
| Q4_K_M | −0.0151 F1 | 661 MB of disk | **none** |
| grammar | −0.0162 F1 | G2 passes; recall +0.0101 | **G2: FAIL → PASS** |

Q4_K_M paid quality for a resource saving that moved no gate. The grammar pays a *precision*
component for a gate flip and a recall gain. The exit gate in ADR 0018 was written for
quantization artifacts, and extending it to decoding configuration would be inventing scope —
but so would waving it away. It is recorded here explicitly so the two decisions can be
audited against each other rather than discovered to disagree later.

**What this decision does not do:** it does not change G1's metric. G1 remains micro-F1 as
pre-committed, and the shipped configuration's G1 number is the lower one. Choosing the
grammar for its recall while quoting the unconstrained F1 for parity would be selecting a
metric per gate to flatter each — the precise gaming the contract exists to prevent. **We take
the F1 hit on the record.**

## Consequences

- **G2: FAIL → PASS.** Schema validity 385/385, structurally rather than empirically.
- **G1: still FAIL, and slightly worse than published.** 0.6198, ratio **0.6537**. The ledger
  carries the shipped artifact's number.
- **Precision regresses to 0.5795** and this is now the system's weakest headline number. It
  is not gated, but it is real: roughly 6% more spans are wrong. Redaction over-fires more.
- **A hallucinated label is now unrepresentable.** The grammar admits only the 19 enum values,
  so an invented type cannot be produced, let alone scored.
- **The three regex fallbacks in `forge/inference.py`** for markdown fences, `<think>` blocks
  and bare-brace extraction are unreachable on the shipped path. They stay for the API-teacher
  path, which is unconstrained.
- **Token cost is unchanged** (+0.7/record), so this neither helps nor hurts G3.

## What would change this decision

If precision were gated, or if a future run brought G1 within 0.02 of the threshold — where
0.0162 would decide the verdict rather than being lost inside a 0.33 gap — this trade would
have to be re-argued rather than inherited. The condition is written down so the re-argument is
triggered by a number, not by someone happening to remember.

## Open, and not claimed

The precision regression is untreated. The grammar makes the model emit everything it wanted
to say, which shows that the underlying model over-enumerates spans — the same conclusion ADR
0013 reached from the opposite direction when run_003's capacity increase collapsed recall.
Constraining the *number* of spans, or filtering low-confidence ones, is untested. This ADR
ships a validity fix and records a precision problem it does not solve.
