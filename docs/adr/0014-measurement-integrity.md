# ADR 0014 — Every gate number gets an interval, and three of them were wrong

**Status:** accepted (2026-09-03)
**Date:** 2026-09-03
**Supersedes:** the uncertainty-free reporting used in `reports/baseline_120b.md` and `reports/eval_run_002.md`
**Depends on:** `forge/ci.py`, `scripts/audit_gold.py`, `scripts/build_validation.py`

## Context

Contract v2 states that all six gates are "measured with 95% CIs". No published number
carried one. `0.5750` and `0.9292` were being compared as if both were exact, on a 385-record
sample.

That is not a cosmetic omission. Three of the six gates are defined *relative to* a teacher
measurement — G1 is `student_f1 ≥ 0.98 × teacher_f1`, G3 is `cost ≤ teacher_cost / 10`, G4 is
`p95 ≤ teacher_p95 / 5`. A gate defined as a multiple of a noisy quantity inherits every bit
of that noise, and nothing in the repo was tracking it.

Adding intervals was expected to be a reporting change. It surfaced four defects instead,
two of which had already influenced planning decisions.

## Decision

Add `forge/ci.py` (bootstrap intervals), `scripts/audit_gold.py` (`make audit` — structural
checks on the *committed bytes*), and `scripts/build_validation.py` (`make validation`).
Report every gate quantity with an interval and a stated measurement condition.

Three implementation choices are load-bearing:

1. **Resample records, never spans.** Spans within a record are correlated — the same fuzzy
   address boundary, the same missed sentence. Treating them as independent draws understates
   the interval, which is the direction that flatters a gate.
2. **Pair the resample across models.** G1's ratio has both sides measured on the *same*
   frozen records. Bootstrapping independently and dividing intervals reports **28% more
   uncertainty than the data holds** (width 0.110 vs 0.086), because it discards the fact that
   a record hard for the student is usually hard for the teacher too.
3. **Zero-failure cases get the exact binomial bound, not the bootstrap.** See finding 2.

## Findings

### 1. The teacher p95 is not reproducible, and the roadmap's explanation of why was wrong

Two runs of the **identical configuration** — same model, endpoint, `reasoning_effort`, and
5 rpm throttle:

| Artifact | n | p50 | **p95** | mean | max |
|---|---|---|---|---|---|
| `teacher_token_sample.meta.json` | 60 | 0.507 | **0.790** | 0.546 | 1.367 |
| `predictions_teacher_120b_test.meta.json` | 302 | 0.586 | **8.024** | 2.608 | — |

The p50s agree within 14%; the p95s differ by an order of magnitude. Bootstrapping the
60-sample p95 gives **[0.689, 1.054]** — the long run's 8.024 sits **7.6× above that upper
bound**, so this is categorically not sampling noise. Solving the long run's mean against its
median implies ≈15% of its calls took ~14 s.

`docs/ROADMAP.md` originally attributed this to client-side rate-limit stalling folded into
the timer. **That hypothesis was wrong and is retracted.** Reading `scripts/run_inference.py`
disproves it: `t0` is set *after* the throttle sleep, and the retry backoff lives in the
`except` branch so every attempt resets it. Only successful attempts append to `latencies`.
Throttle and backoff were already correctly excluded, exactly as the docstring claimed.

The tail is real latency that really happened — episodic congestion on shared free-tier
capacity during a ~77-minute run, absent from a ~12-minute one. A contributing cause is that
the 302-record run predates the `latencies_s` field, so its percentiles came from whichever
resumed segment ran last rather than the pooled distribution; the per-call vector was never
stored, so it cannot be audited after the fact.

**Consequence:** if the clean teacher p95 is ~0.8 s, G4's target is **≤ 0.16 s**, not the
≤ 1.60 s previously assumed — roughly ten times harder, requiring ~660 output tok/s
single-stream from a 1.5B. A re-measurement with the pooling fix in place is in flight.

### 2. "100% recall on the nine high-severity types" overstates the sample

The system (model + validators) misses **zero** of 232 high-severity gold instances. The
bootstrap reports `[1.0000, 1.0000]` — and that is a degenerate artifact, not a strong result:
with no misses in the sample, no resample has any either. Perfect measured recall on 29
instances is not evidence of perfect recall.

The Clopper-Pearson one-sided bound is the honest statement:

| | n | misses | recall | 95% lower bound |
|---|---|---|---|---|
| DRIVER_LICENSE | 15 | 0 | 1.0000 | **0.819** |
| SSN | 18 | 0 | 1.0000 | 0.847 |
| PASSWORD | 19 | 0 | 1.0000 | 0.854 |
| AADHAAR / PAN / API_KEY / BANK_ACCOUNT | 29 | 0 | 1.0000 | 0.902 |
| CREDIT_CARD | 41 | 0 | 1.0000 | 0.930 |
| **pooled** | **232** | **0** | **1.0000** | **0.987** |

So the defensible published claim is **"~99% recall"**, which is what the resume already said.
An earlier recommendation in this project to upgrade that bullet to "100%" was wrong and is
withdrawn: it quotes a point estimate the test set cannot carry. `run_eval --ci` now prints
the bound rather than the bare estimate.

### 3. dev is 79% contaminated and was about to be used for model selection

`make audit` found **150 of 189 dev records (79.4%) appear verbatim in `data/train.jsonl`**.
The data engine was seeded from dev (`SEEDS ?= data/gold/dev.jsonl`), and `forge/dedup.py` was
handed the *test* split to check leakage against — so `train.meta.json`'s
`removed_leakage: 11` counted only what it was asked to look for.

WP-3 was planned as a capacity sweep "on dev only". It would have selected a LoRA rank by
scoring memorised training text, and would have reliably preferred the most overfit
configuration. Fixed by `scripts/build_validation.py`: 533 clean records from seed 4242,
disjointness from train/dev/test asserted on the written bytes, and 339 high-severity
instances against test's 232 — so selection now has at least the resolution of the final
measurement.

### 4. The frozen test set holds 20 duplicate records

385 records, 365 unique texts; the 20 extras are byte-identical in both text and spans.
Effective n is 365, so intervals are marginally optimistic and duplicated records carry double
weight in micro-F1.

Measured impact is immaterial:

| | n=385 | n=365 | Δ |
|---|---|---|---|
| teacher micro-F1 | 0.9482 | 0.9482 | +0.0000 |
| student micro-F1 | 0.5750 | 0.5801 | +0.0050 |
| G1 ratio | 0.6064 | 0.6118 | +0.0054 |

All well inside the ±0.042 interval. **The freeze is kept.** Regenerating a test set after
seeing results is precisely what the contract forbids, and the defect changes no conclusion.
It is documented, pinned at exactly 20 by a test, and disclosed in the model card.

## Consequences

**G1 fails unambiguously.** The paired ratio is **0.6064 [0.5624, 0.6485]** against a 0.98
threshold. No interval rescues it; the gap is structural, not statistical.

**G4's target is probably much harder than planned**, pending the re-measurement.

**The important negative: the frozen test set has zero training leakage (0/385).** Every
published test measurement stands. Had this gone the other way, every number in the repo would
have been void.

**Test suites that regenerate data cannot catch data defects.** The ADR 0011 clock bug
survived a green suite for weeks because every test rebuilt the data the same wrong way. Both
defects here were found by reading committed bytes. `make audit` is now separate from
`make test` for that reason.

## What this does not fix

`PROTOCOL.md` §5 — human verification — remains unperformed. `audit_gold.py` checks what a
machine can check and its output says so explicitly. `contracts/pii_redaction_v2.yaml:114`
still describes the gold set as "a human-verified subset", which is not yet true; correcting a
frozen contract requires a v3 bump and is deliberately left as an open item rather than
silently edited.
