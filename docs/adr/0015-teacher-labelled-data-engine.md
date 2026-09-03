# ADR 0015 — A two-track data engine, because the teacher is not uniformly better than construction

**Status:** accepted for the method, **partial on delivery** — carriers and Track A complete,
Track B labelling rate-limited and still running; P1–P4 unresolved (2026-09-03)
**Date:** 2026-09-03
**Depends on:** ADR 0012 (validator layer), ADR 0013 (capacity × data jointly binding), ADR 0014 (measurement integrity), `reports/baseline_120b.md`
**Work package:** ROADMAP WP-2

## Context

ADR 0013 ended with a sentence that is either the project's central embarrassment or
its next experiment, depending on what happens here:

> Every one of the 837 training records is template-generated: 687 construction-verified,
> and 150 labelled by the **8B development teacher that was later replaced for being too
> weak**. Not a single training record has been touched by the 120B teacher that scores
> 0.9482. A project named for distillation has not yet distilled from its teacher.

That is the untouched lever. The two experiments that preceded it bracket the problem:
`run_002` (r=16, attention-only) **underfit** at loss 1.17; `run_003` (r=64 + MLP, 73.9M
params, *identical data*) **overfit** at loss 0.22, driving precision to 0.86 while recall
collapsed to 0.39 and span ratio halved to 0.46. Capacity and data are jointly binding, so
more capacity alone moves along a frontier rather than toward it.

The obvious move — "have the 120B teacher label everything" — is wrong, and one
measurement says why.

## The measurement that shapes the design

The teacher's predictions on the frozen test set are already committed
(`data/predictions_teacher_120b_test.jsonl`, n=385). Scoring them per type against gold,
separating an *exact* span match from a *same-label overlapping* one:

| type | gold n | exact | boundary only | missed | exact recall |
|---|---|---|---|---|---|
| `PERSON` | 175 | 174 | 1 | 0 | **0.994** |
| `USERNAME` | 42 | 42 | 0 | 0 | **1.000** |
| `AGE` | 18 | 18 | 0 | 0 | **1.000** |
| `STREET_ADDRESS` | 32 | 27 | 4 | 1 | **0.844** |
| `LOCATION` | 22 | 6 | 0 | **16** | **0.273** |
| `DRIVER_LICENSE` | 15 | 8 | 7 | 0 | 0.533 |
| `BANK_ACCOUNT` | 29 | 23 | 0 | 6 | 0.793 |
| `AADHAAR` | 29 | 24 | 1 | 4 | 0.828 |

Two things follow, and they point in opposite directions.

**1. On four of the five types WP-2 targets, the teacher is an excellent labeller.**
`PERSON` 0.994, `USERNAME` 1.000, `AGE` 1.000, `STREET_ADDRESS` 0.844 — it agrees with the
gold boundary convention almost everywhere. Distilling these is well-founded.

**2. On `LOCATION` the teacher misses 16 of 22.** ROADMAP WP-2 lists `LOCATION` as a Track B
type. It cannot be one. ADR 0012 established that "the verification gate can discard teacher
noise but cannot manufacture signal the teacher never produced"; a k=3 self-consistency vote
over three samples that each miss the city produces a confident, unanimous, *empty* label.
The failure is silent by construction.

**And on `LOCATION` the student is already better than the teacher.** Scoring both on the
same 385 records, the headroom a Track B type actually offers is teacher F1 minus student F1:

| type | student F1 | teacher F1 | headroom | Track |
|---|---|---|---|---|
| `STREET_ADDRESS` | 0.0923 | 0.8571 | **+0.7648** | B |
| `USERNAME` | 0.3881 | 1.0000 | **+0.6119** | B |
| `PERSON` | 0.5000 (110 FN) | 0.9915 | **+0.4915** | B |
| `AGE` | 0.6667 | 1.0000 | **+0.3333** | B |
| `LOCATION` | **0.7805** | **0.4138** | **−0.3667** | **A** |

*(Regenerate with `python scripts/analyse_teacher_types.py`; JSON in
`reports/teacher_type_analysis.json`.)*

`LOCATION` is the only model-owned type where distillation would be *regressive*: the student
already scores 0.7805 against a teacher that scores 0.4138, so teaching it the teacher's
labels would cost roughly 0.37 F1 on that type. Distillation is not uniformly an upgrade, and
this is the row that proves it.

And this is worse in training data than in evaluation. A teacher false negative that lands
in a training set is not a scoring error — it is **a labelled example of not detecting an
entity**. The student's diagnosed failure is precisely under-enumeration (span ratio
0.46–0.84, `PERSON` with 110 false negatives). Distilling the teacher's misses would teach
the exact defect this work exists to remove.

Separately, the boundary disagreements are not the teacher being sloppy. It splits
`"01/12, Banik Circle, Ballia"` into `STREET_ADDRESS "Banik Circle"` plus `LOCATION
"Ballia"`, which is a defensible reading — but `data/gold/PROTOCOL.md` §3 fixes the
convention ("a full mailing address → `STREET_ADDRESS`; a bare city used as context →
`LOCATION`"). We are scored against that convention. Adopting the teacher's would train
against our own contract.

**Third measurement, on the other axis.** Carrier shapes, counted as the span-masked
skeleton of each record:

| file | records | distinct shapes |
|---|---|---|
| `data/gold/test.jsonl` | 385 | 109 |
| `data/gold/val.jsonl` | 533 | 100 |
| `data/train_v2.jsonl` | 837 | 208 |

All of them draw on the same ~110 hand-written templates in `scripts/build_gold.py`.
A template that always places the address in the same syntactic slot cannot teach where an
address ends, which is the most plausible explanation for `STREET_ADDRESS` F1 = 0.0923.

## Decision

**Spend teacher tokens on the two things the teacher is actually better at than we are:
writing varied text, and finding entities it was not told about. Keep construction for
labels wherever construction has them.**

Concretely, `data/train_v3.jsonl` is built in two stages.

**Stage 1 — carrier generation (`scripts/generate_carriers.py`).** The teacher writes
*skeletons* with `{{TYPE}}` placeholders across 20 registers (support tickets, chat threads,
log excerpts, clinical notes, voicemail transcripts, form dumps, …) — never PII values,
never labels. ~10 shapes per call makes this the cheap stage. Shapes are validated
(`forge/carriers.validate_shape`) and any shape colliding with an existing split's shape is
dropped, which is stricter than the contract's carrier-sentence rule and free.

**Stage 2 — filling and labelling (`scripts/build_train_v3.py`).** Shapes are filled with
Faker values from the *same* `PIIValueGenerator` the frozen gold set uses, under seed 7717.
Holding the value distribution fixed is deliberate: carrier text is the variable under
study, and ADR 0013 showed what happens when an experiment moves several at once. Then:

| Track | Types | Label source |
|---|---|---|
| **A — construction** | all 19, weighted toward validator-owned | the fill (exact by construction) |
| **B — distillation** | `PERSON`, `STREET_ADDRESS`, `USERNAME`, `AGE` focus | 120B teacher, k=3, majority vote |

### The construction anchor — the part that is new

A Track B record passes `forge/verify.py` (schema + k=3 self-consistency) and then a second
gate that `run_data_engine.py` does not have: its labels are compared against the entities
we *know* are in the text, because we injected them. The teacher never sees them.

1. **A model-owned injected span must be matched exactly, or the record is rejected.**
   Rejected, not repaired — repairing would hide the rate, and the rate is the finding.
   Misses and boundary disagreements are counted separately.
2. **Validator-owned spans are taken from construction, not from the teacher.** On the nine
   high-severity types the deterministic layer now scores 1.0000 on **both** recall and
   precision (commit `4d51d66` took its 55 false positives to 0 without moving recall),
   against a teacher averaging 0.87 recall and failing six of the nine — `DRIVER_LICENSE`
   0.533. Where we hold exact offsets, using the teacher's guess would be choosing the worse
   label deliberately; rejecting the record would discard it over a type the model is not
   being asked to own. Note the dependency runs through `forge.schema.HIGH_SEVERITY`, the
   type roster, not through `forge/validators.py` detection behaviour — the engine never
   calls a validator, so improvements to the validator layer cannot silently change what
   this engine labels.
3. **Teacher spans over text we did not inject are kept.** These are entities occurring
   naturally in the teacher's own prose — the fuzzy, context-dependent cases construction
   cannot manufacture. **This is the only genuinely distilled label content in the corpus,**
   and it rests on the k=3 consensus alone. Stated as a limitation rather than buried.

`LOCATION` is therefore removed from the Track B roster and served by Track A, against
ROADMAP WP-2's listing. The roadmap's own footnote — "the teacher itself scores 0.41" —
already contained the reason; it just was not carried through to the design.

### Two deviations from the brief, recorded rather than silently taken

**High-severity types are down-weighted, not dropped.** "The model's job is the complement
of the validators" (ADR 0012) is right about where capacity should go, and the corpus
reflects it. But removing those nine types from the *label set* would mean the student
stops emitting them, and G1 is measured model-only against gold that contains them. That
changes what G1 measures, which is a contract decision and not this ADR's to make.

**`LOCATION` moves from Track B to Track A**, for the measured reason above.

## Options considered and rejected

**A. Label everything with the teacher.** Rejected: inherits the `LOCATION` blind spot and
the boundary convention, in a corpus where a false negative is a training signal.

**B. Seed carriers from a public corpus (Enron, ai4privacy).** `ai4privacy/pii-masking-200k`
is already rejected by contract v2 and ADR 0003 as academic-use-only. Any other corpus needs
a licence review *before* generation, and teacher-generated text clears ADR 0003
unconditionally — an open-weight Apache-2.0 model's output on our own prompts, regenerable
by a stranger from any host serving that checkpoint. There is no reason to take the risk.

**C. Seed from a gold split, as the previous engine did.** This is the defect ADR 0014
found: `SEEDS ?= data/gold/dev.jsonl` plus `forge/dedup.py` handed only the *test* split,
leaving `dev` 79.4% contaminated and undetected because nothing was asked to look. Carrier
text here comes from no evaluation split at all, so disjointness is structural rather than
filtered-for — and is still asserted, on the written bytes, against all four of
`train.jsonl`, `dev.jsonl`, `val.jsonl`, `test.jsonl`.

**D. More of the same construction data.** ADR 0009 already ran that experiment: 150 → 837
records bought +0.03 F1. Rejected by measurement.

## Two things found while building, before any data was generated

**The design was nearly derived from a prompt the engine would not have used.** The table
above comes from `data/predictions_teacher_120b_test.jsonl`, which `run_inference.py`
produced with the **plain** system prompt. But `run_data_engine.py` — which the new
labeller was copied from — labels with `TEACHER_SYSTEM_PROMPT`, a different prompt that adds
*"Be thorough — missing a PII entity is worse than a false positive"* and asks for a
rationale per span. That instruction targets recall, which is the exact axis `LOCATION`
fails on, so the entire Track A/B split may have been derived from a distribution the
engine was not going to generate from.

`scripts/probe_teacher_prompt.py` settled it on 16 `val` records (never `test` — choosing a
configuration by looking at the frozen split is what the gate discipline exists to prevent),
both prompts, same records, temperature 0:

| type | n | plain recall | "be thorough" recall |
|---|---|---|---|
| `LOCATION` | 10 | 0.100 | **0.100** |
| `PERSON` | 8 | 1.000 | 1.000 |
| `AGE` | 2 | 1.000 | 1.000 |
| `USERNAME` | 1 | 1.000 | 1.000 |

15 of 16 records returned **byte-identical span sets**. The blind spot is a property of the
teacher, not of how it is asked: telling it to be thorough moved `LOCATION` by nothing, and
0.100 on `val` is *worse* than the 0.273 on `test` that motivated the routing. P2's premise
holds, and `LOCATION` stays in Track A.

The single differing record is instructive. The whole micro-F1 gap (0.7541 → 0.8525) sits on
`AADHAAR`, `PAN` and `PASSPORT`, where the plain prompt included the label prefix — `"PAN
MSRPE0506C"` for gold `"MSRPE0506C"` — and the thorough prompt did not. **Those are exactly
the types the construction anchor overrides.** So the prompt choice cannot affect a single
label this engine keeps, and the engine uses the plain prompt: identical where it matters,
matched to the measurement, and cheaper.

**Prompting alone does not keep literal PII out of carriers.** The first 21 generated shapes
included ten chat transcripts with literal speaker names — `"Alice: Could you send the
report to {{PERSON}}?"` — despite an explicit instruction not to write PII values. Every
Track A record built from that shape would contain `Alice` as an **unlabelled** `PERSON`:
a training example asserting that a name is not PII, which is precisely the
under-enumeration the student already fails at. Track B would have partly self-corrected
(the teacher finds the name, the anchor keeps it as a discovery), so the defect would have
been *invisible in Track B metrics while silently poisoning Track A*.

Carriers are now screened rather than merely requested: literal emails, URLs, IPs and long
digit runs by regex, and given names against the ~1,250-name Faker corpus that fills the
placeholders — same provenance as the injected values, so the screen and the fill cannot
disagree. Sentence-initial matches are exempt so that `Will`, `May` and `Grace` as ordinary
words do not cost good carriers. The screen refuses to run if the name set comes back small,
because a screen that silently passes everything is worse than no screen — the first version
of it returned zero names and cleared all ten bad shapes.

## Predictions, recorded before the run

Per ADR 0013's process lesson — a single-number prediction would have read as a clean
success and shipped a recall collapse undetected — these are stated as a conjunction, and
**P3 is the one that decides whether this ADR was worth doing.**

**P1 — Track B accept rate ≥ 0.60.** Below that, the teacher cannot reliably label prose it
wrote itself at k=3, and Track B is not viable at this cost.

**P2 — `LOCATION` is the most-missed model-owned type at the anchor, with a miss rate
> 0.25.** This transfers the frozen-test finding (exact recall 0.273, n=22) to a much larger
sample. If `LOCATION` misses are *rare* on teacher-written carriers, then the test-set
weakness is an artifact of the template pool's bare-city construction rather than a teacher
blind spot — a different and more interesting finding, which would partly vindicate the
roadmap's original Track B roster.

**P3 — teacher discoveries in its own prose ≥ 0.15 spans per accepted Track B record.**
Everything else in this corpus is construction. If this number is near zero, Track B has
bought carrier diversity and a verification signal but **no distilled labels**, and the
honest conclusion is that the engine is construction with extra steps and a teacher bill.
That conclusion will be published in the outcome section if the number says so.

**P4 — `STREET_ADDRESS` boundary disagreement ≥ 0.125**, its rate on the frozen test set
(4/32). Teacher-written prose places addresses in less canonical positions than the
templates do, so the rate should rise, not fall.

**P5 — the mechanical targets:** ≥ 300 distinct carrier shapes, ≥ 40% Track B share,
leakage exactly 0 against all four splits, verified on the written bytes.

Explicitly **not** predicted here: any F1 improvement. That is WP-3's experiment, and
claiming it now would be the post-hoc gate-setting `SUCCESS.md` forbids.

## Consequences

- New modules `forge/carriers.py` (pure, 31 unit tests) and `forge/teacher_client.py`
  (throttle + retry, 9 unit tests), plus three scripts. `run_data_engine.py` is left
  untouched so `make data-engine` still reproduces the historical v1/v2 path.
- `forge/teacher_client.py` exists because a 5 rpm client with no retry was measured
  **dropping half its calls** against this tier while another client was active. A dropped
  call is a missing sample, not a visible error, so every teacher-facing script now shares
  one throttle-and-retry path with the latency timer outside both sleeps (ADR 0014 §4).
- The corpus is regenerable end-to-end from two commands and three seeds, by a stranger, with
  no private credential — the ADR 0003 litmus, applied to the training data for the first time.
- Cost: ~600 tok/record for carriers plus k=3 labelling on Track B, against a 5 req/min,
  1M tok/day free tier. That is days of background generation, so the engine caches teacher
  results per record text and `--resume` costs nothing but re-reading a file.
- The data card reports Track A and Track B separately and never averages them. A reader who
  wants to know how much of this corpus the teacher actually labelled can find out.

## What this does not fix

- Teacher discoveries in prose are accepted on k=3 consensus with no construction anchor
  behind them, because none exists. They carry the teacher's precision behaviour, including
  its habit of labelling context place names the protocol leaves unlabelled.
- Carrier realism is unmeasured. `PROTOCOL.md` §5's human pass remains unperformed here as
  elsewhere; nothing in this ADR claims the text reads naturally, only that it is diverse by
  a counted metric.
- Whether any of this moves F1 is unknown and deliberately untested until WP-3.

---

## Outcome

**Interim, 2026-09-03, at 205 labelled Track B records (8% of the planned 2,618).
Generation continues. P1 and P3 have failed; P2 and P4 are confirmed. The finding that
matters most was not predicted at all.**

### Scoreboard

| | prediction | measured | |
|---|---|---|---|
| **P1** | Track B accept rate ≥ 0.60 | **0.424** (87/205) | **failed** |
| **P2** | `LOCATION` most-missed model-owned type, miss rate > 0.25 | **0.934** (85/91), by far the most-missed | confirmed |
| **P3** | teacher discoveries ≥ 0.15 spans/accepted record | **0.0000** (0 spans over 87 records) | **failed** |
| **P4** | `STREET_ADDRESS` boundary disagreement ≥ 0.125 | **0.153** (11/72) | confirmed |
| **P5** | ≥300 shapes / leakage 0 / ≥40% Track B | **456** / **0,0,0,0** / **5.2%** | 2 of 3 |

### P3 failed, and this ADR said what that means

The prediction was written as the one that decides whether the work was worth doing, with
the consequence stated in advance: *"If this number is near zero, Track B has bought carrier
diversity and a verification signal but no distilled labels, and the honest conclusion is
that the engine is construction with extra steps and a teacher bill."*

It is not near zero. It **is** zero. Across 87 accepted records the teacher found not one
entity that construction had not already injected. On this corpus, distillation contributed
no labels.

Part of that is self-inflicted and worth naming, because attributing it entirely to the
teacher would be wrong. The literal-PII screen added earlier the same day strips names,
emails, IPs and digit runs out of carrier prose so that Track A labels are complete — and
that is precisely the material Track B existed to discover. The screen and P3 are in direct
tension: with literal names left in, the teacher would have "discovered" them, but those
discoveries would only have been repairing a defect the screen now prevents. A carrier whose
prose is clean by construction has nothing left in it to find.

### The finding that was not predicted: construction is wrong ~9% of the time

`anchor.missing` was designed to measure the teacher. Read the other way it measures the
carriers, and that reading is the more valuable one.

Construction labels are exact **by offset** — `fill` accumulates them, so `text[start:end]`
always equals the span. Nothing checks they are correct **by semantics**, and they are
frequently not:

| construction's label | the prose it sits in |
|---|---|
| `AGE` = `46` | "please settle it within **46** days" |
| `AGE` = `27` | "last seen at the sorting facility at **27** hours" |
| `AGE` = `25` | "amount $**25**" |
| `AGE` = `57` | "status **57**" |
| `DATE_OF_BIRTH` = `20/12/1961` | "the claim filed on **20/12/1961**" |
| `DATE_OF_BIRTH` = `1980-02-17` | "the effective date is **1980-02-17**" |

None is the entity its label claims. Training on them teaches that any small integer is an
age and any date is a date of birth — a precision failure manufactured deliberately, in a
corpus whose entire purpose is to fix a precision/recall frontier.

The teacher catches these because it labels the text without knowing what was injected, so
it silently declines them. It declines `AGE` at **0.593** and `DATE_OF_BIRTH` at **0.689**,
having scored **1.000 exact recall on both** against the frozen gold set — so on these two
types the disagreement is evidence about the carrier, not about the teacher. The asymmetry
is the point: on `LOCATION` the same signal means the opposite, because there the teacher is
the one that is wrong (0.273 on test, 0.100 on val, 0.934 miss here). Which side is on trial
depends on which side has an independent measurement behind it.

Track A and Track B are filled from the same 456 shapes, so a disagreement seen on a Track B
instance transfers to every Track A record built from that shape. `scripts/audit_carriers.py`
counts it: **152 of 1,638 Track A records (9.3%)** are built from a (shape, type) pair the
teacher declines at least half the time. Only 35% of shapes have been audited so far, so the
true figure is roughly **27%**. Those records are in `data/train_v3.jsonl` today, unflagged.

**So the teacher's value in this engine is as a critic of construction, not as a labeller.**
That is not what WP-2 was designed to buy, and it is worth more than what it was: P3 says
distillation added no labels, while the audit says a quarter of the construction corpus may
carry semantically wrong ones. A verification signal that finds a defect in the other track
is a better outcome than the one predicted, and it was only observable because the two tracks
share carriers and the anchor compares them.

Not yet done, and not to be claimed until it is: filtering or repairing the affected Track A
records, and constraining carrier generation so `{{AGE}}` and `{{DATE_OF_BIRTH}}` cannot land
in slots the prose reads as a duration, an amount, a status code or a filing date.

### P1 failed for the same reason

The k=3 self-consistency gate accepted **205 of 205** — the teacher agrees with itself.
Every rejection came from the construction anchor, at 0.424 against a 0.60 prediction. But
having read the examples above, a meaningful share of those rejections are the anchor
working correctly on records where **construction** was wrong, not the teacher. The accept
rate is therefore not a clean measure of teacher quality, and P1 as written conflated the
two. It is recorded as failed because that is what it predicted and what was measured, but
the number should not be quoted as "the teacher is unreliable 58% of the time."

### P5 — mechanical targets: two met, one blocked

| target | result |
|---|---|
| ≥ 300 distinct carrier shapes | **456** — met |
| leakage 0 against all four splits, on written bytes | **0 / 0 / 0 / 0** (`dev`, `val`, `test`, `train`; also 0 against `train_v2`) — met |
| ≥ 40% Track B share | **not met — 0.1%**, and the reason is below |

### Why Track B is not there yet

The binding constraint was misidentified in the Consequences section above, which budgeted
in tokens: *"~600 tok/record for carriers plus k=3 labelling, against a 5 req/min, 1M
tok/day free tier."* Tokens were never the limit. Roughly 110k of the day's 1M were spent.

The tier meters **requests against an hourly allowance** and answers exhaustion with
`retry-after: 3600`. At k=3 this engine spends three requests per Track B record — 7,854
for the planned 2,618 — and an hour of cooldown buys back only a small burst. That is a
different economics from the one this ADR assumed, and it is the honest reason the corpus
is partial rather than any property of the method.

The design's response is the obvious one and is **not yet implemented**: under a
request-metered tier, labelling one record per request is the wrong unit. Batching several
records into a single completion would cut request count by that factor at roughly constant
token cost. It is deliberately not being bolted on mid-run — a multi-record prompt is a
different distribution from the single-record prompt every measurement in this ADR was
taken under, and adopting it without re-measuring would repeat exactly the mismatch caught
in "Two things found while building" above.

`make train-v3` resumes from the cache; `make train-v3-card` regenerates the corpus and card
mid-run, and `make carrier-audit` re-runs the audit above as more shapes are covered. The
numbers here are at 205 records and will move; the two failed predictions will not, since a
discovery rate of exactly zero does not become positive with more of the same.
