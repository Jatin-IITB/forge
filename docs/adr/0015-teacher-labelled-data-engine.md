# ADR 0015 — A two-track data engine, because the teacher is not uniformly better than construction

**Status:** proposed — predictions recorded, run in flight (2026-09-03)
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

- New module `forge/carriers.py` (pure, 21 unit tests) and two scripts. `run_data_engine.py`
  is left untouched so `make data-engine` still reproduces the historical v1/v2 path.
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

*To be completed against P1–P5 when generation finishes, including any prediction that
fails. ADR 0013 was rejected by its own experiment; that is the standard here.*
