# ADR 0013 — run_002 diagnosis: the bottleneck is adapter capacity, not training data

**Status:** REJECTED by its own experiment (2026-08-16) — see Outcome
**Date:** 2026-08-15
**Depends on:** run_002 evaluation; supersedes the working hypothesis in ADR 0009.

## Context

ADR 0009 diagnosed run_001's F1 = 0.52 as a **training-data problem**: severe per-type
imbalance (AADHAAR 4 training examples vs 29 in test) plus multi-span blindness. The
intervention was targeted augmentation — 150 → 837 records, 5.6× more data, deliberately
weighted toward the weak types and multi-entity sentences.

`HONEST_ASSESSMENT.md` pre-registered what a null result would mean:

> If run_002 does not substantially beat 0.52, the error-driven loop's central premise —
> that targeted augmentation fixes measured failures — is unsupported, and the honest
> conclusion is that the imbalance diagnosis was wrong.

**run_002 scored F1 ≈ 0.55.** A 5.6× increase in targeted training data bought roughly
+0.03. That is the pre-registered null result, so the ADR 0009 hypothesis is rejected
rather than reinterpreted.

## Evidence

Four measurements, each ruling out a candidate cause.

**1. The model converged; it is not undertrained.**

| step | 10 | 30 | 50 | 70 | 90 | 110 | 130 | 150 |
|---|---|---|---|---|---|---|---|---|
| loss | 2.33 | 1.86 | 1.29 | 1.16 | 1.17 | 1.17 | 1.17 | 1.18 |

Loss falls until step ~70 of 159 and is then flat for the remaining 80 steps. More epochs
on this data cannot help — the optimiser has stopped finding improvements at a loss (≈1.17)
that is high for a structured extraction task.

**2. It is not a generalization or template-memorization problem.**

Splitting the test set by whether its sentence shape appeared in training:

| test subset | n | F1 |
|---|---|---|
| template shape **seen** in training | 68 | 0.5258 |
| template shape **unseen** | 27 | 0.6087 |

The student performs *better* on unseen shapes. Had the model been memorizing templates,
this would be reversed. 52% of test shapes never appear in training and that costs nothing,
so adding template diversity is not the lever.

**3. It is not an output-format or copying failure.**

0 of 116 predicted span texts were absent from the source — the model copies exactly and
emits parseable JSON (schema validity was 100% in run_001). The supervision format works.

**4. The failure is under-detection plus boundary error.**

- Predicted 116 spans where gold has 147 — a **0.79 span ratio**. The model still emits too
  few entities per record, exactly as in run_001, *despite* augmentation explicitly built to
  fix that.
- exact-match F1 0.55 vs **partial-overlap F1 0.69**. Roughly 14 points sit in spans that
  land in the right region with the wrong extent.

A model given 5.6× more examples of multi-entity sentences that still under-predicts by the
same ratio is not short of examples. It is short of the capacity to represent the task.

## The overlooked variable

Both runs used the same adapter configuration:

```python
LORA_DEFAULTS = {"r": 16, "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]}
```

Rank 16, **attention projections only**. The MLP blocks — `gate_proj`, `up_proj`,
`down_proj`, which hold the majority of a transformer's parameters and most of its
factual/format machinery — are untouched. Every experiment so far varied the data and held
capacity fixed, so capacity was never actually tested.

This is the plainest explanation consistent with all four measurements: an adapter that
converges quickly to a mediocre loss and under-produces output is capacity-bound.

## Decision

Run **run_003 as a capacity experiment on unchanged data**, so the variable is isolated:

| | run_002 | run_003 |
|---|---|---|
| LoRA rank | 16 | 64 |
| alpha | 32 | 128 |
| target modules | 4 attention | 7 (attention + MLP) |
| training data | train_v2 (837) | **train_v2 (837), identical** |
| epochs | 3 | 3 |

Holding the data fixed is the point. If F1 moves materially, capacity was the bottleneck and
ADR 0009's data work was solving the wrong problem. If it does not, capacity is excluded too,
and the next suspect is the 1.5B base model itself — at which point the honest options are a
3B base or accepting a measured shortfall against G1.

### Predictions, recorded before the run

- Loss should fall **below 1.0** and keep decreasing past step 70. If it plateaus at ~1.17
  again, the hypothesis is wrong and rank was not the constraint.
- Span ratio should move toward 1.0 from 0.79.
- Both must hold. A loss improvement without a span-ratio improvement would mean the adapter
  is fitting the data better without learning to enumerate entities, which would point at
  the task formulation instead.

## Consequences

- ADR 0009 is **not withdrawn**: its data work is independently sound (per-type coverage is
  genuinely better, and the construction-verified pipeline is reusable). But its causal claim
  is rejected, and `NORTH_STAR.md` must show the augmentation result as a null, not a win.
- The error-driven loop's value proposition is dented. Spending effort on data that a
  capacity ceiling then discards is precisely the failure mode the loop is meant to prevent,
  and the loop did not catch it because no experiment ever varied capacity.
- **Process lesson, generalized:** every future intervention states which variable it
  isolates. Two runs differing in data volume, per-type balance, and multi-span emphasis at
  once cannot attribute a result to any of them — and run_002 could not distinguish "the data
  fix failed" from "the data fix was masked."
- Cost: one 4-hour training run to test a one-line configuration change.

---

## Outcome (2026-08-16) — hypothesis rejected

run_003 ran to step 100 of 159 (1.9 epochs) before being stopped. **The two predictions
split, and the ADR required both.**

### Prediction 1 — met, emphatically

Loss had to break below 1.0 and keep falling past step 70, where run_002 flatlined at 1.16.
On identical data:

| step | 10 | 20 | 30 | 50 | 70 | 90 |
|---|---|---|---|---|---|---|
| run_002 (r=16, attn) | 2.3323 | 2.1307 | 1.8557 | 1.2946 | 1.1572 | 1.1723 |
| run_003 (r=64, +MLP) | 1.9736 | **0.4063** | **0.2803** | 0.2729 | 0.2173 | 0.2476 |

Roughly **5× lower**. Capacity was unambiguously constraining how well the adapter could
fit the data.

### Prediction 2 — failed, and in the opposite direction

Span ratio had to move from 0.79 toward 1.0. Measured on the same 150 test records:

| | F1 | precision | recall | span ratio |
|---|---|---|---|---|
| run_002 (r=16) | **0.5631** | 0.6182 | 0.5171 | 0.84 |
| run_003 (r=64+MLP) | 0.5379 | **0.8583** | 0.3916 | **0.46** |

The ratio **halved**. Precision rose sharply (+0.24) while recall collapsed (−0.13), and net
F1 got slightly *worse*. The model became confident and conservative: it emits fewer spans,
and the ones it emits are usually right.

### Interpretation — written before the result, applied unchanged

This ADR stated in advance:

> A loss improvement without a span-ratio improvement would mean the adapter is fitting the
> data better without learning to enumerate entities, which would point at the task
> formulation instead.

That is exactly what happened, and it is the textbook signature of **overfitting**: 73.9M
trainable parameters against 837 training records. Training loss of 0.22 measures
memorisation of those 837, not competence on the task.

### What the two runs jointly establish

- **run_002 underfit** — capacity too low, loss stuck at 1.17, mediocre precision *and* recall.
- **run_003 overfit** — capacity ample, loss 0.22, precision good but recall collapsed.

Neither capacity nor data volume is independently the bottleneck; they are **jointly
binding**. Adding capacity without data trades recall for precision at roughly constant F1,
which is movement along a frontier rather than progress toward it.

The remaining lever is therefore data that is **more diverse**, not merely more numerous —
and specifically data the project has never had. Every one of the 837 training records is
template-generated: 687 construction-verified, and 150 labelled by the **8B development
teacher that was later replaced for being too weak**. Not a single training record has been
touched by the 120B teacher that scores 0.9482. A project named for distillation has not
yet distilled from its teacher.

### Decision

run_003 stopped at step 100; `checkpoint-100` retained as the evidence for this entry.
Next experiment (ADR 0014) holds capacity near run_003's setting and changes the **data**:
diverse carrier text generated by the 120B teacher with construction-verified span
injection, so the text is varied *and* the labels remain exact.

### Process note

Both predictions were recorded before the run, and only one held. Had this ADR asked for a
single number — "does loss improve?" — the result would have read as a clean success and the
recall collapse would have shipped undetected into the next round. The conjunction is what
made the experiment informative.
