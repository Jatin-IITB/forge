# ADR 0011 — The frozen gold set was not actually frozen (clock-dependent generation)

**Status:** accepted
**Date:** 2026-08-15
**Severity:** CRITICAL — this defect invalidated the project's central reproducibility claim.

## Context

`tests/test_gold_builder.py::test_reproducibility` rebuilds the gold set from
seed 42 and asserts the output matches the committed `data/gold/test.jsonl`.
It had been failing, and was initially misdiagnosed as Faker version drift.

Bisecting seven Faker releases (37.12 through 40.36) found **no version** that
reproduced the committed file. The real cause was in our own code:

```python
d = self.fake_us.date_of_birth(minimum_age=18, maximum_age=80)
```

Faker's `date_of_birth()` derives its sampling window from **`datetime.now()`**.
The seed fixes *where in the window* the sample falls; the window itself slides
one day per day. So the generator was:

- **reproducible within a single day** — which is why it passed when written, and
- **silently different every day after** — a uniform +1 day drift per elapsed day.

The gold set was built 2026-08-07. On 2026-08-15 the rebuild differed in exactly
25 of 385 records, every difference an exact **+8 day** shift on a
`DATE_OF_BIRTH` value. The drift was measured, not assumed: an earlier run on
2026-08-11 showed exactly +4.

### Why this is severe

`SUCCESS.md` treats the frozen test set as the foundation of every claim, and
the contract's data policy promises the set is "fully reproducible from a fixed
seed via `make gold`." That promise was false. Concretely:

- A stranger cloning the repo would rebuild a *different* test set and get
  different scores from ours — the exact failure `adr/0003`'s litmus test exists
  to prevent.
- Any `make gold` re-run would have silently mutated the frozen asset that
  run_001's F1 = 0.52 was measured against, making run-to-run comparisons
  meaningless without anyone noticing.
- The failing test was correctly reporting a real defect. It had been treated as
  environmental noise — a reminder that a red test on a foundational invariant
  deserves diagnosis before dismissal.

## Decision

Anchor date-of-birth generation to a fixed epoch — the date the gold set was
originally built — while keeping Faker's own call path:

```python
DOB_EPOCH = datetime.date(2026, 8, 7)
...
d = self.fake_us.date_of_birth(minimum_age=18, maximum_age=80)
d -= datetime.date.today() - DOB_EPOCH
```

Two rejected alternatives, and why:

1. **Substitute `date_between_dates()` with pinned bounds.** Semantically clean,
   but it consumes the RNG differently from `date_of_birth()`, which shifts every
   subsequent draw and changes 138 records. It would have forced regeneration of
   the frozen set.
2. **Regenerate the gold set with a corrected builder.** Rejected on two grounds:
   the teacher baseline was mid-run against the committed file (changing it would
   have invalidated 100+ scored records), and re-freezing a test set after a model
   has been scored against it is precisely the practice `SUCCESS.md` forbids.

The chosen fix reproduces the committed file **bit-for-bit** — `git status` on
`data/gold/` is clean — so no measurement is invalidated and the freeze holds.

## Consequences

- `test_reproducibility` passes; the full suite is 142/142 green.
- A regression test (`test_gold_is_not_clock_dependent`) asserts the builder
  produces identical output under a simulated future date. The original bug
  passed every test on the day it was written; only a clock-shifted test catches
  that class of defect.
- **Generalized rule:** no generator in this repo may read the wall clock.
  `datetime.now()`, `date.today()`, `time.time()`, and `random` without an
  explicit seed are all disqualifying in any code path that produces committed
  data. Where a date is genuinely needed, it comes from a named constant.
- `DOB_EPOCH` is now load-bearing and documented as such. Changing it changes the
  frozen gold set and therefore requires a new contract version, exactly like
  changing a gate.
