# ADR 0022 — Accept a gated external corpus, and what it costs

**Status:** accepted (2026-09-05)
**Date:** 2026-09-05
**Supersedes:** the blanket rejection of gated corpora implied by contract v2's `rejected_sources`
**Depends on:** `scripts/load_bigcode_pii.py`, `adr/0003` (independence)

## Context

Every Forge number to date is measured on synthetic data. The gold set is Faker
values injected into templates, and the measured consequences are specific:

- **Only 2 of 29 AADHAAR values satisfy Verhoeff** (7%). Real Aadhaar is 100%
  checksum-valid, so the validator's checksum path is essentially never
  exercised against what it would actually meet.
- 385 records span 109 carrier shapes, all one- or two-sentence templates.

External validity is the project's largest untested assumption, and the honest
writeup has said so since `HONEST_ASSESSMENT.md` was written.

`bigcode/bigcode-pii-dataset` is real PII — names, emails, IP addresses, keys,
passwords, usernames — annotated by crowd workers over 12,099 code snippets
drawn from permissively licensed GitHub repositories.

## Decision

**Use it for evaluation. Publish the numbers, never the data.**

This reverses the posture that produced contract v2's rejection of
`ai4privacy/pii-masking-200k`. That rejection was correct on its own terms and
remains so; what changes is the recognition that a project with *zero* real-data
measurement has a worse problem than one with a reproducibility caveat.

## What this costs, stated plainly

### 1. `adr/0003`'s litmus test is weakened, not preserved

> *"If every private credential were revoked tomorrow, could a stranger clone
> this repo and rebuild the model end-to-end?"*

For anything touching this corpus, the answer becomes **no**: a reader must
create a HuggingFace account, share contact details, complete a Google Form and
accept Terms of Use first. The synthetic pipeline still satisfies the test end to
end; the external evaluation does not. Both facts are reported together or
neither is.

### 2. Term 2 is a hard constraint, not a policy

> *"You agree that you will not share the PII dataset or any modified versions
> for whatever purpose."*

`data/external/` is gitignored. The converted corpus, any split of it, any
sample quoted in a report, and any predictions file derived from it stay local.
This is not a preference that can be revisited by editing this ADR — it is a
term accepted at the gate, and the repository is public.

Term 1 permits *"training or evaluating models for PII removal"*, which is
exactly this use, so **measured numbers are publishable**.

### 3. It is code, and code is out of domain

Contract v2 declares code out-of-domain, and `forge/ood.py` refuses
`code_python`, `code_sql`, `code_json`, `code_html` and `code_regex` — scoring
21/21 on the OOD probes with zero false refusals across 1107 in-domain records.

So the shipped system would **decline every file in this corpus by design**. Any
evaluation here must disable the OOD gate, and the result therefore describes
*the model on input the product refuses*, not the product. Reported otherwise it
would be straightforwardly misleading.

That tension is itself informative: BigCode exists *because* credentials leak
into source files, which is precisely the high-severity case the validator layer
handles best. Scoping Forge to prose has a real cost, and this is the evidence
for it.

### 4. It validates six of nineteen types, and two of nine high-severity

| BigCode | Forge | |
|---|---|---|
| EMAIL | `EMAIL` | |
| NAME | `PERSON` | |
| USERNAME | `USERNAME` | |
| IP_ADDRESS | `IP_ADDRESS` | |
| KEY | `API_KEY` | **high-severity** |
| PASSWORD | `PASSWORD` | **high-severity** |

**Absent: AADHAAR, PAN, SSN, CREDIT_CARD, BANK_ACCOUNT, PASSPORT,
DRIVER_LICENSE.** Seven of the nine types carrying the 0.99 recall floor get no
external validation from this corpus — and no public corpus can supply it,
because publishing real values of those *is* the leak the tool prevents. That
limitation is permanent and must stay in the writeup.

## Scoring decisions

`*_EXAMPLE`, `*_LICENSE`, `AMBIGUOUS` and `ID` are excluded from gold. The first
three mark strings that *look* like PII but must not be redacted — placeholder
addresses, names in licence headers, annotator non-decisions. Counting them as
gold would reward exactly the over-redaction the validator layer was tuned
against. They are retained in the converted output under `non_targets` so a
later run can score them as negatives, which is the more interesting experiment:
**a false positive on a licence header is a precision failure with a real cost.**

Offsets are verified against the text and dropped on mismatch, the same rule
`scripts/audit_gold.py` applies to the frozen set. Overlapping spans are dropped
for the same reason they are an error in gold: exact-match scoring is ill-defined
when one prediction can satisfy two references.

## Consequences

- The frozen synthetic test set remains the **contract** measurement. Gate
  verdicts continue to be quoted from it. BigCode is a *supplementary external
  check*, reported separately and never merged into the gate table.
- `data/external/` is gitignored and stays that way.
- Any report citing BigCode numbers must also state: the OOD gate was disabled,
  the domain is code rather than prose, and the corpus is unobtainable without
  accepting a gate.
- The seven unvalidated high-severity types remain the honest headline
  limitation of this project.
