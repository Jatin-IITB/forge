"""Carrier shapes — the skeleton half of the WP-2 data engine (ADR 0015).

A **carrier shape** is a sentence or short document with typed placeholders where
PII values go::

    "Ticket {{PERSON}} reports card {{CREDIT_CARD}} declined at {{LOCATION}}."

Filling a shape with generated values produces text whose spans are **exact by
construction** — offsets come from the fill, never from hand-labelling or from a
model counting characters. That property is what `scripts/build_gold.py` already
relies on; this module generalises it so the shapes can come from the *teacher*
instead of from a hand-written list.

Why that matters: `data/gold/test.jsonl` draws on 109 distinct shapes and
`data/train_v2.jsonl` on 208, all from the same hand-written pool in
`build_gold.py`. A fixed pool always places an entity in the same syntactic slot,
which is why `STREET_ADDRESS` (F1 0.0923) never learned where an address ends.
Teacher-written shapes cost ~60 tokens each and are project-owned synthetic text,
so they clear ADR 0003 without a licence review.

This module is deliberately pure: no network, no torch, no Faker import at module
scope. Everything here is checkable by `pytest` in milliseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from forge.schema import HIGH_SEVERITY, PIIRecord, PIISpan, PIIType

PLACEHOLDER_RE = re.compile(r"\{\{([A-Z_]+)\}\}")

_VALID_NAMES = {t.value for t in PIIType}

# The nine structured identifiers `forge/validators.py` already detects at 1.0000
# recall (ADR 0012). The model does not need to learn them, so they are held to a
# minority of the corpus rather than dropped: dropping them entirely would change
# what G1 measures, which is a contract decision and not this ADR's to make.
VALIDATOR_OWNED: frozenset[PIIType] = HIGH_SEVERITY

# Everything the model actually owns. `find_high_severity` never claims these.
MODEL_OWNED: frozenset[PIIType] = frozenset(set(PIIType) - set(HIGH_SEVERITY))

# The subset of model-owned types the student measurably fails, and that a fixed
# template pool structurally cannot teach (ROADMAP §3 WP-2). A carrier carrying at
# least one of these is eligible for Track B.
TRACK_B_FOCUS: frozenset[PIIType] = frozenset(
    {
        PIIType.PERSON,
        PIIType.STREET_ADDRESS,
        PIIType.USERNAME,
        PIIType.AGE,
        PIIType.LOCATION,
    }
)

MIN_CARRIER_CHARS = 25
MAX_CARRIER_CHARS = 600
MAX_PLACEHOLDERS = 12


class CarrierError(ValueError):
    """A carrier shape that cannot be safely filled."""


@dataclass(frozen=True)
class Carrier:
    """A validated carrier shape plus the provenance needed for the data card."""

    shape: str
    source: str  # e.g. "teacher:gpt-oss-120b" or "template:build_gold"
    register: str = "unspecified"  # support_ticket, chat, email, form, log, ...

    @property
    def slots(self) -> list[PIIType]:
        """Placeholder types in document order (duplicates kept)."""
        return [PIIType(m.group(1)) for m in PLACEHOLDER_RE.finditer(self.shape)]

    @property
    def slot_types(self) -> frozenset[PIIType]:
        return frozenset(self.slots)

    @property
    def is_negative(self) -> bool:
        """A carrier with no PII at all — the false-positive control."""
        return not self.slots

    @property
    def track_b_eligible(self) -> bool:
        return bool(self.slot_types & TRACK_B_FOCUS)

    def normalised(self) -> str:
        """Whitespace-normalised shape, for duplicate detection across sources."""
        return re.sub(r"\s+", " ", self.shape).strip()


def shape_of(record: PIIRecord) -> str:
    """Recover the carrier shape of an already-built record.

    ``PIIRecord.redacted`` replaces each span with ``[LABEL]``; converting that to
    ``{{LABEL}}`` yields the shape the record was (or could have been) built from.
    This is how shapes in the frozen eval splits are compared against generated
    ones without re-running their builder.
    """
    text = re.sub(r"\[([A-Z_]+)\]", lambda m: "{{" + m.group(1) + "}}", record.redacted)
    return re.sub(r"\s+", " ", text).strip()


def validate_shape(shape: str) -> Carrier:
    """Parse and check one carrier shape, raising ``CarrierError`` on anything unsafe.

    The checks exist because the shape comes from a language model, and a malformed
    shape does not fail loudly at fill time — it silently produces a record whose
    spans are wrong, which is the single defect this engine cannot tolerate.
    """
    if not isinstance(shape, str):
        raise CarrierError(f"not a string: {type(shape).__name__}")

    shape = shape.strip()

    if len(shape) < MIN_CARRIER_CHARS:
        raise CarrierError(f"too short ({len(shape)} < {MIN_CARRIER_CHARS} chars)")
    if len(shape) > MAX_CARRIER_CHARS:
        raise CarrierError(f"too long ({len(shape)} > {MAX_CARRIER_CHARS} chars)")
    if "\n" in shape and shape.count("\n") > 8:
        raise CarrierError("too many line breaks")

    # An unknown or misspelled type name would be filled as literal text and then
    # silently carry no span at all.
    names = PLACEHOLDER_RE.findall(shape)
    for name in names:
        if name not in _VALID_NAMES:
            raise CarrierError(f"unknown placeholder type {{{{{name}}}}}")

    # A stray single-brace or unmatched-brace construct means the model produced
    # something we are about to misparse.
    stripped = PLACEHOLDER_RE.sub("", shape)
    if "{" in stripped or "}" in stripped:
        raise CarrierError("unbalanced or non-placeholder braces remain")

    if len(names) > MAX_PLACEHOLDERS:
        raise CarrierError(f"{len(names)} placeholders exceeds cap {MAX_PLACEHOLDERS}")

    # Adjacent placeholders with no separator make the boundary between two
    # entities unrecoverable ("{{PERSON}}{{PHONE}}" -> "Jane Doe+1 555-...").
    if re.search(r"\}\}\s*\{\{", shape):
        raise CarrierError("adjacent placeholders leave an ambiguous boundary")

    return Carrier(shape=shape, source="unknown")


def fill(
    carrier: Carrier,
    record_id: str,
    value_for,
    split: str = "train",
    source: str | None = None,
) -> PIIRecord:
    """Instantiate a carrier into a record with construction-exact spans.

    ``value_for(pii_type) -> str`` supplies the surface value; in production this is
    ``build_gold.PIIValueGenerator.gen`` so generated values follow exactly the same
    distribution as the frozen eval set — the carrier text is the variable under
    study, not the values.

    Offsets are accumulated from the emitted segments, the same way
    ``build_gold.build_record`` does it, so a span can never disagree with its text.
    """
    parts: list[str] = []
    spans: list[PIISpan] = []
    cursor = 0
    last = 0

    for m in PLACEHOLDER_RE.finditer(carrier.shape):
        literal = carrier.shape[last : m.start()]
        parts.append(literal)
        cursor += len(literal)

        pii_type = PIIType(m.group(1))
        value = value_for(pii_type)
        if not value or "\n" in value:
            raise CarrierError(f"generator returned an unusable value for {pii_type.value}")

        spans.append(PIISpan(start=cursor, end=cursor + len(value), label=pii_type, text=value))
        parts.append(value)
        cursor += len(value)
        last = m.end()

    tail = carrier.shape[last:]
    parts.append(tail)

    return PIIRecord(
        id=record_id,
        text="".join(parts),
        spans=spans,
        split=split,
        source=source or carrier.source,
    )


def span_key(s: PIISpan) -> tuple[int, int, str]:
    return (s.start, s.end, s.label.value)


@dataclass(frozen=True)
class AnchorResult:
    """Outcome of checking a teacher's labels against what was actually injected.

    The teacher never sees the construction labels; this compares its independent
    consensus to ground truth we already hold, which is the only reason a
    distillation label set can be trusted to be *complete*.
    """

    ok: bool
    missing: tuple[PIISpan, ...] = ()  # injected, model-owned, teacher found nothing
    boundary: tuple[tuple[PIISpan, PIISpan], ...] = ()  # (injected, teacher's extent)
    repaired: tuple[PIISpan, ...] = ()  # validator-owned, taken from construction
    extra: tuple[PIISpan, ...] = ()  # teacher found PII we did not inject

    @property
    def reason(self) -> str:
        bits = []
        if self.missing:
            bits.append("anchor_missing:" + ",".join(sorted({s.label.value for s in self.missing})))
        if self.boundary:
            bits.append("anchor_boundary:" + ",".join(sorted({a.label.value for a, _ in self.boundary})))
        return "; ".join(bits)


def anchor_against_construction(
    injected: list[PIISpan],
    teacher: list[PIISpan],
) -> AnchorResult:
    """Gate a teacher label set against the entities we know are in the text.

    Three rules, each forced by a measurement rather than by taste:

    1. **Model-owned injected spans must be matched exactly.** A teacher false
       negative in *training* data is a lesson in under-enumeration, and
       under-enumeration is precisely the student's diagnosed failure (span ratio
       0.46-0.84, `PERSON` 110 FN). Training on incomplete labels would teach the
       exact defect the run is trying to remove. Boundary disagreements are
       rejected for a separate reason: `data/gold/PROTOCOL.md` §3 fixes the
       convention ("a full mailing address -> STREET_ADDRESS"), and the teacher
       splits addresses ("01/12, Banik Circle, Ballia" -> "Banik Circle") in
       12.5% of test cases. Adopting the teacher's convention would train against
       the contract we are scored on.

    2. **Validator-owned spans are repaired from construction, not rejected.** On
       the nine high-severity types the teacher is measurably worse than the
       deterministic layer (`DRIVER_LICENSE` exact recall 0.533 on the frozen
       test set). Where we hold the exact injected offsets, using the teacher's
       guess instead would be choosing the worse label on purpose, and rejecting
       the record would discard it for a type the model is not being asked to own.

    3. **Teacher spans over text we did not inject are kept.** These are the
       naturally-occurring entities in the teacher's own prose — the fuzzy,
       context-dependent cases construction cannot manufacture. They rest on the
       k=3 consensus alone, which is stated as a limitation rather than hidden.
    """
    teacher_by_key = {span_key(s): s for s in teacher}

    missing: list[PIISpan] = []
    boundary: list[tuple[PIISpan, PIISpan]] = []
    repaired: list[PIISpan] = []

    for inj in injected:
        if inj.label in VALIDATOR_OWNED:
            repaired.append(inj)
            continue
        if span_key(inj) in teacher_by_key:
            continue
        overlap = next(
            (t for t in teacher if t.label == inj.label and t.start < inj.end and inj.start < t.end),
            None,
        )
        if overlap is not None:
            boundary.append((inj, overlap))
        else:
            missing.append(inj)

    injected_ranges = [(s.start, s.end) for s in injected]
    extra = [
        t
        for t in teacher
        if not any(t.start < e and s < t.end for s, e in injected_ranges)
    ]

    return AnchorResult(
        ok=not missing and not boundary,
        missing=tuple(missing),
        boundary=tuple(boundary),
        repaired=tuple(repaired),
        extra=tuple(extra),
    )


def merge_anchor(result: AnchorResult, injected: list[PIISpan]) -> list[PIISpan]:
    """The final Track B label set for an accepted record.

    Construction labels for everything injected, plus the teacher's own discoveries.
    Overlaps are impossible by construction: ``extra`` excludes anything touching an
    injected range.
    """
    spans = list(injected) + list(result.extra)
    return sorted(spans, key=lambda s: (s.start, s.end))
