"""Carrier shapes and the construction anchor (ADR 0015).

These are the checks that stand between a malformed teacher output and a training
record whose spans are silently wrong. A bad carrier does not raise at fill time
on its own — it produces a record that validates and is simply mislabelled — so
the validation has to be explicit and tested.
"""

from __future__ import annotations

import pytest

from forge.carriers import (
    TRACK_B_FOCUS,
    VALIDATOR_OWNED,
    Carrier,
    CarrierError,
    anchor_against_construction,
    fill,
    merge_anchor,
    shape_of,
    validate_shape,
)
from forge.schema import PIIRecord, PIISpan, PIIType


def const_gen(values: dict[PIIType, str]):
    counters = dict.fromkeys(values, 0)

    def gen(t: PIIType) -> str:
        counters[t] += 1
        return values[t] if counters[t] == 1 else f"{values[t]}-{counters[t]}"

    return gen


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------
def test_valid_shape_round_trips():
    c = validate_shape("Ticket 4451: caller {{PERSON}} reports card {{CREDIT_CARD}} declined.")
    assert c.slots == [PIIType.PERSON, PIIType.CREDIT_CARD]
    assert c.track_b_eligible
    assert not c.is_negative


def test_negative_carrier_has_no_slots():
    c = validate_shape("The quarterly migration window opens on Saturday at 02:00 IST.")
    assert c.is_negative
    assert not c.track_b_eligible


@pytest.mark.parametrize(
    ("shape", "fragment"),
    [
        ("short", "too short"),
        ("Please contact {{PERSONN}} about the outstanding invoice today.", "unknown placeholder"),
        ("Please contact {PERSON} about the outstanding invoice today.", "braces"),
        ("Contact {{PERSON}} {{PHONE}} regarding the outstanding invoice.", "adjacent"),
        ("Contact {{PERSON}}{{PHONE}} regarding the outstanding invoice.", "adjacent"),
        ("x" * 700, "too long"),
    ],
)
def test_rejects_unsafe_shapes(shape, fragment):
    with pytest.raises(CarrierError, match=fragment):
        validate_shape(shape)


def test_placeholder_cap():
    shape = "Record: " + " and ".join(["{{PERSON}}"] * 13) + " were all notified."
    with pytest.raises(CarrierError, match="exceeds cap"):
        validate_shape(shape)


# ---------------------------------------------------------------------------
# Fill: the construction guarantee
# ---------------------------------------------------------------------------
def test_fill_produces_exact_offsets():
    c = validate_shape("Ship to {{PERSON}} at {{STREET_ADDRESS}}, call {{PHONE}} on arrival.")
    rec = fill(c, "r1", const_gen({
        PIIType.PERSON: "Asha Menon",
        PIIType.STREET_ADDRESS: "12 Rowan Lane, Petersfield",
        PIIType.PHONE: "+44 7700 900123",
    }))
    assert rec.text == ("Ship to Asha Menon at 12 Rowan Lane, Petersfield, "
                        "call +44 7700 900123 on arrival.")
    # PIIRecord's own validator already enforces text[start:end] == span.text; assert
    # it here too so a regression names *this* invariant rather than a pydantic error.
    for s in rec.spans:
        assert rec.text[s.start:s.end] == s.text
    assert [s.label for s in rec.spans] == [PIIType.PERSON, PIIType.STREET_ADDRESS, PIIType.PHONE]


def test_fill_handles_repeated_type_in_one_shape():
    c = validate_shape("{{PERSON}} escalated the ticket to {{PERSON}} before the handover.")
    rec = fill(c, "r2", const_gen({PIIType.PERSON: "Lee Park"}))
    assert len(rec.spans) == 2
    assert rec.spans[0].text != rec.spans[1].text
    for s in rec.spans:
        assert rec.text[s.start:s.end] == s.text


def test_fill_placeholder_at_both_edges():
    c = validate_shape("{{PERSON}} signed off on the release before leaving for {{LOCATION}}")
    rec = fill(c, "r3", const_gen({PIIType.PERSON: "Nia Okafor", PIIType.LOCATION: "Kochi"}))
    assert rec.spans[0].start == 0
    assert rec.spans[-1].end == len(rec.text)


def test_shape_of_inverts_fill():
    c = validate_shape("Reset link sent to {{EMAIL}} for user {{USERNAME}} at 09:12.")
    rec = fill(c, "r4", const_gen({PIIType.EMAIL: "a@b.io", PIIType.USERNAME: "abee"}))
    assert shape_of(rec) == c.normalised()


# ---------------------------------------------------------------------------
# The construction anchor
# ---------------------------------------------------------------------------
def _rec(text: str, spans: list[tuple[int, int, PIIType]]) -> list[PIISpan]:
    return [PIISpan(start=a, end=b, label=t, text=text[a:b]) for a, b, t in spans]


TEXT = "Ship to Asha Menon at 12 Rowan Lane, Petersfield, before Friday."
INJECTED = _rec(TEXT, [(8, 18, PIIType.PERSON), (22, 48, PIIType.STREET_ADDRESS)])


def test_anchor_accepts_exact_agreement():
    r = anchor_against_construction(INJECTED, list(INJECTED))
    assert r.ok
    assert not r.missing and not r.boundary and not r.extra


def test_anchor_rejects_a_missed_entity():
    """A teacher false negative in training data teaches under-enumeration.

    This is the student's diagnosed failure (span ratio 0.46-0.84), so it is the
    one thing the gate must never let through.
    """
    teacher = [s for s in INJECTED if s.label is PIIType.PERSON]
    r = anchor_against_construction(INJECTED, teacher)
    assert not r.ok
    assert [s.label for s in r.missing] == [PIIType.STREET_ADDRESS]
    assert "anchor_missing:STREET_ADDRESS" in r.reason


def test_anchor_rejects_a_boundary_disagreement():
    """The teacher splits addresses; PROTOCOL.md §3 says they are one span."""
    teacher = _rec(TEXT, [(8, 18, PIIType.PERSON), (25, 35, PIIType.STREET_ADDRESS)])
    r = anchor_against_construction(INJECTED, teacher)
    assert not r.ok
    assert not r.missing
    assert r.boundary[0][0].label is PIIType.STREET_ADDRESS
    assert "anchor_boundary:STREET_ADDRESS" in r.reason


def test_anchor_repairs_validator_owned_types_instead_of_rejecting():
    """DRIVER_LICENSE exact recall is 0.533 at the teacher and 1.0000 at the
    validators, so where the exact injected offsets are known, using the teacher's
    guess would be choosing the worse label on purpose."""
    text = "Identity proof: DL CA-19384756 on file for Asha Menon."
    injected = _rec(text, [(19, 30, PIIType.DRIVER_LICENSE), (43, 53, PIIType.PERSON)])
    teacher = _rec(text, [(22, 30, PIIType.DRIVER_LICENSE), (43, 53, PIIType.PERSON)])
    r = anchor_against_construction(injected, teacher)
    assert r.ok, "a validator-owned disagreement must not sink the record"
    assert [s.label for s in r.repaired] == [PIIType.DRIVER_LICENSE]
    merged = merge_anchor(r, injected)
    assert any(s.start == 19 and s.end == 30 for s in merged), "construction label wins"


def test_anchor_keeps_teacher_discoveries_in_prose():
    text = "Ship to Asha Menon at 12 Rowan Lane, Petersfield, ask for Ravi at the gate."
    injected = _rec(text, [(8, 18, PIIType.PERSON), (22, 48, PIIType.STREET_ADDRESS)])
    teacher = injected + _rec(text, [(58, 62, PIIType.PERSON)])
    r = anchor_against_construction(injected, teacher)
    assert r.ok
    assert [s.text for s in r.extra] == ["Ravi"]
    merged = merge_anchor(r, injected)
    assert len(merged) == 3
    # Must remain a legal record: sorted, non-overlapping, offsets intact.
    PIIRecord(id="x", text=text, spans=merged, split="train")


def test_merge_never_overlaps_an_injected_span():
    text = "Ship to Asha Menon at 12 Rowan Lane, Petersfield, before Friday."
    injected = _rec(text, [(8, 18, PIIType.PERSON)])
    teacher = _rec(text, [(8, 18, PIIType.PERSON), (8, 12, PIIType.USERNAME)])
    r = anchor_against_construction(injected, teacher)
    assert not r.extra, "a teacher span overlapping an injected one is not an extra"
    PIIRecord(id="x", text=text, spans=merge_anchor(r, injected), split="train")


def test_track_partitions_are_disjoint_and_cover_the_taxonomy():
    assert not (TRACK_B_FOCUS & VALIDATOR_OWNED)
    assert TRACK_B_FOCUS <= set(PIIType) - VALIDATOR_OWNED


def test_carrier_normalisation_collapses_whitespace():
    a = Carrier(shape="Call  {{PERSON}}\n  today.", source="t")
    b = Carrier(shape="Call {{PERSON}} today.", source="t")
    assert a.normalised() == b.normalised()
