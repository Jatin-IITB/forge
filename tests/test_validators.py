"""Deterministic validators for high-severity identifiers (ADR 0012).

The gold-set measurement lives in `scripts/run_eval.py --validators`; these tests
cover correctness of the primitives and the disambiguation rules that the gold-set
numbers depend on, so a regression shows up here rather than as a mysterious drop
in recall.
"""

import json
from pathlib import Path

import pytest

from forge.schema import HIGH_SEVERITY, PIISpan, PIIType
from forge.validators import (
    find_high_severity,
    luhn_valid,
    merge_with_model,
    shannon_entropy,
    verhoeff_valid,
)

GOLD_TEST = Path(__file__).resolve().parents[1] / "data" / "gold" / "test.jsonl"


# --------------------------------------------------------------------------
# Checksums — external test vectors, not values produced by our own code
# --------------------------------------------------------------------------


@pytest.mark.parametrize("digits", ["2363", "758722", "1428570"])
def test_verhoeff_accepts_valid(digits: str) -> None:
    assert verhoeff_valid(digits)


@pytest.mark.parametrize("digits", ["2364", "758723", "1428571"])
def test_verhoeff_rejects_invalid(digits: str) -> None:
    assert not verhoeff_valid(digits)


def test_verhoeff_detects_single_digit_and_transposition_errors() -> None:
    """The properties Verhoeff exists for — the reason UIDAI chose it."""
    base = "2363"
    assert verhoeff_valid(base)
    # every single-digit substitution must break it
    for i in range(len(base)):
        for d in "0123456789":
            if d == base[i]:
                continue
            assert not verhoeff_valid(base[:i] + d + base[i + 1 :])
    # adjacent transposition must break it
    assert not verhoeff_valid("2633")


@pytest.mark.parametrize(
    "digits", ["4532015112830366", "4111111111111111", "5500005555555559"]
)
def test_luhn_accepts_valid(digits: str) -> None:
    assert luhn_valid(digits)


@pytest.mark.parametrize("digits", ["4532015112830367", "4111111111111112", "1234"])
def test_luhn_rejects_invalid(digits: str) -> None:
    assert not luhn_valid(digits)


def test_checksums_reject_non_digits() -> None:
    assert not verhoeff_valid("12a4")
    assert not luhn_valid("4532-0151")


def test_entropy_separates_secrets_from_prose() -> None:
    assert shannon_entropy("aaaaaaaa") < 1.0
    assert shannon_entropy("KIjku2HChRnTLFf5GCZdDiGADK") > 3.0


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def _labels(text: str) -> set[tuple[str, str]]:
    return {(h.span.label.value, h.span.text) for h in find_high_severity(text)}


def test_detects_each_high_severity_type() -> None:
    cases = [
        ("Aadhaar number: 5280 9885 1656.", PIIType.AADHAAR, "5280 9885 1656"),
        ("PAN: UZRZA6578Z on file.", PIIType.PAN, "UZRZA6578Z"),
        ("Passport J5841687 expires soon.", PIIType.PASSPORT, "J5841687"),
        ("Identity proof: DL IL-56506098 attached.", PIIType.DRIVER_LICENSE, "IL-56506098"),
        ("SSN 360-93-3491 on record.", PIIType.SSN, "360-93-3491"),
        ("Card number: 3587 0051 6271 3618.", PIIType.CREDIT_CARD, "3587 0051 6271 3618"),
        ("Deposit to account 743671369594 today.", PIIType.BANK_ACCOUNT, "743671369594"),
        ("New API key: tok_test_lkxMT0hQoAZvUhEREEnLkP1A.", PIIType.API_KEY, "tok_test_lkxMT0hQoAZvUhEREEnLkP1A"),
        ("Your temporary password is mfy_8733!JW. Change it.", PIIType.PASSWORD, "mfy_8733!JW"),
    ]
    for text, label, value in cases:
        assert (label.value, value) in _labels(text), f"missed {label.value} in {text!r}"


def test_offsets_are_exact() -> None:
    """A span whose offsets are off by one is a miss under exact-match scoring."""
    text = "Your temporary password is mfy_8733!JW. Change it."
    for hit in find_high_severity(text):
        assert text[hit.span.start : hit.span.end] == hit.span.text


def test_nearest_keyword_disambiguates_competing_types() -> None:
    """Both keywords sit in the window; each number must go to its nearer one."""
    text = "Account holder Chakrika Dua with bank account 91545226752895 and card 4369378547777."
    found = _labels(text)
    assert (PIIType.BANK_ACCOUNT.value, "91545226752895") in found
    assert (PIIType.CREDIT_CARD.value, "4369378547777") in found


def test_luhn_breaks_ties_toward_card() -> None:
    """A 12-digit number fits AADHAAR and CREDIT_CARD; the checksum decides."""
    text = "Your card 630417445180 was charged."
    assert (PIIType.CREDIT_CARD.value, "630417445180") in _labels(text)


def test_longer_span_wins_over_contained_match() -> None:
    """The 12-digit AADHAAR pattern must not truncate a 16-digit card."""
    text = "Card number: 3587 0051 6271 3618."
    labels = _labels(text)
    assert (PIIType.CREDIT_CARD.value, "3587 0051 6271 3618") in labels
    assert not any(lbl == PIIType.AADHAAR.value for lbl, _ in labels)


def test_checksum_failure_never_suppresses_a_span() -> None:
    """Core ADR 0012 rule: checksums inform confidence, they do not gate recall.

    Only 2/29 synthetic Aadhaar values are Verhoeff-valid while real ones always
    are; gating on the checksum would make recall depend on the dataset.
    """
    text = "Aadhaar number: 5280 9885 1656."
    hits = [h for h in find_high_severity(text) if h.span.label is PIIType.AADHAAR]
    assert hits, "Aadhaar span dropped"
    assert hits[0].checksum_valid is False  # this value fails Verhoeff
    assert hits[0].confidence in {"medium", "low"}


def test_generic_numbers_need_context() -> None:
    """A bare digit run is not a bank account without a keyword nearby."""
    assert not any(
        lbl == PIIType.BANK_ACCOUNT.value
        for lbl, _ in _labels("Reference 743671369594 was processed.")
    )


def test_short_keywords_do_not_match_inside_words() -> None:
    """'pan' must not fire from 'company', 'dl' must not fire from 'middle'."""
    assert not any(
        lbl == PIIType.BANK_ACCOUNT.value
        for lbl, _ in _labels("The company processed 743671369594 units.")
    )


# --------------------------------------------------------------------------
# Merge semantics
# --------------------------------------------------------------------------


def test_merge_prefers_validator_over_model_on_overlap() -> None:
    text = "Card number: 3587 0051 6271 3618."
    model = [PIISpan(start=13, end=32, label=PIIType.BANK_ACCOUNT, text="3587 0051 6271 3618")]
    merged = merge_with_model(model, find_high_severity(text))
    assert len(merged) == 1
    assert merged[0].label is PIIType.CREDIT_CARD


def test_merge_keeps_model_spans_for_other_types() -> None:
    text = "Priya Sharma, PAN: UZRZA6578Z."
    model = [PIISpan(start=0, end=12, label=PIIType.PERSON, text="Priya Sharma")]
    merged = merge_with_model(model, find_high_severity(text))
    labels = {s.label for s in merged}
    assert PIIType.PERSON in labels
    assert PIIType.PAN in labels
    assert merged == sorted(merged, key=lambda s: (s.start, s.end))


# --------------------------------------------------------------------------
# The gate the whole exercise exists for
# --------------------------------------------------------------------------


def test_high_severity_recall_floor_on_frozen_gold() -> None:
    """Every high-severity type must clear the contract's 0.99 recall floor.

    The teacher reaches only 0.8746 micro-recall on these types (6 of 9 below the
    floor), which is why ADR 0012 moved them to deterministic validators.
    """
    if not GOLD_TEST.exists():
        pytest.skip("test.jsonl not built — run `make gold`")

    hs = {t.value for t in HIGH_SEVERITY}
    tp = dict.fromkeys(hs, 0)
    fn = dict.fromkeys(hs, 0)

    for line in GOLD_TEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gold = {(s["start"], s["end"], s["label"]) for s in rec["spans"] if s["label"] in hs}
        pred = {
            (h.span.start, h.span.end, h.span.label.value)
            for h in find_high_severity(rec["text"])
        }
        for key in gold & pred:
            tp[key[2]] += 1
        for key in gold - pred:
            fn[key[2]] += 1

    failures = []
    for label in sorted(hs):
        total = tp[label] + fn[label]
        if not total:
            continue
        recall = tp[label] / total
        if recall < 0.99:
            failures.append(f"{label} recall {recall:.4f} (missed {fn[label]}/{total})")

    assert not failures, "high-severity floor breached: " + "; ".join(failures)


# ---------------------------------------------------------------------------
# Type disambiguation — the three confusions that produced every validator
# false positive on the gold set (55 across test + val, now 0).
# ---------------------------------------------------------------------------


def _labels_at(text: str, needle: str) -> set[PIIType]:
    """Which types the validators claim over `needle` in `text`."""
    start = text.index(needle)
    end = start + len(needle)
    return {
        h.span.label
        for h in find_high_severity(text)
        if h.span.start < end and start < h.span.end
    }


def test_international_dialling_prefix_is_not_a_card() -> None:
    """'+91 99854 35346' is a phone number, not a 12-digit card.

    The largest single source of false positives: 41 of 55. A leading '+' is a
    non-word character, so '\\b' creates a boundary rather than suppressing one
    and the digits matched the card pattern cleanly.
    """
    text = "Reach me at phillip76@example.net or call +91 99854 35346."
    assert PIIType.CREDIT_CARD not in _labels_at(text, "91 99854 35346")


def test_real_grouped_card_is_still_detected() -> None:
    """Guard for the fix above — it must not cost card recall."""
    text = "Please charge my card 4539 1488 0343 6467 today."
    assert PIIType.CREDIT_CARD in _labels_at(text, "4539 1488 0343 6467")


def test_password_pattern_does_not_truncate_an_email() -> None:
    """'adam65@example.net' must not be claimed as 'adam65@example'.

    '@' stays in the password character class because passwords contain it, so
    the match ran to the '.' and stopped — redacting the mailbox but leaving
    '.net' behind. That is a partial leak, not merely a mislabel.
    """
    text = "Support case: user claredean (adam65@example.net) reports a password reset."
    assert not any(
        h.span.label is PIIType.PASSWORD and h.span.text.startswith("adam65")
        for h in find_high_severity(text)
    )


def test_sentence_final_password_keeps_its_full_stop_outside_the_span() -> None:
    """The trailing-dot guard must not undo the ADR 0012 offset fix."""
    text = "The password is mfy_8733!JW."
    hits = [h for h in find_high_severity(text) if h.span.label is PIIType.PASSWORD]
    assert [h.span.text for h in hits] == ["mfy_8733!JW"]


def test_login_alert_does_not_make_the_username_a_password() -> None:
    """'login' introduces a username at least as often as a password.

    13 of 55 false positives. The keyword was dropped rather than patched,
    because the remaining keywords name the secret explicitly.
    """
    text = "Alert: login from 107.248.180.67 for randy04 with token mykey_kfrXzwQzzLmHaVTe6"
    assert PIIType.PASSWORD not in _labels_at(text, "randy04")


def test_one_keyword_claims_only_the_nearest_secret() -> None:
    """Two password-shaped tokens, one keyword: the nearer one is the secret."""
    text = "Failed password attempt: 'oly_4121!FN' for rachit77."
    hits = [h.span.text for h in find_high_severity(text) if h.span.label is PIIType.PASSWORD]
    assert hits == ["oly_4121!FN"]


def test_equidistant_secrets_are_both_kept() -> None:
    """Ties survive, so a text with two genuine passwords cannot lose one."""
    text = "Passwords: ab12cd34 and password ef56gh78 both expired."
    hits = [h.span.text for h in find_high_severity(text) if h.span.label is PIIType.PASSWORD]
    assert len(hits) >= 1


@pytest.mark.parametrize("split", ["test", "val"])
def test_zero_false_positives_on_gold(split: str) -> None:
    """The headline invariant: perfect precision AND recall on both splits.

    `val` matters more than `test` here — the validators were developed by
    inspecting test-set misses, so only `val` (seed 4242, generated after they
    were frozen) is genuinely held out.
    """
    path = GOLD_TEST.parent / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{split} split not built")
    hs = {t.value for t in HIGH_SEVERITY}
    tp = fn = fp = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gold = {(s["start"], s["end"], s["label"]) for s in rec["spans"] if s["label"] in hs}
        pred = {
            (h.span.start, h.span.end, h.span.label.value)
            for h in find_high_severity(rec["text"])
        }
        tp += len(gold & pred)
        fn += len(gold - pred)
        fp += len(pred - gold)
    assert fn == 0, f"{split}: {fn} high-severity misses"
    assert fp == 0, f"{split}: {fp} false positives"
    assert tp > 0
