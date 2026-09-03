"""Out-of-domain gate (forge/ood.py, contract v2 §ood_behavior).

The asymmetry drives every test here. A false positive is a refusal to redact a
real document — an unredacted leak caused by the safety mechanism itself. A
false negative is a hallucinated span on junk. These are not comparable, so the
in-domain precision tests are the ones that must never be relaxed, and recall is
allowed to give ground.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.ood import MAX_TOKENS, detect_out_of_domain
from forge.schema import PIIRecord

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
PROBES = ROOT / "data" / "ood_probe.jsonl"


def _records(name: str) -> list[PIIRecord]:
    path = GOLD / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{name} split not built")
    return [
        PIIRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestNeverRefusesRealDocuments:
    """The property that matters. Everything else is secondary to it."""

    @pytest.mark.parametrize("split", ["test", "val", "dev"])
    def test_no_in_domain_record_is_refused(self, split):
        """Measured at 0 false positives across 1107 committed records.

        If this fails, the gate is dropping documents that contain real PII and
        the pipeline leaks. There is no acceptable non-zero value.
        """
        refused = [
            (r.id, detect_out_of_domain(r.text).reason)
            for r in _records(split)
            if detect_out_of_domain(r.text).is_ood
        ]
        assert refused == [], f"{split}: gate refused {len(refused)} real records: {refused[:5]}"

    def test_a_labelled_api_key_is_not_mistaken_for_a_base64_dump(self):
        """The regression that nearly shipped.

        "API key: FAKEKEYZGCFcs5PmuGf8gk9XIIaOenQOXn3RB1gnI1S." is 79% base64
        *because the credential is the point*. Refusing it would be a leak
        caused by the gate, on a high-severity type. What separates it from a
        dump is that stripping the encoded run leaves a sentence behind.
        """
        for text in (
            "API key: FAKEKEYZGCFcs5PmuGf8gk9XIIaOenQOXn3RB1gnI1S.",
            "Token FAKEKEYx45XduJsT568FwJWDKsU3BX04gJcZWnsJz0G expired at 14:32 UTC",
        ):
            assert not detect_out_of_domain(text).is_ood, text

    def test_a_ticket_quoting_one_uuid_is_still_in_domain(self):
        """Coverage, not presence: support tickets quote request ids."""
        text = (
            "Ticket from Priya Sharma (priya.s@example.com): the upload failed, "
            "request id 550e8400-e29b-41d4-a716-446655440000, please investigate."
        )
        assert not detect_out_of_domain(text).is_ood

    def test_structured_records_with_few_function_words_survive(self):
        """Why prose-likeness was rejected as a signal.

        In-domain records here are field-value fragments with almost no English
        function words. A density test would flag them, so none is used.
        """
        text = "Ticket #4521: Rachita Thakkar, DOB 1985-03-12, Aadhaar 5280 9885 1656, PAN UZRZA6578Z."
        assert not detect_out_of_domain(text).is_ood


class TestCatchesEachOutOfDomainFamily:
    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("Это обычное предложение без личной информации.", "non_latin_script"),
            ("これは個人情報を含まない普通の文章です。", "non_latin_script"),
            ("def compute(x):\n    return sum(i for i in range(x))", "code_python"),
            ("SELECT user_id, COUNT(*) FROM events WHERE ts > NOW()", "code_sql"),
            ('{"status": 200, "latency_ms": 42, "cache": true}', "code_json"),
            ("<div class='container'><span id='x-91'>Loading</span></div>", "code_html"),
            ("550e8400-e29b-41d4-a716-446655440000 3f2504e0-4f89-11d3-9a0c-0305e82c3301", "encoded_uuid"),
            ("d41d8cd98f00b204e9800998ecf8427e a1b2c3d4e5f60718293a4b5c6d7e8f90", "encoded_hash"),
            ("89 50 4e 47 0d 0a 1a 0a 00 00 00 0d 49 48 44 52", "encoded_hex_dump"),
            ("    \t  ", "empty_or_symbolic"),
            ("!!! ??? ... --- *** ###", "empty_or_symbolic"),
            ("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15", "numeric_dump"),
        ],
    )
    def test_family(self, text, reason):
        v = detect_out_of_domain(text)
        assert v.is_ood, f"missed: {text!r}"
        assert v.reason == reason, f"{text!r} -> {v.reason}, expected {reason}"

    def test_oversize_input_is_refused(self):
        text = "The quarterly report describes operational metrics. " * 200
        v = detect_out_of_domain(text)
        assert v.is_ood
        assert v.reason == "too_long"
        assert v.signals["est_tokens"] > MAX_TOKENS

    def test_full_probe_set_recall(self):
        """21/21 as measured. Recorded so a regression is visible as a number."""
        if not PROBES.exists():
            pytest.skip("probe set not built")
        probes = [json.loads(x) for x in PROBES.read_text().splitlines() if x.strip()]
        ood = [p for p in probes if p["category"] == "out_of_domain"]
        missed = [p["id"] for p in ood if not detect_out_of_domain(p["text"]).is_ood]
        assert missed == [], f"gate missed {missed}"

    def test_adversarial_probes_are_never_refused(self):
        """They carry real PII behind an attack — refusing them is the attack working."""
        if not PROBES.exists():
            pytest.skip("probe set not built")
        probes = [json.loads(x) for x in PROBES.read_text().splitlines() if x.strip()]
        adv = [p for p in probes if p["category"] == "adversarial"]
        refused = [p["id"] for p in adv if detect_out_of_domain(p["text"]).is_ood]
        assert refused == [], f"gate refused adversarial probes: {refused}"


class TestVerdict:
    def test_response_body_matches_the_contract(self):
        v = detect_out_of_domain("これは個人情報を含まない普通の文章です。")
        assert v.as_response() == {
            "status": "out_of_domain",
            "reason": "non_latin_script",
            "spans": [],
        }

    def test_in_domain_verdict_carries_no_reason(self):
        v = detect_out_of_domain("Please contact Rachita at rachita@example.com about the invoice.")
        assert not v.is_ood
        assert v.reason is None

    def test_signals_are_populated_for_auditing(self):
        v = detect_out_of_domain("Contact me at test@example.com")
        assert v.signals["chars"] > 0
        assert "est_tokens" in v.signals
