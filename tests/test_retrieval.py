"""Retrieval-based span proposal (forge/retrieval.py).

The property that makes this module legitimate rather than test-set leakage is
measurable, so it is measured here: train and eval splits share ~1% of PII
surface forms, so nothing retrieved can carry an answer with it. What transfers
is template structure, and that is what these tests pin.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from forge.retrieval import SpanRetriever, merge_with_model
from forge.schema import PIIRecord, PIISpan, PIIType

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train_v2.jsonl"
GOLD = ROOT / "data" / "gold"


def _rec(rid: str, text: str, spans: list[tuple[int, int, PIIType]]) -> PIIRecord:
    return PIIRecord(
        id=rid,
        text=text,
        split="train",
        spans=[PIISpan(start=s, end=e, label=lab, text=text[s:e]) for s, e, lab in spans],
    )


def _load(path: Path) -> list[PIIRecord]:
    return [
        PIIRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestAlignment:
    def test_transfers_layout_not_values(self):
        """The core mechanism: a template match yields the *input's* substring.

        The neighbour says "the run between 'Patient ' and ', age ' is a PERSON".
        The value returned must come from the query document, never from the
        indexed one — otherwise this is retrieval of answers, not of structure.
        """
        train = [_rec("t1", "Patient Jane Doe, age 41, was seen today.",
                      [(8, 16, PIIType.PERSON), (22, 24, PIIType.AGE)])]
        r = SpanRetriever(train)
        props = r.propose("Patient Ravi Kumar, age 63, was seen today.")
        got = {(p.span.label.value, p.span.text) for p in props}
        assert got == {("PERSON", "Ravi Kumar"), ("AGE", "63")}

    def test_partial_template_match_contributes_nothing(self):
        """Half an alignment is not half an answer — it is a different template."""
        train = [_rec("t1", "Patient Jane Doe, age 41, was seen today.",
                      [(8, 16, PIIType.PERSON), (22, 24, PIIType.AGE)])]
        r = SpanRetriever(train)
        assert r.propose("Invoice 4102 was posted to the ledger on Tuesday.") == []

    def test_no_neighbour_above_threshold_yields_nothing(self):
        train = [_rec("t1", "Patient Jane Doe, age 41, was seen today.", [(8, 16, PIIType.PERSON)])]
        r = SpanRetriever(train)
        assert r.propose("完全に無関係なテキストです") == []

    def test_proposed_offsets_index_the_query_exactly(self):
        """Grounding: a proposal must be a real slice of the input document."""
        train = [_rec("t1", "Email me at a@b.com please.", [(12, 19, PIIType.EMAIL)])]
        r = SpanRetriever(train)
        query = "Email me at ravi.kumar@example.org please."
        for p in r.propose(query):
            assert query[p.span.start : p.span.end] == p.span.text

    def test_empty_or_whitespace_candidate_is_rejected(self):
        train = [_rec("t1", "Name: Jane Doe.", [(6, 14, PIIType.PERSON)])]
        r = SpanRetriever(train)
        assert r.propose("Name: .") == []


class TestMerge:
    def test_model_wins_every_conflict(self):
        """Retrieval is a recall aid, not a second opinion.

        A template match is weaker evidence than a prediction conditioned on the
        actual text, so it may only fill gaps.
        """
        text = "Patient Ravi Kumar, age 63."
        model = [PIISpan(start=8, end=18, label=PIIType.PERSON, text="Ravi Kumar")]
        train = [_rec("t1", "Patient Jane Doe, age 41.",
                      [(8, 16, PIIType.PERSON), (22, 24, PIIType.AGE)])]
        props = SpanRetriever(train).propose(text)
        merged = merge_with_model(model, props)
        persons = [s for s in merged if s.label is PIIType.PERSON]
        assert len(persons) == 1 and persons[0].text == "Ravi Kumar"

    def test_fills_a_gap_the_model_missed(self):
        text = "Patient Ravi Kumar, age 63, was seen today."
        train = [_rec("t1", "Patient Jane Doe, age 41, was seen today.",
                      [(8, 16, PIIType.PERSON), (22, 24, PIIType.AGE)])]
        props = SpanRetriever(train).propose(text)
        merged = merge_with_model([], props)
        assert {s.label.value for s in merged} == {"PERSON", "AGE"}

    def test_merge_output_is_sorted_and_non_overlapping(self):
        text = "Patient Ravi Kumar, age 63, was seen today."
        train = [_rec("t1", "Patient Jane Doe, age 41, was seen today.",
                      [(8, 16, PIIType.PERSON), (22, 24, PIIType.AGE)])]
        merged = merge_with_model([], SpanRetriever(train).propose(text))
        assert merged == sorted(merged, key=lambda s: (s.start, s.end))
        for a, b in itertools.pairwise(merged):
            assert b.start >= a.end


class TestNotLeakage:
    """The measurement that licenses this whole module."""

    @pytest.mark.parametrize("split", ["test", "val"])
    def test_train_and_eval_share_almost_no_surface_forms(self, split):
        """If values transferred, retrieval would be smuggling answers.

        Measured 7/693 = 1.0% on test, and PERSON shares 0 of 175. Pinned well
        above the observed value so ordinary regeneration noise does not trip
        it, but far below anything that would make a gazetteer viable.
        """
        p = GOLD / f"{split}.jsonl"
        if not (p.exists() and TRAIN.exists()):
            pytest.skip("corpora not built")
        tr = {s.text for r in _load(TRAIN) for s in r.spans}
        ev = [s.text for r in _load(p) for s in r.spans]
        overlap = sum(1 for v in ev if v in tr) / len(ev)
        assert overlap < 0.05, f"{split}: {overlap:.1%} surface overlap — retrieval would leak"

    def test_index_is_built_from_training_data_only(self):
        """A retriever indexed over eval data would score itself."""
        if not TRAIN.exists():
            pytest.skip("no training data")
        r = SpanRetriever(_load(TRAIN))
        assert len(r) > 0
        gold_texts = {rec.text for rec in _load(GOLD / "test.jsonl")}
        assert not any(rec.text in gold_texts for rec in r._records)


class TestCostProperty:
    def test_retrieval_adds_no_prompt_tokens(self):
        """The design constraint: this must not undermine the cost gate.

        Retrieval runs beside the model, not inside its context window. If this
        ever becomes retrieval-augmented *prompting*, the serving cost changes
        and this test should fail rather than the regression appearing later in
        an economics report.
        """
        import inspect

        import forge.retrieval as mod

        src = inspect.getsource(mod)
        for forbidden in ("build_messages", "system", "few_shot", "prompt"):
            assert f"{forbidden}(" not in src, f"retrieval must not build prompts ({forbidden})"
