"""Structural invariants of the committed gold set.

These read the **committed bytes**, unlike the builder tests, which regenerate
the data and can therefore only prove the generator is self-consistent. That
distinction is not academic: the ADR 0011 clock defect survived a fully green
suite for weeks because every test rebuilt the data the same wrong way.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from forge.schema import PIIRecord

GOLD = Path("data/gold")
TRAIN = Path("data/train.jsonl")


def _load(path: Path) -> list[PIIRecord]:
    return [
        PIIRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def test_split() -> list[PIIRecord]:
    return _load(GOLD / "test.jsonl")


@pytest.fixture(scope="module")
def dev_split() -> list[PIIRecord]:
    return _load(GOLD / "dev.jsonl")


@pytest.fixture(scope="module")
def train_texts() -> set[str]:
    return {r.text for r in _load(TRAIN)} if TRAIN.exists() else set()


@pytest.fixture(scope="module")
def val_split() -> list[PIIRecord]:
    path = GOLD / "val.jsonl"
    if not path.exists():
        pytest.skip("validation split not built yet (scripts/build_validation.py)")
    return _load(path)


class TestFrozenTestSplit:
    """The test split carries every published number. It must stay clean."""

    def test_no_training_leakage(self, test_split, train_texts):
        """The invariant the whole parity claim rests on.

        Measured clean at 0/385 on 2026-09-03. If this ever fails, every
        reported F1 is invalid and no gate verdict may be published.
        """
        if not train_texts:
            pytest.skip("no training data present")
        leaked = [r.id for r in test_split if r.text in train_texts]
        assert leaked == [], f"{len(leaked)} test records leaked into training data: {leaked[:5]}"

    def test_spans_index_the_committed_text_exactly(self, test_split):
        for r in test_split:
            for s in r.spans:
                assert r.text[s.start : s.end] == s.text, (
                    f"{r.id}: text[{s.start}:{s.end}]={r.text[s.start:s.end]!r} "
                    f"!= span.text={s.text!r}"
                )

    def test_spans_do_not_overlap(self, test_split):
        """Overlapping gold spans would make exact-match scoring ill-defined."""
        for r in test_split:
            ordered = sorted(r.spans, key=lambda s: (s.start, s.end))
            for x, y in itertools.pairwise(ordered):
                assert y.start >= x.end, f"{r.id}: {x.label.value} and {y.label.value} overlap"

    def test_record_count_is_frozen(self, test_split):
        assert len(test_split) == 385

    def test_known_duplicate_count_has_not_grown(self, test_split):
        """20 byte-identical duplicate records — a known, documented defect.

        Effective n is 365, not 385, so bootstrap intervals are marginally
        optimistic. Measured impact is immaterial (teacher F1 unchanged,
        student +0.0050, both far inside a +/-0.042 CI), so the freeze is kept
        rather than regenerating a test set after seeing results. Pinned here so
        the number cannot drift unnoticed. See WP-0e.
        """
        unique = {r.text for r in test_split}
        assert len(test_split) - len(unique) == 20


class TestLabelSemantics:
    """Labels must be right by *meaning*, not merely exact by offset.

    Construction-based generation guarantees offsets — the filler knows where it
    put the value — but nothing checks the sentence agrees. A {DATE_OF_BIRTH}
    slot in "Order #1234 confirmed on <date>" is offset-perfect and false, and a
    corpus of those teaches that any date is a birth date.

    The frozen gold survives this because its templates are hand-written with
    the anchor built in ("age {AGE}", "born {DOB}"). A teacher-written carrier
    corpus generated later did not: 20/145 AGE and 88/325 DATE_OF_BIRTH spans
    sit in contexts that contradict the label. These tests keep the gold on the
    right side of that line.
    """

    @pytest.mark.parametrize("split", ["test", "val", "dev"])
    def test_no_label_is_contradicted_by_its_context(self, split):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from audit_gold import Audit, audit_semantics

        path = GOLD / f"{split}.jsonl"
        if not path.exists():
            pytest.skip(f"{split} split not built")
        a = Audit()
        stats = audit_semantics(_load(path), split, a)
        offenders = {k: v for k, v in stats.items() if v["contradicted"]}
        assert offenders == {}, f"{split}: semantically contradicted labels {offenders}"

    def test_every_birth_date_has_a_birth_anchor(self, test_split):
        """DATE_OF_BIRTH is the type where a missing anchor is itself damning.

        A bare date means nothing on its own — unlike an Aadhaar number, whose
        shape identifies it. Gold is 25/25 anchored; the generated corpus was
        8/325.
        """
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from audit_gold import _ANCHOR, _ANCHOR_WINDOW

        pat = _ANCHOR["DATE_OF_BIRTH"]
        for r in test_split:
            for s in r.spans:
                if s.label.value != "DATE_OF_BIRTH":
                    continue
                lo = max(0, s.start - _ANCHOR_WINDOW)
                hi = min(len(r.text), s.end + _ANCHOR_WINDOW)
                window = r.text[lo : s.start] + " " + r.text[s.end : hi]
                assert pat.search(window), f"{r.id}: DOB {s.text!r} has no birth anchor in {window!r}"


class TestValidationSplit:
    """The clean model-selection split (WP-0d), built by build_validation.py.

    Its whole purpose is being disjoint from everything else, so that is what
    gets asserted — on the committed bytes, every run.
    """

    def test_disjoint_from_training_data(self, val_split, train_texts):
        if not train_texts:
            pytest.skip("no training data present")
        leaked = [r.id for r in val_split if r.text in train_texts]
        assert leaked == [], f"{len(leaked)} val records leaked from training data"

    def test_disjoint_from_the_frozen_test_set(self, val_split, test_split):
        """Selecting on text that also scores the final number would be circular."""
        test_texts = {r.text for r in test_split}
        overlap = [r.id for r in val_split if r.text in test_texts]
        assert overlap == [], f"{len(overlap)} val records also appear in test"

    def test_disjoint_from_dev(self, val_split, dev_split):
        dev_texts = {r.text for r in dev_split}
        overlap = [r.id for r in val_split if r.text in dev_texts]
        assert overlap == [], f"{len(overlap)} val records also appear in dev"

    def test_has_no_internal_duplicates(self, val_split):
        assert len(val_split) == len({r.text for r in val_split})

    def test_covers_every_high_severity_type_more_densely_than_test(self, val_split, test_split):
        """Model selection needs at least the resolution of the final measurement."""
        from collections import Counter

        from forge.schema import HIGH_SEVERITY

        v = Counter(s.label.value for r in val_split for s in r.spans)
        t = Counter(s.label.value for r in test_split for s in r.spans)
        for label in sorted(x.value for x in HIGH_SEVERITY):
            assert v[label] >= t[label], f"{label}: val has {v[label]} vs test {t[label]}"


class TestDevSplit:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "WP-0d: dev is 79.4% contaminated (150/189 records appear verbatim in "
            "train.jsonl). The data engine was seeded from dev and forge/dedup.py was "
            "given the test split to check against, so this was never caught. Dev is "
            "unusable for model selection until a clean validation split exists. "
            "strict=True so that fixing it forces this test to be updated rather than "
            "silently passing."
        ),
    )
    def test_no_training_leakage(self, dev_split, train_texts):
        if not train_texts:
            pytest.skip("no training data present")
        leaked = [r.id for r in dev_split if r.text in train_texts]
        assert leaked == []

    def test_spans_index_the_committed_text_exactly(self, dev_split):
        for r in dev_split:
            for s in r.spans:
                assert r.text[s.start : s.end] == s.text
