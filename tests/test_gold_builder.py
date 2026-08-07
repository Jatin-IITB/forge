"""Gold set builder: reproducibility, coverage, and schema conformance."""

import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from forge.schema import HIGH_SEVERITY, PIIRecord, PIIType

GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "gold"
DEV_PATH = GOLD_DIR / "dev.jsonl"
TEST_PATH = GOLD_DIR / "test.jsonl"
BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_gold.py"


def _load(path: Path) -> list[PIIRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PIIRecord.model_validate_json(line))
    return records


@pytest.fixture(scope="module")
def dev_records() -> list[PIIRecord]:
    if not DEV_PATH.exists():
        pytest.skip("dev.jsonl not built yet — run `make gold`")
    return _load(DEV_PATH)


@pytest.fixture(scope="module")
def test_records() -> list[PIIRecord]:
    if not TEST_PATH.exists():
        pytest.skip("test.jsonl not built yet — run `make gold`")
    return _load(TEST_PATH)


def test_dev_records_validate(dev_records: list[PIIRecord]) -> None:
    assert len(dev_records) >= 150


def test_test_records_validate(test_records: list[PIIRecord]) -> None:
    assert len(test_records) >= 300


def test_splits_are_correct(dev_records: list[PIIRecord], test_records: list[PIIRecord]) -> None:
    for r in dev_records:
        assert r.split == "dev", f"{r.id} has split={r.split}"
    for r in test_records:
        assert r.split == "test", f"{r.id} has split={r.split}"


def test_ids_are_unique(dev_records: list[PIIRecord], test_records: list[PIIRecord]) -> None:
    all_ids = [r.id for r in dev_records] + [r.id for r in test_records]
    assert len(all_ids) == len(set(all_ids))


def test_all_pii_types_covered_in_test(test_records: list[PIIRecord]) -> None:
    type_counts: Counter[str] = Counter()
    for r in test_records:
        for s in r.spans:
            type_counts[s.label.value] += 1
    for t in PIIType:
        assert type_counts[t.value] >= 10, f"{t.value} has only {type_counts[t.value]} occurrences in test"


def test_high_severity_well_represented(test_records: list[PIIRecord]) -> None:
    type_counts: Counter[str] = Counter()
    for r in test_records:
        for s in r.spans:
            type_counts[s.label.value] += 1
    for t in HIGH_SEVERITY:
        assert type_counts[t.value] >= 15, f"high-severity {t.value} has only {type_counts[t.value]} in test"


def test_negative_examples_exist(test_records: list[PIIRecord]) -> None:
    negatives = [r for r in test_records if not r.spans]
    assert len(negatives) >= 10, f"only {len(negatives)} negative examples in test"


def test_source_is_faker(dev_records: list[PIIRecord], test_records: list[PIIRecord]) -> None:
    for r in dev_records + test_records:
        assert r.source == "synthetic:faker"


def test_span_text_matches_offsets(dev_records: list[PIIRecord], test_records: list[PIIRecord]) -> None:
    for r in dev_records + test_records:
        for s in r.spans:
            assert r.text[s.start : s.end] == s.text, f"offset mismatch in {r.id}"


def test_reproducibility() -> None:
    """Running the builder twice with the same seed produces identical output."""
    if not TEST_PATH.exists():
        pytest.skip("test.jsonl not built yet — run `make gold`")
    committed_test = TEST_PATH.read_text(encoding="utf-8")
    committed_dev = DEV_PATH.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), "--output-dir", tmpdir],
            check=True,
            capture_output=True,
        )
        fresh_test = (Path(tmpdir) / "test.jsonl").read_text(encoding="utf-8")
        fresh_dev = (Path(tmpdir) / "dev.jsonl").read_text(encoding="utf-8")

    assert committed_test == fresh_test, "test.jsonl is not reproducible"
    assert committed_dev == fresh_dev, "dev.jsonl is not reproducible"
