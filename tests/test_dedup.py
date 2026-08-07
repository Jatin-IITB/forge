"""Dedup tests — exact, near-duplicate, and gold leakage detection."""

from forge.dedup import DedupResult, _char_ngrams, dedup_training_data, jaccard
from forge.schema import PIIRecord


def _rec(rid: str, text: str, spans: list | None = None) -> PIIRecord:
    return PIIRecord(id=rid, text=text, spans=spans or [], split="train")


def _gold(rid: str, text: str) -> PIIRecord:
    return PIIRecord(id=rid, text=text, spans=[], split="test")


# --- char_ngrams ---


def test_char_ngrams_basic():
    ngrams = _char_ngrams("abcdef", 3)
    assert ngrams == {"abc", "bcd", "cde", "def"}


def test_char_ngrams_short_text():
    ngrams = _char_ngrams("ab", 5)
    assert ngrams == {"ab"}


def test_char_ngrams_exact_length():
    ngrams = _char_ngrams("abcde", 5)
    assert ngrams == {"abcde"}


# --- jaccard ---


def test_jaccard_identical():
    assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_disjoint():
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial():
    assert abs(jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 0.5) < 1e-9


def test_jaccard_empty():
    assert jaccard(set(), set()) == 1.0


# --- dedup_training_data ---


def test_exact_duplicates_removed():
    train = [
        _rec("r1", "Hello Alice"),
        _rec("r2", "Hello Alice"),
        _rec("r3", "Hello Bob"),
    ]
    result = dedup_training_data(train)
    assert len(result.kept) == 2
    assert result.removed_exact == 1


def test_near_duplicates_removed():
    train = [
        _rec("r1", "Hello Alice, how are you doing today?"),
        _rec("r2", "Hello Alice, how are you doing today!"),
    ]
    result = dedup_training_data(train, near_threshold=0.85)
    assert len(result.kept) == 1
    assert result.removed_near == 1


def test_different_texts_kept():
    train = [
        _rec("r1", "This is a completely different text about weather and climate patterns in the region."),
        _rec("r2", "Contact Alice at alice@example.com for the quarterly financial report details."),
    ]
    result = dedup_training_data(train)
    assert len(result.kept) == 2
    assert result.total_removed == 0


def test_gold_leakage_exact():
    gold = [_gold("g1", "SSN 123-45-6789 belongs to Alice")]
    train = [
        _rec("r1", "SSN 123-45-6789 belongs to Alice"),
        _rec("r2", "Different text entirely about Bob"),
    ]
    result = dedup_training_data(train, gold=gold)
    assert len(result.kept) == 1
    assert result.removed_leakage == 1
    assert result.kept[0].id == "r2"


def test_gold_leakage_near():
    gold = [_gold("g1", "SSN 123-45-6789 belongs to Alice Smith in New York")]
    train = [
        _rec("r1", "SSN 123-45-6789 belongs to Alice Smith in New York!"),
    ]
    result = dedup_training_data(train, gold=gold, near_threshold=0.85)
    assert len(result.kept) == 0
    assert result.removed_leakage == 1


def test_no_gold_no_leakage_check():
    train = [_rec("r1", "Hello world"), _rec("r2", "Goodbye world")]
    result = dedup_training_data(train, gold=None)
    assert len(result.kept) == 2
    assert result.removed_leakage == 0


def test_empty_train():
    result = dedup_training_data([])
    assert len(result.kept) == 0
    assert result.total_removed == 0


def test_summary():
    result = DedupResult(kept=[], removed_exact=2, removed_near=1, removed_leakage=3)
    s = result.summary()
    assert s["total_removed"] == 6
    assert s["kept"] == 0


def test_order_preserved():
    train = [
        _rec("r1", "First unique text about machine learning and natural language processing"),
        _rec("r2", "Second unique text about computer vision and image recognition systems"),
        _rec("r3", "Third unique text about reinforcement learning and decision making processes"),
    ]
    result = dedup_training_data(train)
    assert [r.id for r in result.kept] == ["r1", "r2", "r3"]
