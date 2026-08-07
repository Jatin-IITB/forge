"""Eval harness tests — the scoring logic is the foundation of every parity claim."""


from forge.eval import evaluate
from forge.schema import PIIRecord, PIISpan, PIIType


def _rec(id: str, text: str, spans: list[PIISpan], split: str = "test") -> PIIRecord:
    return PIIRecord(id=id, text=text, spans=spans, split=split)


def _span(start: int, end: int, label: PIIType, text: str) -> PIISpan:
    return PIISpan(start=start, end=end, label=label, text=text)


# --- Perfect match ---

def test_perfect_match():
    text = "Call Raj at raj@x.com please"
    spans = [
        _span(5, 8, PIIType.PERSON, "Raj"),
        _span(12, 21, PIIType.EMAIL, "raj@x.com"),
    ]
    gold = [_rec("r1", text, spans)]
    pred = [_rec("r1", text, spans)]

    report = evaluate(gold, pred)
    assert report.micro_f1 == 1.0
    assert report.micro_precision == 1.0
    assert report.micro_recall == 1.0
    assert report.leak_rate == 0.0
    assert report.partial_overlap_f1 == 1.0
    assert report.n_records == 1
    assert report.n_gold_spans == 2
    assert report.n_pred_spans == 2


# --- Total miss (no predictions) ---

def test_no_predictions():
    text = "SSN 123-45-6789 here"
    gold = [_rec("r1", text, [_span(4, 15, PIIType.SSN, "123-45-6789")])]
    pred = [_rec("r1", text, [])]

    report = evaluate(gold, pred)
    assert report.micro_f1 == 0.0
    assert report.micro_recall == 0.0
    assert report.leak_rate == 1.0


# --- All false positives ---

def test_all_false_positives():
    text = "no pii in this text at all"
    gold = [_rec("r1", text, [])]
    pred = [_rec("r1", text, [_span(0, 2, PIIType.PERSON, "no")])]

    report = evaluate(gold, pred)
    assert report.micro_precision == 0.0
    assert report.micro_f1 == 0.0
    assert report.n_pred_spans == 1
    assert report.n_gold_spans == 0


# --- Partial match (some correct, some missed, some spurious) ---

def test_partial_match():
    text = "Hi Raj, email raj@x.com or call +91 12345 67890."
    gold_spans = [
        _span(3, 6, PIIType.PERSON, "Raj"),
        _span(14, 23, PIIType.EMAIL, "raj@x.com"),
        _span(32, 47, PIIType.PHONE, "+91 12345 67890"),
    ]
    pred_spans = [
        _span(3, 6, PIIType.PERSON, "Raj"),       # TP
        _span(14, 23, PIIType.EMAIL, "raj@x.com"), # TP
        # PHONE missed → FN
        _span(0, 2, PIIType.PERSON, "Hi"),          # FP
    ]
    gold = [_rec("r1", text, gold_spans)]
    pred = [_rec("r1", text, pred_spans)]

    report = evaluate(gold, pred)
    assert report.per_type["PERSON"].tp == 1
    assert report.per_type["PERSON"].fp == 1
    assert report.per_type["EMAIL"].tp == 1
    assert report.per_type["PHONE"].fn == 1
    # micro: TP=2, FP=1, FN=1 → P=2/3, R=2/3, F1=2/3
    assert abs(report.micro_f1 - 2 / 3) < 1e-9


# --- Wrong label (same span, different type) ---

def test_wrong_label_counts_as_fp_and_fn():
    text = "card 4111111111111111 here"
    gold = [_rec("r1", text, [_span(5, 20, PIIType.CREDIT_CARD, "411111111111111")])]
    pred = [_rec("r1", text, [_span(5, 20, PIIType.BANK_ACCOUNT, "411111111111111")])]

    report = evaluate(gold, pred)
    assert report.per_type["CREDIT_CARD"].fn == 1
    assert report.per_type["BANK_ACCOUNT"].fp == 1
    assert report.micro_f1 == 0.0


# --- Off-by-one boundary (exact match is strict) ---

def test_off_by_one_is_a_miss():
    text = "Name: Alice here"
    gold = [_rec("r1", text, [_span(6, 11, PIIType.PERSON, "Alice")])]
    pred = [_rec("r1", text, [_span(6, 12, PIIType.PERSON, "Alice ")])]

    report = evaluate(gold, pred)
    assert report.micro_f1 == 0.0  # exact match fails
    assert report.partial_overlap_f1 > 0.0  # but partial overlap is non-zero


# --- Leak rate ---

def test_leak_rate_partial_coverage():
    text = "SSN 123-45-6789 and PAN ABCDE1234F done"
    gold = [_rec("r1", text, [
        _span(4, 15, PIIType.SSN, "123-45-6789"),
        _span(24, 34, PIIType.PAN, "ABCDE1234F"),
    ])]
    # Only SSN predicted, PAN missed entirely
    pred = [_rec("r1", text, [
        _span(4, 15, PIIType.SSN, "123-45-6789"),
    ])]

    report = evaluate(gold, pred)
    # SSN: 11 chars covered, PAN: 10 chars leaked. Total gold = 21, leaked = 10
    assert abs(report.leak_rate - 10 / 21) < 1e-9


# --- Missing prediction record ---

def test_missing_prediction_record():
    text = "Aadhaar 1234 5678 9012"
    gold = [_rec("r1", text, [_span(8, 22, PIIType.AADHAAR, "1234 5678 9012")])]
    pred = []  # no prediction for this record at all

    report = evaluate(gold, pred)
    assert report.micro_recall == 0.0
    assert report.leak_rate == 1.0


# --- Negative examples (no PII in gold, no PII in pred) ---

def test_negative_example_perfect():
    text = "The weather is nice today."
    gold = [_rec("r1", text, [])]
    pred = [_rec("r1", text, [])]

    report = evaluate(gold, pred)
    assert report.micro_f1 == 0.0  # no spans → 0/0, convention = 0
    assert report.leak_rate == 0.0


# --- Schema validity ---

def test_schema_validity():
    text = "test"
    gold = [_rec("r1", text, []), _rec("r2", text, []), _rec("r3", text, [])]
    pred = [_rec("r1", text, []), _rec("r2", text, []), _rec("r3", text, [])]

    report = evaluate(gold, pred, schema_valid_count=2)
    assert abs(report.schema_validity - 2 / 3) < 1e-9


# --- High severity recall ---

def test_high_severity_recall():
    text = "SSN 123-45-6789 card 4111 1111 1111 1111 done"
    gold = [_rec("r1", text, [
        _span(4, 15, PIIType.SSN, "123-45-6789"),
        _span(21, 40, PIIType.CREDIT_CARD, "4111 1111 1111 1111"),
    ])]
    pred = [_rec("r1", text, [
        _span(4, 15, PIIType.SSN, "123-45-6789"),
        # CREDIT_CARD missed
    ])]

    report = evaluate(gold, pred)
    hs = report.high_severity_recall()
    assert hs["SSN"] == 1.0
    assert hs["CREDIT_CARD"] == 0.0


# --- Multi-record scoring ---

def test_multi_record():
    t1 = "email: a@b.com"
    t2 = "name: Bob"
    gold = [
        _rec("r1", t1, [_span(7, 14, PIIType.EMAIL, "a@b.com")]),
        _rec("r2", t2, [_span(6, 9, PIIType.PERSON, "Bob")]),
    ]
    pred = [
        _rec("r1", t1, [_span(7, 14, PIIType.EMAIL, "a@b.com")]),  # TP
        _rec("r2", t2, []),  # FN
    ]

    report = evaluate(gold, pred)
    assert report.micro_precision == 1.0  # 1 TP, 0 FP
    assert report.micro_recall == 0.5  # 1 TP, 1 FN
    assert report.n_records == 2


# --- Format table smoke test ---

def test_format_table_runs():
    text = "Hello Raj"
    gold = [_rec("r1", text, [_span(6, 9, PIIType.PERSON, "Raj")])]
    pred = [_rec("r1", text, [_span(6, 9, PIIType.PERSON, "Raj")])]

    report = evaluate(gold, pred)
    table = report.format_table()
    assert "micro-F1" in table
    assert "PERSON" in table


# --- Overlapping pred spans (regression: partial overlap must not double-count) ---

def test_overlapping_pred_spans_no_double_count():
    text = "Name: Alice Smith here"
    gold = [_rec("r1", text, [_span(6, 17, PIIType.PERSON, "Alice Smith")])]
    pred = [_rec("r1", text, [
        _span(6, 11, PIIType.PERSON, "Alice"),
        _span(12, 17, PIIType.PERSON, "Smith"),
    ])]

    report = evaluate(gold, pred)
    assert report.partial_overlap_f1 <= 1.0
    assert report.partial_overlap_f1 > 0.0


# --- Extra prediction records not in gold ---

def test_extra_pred_records_do_not_inflate_precision():
    text = "Hello world"
    gold = [_rec("r1", text, [_span(6, 11, PIIType.PERSON, "world")])]
    pred = [
        _rec("r1", text, [_span(6, 11, PIIType.PERSON, "world")]),
        _rec("r99", "phantom text", [_span(0, 7, PIIType.PERSON, "phantom")]),
    ]

    report = evaluate(gold, pred)
    assert report.micro_precision == 1.0
    assert report.micro_f1 == 1.0


# --- Empty gold list ---

def test_empty_gold():
    report = evaluate([], [])
    assert report.micro_f1 == 0.0
    assert report.leak_rate == 0.0
    assert report.schema_validity == 1.0


# --- Wrong label still covers characters for leak rate ---

def test_wrong_label_still_covers_chars_for_leak_rate():
    text = "SSN 123-45-6789"
    gold = [_rec("r1", text, [_span(4, 15, PIIType.SSN, "123-45-6789")])]
    pred = [_rec("r1", text, [_span(4, 15, PIIType.BANK_ACCOUNT, "123-45-6789")])]

    report = evaluate(gold, pred)
    assert report.micro_f1 == 0.0  # wrong label → no exact match
    assert report.leak_rate == 0.0  # but characters ARE covered


# --- Partial overlap F1 value verified ---

def test_partial_overlap_f1_value():
    text = "Name: Alice!"
    gold = [_rec("r1", text, [_span(6, 11, PIIType.PERSON, "Alice")])]
    pred = [_rec("r1", text, [_span(6, 12, PIIType.PERSON, "Alice!")])]

    report = evaluate(gold, pred)
    # gold chars: 5 (positions 6-10), pred chars: 6 (positions 6-11)
    # overlap: 5 (positions 6-10)
    # precision = 5/6, recall = 5/5 = 1.0
    # F1 = 2 * (5/6) * 1 / (5/6 + 1) = (10/6) / (11/6) = 10/11
    assert abs(report.partial_overlap_f1 - 10 / 11) < 1e-9
