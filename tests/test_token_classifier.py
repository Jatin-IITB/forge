"""BIOES alignment and constrained decoding tests.

Skipped unless the training extras are installed. `make install` deliberately
pulls only [dev,data] so the eval path stays lightweight, and a torch import at
module scope made the whole suite uncollectable on a fresh clone rather than
skipping the handful of tests that actually need it.
"""

import pytest

torch = pytest.importorskip("torch", reason="needs the [train] extra")
transformers = pytest.importorskip("transformers", reason="needs the [train] extra")
from transformers import Qwen2Config

from forge.schema import PIIRecord, PIISpan, PIIType
from forge.token_classifier import (
    ID2LABEL,
    LABEL2ID,
    assert_bioes_round_trip,
    constrained_viterbi,
    constrained_viterbi_batch,
    decode_bioes,
    encode_bioes,
)
from forge.token_model import ForgeQwen2ForTokenClassification


def test_overlap_alignment_and_whitespace_trim_round_trip():
    text = "Call Jessica Holmes."
    record = PIIRecord(
        id="r",
        text=text,
        spans=[PIISpan(start=5, end=19, label=PIIType.PERSON, text="Jessica Holmes")],
    )
    offsets = [(0, 4), (4, 12), (12, 19), (19, 20)]
    labels = encode_bioes(record, offsets)
    assert [ID2LABEL[x] for x in labels] == ["O", "B-PERSON", "E-PERSON", "O"]
    decoded = decode_bioes("r", text, offsets, labels)
    assert decoded.spans == record.spans


def test_single_token_entity_uses_s_tag():
    record = PIIRecord(
        id="r",
        text="Alice",
        spans=[PIISpan(start=0, end=5, label=PIIType.PERSON, text="Alice")],
    )
    assert encode_bioes(record, [(0, 5)]) == [LABEL2ID["S-PERSON"]]


def test_url_terminal_period_is_removed():
    text = "See https://example.com/."
    offsets = [(0, 3), (3, 25)]
    labels = [LABEL2ID["O"], LABEL2ID["S-URL"]]
    decoded = decode_bioes("r", text, offsets, labels)
    assert decoded.spans[0].text == "https://example.com/"


def test_cross_token_boundary_clips_round_trip_exactly():
    api_text = "Authorization: =key_sWBe7P69xglpBgnX8tLVQsmcFNC2RVN9HU."
    api_record = PIIRecord(
        id="api",
        text=api_text,
        spans=[
            PIISpan(
                start=16,
                end=54,
                label=PIIType.API_KEY,
                text="key_sWBe7P69xglpBgnX8tLVQsmcFNC2RVN9HU",
            )
        ],
    )
    assert_bioes_round_trip(api_record, [(0, 15), (15, 19), (19, 54), (54, 55)])

    address_text = "Delivery to: H.No. 911, Kara Road, Khora . Call later."
    address_record = PIIRecord(
        id="address",
        text=address_text,
        spans=[
            PIISpan(
                start=13,
                end=41,
                label=PIIType.STREET_ADDRESS,
                text="H.No. 911, Kara Road, Khora ",
            )
        ],
    )
    assert_bioes_round_trip(address_record, [(0, 13), (13, 40), (40, 42), (42, 54)])


def test_token_intersecting_two_spans_fails_loudly():
    text = "Alice/Bob"
    record = PIIRecord(
        id="collision",
        text=text,
        spans=[
            PIISpan(start=0, end=5, label=PIIType.PERSON, text="Alice"),
            PIISpan(start=6, end=9, label=PIIType.USERNAME, text="Bob"),
        ],
    )
    with pytest.raises(ValueError, match="BIOES cannot represent both"):
        encode_bioes(record, [(0, 9)])


def test_viterbi_rejects_illegal_start_and_transition():
    n = len(ID2LABEL)
    logits = [[0.0] * n for _ in range(2)]
    logits[0][LABEL2ID["I-PERSON"]] = 100.0
    logits[0][LABEL2ID["B-PERSON"]] = 10.0
    logits[1][LABEL2ID["I-EMAIL"]] = 100.0
    logits[1][LABEL2ID["E-PERSON"]] = 10.0
    assert constrained_viterbi(logits) == [
        LABEL2ID["B-PERSON"],
        LABEL2ID["E-PERSON"],
    ]


def test_vectorized_viterbi_matches_single_record_decoder():
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(3, 7, len(ID2LABEL), generator=generator).numpy()
    lengths = [7, 4, 1]
    expected = [
        constrained_viterbi(logits[index, :length].tolist())
        for index, length in enumerate(lengths)
    ]
    assert constrained_viterbi_batch(logits, lengths) == expected


def test_all_label_paths_have_expected_width():
    assert len(ID2LABEL) == 77
    assert len(LABEL2ID) == 77


def test_full_attention_token_model_forward_and_weighted_loss():
    config = Qwen2Config(
        vocab_size=100,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_labels=len(ID2LABEL),
    )
    config.forge_full_attention = True
    config._attn_implementation = "eager"
    model = ForgeQwen2ForTokenClassification(config)
    output = model(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        labels=torch.tensor([[0, 1, 3], [0, 5, -100]]),
    )
    assert output.logits.shape == (2, 3, len(ID2LABEL))
    assert torch.isfinite(output.loss)
