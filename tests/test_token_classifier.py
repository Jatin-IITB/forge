"""BIOES alignment and constrained decoding tests."""

import torch
from transformers import Qwen2Config

from forge.schema import PIIRecord, PIISpan, PIIType
from forge.token_classifier import (
    ID2LABEL,
    LABEL2ID,
    constrained_viterbi,
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
