"""Training utilities tests — data formatting and config validation."""

import json
import tempfile
from pathlib import Path

from forge.inference import SYSTEM_PROMPT, USER_TEMPLATE
from forge.schema import PIIRecord, PIISpan, PIIType
from forge.train import LORA_DEFAULTS, SFT_DEFAULTS, load_training_data, record_to_chat


def _make_record(text="Contact Alice at alice@x.com", spans=None):
    if spans is None:
        spans = [
            PIISpan(start=8, end=13, label=PIIType.PERSON, text="Alice"),
            PIISpan(start=17, end=28, label=PIIType.EMAIL, text="alice@x.com"),
        ]
    return PIIRecord(id="test_001", text=text, spans=spans, split="train")


def test_record_to_chat_structure():
    msgs = record_to_chat(_make_record())
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_record_to_chat_system_uses_student_prompt():
    msgs = record_to_chat(_make_record())
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_record_to_chat_user_contains_text():
    rec = _make_record(text="Hello Bob", spans=[])
    msgs = record_to_chat(rec)
    assert "Hello Bob" in msgs[1]["content"]
    assert msgs[1]["content"] == USER_TEMPLATE.format(text="Hello Bob")


def test_record_to_chat_assistant_is_valid_json():
    msgs = record_to_chat(_make_record())
    data = json.loads(msgs[2]["content"])
    assert "spans" in data
    assert isinstance(data["spans"], list)


def test_record_to_chat_spans_have_label_and_text_only():
    msgs = record_to_chat(_make_record())
    data = json.loads(msgs[2]["content"])
    for span in data["spans"]:
        assert set(span.keys()) == {"label", "text"}


def test_record_to_chat_no_offsets_in_output():
    msgs = record_to_chat(_make_record())
    data = json.loads(msgs[2]["content"])
    for span in data["spans"]:
        assert "start" not in span
        assert "end" not in span


def test_record_to_chat_empty_spans():
    rec = _make_record(text="No PII here", spans=[])
    msgs = record_to_chat(rec)
    data = json.loads(msgs[2]["content"])
    assert data["spans"] == []


def test_record_to_chat_preserves_labels():
    rec = _make_record()
    msgs = record_to_chat(rec)
    data = json.loads(msgs[2]["content"])
    labels = {s["label"] for s in data["spans"]}
    assert labels == {"PERSON", "EMAIL"}


def test_load_training_data_roundtrip():
    rec = _make_record()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(rec.model_dump_json() + "\n")
        f.write(rec.model_dump_json() + "\n")
        tmp = Path(f.name)

    conversations = load_training_data(tmp)
    assert len(conversations) == 2
    assert all(len(c) == 3 for c in conversations)
    tmp.unlink()


def test_load_training_data_skips_blank_lines():
    rec = _make_record()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(rec.model_dump_json() + "\n")
        f.write("\n")
        f.write("  \n")
        f.write(rec.model_dump_json() + "\n")
        tmp = Path(f.name)

    conversations = load_training_data(tmp)
    assert len(conversations) == 2
    tmp.unlink()


def test_lora_defaults_valid():
    assert LORA_DEFAULTS["r"] > 0
    assert LORA_DEFAULTS["lora_alpha"] > 0
    assert 0 <= LORA_DEFAULTS["lora_dropout"] < 1
    assert len(LORA_DEFAULTS["target_modules"]) > 0


def test_sft_defaults_valid():
    assert SFT_DEFAULTS["num_train_epochs"] > 0
    assert SFT_DEFAULTS["learning_rate"] > 0
    assert SFT_DEFAULTS["seed"] == 42
    assert SFT_DEFAULTS["max_seq_length"] > 0
