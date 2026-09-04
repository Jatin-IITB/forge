"""One-pass Qwen token classification for exact PII spans.

The generative student spends most of its serving time autoregressively spelling
JSON.  This module keeps the same Qwen2 backbone but replaces the vocabulary
head with 77 BIOES labels and reconstructs the ordinary :class:`PIIRecord` in
deterministic code.
"""

from __future__ import annotations

from collections.abc import Sequence

from forge.schema import PIIRecord, PIISpan, PIIType

BOUNDARY_TAGS = ("B", "I", "E", "S")
ID2LABEL = ["O"] + [
    f"{tag}-{pii_type.value}"
    for pii_type in PIIType
    for tag in BOUNDARY_TAGS
]
LABEL2ID = {label: idx for idx, label in enumerate(ID2LABEL)}


def encode_bioes(record: PIIRecord, offsets: Sequence[tuple[int, int]]) -> list[int]:
    """Align exact character spans to tokenizer offsets using overlap.

    Qwen's byte-level tokenizer commonly includes the preceding space in the
    first entity token.  Overlap alignment deliberately labels that token; the
    decoder removes only the extra whitespace.
    """
    labels = [LABEL2ID["O"]] * len(offsets)
    for span in record.spans:
        token_indices = [
            i
            for i, (start, end) in enumerate(offsets)
            if end > start and end > span.start and start < span.end
        ]
        if not token_indices:
            raise ValueError(f"{record.id}: no tokens overlap span {span}")
        if len(token_indices) == 1:
            tags = ["S"]
        else:
            tags = ["B", *(["I"] * (len(token_indices) - 2)), "E"]
        for index, tag in zip(token_indices, tags):
            labels[index] = LABEL2ID[f"{tag}-{span.label.value}"]
    return labels


def _split_label(label_id: int) -> tuple[str, str | None]:
    label = ID2LABEL[label_id]
    if label == "O":
        return "O", None
    tag, pii_type = label.split("-", 1)
    return tag, pii_type


def _allowed_start(label_id: int) -> bool:
    tag, _ = _split_label(label_id)
    return tag in {"O", "B", "S"}


def _allowed_end(label_id: int) -> bool:
    tag, _ = _split_label(label_id)
    return tag in {"O", "E", "S"}


def _allowed_transition(previous: int, current: int) -> bool:
    prev_tag, prev_type = _split_label(previous)
    tag, pii_type = _split_label(current)
    if prev_tag in {"B", "I"}:
        return tag in {"I", "E"} and pii_type == prev_type
    return tag in {"O", "B", "S"}


def constrained_viterbi(logits: Sequence[Sequence[float]]) -> list[int]:
    """Return the highest-scoring structurally valid BIOES sequence."""
    if not logits:
        return []
    n_labels = len(ID2LABEL)
    neg_inf = float("-inf")
    scores = [
        float(logits[0][label_id]) if _allowed_start(label_id) else neg_inf
        for label_id in range(n_labels)
    ]
    backpointers: list[list[int]] = []

    for token_logits in logits[1:]:
        next_scores = [neg_inf] * n_labels
        pointers = [0] * n_labels
        for current in range(n_labels):
            best_previous = max(
                (
                    (scores[previous], previous)
                    for previous in range(n_labels)
                    if _allowed_transition(previous, current)
                ),
                key=lambda item: item[0],
            )
            next_scores[current] = best_previous[0] + float(token_logits[current])
            pointers[current] = best_previous[1]
        scores = next_scores
        backpointers.append(pointers)

    final = max(
        (label_id for label_id in range(n_labels) if _allowed_end(label_id)),
        key=lambda label_id: scores[label_id],
    )
    path = [final]
    for pointers in reversed(backpointers):
        path.append(pointers[path[-1]])
    return list(reversed(path))


def decode_bioes(
    record_id: str,
    text: str,
    offsets: Sequence[tuple[int, int]],
    label_ids: Sequence[int],
    *,
    split: str = "test",
) -> PIIRecord:
    """Convert one valid BIOES path into exact character spans."""
    spans: list[PIISpan] = []
    index = 0
    while index < len(label_ids):
        tag, label = _split_label(label_ids[index])
        if tag == "O" or label is None or offsets[index][1] <= offsets[index][0]:
            index += 1
            continue

        if tag == "S":
            end_index = index
        elif tag == "B":
            end_index = index + 1
            while end_index < len(label_ids):
                end_tag, end_label = _split_label(label_ids[end_index])
                if end_label != label:
                    break
                if end_tag == "E":
                    break
                end_index += 1
            if end_index >= len(label_ids) or _split_label(label_ids[end_index])[0] != "E":
                index += 1
                continue
        else:
            index += 1
            continue

        start = offsets[index][0]
        end = offsets[end_index][1]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if label == PIIType.URL.value:
            while end > start and text[end - 1] in ".,;:!?":
                end -= 1

        if start < end:
            spans.append(
                PIISpan(
                    start=start,
                    end=end,
                    label=PIIType(label),
                    text=text[start:end],
                )
            )
        index = end_index + 1

    return PIIRecord(id=record_id, text=text, spans=spans, split=split)
