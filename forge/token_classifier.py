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
_TYPE_LABEL_IDS = {
    pii_type.value: {
        tag: LABEL2ID[f"{tag}-{pii_type.value}"]
        for tag in BOUNDARY_TAGS
    }
    for pii_type in PIIType
}
_ENDED_LABEL_IDS = [
    LABEL2ID["O"],
    *[
        label_id
        for label_id, label in enumerate(ID2LABEL)
        if label.startswith(("E-", "S-"))
    ],
]
_B_LABEL_IDS = [ids["B"] for ids in _TYPE_LABEL_IDS.values()]
_I_LABEL_IDS = [ids["I"] for ids in _TYPE_LABEL_IDS.values()]
_E_LABEL_IDS = [ids["E"] for ids in _TYPE_LABEL_IDS.values()]
_S_LABEL_IDS = [ids["S"] for ids in _TYPE_LABEL_IDS.values()]
_ALLOWED_START_IDS = [LABEL2ID["O"], *_B_LABEL_IDS, *_S_LABEL_IDS]
_ALLOWED_END_IDS = [LABEL2ID["O"], *_E_LABEL_IDS, *_S_LABEL_IDS]


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
            if labels[index] != LABEL2ID["O"]:
                raise ValueError(
                    f"{record.id}: tokenizer token {offsets[index]} intersects "
                    "more than one gold span; BIOES cannot represent both"
                )
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

        # O/B/S can follow any completed entity. I/E can only continue the
        # same type's B/I. Exploiting that structure makes decoding O(T * L)
        # instead of scanning all 77x77 transitions at every token. This
        # deterministic post-processing is part of serving latency, so the
        # avoidable quadratic implementation would distort the benchmark.
        ended_score, ended_previous = max(
            (scores[label_id], label_id) for label_id in _ENDED_LABEL_IDS
        )
        next_scores[LABEL2ID["O"]] = ended_score + float(token_logits[LABEL2ID["O"]])
        pointers[LABEL2ID["O"]] = ended_previous

        for type_ids in _TYPE_LABEL_IDS.values():
            for tag in ("B", "S"):
                current = type_ids[tag]
                next_scores[current] = ended_score + float(token_logits[current])
                pointers[current] = ended_previous

            continuing_score, continuing_previous = max(
                (scores[type_ids[tag]], type_ids[tag]) for tag in ("B", "I")
            )
            for tag in ("I", "E"):
                current = type_ids[tag]
                next_scores[current] = continuing_score + float(token_logits[current])
                pointers[current] = continuing_previous
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


def constrained_viterbi_batch(logits, lengths: Sequence[int]) -> list[list[int]]:
    """Vectorized BIOES Viterbi for a padded batch of token logits.

    ``logits`` may be a NumPy array or any object accepted by
    :func:`numpy.asarray`, with shape ``[batch, max_tokens, 77]``. The
    transition graph is identical to :func:`constrained_viterbi`; vectorizing
    across records keeps deterministic reconstruction from dominating the
    one-pass model's serving time.
    """
    import numpy as np

    values = np.asarray(logits)
    if values.ndim != 3 or values.shape[2] != len(ID2LABEL):
        raise ValueError(
            f"expected logits [batch, tokens, {len(ID2LABEL)}], got {values.shape}"
        )
    batch_size, max_tokens, n_labels = values.shape
    sequence_lengths = np.asarray(lengths, dtype=np.int64)
    if sequence_lengths.shape != (batch_size,):
        raise ValueError("lengths must contain one entry per batch row")
    if np.any(sequence_lengths < 1) or np.any(sequence_lengths > max_tokens):
        raise ValueError("every sequence length must be in [1, max_tokens]")

    scores = np.full((batch_size, n_labels), -np.inf, dtype=np.float32)
    scores[:, _ALLOWED_START_IDS] = values[:, 0, _ALLOWED_START_IDS]
    backpointers = np.zeros(
        (max_tokens - 1, batch_size, n_labels),
        dtype=np.int16,
    )

    ended_ids = np.asarray(_ENDED_LABEL_IDS)
    b_ids = np.asarray(_B_LABEL_IDS)
    i_ids = np.asarray(_I_LABEL_IDS)
    e_ids = np.asarray(_E_LABEL_IDS)
    s_ids = np.asarray(_S_LABEL_IDS)
    continuing_ids = np.stack([b_ids, i_ids], axis=1)

    for token_index in range(1, max_tokens):
        next_scores = np.full_like(scores, -np.inf)
        pointers = np.zeros((batch_size, n_labels), dtype=np.int16)

        ended_candidates = scores[:, ended_ids]
        # Python's tuple max used by constrained_viterbi chooses the larger
        # label id on a score tie. Reverse argmax preserves that exact rule.
        ended_choice = ended_candidates.shape[1] - 1 - np.argmax(
            ended_candidates[:, ::-1],
            axis=1,
        )
        ended_previous = ended_ids[ended_choice]
        ended_score = ended_candidates[np.arange(batch_size), ended_choice]

        for current_ids in ([LABEL2ID["O"]], b_ids, s_ids):
            next_scores[:, current_ids] = (
                ended_score[:, None] + values[:, token_index, current_ids]
            )
            pointers[:, current_ids] = ended_previous[:, None]

        continuing_candidates = scores[:, continuing_ids]
        continuing_choice = 1 - np.argmax(
            continuing_candidates[:, :, ::-1],
            axis=2,
        )
        continuing_previous = np.take_along_axis(
            continuing_ids[None, :, :],
            continuing_choice[:, :, None],
            axis=2,
        )[:, :, 0]
        continuing_score = np.take_along_axis(
            continuing_candidates,
            continuing_choice[:, :, None],
            axis=2,
        )[:, :, 0]
        for current_ids in (i_ids, e_ids):
            next_scores[:, current_ids] = (
                continuing_score + values[:, token_index, current_ids]
            )
            pointers[:, current_ids] = continuing_previous

        active = sequence_lengths > token_index
        scores[active] = next_scores[active]
        backpointers[token_index - 1] = pointers

    allowed_end_ids = np.asarray(_ALLOWED_END_IDS)
    final_scores = scores[:, allowed_end_ids]
    finals = allowed_end_ids[np.argmax(final_scores, axis=1)]
    paths: list[list[int]] = []
    for row, (length, final) in enumerate(zip(sequence_lengths, finals)):
        path = [int(final)]
        for token_index in range(int(length) - 2, -1, -1):
            path.append(int(backpointers[token_index, row, path[-1]]))
        paths.append(list(reversed(path)))
    return paths


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

        # Qwen has two merges in train_v2 whose token offsets cross a gold
        # boundary. "=key_..." is one token even though "=" is not part of the
        # API key. Address-ending " ." is also one token, while that record's
        # verified address includes the space but not the period. These
        # deterministic label-specific clips preserve the exact source span;
        # the round-trip assertion below guards against adding an unsafe guess.
        if label == PIIType.API_KEY.value:
            while start < end and text[start] in "=:":
                start += 1
        if label in {PIIType.URL.value, PIIType.STREET_ADDRESS.value}:
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


def assert_bioes_round_trip(
    record: PIIRecord,
    offsets: Sequence[tuple[int, int]],
) -> list[int]:
    """Encode and decode one gold record, failing on any boundary loss.

    Returning the labels avoids doing the alignment twice in the training
    pipeline. Exact equality includes start/end, type, and source substring.
    """
    labels = encode_bioes(record, offsets)
    recovered = decode_bioes(
        record.id,
        record.text,
        offsets,
        labels,
        split=record.split,
    )
    if recovered.spans != record.spans:
        expected = [span.model_dump(mode="json") for span in record.spans]
        actual = [span.model_dump(mode="json") for span in recovered.spans]
        raise ValueError(
            f"{record.id}: BIOES round-trip lost exact gold boundaries; "
            f"expected={expected}, recovered={actual}"
        )
    return labels
