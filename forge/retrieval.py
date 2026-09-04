"""Retrieval-based span proposal — grounding a redaction in retrieved source text.

## Why this shape, and not the obvious one

The obvious design is a gazetteer: index the PII surface forms seen in training
and look them up at inference. **Measured, it is worthless here.** Train and test
share **7 of 693 surface forms — 1.0%**, and `PERSON` shares exactly zero of 175.
The corpus is Faker-generated, so every split draws novel values by construction.
A surface index could recover 1% of test spans at best.

That same measurement is what makes this module legitimate rather than leakage:
**values provably do not transfer between splits, so nothing retrieved here can
carry an answer with it.** What transfers is *structure*.

    test records whose exact carrier shape appears in train   60.0%
    test spans sitting in one of those shapes                 59.4%  (413/695)

So the index is over **carrier shapes**, and what gets retrieved is a *template*,
never a value. A neighbour tells us "in text like this, the run between 'born'
and 'has been enrolled' is a DATE_OF_BIRTH". The value is then read out of the
input document itself.

## Why not retrieval-augmented prompting

Injecting k neighbours as few-shot context is the standard move and it is the
wrong one here. The serving cost gate is measured in tokens, and the project is
chasing an ~80x cost target that a longer prompt directly undermines. This
module adds **zero prompt tokens**: retrieval runs beside the model, not inside
its context window, and proposes candidate spans that are merged afterwards.

## Grounding

Every proposed span is a slice of the *input* document at offsets computed by
alignment. Nothing is generated, so a hallucinated identifier is not
representable: a candidate that cannot be located in the source is never created
in the first place.

## Honest limitation

This works because the corpus is template-generated. On natural text, carrier
shapes would not repeat at 60% and the alignment would find far fewer anchors.
Read the ceiling — 59.4% of spans — as a property of this dataset, not of the
method.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.dedup import _char_ngrams, jaccard
from forge.schema import PIIRecord, PIISpan, PIIType

# Fixed-text fragments shorter than this are too common to anchor on: a two-space
# gap or a lone comma matches everywhere and produces garbage alignments.
_MIN_ANCHOR = 3


@dataclass(frozen=True)
class Neighbour:
    record: PIIRecord
    similarity: float


@dataclass(frozen=True)
class Proposal:
    span: PIISpan
    similarity: float
    source_id: str


def _skeleton(record: PIIRecord) -> tuple[list[str], list[PIIType]]:
    """Split a record into fixed text fragments and the labels between them.

    "Patient Jane Doe, age 41, was seen." with PERSON and AGE spans becomes
    fragments ["Patient ", ", age ", ", was seen."] and labels [PERSON, AGE].
    The fragments are what we align against; the labels are what we transfer.
    """
    spans = sorted(record.spans, key=lambda s: (s.start, s.end))
    fragments: list[str] = []
    labels: list[PIIType] = []
    cursor = 0
    for s in spans:
        if s.start < cursor:  # overlapping gold spans — not alignable
            return [], []
        fragments.append(record.text[cursor : s.start])
        labels.append(s.label)
        cursor = s.end
    fragments.append(record.text[cursor:])
    return fragments, labels


class SpanRetriever:
    """Character n-gram index over training carriers.

    Deliberately not embeddings. Records from one template share their fixed
    prose verbatim, so character n-gram Jaccard separates templates cleanly and
    needs no model, no GPU and no extra dependency — which keeps the
    "runs offline on a laptop" property intact. `forge.dedup` already computes
    exactly this similarity for near-duplicate detection, so the two agree by
    construction.
    """

    def __init__(self, records: list[PIIRecord], ngram_size: int = 5) -> None:
        self.ngram_size = ngram_size
        self._records = [r for r in records if r.spans]
        self._ngrams = [_char_ngrams(r.text, ngram_size) for r in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def retrieve(self, text: str, k: int = 3, min_similarity: float = 0.30) -> list[Neighbour]:
        """Top-k training carriers by character n-gram Jaccard."""
        q = _char_ngrams(text, self.ngram_size)
        if not q:
            return []
        scored = [
            Neighbour(rec, sim)
            for rec, ng in zip(self._records, self._ngrams)
            if (sim := jaccard(q, ng)) >= min_similarity
        ]
        scored.sort(key=lambda n: -n.similarity)
        return scored[:k]

    def propose(self, text: str, k: int = 3, min_similarity: float = 0.30) -> list[Proposal]:
        """Candidate spans for `text`, aligned from retrieved neighbours.

        Neighbours are tried best-first and the first one that aligns wins. A
        partial alignment is discarded rather than kept: half a template match
        means the templates differ, and the surviving half is not evidence.
        """
        for nb in self.retrieve(text, k=k, min_similarity=min_similarity):
            proposals = _align(text, nb)
            if proposals:
                return proposals
        return []


def _align(text: str, neighbour: Neighbour) -> list[Proposal]:
    """Project a neighbour's span layout onto `text` by anchoring on fixed text.

    Walks the neighbour's fixed fragments left to right through `text`. Whatever
    sits between two consecutive anchors is the candidate span, and it inherits
    the neighbour's label for that position. Returns [] unless every fragment
    anchors, so a template that only partly matches contributes nothing.
    """
    fragments, labels = _skeleton(neighbour.record)
    if not labels:
        return []

    proposals: list[Proposal] = []
    cursor = 0
    for i, label in enumerate(labels):
        prefix = fragments[i]
        if len(prefix.strip()) >= _MIN_ANCHOR or i == 0:
            idx = text.find(prefix, cursor) if prefix else cursor
            if idx < 0:
                return []
            start = idx + len(prefix)
        else:
            # Fragment too short to anchor on; fall back to continuing from the
            # previous span's end rather than matching whitespace anywhere.
            start = cursor + len(prefix)

        suffix = fragments[i + 1]
        if suffix.strip():
            end = text.find(suffix, start)
            if end < 0:
                return []
        else:
            end = len(text)  # last span runs to the end of the document

        if end <= start:
            return []
        value = text[start:end]
        if not value.strip():
            return []
        proposals.append(
            Proposal(
                span=PIISpan(start=start, end=end, label=label, text=value),
                similarity=neighbour.similarity,
                source_id=neighbour.record.id,
            )
        )
        cursor = end
    return proposals


def merge_with_model(
    model_spans: list[PIISpan],
    proposals: list[Proposal],
) -> list[PIISpan]:
    """Add retrieved candidates only where the model claimed nothing.

    The model wins every conflict. Retrieval is a recall aid for spans the model
    missed entirely, not a second opinion on spans it already found — a template
    match is weaker evidence than a prediction conditioned on the actual text,
    and letting it override would trade a measured strength for an unmeasured one.
    """
    merged = list(model_spans)
    for p in proposals:
        if any(p.span.start < m.end and m.start < p.span.end for m in merged):
            continue
        merged.append(p.span)
    return sorted(merged, key=lambda s: (s.start, s.end))
