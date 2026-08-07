"""Near-duplicate detection for training data.

Catches:
1. Exact text duplicates within the training set.
2. Near-duplicates (high character n-gram overlap) within the training set.
3. Leakage: training examples that are too similar to gold set examples.

Uses character-level n-gram Jaccard similarity — simple, fast, no external deps.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.schema import PIIRecord


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclass
class DedupResult:
    kept: list[PIIRecord]
    removed_exact: int
    removed_near: int
    removed_leakage: int

    @property
    def total_removed(self) -> int:
        return self.removed_exact + self.removed_near + self.removed_leakage

    def summary(self) -> dict:
        return {
            "kept": len(self.kept),
            "removed_exact": self.removed_exact,
            "removed_near": self.removed_near,
            "removed_leakage": self.removed_leakage,
            "total_removed": self.total_removed,
        }


def dedup_training_data(
    train: list[PIIRecord],
    gold: list[PIIRecord] | None = None,
    near_threshold: float = 0.85,
    ngram_size: int = 5,
) -> DedupResult:
    """Remove exact duplicates, near-duplicates, and gold-set leakage.

    Args:
        train: candidate training records.
        gold: gold set records to check leakage against.
        near_threshold: Jaccard similarity threshold for near-duplicate detection.
        ngram_size: character n-gram size for similarity computation.
    """
    exact_dups = 0
    near_dups = 0
    leakage = 0

    gold_ngrams = []
    gold_texts = set()
    if gold:
        for g in gold:
            gold_texts.add(g.text)
            gold_ngrams.append(_char_ngrams(g.text, ngram_size))

    seen_texts: set[str] = set()
    kept_ngrams: list[set[str]] = []
    kept: list[PIIRecord] = []

    for rec in train:
        if rec.text in seen_texts:
            exact_dups += 1
            continue

        if rec.text in gold_texts:
            leakage += 1
            continue

        rec_ngrams = _char_ngrams(rec.text, ngram_size)

        leaked = False
        for gng in gold_ngrams:
            if jaccard(rec_ngrams, gng) >= near_threshold:
                leakage += 1
                leaked = True
                break
        if leaked:
            continue

        is_near_dup = False
        for kng in kept_ngrams:
            if jaccard(rec_ngrams, kng) >= near_threshold:
                near_dups += 1
                is_near_dup = True
                break
        if is_near_dup:
            continue

        seen_texts.add(rec.text)
        kept_ngrams.append(rec_ngrams)
        kept.append(rec)

    return DedupResult(
        kept=kept,
        removed_exact=exact_dups,
        removed_near=near_dups,
        removed_leakage=leakage,
    )
