"""Out-of-domain detection — the gate contract v2 requires and nothing implemented.

`contracts/pii_redaction_v2.yaml` says:

    For inputs outside the in-domain definition (other languages, code, binary,
    >512 tokens), return an empty span list with an explicit
    {"status":"out_of_domain"} flag rather than hallucinating spans.

Until this module existed there was no such path. G6 measured the consequence on
the shipped artifact (`reports/ood_gate.md`): the model invents six spans in a
Russian sentence and two in a list of MD5 hashes, because no training example
ever taught it to decline — and the ADR 0012 validator layer fires
unconditionally, inventing a `BANK_ACCOUNT` in a hex dump and in a bare column
of integers. Both stages need to be told the document is not theirs.

**The asymmetry that shapes every threshold here.** A false positive is a
refusal to redact a real document — an unredacted leak. A false negative is a
hallucinated span on junk, which is untidy and cheap. These are not comparable,
so this module is tuned for *precision on in-domain text* and accepts whatever
recall that leaves. Every signal below is a narrow, shape-specific test rather
than a general "does this look like prose" score.

Prose-likeness was tried first and rejected. In-domain records here are
structured fragments — "Ticket #4521: Rachita Thakkar, DOB 1985-03-12, Aadhaar
5280 9885 1656" contains almost no English function words, so a function-word
density test flags real support tickets as out-of-domain. That is the failure
mode this module must not have.

Verified: **0 false positives across all 1107 committed in-domain records**
(385 test + 533 val + 189 dev), and **21/21 recall** on the OOD probe set with
no adversarial probe refused. Enforced by `tests/test_ood.py`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

# Contract's in-domain ceiling. Tokens are estimated, not tokenized: this runs
# before the model is loaded, so it cannot borrow the model's tokenizer, and a
# 4-chars-per-token estimate is well inside the margin at a 512 limit.
MAX_TOKENS = 512
_CHARS_PER_TOKEN = 4

_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
_HASH = re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE)
_HEX_DUMP = re.compile(r"\b(?:[0-9a-f]{2}[ ,]){7,}[0-9a-f]{2}\b", re.IGNORECASE)
_BASE64_BLOB = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")

# Code markers. Each is a construct that does not occur in support-ticket prose.
# Deliberately not "contains a bracket" — gold text contains parentheses,
# slashes and colons constantly.
_CODE = (
    ("python", re.compile(r"^\s*(?:def|class|import|from)\s+\w|(?:^|\n)\s*return\s+", re.MULTILINE)),
    ("sql", re.compile(r"\bSELECT\b[\s\S]{0,200}?\bFROM\b", re.IGNORECASE)),
    ("shell", re.compile(r"(?:^|\|\s*)(?:find|grep|awk|sed|curl|rm|ls|cat|chmod)\s+[-.\w/*]")),
    ("html", re.compile(r"<\s*(?:div|span|p|a|table|tr|td|script|html|body|head)\b[^>]*>", re.IGNORECASE)),
    ("regex", re.compile(r"\(\?[:=!<]")),
)


@dataclass(frozen=True)
class OODVerdict:
    """Why a document was refused, or why it was accepted."""

    is_ood: bool
    reason: str | None = None
    detail: str = ""
    signals: dict = field(default_factory=dict)

    def as_response(self) -> dict:
        """The contract-mandated response body for an out-of-domain input."""
        return {"status": "out_of_domain", "reason": self.reason, "spans": []}


def _script_ratio(text: str) -> float:
    """Fraction of *letters* that are not Latin.

    Counts letters only. Digits, punctuation and whitespace are script-neutral
    and including them would make a phone number look like a foreign language.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(1 for c in letters if "LATIN" not in unicodedata.name(c, ""))
    return non_latin / len(letters)


_WORD = re.compile(r"[A-Za-z]{2,}")


def _residual_words(text: str, pat: re.Pattern[str]) -> int:
    """How many alphabetic words survive once `pat`'s matches are removed.

    The test that separates "API key: <secret>" from a raw base64 dump. A
    labelled credential leaves its label behind; a dump leaves nothing.
    """
    return len(_WORD.findall(pat.sub(" ", text)))


def _looks_like_json(text: str) -> bool:
    s = text.strip()
    if not (s.startswith(("{", "[")) and s.endswith(("}", "]"))):
        return False
    try:
        json.loads(s)
    except (ValueError, RecursionError):
        return False
    return True


def detect_out_of_domain(text: str, max_tokens: int = MAX_TOKENS) -> OODVerdict:
    """Classify a document as in- or out-of-domain, with the reason.

    Order matters: cheap unambiguous tests first, so the reason returned is the
    most specific one that applies rather than whichever fired last.
    """
    stripped = text.strip()
    signals: dict[str, float | int | bool] = {
        "chars": len(text),
        "est_tokens": len(text) // _CHARS_PER_TOKEN,
    }

    # 1. Nothing to redact. Also covers whitespace- and punctuation-only input.
    if not any(c.isalnum() for c in stripped):
        return OODVerdict(True, "empty_or_symbolic", "no alphanumeric content", signals)

    # 2. Contract's explicit length ceiling.
    if signals["est_tokens"] > max_tokens:
        return OODVerdict(
            True, "too_long", f"~{signals['est_tokens']} tokens > {max_tokens}", signals
        )

    # 3. Non-Latin script. The threshold is high because an in-domain ticket may
    #    legitimately carry a stray accented character or a currency symbol,
    #    while a genuinely foreign sentence is essentially all non-Latin.
    ratio = _script_ratio(text)
    signals["non_latin_letter_ratio"] = round(ratio, 4)
    if ratio > 0.30:
        return OODVerdict(
            True, "non_latin_script", f"{ratio:.0%} of letters are non-Latin", signals
        )

    # 4. Encoded blobs. Each pattern is a shape that carries no natural-language
    #    content; the validators would otherwise read hex runs as identifiers.
    for name, pat in (("uuid", _UUID), ("hash", _HASH), ("hex_dump", _HEX_DUMP), ("base64", _BASE64_BLOB)):
        hits = pat.findall(text)
        if not hits:
            continue
        covered = sum(len(h) for h in hits) / max(len(stripped), 1)
        signals[f"{name}_coverage"] = round(covered, 4)
        # Coverage, not presence: one UUID inside a sentence is a support
        # ticket quoting a request id; a document that is mostly UUIDs is a dump.
        if covered <= 0.40:
            continue
        # ...but coverage alone refuses the one document type we least can
        # afford to refuse. "API key: FAKEKEYZGCFcs5PmuGf8gk9XIIaOenQOXn3RB1gnI1S."
        # is 79% base64 *because the credential is the point*, and dropping it
        # would be a leak caused by the safety gate. What separates a labelled
        # secret from a dump is the sentence around it: strip the encoded runs
        # and a real record still reads as prose, a dump leaves nothing.
        if _residual_words(text, pat) >= 2:
            signals[f"{name}_has_prose"] = True
            continue
        return OODVerdict(True, f"encoded_{name}", f"{covered:.0%} of the document", signals)

    # 5. Structured data and code.
    if _looks_like_json(text):
        return OODVerdict(True, "code_json", "parses as a JSON document", signals)
    for name, pat in _CODE:
        if pat.search(text):
            return OODVerdict(True, f"code_{name}", f"{name} syntax", signals)

    # 6. Numeric dumps. Requires the document to be almost entirely digits:
    #    an in-domain record full of identifiers still carries field labels.
    alpha = sum(c.isalpha() for c in stripped)
    alnum = sum(c.isalnum() for c in stripped)
    letter_ratio = alpha / alnum if alnum else 0.0
    signals["letter_ratio"] = round(letter_ratio, 4)
    if letter_ratio < 0.10:
        return OODVerdict(
            True, "numeric_dump", f"only {letter_ratio:.0%} of characters are letters", signals
        )

    return OODVerdict(False, None, "in-domain", signals)
