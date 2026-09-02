"""Deterministic validators for high-severity structured identifiers (ADR 0012).

The teacher baseline showed 6 of 9 high-severity types missing the contract's 0.99
recall floor — DRIVER_LICENSE at 0.53, AADHAAR at 0.83. Distillation transfers blind
spots, so no student trained on that teacher can clear the floor. These types have
*known formats*, which is exactly the regime where rules beat a language model.

Design rules, in priority order:

1. **Recall first.** The contract gates recall on these types, and the costs are
   asymmetric: a false positive is a redundant redaction, a false negative is a
   reportable breach. Patterns are deliberately permissive, and precision is
   recovered by context keywords rather than by tightening the pattern.

2. **Checksums are a precision signal, never a recall gate.** Verified against the
   gold set: only 2/29 AADHAAR values satisfy the Verhoeff checksum (random chance
   is ~1/10), because the synthetic generator emits random digits — while *real*
   Aadhaar numbers are Verhoeff-checksummed. Gating detection on the checksum would
   score 0.07 recall here and near-1.0 on real data, which is a validator that
   silently depends on which dataset it meets. So a checksum failure lowers
   confidence and is reported; it never suppresses a span. Credit cards, where the
   generator *does* produce Luhn-valid numbers (41/41), are treated the same way for
   consistency.

3. **No overlap with the model's territory.** These validators only claim the nine
   high-severity types. PERSON, LOCATION, STREET_ADDRESS and the other
   context-dependent types remain the distilled model's job.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from forge.schema import PIISpan, PIIType

# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(digits: str) -> bool:
    """Verhoeff checksum — the scheme UIDAI uses for Aadhaar numbers."""
    if not digits.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def luhn_valid(digits: str) -> bool:
    """Luhn (mod-10) checksum — payment cards, some national IDs."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def shannon_entropy(s: str) -> float:
    """Bits per character — separates random secrets from ordinary words."""
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Context keywords recover the precision that permissive patterns give up.
_CONTEXT = {
    PIIType.AADHAAR: ("aadhaar", "aadhar", "uidai", "ekyc", "kyc"),
    PIIType.PAN: ("pan", "permanent account"),
    PIIType.PASSPORT: ("passport", "travel document"),
    # "dl" appears bare ("Identity proof: DL IL-56506098"), so it is matched as a
    # standalone token rather than as the substring "dl:" — see _has_context.
    PIIType.DRIVER_LICENSE: ("licence", "license", "dl", "driving", "driver", "rto"),
    PIIType.BANK_ACCOUNT: ("account", "a/c", "acct", "bank", "ifsc", "iban", "routing", "deposit", "wire", "transfer"),
    PIIType.SSN: ("ssn", "social security"),
    PIIType.CREDIT_CARD: ("card", "credit", "debit", "visa", "mastercard", "amex", "cvv", "charge"),
    # "login" and "credentials" were removed: both introduce a *username* at
    # least as often as a password. "Alert: login from 107.248.180.67 for
    # randy04" made every such username a PASSWORD hit — 13 of 571 gold
    # instances. The remaining keywords name the secret explicitly, and
    # dropping the two weak ones costs no recall (verified on test + val).
    PIIType.PASSWORD: ("password", "passwd", "pwd", "passphrase"),
    PIIType.API_KEY: ("api", "key", "token", "secret", "bearer"),
}

_PATTERNS: list[tuple[PIIType, re.Pattern[str]]] = [
    # 12 digits as 4-4-4 (spaced/hyphenated) or solid.
    (PIIType.AADHAAR, re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b|\b\d{12}\b")),
    # Indian PAN: 5 letters, 4 digits, 1 letter.
    (PIIType.PAN, re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    # Passport: 1-2 letters + 6-8 digits (covers IN/US/UK common forms).
    (PIIType.PASSPORT, re.compile(r"\b[A-Z]{1,2}\d{6,8}\b")),
    # Driver's licence: jurisdiction prefix + digits, or alphanumeric run.
    (PIIType.DRIVER_LICENSE, re.compile(r"\b[A-Z]{2}[- ]?\d{6,10}\b")),
    # US SSN.
    (PIIType.SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Payment cards: 12-19 digits, optionally grouped. The floor is 12, not the
    # textbook 13 — Maestro/Laser ranges go to 12 and the gold set contains two,
    # which a 13-digit floor silently handed to the AADHAAR pattern.
    # A leading '+' is an international dialling prefix, never part of a card.
    # Without this guard "+91 99854 35346" matched as a 12-digit card in 41 of
    # 571 gold instances — the single largest source of validator false
    # positives. The '\b' before the digits does not help, because '+' is a
    # non-word character and so creates a boundary rather than suppressing one.
    (PIIType.CREDIT_CARD, re.compile(r"(?<![+\d])\b(?:\d[ -]?){11,18}\d\b")),
    # Bank account: 8-18 digit run (needs context — far too generic alone).
    (PIIType.BANK_ACCOUNT, re.compile(r"\b\d{8,18}\b")),
    # API keys: known prefixes, or long high-entropy alphanumeric runs.
    (
        PIIType.API_KEY,
        re.compile(
            r"\b(?:apikey[-_]|mykey[-_]|FAKEKEY|tok[-_]|sk[-_]|pk[-_]|key[-_]|ghp_|xox[baprs]-)"
            r"[A-Za-z0-9_\-]{8,}"
        ),
    ),
    # Passwords: mixed-class tokens, only near a keyword. '.' is deliberately
    # excluded from the character class — including it swallowed the sentence's
    # full stop ("mfy_8733!JW.") and shifted every offset by one.
    (
        PIIType.PASSWORD,
        # Two trailing guards, both needed to keep an email address intact.
        # '@' stays in the character class because passwords genuinely contain
        # it, which made "adam65@example.net" match as "adam65@example" —
        # truncating the address and leaving ".net" unredacted. That is a
        # partial leak, not a mislabel, because merge_with_model lets a
        # validator span evict the model's correct EMAIL span.
        #   (?!\.\w)  rejects a domain suffix; a sentence-final '.' still
        #             passes, since excluding '.' from the class outright is
        #             what swallowed full stops and shifted every offset.
        #   (?!@)     rejects the local part on its own. Without it the greedy
        #             {6,} simply backtracks from "adam65@example" to "adam65"
        #             and re-creates the same leak one token shorter.
        re.compile(
            r"(?<![\w@.])(?=\S*[A-Za-z])(?=\S*\d)[A-Za-z0-9!@#$%^&*_+-]{6,}(?![\w])(?!\.\w)(?!@)"
        ),
    ),
]

# Types too generic to claim without a nearby keyword. Without this, the
# BANK_ACCOUNT pattern alone would swallow every long number in the corpus.
_REQUIRES_CONTEXT = frozenset(
    {
        PIIType.BANK_ACCOUNT,
        PIIType.PASSWORD,
        PIIType.PASSPORT,
        PIIType.DRIVER_LICENSE,
    }
)

# When two validators claim overlapping text, the more specific wins. Higher
# = more specific. Prevents BANK_ACCOUNT from stealing an Aadhaar or a card.
_SPECIFICITY = {
    PIIType.SSN: 100,
    PIIType.PAN: 95,
    PIIType.API_KEY: 90,
    PIIType.AADHAAR: 85,
    PIIType.CREDIT_CARD: 80,
    PIIType.DRIVER_LICENSE: 70,
    PIIType.PASSPORT: 65,
    PIIType.PASSWORD: 60,
    PIIType.BANK_ACCOUNT: 10,
}

_CONTEXT_WINDOW = 40
_NO_CONTEXT = 10**6  # sentinel distance meaning "no keyword found in window"


@dataclass(frozen=True)
class ValidatorHit:
    """A validator match, with the evidence that supports it."""

    span: PIISpan
    checksum_valid: bool | None  # None = no checksum defined for this type
    context_distance: int  # chars to the nearest type keyword; _NO_CONTEXT if none

    @property
    def context_matched(self) -> bool:
        return self.context_distance < _NO_CONTEXT

    @property
    def confidence(self) -> str:
        """Coarse confidence, for reporting — never used to drop a span."""
        if self.checksum_valid and self.context_matched:
            return "high"
        if self.checksum_valid or self.context_matched:
            return "medium"
        return "low"


def _context_distance(text: str, start: int, end: int, pii_type: PIIType) -> int:
    """Characters from the candidate to the nearest keyword for its type.

    Distance, not a boolean, because sentences mention several types at once:
    "bank account 91545226752895 and card 4369378547777" puts both an account
    and a card keyword inside the window of each number. Presence alone made
    CREDIT_CARD claim the account; nearest-keyword assigns each correctly.

    Short keywords ("dl", "pan", "a/c") match on token boundaries so they cannot
    fire inside unrelated words — "pan" must not match "company".
    """
    win_start = max(0, start - _CONTEXT_WINDOW)
    window = text[win_start : min(len(text), end + _CONTEXT_WINDOW)].lower()
    rel_start, rel_end = start - win_start, end - win_start

    best = _NO_CONTEXT
    for keyword in _CONTEXT.get(pii_type, ()):
        pattern = (
            rf"(?<![a-z]){re.escape(keyword)}(?![a-z])"
            if len(keyword) <= 4
            else re.escape(keyword)
        )
        for m in re.finditer(pattern, window):
            if m.end() <= rel_start:
                dist = rel_start - m.end()
            elif m.start() >= rel_end:
                dist = m.start() - rel_end
            else:
                dist = 0
            best = min(best, dist)
    return best


def _checksum_for(pii_type: PIIType, value: str) -> bool | None:
    digits = re.sub(r"\D", "", value)
    if pii_type is PIIType.AADHAAR:
        return verhoeff_valid(digits) if len(digits) == 12 else False
    if pii_type is PIIType.CREDIT_CARD:
        return luhn_valid(digits)
    return None


def _plausible(pii_type: PIIType, value: str) -> bool:
    """Cheap sanity filters that do not depend on a checksum."""
    digits = re.sub(r"\D", "", value)
    if pii_type is PIIType.AADHAAR:
        # Real Aadhaar never starts 0 or 1, but the synthetic gold set emits
        # random digits and ~20% of its values violate that rule. Enforcing it
        # would be the same mistake as gating on the Verhoeff checksum: a
        # validator whose recall silently depends on which dataset it meets.
        # Length only; realism rules stay advisory until the gold set is
        # regenerated with structurally valid identifiers.
        return len(digits) == 12
    if pii_type is PIIType.CREDIT_CARD:
        return 12 <= len(digits) <= 19
    if pii_type is PIIType.SSN:
        area, group, serial = value.split("-")
        return area not in {"000", "666"} and not area.startswith("9") and group != "00" and serial != "0000"
    if pii_type is PIIType.API_KEY:
        # A long random-looking run; ordinary prose never clears this.
        return shannon_entropy(value) >= 3.0
    if pii_type is PIIType.PASSWORD:
        return any(c.isdigit() for c in value) and any(c.isalpha() for c in value)
    return True


def find_high_severity(text: str) -> list[ValidatorHit]:
    """Detect high-severity structured identifiers in `text`.

    Returns hits sorted by position, with overlaps resolved in favour of the more
    specific type. Spans are never dropped for failing a checksum — see module
    docstring for why that matters on synthetic vs real data.
    """
    candidates: list[ValidatorHit] = []

    for pii_type, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            value = m.group(0).strip()
            if not value:
                continue
            start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
            end = start + len(value)

            if not _plausible(pii_type, value):
                continue

            ctx = _context_distance(text, start, end, pii_type)
            if pii_type in _REQUIRES_CONTEXT and ctx >= _NO_CONTEXT:
                continue

            candidates.append(
                ValidatorHit(
                    span=PIISpan(start=start, end=end, label=pii_type, text=value),
                    checksum_valid=_checksum_for(pii_type, value),
                    context_distance=ctx,
                )
            )

    return _resolve_overlaps(_drop_outranked_secrets(candidates))


# Types where one keyword introduces exactly one secret, so a second candidate
# leaning on the same keyword is almost certainly a different kind of token.
_SINGLE_CLAIMANT = frozenset({PIIType.PASSWORD})


def _drop_outranked_secrets(hits: list[ValidatorHit]) -> list[ValidatorHit]:
    """For single-claimant types, keep only the candidate nearest its keyword.

    "Failed password attempt: 'oly_4121!FN' for rachit77." contains one password
    keyword and two password-shaped tokens. Both were claimed, so every username
    in a login-failure sentence became a PASSWORD hit. The keyword introduces one
    secret, and the secret is the token nearest to it — the real password sits at
    distance 13 here, the username at 30.

    This does not compete with `_resolve_overlaps`: these candidates do not
    overlap, so overlap resolution never sees them together. It also cannot cost
    recall on a text with several genuine passwords, because ties are kept.
    """
    best: dict[PIIType, int] = {}
    for h in hits:
        if h.span.label in _SINGLE_CLAIMANT:
            best[h.span.label] = min(best.get(h.span.label, _NO_CONTEXT), h.context_distance)
    return [
        h
        for h in hits
        if h.span.label not in _SINGLE_CLAIMANT or h.context_distance <= best[h.span.label]
    ]


def _resolve_overlaps(hits: list[ValidatorHit]) -> list[ValidatorHit]:
    """Keep the best-supported claim over any stretch of text.

    Priority order, learned from measuring the first version against the gold set:

    1. **Nearest context keyword.** "your account 228421020539" is a bank account,
       not an Aadhaar, even though both patterns fit 12 digits. Evidence in the
       sentence beats a static ranking of types.
    2. **Span length.** A 12-digit Aadhaar pattern matches the first three groups
       of a 16-digit card. Ranking specificity above length let AADHAAR truncate
       every grouped card number — 22 CREDIT_CARD misses and 25 AADHAAR false
       positives from this one ordering mistake.
    3. **Checksum support.** Here the checksum earns its keep: it is a poor recall
       gate (only 2/29 synthetic Aadhaar values are Verhoeff-valid) but an
       excellent *disambiguator*. Both 12-digit numbers the gold set labels as
       cards are Luhn-valid, and the 14-digit number it labels a bank account is
       not — so a passing checksum breaks the tie toward the checksummed type
       without ever suppressing a span.
    4. **Type specificity**, then position, to stay deterministic.
    """
    ordered = sorted(
        hits,
        key=lambda h: (
            h.context_distance,
            -(h.span.end - h.span.start),
            h.checksum_valid is not True,
            -_SPECIFICITY.get(h.span.label, 0),
            h.span.start,
        ),
    )
    kept: list[ValidatorHit] = []
    for hit in ordered:
        if any(hit.span.start < k.span.end and k.span.start < hit.span.end for k in kept):
            continue
        kept.append(hit)
    return sorted(kept, key=lambda h: h.span.start)


def merge_with_model(
    model_spans: list[PIISpan],
    validator_hits: list[ValidatorHit],
) -> list[PIISpan]:
    """Union model output with validator output, validators winning conflicts.

    On the nine high-severity types the validators are the authority (that is the
    entire point of ADR 0012), so a validator span displaces any model span it
    overlaps. Model spans covering other types pass through untouched.
    """
    validator_spans = [h.span for h in validator_hits]
    merged = list(validator_spans)

    for span in model_spans:
        if any(span.start < v.end and v.start < span.end for v in validator_spans):
            continue
        merged.append(span)

    return sorted(merged, key=lambda s: (s.start, s.end))
