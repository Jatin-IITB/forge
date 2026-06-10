"""PII detection/redaction output schema — the data contract for Forge's flagship task.

Defines the entity taxonomy, the labelled-span representation, and the gold-record
model used by the frozen eval set and (later) the training set. Everything here is
*structure*, not modelling: no torch, no transformers. It is the schema the teacher
must emit, the student must learn, and the evaluator scores against.

Design choices:
- Spans are character offsets ``[start, end)`` into the ORIGINAL ``text`` (UTF-8
  codepoints, Python string indices). This is unambiguous and tokenizer-independent.
- Each span carries its surface ``text`` redundantly so records are self-checking
  (``text[start:end] == span.text``) and human-readable in the JSONL.
- Spans must not overlap. Overlapping PII is collapsed to the most specific type at
  labelling time (see ``data/gold/PROTOCOL.md``).
- Redaction is deterministic: each span becomes ``[LABEL]``. The redacted string is
  derived, never hand-written, so it can never silently disagree with the spans.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, model_validator


class PIIType(str, Enum):
    """The labelled PII entity types.

    Chosen to cover globally-common identifiers plus India-specific IDs (the DPDP Act
    is part of this task's reason to exist). Keep this list immutable for a contract
    version — adding a type is a new contract version, not an edit.
    """

    # --- Direct identifiers (global) ---
    PERSON = "PERSON"            # full or partial personal names
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    STREET_ADDRESS = "STREET_ADDRESS"
    USERNAME = "USERNAME"        # handles, logins
    URL = "URL"                  # personal/identifying URLs
    IP_ADDRESS = "IP_ADDRESS"

    # --- Quasi-identifiers ---
    LOCATION = "LOCATION"        # city / region (not a full street address)
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    AGE = "AGE"

    # --- High-severity: a single leak is a reportable breach ---
    CREDIT_CARD = "CREDIT_CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"  # incl. IBAN / account numbers
    SSN = "SSN"                    # US Social Security Number
    AADHAAR = "AADHAAR"            # India national ID (12-digit)
    PAN = "PAN"                    # India tax ID (10-char alphanumeric)
    PASSPORT = "PASSPORT"
    DRIVER_LICENSE = "DRIVER_LICENSE"
    PASSWORD = "PASSWORD"
    API_KEY = "API_KEY"            # secrets/tokens/keys


# A leak of any of these is treated as a compliance breach, so the contract sets a
# hard per-type recall floor on them (see contracts/pii_redaction_v1.yaml).
HIGH_SEVERITY: frozenset[PIIType] = frozenset(
    {
        PIIType.CREDIT_CARD,
        PIIType.BANK_ACCOUNT,
        PIIType.SSN,
        PIIType.AADHAAR,
        PIIType.PAN,
        PIIType.PASSPORT,
        PIIType.DRIVER_LICENSE,
        PIIType.PASSWORD,
        PIIType.API_KEY,
    }
)


class PIISpan(BaseModel):
    """A single labelled PII occurrence: a half-open char range plus its type."""

    model_config = {"frozen": True}

    start: int = Field(ge=0, description="Inclusive start offset into the original text.")
    end: int = Field(gt=0, description="Exclusive end offset into the original text.")
    label: PIIType
    text: str = Field(min_length=1, description="Surface string; must equal text[start:end].")

    @model_validator(mode="after")
    def _check_range(self) -> "PIISpan":
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must be > start ({self.start})")
        return self

    @property
    def is_high_severity(self) -> bool:
        return self.label in HIGH_SEVERITY


class PIIRecord(BaseModel):
    """One gold/eval example: text + its complete set of PII spans + derived redaction."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    spans: List[PIISpan] = Field(default_factory=list)
    lang: str = Field(default="en", description="ISO-639-1 language code.")
    source: str = Field(
        default="synthetic:faker",
        description="Provenance tag, e.g. 'synthetic:faker' or 'public:enron'.",
    )
    split: str = Field(default="test", pattern="^(dev|test|train)$")

    @model_validator(mode="after")
    def _check_spans(self) -> "PIIRecord":
        n = len(self.text)
        ordered = sorted(self.spans, key=lambda s: (s.start, s.end))
        prev_end = -1
        for s in ordered:
            if s.end > n:
                raise ValueError(f"span {s} exceeds text length {n}")
            if self.text[s.start : s.end] != s.text:
                raise ValueError(
                    f"span text {s.text!r} != text[{s.start}:{s.end}] "
                    f"({self.text[s.start : s.end]!r}) in record {self.id}"
                )
            if s.start < prev_end:
                raise ValueError(f"overlapping spans in record {self.id} near offset {s.start}")
            prev_end = s.end
        return self

    @property
    def redacted(self) -> str:
        """Text with every span replaced by ``[LABEL]`` — always derived, never stored."""
        out: list[str] = []
        cursor = 0
        for s in sorted(self.spans, key=lambda s: s.start):
            out.append(self.text[cursor : s.start])
            out.append(f"[{s.label.value}]")
            cursor = s.end
        out.append(self.text[cursor:])
        return "".join(out)
