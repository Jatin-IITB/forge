"""GBNF grammar for constrained decoding — the Phase 3 step that was never built.

`docs/DESIGN.md` §"Reliability" has always said this is how the reliability gate
is met:

    For structured outputs (JSON/enum), the student decodes under a
    grammar/schema constraint, so it *cannot* emit invalid output. This is how a
    1-3B model reaches ~100% schema-validity — a place small models otherwise
    lose to large ones for free.

`ACTION_PLAN.md` Phase 3 step 2 lists it as work. It was never implemented, and
G2 has been failing at 0.9974 (384/385) ever since — one malformed response out
of 385, which is exactly the failure a grammar makes unrepresentable. The
threshold has zero margin at this sample size (0.999 x 385 = 384.6), so that one
record is the entire gap.

Three properties the grammar buys, in decreasing obviousness:

1. **Schema validity becomes structural.** The decoder can only emit tokens the
   grammar permits, so there is no parse-failure path left. Not "fewer errors" —
   no representable error.
2. **The label is drawn from the enum, not from the model's memory.** A
   hallucinated type like `"EMAIL_ADDRESS"` cannot be produced, so it cannot
   become a false positive under a label that does not exist.
3. **No preamble, no code fence, no `<think>` block.** `forge/inference.py`
   carries three regex fallbacks for stripping those; under a grammar they are
   unreachable. That also removes tokens from every response, which is where
   G3's remaining gap has to come from.

The grammar deliberately does **not** try to constrain `text` to a substring of
the input. GBNF is context-free and the input is not in the grammar, so that
check stays where it already is — `reconstruct_offsets` in `forge/inference.py`,
which drops any span it cannot locate verbatim.
"""

from __future__ import annotations

from forge.schema import PIIType


def spans_gbnf(exact_spacing: bool = False) -> str:
    """GBNF for ``{"spans": [{"label": <enum>, "text": <string>}, ...]}``.

    Args:
        exact_spacing: pin the punctuation to the exact byte sequence the model
            was fine-tuned to emit, instead of allowing optional whitespace.

    **Why `exact_spacing` exists.** Grammar-constrained sampling filters at the
    *token* level, not the character level. With flexible whitespace the decoder
    will accept ``{`` and ``"spans"`` as separate tokens, so when the model's
    highest-probability continuation is the single merged token ``{"spans":`` —
    which is what it saw in every training target — the grammar can push it onto
    a different tokenization of the same string. The output is still valid; the
    model is just no longer on the path it was trained on.

    Measured on the full 385-record test set: the permissive grammar fixes G2
    (schema validity 384/385 -> 385/385) but costs **-0.0162 micro-F1**, which is
    larger than the -0.0151 that disqualified Q4_K_M under the <=0.01 exit gate.
    `exact_spacing=True` is the attempt to buy the validity without the loss, by
    making the only legal string the one the model already wants to produce.
    """
    labels = " | ".join(f'"\\"{t.value}\\""' for t in PIIType)
    common = f"""\
label  ::= {labels}
string ::= "\\"" char* "\\""
char   ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt/] | "u" hex hex hex hex)
hex    ::= [0-9a-fA-F]
"""
    if exact_spacing:
        # One space after each colon and after each comma — byte-for-byte the
        # format in data/train_v2.jsonl and in the SYSTEM_PROMPT's example.
        return (
            'root   ::= "{\\"spans\\": " spans "}"\n'
            'spans  ::= "[]" | "[" span (", " span)* "]"\n'
            'span   ::= "{\\"label\\": " label ", \\"text\\": " string "}"\n'
        ) + common
    return (
        'root   ::= "{" ws "\\"spans\\"" ws ":" ws spans ws "}"\n'
        'spans  ::= "[" ws "]" | "[" ws span (ws "," ws span)* ws "]"\n'
        'span   ::= "{" ws "\\"label\\"" ws ":" ws label ws "," ws "\\"text\\"" ws ":" ws string ws "}"\n'
    ) + common + 'ws     ::= [ \\t\\n]*\n'


def compact_spans_gbnf() -> str:
    """GBNF for the minified ``{"s":[{"l":<enum>,"t":<string>}...]}`` format.

    This is intentionally a separate grammar rather than an option on
    :func:`spans_gbnf`: the verbose grammar is the trained contract, while this
    compact shape is an explicitly off-distribution serving experiment. Keeping
    the names distinct makes it difficult to enable the experiment by accident.

    The text value remains mandatory. Offsets are still reconstructed by
    locating that exact substring in the source; asking the model to count
    character offsets would trade output tokens for unreliable boundaries.
    """
    labels = " | ".join(f'"\\"{t.value}\\""' for t in PIIType)
    return f"""\
root   ::= "{{\\"s\\":" spans "}}"
spans  ::= "[]" | "[" span ("," span)* "]"
span   ::= "{{\\"l\\":" label ",\\"t\\":" string "}}"
label  ::= {labels}
string ::= "\\"" char* "\\""
char   ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt/] | "u" hex hex hex hex)
hex    ::= [0-9a-fA-F]
"""


def spans_json_schema() -> dict:
    """The same constraint as a JSON Schema, for servers that take one.

    llama.cpp accepts either; vLLM and the OpenAI API take this form. Kept in
    sync with the GBNF above by `tests/test_grammar.py`, because two encodings
    of one contract will drift apart the moment nothing checks them.
    """
    return {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": [t.value for t in PIIType]},
                        "text": {"type": "string"},
                    },
                    "required": ["label", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["spans"],
        "additionalProperties": False,
    }
