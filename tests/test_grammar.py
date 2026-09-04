"""Constrained-decoding grammar (forge/grammar.py).

The grammar is the mechanism `docs/DESIGN.md` has always named for the
reliability gate — "the student decodes under a grammar/schema constraint, so it
*cannot* emit invalid output". These tests hold it to that: every label the
schema allows must be reachable, and the two encodings of the same contract must
not drift apart.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from forge.grammar import (
    compact_spans_gbnf,
    compact_spans_json_schema,
    line_spans_gbnf,
    spans_gbnf,
    spans_json_schema,
)
from forge.schema import PIIType

LLAMA_GRAMMAR_BIN = Path.home() / "llama.cpp" / "build" / "bin" / "test-grammar-parser"


class TestCoverage:
    def test_every_pii_type_is_reachable(self):
        """A type missing from the grammar can never be predicted.

        That would silently cap recall on it at zero while every test that
        stubs the model still passed.
        """
        g = spans_gbnf()
        for t in PIIType:
            assert f'"\\"{t.value}\\""' in g, f"{t.value} not in grammar"

    def test_json_schema_lists_the_same_types(self):
        schema_labels = set(spans_json_schema()["properties"]["spans"]["items"]["properties"]["label"]["enum"])
        assert schema_labels == {t.value for t in PIIType}

    def test_the_two_encodings_do_not_drift(self):
        """GBNF and JSON Schema express one contract; nothing else checks them.

        llama.cpp takes the grammar, vLLM and the OpenAI API take the schema. If
        a type were added to one and not the other, the same model would be
        constrained differently depending on the server it ran on.
        """
        g = spans_gbnf()
        in_gbnf = set(re.findall(r'"\\"([A-Z_]+)\\""', g))
        in_schema = set(spans_json_schema()["properties"]["spans"]["items"]["properties"]["label"]["enum"])
        assert in_gbnf == in_schema


class TestShape:
    def test_root_produces_the_trained_output_format(self):
        """The grammar must not force a shape the model was not trained on.

        Training targets are `{"spans": [{"label": ..., "text": ...}]}`. A
        grammar demanding different key names or ordering would trade accuracy
        for validity, which is not the deal.
        """
        g = spans_gbnf()
        assert '"spans"' in g.replace("\\", "")
        assert '"label"' in g.replace("\\", "")
        assert '"text"' in g.replace("\\", "")

    def test_empty_span_list_is_representable(self):
        """30 of 385 gold records have no PII. If `{"spans": []}` were
        unreachable the model would be forced to invent one."""
        assert '"[" ws "]"' in spans_gbnf()

    def test_control_characters_are_excluded_from_strings(self):
        """A raw control byte inside a JSON string is invalid JSON — the exact
        class of failure the grammar exists to make unrepresentable."""
        assert "\\x00-\\x1F" in spans_gbnf()

    def test_schema_forbids_extra_keys(self):
        s = spans_json_schema()
        assert s["additionalProperties"] is False
        assert s["properties"]["spans"]["items"]["additionalProperties"] is False


class TestExactSpacingVariant:
    """`exact_spacing=True` — the attempt to buy G2 without paying F1.

    The permissive grammar fixes schema validity but costs -0.0162 micro-F1,
    larger than the -0.0151 that disqualified Q4_K_M. The hypothesis is that
    optional whitespace lets the decoder split `{"spans":` into tokens the model
    never saw, pushing it off its training distribution.
    """

    def test_literals_match_the_prompt_example_byte_for_byte(self):
        """If the grammar and the trained format disagree, the fix cannot work.

        The system prompt shows the model exactly one output shape. The strict
        grammar must admit precisely that string — any difference in spacing
        reintroduces the tokenization mismatch it exists to remove.
        """
        from forge.inference import SYSTEM_PROMPT

        assert '{"spans": [{"label": "PERSON", "text": "Jane Doe"}, ...]}' in SYSTEM_PROMPT
        g = spans_gbnf(exact_spacing=True)
        assert 'root   ::= "{\\"spans\\": " spans "}"' in g
        assert '"{\\"label\\": " label ", \\"text\\": " string "}"' in g
        assert '(", " span)*' in g

    def test_no_optional_whitespace_rule(self):
        assert "ws" not in spans_gbnf(exact_spacing=True).split("\n")[0]
        assert "ws     ::=" not in spans_gbnf(exact_spacing=True)

    def test_permissive_variant_still_has_one(self):
        assert "ws     ::=" in spans_gbnf(exact_spacing=False)

    def test_empty_list_representable_in_both(self):
        assert '"[]"' in spans_gbnf(exact_spacing=True)
        assert '"[" ws "]"' in spans_gbnf(exact_spacing=False)

    @pytest.mark.parametrize("exact", [True, False])
    def test_both_variants_cover_every_type(self, exact):
        g = spans_gbnf(exact_spacing=exact)
        for t in PIIType:
            assert f'"\\"{t.value}\\""' in g


class TestCompactVariant:
    def test_minified_shape_and_every_type_are_reachable(self):
        g = compact_spans_gbnf()
        assert 'root   ::= "{\\"s\\":" spans "}"' in g
        assert 'span   ::= "{\\"l\\":" label ",\\"t\\":" string "}"' in g
        assert "ws" not in g
        for t in PIIType:
            assert f'"\\"{t.value}\\""' in g

    def test_empty_list_is_representable(self):
        assert 'spans  ::= "[]"' in compact_spans_gbnf()

    def test_json_schema_matches_compact_grammar_labels(self):
        schema = compact_spans_json_schema()
        labels = schema["properties"]["s"]["items"]["properties"]["l"]["enum"]
        assert set(labels) == {t.value for t in PIIType}
        assert schema["additionalProperties"] is False
        assert schema["properties"]["s"]["items"]["additionalProperties"] is False

    @pytest.mark.skipif(not LLAMA_GRAMMAR_BIN.exists(), reason="llama.cpp grammar parser not built")
    def test_llama_cpp_parses_compact_grammar(self):
        proc = subprocess.run(
            [str(LLAMA_GRAMMAR_BIN)],
            input=compact_spans_gbnf(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


class TestLineVariant:
    def test_line_shape_and_every_type_are_reachable(self):
        grammar = line_spans_gbnf()
        assert 'root   ::= "-" | row ("\\n" row)*' in grammar
        assert 'row    ::= label "\\t" string' in grammar
        for pii_type in PIIType:
            assert f'"\\"{pii_type.value}\\""' in grammar

    @pytest.mark.skipif(not LLAMA_GRAMMAR_BIN.exists(), reason="llama.cpp grammar parser not built")
    def test_llama_cpp_parses_line_grammar(self):
        proc = subprocess.run(
            [str(LLAMA_GRAMMAR_BIN)],
            input=line_spans_gbnf(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


class TestAcceptedByLlamaCpp:
    @pytest.mark.skipif(not LLAMA_GRAMMAR_BIN.exists(), reason="llama.cpp grammar parser not built")
    def test_llama_cpp_parses_the_grammar(self):
        """Catches escaping errors that only appear at the server.

        A grammar that fails to parse makes llama-server reject every request,
        so this is worth the dependency when the binary happens to be present.
        """
        proc = subprocess.run(
            [str(LLAMA_GRAMMAR_BIN)], input=spans_gbnf(), capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr


class TestOutputStillParses:
    def test_a_grammar_shaped_response_round_trips(self):
        """What the grammar permits must be what the parser accepts.

        Validity at the decoder is worthless if `parse_response` then rejects
        it, so this asserts the two ends agree.
        """
        from forge.inference import parse_response

        text = "Contact Jane Doe at jane@example.com"
        raw = json.dumps({"spans": [{"label": "PERSON", "text": "Jane Doe"},
                                    {"label": "EMAIL", "text": "jane@example.com"}]})
        record, valid = parse_response("t", text, raw, split="test")
        assert valid
        assert {s.label.value for s in record.spans} == {"PERSON", "EMAIL"}

    def test_empty_response_round_trips(self):
        from forge.inference import parse_response

        record, valid = parse_response("t", "no pii here", json.dumps({"spans": []}), split="test")
        assert valid
        assert record.spans == []
