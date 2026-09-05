"""The benchmark's summary artifact must survive being written to JSON.

`scripts/bench_serving.py` had no tests at all. That is how a one-line change --
adding the `machine` label to the cost breakdown so a throughput figure could
never be read against the wrong laptop's price -- crashed the scoring step of an
overnight run *after* 78 minutes of training, on

    "cost_breakdown": {k: round(v, 6) for k, v in econ.items()}

because `machine` is a string and `round()` has no opinion about strings. The
model was fine and the fix was one line, but the failure landed at the most
expensive possible moment for a defect that costs milliseconds to catch here.

These tests do not need a GPU, a model, or a served endpoint: `summarize` is a
pure function of (args, Run, gold), so it can be driven with a hand-built Run.
That was always true; nobody had done it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bench_serving import (
    DEFAULT_MACHINE,
    Run,
    Sample,
    _contention,
    build_parser,
    summarize,
    usd_per_1k,
)

from forge.schema import PIIRecord, PIISpan, PIIType


def _args(**overrides) -> argparse.Namespace:
    """Real CLI defaults, so this cannot drift as flags are added.

    Hand-listing the ~30 attributes `summarize` reads would reproduce the bug
    these tests exist to catch: a copy of the interface that silently stops
    matching it. Parsing the actual parser means a new required attribute shows
    up here automatically.
    """
    args = build_parser().parse_args(
        ["--model", "m", "--backend", "token-classifier", "--config-name", "test-cfg"]
    )
    # main() attaches this after parsing, before the run starts, so that the
    # artifact records whether the machine was already busy. Not a CLI flag.
    args.contention_at_start = _contention()
    for key, value in overrides.items():
        assert hasattr(args, key), f"{key} is not a real bench_serving flag"
        setattr(args, key, value)
    return args


def _run(n: int = 4, wall: float = 8.0) -> Run:
    return Run(
        samples=[
            Sample(rec_id=f"r-{i}", latency_s=1.0 + i * 0.1, prompt_tokens=40,
                   completion_tokens=12, schema_valid=True)
            for i in range(n)
        ],
        wall_clock_s=wall,
    )


def _gold(n: int = 4) -> list[PIIRecord]:
    text = "Contact Tristan Batta today"
    return [
        PIIRecord(
            id=f"r-{i}", text=text, split="test",
            spans=[PIISpan(start=8, end=21, label=PIIType.PERSON, text="Tristan Batta")],
        )
        for i in range(n)
    ]


class TestSummaryIsSerializable:
    def test_summary_survives_json_dump(self):
        """The artifact is written to disk; if it cannot serialize, nothing ships."""
        out = summarize(_args(), _run(), _gold())

        text = json.dumps(out)  # would raise on any stray non-JSON value
        assert json.loads(text)["config_name"] == "test-cfg"

    def test_machine_label_is_preserved_not_rounded(self):
        """The regression. A string in cost_breakdown must pass through intact."""
        out = summarize(_args(machine="asus-vivobook-pro-15-rtx3050ti"), _run(), _gold())

        assert out["cost_breakdown"]["machine"] == "asus-vivobook-pro-15-rtx3050ti"

    def test_machine_defaults_when_unset(self):
        out = summarize(_args(machine=None), _run(), _gold())

        assert out["cost_breakdown"]["machine"] == DEFAULT_MACHINE

    def test_every_numeric_in_cost_breakdown_is_rounded(self):
        out = summarize(_args(hardware_usd=1024.0), _run(), _gold())

        for key, value in out["cost_breakdown"].items():
            if isinstance(value, float):
                assert round(value, 6) == value, f"{key} was not rounded"

    @pytest.mark.parametrize("field", ["machine", "hardware_usd", "watts"])
    def test_cost_provenance_fields_are_all_present(self, field):
        """A cost figure without its price/power/machine basis is unfalsifiable."""
        out = summarize(_args(), _run(), _gold())

        assert field in out["cost_breakdown"]


class TestCostModel:
    def test_hardware_usd_override_moves_the_cost(self):
        """The reference machine changed once; the override is what tracks it."""
        cheap = usd_per_1k(0.01, hardware_usd=500.0)
        dear = usd_per_1k(0.01, hardware_usd=2000.0)

        assert dear["hardware_usd_per_1k"] > cheap["hardware_usd_per_1k"]
        assert dear["hardware_usd"] == 2000.0

    def test_cost_is_linear_in_seconds_per_record(self):
        """usd_per_1k consumes machine-seconds; doubling them doubles the cost."""
        one = usd_per_1k(0.01, hardware_usd=1024.0)
        two = usd_per_1k(0.02, hardware_usd=1024.0)

        assert two["usd_per_1k"] == pytest.approx(2 * one["usd_per_1k"], rel=1e-9)

    def test_only_machine_is_non_numeric(self):
        """Pins the invariant the summarize comprehension relies on."""
        econ = usd_per_1k(0.01)

        non_numeric = {k for k, v in econ.items() if not isinstance(v, (int, float))}
        assert non_numeric == {"machine"}

    def test_zero_throughput_does_not_divide_by_zero(self):
        econ = usd_per_1k(0.0)

        assert econ["usd_per_1k"] == 0.0
