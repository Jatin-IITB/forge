#!/usr/bin/env python3
"""Gates G3 (cost) and G4 (latency) — measured, with the cost model published.

The project's entire premise is that the specialist is cheaper and faster than
the teacher. That claim is worthless unless the arithmetic behind it is visible,
so this script prints every input to every number it reports.

Two honesty rules are enforced here rather than left to the writeup:

1. **The teacher is priced at PAID rates even though we ran it on a free tier.**
   A free tier is a bootstrap subsidy for development, not an economics claim.
   Pricing the teacher at $0 would make the cost ratio infinite and meaningless.
2. **The student's cost is amortized hardware + energy, not $0.** "It runs on my
   laptop" is not free; the laptop cost money and the electricity is metered.

Both models are declared in DEFAULT_PRICING below and can be overridden on the
command line, so a reader who disputes an assumption can re-run with their own.

Usage:
    python scripts/run_economics.py \
        --teacher-meta data/predictions_teacher_120b_test.meta.json \
        --student-meta data/predictions_student_run002.meta.json \
        --contract contracts/pii_redaction_v2.yaml \
        --output reports/economics.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from forge.contracts import load_contract

# ---------------------------------------------------------------------------
# Cost model inputs. Every one of these is an assumption a reader may challenge;
# all are overridable via CLI flags and all are printed in the report.
# ---------------------------------------------------------------------------

DEFAULT_PRICING = {
    # Teacher: public list price for gpt-oss-120b on Cerebras paid tier
    # (USD per 1M tokens). We measured on the free tier; we PRICE at paid.
    "teacher_input_per_1m": 0.25,
    "teacher_output_per_1m": 0.69,
    # Student: Apple M-series laptop, on-device.
    #   hardware: purchase price amortized over a 4-year useful life, assuming
    #   the machine is busy 8h/day (a generous-to-the-teacher assumption: idle
    #   time makes the student look *worse* here, not better).
    "student_hardware_usd": 1599.0,
    "student_life_years": 4.0,
    "student_busy_hours_per_day": 8.0,
    # energy: sustained package power under inference load x local tariff.
    "student_watts": 22.0,
    "student_usd_per_kwh": 0.12,
}


@dataclass
class Side:
    """One side of the comparison, with its cost derivation kept intact."""

    name: str
    records: int
    p50_s: float
    p95_s: float
    avg_s: float
    usd_per_1k: float
    basis: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    notes: list[str] = field(default_factory=list)


def teacher_cost(meta: dict, pricing: dict) -> tuple[float, str, list[str]]:
    """USD per 1k records for a hosted API teacher, priced at paid-tier rates."""
    tin = meta.get("total_tokens_in")
    tout = meta.get("total_tokens_out")
    n = meta.get("total", 0)
    notes = []

    if not tin or not tout or not n:
        notes.append(
            "token counts missing from meta — rerun inference with a build that "
            "records usage; cost reported as UNAVAILABLE rather than guessed"
        )
        return float("nan"), "token usage unavailable", notes

    in_per_rec = tin / n
    out_per_rec = tout / n
    usd_per_1k = 1000 * (
        in_per_rec * pricing["teacher_input_per_1m"] / 1e6
        + out_per_rec * pricing["teacher_output_per_1m"] / 1e6
    )
    basis = (
        f"{in_per_rec:.0f} in + {out_per_rec:.0f} out tokens/record @ "
        f"${pricing['teacher_input_per_1m']}/1M in, "
        f"${pricing['teacher_output_per_1m']}/1M out"
    )
    notes.append(
        "priced at PAID list rates although measured on the free tier — a free "
        "tier is a development subsidy, not an economics claim"
    )
    return usd_per_1k, basis, notes


def student_cost(meta: dict, pricing: dict) -> tuple[float, str, list[str]]:
    """USD per 1k records for on-device inference: amortized hardware + energy."""
    avg_s = meta.get("avg_latency_s", 0.0)
    if not avg_s:
        return float("nan"), "latency unavailable", ["no latency recorded"]

    busy_hours = pricing["student_life_years"] * 365 * pricing["student_busy_hours_per_day"]
    hw_per_hour = pricing["student_hardware_usd"] / busy_hours
    hw_per_1k = hw_per_hour * (avg_s * 1000 / 3600)

    energy_per_1k = (
        pricing["student_watts"] / 1000 * (avg_s * 1000 / 3600) * pricing["student_usd_per_kwh"]
    )

    basis = (
        f"{avg_s:.2f}s/record x 1000 = {avg_s * 1000 / 3600:.2f} machine-hours; "
        f"hardware ${hw_per_hour:.4f}/h "
        f"(${pricing['student_hardware_usd']:.0f} / {busy_hours:.0f} busy hours) "
        f"+ energy {pricing['student_watts']:.0f}W @ ${pricing['student_usd_per_kwh']}/kWh"
    )
    notes = [
        (
            "on-device is NOT free: hardware amortized over "
            f"{pricing['student_life_years']:.0f}y at "
            f"{pricing['student_busy_hours_per_day']:.0f}h/day busy + metered energy"
        ),
        "latency measured on Apple Silicon MPS, batch size 1, unquantized",
    ]
    return hw_per_1k + energy_per_1k, basis, notes


def build_side(name: str, meta: dict, pricing: dict, is_teacher: bool) -> Side:
    cost_fn = teacher_cost if is_teacher else student_cost
    usd, basis, notes = cost_fn(meta, pricing)
    return Side(
        name=name,
        records=meta.get("total", 0),
        p50_s=meta.get("p50_latency_s", 0.0),
        p95_s=meta.get("p95_latency_s", 0.0),
        avg_s=meta.get("avg_latency_s", 0.0),
        usd_per_1k=usd,
        basis=basis,
        tokens_in=meta.get("total_tokens_in"),
        tokens_out=meta.get("total_tokens_out"),
        notes=notes,
    )


def render(teacher: Side, student: Side, gates: dict, pricing: dict) -> str:
    def ratio(s: float, t: float) -> float:
        return s / t if t else float("nan")

    cost_ratio = ratio(student.usd_per_1k, teacher.usd_per_1k)
    p95_ratio = ratio(student.p95_s, teacher.p95_s)
    cost_max = gates.get("cost_ratio_max", 0.10)
    p95_max = gates.get("p95_ratio_max", 0.20)

    def verdict(r: float, limit: float) -> str:
        if math.isnan(r):
            return "UNMEASURED"
        return "PASS" if r <= limit else "FAIL"

    cost_v = verdict(cost_ratio, cost_max)
    p95_v = verdict(p95_ratio, p95_max)

    def money(x: float) -> str:
        return "unavailable" if math.isnan(x) else f"${x:.4f}"

    def mult(r: float) -> str:
        return "n/a" if math.isnan(r) or r == 0 else f"{1 / r:.1f}x"

    def ratio_str(r: float) -> str:
        return "n/a" if math.isnan(r) else f"{r:.4f}"

    lines = [
        "# Economics & latency — gates G3 and G4",
        "",
        "Generated by `scripts/run_economics.py`. Every number below is derived",
        "from a measured run; the cost model is stated in full so the arithmetic",
        "can be checked or disputed.",
        "",
        "## Verdict",
        "",
        "| Gate | Measure | Student | Teacher | Ratio | Threshold | Verdict |",
        "|---|---|---|---|---|---|---|",
        (
            f"| G3 | cost per 1k records | {money(student.usd_per_1k)} | "
            f"{money(teacher.usd_per_1k)} | {ratio_str(cost_ratio)} | "
            f"<= {cost_max} | **{cost_v}** |"
        ),
        (
            f"| G4 | p95 latency | {student.p95_s:.2f}s | {teacher.p95_s:.2f}s | "
            f"{ratio_str(p95_ratio)} | <= {p95_max} | **{p95_v}** |"
        ),
        "",
        (
            f"**Cost reduction: {mult(cost_ratio)}** &nbsp;&nbsp; "
            f"**Latency reduction (p95): {mult(p95_ratio)}**"
        ),
        "",
        "## Measured latency",
        "",
        "| Side | Records | avg | p50 | p95 |",
        "|---|---|---|---|---|",
        f"| {teacher.name} | {teacher.records} | {teacher.avg_s:.2f}s | {teacher.p50_s:.2f}s | {teacher.p95_s:.2f}s |",
        f"| {student.name} | {student.records} | {student.avg_s:.2f}s | {student.p50_s:.2f}s | {student.p95_s:.2f}s |",
        "",
        "Latency is the successful request's round trip only; throttle sleeps and",
        "retry backoff are excluded, so free-tier rate limiting does not flatter",
        "the student.",
        "",
        "## Cost model",
        "",
        f"**{teacher.name}** — {teacher.basis}",
        "",
        f"**{student.name}** — {student.basis}",
        "",
        "### Assumptions (override with CLI flags to test sensitivity)",
        "",
        "| Input | Value |",
        "|---|---|",
    ]
    for k, v in pricing.items():
        lines.append(f"| `{k}` | {v} |")

    lines += ["", "### Honesty notes", ""]
    for side in (teacher, student):
        for n in side.notes:
            lines.append(f"- **{side.name}:** {n}")

    lines += [
        "",
        "### What this does NOT claim",
        "",
        "- Teacher latency is measured against one hosted provider on its free",
        "  tier; a dedicated deployment would be faster and the ratio smaller.",
        "- Student latency is unquantized batch-1 on Apple Silicon. Quantized",
        "  (GGUF/AWQ) numbers are reported separately once those artifacts pass",
        "  their own gate run.",
        "- No throughput batching is modelled on either side.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure G3 (cost) and G4 (latency).")
    ap.add_argument("--teacher-meta", type=Path, required=True, help="Teacher .meta.json")
    ap.add_argument("--student-meta", type=Path, required=True, help="Student .meta.json")
    ap.add_argument("--contract", type=Path, default=Path("contracts/pii_redaction_v2.yaml"))
    ap.add_argument("--output", type=Path, default=None, help="Write markdown report here")
    for key, val in DEFAULT_PRICING.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=float, default=val)
    args = ap.parse_args()

    pricing = {k: getattr(args, k) for k in DEFAULT_PRICING}

    for p in (args.teacher_meta, args.student_meta):
        if not p.exists():
            print(f"missing meta file: {p}", file=sys.stderr)
            return 1

    t_meta = json.loads(args.teacher_meta.read_text(encoding="utf-8"))
    s_meta = json.loads(args.student_meta.read_text(encoding="utf-8"))

    contract = load_contract(args.contract)
    gates = {
        "cost_ratio_max": contract.gates.cost_ratio_max,
        "p95_ratio_max": contract.gates.p95_ratio_max,
    }

    teacher = build_side(t_meta.get("model", "teacher"), t_meta, pricing, is_teacher=True)
    student_name = s_meta.get("model", "student")
    if s_meta.get("adapter"):
        student_name = f"{student_name} + {Path(s_meta['adapter']).parent.name}"
    student = build_side(student_name, s_meta, pricing, is_teacher=False)

    report = render(teacher, student, gates, pricing)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
