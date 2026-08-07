"""The TaskContract — Forge's load-bearing object (DESIGN §1, ADR 0001).

A contract declares, *before any modelling*, exactly what is being built and how
success is measured: the IO schema, the metric, the parity gate, the teacher/base
models (with licences), the deployment constraints, and the guardrails. It is
committed to git first and is immutable for a training run.

This module is the *code* form of that contract: load a ``contracts/*.yaml`` and it is
validated against these models, so a malformed or under-specified contract fails loudly
instead of silently producing a meaningless run. Pure structure — no modelling deps.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from forge.schema import PIIType


class ModelRef(BaseModel):
    """A teacher or base model, pinned with its licence so independence is auditable."""

    name: str = Field(min_length=1, description="HF repo id, e.g. 'Qwen/Qwen2.5-32B-Instruct'.")
    license: str = Field(min_length=1, description="SPDX-ish licence id, e.g. 'Apache-2.0'.")
    open_weight: bool = Field(description="True iff weights are downloadable (independence).")
    distillation_permitted: bool = Field(
        description="True iff the licence permits training other models on its outputs."
    )
    notes: str | None = None


class MetricSpec(BaseModel):
    """Primary + secondary metrics. The primary is what the parity gate is computed on."""

    primary: str = Field(min_length=1)
    secondary: list[str] = Field(default_factory=list)
    span_match: str = Field(
        default="exact",
        pattern="^(exact|partial|overlap)$",
        description="How a predicted span is credited against gold.",
    )


class RecallFloor(BaseModel):
    """A hard per-type recall floor — a leak of a high-severity type is a breach."""

    label: str
    min_recall: float = Field(ge=0.0, le=1.0)

    @field_validator("label")
    @classmethod
    def _label_is_valid_pii_type(cls, v: str) -> str:
        try:
            PIIType(v)
        except ValueError:
            valid = ", ".join(t.value for t in PIIType)
            raise ValueError(f"{v!r} is not a valid PIIType. Valid: {valid}") from None
        return v


class Gates(BaseModel):
    """The six pre-committed gates (DESIGN §3, SUCCESS.md). All measured with CIs."""

    parity_target: float = Field(
        gt=0.0, le=1.0, description="student_score >= parity_target * teacher_score."
    )
    schema_validity_min: float = Field(ge=0.0, le=1.0, default=0.999)
    cost_ratio_max: float = Field(
        gt=0.0, description="student_cost_per_1k <= teacher_cost_per_1k * cost_ratio_max."
    )
    p95_ratio_max: float = Field(
        gt=0.0, description="student_p95 <= teacher_p95 * p95_ratio_max."
    )
    high_severity_recall_floors: list[RecallFloor] = Field(default_factory=list)
    ood_refusal_min: float = Field(ge=0.0, le=1.0, default=0.90)


class Constraints(BaseModel):
    max_params_b: float = Field(gt=0.0, description="Max student size in billions of params.")
    target_hardware: str = Field(min_length=1)
    privacy: str = Field(min_length=1, description="e.g. 'on-device / air-gapped'.")


class Guardrails(BaseModel):
    in_domain: str = Field(min_length=1, description="What counts as a valid input.")
    ood_behavior: str = Field(min_length=1, description="What to do with out-of-domain input.")


class DataProvenance(BaseModel):
    """Where eval/training data comes from — gated by the independence litmus test."""

    gold_source: str = Field(min_length=1)
    gold_license: str = Field(min_length=1)
    rejected_sources: list[str] = Field(default_factory=list)
    leakage_policy: str = Field(min_length=1)


class IOSchema(BaseModel):
    input: str = Field(min_length=1)
    output: str = Field(min_length=1)
    schema_module: str = Field(
        default="forge.schema",
        description="Python module defining the output record model.",
    )


class TaskContract(BaseModel):
    """The immutable spec a Forge run compiles into a model."""

    model_config = {"frozen": True}

    task_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    io_schema: IOSchema
    metric: MetricSpec
    gates: Gates
    teacher: ModelRef
    base_model: ModelRef
    constraints: Constraints
    guardrails: Guardrails
    data: DataProvenance

    @model_validator(mode="after")
    def _independence(self) -> TaskContract:
        # ADR 0003: the asset must be rebuildable by a stranger. Enforce it structurally.
        if not self.teacher.distillation_permitted:
            raise ValueError(
                f"teacher {self.teacher.name} licence does not permit distillation — "
                "violates ADR 0003 independence."
            )
        if not self.base_model.open_weight:
            raise ValueError(
                f"base model {self.base_model.name} is not open-weight — "
                "violates ADR 0003 independence."
            )
        floors = {f.label for f in self.gates.high_severity_recall_floors}
        if not floors:
            raise ValueError(
                "PII contract must set at least one high-severity recall floor "
                "(a leaked credential is a breach)."
            )
        return self


def load_contract(path: str | Path) -> TaskContract:
    """Parse and validate a contract YAML. Raises on any violation."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TaskContract.model_validate(raw)
