"""Forge — task-specialization distillation pipeline (Phase 0 scaffolding).

Phase 0 ships only *structure*: the task contract and the IO/gold schema. No modelling
code yet, by design (eval-first, ADR 0001).
"""

from forge.contracts import TaskContract, load_contract
from forge.schema import HIGH_SEVERITY, PIIRecord, PIISpan, PIIType

__all__ = [
    "TaskContract",
    "load_contract",
    "PIIType",
    "PIISpan",
    "PIIRecord",
    "HIGH_SEVERITY",
]
