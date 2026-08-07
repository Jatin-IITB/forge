"""Forge — task-specialization distillation pipeline.

Exports the task contract, PII schema, evaluation harness, and inference adapter.
"""

from forge.contracts import TaskContract, load_contract
from forge.eval import EvalReport, evaluate
from forge.inference import build_messages, parse_response, reconstruct_offsets
from forge.schema import HIGH_SEVERITY, PIIRecord, PIISpan, PIIType

__all__ = [
    "HIGH_SEVERITY",
    "EvalReport",
    "PIIRecord",
    "PIISpan",
    "PIIType",
    "TaskContract",
    "build_messages",
    "evaluate",
    "load_contract",
    "parse_response",
    "reconstruct_offsets",
]
