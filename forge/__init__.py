"""Forge — task-specialization distillation pipeline.

Exports the task contract, PII schema, evaluation harness, inference adapter,
verification gate, and dedup utilities.
"""

from forge.contracts import TaskContract, load_contract
from forge.dedup import dedup_training_data
from forge.eval import EvalReport, evaluate
from forge.inference import build_messages, parse_response, reconstruct_offsets
from forge.schema import HIGH_SEVERITY, PIIRecord, PIISpan, PIIType
from forge.train import LORA_DEFAULTS, SFT_DEFAULTS, load_training_data, record_to_chat
from forge.verify import RejectReason, VerifiedRecord, majority_vote_spans, verify_record

__all__ = [
    "HIGH_SEVERITY",
    "LORA_DEFAULTS",
    "SFT_DEFAULTS",
    "EvalReport",
    "PIIRecord",
    "PIISpan",
    "PIIType",
    "RejectReason",
    "TaskContract",
    "VerifiedRecord",
    "build_messages",
    "dedup_training_data",
    "evaluate",
    "load_contract",
    "load_training_data",
    "majority_vote_spans",
    "parse_response",
    "reconstruct_offsets",
    "record_to_chat",
    "verify_record",
]
