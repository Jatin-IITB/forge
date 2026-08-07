"""The committed contract must load, validate, and satisfy independence invariants."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from forge.contracts import TaskContract, load_contract

CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "pii_redaction_v1.yaml"


def test_contract_loads_and_validates():
    c = load_contract(CONTRACT)
    assert c.task_id == "pii_redaction_v1"
    assert 0 < c.gates.parity_target <= 1


def test_contract_is_immutable():
    c = load_contract(CONTRACT)
    with pytest.raises(ValidationError):  # frozen model
        c.task_id = "mutated"  # type: ignore[misc]


def test_independence_invariants_hold():
    c = load_contract(CONTRACT)
    assert c.teacher.open_weight and c.teacher.distillation_permitted
    assert c.base_model.open_weight
    # high-severity recall floors must exist for a PII task
    assert c.gates.high_severity_recall_floors


def test_teacher_without_distillation_rights_is_rejected():
    bad = {
        "task_id": "x",
        "version": "1",
        "description": "d",
        "io_schema": {"input": "i", "output": "o"},
        "metric": {"primary": "f1"},
        "gates": {
            "parity_target": 0.98,
            "cost_ratio_max": 0.1,
            "p95_ratio_max": 0.2,
            "high_severity_recall_floors": [{"label": "SSN", "min_recall": 0.99}],
        },
        "teacher": {
            "name": "Closed/Model",
            "license": "proprietary",
            "open_weight": False,
            "distillation_permitted": False,
        },
        "base_model": {
            "name": "Open/Base",
            "license": "Apache-2.0",
            "open_weight": True,
            "distillation_permitted": True,
        },
        "constraints": {"max_params_b": 3, "target_hardware": "cpu", "privacy": "local"},
        "guardrails": {"in_domain": "x", "ood_behavior": "y"},
        "data": {
            "gold_source": "s",
            "gold_license": "MIT",
            "leakage_policy": "p",
        },
    }
    with pytest.raises(ValueError):
        TaskContract.model_validate(bad)
