#!/usr/bin/env python3
"""Validate a TaskContract YAML and (optionally) a gold JSONL against the schema.

Phase 0's exit gate is partly mechanical: the contract must parse and satisfy the
independence invariants, and any committed gold records must conform to PIIRecord.
This script makes that checkable in CI / `make validate`.

Usage:
    python scripts/validate_contract.py contracts/pii_redaction_v1.yaml
    python scripts/validate_contract.py contracts/pii_redaction_v1.yaml --gold data/gold/sample.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.contracts import load_contract
from forge.schema import PIIRecord


def validate_gold(path: Path) -> int:
    n = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            PIIRecord.model_validate_json(line)
        except Exception as e:
            print(f"  [FAIL] {path}:{i}: {e}")
            raise
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("contract", type=Path)
    ap.add_argument("--gold", type=Path, default=None)
    args = ap.parse_args()

    c = load_contract(args.contract)
    print(f"[OK] contract '{c.task_id}' v{c.version} valid")
    print(f"     teacher  : {c.teacher.name} ({c.teacher.license})")
    print(f"     base     : {c.base_model.name} ({c.base_model.license})")
    print(f"     parity   : student >= {c.gates.parity_target} x teacher")
    print(f"     gold src : {c.data.gold_source.splitlines()[0].strip()}")

    if args.gold:
        if not args.gold.exists():
            print(f"[FAIL] gold file not found: {args.gold}")
            return 1
        n = validate_gold(args.gold)
        print(f"[OK] {n} gold records conform to forge.schema.PIIRecord")
    return 0


if __name__ == "__main__":
    sys.exit(main())
