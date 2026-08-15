#!/usr/bin/env bash
# Score a finished training run end to end: inference -> gates -> error analysis.
#
# Every stage resumes, so an interrupted evaluation costs minutes rather than
# restarting the 385-record inference pass.
#
# Usage: scripts/evaluate_run.sh run_002 [teacher_f1]

set -euo pipefail

RUN="${1:?usage: $0 <run_id> [teacher_f1]}"
TEACHER_F1="${2:-0.9482}"   # GPT-OSS-120B on the frozen test set, reports/baseline_120b.md

BASE="Qwen/Qwen2.5-1.5B-Instruct"
GOLD="data/gold/test.jsonl"
ADAPTER="checkpoints/${RUN}/final"
PREDS="data/predictions_student_${RUN}.jsonl"
CONTRACT="contracts/pii_redaction_v2.yaml"
PY=".venv/bin/python"

if [ ! -f "${ADAPTER}/adapter_config.json" ]; then
    echo "no adapter at ${ADAPTER} — has training finished?" >&2
    exit 1
fi

echo "=== [1/4] inference: ${RUN} over ${GOLD}"
"$PY" -u scripts/run_inference.py "$GOLD" "$PREDS" \
    --model "$BASE" --adapter "$ADAPTER" --resume

echo
echo "=== [2/4] gates (model-only, vs teacher F1 ${TEACHER_F1})"
"$PY" scripts/run_eval.py "$GOLD" "$PREDS" \
    --contract "$CONTRACT" --check-gates --teacher-f1 "$TEACHER_F1" || true

echo
echo "=== [3/4] model-only vs system (ADR 0012 validators)"
"$PY" scripts/run_eval.py "$GOLD" "$PREDS" \
    --contract "$CONTRACT" --validators

echo
echo "=== [4/4] error analysis -> next augmentation targets"
"$PY" scripts/error_analysis.py "$GOLD" "$PREDS" \
    --train-data data/train_v2.jsonl \
    --output "data/error_analysis_${RUN}.json" > /dev/null

echo
echo "done. predictions=${PREDS}  analysis=data/error_analysis_${RUN}.json"
