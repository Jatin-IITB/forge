#!/usr/bin/env bash
# Full-385, repeated A/B/C for the G3 output-shortening hypotheses.
#
# Run through scripts/nohup_run.sh: two passes of three arms take longer than
# ten minutes, and laptop sleep has invalidated long measurements before.
set -euo pipefail

MODEL="${1:-models/pii-1.5b-gguf/model-Q8_0.gguf}"
if [[ "$MODEL" != /* ]]; then
    MODEL="$(pwd)/$MODEL"
fi
SERVER="$HOME/llama.cpp/build/bin/llama-server"
PY="${PY:-.venv/bin/python}"
PORT="${PORT:-8091}"
SERVER_CMD="llama-server -m $MODEL -ngl 99 -np 32 -c 32768 --mlock"

"$SERVER" -m "$MODEL" -ngl 99 --host 127.0.0.1 --port "$PORT" \
    -np 32 -c 32768 --mlock > logs/compact_ab_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null

LLAMA_COMMIT="$(git -C "$HOME/llama.cpp" rev-parse --short HEAD)"
"$PY" scripts/bench_grammar.py \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --experiment compact \
    --concurrency 32 \
    --repeat 2 \
    --max-tokens 1024 \
    --server-cmd "$SERVER_CMD" \
    --llama-cpp-commit "$LLAMA_COMMIT" \
    --out reports/bench/compact_abcd.json \
    --save-predictions data/predictions_student_q8_compact_v2.jsonl

for arm in verbose_baseline compact_grammar compact_prompt compact_prompt_grammar; do
    "$PY" scripts/run_eval.py \
        data/gold/test.jsonl \
        "data/predictions_student_q8_compact_v2_${arm}.jsonl" \
        --ci \
        --validators \
        --teacher-preds data/predictions_teacher_120b_relat.jsonl \
        > "reports/bench/compact_v2_${arm}_eval.txt"
done

echo "COMPACT_AB_DONE"
