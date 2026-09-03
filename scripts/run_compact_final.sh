#!/usr/bin/env bash
# Definitive G3 run: standard serving harness, full 385, three passes.
# Launch with scripts/nohup_run.sh because this takes longer than ten minutes.
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
    -np 32 -c 32768 --mlock > logs/compact_final_server.log 2>&1 &
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
COMMON=(
    --backend openai
    --model m
    --base-url "http://127.0.0.1:$PORT/v1"
    --concurrency 32
    --repeat 3
    --max-tokens 1024
    --quant Q8_0
    --server-cmd "$SERVER_CMD"
    --llama-cpp-commit "$LLAMA_COMMIT"
)

"$PY" scripts/bench_serving.py \
    "${COMMON[@]}" \
    --config-name q8-fresh-baseline \
    --out reports/bench/FINAL_q8_fresh_baseline.json

"$PY" scripts/bench_serving.py \
    "${COMMON[@]}" \
    --compact-prompt \
    --compact-grammar \
    --config-name q8-compact-prompt-grammar \
    --out reports/bench/FINAL_compact_throughput.json \
    --save-predictions data/predictions_student_q8_compact_final.jsonl

"$PY" scripts/run_eval.py \
    data/gold/test.jsonl \
    data/predictions_student_q8_compact_final.jsonl \
    --ci \
    --validators \
    --teacher-preds data/predictions_teacher_120b_relat.jsonl \
    > reports/bench/FINAL_compact_eval.txt

echo "COMPACT_FINAL_DONE"
