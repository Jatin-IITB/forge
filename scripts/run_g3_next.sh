#!/usr/bin/env bash
# Definitive full-set comparison of compact fallback and LLGuidance paths.
# Invoke through scripts/nohup_run.sh; five passes take well over ten minutes.
set -euo pipefail

MODEL="${MODEL:-models/pii-1.5b-gguf/model-Q8_0.gguf}"
if [[ "$MODEL" != /* ]]; then MODEL="$(pwd)/$MODEL"; fi

SERVER="${SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
PY="${PY:-.venv/bin/python}"
PORT="${PORT:-8091}"
SESSION="${SESSION:-a}"
REPEAT="${REPEAT:-5}"
LLAMA_COMMIT="$(git -C "$HOME/llama.cpp" rev-parse --short HEAD)"
SERVER_CMD="llama-server -m $MODEL -ngl 99 -np 32 -c 32768 --mlock (LLGuidance)"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

"$SERVER" -m "$MODEL" -ngl 99 --host 127.0.0.1 --port "$PORT" \
    -np 32 -c 32768 --mlock > "logs/g3_next_${SESSION}_server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null

"$PY" scripts/bench_grammar.py \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --experiment compact \
    --concurrency 32 \
    --repeat "$REPEAT" \
    --max-tokens 1024 \
    --server-cmd "$SERVER_CMD" \
    --llama-cpp-commit "$LLAMA_COMMIT" \
    --out "reports/bench/g3_next_${SESSION}.json" \
    --save-predictions "data/predictions_g3_next_${SESSION}.jsonl"

for arm in \
    verbose_baseline \
    compact_prompt_grammar \
    compact_prompt_retry \
    compact_prompt_json_schema \
    compact_prompt_json_retry
do
    "$PY" scripts/run_eval.py \
        data/gold/test.jsonl \
        "data/predictions_g3_next_${SESSION}_${arm}.jsonl" \
        --ci \
        --validators \
        --teacher-preds data/predictions_teacher_120b_relat.jsonl \
        > "reports/bench/g3_next_${SESSION}_${arm}_eval.txt"
done

echo "G3_NEXT_${SESSION}_DONE"
