#!/usr/bin/env bash
# Full-385 speculative-decoding sweep with a fresh repeated baseline.
# Run through scripts/nohup_run.sh; the complete sweep takes over ten minutes.
set -euo pipefail

TARGET="${TARGET:-models/pii-1.5b-gguf/model-Q8_0.gguf}"
DRAFT="${DRAFT:-models/qwen2.5-0.5b-instruct-gguf/model-Q8_0.gguf}"
if [[ "$TARGET" != /* ]]; then TARGET="$(pwd)/$TARGET"; fi
if [[ "$DRAFT" != /* ]]; then DRAFT="$(pwd)/$DRAFT"; fi

SERVER="$HOME/llama.cpp/build/bin/llama-server"
PY="${PY:-.venv/bin/python}"
PORT="${PORT:-8091}"
LLAMA_COMMIT="$(git -C "$HOME/llama.cpp" rev-parse --short HEAD)"
SERVER_PID=""

cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup_server EXIT

run_config() {
    local name="$1"
    shift
    local -a extra=()
    if (( $# > 0 )); then
        extra=("$@")
    fi
    local server_cmd="llama-server -m $TARGET -ngl 99 -np 32 -c 32768 --mlock"
    if (( ${#extra[@]} > 0 )); then
        server_cmd+=" ${extra[*]}"
    fi

    if (( ${#extra[@]} > 0 )); then
        "$SERVER" -m "$TARGET" -ngl 99 --host 127.0.0.1 --port "$PORT" \
            -np 32 -c 32768 --mlock "${extra[@]}" \
            > "logs/${name}_server.log" 2>&1 &
    else
        "$SERVER" -m "$TARGET" -ngl 99 --host 127.0.0.1 --port "$PORT" \
            -np 32 -c 32768 --mlock \
            > "logs/${name}_server.log" 2>&1 &
    fi
    SERVER_PID=$!

    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null

    "$PY" scripts/bench_serving.py \
        --backend openai \
        --model m \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --concurrency 32 \
        --repeat 2 \
        --max-tokens 1024 \
        --quant Q8_0 \
        --config-name "$name" \
        --server-cmd "$server_cmd" \
        --llama-cpp-commit "$LLAMA_COMMIT" \
        --out "reports/bench/${name}.json" \
        --save-predictions "data/predictions_${name}.jsonl"

    "$PY" scripts/run_eval.py \
        data/gold/test.jsonl \
        "data/predictions_${name}.jsonl" \
        --ci \
        --validators \
        --teacher-preds data/predictions_teacher_120b_relat.jsonl \
        > "reports/bench/${name}_eval.txt"

    cleanup_server
}

run_config spec_fresh_baseline
run_config spec_draft3 \
    -md "$DRAFT" -ngld 99 --spec-type draft-simple \
    --spec-draft-n-max 3 --spec-draft-n-min 0
run_config spec_draft8 \
    -md "$DRAFT" -ngld 99 --spec-type draft-simple \
    --spec-draft-n-max 8 --spec-draft-n-min 0

echo "SPEC_SWEEP_DONE"
