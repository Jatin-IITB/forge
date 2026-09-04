#!/usr/bin/env bash
# Full-set draftless n-gram sweep. Short drafts are deliberate at -np 32.
set -euo pipefail

MODEL="${MODEL:-models/pii-1.5b-gguf/model-Q8_0.gguf}"
if [[ "$MODEL" != /* ]]; then MODEL="$(pwd)/$MODEL"; fi

SERVER="${SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
PY="${PY:-.venv/bin/python}"
PORT="${PORT:-8091}"
REPEAT="${REPEAT:-2}"
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
    local -a extra=("$@")
    local -a server_args=(
        -m "$MODEL" -ngl 99 --host 127.0.0.1 --port "$PORT"
        -np 32 -c 32768 --mlock
    )
    if (( ${#extra[@]} )); then server_args+=("${extra[@]}"); fi
    local server_cmd="llama-server -m $MODEL -ngl 99 -np 32 -c 32768 --mlock"
    if (( ${#extra[@]} )); then server_cmd+=" ${extra[*]}"; fi

    "$SERVER" "${server_args[@]}" > "logs/${name}_server.log" 2>&1 &
    SERVER_PID=$!

    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
        sleep 1
    done
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null

    "$PY" scripts/bench_serving.py \
        --backend openai \
        --model m \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --concurrency 32 \
        --repeat "$REPEAT" \
        --max-tokens 1024 \
        --compact-prompt \
        --retry-invalid json-schema \
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

run_config ngram_fresh_baseline
run_config ngram_map_n2_m2 \
    --spec-type ngram-map-k --spec-draft-n-max 2 \
    --spec-ngram-map-k-size-n 2 --spec-ngram-map-k-size-m 2 \
    --spec-ngram-map-k-min-hits 1
run_config ngram_map_n3_m3 \
    --spec-type ngram-map-k --spec-draft-n-max 3 \
    --spec-ngram-map-k-size-n 3 --spec-ngram-map-k-size-m 3 \
    --spec-ngram-map-k-min-hits 1
run_config ngram_simple_n2_m2 \
    --spec-type ngram-simple --spec-draft-n-max 2 \
    --spec-ngram-simple-size-n 2 --spec-ngram-simple-size-m 2 \
    --spec-ngram-simple-min-hits 1
run_config ngram_simple_n3_m3 \
    --spec-type ngram-simple --spec-draft-n-max 3 \
    --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 3 \
    --spec-ngram-simple-min-hits 1

echo "G3_NGRAM_DONE"
