#!/usr/bin/env bash
# Retrain the shipped task checkpoint on the compact escaped-line protocol,
# export Q8_0, then run full quality/economics evidence.
set -euo pipefail

PY="${PY:-.venv/bin/python}"
BASE="${BASE:-models/pii-1.5b-merged}"
CHECKPOINT="${CHECKPOINT:-checkpoints/g3_line}"
MERGED="${MERGED:-models/pii-g3-line-merged}"
GGUF_DIR="${GGUF_DIR:-models/pii-g3-line-gguf}"
EPOCHS="${EPOCHS:-2}"
REPEAT="${REPEAT:-3}"
PORT="${PORT:-8091}"
SERVER="${SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
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

"$PY" -u scripts/run_train.py \
    --train-data data/train_v2.jsonl \
    --train-data data/train_v3.jsonl \
    --val-data data/gold/val.jsonl \
    --base-model "$BASE" \
    --output-dir "$CHECKPOINT" \
    --output-format line \
    --epochs "$EPOCHS" \
    --batch-size 4 \
    --grad-accum 4 \
    --lora-target-mlp \
    --save-steps 100 \
    --resume

"$PY" scripts/export_model.py merge \
    --base "$BASE" \
    --adapter "$CHECKPOINT/final" \
    --output "$MERGED"
"$PY" scripts/export_model.py gguf \
    --merged "$MERGED" \
    --output "$GGUF_DIR" \
    --quant Q8_0

run_model() {
    local name="$1"
    local model="$2"
    shift 2
    local -a format_args=("$@")
    local -a bench_format_args=()
    if (( ${#format_args[@]} )); then bench_format_args=("${format_args[@]}"); fi
    if [[ "$model" != /* ]]; then model="$(pwd)/$model"; fi
    local server_cmd="llama-server -m $model -ngl 99 -np 32 -c 32768 --mlock"

    "$SERVER" -m "$model" -ngl 99 --host 127.0.0.1 --port "$PORT" \
        -np 32 -c 32768 --mlock > "logs/${name}_server.log" 2>&1 &
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
        --quant Q8_0 \
        --config-name "$name" \
        --server-cmd "$server_cmd" \
        --llama-cpp-commit "$LLAMA_COMMIT" \
        ${bench_format_args[@]+"${bench_format_args[@]}"} \
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

run_model line_retrain_fresh_baseline models/pii-1.5b-gguf/model-Q8_0.gguf
run_model line_retrain_unconstrained "$GGUF_DIR/model-Q8_0.gguf" --line-prompt
run_model line_retrain_retry "$GGUF_DIR/model-Q8_0.gguf" \
    --line-prompt --retry-invalid line-gbnf

echo "G3_LINE_RETRAIN_DONE"
