#!/usr/bin/env bash
# Convert the exact merged checkpoint and compare MLX BatchGenerator directly
# against a fresh llama.cpp compact/retry baseline.
set -euo pipefail

PY="${PY:-.venv/bin/python}"
HF_MODEL="${HF_MODEL:-models/pii-1.5b-merged}"
MLX_FP16="${MLX_FP16:-models/pii-1.5b-mlx-fp16}"
MLX_Q8="${MLX_Q8:-models/pii-1.5b-mlx-q8}"
REPEAT="${REPEAT:-3}"
PORT="${PORT:-8091}"
SERVER="${SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
GGUF="${GGUF:-models/pii-1.5b-gguf/model-Q8_0.gguf}"
if [[ "$GGUF" != /* ]]; then GGUF="$(pwd)/$GGUF"; fi
LLAMA_COMMIT="$(git -C "$HOME/llama.cpp" rev-parse --short HEAD)"

"$PY" -m mlx_lm convert \
    --hf-path "$HF_MODEL" --mlx-path "$MLX_FP16" --dtype float16
"$PY" -m mlx_lm convert \
    --hf-path "$HF_MODEL" --mlx-path "$MLX_Q8" --quantize --q-bits 8

"$SERVER" -m "$GGUF" -ngl 99 --host 127.0.0.1 --port "$PORT" \
    -np 32 -c 32768 --mlock > logs/mlx_fresh_baseline_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
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
    --compact-prompt \
    --retry-invalid json-schema \
    --config-name mlx-fresh-llama-baseline \
    --server-cmd "llama-server -m $GGUF -ngl 99 -np 32 -c 32768 --mlock" \
    --llama-cpp-commit "$LLAMA_COMMIT" \
    --out reports/bench/mlx_fresh_llama_baseline.json \
    --save-predictions data/predictions_mlx_fresh_llama_baseline.jsonl
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

for format in fp16 q8; do
    model="$MLX_FP16"
    if [[ "$format" == "q8" ]]; then model="$MLX_Q8"; fi
    for batch_size in 4 8 16 24 32; do
        name="mlx_${format}_b${batch_size}"
        "$PY" scripts/bench_mlx_batch.py \
            --model "$model" \
            --completion-batch-size "$batch_size" \
            --prefill-batch-size 8 \
            --repeat "$REPEAT" \
            --config-name "$name" \
            --out "reports/bench/${name}.json" \
            --save-predictions "data/predictions_${name}.jsonl"
        "$PY" scripts/run_eval.py \
            data/gold/test.jsonl \
            "data/predictions_${name}.jsonl" \
            --ci \
            --validators \
            --teacher-preds data/predictions_teacher_120b_relat.jsonl \
            > "reports/bench/${name}_eval.txt"
    done
done

echo "G3_MLX_DONE"
