#!/usr/bin/env bash
# G3 throughput levers WP-1 did not test: KV-cache quantization and flash attention.
#
# WP-1 swept -np, -c, -ub, --mlock and --kv-unified and concluded G3 needs a
# further 1.89x that "more of what has already been tried" cannot buy. Two flags
# were never exercised:
#
#   -ctk/-ctv q8_0   The KV cache is f16 by default. At -np 32 the cache is read
#                    every decode step for all 32 sequences, so at batch 32 --
#                    which WP-1 measured as compute/bandwidth bound rather than
#                    weight bound -- halving KV bandwidth attacks the actual
#                    bottleneck, unlike quantizing the weights further (measured
#                    to buy nothing at that batch size).
#
#   -fa on           Defaults to 'auto'. Whether Metal actually selects it is not
#                    recorded in any artifact, so 'auto' vs explicit 'on' is
#                    worth one run to remove the ambiguity.
#
# Quality is NOT assumed unchanged: a quantized KV cache can move logits, so any
# config that wins here must re-pass the gate suite before it ships. That check
# is the caller's job and is stated in the report.
#
# Usage: scripts/bench_kv_sweep.sh [model.gguf]
set -euo pipefail

MODEL="${1:-models/pii-1.5b-gguf/model-Q8_0.gguf}"
SERVER="$HOME/llama.cpp/build/bin/llama-server"
PORT=8080
PY="${PY:-.venv/bin/python}"
GOLD="data/gold/test.jsonl"

run_config() {
    local name="$1"; shift
    echo "=== $name : $* ==="
    pkill -f "llama-server -m $MODEL" 2>/dev/null || true
    sleep 3
    nohup "$SERVER" -m "$MODEL" -ngl 99 --host 127.0.0.1 --port "$PORT" \
        "$@" > "logs/kvsweep_${name}.log" 2>&1 &
    for _ in $(seq 1 30); do
        sleep 3
        curl -s -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1 && break
    done
    "$PY" scripts/bench_serving.py \
        --backend openai --model m --base-url "http://localhost:$PORT/v1" \
        --gold "$GOLD" --concurrency 32 --max-tokens 1024 \
        --quant Q8_0 --config-name "$name" \
        --out "reports/bench/${name}.json" 2>&1 | tail -6
}

mkdir -p logs reports/bench
# Baseline reproduces WP-1's shipped throughput config on this machine's current
# state, so the deltas below are not read against a number measured hours ago.
run_config kv_baseline   -np 32 -c 32768 --mlock
run_config kv_q8         -np 32 -c 32768 --mlock -ctk q8_0 -ctv q8_0
run_config kv_q8_fa      -np 32 -c 32768 --mlock -ctk q8_0 -ctv q8_0 -fa on
run_config kv_q4         -np 32 -c 32768 --mlock -ctk q4_0 -ctv q4_0

pkill -f "llama-server -m $MODEL" 2>/dev/null || true
echo "done — compare with: $PY -c \"import json,glob; [print(f, json.load(open(f))['sustained_s_per_record']) for f in sorted(glob.glob('reports/bench/kv_*.json'))]\""
