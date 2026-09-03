#!/usr/bin/env bash
# Wait for any in-flight benchmark to release the GPU, then run the three-arm
# decoding-constraint A/B on a server configured to match the first run.
#
# Split out of an inline command because the work outlives a tool-call timeout:
# three arms x 385 records is ~20 minutes, and it may first have to wait out a
# KV sweep. Launch through scripts/nohup_run.sh so it survives that.
#
# Usage: scripts/nohup_run.sh logs/grammar_ab.log scripts/run_grammar_ab.sh [wait_pid]
set -euo pipefail

WAIT_PID="${1:-}"
MODEL=models/pii-1.5b-gguf/model-Q8_0.gguf
SERVER="$HOME/llama.cpp/build/bin/llama-server"
PY="${PY:-.venv/bin/python}"

if [ -n "$WAIT_PID" ]; then
    echo "waiting for pid $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
    echo "released."
fi

pkill -f "llama-server -m $MODEL" 2>/dev/null || true
sleep 4

# -np 8 -c 8192 reproduces the server used for the first A/B. The decoding
# constraint must be the only variable: batched Metal decode is not
# bit-deterministic, so a different -np would confound the two.
nohup "$SERVER" -m "$MODEL" -ngl 99 --host 127.0.0.1 --port 8080 \
    -np 8 -c 8192 --mlock > logs/grammar_ab_server.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 40); do
    sleep 3
    curl -s -m 2 http://localhost:8080/health >/dev/null 2>&1 && break
done
echo "server up (np=8 c=8192)"

"$PY" scripts/bench_grammar.py \
    --base-url http://localhost:8080/v1 \
    --out reports/bench/grammar_abc.json \
    --save-predictions data/predictions_student_q8.jsonl

kill "$SERVER_PID" 2>/dev/null || true
echo "GRAMMAR_AB_DONE"
