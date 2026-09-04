#!/usr/bin/env bash
# Start llama-server with a given config, benchmark it, tear it down.
#
# Exists because a serving measurement is only meaningful with the server
# arguments attached to it. Starting the server by hand and running the
# benchmark separately makes it easy to publish a number whose config nobody can
# reconstruct. Here the exact command line, the resident set size, and the
# per-slot context the server actually chose are captured into the same artifact
# as the timings.
#
# Usage:
#   scripts/bench_sweep.sh <name> <gguf> <server-args...> -- <bench-args...>
#
# Example:
#   scripts/bench_sweep.sh lat_c2048 models/pii-1.5b-gguf/model-Q4_K_M.gguf \
#       -np 1 -c 2048 -- --concurrency 1 --limit 48

set -euo pipefail

NAME="$1"; shift
GGUF="$1"; shift

SERVER_ARGS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do SERVER_ARGS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
BENCH_ARGS=("$@")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${LLAMA_CPP:-$HOME/llama.cpp}"
PORT="${PORT:-8080}"
LOG="/tmp/srv_${NAME}.log"

cd "$REPO"

CMD="llama-server -m $GGUF -ngl 99 ${SERVER_ARGS[*]}"
echo "=== $NAME"
echo "    $CMD"

"$LLAMA/build/bin/llama-server" -m "$GGUF" -ngl 99 "${SERVER_ARGS[@]}" \
    --port "$PORT" --host 127.0.0.1 >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true' EXIT

# Wait for readiness rather than sleeping a fixed interval: model load time
# varies with page cache state, and a fixed sleep either wastes time or starts
# the benchmark against a server that is still mapping weights.
for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done

# Resident set size AFTER load but BEFORE traffic. This is the number the
# "-c reduction" question is really asking about: how much memory does this
# configuration reserve just by existing.
RSS_MB=$(ps -o rss= -p "$SRV" | awk '{printf "%.0f", $1/1024}')
NCTX=$(grep -o "n_ctx_slot = [0-9]*" "$LOG" | head -1 | grep -o "[0-9]*" || echo "?")
NSLOTS=$(grep -o "n_slots = [0-9]*" "$LOG" | head -1 | grep -o "[0-9]*" || echo "?")
echo "    n_slots=$NSLOTS  n_ctx_slot=$NCTX  RSS=${RSS_MB}MB  load=$(uptime | sed 's/.*averages: //')"

.venv/bin/python scripts/bench_serving.py \
    --backend openai --model "$(basename "$GGUF")" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --config-name "$NAME" --quant "$(basename "$GGUF" .gguf | sed 's/^model-//')" \
    --server-cmd "$CMD" \
    --llama-cpp-commit "$(git -C "$LLAMA" rev-parse --short HEAD)" \
    "${BENCH_ARGS[@]}"

# Fold the server-side facts into the artifact the benchmark just wrote, so the
# config and its cost live in one file.
OUT=""
for i in "${!BENCH_ARGS[@]}"; do
    [[ "${BENCH_ARGS[$i]}" == "--out" ]] && OUT="${BENCH_ARGS[$((i+1))]}"
done
if [[ -n "$OUT" && -f "$OUT" ]]; then
    RSS_MB="$RSS_MB" NCTX="$NCTX" NSLOTS="$NSLOTS" OUT="$OUT" .venv/bin/python - <<'PY'
import json, os
p = os.environ["OUT"]
d = json.load(open(p))
d["server"] = {
    "rss_mb_after_load": int(os.environ["RSS_MB"]),
    "n_ctx_slot": int(os.environ["NCTX"]) if os.environ["NCTX"].isdigit() else None,
    "n_slots": int(os.environ["NSLOTS"]) if os.environ["NSLOTS"].isdigit() else None,
}
json.dump(d, open(p, "w"), indent=2)
open(p, "a").write("\n")
PY
fi
