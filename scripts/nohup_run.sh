#!/usr/bin/env bash
# Launch a long-running Forge job detached from the calling shell.
#
# Long jobs (training, teacher scoring) must survive terminal close, IDE
# restarts, and machine idle. macOS has no setsid, so we rely on:
#   nohup      — ignore SIGHUP when the parent shell exits
#   caffeinate — prevent idle sleep for the job's lifetime (-i idle, -m disk,
#                -s while on AC power)
#   disown     — detach from the shell's job table
#
# Lid-close on battery still sleeps the machine; nothing in userland prevents
# that. Every long job in this repo therefore checkpoints incrementally and
# supports --resume, so a sleep costs minutes, not hours.
#
# Usage:
#   scripts/nohup_run.sh <logfile> <command> [args...]
#
# Example:
#   scripts/nohup_run.sh logs/train.log .venv/bin/python -u scripts/run_train.py --resume ...

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <logfile> <command> [args...]" >&2
    exit 2
fi

LOGFILE="$1"
shift

mkdir -p "$(dirname "$LOGFILE")"

nohup caffeinate -ims "$@" > "$LOGFILE" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true

echo "$PID" > "${LOGFILE}.pid"
echo "launched: pid=$PID log=$LOGFILE"
