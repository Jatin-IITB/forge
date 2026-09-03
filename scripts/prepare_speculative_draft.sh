#!/usr/bin/env bash
# Download and convert the independent speculative draft model.
#
# Source: Qwen/Qwen2.5-0.5B-Instruct, Apache-2.0, public and ungated.
# Pinned revision is the model repository SHA recorded by Hugging Face.
set -euo pipefail

HF_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
HF_REVISION="7ae557604adf67be50417f59c2c2f167def9a775"
HF_DIR="models/qwen2.5-0.5b-instruct"
GGUF_DIR="models/qwen2.5-0.5b-instruct-gguf"
PY="${PY:-.venv/bin/python}"

.venv/bin/hf download "$HF_MODEL" \
    --revision "$HF_REVISION" \
    --local-dir "$HF_DIR"

"$PY" scripts/export_model.py gguf \
    --merged "$HF_DIR" \
    --output "$GGUF_DIR" \
    --quant Q8_0 \
    --llama-cpp "$HOME/llama.cpp"

echo "SPEC_DRAFT_READY"
