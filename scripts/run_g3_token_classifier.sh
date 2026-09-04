#!/usr/bin/env bash
# Train and gate full-attention BIOES classification, with causal ablation.
set -euo pipefail

PY="${PY:-.venv/bin/python}"
BASE="${BASE:-models/pii-1.5b-merged}"
EPOCHS="${EPOCHS:-3}"
REPEAT="${REPEAT:-3}"

"$PY" scripts/bench_token_floor.py \
    --model "$BASE" \
    --batch-sizes 8 16 32 \
    --repeat "$REPEAT" \
    --out reports/bench/token_classifier_floor.json

for attention in full causal; do
    output="checkpoints/g3_token_${attention}"
    attention_args=()
    if [[ "$attention" == "full" ]]; then attention_args=(--full-attention); fi

    "$PY" -u scripts/train_token_classifier.py \
        --train-data data/train_v2.jsonl \
        --train-data data/train_v3.jsonl \
        --val-data data/gold/val.jsonl \
        --base-model "$BASE" \
        --output-dir "$output" \
        --epochs "$EPOCHS" \
        --batch-size 2 \
        --grad-accum 8 \
        --resume \
        ${attention_args[@]+"${attention_args[@]}"}

    for batch_size in 8 16 32; do
        name="token_${attention}_b${batch_size}"
        "$PY" scripts/bench_token_classifier.py \
            --model "$output/final-merged" \
            --batch-size "$batch_size" \
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

echo "G3_TOKEN_CLASSIFIER_DONE"
