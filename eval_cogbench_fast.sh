#!/bin/bash

set -euo pipefail

MODEL_PATH=""
REVISION_NAME=${REVISION_NAME:-""}
EVAL_DIR="evaluation_data/cogbench"
TASKS="word_fmri,fmri,meg,eye_tracking"
OUTPUT_DIR=${OUTPUT_DIR:-"$PWD"}
BACKEND=${BACKEND:-"causal"}

# Parse command-line arguments.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path|--model_path_or_name|-m)
            MODEL_PATH="$2"
            shift 2
            ;;
        --revision_name)
            REVISION_NAME="$2"
            shift 2
            ;;
        --eval_dir)
            EVAL_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --task|--tasks)
            TASKS="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash eval_cogbench_fast.sh --model_path <path_or_hf_name> [--backend mlm|causal|mntp|enc_dec_mask|enc_dec_prefix] [--task word_fmri|fmri|meg|comma_list] [--eval_dir <path>] [--output_dir <path, default: current directory>] [--revision_name <name>]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

case "$BACKEND" in
    mlm|causal|mntp|enc_dec_mask|enc_dec_prefix) ;;
    *)
        echo "Error: unsupported backend '$BACKEND'. Expected one of: mlm, causal, mntp, enc_dec_mask, enc_dec_prefix"
        exit 1
        ;;
esac

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: model path is required. Use --model_path <path_or_hf_name>."
    exit 1
fi


# Normalize comma-separated task list for membership checks.
TASKS=",${TASKS},"

REVISION_ARGS=()
if [[ -n "$REVISION_NAME" ]]; then
    REVISION_ARGS+=(--revision_name "$REVISION_NAME")
fi

if [[ "$TASKS" == *",word_fmri,"* ]]; then
    python -m evaluation_pipeline.cogbench.run \
        --model_path_or_name "$MODEL_PATH" \
        --backend "$BACKEND" \
        --task word_fmri \
        --data_path "${EVAL_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --save_predictions \
        --fast \
        "${REVISION_ARGS[@]}"
fi

if [[ "$TASKS" == *",fmri,"* ]]; then
    python -m evaluation_pipeline.cogbench.run \
        --model_path_or_name "$MODEL_PATH" \
        --backend "$BACKEND" \
        --task fmri \
        --data_path "${EVAL_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --save_predictions \
        --fast \
        "${REVISION_ARGS[@]}"
fi

if [[ "$TASKS" == *",meg,"* ]]; then
    python -m evaluation_pipeline.cogbench.run \
        --model_path_or_name "$MODEL_PATH" \
        --backend "$BACKEND" \
        --task meg \
        --data_path "${EVAL_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --save_predictions \
        --fast \
        "${REVISION_ARGS[@]}"
fi

if [[ "$TASKS" == *",eye_tracking,"* ]]; then
    python -m evaluation_pipeline.cogbench.run \
        --model_path_or_name "$MODEL_PATH" \
        --backend "$BACKEND" \
        --task eye_tracking \
        --data_path "${EVAL_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --save_predictions \
        --fast \
        "${REVISION_ARGS[@]}"
fi