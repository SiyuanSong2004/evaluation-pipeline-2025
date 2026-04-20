#!/bin/bash

MODEL_PATH=$1
BACKEND=$2
LR=${3:-3e-5}           # default: 3e-5
BSZ=${4:-64}            # default: 64
MAX_EPOCHS=${5:-5}      # default: 5
WSC_EPOCHS=${6:-5}      # default: 5
SEED=${7:-42}           # default: 42

if [[ -z "$BACKEND" ]]; then
    echo "Usage: $0 <model_path> <causal|mlm|mntp|enc_dec_mask|enc_dec_prefix> [lr] [batch_size] [max_epochs] [wsc_epochs] [seed]"
    exit 1
fi

if [[ "$BACKEND" == "causal" ]]; then
    BACKEND_FLAGS="--causal --take_final"
elif [[ "$BACKEND" == enc_dec* ]]; then
    BACKEND_FLAGS="--enc_dec"
else
    BACKEND_FLAGS=""
fi

model_basename=$(basename $MODEL_PATH)

# ===== Chinese CLUE Fine-Tuning Tasks =====

# AFQMC — sentence similarity, 2 labels
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/clue/afqmc.train.jsonl" \
    --valid_data "evaluation_data/full_eval/clue/afqmc.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/clue/afqmc.valid.jsonl" \
    --task afqmc \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    $BACKEND_FLAGS \
    --verbose

# OCNLI — natural language inference, 3 labels
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/clue/ocnli.train.jsonl" \
    --valid_data "evaluation_data/full_eval/clue/ocnli.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/clue/ocnli.valid.jsonl" \
    --task ocnli \
    --num_labels 3 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy \
    --metric_for_valid accuracy \
    --seed $SEED \
    $BACKEND_FLAGS \
    --verbose

# TNEWS — news topic classification, 15 labels
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/clue/tnews.train.jsonl" \
    --valid_data "evaluation_data/full_eval/clue/tnews.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/clue/tnews.valid.jsonl" \
    --task tnews \
    --num_labels 15 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy \
    --metric_for_valid accuracy \
    --seed $SEED \
    $BACKEND_FLAGS \
    --verbose

# CLUEWSC2020 — pronoun disambiguation, 2 labels
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/clue/cluewsc2020.train.jsonl" \
    --valid_data "evaluation_data/full_eval/clue/cluewsc2020.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/clue/cluewsc2020.valid.jsonl" \
    --task cluewsc2020 \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $WSC_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    $BACKEND_FLAGS \
    --verbose
