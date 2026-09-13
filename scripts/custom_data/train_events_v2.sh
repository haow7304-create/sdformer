#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PHYSICAL_GPU="${PHYSICAL_GPU:-6}"
PYTHON_BIN="${PYTHON_BIN:-/home/liangjing/miniconda3/envs/llm/bin/python}"
PROCESSED_DIR="${PROCESSED_DIR:-$PROJECT_ROOT/dataset/processed_v2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/new_checkpoints_v2}"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
cd "$PROJECT_ROOT"

EVENT_KEYS=("russia_ukraine" "america_iran" "china_japan")
MODEL_IDS=("russia_ukraine_v2" "america_iran_v2" "china_japan_v2")
DATA_FILES=("俄乌冲突.csv" "美伊战争.csv" "中日外交.csv")

NUM_VARIATES=320
SEQ_LEN=96
PRED_LEN=48
E_LAYERS=4
D_MODEL=128
D_FF=128
BATCH_SIZE=8
LEARNING_RATE=0.001
TRAIN_EPOCHS=20
PATIENCE=4

RUN_CHECKPOINTS="$CHECKPOINT_ROOT/runs"
mkdir -p "$RUN_CHECKPOINTS"

for index in "${!EVENT_KEYS[@]}"; do
  event_key="${EVENT_KEYS[$index]}"
  model_id="${MODEL_IDS[$index]}"
  data_file="${DATA_FILES[$index]}"

  "$PYTHON_BIN" -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$PROCESSED_DIR/" \
    --data_path "$data_file" \
    --model_id "$model_id" \
    --model SDformer \
    --data custom \
    --features M \
    --target OT \
    --seq_len "$SEQ_LEN" \
    --pred_len "$PRED_LEN" \
    --e_layers "$E_LAYERS" \
    --enc_in "$NUM_VARIATES" \
    --dec_in "$NUM_VARIATES" \
    --c_out "$NUM_VARIATES" \
    --des V2 \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --train_epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --num_workers 4 \
    --checkpoints "$RUN_CHECKPOINTS/" \
    --grouped_loss \
    --itr 1

  setting="new_long_term_forecast_${model_id}_SDformer_custom_ftM_sl96_ll48_pl48_dm128_nh8_el4_dl1_df128_fc1_ebtimeF_dtTrue_V2_0"
  destination="$CHECKPOINT_ROOT/$event_key"
  mkdir -p "$destination"
  cp "$RUN_CHECKPOINTS/$setting/checkpoint.pth" "$destination/checkpoint.pth"
  cp "$RUN_CHECKPOINTS/$setting/training_config.json" "$destination/training_config.json"
  cp "$PROCESSED_DIR/${data_file%.csv}.preprocessor.json" "$destination/preprocessor.json"
  cp "$PROCESSED_DIR/feature_schema.json" "$destination/feature_schema.json"
  cp "$PROJECT_ROOT/results/$setting/metrics.npy" "$destination/metrics.npy"
done

echo "V2 training completed for all three events."
