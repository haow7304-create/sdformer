#!/bin/bash
# ============================================================
# iTransformer 自定义数据 - 多变量时序预测（Multivariate Forecasting）
# 与 scripts/multivariate_forecasting/ 下脚本格式一致，预测所有变量而非单目标列
# CSV：第一列为 date，其余列为多元变量；划分 70% 训练 / 10% 验证 / 20% 测试
# ============================================================

export CUDA_VISIBLE_DEVICES=0

model_name=SDformer

# ---------- 自定义数据配置（请按你的数据修改） ----------
# ROOT_PATH=/mnt/user/ai-car-miks/public_data/big_data/defect_foundation_model_weihao/ref_code_old/time_series/dataset/dexianjinzhi/
# ROOT_PATH=/mnt/user/ai-car-miks/public_data/big_data/defect_foundation_model_weihao/ref_code_old/time_series_full/dataset/war/
ROOT_PATH=/mnt/user/ai-car-miks/public_data/big_data/defect_foundation_model_weihao/ref_code_old/time_series_full/dataset/diplomacy/
DATA_PATH=train_data_file3.csv
# 变量数 = CSV 列数 - 1（去掉 date 列） 总列数：  file1:733 file2:44 file3:220
NUM_VARIATES=219

# 多变量预测时 --target 仅用于 DataLoader 列顺序，可填任意存在的列名（如最后一列）
TARGET=OT

# ---------- 序列与预测长度（与 multivariate_forecasting 一致：96/192/336/720） ----------
SEQ_LEN=96

# ---------- 模型与训练超参 ----------
E_LAYERS=4
D_MODEL=128
D_FF=128
BATCH_SIZE=8
LEARNING_RATE=0.001
ITR=1

# ---------- 从 iTransformer 根目录运行 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --model_id custom_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len 48 \
  --e_layers $E_LAYERS \
  --enc_in $NUM_VARIATES \
  --dec_in $NUM_VARIATES \
  --c_out $NUM_VARIATES \
  --des 'Exp' \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --itr $ITR

# python -u run.py \
#   --is_training 1 \
#   --root_path "$ROOT_PATH" \
#   --data_path "$DATA_PATH" \
#   --model_id custom_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $SEQ_LEN \
#   --pred_len 96 \
#   --e_layers $E_LAYERS \
#   --enc_in $NUM_VARIATES \
#   --dec_in $NUM_VARIATES \
#   --c_out $NUM_VARIATES \
#   --des 'Exp' \
#   --d_model $D_MODEL \
#   --d_ff $D_FF \
#   --batch_size $BATCH_SIZE \
#   --learning_rate $LEARNING_RATE \
#   --itr $ITR

# python -u run.py \
#   --is_training 1 \
#   --root_path "$ROOT_PATH" \
#   --data_path "$DATA_PATH" \
#   --model_id custom_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $SEQ_LEN \
#   --pred_len 192 \
#   --e_layers $E_LAYERS \
#   --enc_in $NUM_VARIATES \
#   --dec_in $NUM_VARIATES \
#   --c_out $NUM_VARIATES \
#   --des 'Exp' \
#   --d_model $D_MODEL \
#   --d_ff $D_FF \
#   --batch_size $BATCH_SIZE \
#   --learning_rate $LEARNING_RATE \
#   --itr $ITR

# python -u run.py \
#   --is_training 1 \
#   --root_path "$ROOT_PATH" \
#   --data_path "$DATA_PATH" \
#   --model_id custom_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $SEQ_LEN \
#   --pred_len 336 \
#   --e_layers $E_LAYERS \
#   --enc_in $NUM_VARIATES \
#   --dec_in $NUM_VARIATES \
#   --c_out $NUM_VARIATES \
#   --des 'Exp' \
#   --d_model $D_MODEL \
#   --d_ff $D_FF \
#   --batch_size $BATCH_SIZE \
#   --learning_rate $LEARNING_RATE \
  --itr $ITR

echo ">>>>>>> Custom multivariate forecasting finished <<<<<<<"
