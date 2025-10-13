#!/bin/bash

# 从环境变量读取路径配置
MODEL_PATH=${MODEL_PATH:-"/path/to/your/model"}
DATASET_PATH=${DATASET_PATH:-"/path/to/your/dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/your/output"}

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=2,3,4,5 \
swift sft \
    --model "$MODEL_PATH" \
    --train_type full \
    --dataset "$DATASET_PATH" \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --streaming false \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --packing false \
    --eval_steps 200 \
    --save_steps 2000 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 4 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn
