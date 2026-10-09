#!/bin/bash
# 优化后的蒸馏脚本
export CUDA_VISIBLE_DEVICES=1,4
export NPROC_PER_NODE=2

TEACHER_MODEL="/data/xujianjun/workspace/dianhua_sft/output/v19_0527_simple_prompt/merged_sft_0527_simple_prompt"
STUDENT_MODEL="/data/xujianjun/models/Qwen/Qwen3-4B-Instruct-2507"
DATASET="/data/xujianjun/workspace/dianhua_sft/datasets/0520_dianhua_train_sft_simple_prompt_converted.json"
VAL_DATASET="/data/xujianjun/workspace/dianhua_sft/datasets/0520_dianhua_eval_sft_simple_prompt_converted.json"
OUTPUT_DIR="/data/xujianjun/workspace/dianhua_sft/output/v19_0527_simple_prompt/distilled-model"

# 启动蒸馏训练
nohup swift rlhf \
    --rlhf_type gkd \
    --template chatml \
    --response_prefix '' \
    --loss_scale last_round \
    --model $STUDENT_MODEL \
    --teacher_model $TEACHER_MODEL \
    --teacher_model_type qwen3 \
    --dataset $DATASET \
    --model_type qwen3 \
    --val_dataset $VAL_DATASET \
    --eval_strategy steps \
    --eval_steps 100 \
    --metric_for_best_model loss \
    --greater_is_better false \
    --load_best_model_at_end true \
    --train_type lora \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1.5e-4 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --output_dir $OUTPUT_DIR \
    --save_steps 100 \
    --logging_steps 10 \
    --warmup_ratio 0.05 \
    --offload_teacher_model false \
    --sft_alpha 0.5 \
    --beta 0.5 \
    --lmbda 0 \
    --save_only_model true  > zliu.log 2>&1 &
    
# 移除了 --deepspeed zero3（单卡环境）