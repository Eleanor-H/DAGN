#!/usr/bin/env bash
export RECLOR_DIR=reclor_data
export TASK_NAME=reclor
export MODEL_DIR=roberta-large
export MODEL_TYPE=PLM
export SAVE_DIR=plm

CUDA_VISIBLE_DEVICES=0 python3 run_multiple_choice.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_DIR \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 8 \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 3 \
    --learning_rate 1e-05 \
    --num_train_epochs 10.0 \
    --output_dir Checkpoints/$TASK_NAME/${SAVE_DIR} \
    --fp16 \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --warmup_steps 1932 \
    --weight_decay 0.01