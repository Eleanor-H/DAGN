#!/usr/bin/env bash
export RECLOR_DIR=logiqa_data
export TASK_NAME=logiqa
export MODEL_DIR=roberta-large
export MODEL_TYPE=DAGN
export GRAPH_VERSION=4
export DATA_PROCESSING_VERSION=32
export MODEL_VERSION=2132
export GNN_VERSION=GCN_reversededges_double
export SAVE_DIR=dagn_aug

CUDA_VISIBLE_DEVICES=0 python3 run_multiple_choice.py \
    --disable_tqdm \
    --task_name $TASK_NAME \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_DIR \
    --init_weights \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $RECLOR_DIR \
    --graph_building_block_version $GRAPH_VERSION \
    --data_processing_version $DATA_PROCESSING_VERSION \
    --model_version $MODEL_VERSION \
    --merge_type 4 \
    --gnn_version $GNN_VERSION \
    --use_gcn \
    --gcn_steps 2 \
    --max_seq_length 256 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --roberta_lr 1e-5 \
    --gcn_lr 5e-6 \
    --proj_lr 5e-6 \
    --num_train_epochs 10 \
    --output_dir Checkpoints/$TASK_NAME/${SAVE_DIR} \
    --fp16 \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --numnet_drop 0.2
